# -*- coding: utf-8 -*-

import argparse
import math
from functools import partial
from typing import Any, Dict, Iterator, List, Optional

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from fla.modules.fused_cross_entropy import FusedCrossEntropyLoss


def _shift_labels_and_mask(
    input_ids: torch.Tensor,
    pad_token_id: Optional[int],
) -> torch.Tensor:

    B, L = input_ids.shape
    labels = input_ids.clone()
    labels[:, :-1] = input_ids[:, 1:]
    labels[:, -1] = -100  

    if pad_token_id is not None:
        pad_next = input_ids[:, 1:] == pad_token_id
        labels[:, :-1][pad_next] = -100

    return labels


class VarLenPerplexityEvaluator:
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizer,
        device: str = "cuda",
        bucket_size: int = 2048,
        batch_size: int = 1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.bucket_size = bucket_size
        self.batch_size = batch_size


        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.pad_token_id = self.tokenizer.pad_token_id
        self.loss_reduction = "none" 
        self.ce_ignore_index = -100

    @staticmethod
    def preprocess(
        examples: Dict[str, List[Any]],
        tokenizer: PreTrainedTokenizer,
        column_name: str = "text",
    ) -> Dict[str, List[List[int]]]:
        tok = tokenizer(examples[column_name])
        return {
            "input_ids": tok["input_ids"],
            "length": [len(x) for x in tok["input_ids"]],
        }

    def _collate_varlen(self, batch: List[List[int]]) -> Dict[str, torch.Tensor]:
        seq_lens = [len(x) for x in batch]
        Lmax = max(seq_lens)
        B = len(batch)

        input_ids = torch.full((B, Lmax), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((B, Lmax), dtype=torch.long)

        for i, ids in enumerate(batch):
            Li = len(ids)
            input_ids[i, :Li] = torch.tensor(ids, dtype=torch.long)
            attention_mask[i, :Li] = 1

        return {
            "input_ids": input_ids.to(self.device),
            "attention_mask": attention_mask.to(self.device),
            "seq_lens": torch.tensor(seq_lens, dtype=torch.long, device=self.device),
        }

    def _iter_varlen_batches(self, dataset: Dataset) -> Iterator[List[List[int]]]:
        cur: List[List[int]] = []
        for ex in dataset:
            ids = ex["input_ids"]
            ids = ids.tolist() if torch.is_tensor(ids) else list(ids)
            if not ids:
                continue
            cur.append(ids)
            if len(cur) == self.batch_size:
                yield cur
                cur = []
        if cur:
            yield cur

    @torch.no_grad()
    def evaluate(self, dataset: Dataset) -> Dict[str, Any]:

        sample_stats: List[Dict[str, Any]] = []
        longest_length = 0

        total_rows = len(dataset)
        bar = tqdm(self._iter_varlen_batches(dataset), total=(total_rows + self.batch_size - 1) // self.batch_size)

        sample_index = 0

        for batch in bar:
            pack = self._collate_varlen(batch)  
            input_ids = pack["input_ids"]        # [B, Lmax]
            attention_mask = pack["attention_mask"]
            seq_lens = pack["seq_lens"]          # [B]

            longest_length = max(longest_length, int(seq_lens.max().item()))

            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, Lmax, V]

            logp = logits.log_softmax(dim=-1)    # [B, Lmax, V]

            labels = _shift_labels_and_mask(input_ids, self.pad_token_id)  # [B, Lmax]

            B, Lmax = input_ids.shape
            for b in range(B):
                L = int(seq_lens[b].item())
                if L <= 1:
                    sample_stats.append({
                        "index": sample_index,
                        "length": L,
                        "ppl": float("nan"),
                        "bucket_ppls": [],
                        "bucket_token_counts": [],
                        "bucket_ranges": [],
                    })
                    sample_index += 1
                    continue

                lbl = labels[b, :L]  
                valid_mask = (lbl != -100)
                valid_positions = valid_mask.nonzero(as_tuple=False).flatten()  

                if valid_positions.numel() == 0:
                    sample_stats.append({
                        "index": sample_index,
                        "length": L,
                        "ppl": float("nan"),
                        "bucket_ppls": [],
                        "bucket_token_counts": [],
                        "bucket_ranges": [],
                    })
                    sample_index += 1
                    continue

                # logp[b, t, vocab] -> labels[b, t]
                tok_logp = logp[b, :L, :]  # [L, V]
                nll_all = -tok_logp.gather(dim=-1, index=lbl[:L].unsqueeze(-1)).squeeze(-1)  # [L]
                nll_all = nll_all[valid_mask]  

                total_nll = nll_all.sum()
                total_tok = valid_mask.sum()
                ppl = math.exp((total_nll / total_tok).item())


                bucket_ppls: List[float] = []
                bucket_token_counts: List[int] = []
                bucket_ranges: List[tuple] = []


                for start in range(0, L, self.bucket_size):
                    end = min(start + self.bucket_size, L)
                    bucket_mask = valid_mask[start:end]
                    if bucket_mask.any():
                        bucket_nll = nll_all[ (valid_positions >= start) & (valid_positions < end) ]
                       
                        tok_count = int(bucket_mask.sum().item())
                        ppl_i = math.exp((bucket_nll.sum() / tok_count).item())
                    else:
                        tok_count = 0
                        ppl_i = float("nan")

                    bucket_ppls.append(ppl_i)
                    bucket_token_counts.append(tok_count)
                    bucket_ranges.append((start, end))

                sample_stats.append({
                    "index": sample_index,
                    "length": L,
                    "ppl": ppl,
                    "bucket_ppls": bucket_ppls,
                    "bucket_token_counts": bucket_token_counts,
                    "bucket_ranges": bucket_ranges,
                })

                sample_index += 1

            bar.set_description_str(f"[processed {sample_index}/{total_rows}] longest={longest_length}")

        return {
            "longest_length": longest_length,
            "sample_stats": sample_stats,
        }


def main():
    parser = argparse.ArgumentParser(description="VarLen per-sample perplexity (with per-sample bucketing)")
    parser.add_argument("-p", "--path", type=str, default="fla-hub/gla-1.3B-100B")
    parser.add_argument("-d", "--data", type=str, default="fla-hub/pg19")
    parser.add_argument("-s", "--split", type=str, default="train")
    parser.add_argument("-n", "--column_name", type=str, default="text")
    parser.add_argument("--bucket_size", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.device is None:
        from fla.utils import device
    else:
        device = args.device
    torch.manual_seed(0)

    print(f"Loading model {args.path}")
    tokenizer = AutoTokenizer.from_pretrained(args.path)
    model = AutoModelForCausalLM.from_pretrained(args.path, device_map={"": device}).bfloat16().eval()

    print(f"Loading data {args.data}")
    dataset = load_dataset(args.data, split=args.split)
    dataset = dataset.map(
        partial(VarLenPerplexityEvaluator.preprocess, tokenizer=tokenizer, column_name=args.column_name),
        batched=True,
        num_proc=32,
    )
    print(dataset)

    evaluator = VarLenPerplexityEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        bucket_size=args.bucket_size,
        batch_size=args.batch_size,
    )

    with torch.no_grad():
        results = evaluator.evaluate(dataset)

    print("\nEvaluation Results (VarLen / per-sample):")
    print(f"Longest tokenized length: {results['longest_length']}")
    print(f"Num samples: {len(results['sample_stats'])}")

    for s in results["sample_stats"][:3]:
        print(f"\nSample #{s['index']}  len={s['length']}  ppl={s['ppl']:.2f}")
        for (r, p, c) in zip(s["bucket_ranges"], s["bucket_ppls"], s["bucket_token_counts"]):
            l, r_ = r
            print(f"  bucket[{l}:{r_}]  tokens={c:4d}  ppl={p if isinstance(p, float) and not math.isnan(p) else 'NaN'}")


if __name__ == "__main__":
    main()
