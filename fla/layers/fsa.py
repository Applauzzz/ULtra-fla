# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

logger = logging.get_logger(__name__)


from fla.ops.native_sparse_attention.ops import (
    compressed_attention,
    topk_sparse_attention,
    linear_compress,
    compressed_attention_decode,
    topk_sparse_attention_decode,
    v2b,
)

def fsa_func_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    compress_key: torch.Tensor,
    compress_value: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    cu_seqlens_k: torch.Tensor,
    max_seqlen_q: int,
    max_seqlen_k: int,
    num_heads: int,
    num_kv_heads: int,
    init_blocks: int = 1,
    local_blocks: int = 2,
    window_size: int = 512,
    kernel_size: int = 32,
    kernel_stride: int = 16,
    block_size: int = 64,
    topk: int = 16,
):
    batch_size = cu_seqlens_q.shape[0] - 1
    if torch.equal(cu_seqlens_q, cu_seqlens_k):
        mode = "prefill"
    elif torch.equal(cu_seqlens_q, torch.arange(0, batch_size + 1, dtype=cu_seqlens_q.dtype, device=cu_seqlens_q.device)):
        mode = "decode"
    else:
        raise ValueError(
            f"cu_seqlens_q and cu_seqlens_k must be equal or cu_seqlens_q must be cu_seqlens_k + 1, but got {cu_seqlens_q} and {cu_seqlens_k}"
        )
    if mode == "prefill":
        # compressed key and value before rope
        compressed_k, compressed_cu_seqlens = linear_compress(
            k,
            compress_key,
            cu_seqlens_k,
            kernel_size,
            kernel_stride,
            None,
        )
        compressed_v, _ = linear_compress(
            v,
            compress_value,
            cu_seqlens_k,
            kernel_size,
            kernel_stride,
            None,
        )

        # seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
        # max_seqlen_q = seqlens_q.max().item()
        # seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        # max_seqlen_k = seqlens_k.max().item()
        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        max_seqlens_compressed = compressed_seqlens.max().item()
        
        compressed_attn_output, topk_idx = compressed_attention(
            q,
            compressed_k,
            compressed_v,
            kernel_size,
            kernel_stride,
            block_size,
            topk,
            cu_seqlens_q,
            compressed_cu_seqlens,
            max_seqlen_q,
            max_seqlens_compressed,
            None,
            init_blocks,
            local_blocks,
            parallel_topk_compute=False,
        )

        sparse_attn_output = topk_sparse_attention(
            q, k, v, topk_idx, block_size, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, None
        )
        
        # sliding window attention
        sliding_attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            window_size=(window_size, -1),
        )
        
        attn_output = (
            gate[:, 0:1, None] * compressed_attn_output
            + gate[:, 1:2, None] * sparse_attn_output
            + gate[:, 2:3, None] * sliding_attn_output
        )

        return attn_output
    elif mode == "decode":

        # compressed key and value before rope
        compressed_k, compressed_cu_seqlens = linear_compress(
            k,
            compress_key,
            cu_seqlens_k,
            kernel_size,
            kernel_stride,
            None,
        )
        compressed_v, _ = linear_compress(
            v,
            compress_value,
            cu_seqlens_k,
            kernel_size,
            kernel_stride,
            None,
        )

        seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
        max_seqlen_k = seqlens_k.max().item()
        k_flat = v2b(
            k,
            cu_seqlens_k,
            seqlens_k,
            max_seqlen_k,
        )
        v_flat = v2b(
            v,
            cu_seqlens_k,
            seqlens_k,
            max_seqlen_k,
        )

        compressed_seqlens = compressed_cu_seqlens[1:] - compressed_cu_seqlens[:-1]
        max_seqlen_compressed = compressed_seqlens.max().item()

        compressed_k_flat = v2b(
            compressed_k,
            compressed_cu_seqlens,
            compressed_seqlens,
            max_seqlen_compressed,
        )
        compressed_v_flat = v2b(
            compressed_v,
            compressed_cu_seqlens,
            compressed_seqlens,
            max_seqlen_compressed,
        )
        compressed_attn_output, topk_idx = compressed_attention_decode(
            q,
            k_flat,
            v_flat,
            seqlens_k,
            compressed_seqlens,
            kernel_size,
            kernel_stride,
            block_size,
            topk,
            init_blocks,
            local_blocks,
        )

        sparse_attn_output = topk_sparse_attention_decode(
            q,
            k_flat,
            v_flat,
            topk_idx,
            block_size,
            seqlens_k,
        )

        # sliding window attention
        sliding_attn_output = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            causal=True,
            window_size=(window_size, -1),
        )

        attn_output = (
            gate[:, 0:1, None] * compressed_attn_output
            + gate[:, 1:2, None] * sparse_attn_output
            + gate[:, 2:3, None] * sliding_attn_output
        )

        return attn_output

    else:
        raise ValueError("Only prefill (or Training) and Decode modes are supported for now")


class Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int = 2048,
        num_heads: int = 32,
        num_kv_heads: Optional[int] = None,
        head_dim: Optional[int] = None,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        window_size: Optional[int] = 512,
        init_blocks: Optional[int] = 1,
        local_blocks: Optional[int] = 2,
        kernel_size: Optional[int] = 32,
        kernel_stride: Optional[int] = 16,
        block_size: Optional[int] = 64,
        topk: Optional[int] = 16,
        rope_theta: Optional[float] = 10000.,
        max_position_embeddings: Optional[int] = None,
        layer_idx: int = None,
        use_rope: bool = True,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = head_dim if head_dim is not None else hidden_size // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        if flash_attn_func is None:
            raise ImportError("Please install Flash Attention via `pip install flash-attn --no-build-isolation` first")
    
        # self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        # self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        # self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        # self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.qkv_bias
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_kv_heads * self.head_dim, bias=self.qkv_bias
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )
        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)
        self.use_rope = use_rope
        if self.use_rope:   
            self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)
        
        # NOTE added for NSA computation
        self.init_blocks = init_blocks
        self.local_blocks = local_blocks
        self.kernel_size = kernel_size
        self.kernel_stride = kernel_stride
        self.block_size = block_size
        self.topk = topk

        # nsa parameteres
        self.compress_key = torch.nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim * self.kernel_size, self.head_dim)
        )
        self.compress_value = torch.nn.Parameter(
            torch.zeros(self.num_kv_heads, self.head_dim * self.kernel_size, self.head_dim)
        )

        # gate function
        self.gate = torch.nn.Sequential(torch.nn.Linear(self.hidden_size, 3, bias=False), torch.nn.Sigmoid())

        # init parameters
        self.init_params()

    def init_params(self):
        for p in self.parameters():
            torch.nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        # NOTE gate is for native sparse attention
        gate = self.gate(hidden_states)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)
        
        if self.use_rope:
            q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Contains at least one padding token in the sequence
        if attention_mask is not None:
            if q.shape[1] == 1 and self.window_size is not None:
                attention_mask = attention_mask[:, -self.window_size:]
            q, (k, v), indices_q, cu_seqlens, max_seq_lens = unpad_input(q, (k, v), attention_mask, q_len)
            cu_seqlens_q, cu_seqlens_k = cu_seqlens
            max_seqlen_q, max_seqlen_k = max_seq_lens
            o = fsa_func_varlen(
                q, k, v, gate.squeeze(0),
                compress_key=self.compress_key,
                compress_value=self.compress_value,
                cu_seqlens_q=cu_seqlens_q,
                cu_seqlens_k=cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                max_seqlen_k=max_seqlen_k,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                window_size=self.window_size,
                kernel_size=self.kernel_size,
                kernel_stride=self.kernel_stride,
                block_size=self.block_size,
                topk=self.topk,
            )
            o = pad_input(o, indices_q, batch_size, q_len)
        elif cu_seqlens is not None:
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            o = fsa_func_varlen(
                q.squeeze(0), k.squeeze(0), v.squeeze(0), gate.squeeze(0),
                compress_key=self.compress_key,
                compress_value=self.compress_value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                window_size=self.window_size,
                kernel_size=self.kernel_size,
                kernel_stride=self.kernel_stride,
                block_size=self.block_size,
                topk=self.topk,
            ).unsqueeze(0)
        else:
            cu_seqlens = torch.arange(0, (batch_size + 1) * q_len, step=q_len, device=q.device).to(torch.int32)
            max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max().item()
            o = fsa_func_varlen(
                q.reshape(batch_size*q_len, self.num_heads, self.head_dim), 
                k.reshape(batch_size*q_len, self.num_kv_heads, self.head_dim), 
                v.reshape(batch_size*q_len, self.num_kv_heads, self.head_dim), 
                gate.reshape(batch_size*q_len, -1),
                compress_key=self.compress_key,
                compress_value=self.compress_value,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                num_heads=self.num_heads,
                num_kv_heads=self.num_kv_heads,
                init_blocks=self.init_blocks,
                local_blocks=self.local_blocks,
                window_size=self.window_size,
                kernel_size=self.kernel_size,
                kernel_stride=self.kernel_stride,
                block_size=self.block_size,
                topk=self.topk,
            ).reshape(batch_size, q_len, self.num_heads, self.head_dim)
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        if not output_attentions:
            attentions = None

        return o, attentions, past_key_values

if __name__ == "__main__":
    # =================================================================================
    # Original Test Case (for sequences of the same length)
    # =================================================================================
    print("--- Running Original Test (Equal Sequence Lengths) ---")
    B, L, D = 2, 1024, 1024  # Reduced sequence length for faster testing
    hidden_size = D
    num_heads = 8
    num_kv_heads = 8
    print("Creating model...")
    try:
        module = Attention(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
        ).to(torch.bfloat16).cuda()
        print("Model created successfully.")
        hidden_state = torch.randn(B, L, D, dtype=torch.bfloat16, device='cuda')
        print("Computing with equal length inputs...")
        # This tests the `else` block in the forward pass
        output, _, _ = module(hidden_state)
        print(f"Output shape (equal lengths): {output.shape}")
        print("Original test completed successfully.\n")
    except Exception as e:
        print(f"An error occurred during the original test: {e}")
 # =================================================================================
    # New Test Case for Mismatched cu_seqlens_q and cu_seqlens_k
    # =================================================================================
    print("--- Running New Test (Mismatched Sequence Lengths for Q and K/V) ---")
    # Use different parameters to avoid confusion
    test_batch_size = 4
    test_hidden_size = 1024
    test_num_heads = 8
    test_num_kv_heads = 4  # Using GQA is a good practice for testing
    head_dim = test_hidden_size // test_num_heads
    # Define different sequence lengths for Q and K/V
    # For example, Q has shorter sequences than K/V
    q_seqlens = [1, 1, 1, 1]
    k_seqlens = [300, 600, 150, 40000]
    # Ensure the test setup is valid
    assert test_batch_size == len(q_seqlens), "Batch size must match the number of sequence lengths."
    assert len(q_seqlens) == len(k_seqlens), "Q and K/V must have the same number of sequences in the batch."
    max_seqlen_q = max(q_seqlens)
    max_seqlen_k = max(k_seqlens)
    print(f"Batch Size: {test_batch_size}")
    print(f"Q Sequence Lengths: {q_seqlens} (Max: {max_seqlen_q})")
    print(f"K/V Sequence Lengths: {k_seqlens} (Max: {max_seqlen_k})")
    # Manually create cu_seqlens
    cu_seqlens_q = torch.tensor([0] + list(torch.cumsum(torch.tensor(q_seqlens), 0)), dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0] + list(torch.cumsum(torch.tensor(k_seqlens), 0)), dtype=torch.int32, device='cuda')
    print(f"cu_seqlens_q: {cu_seqlens_q}")
    print(f"cu_seqlens_k: {cu_seqlens_k}")
    total_len_q = sum(q_seqlens)
    total_len_k = sum(k_seqlens)
    # Create ragged tensors (concatenated along the sequence dimension)
    # These mimic the output of `unpad_input`
    q_unpadded = torch.randn(total_len_q, test_num_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    k_unpadded = torch.randn(total_len_k, test_num_kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    v_unpadded = torch.randn(total_len_k, test_num_kv_heads, head_dim, dtype=torch.bfloat16, device='cuda')
    gate_unpadded = torch.randn(total_len_q, 3, dtype=torch.bfloat16, device='cuda')
    print(f"Shape of unpadded Q: {q_unpadded.shape}")
    print(f"Shape of unpadded K: {k_unpadded.shape}")
    print(f"Shape of unpadded V: {v_unpadded.shape}")
    # Create a new model instance for this test
    try:
        varlen_module = Attention(
            hidden_size=test_hidden_size,
            num_heads=test_num_heads,
            num_kv_heads=test_num_kv_heads,
        ).to(torch.bfloat16).cuda()
        print("Varlen model created successfully.")
        print("Computing with fsa_func_varlen and mismatched cu_seqlens...")
        # This call directly tests the target function
        output_varlen = fsa_func_varlen(
            q=q_unpadded,
            k=k_unpadded,
            v=v_unpadded,
            gate=gate_unpadded,
            compress_key=varlen_module.compress_key,
            compress_value=varlen_module.compress_value,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            num_heads=varlen_module.num_heads,
            num_kv_heads=varlen_module.num_kv_heads,
            init_blocks=varlen_module.init_blocks,
            local_blocks=varlen_module.local_blocks,
            window_size=varlen_module.window_size,
            kernel_size=varlen_module.kernel_size,
            kernel_stride=varlen_module.kernel_stride,
            block_size=varlen_module.block_size,
            topk=varlen_module.topk,
        )
        print(f"Output shape from fsa_func_varlen: {output_varlen.shape}")
        # The output should have a total length equal to the total length of Q
        assert output_varlen.shape[0] == total_len_q
        assert output_varlen.shape[1] == test_num_heads
        assert output_varlen.shape[2] == head_dim
        print("New test for mismatched sequence lengths completed successfully!")
    except Exception as e:
        print(f"An error occurred during the mismatched lengths test: {e}")
        import traceback
        traceback.print_exc()