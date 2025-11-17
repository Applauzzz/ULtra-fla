import torch
from torch import Tensor
from torch import nn
from typing import Any, Callable, Optional, Union
import matplotlib.pyplot as plt
import os

from torch.nn.attention.flex_attention import (
    flex_attention, 
    create_block_mask,
    create_mask,
    and_masks,
    or_masks,
    _score_mod_signature,
    BlockMask,
)

import pdb


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx

def diff_length_causal_mask(mask_mod, Q_LEN, K_LEN):
    offset = K_LEN - Q_LEN
    assert offset >= 0
    def mask_fn(b, h, q_idx, kv_idx):
        q_idx_logic = offset + q_idx
        return mask_mod(b, h, q_idx_logic, kv_idx)
    return mask_fn

def varlen_causal_mask(mask_mod, cu_seqlens_q, cu_seqlens_k, document_id_q, document_id_k):
    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    
    def mask_fn(b, h, q_idx, k_idx):
        sample_idx_q = document_id_q[q_idx]
        sample_idx_k = document_id_k[k_idx]
        sample_mask = (sample_idx_q == sample_idx_k)

        q_len = seqlens_q[sample_idx_q]
        k_len = seqlens_k[sample_idx_k]
        q_offset = cu_seqlens_q[sample_idx_q]
        k_offset = cu_seqlens_k[sample_idx_k]
        intra_offset = k_len - q_len
        q_idx_logic = q_idx - q_offset + intra_offset
        k_idx_logic = k_idx - k_offset

        # sample_mask = (document_id_q[q_idx] == document_id_k[k_idx])
        inner_mask = mask_mod(b, h, q_idx_logic, k_idx_logic)

        return sample_mask &inner_mask
    return mask_fn

def fa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    score_mod: Optional[_score_mod_signature] = None,
    block_mask: Optional[BlockMask] = None,
    scale: Optional[float] = None,
    enable_gqa: bool = False,
    return_lse: bool = False,
    kernel_options: Optional[dict[str, Any]] = None,
):
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    return flex_attention(
        query,
        key,
        value,
        score_mod=score_mod,
        block_mask=block_mask,
        scale=scale,
        enable_gqa=enable_gqa,
        return_lse=return_lse,
        kernel_options=kernel_options,
    )

if __name__ == "__main__":
    B, H, D = 1, 1, 128
    seqlens_q = [1, 1, 1024, 4049]
    seqlens_k = [256, 500, 2048, 5001]
    seqlens_q = torch.tensor(seqlens_q, dtype=torch.int32, device="cuda")
    seqlens_k = torch.tensor(seqlens_k, dtype=torch.int32, device="cuda")
    cu_seqlens_q = torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_q.cumsum(0)])
    cu_seqlens_k = torch.cat([torch.tensor([0], dtype=torch.int32, device="cuda"), seqlens_k.cumsum(0)])
    document_id_q = torch.repeat_interleave(torch.arange(len(seqlens_q), device=cu_seqlens_q.device), seqlens_q)
    document_id_k = torch.repeat_interleave(torch.arange(len(seqlens_k), device=cu_seqlens_k.device), seqlens_k)
    Q_LEN = cu_seqlens_q[-1].item()
    K_LEN = cu_seqlens_k[-1].item()
    causal_mask_log = varlen_causal_mask(causal_mask, cu_seqlens_q, cu_seqlens_k, document_id_q, document_id_k)
    mask = create_mask(causal_mask_log, B, H, Q_LEN, K_LEN, device="cuda")
    # ==================== 绘图并保存代码开始 ====================
    # 创建一个文件夹来存放图像，避免文件混乱
    output_dir = "attention_visuals"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # --- 绘制并保存完整的 Attention Mask ---
    print("正在绘制并保存完整的 Attention Mask 图像...")
    mask_to_plot = mask[0, 0, :, :].cpu().to(torch.float)
    plt.figure(figsize=(12, 10))
    # 将 origin 从 'lower' 改为 'upper'
    plt.imshow(mask_to_plot, cmap='gray_r', origin='upper', aspect='auto') # <--- 主要改动在这里
    plt.title("Variable-Length Causal Attention Mask (Origin at Top-Left)")
    plt.xlabel("Key Sequence Index (k_idx)")
    plt.ylabel("Query Sequence Index (q_idx)")
    cbar = plt.colorbar()
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Masked (False)', 'Computed (True)'])
    plt.tight_layout()
    output_path_mask = os.path.join(output_dir, "attention_mask_origin_upper.png")
    plt.savefig(
        output_path_mask, 
        dpi=300,
        bbox_inches='tight'
    )
    print(f"完整的 Attention Mask 图像已保存到: {output_path_mask}")
    plt.close()