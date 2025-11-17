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
    _score_mod_signature,
    BlockMask,
)
import pdb


def cu_seqlens_to_document_id(
    cu_seqlens,
):
    '''
    Here we return the document id given cu_seqlens
    '''
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    document_id = torch.repeat_interleave(torch.arange(len(seqlens), device=cu_seqlens.device), seqlens)
    return document_id


def hierarchy_cross_causal_mask_varlen(
    cu_seqlens_q: torch.Tensor,
    cu_chunklens: torch.Tensor,
    chunk_size: int,
):
    document_id_q = cu_seqlens_to_document_id(cu_seqlens_q)
    document_id_chunk = cu_seqlens_to_document_id(cu_chunklens)

    # we should create a mask_fn according to varlen arguments
    def mask_fn(b, h, q_idx, k_idx):
        '''
        k_idx is for chunk
        '''
        # only same sample should be computed
        sample_mask = (document_id_q[q_idx] == document_id_chunk[k_idx])
        # only causal part should be computed
        k_logic_idx = ((k_idx - cu_chunklens[document_id_chunk[k_idx]]) + 1) * (chunk_size + 1) - 1
        q_logic_idx = q_idx - cu_seqlens_q[document_id_q[q_idx]]
        inner_mask = (q_logic_idx >= k_logic_idx)

        return sample_mask & inner_mask
    
    return mask_fn

def hierarchy_cross_causal_mask_packing(
    chunk_size: int,
):
    # we should create a mask_fn according to varlen arguments
    def mask_fn(b, h, q_idx, k_idx):
        '''
        k_idx is for chunk
        '''
        # only causal part should be computed
        k_logic_idx = (k_idx + 1) * (chunk_size + 1) - 1
        q_logic_idx = q_idx
        inner_mask = (q_logic_idx >= k_logic_idx)

        return inner_mask
    
    return mask_fn

def hierarchy_cross_causal_mask(
    chunk_size: int,
    cu_seqlens_q: torch.Tensor = None,
    cu_chunklens: torch.Tensor = None,
):
    if cu_seqlens_q is None:
        assert cu_chunklens is None, "cu_chunklens should be None when cu_seqlens_q is None"
        mode = "packing"
        return hierarchy_cross_causal_mask_packing(chunk_size), mode
    else:
        mode = "varlen"
        return hierarchy_cross_causal_mask_varlen(cu_seqlens_q, cu_chunklens, chunk_size), mode

def fa_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    block_mask: Optional[BlockMask] = None,
):
    '''
    usually we use q, k, v as [B, L, H, D]
    However, in flex attention the input is [B, H, L, D]
    So, Here we wrap the input to [B, H, L, D] for convenience
    '''
    query = query.transpose(1, 2)
    key = key.transpose(1, 2)
    value = value.transpose(1, 2)
    return flex_attention(
        query,
        key,
        value,
        score_mod=None,
        block_mask=block_mask,
        scale=None,
        enable_gqa=False,
        return_lse=False,
        kernel_options=None,
    )

if __name__ == "__main__":
    pass