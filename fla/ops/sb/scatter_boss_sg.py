from typing import Optional, Tuple
from dataclasses import dataclass
import torch
import torch.nn.functional as F
import triton
from triton import language as tl
from cubsort import sort_varlen
import math
import time
import numpy as np
import pdb
from flash_attn.flash_attn_interface import _wrapped_flash_attn_forward as _flash_attn_forward
from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_forward as _flash_attn_varlen_forward
from flash_attn.flash_attn_interface import _wrapped_flash_attn_backward as _flash_attn_backward
from flash_attn.flash_attn_interface import _wrapped_flash_attn_varlen_backward as _flash_attn_varlen_backward

'''
==================================================
NOTE scatter boss autograd function class
==================================================
'''
def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x

class BOSS_Attention(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        q: torch.Tensor, # [B L H D]
        k: torch.Tensor, # [B L H D]
        v: torch.Tensor, # [B L H D]
        top_states_q: torch.Tensor, # [B L H]
        top_states_k: torch.Tensor, # [B L H]
        sg_indices_q: torch.Tensor, # [B L H]
        sg_indices_k: torch.Tensor, # [B L H]
        num_classes: int,
        local_size: int,
        global_size: int,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        assert q.shape[1] == k.shape[1], "q, k should have same length"
        assert q.dtype in [torch.float16, torch.bfloat16], "only support fp16 and bf16"
        assert q.dtype == k.dtype
        assert q.dtype == v.dtype
        assert top_states_q.dtype == torch.int32
        assert top_states_k.dtype == torch.int32
        B, L, H, D = q.shape
        if cu_seqlens is None:
            USE_CUSEQLENS = False
        else:
            USE_CUSEQLENS = True
        softmax_scale = 1 / math.sqrt(D)
        # NOTE local part computation
        if USE_CUSEQLENS:
            o, lse, _, _ = _flash_attn_varlen_forward(
                q=q.view((L, H, D)),
                k=k.view((L, H, D)),
                v=v.view((L, H, D)),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=L,
                max_seqlen_k=L,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size_left=local_size-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
            o=o.view((1, L, H, D))
            lse = lse.view((1, H, L))
        else:
            o, lse, _, _ = _flash_attn_forward(
                q=q,
                k=k,
                v=v,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size_left=local_size-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                return_softmax=False,
            )
        # NOTE boss computation
        boss_info = boss_prepare(
            top_states_q=top_states_q,
            top_states_k=top_states_k,
            shift_size=local_size,
            num_classes=num_classes,
            cu_seqlens=cu_seqlens,
        )
        
        if boss_info.valid:
            o_boss, lse_boss = scatter_boss_fwd(
                q=q,
                k=k,
                v=v,
                softmax_scale=softmax_scale,
                state_lambda_q=boss_info.state_lambda_q,
                indices_q=boss_info.indices_q,
                indices_k=boss_info.indices_k,
                hist_q=boss_info.hist_q,
                hist_k=boss_info.hist_k,
                local_size=local_size,
                global_size=global_size,
                max_bin_q=boss_info.max_bin_q,
                cu_seqlens=cu_seqlens,
                cu_trunclens=boss_info.cu_trunclens,
            )
            # NOTE magic chunk size:
            # the compute will fail if chun size = 64
            CHUNK_SIZE = 128
            if USE_CUSEQLENS:
                weighted_merge_o[(boss_info.B, H, triton.cdiv(boss_info.max_seqlen, CHUNK_SIZE))](
                    o=o,
                    lse=lse,
                    o_boss=o_boss,
                    lse_boss=lse_boss,
                    cu_seqlens=cu_seqlens,
                    cu_trunclens=boss_info.cu_trunclens,
                    L=L,
                    LOCAL_SIZE=local_size,
                    USE_CUSEQLENS=boss_info.USE_CUSEQLENS,
                    CHUNK_SIZE=CHUNK_SIZE,
                    H=H,
                    D=D,
                )
            else:
                weighted_merge_o[(boss_info.B, H, triton.cdiv(boss_info.L_S, CHUNK_SIZE))](
                    o=o,
                    lse=lse,
                    o_boss=o_boss,
                    lse_boss=lse_boss,
                    cu_seqlens=cu_seqlens,
                    cu_trunclens=boss_info.cu_trunclens,
                    L=L,
                    LOCAL_SIZE=local_size,
                    USE_CUSEQLENS=boss_info.USE_CUSEQLENS,
                    CHUNK_SIZE=CHUNK_SIZE,
                    H=H,
                    D=D,
                )
        else:
            o_boss = None
        ctx.save_for_backward(
            q,
            k,
            v,
            o,
            lse,
            boss_info.indices_q,
            boss_info.indices_k,
            boss_info.state_lambda_q,
            boss_info.state_lambda_k,
            boss_info.hist_q,
            boss_info.hist_k,
            cu_seqlens,
            boss_info.cu_trunclens,
        )
        ctx.num_sample = boss_info.B
        ctx.USE_CUSEQLENS = boss_info.USE_CUSEQLENS
        ctx.valid = boss_info.valid
        ctx.L_S = boss_info.L_S
        ctx.max_seqlen = boss_info.max_seqlen
        ctx.max_bin_q = boss_info.max_bin_q
        ctx.max_bin_k = boss_info.max_bin_k
        ctx.shape = (B, L, H, D)
        ctx.softmax_scale = softmax_scale
        ctx.local_size = local_size
        ctx.global_size = global_size
        return o
    
    @staticmethod
    def backward(
        ctx,
        do: torch.Tensor,
    ):
        if not do.is_contiguous():
            do = do.contiguous()
        (
            q,
            k,
            v,
            o,
            lse,
            indices_q,
            indices_k,
            state_lambda_q,
            state_lambda_k,
            hist_q,
            hist_k,
            cu_seqlens,
            cu_trunclens,
        ) = ctx.saved_tensors
        num_sample = ctx.num_sample
        USE_CUSEQLENS = ctx.USE_CUSEQLENS
        valid = ctx.valid
        L_S = ctx.L_S
        max_seqlen = ctx.max_seqlen
        (B, L, H, D) = ctx.shape
        softmax_scale = ctx.softmax_scale
        local_size = ctx.local_size
        global_size = ctx.global_size
        max_bin_q = ctx.max_bin_q
        max_bin_k = ctx.max_bin_k
        boss_info = BossPrepareResult(
            indices_q=indices_q,
            indices_k=indices_k,
            state_lambda_q=state_lambda_q,
            state_lambda_k=state_lambda_k,
            hist_q=hist_q,
            hist_k=hist_k,
            cu_trunclens=cu_trunclens,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=num_sample,
            L_S=L_S,
            max_seqlen=max_seqlen,
            max_bin_q=max_bin_q,
            max_bin_k=max_bin_k,
            valid=valid,
        )
        dq = torch.empty_like(q)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)
        dq_boss = torch.zeros_like(q)
        dk_boss = torch.zeros_like(k)
        dv_boss = torch.zeros_like(v)
        
        if cu_seqlens is not None:
            USE_CUSEQLENS = True
        else:
            USE_CUSEQLENS = False
        if USE_CUSEQLENS:
            _flash_attn_varlen_backward(
                dout=do.view((L, H, D)),
                q=q.view((L, H, D)),
                k=k.view((L, H, D)),
                v=v.view((L, H, D)),
                out=o.view((L, H, D)),
                softmax_lse=lse.view((H, L)),
                dq=dq.view((L, H, D)),
                dk=dk.view((L, H, D)),
                dv=dv.view((L, H, D)),
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=L,
                max_seqlen_k=L,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size_left=local_size-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
            )
            dq = dq.view((1, L, H, D))
            dk = dk.view((1, L, H, D))
            dv = dv.view((1, L, H, D))
        else:
            _flash_attn_backward(
                dout=do,
                q=q,
                k=k,
                v=v,
                out=o,
                softmax_lse=lse,
                dq=dq,
                dk=dk,
                dv=dv,
                dropout_p=0.0,
                softmax_scale=softmax_scale,
                causal=True,
                window_size_left=local_size-1,
                window_size_right=-1,
                softcap=0.0,
                alibi_slopes=None,
                deterministic=False,
            )
        if boss_info.valid:
            delta = torch.zeros_like(lse)
            d_sgq = torch.zeros((B, L, H), device=dk.device)
            d_sgk = torch.zeros((B, L, H), device=dk.device)
            do_boss = torch.zeros_like(do)
            CHUNK_SIZE = 128
            if USE_CUSEQLENS:
                dweight_scatter_o[(boss_info.B, H, triton.cdiv(boss_info.max_seqlen, CHUNK_SIZE))](
                    do=do,
                    o=o,
                    do_boss=do_boss,
                    delta=delta,
                    cu_seqlens=cu_seqlens,
                    cu_trunclens=boss_info.cu_trunclens,
                    L=L,
                    LOCAL_SIZE=local_size,
                    USE_CUSEQLENS=boss_info.USE_CUSEQLENS,
                    CHUNK_SIZE=CHUNK_SIZE,
                    H=H,
                    D=D,
                )
            else:
                dweight_scatter_o[(boss_info.B, H, triton.cdiv(boss_info.L_S, CHUNK_SIZE))](
                    do=do,
                    o=o,
                    do_boss=do_boss,
                    delta=delta,
                    cu_seqlens=cu_seqlens,
                    cu_trunclens=boss_info.cu_trunclens,
                    L=L,
                    LOCAL_SIZE=local_size,
                    USE_CUSEQLENS=boss_info.USE_CUSEQLENS,
                    CHUNK_SIZE=CHUNK_SIZE,
                    H=H,
                    D=D,
                )
            dq_boss, dk_boss, dv_boss = scatter_boss_bwd(
                do=do_boss,
                q=q,
                k=k,
                v=v,
                dq=dq_boss,
                dk=dk_boss,
                dv=dv_boss,
                d_sgq=d_sgq,
                d_sgk=d_sgk,
                lse=lse,
                delta=delta,
                softmax_scale=softmax_scale,
                state_lambda_q=boss_info.state_lambda_q,
                state_lambda_k=boss_info.state_lambda_k,
                indices_q=boss_info.indices_q,
                indices_k=boss_info.indices_k,
                hist_q=boss_info.hist_q,
                hist_k=boss_info.hist_k,
                local_size=local_size,
                global_size=global_size,
                max_bin_q=boss_info.max_bin_q,
                max_bin_k=boss_info.max_bin_k,
                cu_seqlens=cu_seqlens,
                cu_trunclens=boss_info.cu_trunclens,
            )
            dq += dq_boss
            dk += dk_boss
            dv += dv_boss
            d_sgq = torch.einsum("bnhd,bnhd->bnh", dq.float(), q.float())
            d_sgk = torch.einsum("bnhd,bnhd->bnh", dk.float(), k.float())
        return dq, dk, dv, None, None, d_sgq, d_sgk, None, None, None, None

def scatter_boss_function(
    q: torch.Tensor, # [B L H D]
    k: torch.Tensor, # [B L H D]
    v: torch.Tensor, # [B L H D]
    top_states_q: torch.Tensor, # [B L H]
    top_states_k: torch.Tensor, # [B L H]
    sg_indices_q: torch.Tensor, # [B L H]
    sg_indices_k: torch.Tensor, # [B L H]
    num_classes: int,
    local_size: int,
    global_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    return BOSS_Attention.apply(
        q,
        k,
        v,
        top_states_q,
        top_states_k,
        sg_indices_q,
        sg_indices_k,
        num_classes,
        local_size,
        global_size,
        cu_seqlens,
    )


'''
==================================================
NOTE scatter boss python functions
==================================================
'''
def scatter_boss_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    softmax_scale: float,
    state_lambda_q: torch.Tensor, # [B H L_S]
    indices_q: torch.Tensor, # [B H L_S]
    indices_k: torch.Tensor, # [B H L_S]
    hist_q: torch.Tensor, # [B H NUMSTATES + 1]
    hist_k: torch.Tensor, # [B H NUMSTATES + 1]
    local_size: int,
    global_size: int,
    max_bin_q: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_trunclens: Optional[torch.Tensor] = None,
):
    

    B, H, L_S = state_lambda_q.shape
    NUM_STATES = hist_q.shape[-1] - 1
    _, L, _, D = q.shape
    
    
    if cu_trunclens is not None:
        USE_CUSEQLENS = True
        num_samples = cu_seqlens.shape[0] - 1
    else:
        USE_CUSEQLENS = False
        num_samples = B
    CHUNK_SIZE = 64
    NT = triton.cdiv(max_bin_q, CHUNK_SIZE)
    o = torch.zeros_like(q)
    lse = torch.zeros((B, H, L), dtype=torch.float32, device=q.device)
    grid = (num_samples*H, NUM_STATES, NT)
    scatter_boss_fwd_kernel[grid](
        q=q,
        k=k,
        v=v,
        o=o,
        lse=lse,
        softmax_scale=softmax_scale,
        state_lambda_q=state_lambda_q,
        hist_q=hist_q,
        hist_k=hist_k,
        indices_q=indices_q,
        indices_k=indices_k,
        cu_seqlens=cu_seqlens,
        cu_trunclens=cu_trunclens,
        L=L,
        L_S=L_S,
        USE_CUSEQLENS=USE_CUSEQLENS,
        NUM_STATES=NUM_STATES,
        LOCAL_SIZE=local_size,
        GLOBAL_SIZE=global_size,
        BLOCK_M=CHUNK_SIZE,
        BLOCK_N=64,
        H=H,
        D=D,
    )
    return o, lse

def scatter_boss_bwd(
    do: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    d_sgq: torch.Tensor,
    d_sgk: torch.Tensor,
    lse: torch.Tensor,
    delta: torch.Tensor,
    softmax_scale: float,
    state_lambda_q: torch.Tensor, # [B H L_S]
    state_lambda_k: torch.Tensor, # [B H L_S]
    indices_q: torch.Tensor, # [B H L_S]
    indices_k: torch.Tensor, # [B H L_S]
    hist_q: torch.Tensor, # [B H NUMSTATES + 1]
    hist_k: torch.Tensor, # [B H NUMSTATES + 1]
    local_size: int,
    global_size: int,
    max_bin_q: Optional[int] = None,
    max_bin_k: Optional[int] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_trunclens: Optional[torch.Tensor] = None,
):
    B, H, L_S = state_lambda_q.shape
    NUM_STATES = hist_q.shape[-1] - 1
    _, L, _, D = q.shape
    if cu_trunclens is not None:
        USE_CUSEQLENS = True
        num_samples = cu_seqlens.shape[0] - 1
    else:
        USE_CUSEQLENS = False
        num_samples = B
    CHUNK_SIZE = 64
    NT = triton.cdiv(max_bin_k, CHUNK_SIZE)
    grid = (num_samples*H, NUM_STATES, NT)

    scatter_boss_bwd_dkvsg_kernel[grid](
        q=q,
        k=k,
        v=v,
        do=do,
        dk=dk,
        dv=dv,
        d_sgk=d_sgk,
        lse=lse,
        delta=delta,
        softmax_scale=softmax_scale,
        state_lambda_q=state_lambda_q,
        state_lambda_k=state_lambda_k,
        hist_q=hist_q,
        hist_k=hist_k,
        indices_q=indices_q,
        indices_k=indices_k,
        cu_seqlens=cu_seqlens,
        cu_trunclens=cu_trunclens,
        L=L,
        L_S=L_S,
        USE_CUSEQLENS=USE_CUSEQLENS,
        NUM_STATES=NUM_STATES,
        LOCAL_SIZE=local_size,
        GLOBAL_SIZE=global_size,
        BLOCK_M=64,
        BLOCK_N=CHUNK_SIZE,
        H=H,
        D=D,
    )
    NT = triton.cdiv(max_bin_q, CHUNK_SIZE)
    grid = (num_samples*H, NUM_STATES, NT)
    scatter_boss_bwd_dqsg_kernel[grid](
        q=q,
        k=k,
        v=v,
        do=do,
        dq=dq,
        d_sgq=d_sgq,
        lse=lse,
        delta=delta,
        softmax_scale=softmax_scale,
        state_lambda_q=state_lambda_q,
        hist_q=hist_q,
        hist_k=hist_k,
        indices_q=indices_q,
        indices_k=indices_k,
        cu_seqlens=cu_seqlens,
        cu_trunclens=cu_trunclens,
        L=L,
        L_S=L_S,
        USE_CUSEQLENS=USE_CUSEQLENS,
        NUM_STATES=NUM_STATES,
        LOCAL_SIZE=local_size,
        GLOBAL_SIZE=global_size,
        BLOCK_M=CHUNK_SIZE,
        BLOCK_N=64,
        H=H,
        D=D,
    )
    return dq, dk, dv

'''
==================================================
NOTE scatter boss triton kernels
==================================================
'''
@triton.jit
def scatter_boss_fwd_kernel(
    q, # [B L H D] / [1 L H D]
    k, # [B L H D] / [1 L H D]
    v, # [B L H D] / [1 L H D]
    o, # [B L H D] / [1 L H D]
    lse, # [B, H, L]
    softmax_scale, # float
    state_lambda_q, # [B H L_S] / [1 H L_S]
    hist_q, # [B H NUMSTATES + 1]
    hist_k,  # [B H NUMSTATES + 1]
    indices_q, # [B H L_S] / [1 H L_S]
    indices_k, # [B H L_S] / [1 H L_S]
    cu_seqlens, # [B + 1]
    cu_trunclens, # [B + 1]
    L,
    L_S,
    USE_CUSEQLENS:tl.constexpr,
    NUM_STATES:tl.constexpr,
    LOCAL_SIZE:tl.constexpr,
    GLOBAL_SIZE:tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # NOTE get block position
    bh_id, e_id, m_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_id, h_id = bh_id // H, bh_id % H
    # NOTE now we need to find out the length of current state bin
    ns_offset = (b_id*H + h_id)*(NUM_STATES+1) + e_id
    qbos, qeos = tl.load(hist_q + ns_offset).to(tl.int32), tl.load(hist_q + ns_offset + 1).to(tl.int32)
    qT = qeos - qbos
    # NOTE early return mechanism
    if m_id * BLOCK_M >= qT:
        return
    # NOTE kT is the bound of row length
    kbos, keos = tl.load(hist_k + ns_offset).to(tl.int32), tl.load(hist_k + ns_offset + 1).to(tl.int32)
    kT = keos - kbos
    # NOTE get the indices of indices_q specified bin and get the mask
    qidx_offset = m_id * BLOCK_M + tl.arange(0, BLOCK_M)  
    qidx_mask = qidx_offset < qT
    # NOTE process scattered columns
    if USE_CUSEQLENS:
        lens_bos, trun_bos = tl.load(cu_seqlens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id).to(tl.int32)
        bhn_offset = h_id*L_S + trun_bos
    else:
        bhn_offset = (b_id*H + h_id)*L_S
    qidx_bias = bhn_offset + qbos
    # NOTE get the ptr of Q
    p_qidx = indices_q + qidx_bias + qidx_offset
    b_qidx = tl.load(p_qidx, mask=qidx_mask, other=0)
    if USE_CUSEQLENS:
        qo_offset = ((lens_bos+LOCAL_SIZE+b_qidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        p_lse = lse + h_id*L + (lens_bos+LOCAL_SIZE+b_qidx)
        p_lmd = state_lambda_q + bhn_offset + b_qidx
        p_start = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + m_id * BLOCK_M).to(tl.int32)
        p_end = state_lambda_q + bhn_offset+tl.load(indices_q + qidx_bias + tl.minimum((m_id + 1) * BLOCK_M, qT) - 1).to(tl.int32)
    else:
        qo_offset = ((b_id*L + LOCAL_SIZE+b_qidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        p_lse = lse + (b_id*H + h_id)*L + LOCAL_SIZE+b_qidx
        p_lmd = state_lambda_q + bhn_offset + b_qidx
        p_start = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + m_id * BLOCK_M).to(tl.int32)
        p_end = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + tl.minimum((m_id + 1) * BLOCK_M, qT) - 1).to(tl.int32)
    
    # NOTE get the start and the end position of BLOCK_N moving
    N_START = tl.maximum(tl.load(p_start) - GLOBAL_SIZE + 1, 0)
    N_END = tl.minimum(tl.load(p_end) + 1, kT)
    # N_START = 0
    # N_END = kT

    if N_END == 0:
        return
    
    # NOTE row part preparation
    kidx_bias = bhn_offset + kbos
    kidx_offset = N_START + tl.arange(0, BLOCK_N)
    kidx_mask = kidx_offset < kT
    p_kidx = indices_k + kidx_bias + kidx_offset
    # NOTE for aux tensors
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - 1e8
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32) + 1.0
    acc_o = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    # NOTE prepare the q, o ptr
    p_q = q + qo_offset
    p_o = o + qo_offset
    b_q = tl.load(p_q, mask=qidx_mask[:, None], other=0.0)
    b_lmd = tl.load(p_lmd, mask=qidx_mask, other=-1) # the -1 padding can avoid bad attention mask
    b_q = (b_q * softmax_scale).to(b_q.dtype)
    # NOTE flash attention, start!
    for n_position in range(N_START, N_END, BLOCK_N):
        # NOTE get the ptr of K
        b_kidx = tl.load(p_kidx, mask=kidx_mask, other=0).to(tl.int32)
        if USE_CUSEQLENS:
            kv_offset = ((lens_bos + b_kidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        else:
            kv_offset = ((b_id*L + b_kidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        p_k = k + kv_offset
        p_v = v + kv_offset
        b_k = tl.load(p_k, mask=kidx_mask[:, None], other=0.0)
        # NOTE here we compute flash attention
        # NOTE compute the qk
        b_qk = tl.dot(b_q, tl.trans(b_k))
        rel_map = b_lmd[:, None] - kidx_offset[None, :]
        b_qk = tl.where((rel_map >= 0) & (rel_map < GLOBAL_SIZE), b_qk, float("-inf"))
        # NOTE the numerical stable of exp ops
        m_ij = tl.maximum(m_i, tl.max(b_qk, 1))
        b_p = tl.exp(b_qk - m_ij[:, None])
        num_scale = tl.exp(m_i - m_ij)
        l_i = num_scale * l_i + tl.sum(b_p, 1)
        m_i = m_ij
        acc_o = acc_o * num_scale[:, None]
        # NOTE accumulate the output
        b_v = tl.load(p_v, mask=kidx_mask[:, None], other=0.0)
        b_o = tl.dot(b_p.to(b_v.dtype), b_v).to(acc_o.dtype)
        acc_o += b_o
        # NOTE move the index a BLOCK_N offset
        kidx_offset += BLOCK_N
        kidx_mask = kidx_offset < kT
        p_kidx = indices_k + kidx_bias + kidx_offset
    # NOTE update the output
    acc_o = acc_o / l_i[:, None]
    L_i = m_i + tl.log(l_i)
    # NOTE store the output
    tl.store(p_o, acc_o.to(p_o.dtype.element_ty), mask=tl.broadcast_to(qidx_mask[:, None], (BLOCK_M, D)))
    # NOTE process the logsumexp L
    tl.store(p_lse, L_i.to(p_lse.dtype.element_ty), mask=qidx_mask)

@triton.jit
def scatter_boss_bwd_dqsg_kernel(
    q, # [B L H D] / [1 L H D]
    k, # [B L H D] / [1 L H D]
    v, # [B L H D] / [1 L H D]
    do, # [B L H D] / [1 L H D]
    dq, # [B L H D] / [1 L H D]
    d_sgq, # [B L H]
    lse, # [B, H, L]
    delta, # [B, H, L]
    softmax_scale, # float
    state_lambda_q, # [B H L_S] / [1 H L_S]
    hist_q, # [B H NUMSTATES + 1]
    hist_k,  # [B H NUMSTATES + 1]
    indices_q, # [B H L_S] / [1 H L_S]
    indices_k, # [B H L_S] / [1 H L_S]
    cu_seqlens, # [B + 1]
    cu_trunclens, # [B + 1]
    L,
    L_S,
    USE_CUSEQLENS:tl.constexpr,
    NUM_STATES:tl.constexpr,
    LOCAL_SIZE:tl.constexpr,
    GLOBAL_SIZE:tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # NOTE get block position
    bh_id, e_id, m_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_id, h_id = bh_id // H, bh_id % H
    # NOTE now we need to find out the length of current state bin
    ns_offset = (b_id*H + h_id)*(NUM_STATES+1) + e_id
    qbos, qeos = tl.load(hist_q + ns_offset).to(tl.int32), tl.load(hist_q + ns_offset + 1).to(tl.int32)
    qT = qeos - qbos
    # NOTE early return mechanism
    if m_id * BLOCK_M >= qT:
        return
    # NOTE kT is the bound of row length
    kbos, keos = tl.load(hist_k + ns_offset).to(tl.int32), tl.load(hist_k + ns_offset + 1).to(tl.int32)
    kT = keos - kbos
    # NOTE get the indices of indices_q specified bin and get the mask
    qidx_offset = m_id * BLOCK_M + tl.arange(0, BLOCK_M)  
    qidx_mask = qidx_offset < qT
    # NOTE process scattered columns
    if USE_CUSEQLENS:
        lens_bos, trun_bos = tl.load(cu_seqlens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id).to(tl.int32)
        bhn_offset = h_id*L_S + trun_bos
    else:
        bhn_offset = (b_id*H + h_id)*L_S
    qidx_bias = bhn_offset + qbos
    # NOTE get the ptr of Q
    p_qidx = indices_q + qidx_bias + qidx_offset
    b_qidx = tl.load(p_qidx, mask=qidx_mask, other=0)
    if USE_CUSEQLENS:
        qo_offset = ((lens_bos+LOCAL_SIZE+b_qidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        # dsgq_offset = ((lens_bos+LOCAL_SIZE+b_qidx)*H + h_id)

        p_lse = lse + h_id*L + (lens_bos+LOCAL_SIZE+b_qidx)
        p_delta = delta + h_id*L + (lens_bos+LOCAL_SIZE+b_qidx)
        p_lmd = state_lambda_q + bhn_offset +b_qidx
        p_start = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + m_id * BLOCK_M)
        p_end = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + tl.minimum((m_id + 1) * BLOCK_M, qT) - 1)
    else:
        qo_offset = ((b_id*L + LOCAL_SIZE+b_qidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        # dsgq_offset = ((b_id*L + LOCAL_SIZE+b_qidx)*H + h_id)

        p_lse = lse + (b_id*H + h_id)*L + LOCAL_SIZE+b_qidx
        p_delta = delta + (b_id*H + h_id)*L + LOCAL_SIZE+b_qidx
        p_lmd = state_lambda_q + bhn_offset + b_qidx
        p_start = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + m_id * BLOCK_M)
        p_end = state_lambda_q + bhn_offset + tl.load(indices_q + qidx_bias + tl.minimum((m_id + 1) * BLOCK_M, qT) - 1)
    
    # NOTE get the start and the end position of BLOCK_N moving
    N_START = tl.maximum(tl.load(p_start) - GLOBAL_SIZE + 1, 0)
    N_END = tl.load(p_end) + 1
    if N_END == 0:
        return
    # N_START = 0
    # N_END = kT

    # NOTE row part preparation
    kidx_bias = bhn_offset + kbos
    kidx_offset = N_START + tl.arange(0, BLOCK_N)
    kidx_mask = kidx_offset < kT
    p_kidx = indices_k + kidx_bias + kidx_offset
    # NOTE aux tensor
    acc_dq = tl.zeros([BLOCK_M, D], dtype=tl.float32)
    # NOTE prepare the q, o ptr
    p_q = q + qo_offset

    # p_dsgq = d_sgq + dsgq_offset

    p_do = do + qo_offset
    p_dq = dq + qo_offset
    b_q = tl.load(p_q, mask=qidx_mask[:, None], other=0)
    b_do = tl.load(p_do, mask=qidx_mask[:, None], other=0)
    b_lmd = tl.load(p_lmd, mask=qidx_mask, other=-1) # the -1 padding can avoid bad attention mask
    b_lse = tl.load(p_lse, mask=qidx_mask, other=0)
    b_delta = tl.load(p_delta, mask=qidx_mask, other=0)
    b_q = (b_q * softmax_scale).to(b_q.dtype)
    # NOTE flash attention bwd, start!
    for n_position in range(N_START, N_END, BLOCK_N):
        # NOTE get the ptr of K
        b_kidx = tl.load(p_kidx, mask=kidx_mask, other=0)
        if USE_CUSEQLENS:
            kv_offset = ((lens_bos+b_kidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        else:
            kv_offset = ((b_id*L + b_kidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        p_k = k + kv_offset
        p_v = v + kv_offset
        b_k = tl.load(p_k, mask=kidx_mask[:, None], other=0)
        # NOTE here we compute the qk
        b_qk = tl.dot(b_q, tl.trans(b_k))
        rel_map = b_lmd[:, None] - kidx_offset[None, :]
        b_qk = tl.where((rel_map >= 0) & (rel_map < GLOBAL_SIZE), b_qk, float("-inf"))
        # NOTE numerical stablization with global logsumexp
        b_p = tl.exp(b_qk - b_lse[:, None])
        # NOTE load v and compute the related
        b_v = tl.load(p_v, mask=kidx_mask[:, None], other=0)
        b_dp = tl.dot(b_do.to(b_v.dtype), tl.trans(b_v))
        b_dqk = b_p * (b_dp - b_delta[:, None]) * softmax_scale
        acc_dq += tl.dot(b_dqk.to(b_k.dtype), b_k).to(acc_dq.dtype)
        # NOTE update the index a BLOCK_N offset
        kidx_offset += BLOCK_N
        kidx_mask = kidx_offset < kT
        p_kidx = indices_k + kidx_bias + kidx_offset
    # NOTE store the dq
    tl.store(p_dq, acc_dq.to(b_q.dtype), mask=qidx_mask[:, None])
    # # NOTE store d_sgq
    # b_dsgq = tl.sum((acc_dq * b_q), axis=1)
    # tl.store(p_dsgq, b_dsgq.to(p_dsgq.dtype.element_ty), mask=qidx_mask)

@triton.jit
def scatter_boss_bwd_dkvsg_kernel(
    q, # [B L H D] / [1 L H D]
    k, # [B L H D] / [1 L H D]
    v, # [B L H D] / [1 L H D]
    do, # [B L H D] / [1 L H D]
    dk, # [B L H D] / [1 L H D]
    dv, # [B L H D] / [1 L H D]
    d_sgk, # [B L H]
    lse, # [B, H, L]
    delta, # [B, H, L]
    softmax_scale, # float
    state_lambda_q, # [B H L_S] / [1 H L_S]
    state_lambda_k, # [B H L_S] / [1 H L_S]
    hist_q, # [B H NUMSTATES + 1]
    hist_k,  # [B H NUMSTATES + 1]
    indices_q, # [B H L_S] / [1 H L_S]
    indices_k, # [B H L_S] / [1 H L_S]
    cu_seqlens, # [B + 1]
    cu_trunclens, # [B + 1]
    L,
    L_S,
    USE_CUSEQLENS:tl.constexpr,
    NUM_STATES:tl.constexpr,
    LOCAL_SIZE:tl.constexpr,
    GLOBAL_SIZE:tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # NOTE get block position
    bh_id, e_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    b_id, h_id = bh_id // H, bh_id % H
    # NOTE now we need to find out the length of current state bin
    ns_offset = (b_id*H + h_id)*(NUM_STATES+1) + e_id
    # NOTE kT is the bound of row length
    kbos, keos = tl.load(hist_k + ns_offset).to(tl.int32), tl.load(hist_k + ns_offset + 1).to(tl.int32)
    kT = keos - kbos
    # NOTE early return mechanism
    if n_id * BLOCK_N >= kT:
        return
    qbos, qeos = tl.load(hist_q + ns_offset).to(tl.int32), tl.load(hist_q + ns_offset + 1).to(tl.int32)
    qT = qeos - qbos
    kidx_offset = n_id * BLOCK_N + tl.arange(0, BLOCK_N)
    kidx_mask = kidx_offset < kT
    # NOTE process the scatter row
    if USE_CUSEQLENS:
        lens_bos, trun_bos = tl.load(cu_seqlens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id).to(tl.int32)
        bhn_offset = h_id*L_S + trun_bos
    else:
        bhn_offset = (b_id*H + h_id)*L_S
    kidx_bias = bhn_offset + kbos
    p_kidx = indices_k + kidx_bias + kidx_offset
    b_kidx = tl.load(p_kidx, mask=kidx_mask, other=0)
    if USE_CUSEQLENS:
        kv_offset = ((lens_bos+b_kidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        # dsgk_offset = ((lens_bos+b_kidx)*H + h_id)

        p_start = state_lambda_k + bhn_offset + tl.load(indices_k + kidx_bias + n_id * BLOCK_N)
    else:
        kv_offset = ((b_id*L + b_kidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
        # dsgk_offset = ((b_id*L + b_kidx)*H + h_id)

        p_start = state_lambda_k + bhn_offset + tl.load(indices_k + kidx_bias + n_id * BLOCK_N)
        
    M_START = tl.maximum(tl.load(p_start), 0)
    if (n_id + 1) * BLOCK_N + GLOBAL_SIZE >= kT:
        M_END = qT
    else:
        if USE_CUSEQLENS:
            p_end = state_lambda_k + bhn_offset + tl.load(indices_k + kidx_bias + tl.minimum((n_id + 1) * BLOCK_N + GLOBAL_SIZE, kT) - 1)
        else:
            p_end = state_lambda_k + bhn_offset + tl.load(indices_k + kidx_bias + tl.minimum((n_id + 1) * BLOCK_N + GLOBAL_SIZE, kT) - 1)
        M_END = tl.load(p_end) + 1
    # M_END = tl.load(p_end) + 1

    # M_START = 0
    # M_END = qT
    
    # NOTE column preparation
    qidx_bias = bhn_offset + qbos
    qidx_offset = M_START + tl.arange(0, BLOCK_M)
    qidx_mask = qidx_offset < qT
    p_qidx = indices_q + qidx_bias + qidx_offset

    p_k = k + kv_offset
    p_v = v + kv_offset
    p_dk = dk + kv_offset
    p_dv = dv + kv_offset

    # p_dsgk = d_sgk + dsgk_offset
    
    b_k = tl.load(p_k, mask=kidx_mask[:, None], other=0)
    b_v = tl.load(p_v, mask=kidx_mask[:, None], other=0)
    b_k = (b_k * softmax_scale).to(b_k.dtype)
    # NOTE get accumulate diff k, v
    acc_dk = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    acc_dv = tl.zeros([BLOCK_N, D], dtype=tl.float32)
    for m_position in range(M_START, M_END, BLOCK_M):
        # NOTE get the ptr of Q
        b_qidx = tl.load(p_qidx, mask=qidx_mask, other=0)
        if USE_CUSEQLENS:
            qo_offset = ((lens_bos+LOCAL_SIZE+b_qidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
            dl_offset = h_id*L + (lens_bos+LOCAL_SIZE+b_qidx)
            p_lmd = state_lambda_q + h_id*L_S + (trun_bos+b_qidx)
        else:
            qo_offset = ((b_id*L + LOCAL_SIZE+b_qidx[:, None])*H + h_id)*D + tl.arange(0, D)[None, :]
            dl_offset = (b_id*H + h_id)*L + LOCAL_SIZE+b_qidx
            p_lmd = state_lambda_q + (b_id*H + h_id)*L_S + b_qidx
        p_q = q + qo_offset
        p_do = do + qo_offset
        p_lse = lse + dl_offset
        p_delta = delta + dl_offset

        b_q = tl.load(p_q, mask=qidx_mask[:, None], other=0)
        b_do = tl.load(p_do, mask=qidx_mask[:, None], other=0)
        b_lmd = tl.load(p_lmd, mask=qidx_mask, other=-1) # the -1 padding can avoid bad attention mask
        b_lse = tl.load(p_lse, mask=qidx_mask, other=0)
        b_delta = tl.load(p_delta, mask=qidx_mask, other=0)
        # NOTE compute 
        b_qk = tl.dot(b_q, tl.trans(b_k))
        rel_map = b_lmd[:, None] - kidx_offset[None, :]
        b_qk = tl.where((rel_map >= 0) & (rel_map < GLOBAL_SIZE), b_qk, float("-inf"))
        # NOTE numerical stablization with global logsumexp
        b_p = tl.exp(b_qk - b_lse[:, None])
        acc_dv += tl.dot(tl.trans(b_p.to(b_do.dtype)), b_do).to(acc_dv.dtype)
        b_dp = tl.dot(b_do.to(b_v.dtype), tl.trans(b_v))
        b_dqk = b_p * (b_dp - b_delta[:, None]) * softmax_scale
        # NOTE accumulate dk
        acc_dk += tl.dot(tl.trans(b_dqk.to(b_q.dtype)), b_q).to(acc_dk.dtype)
        # NOTE update the index a BLOCK_M offset
        qidx_offset += BLOCK_M
        qidx_mask = qidx_offset < qT
        p_qidx = indices_q + qidx_bias + qidx_offset
    # NOTE store acc_k, acc_v
    tl.store(p_dk, acc_dk.to(p_dk.dtype.element_ty), mask=kidx_mask[:, None])
    tl.store(p_dv, acc_dv.to(p_dv.dtype.element_ty), mask=kidx_mask[:, None])
    # # NOTE store d_sgk
    # b_dsgk = tl.sum((acc_dk * b_k), axis=1)
    # tl.store(p_dsgk, b_dsgk.to(p_dsgk.dtype.element_ty), mask=kidx_mask)

@triton.jit
def weighted_merge_o(
    o, # [B L H D]
    lse, # [B H L]
    o_boss, # [B L H D]
    lse_boss, # [B H L]
    cu_seqlens, # [B + 1]
    cu_trunclens, # [B + 1]
    L,
    LOCAL_SIZE: tl.constexpr,
    USE_CUSEQLENS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # NOTE get block position
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        lens_bos = tl.load(cu_seqlens + b_id).to(tl.int32)
        trunc_bos, trunc_eos = tl.load(cu_trunclens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id + 1).to(tl.int32)
        T = trunc_eos - trunc_bos
    else:
        T = L - LOCAL_SIZE
    if n_id * CHUNK_SIZE >= T:
        return
    # NOTE get p_o, p_l, p_ob, p_lb
    if USE_CUSEQLENS:
        p_o1 = tl.make_block_ptr(o + ((lens_bos+LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_o2 = tl.make_block_ptr(o_boss + ((lens_bos+LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_l1 = tl.make_block_ptr(lse + (h_id*L + lens_bos+LOCAL_SIZE), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_l2 = tl.make_block_ptr(lse_boss + (h_id*L + lens_bos+LOCAL_SIZE), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    else:
        p_o1 = tl.make_block_ptr(o + (((b_id*L) + LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_o2 = tl.make_block_ptr(o_boss + (((b_id*L) + LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_l1 = tl.make_block_ptr(lse + (((b_id*H) + h_id)*L + LOCAL_SIZE), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_l2 = tl.make_block_ptr(lse_boss + (((b_id*H) + h_id)*L + LOCAL_SIZE), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    b_l1 = tl.load(p_l1, boundary_check=(0,))
    b_l2 = tl.load(p_l2, boundary_check=(0,))
    b_l = b_l1 - tl.log(tl.sigmoid(b_l1 - b_l2))
    b_dl = tl.sigmoid(b_l2 - b_l1)
    
    b_o2 = tl.load(p_o2, boundary_check=(0, 1)).to(tl.float32)
    b_ob = b_o2 * b_dl[:, None]
    
    b_o1 = tl.load(p_o1, boundary_check=(0, 1)).to(tl.float32)
    b_o = b_o1 + (b_o2 - b_o1) * b_dl[:, None]   
    tl.debug_barrier()
    tl.store(p_l1, b_l.to(p_l1.dtype.element_ty), boundary_check=(0,))
    tl.store(p_o1, b_o.to(p_o1.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_o2, b_ob.to(p_o2.dtype.element_ty), boundary_check=(0, 1))
    
    

@triton.jit
def dweight_scatter_o(
    do, # [B L H D]
    o, # [B L H D]
    do_boss, # [B L H D]
    delta, # [B H L]
    cu_seqlens, # [B + 1]
    cu_trunclens, # [B + 1]
    L,
    LOCAL_SIZE: tl.constexpr,
    USE_CUSEQLENS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
):
    # NOTE get block position
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        lens_bos = tl.load(cu_seqlens + b_id).to(tl.int32)
        trunc_bos, trunc_eos = tl.load(cu_trunclens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id + 1).to(tl.int32)
        T = trunc_eos - trunc_bos
    else:
        T = L - LOCAL_SIZE
    if n_id * CHUNK_SIZE >= T:
        return
    # NOTE get p_o, p_l, p_ob, p_lb, p_w
    if USE_CUSEQLENS:
        p_o = tl.make_block_ptr(o + ((lens_bos+LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_do = tl.make_block_ptr(do + ((lens_bos+LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_dob = tl.make_block_ptr(do_boss + ((lens_bos+LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_d = tl.make_block_ptr(delta + (h_id*L + lens_bos+LOCAL_SIZE), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))

        # p_ob = tl.make_block_ptr(o_boss + ((lens_bos+LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        # p_w = tl.make_block_ptr(router_weight + ((lens_bos+LOCAL_SIZE)*H + h_id), (T,), (H,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        # p_dw = tl.make_block_ptr(drouter_weight + ((lens_bos+LOCAL_SIZE)*H + h_id), (T,), (H,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    else:
        p_o = tl.make_block_ptr(o + (((b_id*L) + LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_do = tl.make_block_ptr(do + (((b_id*L) + LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_dob = tl.make_block_ptr(do_boss + (((b_id*L) + LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        p_d = tl.make_block_ptr(delta + (((b_id*H) + h_id)*L + LOCAL_SIZE), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        # p_ob = tl.make_block_ptr(o_boss + (((b_id*L) + LOCAL_SIZE)*H + h_id)*D, (T, D), (H*D, 1), (n_id*CHUNK_SIZE, 0), (CHUNK_SIZE, D), (1, 0))
        # p_w = tl.make_block_ptr(router_weight + (((b_id*L) + LOCAL_SIZE)*H + h_id), (T,), (H,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        # p_dw = tl.make_block_ptr(drouter_weight + (((b_id*L) + LOCAL_SIZE)*H + h_id), (T,), (H,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    
    b_do = tl.load(p_do, boundary_check=(0, 1)).to(tl.float32)
    b_o = tl.load(p_o, boundary_check=(0, 1)).to(tl.float32)

    # b_w = tl.load(p_w, boundary_check=(0,))
    # b_ob = tl.load(p_ob, boundary_check=(0, 1)).to(tl.float32)
    # b_dob = b_do * b_w[:, None]
    
    # b_dw = tl.sum(b_do*b_ob, axis=1)

    b_d = tl.sum(b_do*b_o, axis=1)

    tl.debug_barrier()
    tl.store(p_dob, b_do.to(p_dob.dtype.element_ty), boundary_check=(0, 1))
    tl.store(p_d, b_d.to(p_d.dtype.element_ty), boundary_check=(0,))

    # tl.store(p_dw, b_dw.to(p_dw.dtype.element_ty), boundary_check=(0,))

'''
==================================================
NOTE scatter boss prepare functions
==================================================
'''

@dataclass
class BossPrepareResult:
    indices_q: Optional[torch.Tensor]
    indices_k: Optional[torch.Tensor]
    state_lambda_q: Optional[torch.Tensor]
    state_lambda_k: Optional[torch.Tensor]
    hist_q: Optional[torch.Tensor]
    hist_k: Optional[torch.Tensor]
    cu_trunclens: Optional[torch.Tensor]
    USE_CUSEQLENS: Optional[bool]
    B: Optional[int]
    L_S: Optional[int]
    max_seqlen: Optional[int]
    max_bin_q: Optional[int]
    max_bin_k: Optional[int]
    valid: bool

def boss_prepare(
    top_states_q: torch.Tensor,
    top_states_k: torch.Tensor,
    shift_size: int,
    num_classes: int,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> BossPrepareResult:
    '''
    return: indices_q, indices_k, state_lambda_q, state_lambda_k, hist_q, hist_k, cu_trunclens, cu_chunklens, True
    '''

    NUM_STATES = next_power_of_2(num_classes)
    B, L, H = top_states_q.shape
    CHUNK_SIZE = 128
    
    if cu_seqlens is not None:
        USE_CUSEQLENS = True
        assert cu_seqlens.dtype == torch.int32
        assert B == 1
        seqlens = torch.diff(cu_seqlens, dim=0)
        B = seqlens.shape[0]
        cu_trunclens = torch.zeros((B + 1,), dtype=torch.int32, device=top_states_q.device)
        cu_chunklens = torch.zeros((B + 1,), dtype=torch.int32, device=top_states_q.device)
        cu_trunc_chunk_kernel[(1,)](
            seqlens=seqlens,
            cu_trunclens=cu_trunclens,
            cu_chunklens=cu_chunklens,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
        )
        trunc_seqlens = torch.diff(cu_trunclens, dim=0)
        max_seqlen = trunc_seqlens.max().item()
        # NOTE if length is shorter than shift size, we don't have to compute boss part
        if max_seqlen == 0:
            return BossPrepareResult(
                indices_q=None,
                indices_k=None,
                state_lambda_q=None,
                state_lambda_k=None,
                hist_q=None,
                hist_k=None,
                cu_trunclens=None,
                USE_CUSEQLENS=None,
                B=None,
                L_S=None,
                max_seqlen=None,
                max_bin_q=None,
                max_bin_k=None,
                valid=False,
            )
        L_S = cu_trunclens[-1].item()
        NT_L = cu_chunklens[-1].item()
        NT = triton.cdiv(max_seqlen, CHUNK_SIZE)
        # NOTE shift copy q, k
        top_states_q_shift = torch.zeros((1, H, L_S), dtype=torch.int32, device=top_states_q.device)
        top_states_k_shift = torch.zeros((1, H, L_S), dtype=torch.int32, device=top_states_q.device)
        shift_copy_q_kernel[(B, H, NT)](
            top_states=top_states_q,
            output=top_states_q_shift,
            cu_seqlens=cu_seqlens,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        shift_copy_k_kernel[(B, H, NT)](
            top_states=top_states_k,
            output=top_states_k_shift,
            cu_seqlens=cu_seqlens,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        # NOTE start to do block scan and histogram
        global_aux = torch.empty((1, H, NT_L, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_aux = torch.empty((1, H, NT_L, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_q = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        hist_k = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        state_lambda_q = torch.empty_like(top_states_q_shift)
        state_lambda_k = torch.empty_like(top_states_k_shift)
        state_lambda_k, hist_q = scan_hist_fused(
            top_states=top_states_q_shift,
            rel_states=top_states_k_shift,
            state_lambda=state_lambda_k,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_q,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT_L,
            CHUNK_SIZE=CHUNK_SIZE,
            cu_trunclens=cu_trunclens,
            cu_chunklens=cu_chunklens,
        )
        state_lambda_q, hist_k = scan_hist_fused(
            top_states=top_states_k_shift,
            rel_states=top_states_q_shift,
            state_lambda=state_lambda_q,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_k,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT_L,
            CHUNK_SIZE=CHUNK_SIZE,
            cu_trunclens=cu_trunclens,
            cu_chunklens=cu_chunklens,
        )
        radix_cu_seqlens = flatten_cuseqlens(B, L_S, H, top_states_q.device, cu_trunclens)
        bin_idx_q, indices_q = sort_varlen(top_states_q_shift.view(-1), radix_cu_seqlens, end_bit(num_classes))
        bin_idx_k, indices_k = sort_varlen(top_states_k_shift.view(-1), radix_cu_seqlens, end_bit(num_classes))
        indices_q_revised = torch.empty((1, H, L_S), dtype=torch.int32, device=top_states_q.device)
        indices_k_revised = torch.empty((1, H, L_S), dtype=torch.int32, device=top_states_q.device)
        indices_revise[(B, H, NT)](
            indices=indices_q,
            indices_revise=indices_q_revised,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            H=H,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        indices_revise[(B, H, NT)](
            indices=indices_k,
            indices_revise=indices_k_revised,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            H=H,
            CHUNK_SIZE=CHUNK_SIZE,
        )

    else:
        USE_CUSEQLENS = False
        # NOTE if length is shorter than shift size, we don't have to compute boss part
        if L <= shift_size:
            return BossPrepareResult(
                indices_q=None,
                indices_k=None,
                state_lambda_q=None,
                state_lambda_k=None,
                hist_q=None,
                hist_k=None,
                cu_trunclens=None,
                USE_CUSEQLENS=None,
                B=None,
                L_S=None,
                max_seqlen=None,
                max_bin_q=None,
                max_bin_k=None,
                valid=False,
            )
        max_seqlen=None
        L_S = L - shift_size
        NT = triton.cdiv(L_S, CHUNK_SIZE)

        # NOTE shift copy q, k
        top_states_q_shift = torch.zeros((B, H, L_S), dtype=torch.int32, device=top_states_q.device)
        top_states_k_shift = torch.zeros((B, H, L_S), dtype=torch.int32, device=top_states_q.device)
        shift_copy_q_kernel[(B, H, NT)](
            top_states=top_states_q,
            output=top_states_q_shift,
            cu_seqlens=None,
            cu_trunclens=None,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        shift_copy_k_kernel[(B, H, NT)](
            top_states=top_states_k,
            output=top_states_k_shift,
            cu_seqlens=None,
            cu_trunclens=None,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        # NOTE start to do block scan and histogram
        global_aux = torch.empty((B, H, NT, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_aux = torch.empty((B, H, NT, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_q = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        hist_k = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        state_lambda_q = torch.empty_like(top_states_q_shift)
        state_lambda_k = torch.empty_like(top_states_k_shift)
        state_lambda_k, hist_q = scan_hist_fused(
            top_states=top_states_q_shift,
            rel_states=top_states_k_shift,
            state_lambda=state_lambda_k,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_q,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        state_lambda_q, hist_k = scan_hist_fused(
            top_states=top_states_k_shift,
            rel_states=top_states_q_shift,
            state_lambda=state_lambda_q,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_k,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        cu_trunclens, cu_chunklens = None, None
        radix_cu_seqlens = flatten_cuseqlens(B, L_S, H, top_states_q.device, None)
        bin_idx_q, indices_q = sort_varlen(top_states_q_shift.view(-1), radix_cu_seqlens, end_bit(num_classes))
        bin_idx_k, indices_k = sort_varlen(top_states_k_shift.view(-1), radix_cu_seqlens, end_bit(num_classes))
        indices_q_revised = torch.empty((B, H, L_S), dtype=torch.int32, device=top_states_q.device)
        indices_k_revised = torch.empty((B, H, L_S), dtype=torch.int32, device=top_states_q.device)
        indices_revise[(B, H, NT)](
            indices=indices_q,
            indices_revise=indices_q_revised,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            H=H,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        indices_revise[(B, H, NT)](
            indices=indices_k,
            indices_revise=indices_k_revised,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            H=H,
            CHUNK_SIZE=CHUNK_SIZE,
        )
    max_bin_q = torch.diff(hist_q, dim=-1).max().item()
    max_bin_k = torch.diff(hist_k, dim=-1).max().item()
    
    return BossPrepareResult(
        indices_q=indices_q_revised,
        indices_k=indices_k_revised,
        state_lambda_q=state_lambda_q,
        state_lambda_k=state_lambda_k,
        hist_q=hist_q,
        hist_k=hist_k,
        cu_trunclens=cu_trunclens,
        USE_CUSEQLENS=USE_CUSEQLENS,
        B=B,
        L_S=L_S,
        max_seqlen=max_seqlen,
        max_bin_q=max_bin_q,
        max_bin_k=max_bin_k,
        valid=True,
    )

def next_power_of_2(n: int) -> int:
    if n <= 8:
        return 8
    return 1 << (n - 1).bit_length()

def end_bit(n: int) -> int:

    if n == 0:
        return -1
    return n.bit_length() - 1

'''
==================================================
NOTE scatter boss prepare triton kernel
==================================================
'''


@triton.jit
def cu_trunc_kernel(
    seqlens,
    cu_trunclens,
    SHIFT_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    '''
    This is the kernel to compute the truncated length of sequences
    '''
    NT = tl.cdiv(B, CHUNK_SIZE)
    tl.store(cu_trunclens, 0)
    p_lens = tl.make_block_ptr(seqlens, (B,), (1,), (0,), (CHUNK_SIZE,), (0,))
    p_trun = tl.make_block_ptr(cu_trunclens, (B + 1,), (1,), (1,), (CHUNK_SIZE,), (0,))
    b_s = tl.zeros((1,), dtype=tl.int32)
    for i in range(NT):
        LAST = tl.minimum(CHUNK_SIZE, B - i * CHUNK_SIZE) - 1
        b_idx = tl.full((1,), LAST, dtype=tl.int32)
        
        b_lens = tl.load(p_lens, boundary_check=(0,))
        b_sft = tl.maximum(0, b_lens - SHIFT_SIZE)
        b_cs = tl.cumsum(b_sft, axis=0) + b_s
        tl.store(p_trun, b_cs.to(p_trun.dtype.element_ty), boundary_check=(0,))
        
        b_s = tl.gather(b_cs, b_idx, axis=0)
        
        p_lens = tl.advance(p_lens, (CHUNK_SIZE,))
        p_trun = tl.advance(p_trun, (CHUNK_SIZE,))

@triton.jit
def cu_trunc_chunk_kernel(
    seqlens,
    cu_trunclens,
    cu_chunklens,
    SHIFT_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
):
    '''
    This is the kernel to compute the truncated length of sequences
    '''
    NT = tl.cdiv(B, CHUNK_SIZE)
    tl.store(cu_trunclens, 0)
    p_lens = tl.make_block_ptr(seqlens, (B,), (1,), (0,), (CHUNK_SIZE,), (0,))
    p_trun = tl.make_block_ptr(cu_trunclens, (B + 1,), (1,), (1,), (CHUNK_SIZE,), (0,))
    p_chun = tl.make_block_ptr(cu_chunklens, (B + 1,), (1,), (1,), (CHUNK_SIZE,), (0,))
    b_s = tl.zeros((1,), dtype=tl.int32)
    b_chunk_s = tl.zeros((1,), dtype=tl.int32)
    for i in range(NT):
        LAST = tl.minimum(CHUNK_SIZE, B - i * CHUNK_SIZE) - 1
        b_idx = tl.full((1,), LAST, dtype=tl.int32)
        
        b_lens = tl.load(p_lens, boundary_check=(0,))
        b_sft = tl.maximum(0, b_lens - SHIFT_SIZE)
        b_cs = tl.cumsum(b_sft, axis=0) + b_s
        tl.store(p_trun, b_cs.to(p_trun.dtype.element_ty), boundary_check=(0,))
        
        b_nt = tl.cdiv(b_sft, CHUNK_SIZE)
        b_nt = tl.where(b_sft == 0, 0, b_nt)
        b_chunk_cs = tl.cumsum(b_nt, axis=0) + b_chunk_s
        tl.store(p_chun, b_chunk_cs.to(p_chun.dtype.element_ty), boundary_check=(0,))

        b_s = tl.gather(b_cs, b_idx, axis=0)
        b_chunk_s = tl.gather(b_chunk_cs, b_idx, axis=0)
        
        p_lens = tl.advance(p_lens, (CHUNK_SIZE,))
        p_trun = tl.advance(p_trun, (CHUNK_SIZE,))
        p_chun = tl.advance(p_chun, (CHUNK_SIZE,))

@triton.jit
def shift_copy_q_kernel(
    top_states,
    output,
    cu_seqlens,
    cu_trunclens,
    L_S,
    USE_CUSEQLENS: tl.constexpr,
    SHIFT_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
    L,
    H: tl.constexpr,
):
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        bos, eos = tl.load(cu_seqlens + b_id).to(tl.int32), tl.load(cu_seqlens + b_id + 1).to(tl.int32)
        trunc_bos = tl.load(cu_trunclens + b_id).to(tl.int32)
        T = eos - bos
    else:
        T = L
    T_S = T - SHIFT_SIZE
    if n_id * CHUNK_SIZE >= T_S:
        return
    if USE_CUSEQLENS:
        p_ts = tl.make_block_ptr(top_states + ((bos + SHIFT_SIZE) * H + h_id), (T_S,), (H,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_o = tl.make_block_ptr(output + (h_id * L_S + trunc_bos), (T_S,), (1,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    else:
        p_ts = tl.make_block_ptr(top_states + ((b_id * L + SHIFT_SIZE) * H + h_id), (T_S,), (H,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_o = tl.make_block_ptr(output + ((b_id * H + h_id) * T_S), (T_S,), (1,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    b_ts = tl.load(p_ts, boundary_check=(0,))
    tl.store(p_o, b_ts.to(p_o.dtype.element_ty), boundary_check=(0,))

@triton.jit
def shift_copy_k_kernel(
    top_states,
    output,
    cu_seqlens,
    cu_trunclens,
    L_S,
    USE_CUSEQLENS: tl.constexpr,
    SHIFT_SIZE: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
    L,
    H: tl.constexpr,
):
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        bos, eos = tl.load(cu_seqlens + b_id).to(tl.int32), tl.load(cu_seqlens + b_id + 1).to(tl.int32)
        trunc_bos = tl.load(cu_trunclens + b_id).to(tl.int32)
        T = eos - bos
    else:
        T = L
    T_S = T - SHIFT_SIZE
    if n_id * CHUNK_SIZE >= T_S:
        return
    if USE_CUSEQLENS:
        p_ts = tl.make_block_ptr(top_states + (bos * H + h_id), (T_S,), (H,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_o = tl.make_block_ptr(output + (h_id * L_S + trunc_bos), (T_S,), (1,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    else:
        p_ts = tl.make_block_ptr(top_states + (b_id * L * H + h_id), (T_S,), (H,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_o = tl.make_block_ptr(output + ((b_id * H + h_id) * T_S), (T_S,), (1,), (n_id * CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    b_ts = tl.load(p_ts, boundary_check=(0,))
    tl.store(p_o, b_ts.to(p_o.dtype.element_ty), boundary_check=(0,))

def scan_hist_fused(
    top_states: torch.Tensor,
    rel_states: torch.Tensor,
    state_lambda: torch.Tensor,
    global_aux: torch.Tensor,
    hist_aux: torch.Tensor,
    hist: torch.Tensor,
    NUM_STATES: int,
    USE_CUSEQLENS: bool,
    B: int,
    L_S: int,
    H: int,
    NT: int,
    CHUNK_SIZE: int,
    cu_trunclens: Optional[torch.Tensor] = None,
    cu_chunklens: Optional[torch.Tensor] = None,
):
    local_scan_kernel[(B, H, NT)](
        top_states=top_states,
        rel_states=rel_states,
        state_lambda=state_lambda,
        global_aux=global_aux,
        hist_aux=hist_aux,
        cu_trunclens=cu_trunclens,
        cu_chunklens=cu_chunklens,
        USE_CUSEQLENS=USE_CUSEQLENS,
        NUM_STATES=NUM_STATES,
        CHUNK_SIZE=CHUNK_SIZE,
        B=B,
        L_S=L_S,
        H=H,
        NT=NT,
    )
    global_scan_kernel[(B, H)](
        global_aux=global_aux,
        hist_aux=hist_aux,
        hist=hist,
        cu_chunklens=cu_chunklens,
        USE_CUSEQLENS=USE_CUSEQLENS,
        NUM_STATES=NUM_STATES,
        CHUNK_SIZE=CHUNK_SIZE,
        B=B,
        H=H,
        NT=NT,
    )
    global_revise_kernel[(B, H, NT)](
        rel_states=rel_states,
        state_lambda=state_lambda,
        global_aux=global_aux,
        cu_trunclens=cu_trunclens,
        cu_chunklens=cu_chunklens,
        USE_CUSEQLENS=USE_CUSEQLENS,
        NUM_STATES=NUM_STATES,
        CHUNK_SIZE=CHUNK_SIZE,
        B=B,
        L_S=L_S,
        H=H,
        NT=NT,
    )
    return state_lambda, hist
    

@triton.jit
def local_scan_kernel(
    top_states, # [B, H, L_S]
    rel_states,
    state_lambda,
    global_aux, # [B, H, NT NUM_STATES]
    hist_aux,
    cu_trunclens,
    cu_chunklens,
    USE_CUSEQLENS: tl.constexpr,
    NUM_STATES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
    L_S,
    H: tl.constexpr,
    NT,
):
    '''
    Here we use the shifted version of inputs
    '''
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        bos, eos = tl.load(cu_trunclens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id + 1).to(tl.int32)
        T_S = eos - bos
        if n_id * CHUNK_SIZE >= T_S:
            return
        chunk_bos = tl.load(cu_chunklens + b_id).to(tl.int32)
    else:
        T_S = L_S
    # NOTE get the ptr
    ts_ptr = n_id * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    idx_mask = ts_ptr < T_S
    if USE_CUSEQLENS:
        p_ts = top_states + (h_id*L_S + bos) + ts_ptr
        p_rs = rel_states + (h_id*L_S + bos) + ts_ptr
        p_sl = state_lambda + (h_id*L_S + bos) + ts_ptr
        p_ga = tl.make_block_ptr(global_aux + (h_id*NT + chunk_bos+n_id)*NUM_STATES, (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
        p_ha = tl.make_block_ptr(hist_aux + (h_id*NT + chunk_bos+n_id)*NUM_STATES, (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
    else:
        p_ts = top_states + (b_id*H + h_id)*L_S + ts_ptr
        p_rs = rel_states + (b_id*H + h_id)*L_S + ts_ptr
        p_sl = state_lambda + (b_id*H + h_id)*L_S + ts_ptr
        p_ga = tl.make_block_ptr(global_aux + ((b_id*H + h_id)*NT + n_id)*NUM_STATES, (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
        p_ha = tl.make_block_ptr(hist_aux + ((b_id*H + h_id)*NT + n_id)*NUM_STATES, (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
    # NOTE load data
    b_aux = tl.arange(0, NUM_STATES)
    b_ts = tl.load(p_ts, mask=idx_mask, other=NUM_STATES) # the other is NUM_STATE to avoid padding values be histgramed
    b_hist = tl.histogram(b_ts, NUM_STATES)
    tl.store(p_ha, b_hist.to(p_ha.dtype.element_ty), boundary_check=(0,))

    # NOTE if the b_tos is on the boundary, dont worry since it will not be stored
    LAST = tl.minimum(CHUNK_SIZE, T_S - n_id * CHUNK_SIZE) - 1
    b_last = tl.full((1, NUM_STATES), LAST, dtype=tl.int32)
    b_toh = (b_ts[:, None] == b_aux[None, :]).to(tl.int32)
    b_cs = tl.cumsum(b_toh, axis=0)
    b_ga = tl.gather(b_cs, b_last, axis=0).view(NUM_STATES)
    tl.store(p_ga, b_ga.to(p_ga.dtype.element_ty), boundary_check=(0,))
    b_rs = tl.load(p_rs, mask=idx_mask, other=0)
    b_sl = tl.gather(b_cs, b_rs[:, None], axis=1).view(CHUNK_SIZE)
    tl.store(p_sl, b_sl.to(p_sl.dtype.element_ty), mask=idx_mask)

@triton.jit
def global_scan_kernel(
    global_aux, # [B, H, NT NUM_STATES]
    hist_aux, # [B, H, NT NUM_STATES]
    hist, # [B H NUM_STATES]
    cu_chunklens,
    USE_CUSEQLENS: tl.constexpr,
    NUM_STATES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    NT,
):
    '''
    Here we process the aux for global data
    '''
    b_id, h_id = tl.program_id(0), tl.program_id(1)
    if USE_CUSEQLENS:
        chunk_bos, chunk_eos = tl.load(cu_chunklens + b_id).to(tl.int32), tl.load(cu_chunklens + b_id + 1).to(tl.int32)
        NT_S = chunk_eos - chunk_bos
        if NT_S <= 0:
            return
    else:
        NT_S = NT
    NT_G = tl.cdiv(NT_S, CHUNK_SIZE)
    # NOTE do the ptr
    if USE_CUSEQLENS:
        p_ga = tl.make_block_ptr(global_aux + (h_id*NT + chunk_bos)*NUM_STATES, (NT_S, NUM_STATES), (NUM_STATES, 1), (0, 0), (CHUNK_SIZE, NUM_STATES), (1, 0))
        p_ha = tl.make_block_ptr(hist_aux + (h_id*NT + chunk_bos)*NUM_STATES, (NT_S, NUM_STATES), (NUM_STATES, 1), (0, 0), (CHUNK_SIZE, NUM_STATES), (1, 0))
        p_hist = tl.make_block_ptr(hist + ((b_id*H + h_id)*(NUM_STATES+1) + 1), (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
    else:
        p_ga = tl.make_block_ptr(global_aux + ((b_id*H + h_id)*NT)*NUM_STATES, (NT_S, NUM_STATES), (NUM_STATES, 1), (0, 0), (CHUNK_SIZE, NUM_STATES), (1, 0))
        p_ha = tl.make_block_ptr(hist_aux + ((b_id*H + h_id)*NT)*NUM_STATES, (NT_S, NUM_STATES), (NUM_STATES, 1), (0, 0), (CHUNK_SIZE, NUM_STATES), (1, 0))
        p_hist = tl.make_block_ptr(hist + ((b_id*H + h_id)*(NUM_STATES+1) + 1), (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
    # NOTE do the scan
    b_ss = tl.zeros((1, NUM_STATES), dtype=tl.int32)
    b_hs = tl.zeros((NUM_STATES,), dtype=tl.int32)
    b_last = tl.full((1, NUM_STATES), CHUNK_SIZE - 1, dtype=tl.int32)
    for i in range(NT_G):
        # NOTE process the scan
        b_ga = tl.load(p_ga, boundary_check=(0, 1))
        b_cs = tl.cumsum(b_ga, axis=0) + b_ss
        b_ss = tl.gather(b_cs, b_last, axis=0)
        tl.store(p_ga, b_cs.to(p_ga.dtype.element_ty), boundary_check=(0, 1))
        # NOTE process the histogram
        b_ha = tl.load(p_ha, boundary_check=(0, 1))
        b_hs += tl.sum(b_ha, axis=0)
        # NOTE move the ptrs
        p_ga = tl.advance(p_ga, (CHUNK_SIZE, 0))
        p_ha = tl.advance(p_ha, (CHUNK_SIZE, 0))
    b_hs = tl.cumsum(b_hs)
    tl.debug_barrier()
    tl.store(p_hist, b_hs.to(p_hist.dtype.element_ty), boundary_check=(0,))

@triton.jit
def global_revise_kernel(
    rel_states, # [B H L_S]
    state_lambda, # [B H L_S]
    global_aux, # [B, H, NT NUM_STATES]
    cu_trunclens,
    cu_chunklens,
    USE_CUSEQLENS: tl.constexpr,
    NUM_STATES: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    B: tl.constexpr,
    L_S,
    H: tl.constexpr,
    NT,
):
    '''
    Here we use the shifted version of inputs
    '''
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        bos, eos = tl.load(cu_trunclens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id + 1).to(tl.int32)
        T_S = eos - bos
        if n_id * CHUNK_SIZE >= T_S:
            return
        chunk_bos = tl.load(cu_chunklens + b_id).to(tl.int32)
    else:
        T_S = L_S
    # NOTE get the ptr
    rs_ptr = n_id * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    idx_mask = rs_ptr < T_S
    
    if USE_CUSEQLENS:
        p_rs = rel_states + (h_id*L_S + bos) + rs_ptr
        p_sl = state_lambda + (h_id*L_S + bos) + rs_ptr
        p_ga = tl.make_block_ptr(global_aux + (h_id*NT + chunk_bos+n_id-1)*NUM_STATES, (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
    else:
        p_rs = rel_states + (b_id*H + h_id)*L_S + rs_ptr
        p_sl = state_lambda + (b_id*H + h_id)*L_S + rs_ptr
        p_ga = tl.make_block_ptr(global_aux + ((b_id*H + h_id)*NT + n_id-1)*NUM_STATES, (NUM_STATES,), (1,), (0,), (NUM_STATES,), (0,))
    # NOTE load data
    b_aux = tl.arange(0, NUM_STATES)
    b_sl = tl.load(p_sl, mask=idx_mask, other=0)
    if n_id == 0:
        b_sl -= 1
        tl.debug_barrier()
        tl.store(p_sl, b_sl.to(p_rs.dtype.element_ty), mask=idx_mask)
    else:
        b_rs = tl.load(p_rs, mask=idx_mask, other=NUM_STATES)
        b_roh = (b_rs[:, None] == b_aux[None, :]).to(tl.int32)
        b_ga = tl.load(p_ga, boundary_check=(0,))[None, :]
        b_sl += tl.sum(b_roh * b_ga, axis=1) - 1
        tl.debug_barrier()
        tl.store(p_sl, b_sl.to(p_rs.dtype.element_ty), mask=idx_mask)

def flatten_cuseqlens(
    B: int,
    L: int,
    H: int,
    device: torch.device,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    if cu_seqlens is None:
        output = torch.zeros((1 + B * H), dtype=torch.int32, device=device)
        pack_cucp_kernel[(B, H)](
            output=output,
            B=B,
            H=H,
            L=L,
        )
    else:
        B = cu_seqlens.shape[0] - 1
        output = torch.zeros((1 + B * H), dtype=torch.int32, device=device)
        varlen_cucp_kernel[(B, H)](
            output=output,
            cu_seqlens=cu_seqlens,
            B=B,
            H=H,
            L=L,
        )
    return output

@triton.jit
def pack_cucp_kernel(
    output,
    B: tl.constexpr,
    H: tl.constexpr,
    L,
):
    b_id, h_id = tl.program_id(0), tl.program_id(1)
    bh_offset = b_id * H + h_id + 1
    tl.store(output + bh_offset, bh_offset * L)

@triton.jit
def varlen_cucp_kernel(
    output,
    cu_seqlens,
    B: tl.constexpr,
    H: tl.constexpr,
    L,
):
    b_id, h_id = tl.program_id(0), tl.program_id(1)
    eos = tl.load(cu_seqlens + b_id + 1).to(tl.int32)
    bh_offset = h_id * B + b_id + 1
    tl.store(output + bh_offset, h_id * L + eos)

@triton.jit
def indices_revise(
    indices,
    indices_revise,
    cu_trunclens,
    L_S,
    USE_CUSEQLENS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    b_id, h_id, n_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    if USE_CUSEQLENS:
        bos, eos = tl.load(cu_trunclens + b_id).to(tl.int32), tl.load(cu_trunclens + b_id + 1).to(tl.int32)
        T = eos - bos
        if n_id * CHUNK_SIZE >= T:
            return
        bias = (h_id * L_S) + bos
        p_idx = tl.make_block_ptr(indices + (h_id*L_S + bos), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_irv = tl.make_block_ptr(indices_revise + (h_id*L_S + bos), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
    else:
        T = L_S
        bias = (b_id*H + h_id)*L_S
        p_idx = tl.make_block_ptr(indices + ((b_id*H + h_id)*L_S), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))
        p_irv = tl.make_block_ptr(indices_revise + ((b_id*H + h_id)*L_S), (T,), (1,), (n_id*CHUNK_SIZE,), (CHUNK_SIZE,), (0,))        
    # NOTE load the data
    b_idx = tl.load(p_idx, boundary_check=(0,))
    b_idx -= bias
    tl.store(p_irv, b_idx.to(p_idx.dtype.element_ty), boundary_check=(0,))

from einops import einsum, rearrange, repeat

def boss_prepare_ref(
    top_states_q: torch.Tensor,
    top_states_k: torch.Tensor,
    shift_size: int,
    num_classes: int,
    cu_seqlens: Optional[torch.Tensor] = None,
) -> BossPrepareResult:
    '''
    return: indices_q, indices_k, state_lambda_q, state_lambda_k, hist_q, hist_k, cu_trunclens, cu_chunklens, True
    '''

    NUM_STATES = next_power_of_2(num_classes)
    B, L, H = top_states_q.shape
    CHUNK_SIZE = 128
    
    if cu_seqlens is not None:
        USE_CUSEQLENS = True
        assert cu_seqlens.dtype == torch.int32
        assert B == 1
        seqlens = torch.diff(cu_seqlens, dim=0)
        B = seqlens.shape[0]
        cu_trunclens = torch.zeros((B + 1,), dtype=torch.int32, device=top_states_q.device)
        cu_chunklens = torch.zeros((B + 1,), dtype=torch.int32, device=top_states_q.device)
        cu_trunc_chunk_kernel[(1,)](
            seqlens=seqlens,
            cu_trunclens=cu_trunclens,
            cu_chunklens=cu_chunklens,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
        )
        trunc_seqlens = torch.diff(cu_trunclens, dim=0)
        max_seqlen = trunc_seqlens.max().item()
        # NOTE if length is shorter than shift size, we don't have to compute boss part
        if max_seqlen == 0:
            return None, None, None, None, None, False
        L_S = cu_trunclens[-1].item()
        NT_L = cu_chunklens[-1].item()
        NT = triton.cdiv(max_seqlen, CHUNK_SIZE)
        # NOTE shift copy q, k
        top_states_q_shift = torch.zeros((1, H, L_S), dtype=torch.int32, device=top_states_q.device)
        top_states_k_shift = torch.zeros((1, H, L_S), dtype=torch.int32, device=top_states_q.device)
        shift_copy_q_kernel[(B, H, NT)](
            top_states=top_states_q,
            output=top_states_q_shift,
            cu_seqlens=cu_seqlens,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        shift_copy_k_kernel[(B, H, NT)](
            top_states=top_states_k,
            output=top_states_k_shift,
            cu_seqlens=cu_seqlens,
            cu_trunclens=cu_trunclens,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        # NOTE start to do block scan and histogram
        global_aux = torch.empty((1, H, NT_L, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_aux = torch.empty((1, H, NT_L, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_q = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        hist_k = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        state_lambda_q = torch.empty_like(top_states_q_shift)
        state_lambda_k = torch.empty_like(top_states_k_shift)
        state_lambda_k, hist_q = scan_hist_fused(
            top_states=top_states_k_shift,
            rel_states=top_states_k_shift,
            state_lambda=state_lambda_k,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_q,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT_L,
            CHUNK_SIZE=CHUNK_SIZE,
            cu_trunclens=cu_trunclens,
            cu_chunklens=cu_chunklens,
        )
        state_lambda_q, hist_k = scan_hist_fused(
            top_states=top_states_k_shift,
            rel_states=top_states_q_shift,
            state_lambda=state_lambda_q,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_k,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT_L,
            CHUNK_SIZE=CHUNK_SIZE,
            cu_trunclens=cu_trunclens,
            cu_chunklens=cu_chunklens,
        )

    else:
        USE_CUSEQLENS = False
        # NOTE if length is shorter than shift size, we don't have to compute boss part
        if L <= shift_size:
            return None, None, None, None, None, False
        max_seqlen=None
        L_S = L - shift_size
        NT = triton.cdiv(L_S, CHUNK_SIZE)

        # NOTE shift copy q, k
        top_states_q_shift = torch.zeros((B, H, L_S), dtype=torch.int32, device=top_states_q.device)
        top_states_k_shift = torch.zeros((B, H, L_S), dtype=torch.int32, device=top_states_q.device)
        shift_copy_q_kernel[(B, H, NT)](
            top_states=top_states_q,
            output=top_states_q_shift,
            cu_seqlens=None,
            cu_trunclens=None,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        shift_copy_k_kernel[(B, H, NT)](
            top_states=top_states_k,
            output=top_states_k_shift,
            cu_seqlens=None,
            cu_trunclens=None,
            L_S=L_S,
            USE_CUSEQLENS=USE_CUSEQLENS,
            SHIFT_SIZE=shift_size,
            CHUNK_SIZE=CHUNK_SIZE,
            B=B,
            L=L,
            H=H,
        )
        # NOTE start to do block scan and histogram
        global_aux = torch.empty((B, H, NT, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_aux = torch.empty((B, H, NT, NUM_STATES), dtype=torch.int32, device=top_states_q.device)
        hist_q = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        hist_k = torch.zeros((B, H, NUM_STATES + 1), dtype=torch.int32, device=top_states_q.device)
        state_lambda_q = torch.empty_like(top_states_q_shift)
        state_lambda_k = torch.empty_like(top_states_k_shift)
        state_lambda_k, hist_q = scan_hist_fused(
            top_states=top_states_k_shift,
            rel_states=top_states_k_shift,
            state_lambda=state_lambda_k,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_q,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        state_lambda_q, hist_k = scan_hist_fused(
            top_states=top_states_k_shift,
            rel_states=top_states_q_shift,
            state_lambda=state_lambda_q,
            global_aux=global_aux,
            hist_aux=hist_aux,
            hist=hist_k,
            NUM_STATES=NUM_STATES,
            USE_CUSEQLENS=USE_CUSEQLENS,
            B=B,
            L_S=L_S,
            H=H,
            NT=NT,
            CHUNK_SIZE=CHUNK_SIZE,
        )
        cu_trunclens = None
        
    
    return state_lambda_q, state_lambda_k, top_states_q_shift, top_states_k_shift, cu_trunclens, True

def scatter_boss_attention_torch_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    top_states_q: torch.Tensor, # [B L H]
    top_states_k: torch.Tensor, # [B L H]
    sg_indices_q: torch.Tensor, # [B L H]
    sg_indices_k: torch.Tensor, # [B L H]
    local_size: int,
    global_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    B, L, H, D = q.shape
    state_lambda_q, state_lambda_k, top_states_q_shift, top_states_k_shift, cu_trunclens, valid = boss_prepare_ref(
        top_states_q=top_states_q,
        top_states_k=top_states_k,
        shift_size=local_size,
        num_classes=num_states,
        cu_seqlens=cu_seqlens,
    )
    assert valid, "it should be True to make boss part compute"
    o = torch.zeros_like(q)
    o_b = torch.zeros_like(q)
    q = q * sg_indices_q[..., None]
    k = k * sg_indices_k[..., None]
    def head_attention(
        q: torch.Tensor, # [L, D]
        k: torch.Tensor, # [L, D]
        v: torch.Tensor, # [L, D]
        top_states_q_shift: torch.Tensor, # [L_S]
        top_states_k_shift: torch.Tensor, # [L_S]
        sg_indices_q: torch.Tensor, # [L_S]
        sg_indices_k: torch.Tensor, # [L_S]
        state_lambda_q: torch.Tensor, # [L_S]
        state_lambda_k: torch.Tensor, # [L_S]
        local_size: int,
        global_size: int,
        cu_trunclens: Optional[torch.Tensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
    ):
        L, D = q.shape
        softmax_scale = 1.0 / math.sqrt(D)
        
        q = q * softmax_scale
        qk = (q @ k.T)
        if cu_seqlens is not None: 
            M_swa = torch.zeros(qk.shape, dtype=torch.bool, device=q.device)
            M_boss = torch.zeros(qk.shape, dtype=torch.bool, device=q.device)

            num_samples = cu_seqlens.shape[0] - 1
            for i in range(num_samples):
                start = cu_seqlens[i]
                end = cu_seqlens[i + 1]
                l = end - start
                assert l > 0, "cu_seqlens plz larger than zero"
                if l > local_size:
                    
                    index = torch.arange(0, l).to(q.device)
                    m_diff = index[:, None] - index[None, :]
                    M_swa[start:end, start:end] = (m_diff >= 0) & (m_diff < local_size)

                    trunc_bos = cu_trunclens[i]
                    trunc_eos = cu_trunclens[i + 1]
                    slq = state_lambda_q[trunc_bos:trunc_eos]
                    slk = state_lambda_k[trunc_bos:trunc_eos]
                    tsq = top_states_q_shift[trunc_bos:trunc_eos]
                    tsk = top_states_k_shift[trunc_bos:trunc_eos]
                    ts_eq = tsq[:, None] == tsk[None, :]
                    sl_diff = slq[:, None] - slk[None, :]
                    M_boss[start+local_size:end, start:end-local_size] = (ts_eq & (sl_diff >= 0) & (sl_diff < global_size))
                    
                else: # l < local_size
                    index = torch.arange(0, l)
                    m_diff = index[:, None] - index[None, :]
                    M_swa[start:end, start:end] = (m_diff >= 0) & (m_diff < local_size)
                    # NOTE no boss mask if l < local_size
            M_total = M_swa | M_boss
            
            
            qk = torch.where(M_total, qk, float('-inf'))
            score = qk.softmax(dim=-1)
            o_swa = torch.where(M_swa, score, 0).to(q.dtype) @ v
            o_boss = (torch.where(M_boss, score, 0).to(q.dtype) @ v)
            lse = torch.logsumexp(qk, dim=-1)
            o = o_swa
        else:
            qk = (q @ k.T)

            index = torch.arange(0, L).to(q.device)
            diff = index[:, None] - index[None, :]
            M_swa = (diff >= 0) & (diff < local_size)

            L_S= L = local_size
            lambda_diff = state_lambda_q[:, None] - state_lambda_k[None, :]
            state_eq = top_states_q_shift[:, None] == top_states_k_shift[None, :]
            M_boss = torch.zeros(M_swa.shape, dtype=torch.bool, device=q.device)
            M_boss[local_size:, :-local_size] = (lambda_diff >= 0) & (lambda_diff < global_size) & state_eq
            M_total = M_swa | M_boss
            qk = torch.where(M_total, qk, float('-inf'))
            lse = torch.logsumexp(qk, dim=-1)
            score = qk.softmax(dim=-1)
            o_swa = torch.where(M_swa, score, 0).to(q.dtype) @ v
            o_boss = (torch.where(M_boss, score, 0).to(q.dtype) @ v)
            o = o_swa
        return o, o_boss, lse

    o1_list = []
    o2_list = []
    lse_list = []
    for b in range(B):
        o1_list_b = []
        o2_list_b = []
        lse_list_b = []
        for h in range(H):
            o1, o2, lse = head_attention(
                q=q[b, :, h, :],
                k=k[b, :, h, :],
                v=v[b, :, h, :],
                top_states_q_shift=top_states_q_shift[b, h, :],
                top_states_k_shift=top_states_k_shift[b, h, :],
                sg_indices_q=sg_indices_q[b, :, h],
                sg_indices_k=sg_indices_k[b, :, h],
                state_lambda_q=state_lambda_q[b, h, :],
                state_lambda_k=state_lambda_k[b, h, :],
                local_size=local_size,
                global_size=global_size,
                cu_trunclens=cu_trunclens,
                cu_seqlens=cu_seqlens,
            )
            o1_list_b.append(o1)
            o2_list_b.append(o2)
            lse_list_b.append(lse)
        # 这一层拼成 [L, H, D]
        o1_list.append(torch.stack(o1_list_b, dim=1))
        o2_list.append(torch.stack(o2_list_b, dim=1))
        lse_list.append(torch.stack(lse_list_b, dim=0))
    # 最后拼成 [B, H, L, D]
    o  = torch.stack(o1_list, dim=0)
    o_b = torch.stack(o2_list, dim=0)
    lse = torch.stack(lse_list, dim=0)
    o = o + o_b
    return o, o_b, lse

def scatter_verify(
    state_lambda,
    indices,
    hist,
    b_id,
    h_id,
    e_id,
):
    bos = hist[b_id, h_id, e_id]
    eos = hist[b_id, h_id, e_id + 1]
    idx = indices[b_id, h_id, bos:eos]
    return state_lambda[b_id, h_id, idx]

def do_state_lambda(
    top_states_q: torch.Tensor, # [B L H]
    top_states_k: torch.Tensor, # [B L H]
    num_classes: int,
    local_size: int,
    cu_seqlens: Optional[torch.Tensor] = None,
):
    # NOTE boss computation
    boss_info = boss_prepare(
        top_states_q=top_states_q,
        top_states_k=top_states_k,
        shift_size=local_size,
        num_classes=num_classes,
        cu_seqlens=cu_seqlens,
    )
    state_lambda_q = boss_info.state_lambda_q
    state_lambda_k = boss_info.state_lambda_k
    indices_q = boss_info.indices_q
    indices_k = boss_info.indices_k
    hist_q = boss_info.hist_q
    hist_k = boss_info.hist_k

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
def save_bool_heatmap(mat, filename='bool_matrix_heatmap.png', show_plot=True):
    """
    可视化并保存一个bool型或0/1型矩阵的heatmap。
    参数:
        mat: torch.Tensor 或 np.ndarray，形状为 (L, L)，类型为bool或0/1
        filename: 保存图片的文件名
        show_plot: 是否显示图片窗口
    """
    # 支持 torch.Tensor 和 np.ndarray
    if isinstance(mat, torch.Tensor):
        mat_np = mat.cpu().numpy()
    elif isinstance(mat, np.ndarray):
        mat_np = mat
    else:
        raise ValueError("mat 必须是 torch.Tensor 或 np.ndarray")
    # 如果是bool类型，转为int
    if mat_np.dtype == np.bool_:
        mat_np = mat_np.astype(int)
    plt.figure(figsize=(6, 6), dpi=300)
    # 使用灰度色阶，0为白，1为黑
    plt.imshow(mat_np, cmap='gray_r', interpolation='nearest', origin='upper')
    plt.title('Bool Matrix Heatmap')
    plt.xlabel('Column')
    plt.ylabel('Row')
    plt.colorbar(label='Value')
    plt.savefig(filename, bbox_inches='tight')
    if show_plot:
        plt.show()
    plt.close()
    print(f"Heatmap 已保存为 {filename}")

def get_err_ratio(x, y):
    err = (x - y).flatten().square().mean().sqrt().item()
    base = x.flatten().square().mean().sqrt().item()
# get_err_ratio(o_ref, o)
    return err / base
import random
def set_seed(seed=42):
    """
    固定所有常用随机性的种子，保证结果可复现。
    """
    random.seed(seed)                          # Python 内置随机库
    np.random.seed(seed)                       # numpy 随机库
    torch.manual_seed(seed)                    # torch CPU
    torch.cuda.manual_seed(seed)               # torch 单个GPU
    torch.cuda.manual_seed_all(seed)           # torch 多个GPU
    # 保证 cudnn 的确定性
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
if __name__ == "__main__":
    
    set_seed(42)
    num_states = 8
    local_size = 128
    global_size = 2048
    B, L, H, D = 1, 8192, 16, 128
    top_states_q = torch.randint(0, num_states, (B, L, H), dtype=torch.int32, device="cuda")
    top_states_k = torch.randint(0, num_states, (B, L, H), dtype=torch.int32, device="cuda")

    sg_indices_q =  torch.full((B, L, H), 1., dtype=torch.float32, device="cuda", requires_grad=True)
    sg_indices_k =  torch.full((B, L, H), 1., dtype=torch.float32, device="cuda", requires_grad=True)
    cu_seqlens = torch.tensor([0, 1024, 1256, 2048, 8192], dtype=torch.int32, device="cuda")
    # cu_seqlens = None
    q = torch.randn((B, L, H, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    k = torch.randn((B, L, H, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    v = torch.randn((B, L, H, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    g = -torch.rand((B, L, H, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    do = torch.randn((B, L, H, D), dtype=torch.bfloat16, device="cuda", requires_grad=True)
    router_weight = torch.full((B, L, H), 1., dtype=torch.float32, device="cuda", requires_grad=True)

    q_ref = q.clone().detach().requires_grad_()
    k_ref = k.clone().detach().requires_grad_()
    v_ref = v.clone().detach().requires_grad_()
    do_ref = do.clone().detach().requires_grad_()
    sg_indices_q_ref = sg_indices_q.clone().detach().requires_grad_()
    sg_indices_k_ref = sg_indices_k.clone().detach().requires_grad_()
    router_weight_ref = router_weight.clone().detach().requires_grad_()
    
    o_ref, o_boss_ref, lse_ref = scatter_boss_attention_torch_ref(
        q=q_ref.to(torch.float32),
        k=k_ref.to(torch.float32),
        v=v_ref.to(torch.float32),  
        top_states_q=top_states_q,
        top_states_k=top_states_k,
        sg_indices_q=sg_indices_q_ref,
        sg_indices_k=sg_indices_k_ref,
        local_size=local_size,
        global_size=global_size,
        cu_seqlens=cu_seqlens,
    )
    o_ref = o_ref.to(torch.bfloat16)
    o_boss_ref = o_boss_ref.to(torch.bfloat16)
    (o_ref * do_ref).sum().backward()
    
    o = scatter_boss_function(
        q=q,
        k=k,
        v=v,
        top_states_q=top_states_q,
        top_states_k=top_states_k,
        sg_indices_q=sg_indices_q,
        sg_indices_k=sg_indices_k,
        num_classes=num_states,
        local_size=local_size,
        global_size=global_size,
        cu_seqlens=cu_seqlens,
    )
    (o * do).sum().backward()
    

    print("=====the output error=====")
    print("relative error \n", get_err_ratio(o, o_ref))
    print("absolute error \n", (o - o_ref).abs().max().item())

    # print("=====the lse error=====")
    # print("relative error \n", get_err_ratio(lse, lse_ref))
    # print("absolute error \n", (lse - lse_ref).abs().max().item())

    # print("=====the output boss error=====")
    # print("relative error \n", get_err_ratio(o_boss, o_boss_ref))
    # print("absolute error \n", (o_boss - o_boss_ref).abs().max().item())
    
    print("=====the q grad error=====")
    print("relative error \n", get_err_ratio(q_ref.grad, q.grad))
    print("absolute error \n", (q.grad - q_ref.grad).abs().max().item())
    
    print("=====the k grad error=====")
    print("relative error \n", get_err_ratio(k_ref.grad, k.grad))
    print("absolute error \n", (k.grad - k_ref.grad).abs().max().item())

    print("=====the v grad error=====")
    print("relative error \n", get_err_ratio(v_ref.grad, v.grad))
    print("absolute error \n", (v.grad - v_ref.grad).abs().max().item())

    print("=====the sgq grad error=====")
    print("relative error \n", get_err_ratio(sg_indices_q_ref.grad, sg_indices_q.grad))
    print("absolute error \n", (sg_indices_q.grad - sg_indices_q_ref.grad).abs().max().item())

    print("=====the sgk grad error=====")
    print("relative error \n", get_err_ratio(sg_indices_k_ref.grad, sg_indices_k.grad))
    print("absolute error \n", (sg_indices_k.grad - sg_indices_k_ref.grad).abs().max().item())

    # d_sgk = torch.einsum("bnhd,bnhd->bnh", q_ref.float(), q_ref.grad.float()) get_err_ratio(x, sg_indices_q.grad)

    # print("=====the w grad error=====")
    # print("relative error \n", get_err_ratio(router_weight_ref.grad, router_weight.grad))
    # print("absolute error \n", (router_weight.grad - router_weight_ref.grad).abs().max().item())

    pdb.set_trace()