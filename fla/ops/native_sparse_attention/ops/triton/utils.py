# Copyright 2025 Xunhao Lai & Jianqiao Lu.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

import triton
from triton import language as tl


def is_hopper_gpu():
    if torch.cuda.is_available():
        device_capability = torch.cuda.get_device_capability()
        major, minor = device_capability
        return major == 9
    return False


def get_compressed_seqlens(
    cu_seqlens: torch.Tensor, kernel_size: int, kernel_stride: int
):
    # compute seqlens after compression
    seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
    y_seqlens = torch.floor((seqlens - kernel_size) / kernel_stride).to(torch.int32) + 1
    # corner case, if sequence_length < kernel_size, no compression for this sequence
    y_seqlens[seqlens < kernel_size] = 0 # make edge cases to zero
    y_cu_seqlens = torch.zeros(
        y_seqlens.shape[0] + 1, dtype=torch.int32, device=cu_seqlens.device
    )
    y_cu_seqlens[1:] = torch.cumsum(y_seqlens, dim=0)
    return y_seqlens, y_cu_seqlens # get compressed seqlens and lens


def get_num_warps_stages(head_dim, block_size, is_hopper_gpu):
    """
    Returns recommended num_warps and num_stages for a Sparse Attention kernel in Triton.

    Args:
        head_dim (int): Size of the head dimension.
        block_size (int): Size of the block in the attention matrix.
        is_hopper_gpu (bool): True if Hopper GPU, False if Ampere GPU.

    Returns:
        tuple: (num_warps, num_stages) recommended values.
    """
    # Determine if head_dim and block_size exceed 64
    head_large = head_dim > 64
    block_large = block_size > 64

    if is_hopper_gpu:
        # Hopper GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 4
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    else:
        # Ampere GPU recommendations
        if head_large and block_large:
            num_warps = 8
            num_stages = 3
        elif head_large or block_large:
            num_warps = 8
            num_stages = 3
        else:
            num_warps = 2
            num_stages = 2
    if head_dim > 128:
        num_stages = 2
    return num_warps, num_stages

@triton.jit
def varlen_to_batch(
    x, # total_len, num_heads, head_dim
    o, # batch_size, max_seqlen, num_heads, head_dim
    cu_seqlens, # batch_size + 1
    L, # max_seqlen
    H: tl.constexpr,
    D: tl.constexpr,
    BLOCK_T: tl.constexpr,
):
    # grid: B, NT, H
    b_id, t_id, h_id = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    
    # get cu_seqlens
    start = tl.load(cu_seqlens + b_id)
    end = tl.load(cu_seqlens + b_id + 1)
    seqlen = end - start
    
    # seqlens offsets
    seq_start = t_id * BLOCK_T
    if seq_start >= seqlen:
        return

    idx = seq_start + tl.arange(0, BLOCK_T)
    mask = (idx < seqlen)[:, None]
    p_x = x + ((start + idx) * (H*D))[:, None] + (h_id*D) + tl.arange(0, D)[None, :]
    p_o = o + (((b_id*L) + idx) * (H*D))[:, None] + (h_id*D) + tl.arange(0, D)[None, :]

    b_x = tl.load(p_x, mask=mask)
    b_o = tl.load(p_o, mask=mask)

def v2b(
    x,
    cu_seqlens,
    seqlens,
    max_seqlen,
):
    L = max_seqlen
    B = seqlens.shape[0]
    _, H, D = x.shape
    o = torch.zeros((B, L, H, D), device=x.device, dtype=x.dtype)

    BLOCK_T = 128
    NT = triton.cdiv(L, BLOCK_T)

    grid = (B, NT, H)
    varlen_to_batch[grid](
        x,  
        o, 
        cu_seqlens, 
        L, 
        H, 
        D, 
        BLOCK_T
    )
    return o
