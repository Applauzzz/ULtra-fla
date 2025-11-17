
from typing import Any, Callable, Optional, Union

import torch

import triton
from triton import language as tl

# NOTE here we give the referecne code
def _insert_single_sequence(sequence: torch.Tensor, chunk_size: int, insert_id: int) -> torch.Tensor:
    """[内部辅助函数] 对单个 1D 序列执行插入操作。"""
    # 如果长度不足以插入，直接返回
    if sequence.numel() <= chunk_size:
        return sequence.clone()
    chunks = list(torch.split(sequence, chunk_size))
    result = []
    for i, chunk in enumerate(chunks):
        result.append(chunk)
        # 如果不是最后一个 chunk，就在后面加上分隔符
        if i < len(chunks) - 1:
            result.append(torch.tensor([insert_id], dtype=sequence.dtype, device=sequence.device))
    return torch.cat(result)
def ground_truth_insert(
    input_ids: torch.Tensor,
    chunk_size: int,
    insert_id: int,
    cu_seqlens: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    [参考实现] 使用纯 PyTorch/Python 插入特殊 token，支持单序列和批处理 (cu_seqlens)。
    Args:
        input_ids (torch.Tensor): 输入的 token ID。
            - 如果 cu_seqlens is None, 假定为 (1, N) 或 (N,)。
            - 如果 cu_seqlens is not None, 假定为 (1, N) 或 (N,) 的 packed 张量。
        chunk_size (int): 插入间隔。
        insert_id (int): 要插入的 token ID。
        cu_seqlens (Optional[torch.Tensor]): 累积序列长度，用于批处理。
    Returns:
        torch.Tensor: 插入 token 后的 1D 张量。对于批处理，返回一个 packed 的 1D 张量。
    """
    # 确保处理的是 1D 张量
    if input_ids.dim() > 1:
        input_ids = input_ids.squeeze(0)
    # 情况一：单序列处理
    if cu_seqlens is None:
        return _insert_single_sequence(input_ids, chunk_size, insert_id)
    # 情况二：批处理 (cu_seqlens)
    else:
        processed_sequences = []
        num_sequences = len(cu_seqlens) - 1
        for i in range(num_sequences):
            start_idx = cu_seqlens[i]
            end_idx = cu_seqlens[i+1]
            current_seq = input_ids[start_idx:end_idx]
            # 对每个子序列调用单序列处理函数
            processed_seq = _insert_single_sequence(current_seq, chunk_size, insert_id)
            processed_sequences.append(processed_seq)
        # 将所有处理完的序列拼接成一个大的 packed 张量
        return torch.cat(processed_sequences)

# NOTE here we propose a function to insert special tokens
def insert_special_tokens(
    input_ids: torch.Tensor,
    cu_seqlens: torch.Tensor = None,
    chunk_size: int = 4, # we want 4, 16, 32 ...
    insert_id: int = -1,
) -> torch.Tensor:
    # BLOCK_SIZE = 64
    assert insert_id != -1, "insert_id should be different from -1, which means you must specify the insert_id"
    # assert BLOCK_SIZE % chunk_size == 0, "128 should be divisible by chunk_size"

    batch_size, total_length = input_ids.shape
    
    if cu_seqlens is None:
        assert batch_size == 1, "cu_seqlens is None, but batch_size is not 1"
        USE_CU_SEQLENS =  False
        num_inserts_total = (total_length - 1) // chunk_size
        max_seqlen = total_length
        output_ids = torch.empty(size=(batch_size, max_seqlen + num_inserts_total), dtype=input_ids.dtype, device=input_ids.device)
        # prepare the scatter indices
        scatter_sep = torch.empty(size=(batch_size, num_inserts_total), dtype=input_ids.dtype, device=input_ids.device)
        scatter_inp = torch.empty_like(input_ids)
        src_sep = torch.full_like(scatter_sep, insert_id)
        
        cu_chunklens = None
        cu_seqlens_q = None
    
    else:
        USE_CU_SEQLENS = True
        assert batch_size == 1, "cu_seqlens is not None, but batch_size is not 1"
        seqlens = cu_seqlens[1:] - cu_seqlens[:-1]
        max_seqlen = seqlens.max().item()
        num_inserts = ((seqlens - 1) // chunk_size)
        num_inserts = torch.clamp(num_inserts, max=0)
        cu_chunklens = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype, device=cu_seqlens.device), num_inserts.cumsum(0)], dim=0)
        cu_seqlens_q = torch.cat([torch.tensor([0], dtype=cu_seqlens.dtype, device=cu_seqlens.device), (seqlens + num_inserts).cumsum(0)], dim=0)
        num_inserts_total = cu_chunklens[-1].item()
        output_ids = torch.empty(size=(1, total_length + num_inserts_total), dtype=input_ids.dtype, device=input_ids.device)
        # prepare the scatter indices
        scatter_sep = torch.empty(size=(batch_size, num_inserts_total), dtype=input_ids.dtype, device=input_ids.device)
        scatter_inp = torch.empty_like(input_ids)
        src_sep = torch.full_like(scatter_sep, insert_id)
        
        batch_size = seqlens.shape[0]

    # 16 is the minimum requirement for loading
    BLOCK_SIZE = max(16, chunk_size)
    NT = triton.cdiv(max_seqlen, chunk_size)
    grid = (batch_size, NT)
    scatter_index_prepare[grid](
        scatter_inp,
        scatter_sep,
        cu_seqlens,
        cu_chunklens,
        max_seqlen,
        USE_CU_SEQLENS,
        chunk_size,
        BLOCK_SIZE,
    )
    # scatter the input_ids and sep_ids
    output_ids = torch.scatter(output_ids, 1, scatter_inp, input_ids)
    output_ids = torch.scatter(output_ids, 1, scatter_sep, src_sep)

    return output_ids, scatter_inp, scatter_sep, cu_seqlens_q, cu_chunklens

@triton.jit
def scatter_index_prepare(
    scatter_inp,
    scatter_sep,
    cu_seqlens,
    cu_chunklens,
    L,
    USE_CU_SEQLENS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr, # real logic loading size for the sequnece
    BLOCK_SIZE: tl.constexpr, # load size, will padding if chun_size smaller than 16
):    
    b_id, t_id = tl.program_id(0), tl.program_id(1)
    
    if USE_CU_SEQLENS:
        start, end = tl.load(cu_seqlens + b_id), tl.load(cu_seqlens + b_id + 1)
        inserted_num = tl.load(cu_chunklens + b_id)

        sample_length = end - start
        if t_id * CHUNK_SIZE >= sample_length:
            return
        # here is the pointer for scatter_inp, while we save the scatter indices back
        p_inp_ids = scatter_inp + start + t_id * CHUNK_SIZE + tl.arange(0, BLOCK_SIZE)
        # we create mask for the store operation
        logic_mask = tl.arange(0, BLOCK_SIZE) < CHUNK_SIZE # logic loading mask
        length_mask = t_id * CHUNK_SIZE + tl.arange(0, BLOCK_SIZE) < sample_length # length mask
        mask = logic_mask & length_mask # the mask should both satisfy length and logic constraint

        b_insert_ids = (start + inserted_num) + t_id * (CHUNK_SIZE + 1) - 1 # the position of insert id
        b_inp_ids = b_insert_ids + 1 + tl.arange(0, BLOCK_SIZE) # the chunk should be placed right after the insert id
        # For the start position we do not insert at begin
        if t_id == 0:
            tl.store(p_inp_ids, b_inp_ids, mask=mask)
        # For the other position we do insert at begin
        else:
            p_sep = scatter_sep + inserted_num + t_id - 1
            tl.store(p_inp_ids, b_inp_ids, mask=mask)
            tl.store(p_sep, b_insert_ids)
    
    else:
        # here is the pointer for scatter_inp, while we save the scatter indices back
        sample_length = L
        p_inp_ids = scatter_inp + (b_id * L) + t_id * CHUNK_SIZE + tl.arange(0, BLOCK_SIZE)
        # we create mask for the store operation
        logic_mask = tl.arange(0, BLOCK_SIZE) < CHUNK_SIZE # logic loading mask
        length_mask = t_id * CHUNK_SIZE + tl.arange(0, BLOCK_SIZE) < sample_length # length mask
        mask = logic_mask & length_mask # the mask should both satisfy length and logic constraint

        b_insert_ids = t_id * (CHUNK_SIZE + 1) - 1 # the position of insert id
        b_inp_ids = b_insert_ids + 1 + tl.arange(0, BLOCK_SIZE) # the chunk should be placed right after the insert id
        # For the start position we do not insert at begin
        if t_id == 0:
            tl.store(p_inp_ids, b_inp_ids, mask=mask)
        # For the other position we do insert at begin
        else:
            p_sep = scatter_sep + t_id - 1
            tl.store(p_inp_ids, b_inp_ids, mask=mask)
            tl.store(p_sep, b_insert_ids)

if __name__ == "__main__":
    CHUNK_SIZE = 4
    INSERT_ID = -100
    # --- 测试案例 1, 2, 3 保持不变 ---
    print("-" * 60)
    print(f"✅ 测试案例 1: 单序列, 长度 11, chunk_size {CHUNK_SIZE}")
    input_ids_1 = torch.arange(11, device='cuda', dtype=torch.long).unsqueeze(0)
    output_ids_1, _, _ = insert_special_tokens(input_ids_1, chunk_size=CHUNK_SIZE, insert_id=INSERT_ID)
    expected_output_1 = ground_truth_insert(input_ids_1, CHUNK_SIZE, INSERT_ID)
    assert (output_ids_1 - expected_output_1).abs().sum().item() == 0, "测试案例 1 失败!"
    print("🎉 测试案例 1 通过!\n")
    # ... (案例2和3的测试代码也类似，此处省略以保持简洁) ...
    # ==============================================================================
    # 测试案例 4: cu_seqlens is not None, 使用了新的 ground_truth 函数
    # ==============================================================================
    print("-" * 60)
    print(f"✅ 测试案例 4: 批处理 (cu_seqlens), 长度 [11, 8, 3]")
    print("-" * 60)
    seq1 = torch.arange(11, device='cuda', dtype=torch.long)
    seq2 = torch.arange(100, 108, device='cuda', dtype=torch.long)
    seq3 = torch.arange(200, 203, device='cuda', dtype=torch.long)
    packed_input_ids = torch.cat([seq1, seq2, seq3]).unsqueeze(0)
    cu_seqlens = torch.tensor([0, 11, 19, 22], device='cuda', dtype=torch.int32)
    print("打包的原始输入:", packed_input_ids)
    print("cu_seqlens:", cu_seqlens)
    # 计算期望的、紧凑的输出
    expected_packed_output = ground_truth_insert(packed_input_ids, CHUNK_SIZE, INSERT_ID, cu_seqlens)
    print("\n期望的正确输出 (packed):\n", expected_packed_output)
    output_ids_1, _, _ = insert_special_tokens(packed_input_ids, cu_seqlens, CHUNK_SIZE, INSERT_ID)
    assert (output_ids_1 - expected_packed_output).abs().sum().item() == 0, "测试案例 2 失败!"
    print("🎉 测试案例 2 通过!\n")