import torch
import pdb

if __name__ == "__main__":
    # --- 1. 准备输入数据 ---
    # 定义维度
    batch_size = 2
    original_seq_length = 14  # L_orig，一个不是4的倍数的长度
    hidden_dim = 8
    sampling_rate = 4
    # a. 先创建原始张量
    original_tensor = torch.arange(original_seq_length).float().reshape(1, original_seq_length, 1).repeat(batch_size, 1, hidden_dim)
    # b. 进行下采样，得到我们的输入
    sampled_tensor = original_tensor[:, ::sampling_rate, :]
    downsampled_seq_length = sampled_tensor.shape[1] # L_down
    print("--- 准备阶段 ---")
    print(f"原始序列长度 L_orig: {original_seq_length}")
    print(f"下采样后的张量形状: {sampled_tensor.shape}")
    print(f"下采样后张量第一个样本的内容 (L维度索引): \n{sampled_tensor[0, :, 0]}\n")
    pdb.set_trace()
    # --- 2. 执行精确上采样操作 ---
    # a. 计算每个元素需要重复的次数
    #    对于前 L_down - 1 个元素，每个重复 sampling_rate 次
    repeats = torch.full((downsampled_seq_length,), sampling_rate, dtype=torch.long)
    # b. 计算最后一个元素需要重复的次数
    last_element_repeats = original_seq_length - (downsampled_seq_length - 1) * sampling_rate
    repeats[-1] = last_element_repeats
    print(f"--- 上采样计算 ---")
    print(f"每个元素需要重复的次数: {repeats}")
    print(f"计算出的最后一个元素重复次数: {last_element_repeats}\n")
    pdb.set_trace()
    # c. 使用 repeat_interleave 执行上采样
    #    dim=1 表示我们希望在 L 维度上进行重复操作
    upsampled_tensor = torch.repeat_interleave(sampled_tensor, repeats, dim=1)
    # --- 3. 验证结果 ---
    print("--- 验证结果 ---")
    print(f"精确上采样后的张量形状: {upsampled_tensor.shape}")
    print(f"上采样后张量第一个样本的内容 (L维度索引): \n{upsampled_tensor[0, :, 0]}\n")
    # 验证长度是否与原始长度一致
    assert upsampled_tensor.shape[1] == original_seq_length
    print(f"验证成功！上采样后的长度 ({upsampled_tensor.shape[1]}) 与原始长度 ({original_seq_length}) 一致。")
    pdb.set_trace()