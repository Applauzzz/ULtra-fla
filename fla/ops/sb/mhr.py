import torch
from torch import nn
import triton
from triton import language as tl
import pdb

class GradTop1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, score, indices):
        ctx.save_for_backward(indices)
        output = torch.ones(indices.shape, dtype=score.dtype, device=score.device)
        ctx.score_shape = score.shape
        return output
    @staticmethod
    def backward(ctx, grad_output):
        indices, = ctx.saved_tensors
        grad_score = torch.zeros(ctx.score_shape, dtype=grad_output.dtype, device=grad_output.device)
        index = indices.unsqueeze(-1)
        src = grad_output.unsqueeze(-1)
        grad_score.scatter_add_(-1, index, src)
        return grad_score, None

grad_top1 = GradTop1.apply

class MultiHeadRouter(nn.Module):
    def __init__(self, 
                num_heads, 
                num_states, 
                head_dim=128):
        super().__init__()
        self.num_heads = num_heads
        self.num_states = num_states
        self.head_dim = head_dim
        self.weight = nn.Parameter(torch.randn(num_heads, num_states, head_dim))
        self.bias = nn.Parameter(torch.randn(num_heads, num_states))

    def forward(self, x):
        B, L, H, _ = x.shape
        T = B * L
        dtype = x.dtype
        
        output = torch.einsum('bnhd, hsd -> bnhs', x.float(), self.weight.float()) + self.bias.float()
        score = output.softmax(dim=-1) # [B L H NS]
        indices = score.argmax(dim=-1).to(torch.int32)
        sg_indices = grad_top1(score, indices)

        bin_indices = bincount(indices, self.num_states)
        mean_indices = bin_indices.float() / T # [H, NS]

        balance_loss = self.num_states * (mean_indices * score).mean((0, 1)).sum()


        return sg_indices.to(dtype), indices, balance_loss.to(dtype)
    
    def extra_repr(self) -> str:
        return f"num_heads={self.num_heads}, num_states={self.num_states}, head_dim={self.head_dim}"


def bincount(
    indices: torch.Tensor,
    NUM_STATES: int,
):
    CHUNK_SIZE = 128
    endbit = NUM_STATES.bit_length()
    B, L, H = indices.shape
    T = B * L

    if indices.dtype != torch.int32:
        indices = indices.to(torch.int32)

    NT = triton.cdiv(T, CHUNK_SIZE)
    hist_aux = torch.empty((NT, H, 2**endbit), dtype=indices.dtype, device=indices.device)
    grid = (H, NT)
    bincount_kernel[grid](
        indices, 
        hist_aux, 
        T,
        H, 
        NS=NUM_STATES,
        NUM_BINS=2**endbit,
        CHUNK_SIZE=CHUNK_SIZE,
    )
    bin_indices = hist_aux[:, :, :NUM_STATES].sum(dim=0)
    # return bin_indices [H, NUM_STATES]
    return bin_indices

@triton.jit
def bincount_kernel(
    indices,
    hist_aux,
    T,
    H: tl.constexpr,
    NS: tl.constexpr,
    NUM_BINS: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
):
    h_id, n_id = tl.program_id(0), tl.program_id(1)
    t_ptr = n_id * CHUNK_SIZE + tl.arange(0, CHUNK_SIZE)
    mask = t_ptr < T
    indices_ptr = indices + (t_ptr * H) + h_id
    aux_ptr = hist_aux + n_id * (H * NUM_BINS) + h_id * NUM_BINS + tl.arange(0, NUM_BINS)
    aux_mask = (tl.arange(0, NUM_BINS) < NS)

    b_indices = tl.load(indices_ptr, mask=mask)
    b_hist = tl.histogram(b_indices, NUM_BINS, mask=mask)
    tl.store(aux_ptr, b_hist, mask=aux_mask)

if __name__ == "__main__":
    # 定义测试维度
    B, L, H, D, NS = 2, 1024, 32, 128, 6
    print("--- 模型和输入参数 ---")
    print(f"Batch Size (B): {B}")
    print(f"Sequence Length (L): {L}")
    print(f"Number of Heads (H): {H}")
    print(f"Head Dimension (D): {D}")
    print(f"Number of States/Experts (NS): {NS}\n")
    # 实例化模型并移动到 GPU
    device = 'cuda'
    router = MultiHeadRouter(num_heads=H, num_states=NS, head_dim=D).to(device)
    # 创建随机输入张量并移动到 GPU
    # 输入形状应为 (B, L, H, D)
    input_tensor = torch.randn(B, L, H, D, device=device, requires_grad=True)
    print("--- 执行前向传播 ---")
    # 执行 forward
    ste_indices, indices, balance_loss = router(input_tensor)
    print("\n--- 输出结果 ---")
    print(f"STE Indices (sg_indices) 的形状: {ste_indices.shape}")
    print(f"Balance Loss 的值: {balance_loss.item()}")
    # 简单验证梯度传播
    print("\n--- 梯度传播测试 ---")
    # 对 balance_loss 和 sg_indices 的和进行反向传播，以确保所有路径都有梯度
    grad_output = torch.randn(ste_indices.shape, dtype=ste_indices.dtype, device=ste_indices.device)
    total_loss = balance_loss + (ste_indices * grad_output).sum()
    total_loss.backward()
    if router.weight.grad is not None:
        print("✅ 成功: 路由器的权重 (self.weight) 接收到了梯度。")
    else:
        print("❌ 失败: 路由器的权重没有梯度。")
    if input_tensor.grad is not None:
        print("✅ 成功: 输入张量 (x) 接收到了梯度。")
    else:
        print("❌ 失败: 输入张量没有梯度。")

    pdb.set_trace()
    