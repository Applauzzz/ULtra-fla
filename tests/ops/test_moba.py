import argparse
import time
import torch

from fla.ops.moba import moba_attn_varlen_naive
from fla.ops.moba import moba_attn_varlen


def gen_qkv_and_cu_seqlens(
    seqlens,
    num_heads,
    head_dim,
    device,
    dtype,
):
    torch.manual_seed(0)
    total_seqlen = sum(seqlens)
    max_seqlen = max(seqlens)

    q = torch.randn(
        total_seqlen, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    k = torch.randn(
        total_seqlen, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        total_seqlen, num_heads, head_dim,
        device=device, dtype=dtype, requires_grad=True
    )

    cu_seqlens = torch.tensor(
        [0] + seqlens, device=device, dtype=torch.int32
    ).cumsum(dim=0)

    return q, k, v, cu_seqlens, max_seqlen


@torch.no_grad()
def compare_forward(
    seqlens,
    num_heads,
    head_dim,
    moba_chunk_size,
    moba_topk,
    device,
    dtype,
):
    q, k, v, cu_seqlens, max_seqlen = gen_qkv_and_cu_seqlens(
        seqlens, num_heads, head_dim, device, dtype
    )

    out_naive = moba_attn_varlen_naive(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

    out_eff = moba_attn_varlen(
        q=q,
        k=k,
        v=v,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

    diff = out_naive - out_eff
    abs_err = diff.abs()
    denom = out_naive.abs().clamp_min(1e-6)
    rel_err = abs_err / denom

    print("=== Forward 精度对比 ===")
    print(f"out_naive shape = {out_naive.shape}, out_eff shape = {out_eff.shape}")
    print(f"max |diff|   = {abs_err.max().item():.6e}")
    print(f"mean |diff|  = {abs_err.mean().item():.6e}")
    print(f"max rel diff = {rel_err.max().item():.6e}")
    print(f"mean rel diff= {rel_err.mean().item():.6e}")
    print()


def run_backward_once(
    impl,
    q,
    k,
    v,
    cu_seqlens,
    max_seqlen,
    moba_chunk_size,
    moba_topk,
    grad_out,
):
    q_ = q.clone().detach().requires_grad_(True)
    k_ = k.clone().detach().requires_grad_(True)
    v_ = v.clone().detach().requires_grad_(True)

    out = impl(
        q=q_,
        k=k_,
        v=v_,
        cu_seqlens=cu_seqlens,
        max_seqlen=max_seqlen,
        moba_chunk_size=moba_chunk_size,
        moba_topk=moba_topk,
    )

    out.backward(grad_out)

    return q_.grad, k_.grad, v_.grad


def compare_backward(
    seqlens,
    num_heads,
    head_dim,
    moba_chunk_size,
    moba_topk,
    device,
    dtype,
):
    torch.manual_seed(1)
    q_base, k_base, v_base, cu_seqlens, max_seqlen = gen_qkv_and_cu_seqlens(
        seqlens, num_heads, head_dim, device, dtype
    )

    grad_out = torch.randn_like(q_base)

    qg_naive, kg_naive, vg_naive = run_backward_once(
        moba_attn_varlen_naive,
        q_base,
        k_base,
        v_base,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size,
        moba_topk,
        grad_out,
    )

    qg_eff, kg_eff, vg_eff = run_backward_once(
        moba_attn_varlen,
        q_base,
        k_base,
        v_base,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size,
        moba_topk,
        grad_out,
    )

    def _print_grad_stats(name, g1, g2):
        diff = g1 - g2
        abs_err = diff.abs()
        denom = g1.abs().clamp_min(1e-6)
        rel_err = abs_err / denom
        print(f"--- {name} grad ---")
        print(f"max |diff|   = {abs_err.max().item():.6e}")
        print(f"mean |diff|  = {abs_err.mean().item():.6e}")
        print(f"max rel diff = {rel_err.max().item():.6e}")
        print(f"mean rel diff= {rel_err.mean().item():.6e}")
        print()

    print("=== Backward 精度对比 ===")
    _print_grad_stats("dq", qg_naive, qg_eff)
    _print_grad_stats("dk", kg_naive, kg_eff)
    _print_grad_stats("dv", vg_naive, vg_eff)


def run_once_for_stress(
    seqlens,
    num_heads,
    head_dim,
    moba_chunk_size,
    moba_topk,
    device,
    dtype,
):
    """
    单次 forward + backward，用于压力测试，返回误差和耗时统计。
    """
    # 生成数据（随机）
    q_base, k_base, v_base, cu_seqlens, max_seqlen = gen_qkv_and_cu_seqlens(
        seqlens, num_heads, head_dim, device, dtype
    )

    # ========== forward ==========
    with torch.no_grad():
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        out_naive = moba_attn_varlen_naive(
            q=q_base,
            k=k_base,
            v=v_base,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        out_eff = moba_attn_varlen(
            q=q_base,
            k=k_base,
            v=v_base,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            moba_chunk_size=moba_chunk_size,
            moba_topk=moba_topk,
        )
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()

    diff = out_naive - out_eff
    abs_err = diff.abs()
    denom = out_naive.abs().clamp_min(1e-6)
    rel_err = abs_err / denom

    fwd_stats = {
        "max_abs": abs_err.max().item(),
        "mean_abs": abs_err.mean().item(),
        "max_rel": rel_err.max().item(),
        "mean_rel": rel_err.mean().item(),
        "time_naive": t1 - t0,
        "time_eff": t2 - t1,
    }

    # ========== backward ==========
    grad_out = torch.randn_like(q_base)

    if device.type == "cuda":
        torch.cuda.synchronize()
    tb0 = time.perf_counter()
    qg_naive, kg_naive, vg_naive = run_backward_once(
        moba_attn_varlen_naive,
        q_base,
        k_base,
        v_base,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size,
        moba_topk,
        grad_out,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb1 = time.perf_counter()
    qg_eff, kg_eff, vg_eff = run_backward_once(
        moba_attn_varlen,
        q_base,
        k_base,
        v_base,
        cu_seqlens,
        max_seqlen,
        moba_chunk_size,
        moba_topk,
        grad_out,
    )
    if device.type == "cuda":
        torch.cuda.synchronize()
    tb2 = time.perf_counter()

    def _grad_err(g1, g2):
        d = g1 - g2
        ae = d.abs()
        denom = g1.abs().clamp_min(1e-6)
        re = ae / denom
        return {
            "max_abs": ae.max().item(),
            "mean_abs": ae.mean().item(),
            "max_rel": re.max().item(),
            "mean_rel": re.mean().item(),
        }

    bwd_stats = {
        "dq": _grad_err(qg_naive, qg_eff),
        "dk": _grad_err(kg_naive, kg_eff),
        "dv": _grad_err(vg_naive, vg_eff),
        "time_naive": tb1 - tb0,
        "time_eff": tb2 - tb1,
    }

    return fwd_stats, bwd_stats


def stress_test(args, device, dtype):
    """
    压力测试：随机多组 varlen seqlens，反复跑 forward + backward。
    """
    if args.stress_iters <= 0:
        return

    print("=== Stress Test 开始 ===")
    print(
        f"[stress] iters={args.stress_iters}, batch_size={args.stress_batch_size}, "
        f"len_range=[{args.stress_min_seqlen}, {args.stress_max_seqlen}]"
    )

    global_fwd_max_abs = 0.0
    global_fwd_max_rel = 0.0
    global_bwd_max_abs = {"dq": 0.0, "dk": 0.0, "dv": 0.0}
    global_bwd_max_rel = {"dq": 0.0, "dk": 0.0, "dv": 0.0}

    sum_time_fwd_naive = 0.0
    sum_time_fwd_eff = 0.0
    sum_time_bwd_naive = 0.0
    sum_time_bwd_eff = 0.0

    for it in range(args.stress_iters):
        torch.manual_seed(args.stress_seed + it)

        # 随机生成一组 varlen 的 seqlens
        seqlens = torch.randint(
            low=args.stress_min_seqlen,
            high=args.stress_max_seqlen + 1,
            size=(args.stress_batch_size,),
        ).tolist()

        fwd_stats, bwd_stats = run_once_for_stress(
            seqlens=seqlens,
            num_heads=args.num_heads,
            head_dim=args.head_dim,
            moba_chunk_size=args.moba_chunk_size,
            moba_topk=args.moba_topk,
            device=device,
            dtype=dtype,
        )

        global_fwd_max_abs = max(global_fwd_max_abs, fwd_stats["max_abs"])
        global_fwd_max_rel = max(global_fwd_max_rel, fwd_stats["max_rel"])

        for name in ["dq", "dk", "dv"]:
            global_bwd_max_abs[name] = max(
                global_bwd_max_abs[name], bwd_stats[name]["max_abs"]
            )
            global_bwd_max_rel[name] = max(
                global_bwd_max_rel[name], bwd_stats[name]["max_rel"]
            )

        sum_time_fwd_naive += fwd_stats["time_naive"]
        sum_time_fwd_eff += fwd_stats["time_eff"]
        sum_time_bwd_naive += bwd_stats["time_naive"]
        sum_time_bwd_eff += bwd_stats["time_eff"]

        if (it + 1) % max(1, args.stress_iters // 5) == 0:
            print(f"[stress] iter {it+1}/{args.stress_iters} done.")

    print("\n=== Stress Test 汇总结果 ===")
    print(f"[Forward] worst max |diff| = {global_fwd_max_abs:.6e}")
    print(f"[Forward] worst max rel   = {global_fwd_max_rel:.6e}")
    print()
    for name in ["dq", "dk", "dv"]:
        print(f"[Backward-{name}] worst max |diff| = {global_bwd_max_abs[name]:.6e}")
        print(f"[Backward-{name}] worst max rel   = {global_bwd_max_rel[name]:.6e}")
    print()
    n = float(args.stress_iters)
    print(f"[Time] forward naive avg  = {sum_time_fwd_naive / n:.6e} s")
    print(f"[Time] forward eff   avg  = {sum_time_fwd_eff / n:.6e} s")
    print(f"[Time] backward naive avg = {sum_time_bwd_naive / n:.6e} s")
    print(f"[Time] backward eff   avg = {sum_time_bwd_eff / n:.6e} s")
    print("=== Stress Test 结束 ===\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dtype", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    parser.add_argument("--seqlens", type=int, nargs="+", default=[512, 1024, 256])
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=64)
    parser.add_argument("--moba-chunk-size", type=int, default=256)
    parser.add_argument("--moba-topk", type=int, default=2)

    parser.add_argument("--stress-iters", type=int, default=0,
                        help=">0 时开启压力测试，迭代次数")
    parser.add_argument("--stress-batch-size", type=int, default=4)
    parser.add_argument("--stress-min-seqlen", type=int, default=64)
    parser.add_argument("--stress-max-seqlen", type=int, default=2048)
    parser.add_argument("--stress-seed", type=int, default=1234)

    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA 不可用，自动切到 CPU。")
        args.device = "cpu"

    device = torch.device(args.device)
    if args.dtype == "bf16":
        dtype = torch.bfloat16
    elif args.dtype == "fp16":
        dtype = torch.float16
    else:
        dtype = torch.float32

    print("=== 配置 ===")
    print(f"device          = {device}")
    print(f"dtype           = {dtype}")
    print(f"seqlens         = {args.seqlens}")
    print(f"num_heads       = {args.num_heads}")
    print(f"head_dim        = {args.head_dim}")
    print(f"moba_chunk_size = {args.moba_chunk_size}")
    print(f"moba_topk       = {args.moba_topk}")
    print(f"stress_iters    = {args.stress_iters}")
    print("====================\n")

    compare_forward(
        seqlens=args.seqlens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        moba_chunk_size=args.moba_chunk_size,
        moba_topk=args.moba_topk,
        device=device,
        dtype=dtype,
    )

    compare_backward(
        seqlens=args.seqlens,
        num_heads=args.num_heads,
        head_dim=args.head_dim,
        moba_chunk_size=args.moba_chunk_size,
        moba_topk=args.moba_topk,
        device=device,
        dtype=dtype,
    )

    stress_test(args, device, dtype)


if __name__ == "__main__":
    main()

# python test_moba_accuracy.py --device cuda --dtype bf16

# python test_moba.py \
#   --device cuda \
#   --dtype bf16 \
#   --stress-iters 20 \
#   --stress-batch-size 8 \
#   --stress-min-seqlen 64 \
#   --stress-max-seqlen 4096
