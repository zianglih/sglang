import os

import torch
import torch.nn.functional as F
import triton
import triton.testing
from sgl_kernel import bf16_bf16_fp32_nosplitk_cublaslt_gemm

# DeepSeek-V3.2 indexer weights_proj workload:
# - hidden_size (K): 7168
# - index_n_heads (N): 64
# - index_topk: 2048
# Reference:
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/config.json
#   https://huggingface.co/deepseek-ai/DeepSeek-V3.2/blob/main/inference/model.py
HIDDEN_SIZE = 7168
INDEX_N_HEADS = 64
HEAD_SCALE = INDEX_N_HEADS**-0.5
BATCH_SIZES = [1, 2, 4, 8, 16]
CONTEXT_LENGTHS = [1024, 8192, 65536]  # 1k, 8k, 64k

# CI environment detection
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

if IS_CI:
    context_length_vals = [1024]
    batch_size_vals = [1, 2]
else:
    context_length_vals = CONTEXT_LENGTHS
    batch_size_vals = BATCH_SIZES

IMPLS = ["original", "sgl-kernel"]
IMPL_STYLES = {
    "original": ("blue", "-"),
    "sgl-kernel": ("orange", "-"),
}
IMPL_NAMES = {
    "original": "original weights_proj (torch bf16 linear -> fp32)",
    "sgl-kernel": "sgl-kernel no-splitK (bf16xbf16->fp32)",
}

line_vals = []
line_names = []
styles = []
for batch_size in batch_size_vals:
    for impl in IMPLS:
        line_vals.append(f"{impl}|bs{batch_size}")
        line_names.append(f"{IMPL_NAMES[impl]} bs={batch_size}")
        styles.append(IMPL_STYLES[impl])


def _estimate_case_bytes(num_tokens: int, impl: str) -> int:
    # Rough peak estimate for allocations from inputs + outputs per benchmark case.
    mat_a = num_tokens * HIDDEN_SIZE * torch.finfo(torch.bfloat16).bits // 8
    mat_b = INDEX_N_HEADS * HIDDEN_SIZE * torch.finfo(torch.bfloat16).bits // 8
    out_fp32 = num_tokens * INDEX_N_HEADS * torch.finfo(torch.float32).bits // 8
    # The torch original path materializes bf16 output before casting to fp32.
    out_bf16 = (
        num_tokens * INDEX_N_HEADS * torch.finfo(torch.bfloat16).bits // 8
        if impl == "original"
        else 0
    )
    return int(mat_a + mat_b + out_fp32 + out_bf16)


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["context_length"],
        x_vals=context_length_vals,
        x_log=False,
        line_arg="impl",
        line_vals=line_vals,
        line_names=line_names,
        styles=styles,
        ylabel="TFLOPs",
        plot_name="deepseek_v3_2_indexer_weights_proj_bs_ctx",
        args={},
    )
)
def benchmark(context_length, impl):
    impl_name, batch_size_token = impl.split("|")
    batch_size = int(batch_size_token.replace("bs", ""))
    m = batch_size * context_length
    k = HIDDEN_SIZE
    n = INDEX_N_HEADS
    estimated_case_bytes = _estimate_case_bytes(m, impl_name)
    free_bytes, _ = torch.cuda.mem_get_info()
    if estimated_case_bytes > int(free_bytes * 0.8):
        return float("nan"), float("nan"), float("nan")

    mat_a = torch.randn((m, k), dtype=torch.bfloat16, device="cuda").contiguous()
    mat_b_bf16 = torch.randn((n, k), dtype=torch.bfloat16, device="cuda").contiguous()

    quantiles = [0.5, 0.2, 0.8]
    if impl_name == "original":
        runner = lambda: (
            F.linear(mat_a, mat_b_bf16).to(torch.float32) * HEAD_SCALE
        )
    elif impl_name == "sgl-kernel":
        runner = lambda: (
            bf16_bf16_fp32_nosplitk_cublaslt_gemm(mat_a, mat_b_bf16) * HEAD_SCALE
        )
    else:
        raise ValueError(f"Unknown impl: {impl_name}")

    # The contribution guide recommends do_bench_cudagraph for kernel benchmarking.
    try:
        ms, min_ms, max_ms = triton.testing.do_bench_cudagraph(
            runner, quantiles=quantiles
        )
    except RuntimeError as err:
        if "out of memory" in str(err).lower():
            torch.cuda.empty_cache()
            return float("nan"), float("nan"), float("nan")
        raise

    def tflops(t_ms):
        flops = 2 * m * n * k
        return flops / (t_ms * 1e-3) / 1e12

    return tflops(ms), tflops(max_ms), tflops(min_ms)


if __name__ == "__main__":
    benchmark.run(print_data=True)
