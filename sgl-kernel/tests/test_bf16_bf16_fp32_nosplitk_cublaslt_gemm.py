import os

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import bf16_bf16_fp32_nosplitk_cublaslt_gemm

# DeepSeek-V3.2 indexer weights_proj workload:
# - hidden_size (K): 7168
# - index_n_heads (N): 64
HIDDEN_SIZE = 7168
INDEX_N_HEADS = 64
BATCH_SIZES = [1, 2, 4, 8, 16]
CONTEXT_LENGTHS = [1024, 8192, 65536]  # 1k, 8k, 64k
MAX_TEST_TOKENS = int(os.getenv("SGL_KERNEL_BF16_GEMM_MAX_TEST_TOKENS", "4096"))


def _deepseek_ref_weights_proj(
    mat_a_bf16: torch.Tensor, mat_b_bf16: torch.Tensor
) -> torch.Tensor:
    # Matches DeepSeek reference implementation in inference/model.py:
    # weights = self.weights_proj(x.float())
    return F.linear(mat_a_bf16.to(torch.float32), mat_b_bf16.to(torch.float32))


def _original_weights_proj(
    mat_a_bf16: torch.Tensor, mat_b_bf16: torch.Tensor
) -> torch.Tensor:
    # Matches original sglang fallback path:
    # weights, _ = self.weights_proj(x)
    # weights = weights.float()
    return F.linear(mat_a_bf16, mat_b_bf16).to(torch.float32)


@pytest.mark.parametrize(
    "batch_size",
    BATCH_SIZES,
)
@pytest.mark.parametrize("context_length", CONTEXT_LENGTHS)
def test_weights_proj_real_workload_accuracy(batch_size, context_length):
    num_tokens = batch_size * context_length
    if num_tokens > MAX_TEST_TOKENS:
        pytest.skip(
            f"Skipping very large case for unit tests: "
            f"batch_size={batch_size}, context_length={context_length}, "
            f"num_tokens={num_tokens}, max={MAX_TEST_TOKENS}"
        )

    mat_a = torch.randn(
        (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    mat_b = torch.randn(
        (INDEX_N_HEADS, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    ).contiguous()

    output = bf16_bf16_fp32_nosplitk_cublaslt_gemm(mat_a, mat_b)
    ref = _deepseek_ref_weights_proj(mat_a, mat_b)
    original = _original_weights_proj(mat_a, mat_b)

    assert output.dtype == torch.float32
    assert output.shape == (num_tokens, INDEX_N_HEADS)
    assert original.dtype == torch.float32
    assert original.shape == (num_tokens, INDEX_N_HEADS)
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=2e-2)


def test_weights_proj_real_workload_preallocated_output():
    batch_size = 2
    context_length = 1024
    num_tokens = batch_size * context_length
    mat_a = torch.randn(
        (num_tokens, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    mat_b = torch.randn(
        (INDEX_N_HEADS, HIDDEN_SIZE), dtype=torch.bfloat16, device="cuda"
    ).contiguous()
    output = torch.empty(
        (num_tokens, INDEX_N_HEADS), dtype=torch.float32, device="cuda"
    )

    returned = bf16_bf16_fp32_nosplitk_cublaslt_gemm(mat_a, mat_b, out=output)
    ref = _deepseek_ref_weights_proj(mat_a, mat_b)

    assert returned.data_ptr() == output.data_ptr()
    torch.testing.assert_close(output, ref, rtol=1e-2, atol=2e-2)


if __name__ == "__main__":
    pytest.main([__file__])
