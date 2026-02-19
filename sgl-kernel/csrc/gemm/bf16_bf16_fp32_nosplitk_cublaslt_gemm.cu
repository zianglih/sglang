#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <torch/all.h>

void bf16_bf16_fp32_nosplitk_cublaslt_gemm(
    at::Tensor& output, const at::Tensor& mat_a, const at::Tensor& mat_b, int64_t cublas_handle) {
  TORCH_CHECK(mat_a.is_cuda(), "mat_a must be a CUDA tensor");
  TORCH_CHECK(mat_b.is_cuda(), "mat_b must be a CUDA tensor");
  TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
  TORCH_CHECK(mat_a.dim() == 2, "mat_a must be a 2D tensor");
  TORCH_CHECK(mat_b.dim() == 2, "mat_b must be a 2D tensor");
  TORCH_CHECK(output.dim() == 2, "output must be a 2D tensor");
  TORCH_CHECK(mat_a.scalar_type() == at::kBFloat16, "mat_a must be bf16");
  TORCH_CHECK(mat_b.scalar_type() == at::kBFloat16, "mat_b must be bf16");
  TORCH_CHECK(output.scalar_type() == at::kFloat, "output must be fp32");
  TORCH_CHECK(mat_a.size(1) == mat_b.size(1), "mat_a and mat_b must have the same K dimension");
  TORCH_CHECK(mat_a.is_contiguous(), "mat_a must be contiguous");
  TORCH_CHECK(mat_b.is_contiguous(), "mat_b must be contiguous");
  TORCH_CHECK(output.is_contiguous(), "output must be contiguous");

  const int64_t m = mat_a.size(0);
  const int64_t k = mat_a.size(1);
  const int64_t n = mat_b.size(0);
  TORCH_CHECK(output.size(0) == m && output.size(1) == n, "output shape must be [mat_a.size(0), mat_b.size(0)]");

  auto lt_handle = reinterpret_cast<cublasLtHandle_t>(cublas_handle);
  auto stream = at::cuda::getCurrentCUDAStream();

  cublasLtMatmulDesc_t matmul_desc = nullptr;
  cublasLtMatrixLayout_t a_desc = nullptr;
  cublasLtMatrixLayout_t b_desc = nullptr;
  cublasLtMatrixLayout_t c_desc = nullptr;
  cublasLtMatmulPreference_t preference = nullptr;

  auto cleanup = [&]() {
    if (preference != nullptr) {
      cublasLtMatmulPreferenceDestroy(preference);
    }
    if (c_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(c_desc);
    }
    if (b_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(b_desc);
    }
    if (a_desc != nullptr) {
      cublasLtMatrixLayoutDestroy(a_desc);
    }
    if (matmul_desc != nullptr) {
      cublasLtMatmulDescDestroy(matmul_desc);
    }
  };

  auto check_status = [&](cublasStatus_t status, const char* msg) {
    if (status != CUBLAS_STATUS_SUCCESS) {
      cleanup();
      TORCH_CHECK(false, msg, ": ", cublasGetStatusString(status));
    }
  };

  check_status(cublasLtMatmulDescCreate(&matmul_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F), "cublasLtMatmulDescCreate failed");

  cublasOperation_t trans_a = CUBLAS_OP_T;
  cublasOperation_t trans_b = CUBLAS_OP_N;
  check_status(
      cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSA, &trans_a, sizeof(trans_a)),
      "cublasLtMatmulDescSetAttribute TRANSA failed");
  check_status(
      cublasLtMatmulDescSetAttribute(matmul_desc, CUBLASLT_MATMUL_DESC_TRANSB, &trans_b, sizeof(trans_b)),
      "cublasLtMatmulDescSetAttribute TRANSB failed");

  // cuBLASLt uses column-major descriptors. For row-major GEMM C = A * B^T:
  // C^T = B * A^T.
  check_status(cublasLtMatrixLayoutCreate(&a_desc, CUDA_R_16BF, k, n, k), "cublasLtMatrixLayoutCreate A failed");
  check_status(cublasLtMatrixLayoutCreate(&b_desc, CUDA_R_16BF, k, m, k), "cublasLtMatrixLayoutCreate B failed");
  check_status(cublasLtMatrixLayoutCreate(&c_desc, CUDA_R_32F, n, m, n), "cublasLtMatrixLayoutCreate C failed");

  check_status(cublasLtMatmulPreferenceCreate(&preference), "cublasLtMatmulPreferenceCreate failed");
  constexpr size_t workspace_size = 0;
  check_status(
      cublasLtMatmulPreferenceSetAttribute(
          preference, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &workspace_size, sizeof(workspace_size)),
      "cublasLtMatmulPreferenceSetAttribute failed");
  constexpr uint32_t reduction_scheme_mask = CUBLASLT_REDUCTION_SCHEME_NONE;
  check_status(
      cublasLtMatmulPreferenceSetAttribute(
          preference,
          CUBLASLT_MATMUL_PREF_REDUCTION_SCHEME_MASK,
          &reduction_scheme_mask,
          sizeof(reduction_scheme_mask)),
      "cublasLtMatmulPreferenceSetAttribute reduction scheme mask failed");

  constexpr int kMaxAlgos = 32;
  cublasLtMatmulHeuristicResult_t heuristic_results[kMaxAlgos];
  int returned_results = 0;
  check_status(
      cublasLtMatmulAlgoGetHeuristic(
          lt_handle,
          matmul_desc,
          a_desc,
          b_desc,
          c_desc,
          c_desc,
          preference,
          kMaxAlgos,
          heuristic_results,
          &returned_results),
      "cublasLtMatmulAlgoGetHeuristic failed");
  TORCH_CHECK(
      returned_results > 0, "No cublasLt algorithm found for bf16_bf16_fp32_nosplitk_cublaslt_gemm");

  constexpr float alpha = 1.0f;
  constexpr float beta = 0.0f;
  cublasStatus_t matmul_status = CUBLAS_STATUS_NOT_SUPPORTED;
  for (int i = 0; i < returned_results; ++i) {
    auto algo = heuristic_results[i].algo;
    uint64_t numerical_impl_flags = 0;
    size_t size_written = 0;
    auto numerical_impl_status = cublasLtMatmulAlgoCapGetAttribute(
        &algo,
        CUBLASLT_ALGO_CAP_NUMERICAL_IMPL_FLAGS,
        &numerical_impl_flags,
        sizeof(numerical_impl_flags),
        &size_written);
    if (numerical_impl_status != CUBLAS_STATUS_SUCCESS || size_written != sizeof(numerical_impl_flags)) {
      continue;
    }
    if ((numerical_impl_flags & CUBLASLT_NUMERICAL_IMPL_FLAGS_ACCUMULATOR_32F) == 0) {
      continue;
    }
    constexpr int split_k = 1;
    auto splitk_status = cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_SPLITK_NUM, &split_k, sizeof(split_k));
    if (splitk_status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    constexpr int reduction_scheme = CUBLASLT_REDUCTION_SCHEME_NONE;
    auto reduction_status = cublasLtMatmulAlgoConfigSetAttribute(
        &algo, CUBLASLT_ALGO_CONFIG_REDUCTION_SCHEME, &reduction_scheme, sizeof(reduction_scheme));
    if (reduction_status != CUBLAS_STATUS_SUCCESS) {
      continue;
    }
    matmul_status = cublasLtMatmul(
        lt_handle,
        matmul_desc,
        &alpha,
        mat_b.data_ptr(),
        a_desc,
        mat_a.data_ptr(),
        b_desc,
        &beta,
        output.data_ptr(),
        c_desc,
        output.data_ptr(),
        c_desc,
        &algo,
        nullptr,
        0,
        stream);
    if (matmul_status == CUBLAS_STATUS_SUCCESS) {
      break;
    }
  }

  cleanup();
  TORCH_CHECK(
      matmul_status == CUBLAS_STATUS_SUCCESS,
      "bf16_bf16_fp32_nosplitk_cublaslt_gemm cublasLtMatmul failed: ",
      cublasGetStatusString(matmul_status));
}
