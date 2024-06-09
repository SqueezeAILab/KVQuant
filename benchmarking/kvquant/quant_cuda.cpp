#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, torch::Tensor scalingfactor, torch::Tensor zeropoint, int kcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startrows,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz,
  float rope_theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, torch::Tensor scalingfactor, torch::Tensor zeropoint, int kcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startrows,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz,
  float rope_theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_cuda(vec, mat, mul, lookup_table, scalingfactor, zeropoint, kcachelen, rows, cols, startrows, spmat, num_rows, num_threads, nnz, rope_theta, pos_offset);
}


void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, torch::Tensor scalingfactor, torch::Tensor zeropoint, int vcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startcols,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, torch::Tensor scalingfactor, torch::Tensor zeropoint, int vcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startcols,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_cuda(vec, mat, mul, lookup_table, scalingfactor, zeropoint, vcachelen, rows, cols, startcols, spmat, num_rows, num_threads, nnz);
}

std::vector<torch::Tensor> vecquant4appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_rows, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
std::vector<torch::Tensor> vecquant4appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_rows, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return vecquant4appendvecKsparse_cuda(mat, lookup_table, newvec, zeropoint, row, col, val, start_rows, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

std::vector<torch::Tensor> vecquant4appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_cols, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
std::vector<torch::Tensor> vecquant4appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_cols, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return vecquant4appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, row, col, val, start_cols, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant4appendvecKsparse", &vecquant4appendvecKsparse, "Append 4-bit key vector to the key cache (including sparsity)");
  m.def("vecquant4appendvecVsparse", &vecquant4appendvecVsparse, "Append 4-bit value vector to the value cache (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused, "4-bit key cache matrix-vector operation");
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused, "4-bit value cache matrix-vector operation (including sparsity)");
}
