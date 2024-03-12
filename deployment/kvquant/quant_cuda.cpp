#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

void vecquant4appendvecK_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
);
void vecquant4appendvecK(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecK_cuda(mat, lookup_table, newvec, kcachelen);
}

void vecquant3appendvecK_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
);
void vecquant3appendvecK(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecK_cuda(mat, lookup_table, newvec, kcachelen);
}

void vecquant2appendvecK_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
);
void vecquant2appendvecK(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecK_cuda(mat, lookup_table, newvec, kcachelen);
}

void vecquant4appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
void vecquant4appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecKsparse_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

void vecquant4appendvecKsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant4appendvecKsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecKsparseParallel_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper);
}

void vecquant3appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
void vecquant3appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecKsparse_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

void vecquant3appendvecKsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant3appendvecKsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecKsparseParallel_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper);
}

void vecquant2appendvecKsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
void vecquant2appendvecKsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecKsparse_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

void vecquant2appendvecKsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant2appendvecKsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outliers_rescaled, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecKsparseParallel_cuda(mat, lookup_table, newvec, outliers_rescaled, outlier_threshold_lower, outlier_threshold_upper);
}


void vecquant4appendvecV_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
);
void vecquant4appendvecV(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecV_cuda(mat, lookup_table, newvec, vcachelen);
}

void vecquant3appendvecV_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
);
void vecquant3appendvecV(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecV_cuda(mat, lookup_table, newvec, vcachelen);
}

void vecquant2appendvecV_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
);
void vecquant2appendvecV(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecV_cuda(mat, lookup_table, newvec, vcachelen);
}

void vecquant4appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
void vecquant4appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

void vecquant4appendvecVsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant4appendvecVsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant4appendvecVsparseParallel_cuda(mat, lookup_table, newvec, outlier_threshold_lower, outlier_threshold_upper);
}


void vecquant3appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
void vecquant3appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

void vecquant3appendvecVsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant3appendvecVsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant3appendvecVsparseParallel_cuda(mat, lookup_table, newvec, outlier_threshold_lower, outlier_threshold_upper);
}

void vecquant2appendvecVsparse_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
void vecquant2appendvecVsparse(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecVsparse_cuda(mat, lookup_table, newvec, zeropoint, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

void vecquant2appendvecVsparseParallel_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
);
void vecquant2appendvecVsparseParallel(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  vecquant2appendvecVsparseParallel_cuda(mat, lookup_table, newvec, outlier_threshold_lower, outlier_threshold_upper);
}


void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, kcachelen, theta, pos_offset);
}

void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, kcachelen, outliers, outlier_indices, theta, pos_offset);
}

void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, vcachelen);
}

void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, vcachelen, outliers, outlier_indices);
}



void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
);
void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, kcachelen, theta, pos_offset);
}

void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
);
void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, kcachelen, outliers, outlier_indices, theta, pos_offset);
}

void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
);
void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, vcachelen);
}

void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
);
void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, vcachelen, outliers, outlier_indices);
}


void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
);
void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, kcachelen, theta, pos_offset);
}

void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
);
void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices, float theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, kcachelen, outliers, outlier_indices, theta, pos_offset);
}

void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
);
void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(vec, mat, mul, lookup_table, vcachelen);
}

void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
);
void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor outliers, torch::Tensor outlier_indices
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(vec, mat, mul, lookup_table, vcachelen, outliers, outlier_indices);
}

void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startrows,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz,
  float rope_theta, int pos_offset
);
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int kcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startrows,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz,
  float rope_theta, int pos_offset
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig_cuda(vec, mat, mul, lookup_table, kcachelen, rows, cols, startrows, spmat, num_rows, num_threads, nnz, rope_theta, pos_offset);
}

void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startcols,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz
);
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor lookup_table, int vcachelen,
  torch::Tensor rows, torch::Tensor cols, torch::Tensor startcols,
  torch::Tensor spmat, int num_rows, int num_threads, int nnz
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig_cuda(vec, mat, mul, lookup_table, vcachelen, rows, cols, startcols, spmat, num_rows, num_threads, nnz);
}

std::vector<torch::Tensor> vecquant4appendvecKsparseorig_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_rows, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
);
std::vector<torch::Tensor> vecquant4appendvecKsparseorig(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, torch::Tensor zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_rows, torch::Tensor outlier_threshold_lower, torch::Tensor outlier_threshold_upper, int kcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return vecquant4appendvecKsparseorig_cuda(mat, lookup_table, newvec, zeropoint, row, col, val, start_rows, outlier_threshold_lower, outlier_threshold_upper, kcachelen);
}

std::vector<torch::Tensor> vecquant4appendvecVsparseorig_cuda(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_cols, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
);
std::vector<torch::Tensor> vecquant4appendvecVsparseorig(
  torch::Tensor mat, torch::Tensor lookup_table, torch::Tensor newvec, float zeropoint, torch::Tensor row, torch::Tensor col, torch::Tensor val, torch::Tensor start_cols, float outlier_threshold_lower, float outlier_threshold_upper, int vcachelen
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(mat));
  return vecquant4appendvecVsparseorig_cuda(mat, lookup_table, newvec, zeropoint, row, col, val, start_cols, outlier_threshold_lower, outlier_threshold_upper, vcachelen);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt, "4-bit value cache matrix-vector operation");
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2, "4-bit value cache matrix-vector operation (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt, "4-bit key cache matrix-vector operation");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2, "4-bit key cache matrix-vector operation (including sparsity)");
  m.def("vecquant4appendvecK", &vecquant4appendvecK, "Append 4-bit key vector to the key cache");
  m.def("vecquant4appendvecKsparse", &vecquant4appendvecKsparse, "Append 4-bit key vector to the key cache (including sparsity)");
  m.def("vecquant4appendvecKsparseParallel", &vecquant4appendvecKsparseParallel, "Append 4-bit key vectors to the key cache (including sparsity)");
  m.def("vecquant4appendvecV", &vecquant4appendvecV, "Append 4-bit value vector to the value cache");
  m.def("vecquant4appendvecVsparse", &vecquant4appendvecVsparse, "Append 4-bit value vector to the value cache (including sparsity)");
  m.def("vecquant4appendvecVsparseParallel", &vecquant4appendvecVsparseParallel, "Append 4-bit value vectors to the value cache (including sparsity)");
  m.def("vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt", &vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt, "3-bit value cache matrix-vector operation");
  m.def("vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2", &vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2, "3-bit value cache matrix-vector operation (including sparsity)");
  m.def("vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt", &vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt, "3-bit key cache matrix-vector operation");
  m.def("vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2", &vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2, "3-bit key cache matrix-vector operation (including sparsity)");
  m.def("vecquant3appendvecK", &vecquant3appendvecK, "Append 3-bit key vector to the key cache");
  m.def("vecquant3appendvecKsparse", &vecquant3appendvecKsparse, "Append 3-bit key vector to the key cache (including sparsity)");
  m.def("vecquant3appendvecKsparseParallel", &vecquant3appendvecKsparseParallel, "Append 3-bit key vectors to the key cache (including sparsity)");
  m.def("vecquant3appendvecV", &vecquant3appendvecV, "Append 3-bit value vector to the value cache");
  m.def("vecquant3appendvecVsparse", &vecquant3appendvecVsparse, "Append 3-bit value vector to the value cache (including sparsity)");
  m.def("vecquant3appendvecVsparseParallel", &vecquant3appendvecVsparseParallel, "Append 3-bit value vectors to the value cache (including sparsity)");
  m.def("vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt", &vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt, "2-bit value cache matrix-vector operation");
  m.def("vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2", &vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2, "2-bit value cache matrix-vector operation (including sparsity)");
  m.def("vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt", &vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt, "2-bit key cache matrix-vector operation");
  m.def("vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2", &vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2, "2-bit key cache matrix-vector operation (including sparsity)");
  m.def("vecquant2appendvecK", &vecquant2appendvecK, "Append 2-bit key vector to the key cache");
  m.def("vecquant2appendvecKsparse", &vecquant2appendvecKsparse, "Append 2-bit key vector to the key cache (including sparsity)");
  m.def("vecquant2appendvecKsparseParallel", &vecquant2appendvecKsparseParallel, "Append 2-bit key vectors to the key cache (including sparsity)");
  m.def("vecquant2appendvecV", &vecquant2appendvecV, "Append 2-bit value vector to the value cache");
  m.def("vecquant2appendvecVsparse", &vecquant2appendvecVsparse, "Append 2-bit value vector to the value cache (including sparsity)");
  m.def("vecquant2appendvecVsparseParallel", &vecquant2appendvecVsparseParallel, "Append 2-bit value vectors to the value cache (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig", &vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig, "4-bit key cache matrix-vector operation");
  m.def("vecquant4appendvecKsparseorig", &vecquant4appendvecKsparseorig, "Append 4-bit key vector to the key cache (including sparsity)");
  m.def("vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig", &vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig, "4-bit value cache matrix-vector operation");
  m.def("vecquant4appendvecVsparseorig", &vecquant4appendvecVsparseorig, "Append 4-bit value vector to the value cache (including sparsity)");
}
