#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

// half-tensor
#include <c10/cuda/CUDAStream.h>
#include <ATen/cuda/CUDATensorMethods.cuh>

// Thrust
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <thrust/functional.h>
#include <thrust/device_vector.h>

// atomicAdd for double-precision floating-point numbers on hardware with
// compute capability < 6.0 from:
// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
__device__ double atomicAdd(
    double* address,
    double val
) {
  unsigned long long int* address_as_ull = (unsigned long long int*)address;
  unsigned long long int old = *address_as_ull, assumed;

  do {
    assumed = old;
    old = atomicCAS(
      address_as_ull,
      assumed,
      __double_as_longlong(val + __longlong_as_double(assumed))
    );

  // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
  } while (assumed != old);

  return __longlong_as_double(old);
}
#endif

const int BLOCKWIDTH  = 128;
const int BLOCKHEIGHT2 =  8;
const int BLOCKHEIGHT3 =  12;
const int BLOCKHEIGHT4 =  16;

const int PBLOCKWIDTH  = 32;
const int PBLOCKHEIGHT2 =  2;
const int PBLOCKHEIGHT3 =  3;
const int PBLOCKHEIGHT4 =  4;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_ROPE_BALANCED(
    const  scalar_t* __restrict__ outliers,
    const  int*      __restrict__ outlier_indices,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int kcachelen,
    const  int full_kcachelen,
    int numheads,
    int headdim,
    int num_outliers,
    float rope_theta,
    int pos_offset
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_CSR_ROPE_BALANCED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const       int* __restrict__ startrows,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_rows,
    int numheads,
    int seqlen,
    int headdim,
    int num_threads,
    int nnz,
    float rope_theta,
    int pos_offset
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_CSC_BALANCED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const       int* __restrict__ startcols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int num_cols,
    int numheads,
    int seqlen,
    int headdim,
    int num_threads,
    int nnz
);


template <typename scalar_t>
__global__ void SPMV_ATOMIC_BALANCED(
    const  scalar_t* __restrict__ outliers,
    const       int* __restrict__ outlier_indices,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    int seqlen,
    int fullwidth,
    int numheads,
    int headdim,
    int num_outliers
);

__global__ void VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size
);

__global__ void VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size,
    float rope_theta,
    int pos_offset
);

__global__ void VecQuant3MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size
);

__global__ void VecQuant3MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size,
    float rope_theta,
    int pos_offset
);

__global__ void VecQuant2MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size
);

__global__ void VecQuant2MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size,
    float rope_theta,
    int pos_offset
);


__global__ void VecQuant4AppendVecK(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
);

__global__ void VecQuant4AppendVecV(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
);

__global__ void VecQuant3AppendVecK(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
);

__global__ void VecQuant3AppendVecV(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
);

__global__ void VecQuant2AppendVecK(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
);

__global__ void VecQuant2AppendVecV(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
);


__global__ void VecQuant4AppendVecKSparse(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth
);

__global__ void VecQuant4AppendVecKSparseParallel(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth,
  int headdim
);

__global__ void VecQuant4AppendVecVSparse(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
);

__global__ void VecQuant4AppendVecVSparseParallel(
    int* __restrict__ mat,
  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth,
  int headdim
);

__global__ void VecQuant3AppendVecKSparse(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth
);

__global__ void VecQuant3AppendVecKSparseParallel(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth,
  int headdim
);

__global__ void VecQuant3AppendVecVSparse(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
);

__global__ void VecQuant3AppendVecVSparseParallel(
    int* __restrict__ mat,
  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth,
  int headdim
);

__global__ void VecQuant2AppendVecKSparse(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth
);

__global__ void VecQuant2AppendVecKSparseParallel(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth,
  int headdim
);

__global__ void VecQuant2AppendVecVSparse(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
);

__global__ void VecQuant2AppendVecVSparseParallel(
    int* __restrict__ mat,
  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth,
  int headdim
);

__global__ void VecQuant4AppendVecKSparseOrig(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count,
  int*  __restrict__ outlier_count_per_block,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth
);

__global__ void VecQuant4AppendVecKSparse2Orig(
  const  float* __restrict__ newvec,
  float* __restrict__ zeropoint,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count_per_block,
  int*  __restrict__ dst_indices,
  float*  __restrict__ dst_values,
  int num_blocks
);

__global__ void VecQuant4AppendVecVSparseOrig(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int* __restrict__ mask,
  int* __restrict__ outlier_count,
  int*  __restrict__ outlier_count_per_block,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
);

__global__ void VecQuant4AppendVecVSparse2Orig(
  const  float* __restrict__ newvec,
  float zeropoint,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count_per_block,
  int*  __restrict__ dst_indices,
  float*  __restrict__ dst_values,
  int num_blocks
);

template <typename scalar_t>
__global__ void SPMV_ATOMIC_BALANCED(
    const  scalar_t* __restrict__ outliers,
    const       int* __restrict__ outlier_indices,
    const  scalar_t* __restrict__ vec,
           scalar_t* __restrict__ mul,
    int seqlen,
    int fullwidth,
    int numheads,
    int headdim,
    int num_outliers
) {

    int threadid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadid < seqlen) {

        int start = threadid * num_outliers;
        int end = (threadid+1) * num_outliers;
        int col = threadid;

        float dot = 0;

        for (int i = start; i < end; i++) {

            int row = outlier_indices[i];
            int headid = row / headdim;
            float vval = vec[headid * seqlen + col];

            dot = outliers[i] * vval;

            atomicAdd(&mul[row], dot);
        }
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_ROPE_BALANCED(
    const  scalar_t* __restrict__ outliers,
    const  int*      __restrict__ outlier_indices,
    const  scalar_t* __restrict__ vec,
	         scalar_t* __restrict__ mul,
    const  int kcachelen,
    const  int full_kcachelen,
    int numheads,
    int headdim,
    int num_outliers,
    float rope_theta,
    int pos_offset
) {

    int threadid = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadid < kcachelen) {

        float theta;
        float sign;
        float c, s;

        float dot = 0;

        int start = threadid * num_outliers;
        int end = (threadid+1) * num_outliers;
        int row = threadid;

        for (int i = start; i < end; i++) {
            int col = outlier_indices[i];
            float mat_tmp = outliers[i];

            int headid = col / headdim;
            int channel_head_off = col % headdim; // needed for RoPE pos

            // RoPE embeddings
            theta = powf ( rope_theta , (-2 * __int2float_rd(channel_head_off % (headdim/2)) / headdim) );
            sign = (channel_head_off < (headdim/2)) ? 1 : -1;
            c = cosf(theta * (row + pos_offset));
            s = sinf(theta * (row + pos_offset));

            // compute dot products
            int col2 = ((channel_head_off + (headdim/2)) % headdim ) + headid * headdim;
            dot = mat_tmp * c * vec[col];
            dot += sign * mat_tmp * s * vec[col2];

            atomicAdd(&mul[headid * kcachelen + row], dot);
        }
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_CSR_ROPE_BALANCED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const       int* __restrict__ startrows,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
           scalar_t* __restrict__ mul,
    const  int num_rows,
    int numheads,
    int seqlen,
    int headdim,
    int num_threads,
    int nnz,
    float rope_theta,
    int pos_offset
) {

    int nnz_per_thread = (nnz + num_threads - 1) / num_threads;
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadid < num_threads) {

      int row = startrows[threadid];
      int nextrow = -1;
      if (row != -1) {
        nextrow = rows[row+1];

        // extra check to make sure we don't start with an empty row!
        while (nextrow == threadid * nnz_per_thread) {
          row += 1;
          if (row < num_rows) {
            nextrow = rows[row+1];
          } else {
            break;
          }
        }
      }

      float theta;
      float sign;
      float c, s;

      if (threadid*nnz_per_thread < nnz && row != -1) {

          int max = (threadid+1)*nnz_per_thread;
          if (nnz < max) {
            max = nnz;
          }

          float dot = 0;

          for (int i = threadid * nnz_per_thread; i < max; i++) {

              int col = cols[i];
              float mat_tmp = mat[i];

              int headid = col / headdim;
              int channel_head_off = col % headdim; // needed for RoPE pos

              // RoPE embeddings
              theta = powf ( rope_theta , (-2 * __int2float_rd(channel_head_off % (headdim/2)) / headdim) );
              sign = (channel_head_off < (headdim/2)) ? 1 : -1;
              c = cosf(theta * (row + pos_offset));
              s = sinf(theta * (row + pos_offset));

              // compute dot products
              int col2 = ((channel_head_off + (headdim/2)) % headdim ) + headid * headdim;
              dot = mat_tmp * c * vec[col];
              dot += sign * mat_tmp * s * vec[col2];

              atomicAdd(&mul[headid * seqlen + row], dot);

              if (i + 1 == nextrow) { // finish & move on to next row

                  dot = 0;

                  while (i + 1 == nextrow) { // while loop is to deal with cases where there are entire zero rows
                      row += 1;
                      if (row < num_rows) {
                          nextrow = rows[row+1];
                      } else {
                          nextrow = -1;
                          break;
                      }
                  }
              }

          }
      }
    }
}

template <typename scalar_t>
__global__ void SPMV_ATOMIC_CSC_BALANCED(
    const       int* __restrict__ rows,
    const       int* __restrict__ cols,
    const       int* __restrict__ startcols,
    const  scalar_t* __restrict__ mat,
    const  scalar_t* __restrict__ vec,
           scalar_t* __restrict__ mul,
    const  int num_cols,
    int numheads,
    int seqlen,
    int headdim,
    int num_threads,
    int nnz
) {

    int nnz_per_thread = (nnz + num_threads - 1) / num_threads;
    int threadid = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadid < num_threads) {

      int col = startcols[threadid];
      int nextcol = -1;
      if (col != -1) {
        nextcol = cols[col+1];

        // extra check to make sure we don't start with an empty row!
        while (nextcol == threadid * nnz_per_thread) {
          col += 1;
          if (col < num_cols) {
            nextcol = cols[col+1];
          } else {
            break;
          }
        }
      }


      if (threadid*nnz_per_thread < nnz && col != -1) {

          int max = (threadid+1)*nnz_per_thread;
          if (nnz < max) {
            max = nnz;
          }

          float dot = 0;

          for (int i = threadid * nnz_per_thread; i < max; i++) {

              int row = rows[i];
              int headid = row / headdim;
              float vval = vec[headid * seqlen + col];

              dot = mat[i] * vval;

              atomicAdd(&mul[row], dot);

              if (i + 1 == nextcol) { // finish & move on to next row
                  dot = 0;

                  while (i + 1 == nextcol) { // while loop is to deal with cases where there are entire zero rows
                      col += 1;
                      if (col < num_cols) {
                          nextcol = cols[col+1];
                      } else {
                          nextcol = -1;
                          break;
                      }
                  }
              }
          }
      }
    }
}

std::vector<torch::Tensor> vecquant4appendvecKsparseorig_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor zeropoint,
  torch::Tensor row,
  torch::Tensor col,
  torch::Tensor val,
  torch::Tensor start_rows,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  auto options1 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor outlier_count_per_block = torch::zeros(num_blocks,options1);

  auto options2 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor outlier_count = torch::zeros(1,options2);

  auto options3 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor mask = torch::zeros_like(newvec,options3);

  VecQuant4AppendVecKSparseOrig<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    mask.data_ptr<int>(),
    outlier_count.data_ptr<int>(),
    outlier_count_per_block.data_ptr<int>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    width,
    fullwidth
  );

  torch::Tensor hostcount = outlier_count.to(torch::kCPU);
  int* count = hostcount.data_ptr<int>();
  int intcount = count[0];
  auto options4 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor dst_indices = torch::zeros(intcount,options4);
  auto options5 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor dst_values = torch::zeros(intcount, options5);

  VecQuant4AppendVecKSparse2Orig<<<num_blocks, block_size>>>(
    newvec.data_ptr<float>(),
    zeropoint.data_ptr<float>(),
    mask.data_ptr<int>(),
    outlier_count_per_block.data_ptr<int>(),
    dst_indices.data_ptr<int>(),
    dst_values.data_ptr<float>(),
    num_blocks
  );

  torch::Tensor row2, col2, val2, start_rows2;
  int num_threads2;

  // Deal w/ rows / cols / vals
  if (!row.numel()) {
    auto options6 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor dst_row_cpu = torch::zeros(2,options6);
    int* row_ptr = dst_row_cpu.data_ptr<int>();
    row_ptr[1] = intcount;
    torch::Tensor dst_row = dst_row_cpu.to(torch::kCUDA);

    row2 = dst_row;
    col2 = dst_indices;
    val2 = dst_values;

    // balanced part - TODO make parameterizable (currently assumes 10 nnz per thread)
    int nnz_per_thread = 10;
    int num_nonzeros = intcount;
    num_threads2 = (num_nonzeros+9) / 10 ;

    // currently initialize on CPU and copy, see if this is fast enough
    auto options8 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    start_rows2 = torch::full(num_threads2, kcachelen, options8);

  } else {

    auto options7 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor dst_row_cuda = torch::full(1, col.size(0) + intcount, options7);

    int prev_num_nonzeros = col.size(0);
    int prevmax = start_rows.size(0);
    int prev_num_threads = (prev_num_nonzeros+9)/10;

    row2 = torch::cat({row,dst_row_cuda}, 0);
    if (intcount > 0) {
      col2 = torch::cat({col,dst_indices}, 0);
      val2 = torch::cat({val,dst_values}, 0);

      int nnz_per_thread = 10;
      int num_nonzeros = col2.size(0);
      num_threads2 = (num_nonzeros+9) / 10 ;
      int new_alloc = num_threads2 - prevmax;

      if (new_alloc > 0) {
        auto options9 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        torch::Tensor start_rows2_tmp = torch::full(new_alloc, kcachelen, options9);
        start_rows2 = torch::cat({start_rows,start_rows2_tmp}, 0);
      } else {
        start_rows2 = start_rows;
      }

    } else {
      col2 = col;
      val2 = val;
      start_rows2 = start_rows;
      int num_nonzeros = col2.size(0);
      num_threads2 = (num_nonzeros+9) / 10 ;
    }
  }

  // hack to return int
  auto options10 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor num_threads = torch::zeros(1, options10);
  int* num_threads_ptr = num_threads.data_ptr<int>();
  num_threads_ptr[0] = num_threads2;

  return {row2, col2, val2, start_rows2, num_threads, outlier_count};
}

__global__ void VecQuant4AppendVecKSparse2Orig(
  const  float* __restrict__ newvec,
  float* __restrict__ zeropoint,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count_per_block,
  int*  __restrict__ dst_indices,
  float*  __restrict__ dst_values,
  int num_blocks
) {
  __shared__ int limits[2];
  if (threadIdx.x == 0) {
      int outlier_offset = 0;
      for (int i=0; i<blockIdx.x; i++) {
        outlier_offset += outlier_count_per_block[i];
      }
      limits[0] = outlier_offset;
      limits[1] = outlier_offset + outlier_count_per_block[blockIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int insert_loc = limits[0];
    for (int i=0; i<PBLOCKWIDTH; i++) {
      int outlier_offset = PBLOCKWIDTH * blockIdx.x + i;
      if (mask[outlier_offset] != 0) {
        dst_indices[insert_loc] = outlier_offset;
        float newvecval = newvec[outlier_offset];
        float zeropointval = zeropoint[outlier_offset];
        float insert_val = newvecval - zeropointval;
        dst_values[insert_loc] = insert_val;
        insert_loc += 1;
      }
    }
  }
}

__global__ void VecQuant4AppendVecKSparseOrig(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count,
  int*  __restrict__ outlier_count_per_block,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT4 * blockIdx.x;

  //Modified dequant block
  __shared__ float deq2[16][PBLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 16;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower[lut_row];
  float upper_threshold = outlier_threshold_upper[lut_row];

  int num_outliers = 0;

  int smallest_idx = 0;
  if ((newvecval < lower_threshold) || (newvecval > upper_threshold)) {
    smallest_idx = 7; //zero-point
    mask[lut_row] = 1; // set boolean mask
    num_outliers += 1;
  } else {
    // find index of smallest entry in lut
    float prev_val = deq2[0][off];

    for (int val = 1; val < 16; val += 1) {
      if (deq2[val][off] < prev_val) {
        prev_val = deq2[val][off];
        smallest_idx = val;
      }
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 8);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 8) * 4;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
  atomicAdd(&outlier_count[0], num_outliers);
  atomicAdd(&outlier_count_per_block[blockIdx.x], num_outliers);
}

std::vector<torch::Tensor> vecquant4appendvecVsparseorig_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  float zeropoint,
  torch::Tensor row,
  torch::Tensor col,
  torch::Tensor val,
  torch::Tensor start_cols,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  auto options1 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor outlier_count_per_block = torch::zeros(num_blocks,options1);

  auto options2 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor outlier_count = torch::zeros(1,options2);

  auto options3 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor mask = torch::zeros_like(newvec,options3);

  VecQuant4AppendVecVSparseOrig<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    mask.data_ptr<int>(),
    outlier_count.data_ptr<int>(),
    outlier_count_per_block.data_ptr<int>(),
    outlier_threshold_lower,
    outlier_threshold_upper,
    width,
    fullwidth
  );

  torch::Tensor hostcount = outlier_count.to(torch::kCPU);
  int* count = hostcount.data_ptr<int>();
  int intcount = count[0];
  auto options4 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  torch::Tensor dst_indices = torch::zeros(intcount,options4);
  auto options5 = torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA);
  torch::Tensor dst_values = torch::zeros(intcount, options5);

  VecQuant4AppendVecVSparse2Orig<<<num_blocks, block_size>>>(
    newvec.data_ptr<float>(),
    zeropoint,
    mask.data_ptr<int>(),
    outlier_count_per_block.data_ptr<int>(),
    dst_indices.data_ptr<int>(),
    dst_values.data_ptr<float>(),
    num_blocks
  );

  torch::Tensor row2, col2, val2, start_cols2;
  int num_threads2;

  // Deal w/ rows / cols / vals
  if (!col.numel()) {
    auto options6 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
    torch::Tensor dst_col_cpu = torch::zeros(2,options6);
    int* col_ptr = dst_col_cpu.data_ptr<int>();
    col_ptr[1] = intcount;
    torch::Tensor dst_col = dst_col_cpu.to(torch::kCUDA);

    row2 = dst_indices;
    col2 = dst_col;
    val2 = dst_values;

    // balanced part - TODO make parameterizable
    int nnz_per_thread = 10;
    int num_nonzeros = intcount;
    num_threads2 = (num_nonzeros+9) / 10 ;

    // currently initialize on CPU and copy, see if this is fast enough
    auto options8 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    start_cols2 = torch::full(num_threads2, vcachelen, options8);

  } else {
    auto options7 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
    torch::Tensor dst_col = torch::full(1, row.size(0) + intcount, options7);

    int prev_num_nonzeros = row.size(0);
    int prevmax = start_cols.size(0);
    int prev_num_threads = (prev_num_nonzeros+9)/10;

    col2 = torch::cat({col,dst_col}, 0);
    if (intcount > 0) {
      row2 = torch::cat({row,dst_indices}, 0);
      val2 = torch::cat({val,dst_values}, 0);

      int nnz_per_thread = 10;
      int num_nonzeros = row2.size(0);
      num_threads2 = (num_nonzeros+9) / 10 ;
      int new_alloc = num_threads2 - prevmax;

      if (new_alloc > 0) {
        auto options9 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
        torch::Tensor start_cols2_tmp = torch::full(new_alloc, vcachelen, options9);
        start_cols2 = torch::cat({start_cols,start_cols2_tmp}, 0);
      } else {
        start_cols2 = start_cols;
      }

    } else {
      row2 = row;
      val2 = val;
      start_cols2 = start_cols;
    }
  }

  // hack to return int
  auto options10 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCPU);
  torch::Tensor num_threads = torch::zeros(1, options10);
  int* num_threads_ptr = num_threads.data_ptr<int>();
  num_threads_ptr[0] = num_threads2;

  return {row2, col2, val2, start_cols2, num_threads, outlier_count};

}

__global__ void VecQuant4AppendVecVSparse2Orig(
  const  float* __restrict__ newvec,
  float zeropoint,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count_per_block,
  int*  __restrict__ dst_indices,
  float*  __restrict__ dst_values,
  int num_blocks
) {
  __shared__ int limits[2];
  if (threadIdx.x == 0) {
      int outlier_offset = 0;
      for (int i=0; i<blockIdx.x; i++) {
        outlier_offset += outlier_count_per_block[i];
      }
      limits[0] = outlier_offset;
      limits[1] = outlier_offset + outlier_count_per_block[blockIdx.x];
  }
  __syncthreads();

  if (threadIdx.x == 0) {
    int insert_loc = limits[0];
    for (int i=0; i<PBLOCKWIDTH; i++) {
      int outlier_offset = PBLOCKWIDTH * blockIdx.x + i;
      if (mask[outlier_offset] != 0) {
        dst_indices[insert_loc] = outlier_offset;
        float newvecval = newvec[outlier_offset];
        float insert_val = newvecval - zeropoint;
        dst_values[insert_loc] = insert_val;
        insert_loc += 1;
      }
    }
  }
}

__global__ void VecQuant4AppendVecVSparseOrig(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int* __restrict__ mask,
  int* __restrict__ outlier_count,
  int*  __restrict__ outlier_count_per_block,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT4 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[16][PBLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 16;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower;
  float upper_threshold = outlier_threshold_upper;

  int num_outliers = 0;

  int smallest_idx = 0;
  if ((newvecval < lower_threshold) || (newvecval > upper_threshold)) {
    smallest_idx = 7; //zero-point
    mask[offset] = 1; // set boolean mask
    num_outliers += 1;
  } else {
    // find index of smallest entry in lut
    float prev_val = deq2[0][off];
    for (int val = 1; val < 16; val += 1) {
      if (deq2[val][off] < prev_val) {
        prev_val = deq2[val][off];
        smallest_idx = val;
      }
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 8);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 8) * 4;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
  atomicAdd(&outlier_count[0], num_outliers);
  atomicAdd(&outlier_count_per_block[blockIdx.x], num_outliers);
}



void vecquant4appendvecK_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = BLOCKWIDTH;
  int num_blocks = (newveclen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  VecQuant4AppendVecK<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    width,
    fullwidth
  );

}

__global__ void VecQuant4AppendVecK(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
) {

  int packedoffset = BLOCKHEIGHT4 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 16;

  // get value of vec to pack
  int offset = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // find index of smallest entry in lut
  int smallest_idx = 0;
  float prev_val = deq2[0][off];
  for (int val = 1; val < 16; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 8);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 8) * 4;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}


void vecquant4appendvecV_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = BLOCKWIDTH;
  int num_blocks = (newveclen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  VecQuant4AppendVecV<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    width,
    fullwidth
  );

}

__global__ void VecQuant4AppendVecV(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
) {

  int packedoffset = BLOCKHEIGHT4 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 16;

  // get value of vec to pack
  int offset = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // find index of smallest entry in lut
  int smallest_idx = 0;
  float prev_val = deq2[0][off];
  for (int val = 1; val < 16; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 8);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 8) * 4;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}

void vecquant3appendvecK_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 8)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = BLOCKWIDTH;
  int num_blocks = (newveclen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  VecQuant3AppendVecK<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    width,
    fullwidth
  );

}

__global__ void VecQuant3AppendVecK(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
) {

  int packedoffset = BLOCKHEIGHT3 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 8;

  // get value of vec to pack
  int offset = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // find index of smallest entry in lut
  int smallest_idx = 0;
  float prev_val = deq2[0][off];
  for (int val = 1; val < 8; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int loc = (threadIdx.x % 32);
  if (loc == 10) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 30);
    int word_to_add2 = (smallest_idx >> 2);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else if (loc == 21) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 2;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 31);
    int word_to_add2 = (smallest_idx >> 1);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else {
    int row = packedoffset + (threadIdx.x / 32) * 3 + (loc / 11);
    int i = fullwidth * row + width;
    int word_offset = (loc * 3) % 32;
    int word_to_add = (smallest_idx << word_offset);
    atomicAdd(&mat[i], word_to_add);
  }
}


void vecquant3appendvecV_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = BLOCKWIDTH;
  int num_blocks = (newveclen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  VecQuant3AppendVecV<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    width,
    fullwidth
  );

}

__global__ void VecQuant3AppendVecV(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
) {

  int packedoffset = BLOCKHEIGHT3 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 8;

  // get value of vec to pack
  int offset = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // find index of smallest entry in lut
  int smallest_idx = 0;
  float prev_val = deq2[0][off];
  for (int val = 1; val < 8; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int loc = (threadIdx.x % 32);
  if (loc == 10) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 30);
    int word_to_add2 = (smallest_idx >> 2);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else if (loc == 21) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 2;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 31);
    int word_to_add2 = (smallest_idx >> 1);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else {
    int row = packedoffset + (threadIdx.x / 32) * 3 + (loc / 11);
    int i = fullwidth * row + width;
    int word_offset = (loc * 3) % 32;
    int word_to_add = (smallest_idx << word_offset);
    atomicAdd(&mat[i], word_to_add);
  }
}


void vecquant2appendvecK_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = BLOCKWIDTH;
  int num_blocks = (newveclen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  VecQuant2AppendVecK<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    width,
    fullwidth
  );

}

__global__ void VecQuant2AppendVecK(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
) {

  int packedoffset = BLOCKHEIGHT2 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[4][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 4;

  // get value of vec to pack
  int offset = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 4; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // find index of smallest entry in lut
  int smallest_idx = 0;
  float prev_val = deq2[0][off];
  for (int val = 1; val < 4; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 16);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 16) * 2;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}


void vecquant2appendvecV_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = BLOCKWIDTH;
  int num_blocks = (newveclen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  VecQuant2AppendVecV<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    width,
    fullwidth
  );

}

__global__ void VecQuant2AppendVecV(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  int width,
  int fullwidth
) {

  int packedoffset = BLOCKHEIGHT2 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[4][BLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 4;

  // get value of vec to pack
  int offset = BLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 4; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // find index of smallest entry in lut
  int smallest_idx = 0;
  float prev_val = deq2[0][off];
  for (int val = 1; val < 4; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 16);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 16) * 2;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}


void vecquant4appendvecKsparse_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outliers_rescaled,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  // TODO: modify this function to only subtract the outlier from the nearest signpost
  VecQuant4AppendVecKSparse<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    outliers_rescaled.data_ptr<float>(),
    width,
    fullwidth
  );
}

__global__ void VecQuant4AppendVecKSparse(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled, // cloned copy of newvec
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT4 * blockIdx.x;

  //Modified dequant block
  __shared__ float deq2[16][PBLOCKWIDTH];
  int off = threadIdx.x;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];
  int row_offset = offset * 16;

  // loop over LUT to find smallest entry
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower[offset];
  float upper_threshold = outlier_threshold_upper[offset];

  unsigned int smallest_idx = 0;

  float rangeval = (upper_threshold - lower_threshold)/2;
  float zeropoint = (upper_threshold + lower_threshold)/2;

  // TODO: could instead use delta from the signpost (instead of delta from the outlier threshold)
  //       -> the downside is this is harder to calibrate for
  outliers_rescaled[offset] = (newvecval - zeropoint) / rangeval;

  // find index of smallest entry in lut
  float prev_val = deq2[0][off];
  for (int val = 1; val < 16; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 8);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 8) * 4;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}

void vecquant4appendvecKsparseParallel_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outliers_rescaled,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0); // numheads
  int height = mat.size(1); // packed headdim
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newvec_numheads = newvec.size(0);
  int newvec_height = newvec.size(1);
  int newvec_seqlen = newvec.size(2);
  assert (newvec_numheads == numheads2);
  assert (newvec_height == headdim);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (newvec_seqlen + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4AppendVecKSparseParallel<<<blocks, threads>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    outliers_rescaled.data_ptr<float>(),
    newvec_seqlen,
    fullwidth,
    headdim
  );
}

__global__ void VecQuant4AppendVecKSparseParallel(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth,
  int headdim
) {
    int headid = blockIdx.z;
    int headoffset = headdim * headid; // in terms of number of logical rows
    int packedheadoffset = (headoffset * BLOCKHEIGHT4) / BLOCKWIDTH;

    // src and dst indices
    int srcrow = headoffset + BLOCKWIDTH * blockIdx.x;
    int dstrow = packedheadoffset + BLOCKHEIGHT4 * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    //Modified dequant block
    __shared__ float deq2[18][BLOCKWIDTH];
    int off = threadIdx.x;

    int lut_row = headoffset + BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int row_offset = lut_row * 16;
    int offset = lut_row;

    for (int val = 0; val < 16; val += 1) {
      int lut_index = row_offset + val;
      deq2[val][off] = lookup_table[lut_index];
    }

    // outlier thresholds
    float lower_threshold = outlier_threshold_lower[lut_row];
    float upper_threshold = outlier_threshold_upper[lut_row];
    deq2[16][off] = (upper_threshold - lower_threshold)/2; // rangeval
    deq2[17][off] = (upper_threshold + lower_threshold)/2; // zeropoint

    if (col < width) {
        int k = 0;

        while (k < BLOCKWIDTH) {

            int srcaddr = (srcrow + k) * width + col;
            int k_div_8 = k / 8;
            int dstaddr = (dstrow + k_div_8) * fullwidth + col;
            float newvecval = newvec[srcaddr];
            outliers_rescaled[srcaddr] = (newvecval - deq2[17][k]) / deq2[16][k];

            // find index of smallest entry in lut
            unsigned int smallest_idx = 0;
            float prev_val = fabsf(newvecval - deq2[0][k]);
            for (int val = 1; val < 16; val += 1) {
              float sub = fabsf(newvecval - deq2[val][k]);
              if (sub < prev_val) {
                prev_val = sub;
                smallest_idx = val;
              }
            }

            // update mat entry using computed idx
            int word_offset = (k % 8) * 4;
            int word_to_add = (smallest_idx << word_offset);

            mat[dstaddr] += word_to_add;
            k += 1;
        }
    }
}

void vecquant4appendvecVsparseParallel_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0); // numheads
  int height = mat.size(1); // packed headdim
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (vseqlen, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  assert(lut_seqlen == fullwidth);

  // newvec shape
  int newvec_numheads = newvec.size(0);
  int newvec_height = newvec.size(1);
  int newvec_seqlen = newvec.size(2);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (newvec_seqlen + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4AppendVecVSparseParallel<<<blocks, threads>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    newvec_seqlen, // input seqlen
    fullwidth, // max seqlen
    newvec_height // headdim
  );
}


__global__ void VecQuant4AppendVecVSparseParallel(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth,
  int headdim
) {

    int headid = blockIdx.z;
    int headoffset = headdim * headid; // in terms of number of logical rows
    int packedheadoffset = (headoffset * BLOCKHEIGHT4) / BLOCKWIDTH;

    // src and dst indices
    int srcrow = headoffset + BLOCKWIDTH * blockIdx.x;
    int dstrow = packedheadoffset + BLOCKHEIGHT4 * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    //Modified dequant block
    __shared__ float deq2[16][BLOCKWIDTH];
    int off = threadIdx.x;
    int col_offset = col * 16;

    if (col < width) {
        for (int val = 0; val < 16; val += 1) {
          int lut_index = col_offset + val;
          deq2[val][off] = lookup_table[lut_index];
        }

        int k = 0;

        float threshold_lower = outlier_threshold_lower[col];
        float threshold_upper = outlier_threshold_upper[col];

        while (k < BLOCKWIDTH) {

            int srcaddr = (srcrow + k) * width + col;
            int k_div_8 = k / 8;
            int dstaddr = (dstrow + k_div_8) * fullwidth + col;
            float newvecval = newvec[srcaddr];

            // find index of smallest entry in lut
            unsigned int smallest_idx = 0;
            float prev_val = fabsf(newvecval - deq2[0][off]);
            if ((newvecval < threshold_lower) || (newvecval > threshold_upper)) {
              smallest_idx = 7; // set to zero-point
            } else {
              for (int val = 1; val < 16; val += 1) {
                float sub = fabsf(newvecval - deq2[val][off]);
                if (sub < prev_val) {
                  prev_val = sub;
                  smallest_idx = val;
                }
              }
            }

            // update mat entry using computed idx
            int word_offset = (k % 8) * 4;
            int word_to_add = (smallest_idx << word_offset);

            mat[dstaddr] += word_to_add;
            k += 1;
        }
    }
}

void vecquant4appendvecVsparse_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  float zeropoint,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  VecQuant4AppendVecVSparse<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower,
    outlier_threshold_upper,
    width,
    fullwidth
  );

}



__global__ void VecQuant4AppendVecVSparse(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT4 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[16][PBLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 16;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 16; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower;
  float upper_threshold = outlier_threshold_upper;

  int num_outliers = 0;

  int smallest_idx = 0;
  if ((newvecval < lower_threshold) || (newvecval > upper_threshold)) {
    smallest_idx = 7; //zero-point
  } else {
    // find index of smallest entry in lut
    float prev_val = deq2[0][off];
    for (int val = 1; val < 16; val += 1) {
      if (deq2[val][off] < prev_val) {
        prev_val = deq2[val][off];
        smallest_idx = val;
      }
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 8);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 8) * 4;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}

void vecquant3appendvecKsparse_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outliers_rescaled,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  // TODO: modify this function to only subtract the outlier from the nearest signpost
  VecQuant3AppendVecKSparse<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    outliers_rescaled.data_ptr<float>(),
    width,
    fullwidth
  );
}

__global__ void VecQuant3AppendVecKSparse(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled, // cloned copy of newvec
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT3 * blockIdx.x;

  //Modified dequant block
  __shared__ float deq2[8][PBLOCKWIDTH];
  int off = threadIdx.x;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];
  int row_offset = offset * 8;

  // loop over LUT to find smallest entry
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower[offset];
  float upper_threshold = outlier_threshold_upper[offset];

  unsigned int smallest_idx = 0;

  float rangeval = (upper_threshold - lower_threshold)/2;
  float zeropoint = (upper_threshold + lower_threshold)/2;

  // TODO: could instead use delta from the signpost (instead of delta from the outlier threshold)
  //       -> the downside is this is harder to calibrate for
  outliers_rescaled[offset] = (newvecval - zeropoint) / rangeval;

  // find index of smallest entry in lut
  float prev_val = deq2[0][off];
  for (int val = 1; val < 8; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int loc = (threadIdx.x % 32);
  if (loc == 10) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;

    unsigned int word_to_add1 = (smallest_idx << 30);
    unsigned int word_to_add2 = (smallest_idx >> 2);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else if (loc == 21) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 2;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 31);
    int word_to_add2 = (smallest_idx >> 1);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else {
    int row = packedoffset + (threadIdx.x / 32) * 3 + (loc / 11);
    int i = fullwidth * row + width;
    int word_offset = (loc * 3) % 32;
    int word_to_add = (smallest_idx << word_offset);
    atomicAdd(&mat[i], word_to_add);
  }
}

void vecquant3appendvecKsparseParallel_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outliers_rescaled,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0); // numheads
  int height = mat.size(1); // packed headdim
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newvec_numheads = newvec.size(0);
  int newvec_height = newvec.size(1);
  int newvec_seqlen = newvec.size(2);
  assert (newvec_numheads == numheads2);
  assert (newvec_height == headdim);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (newvec_seqlen + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3AppendVecKSparseParallel<<<blocks, threads>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    outliers_rescaled.data_ptr<float>(),
    newvec_seqlen,
    fullwidth,
    headdim
  );
}

__global__ void VecQuant3AppendVecKSparseParallel(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth,
  int headdim
) {
    int headid = blockIdx.z;
    int headoffset = headdim * headid; // in terms of number of logical rows
    int packedheadoffset = (headoffset * BLOCKHEIGHT3) / BLOCKWIDTH;

    // src and dst indices
    int srcrow = headoffset + BLOCKWIDTH * blockIdx.x;
    int dstrow = packedheadoffset + BLOCKHEIGHT3 * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    //Modified dequant block
    __shared__ float deq2[10][BLOCKWIDTH];
    int off = threadIdx.x;

    int lut_row = headoffset + BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int row_offset = lut_row * 8;
    int offset = lut_row;

    for (int val = 0; val < 8; val += 1) {
      int lut_index = row_offset + val;
      deq2[val][off] = lookup_table[lut_index];
    }

    // outlier thresholds
    float lower_threshold = outlier_threshold_lower[lut_row];
    float upper_threshold = outlier_threshold_upper[lut_row];
    deq2[8][off] = (upper_threshold - lower_threshold)/2; // rangeval
    deq2[9][off] = (upper_threshold + lower_threshold)/2; // zeropoint

    if (col < width) {
        int k = 0;
        int srcaddr, dstaddr;

        while (k < BLOCKWIDTH) {

            srcaddr = (srcrow + k) * width + col;
            float newvecval = newvec[srcaddr];
            outliers_rescaled[srcaddr] = (newvecval - deq2[9][k]) / deq2[8][k];

            // find index of smallest entry in lut
            unsigned int smallest_idx = 0;
            float prev_val = fabsf(newvecval - deq2[0][k]);
            for (int val = 1; val < 8; val += 1) {
              float sub = fabsf(newvecval - deq2[val][k]);
              if (sub < prev_val) {
                prev_val = sub;
                smallest_idx = val;
              }
            }

            // compute dst addr
            int loc = (k % 32);
            int k_div_32 = (k / 32) * 3;
            int word_offset, word_to_add;
            if (loc == 10) {
                dstaddr = (dstrow + k_div_32) * fullwidth + col;
                word_to_add = (smallest_idx << 30);
                mat[dstaddr] += word_to_add;

                dstaddr = (dstrow + k_div_32 + 1) * fullwidth + col;
                word_to_add = (smallest_idx >> 2);
                mat[dstaddr] += word_to_add;
            } else if (loc == 21) {
                dstaddr = (dstrow + k_div_32 + 1) * fullwidth + col;
                word_to_add = (smallest_idx << 31);
                mat[dstaddr] += word_to_add;

                dstaddr = (dstrow + k_div_32 + 2) * fullwidth + col;
                word_to_add = (smallest_idx >> 1);
                mat[dstaddr] += word_to_add;
            } else {
                dstaddr = (dstrow + k_div_32 + (loc / 11)) * fullwidth + col;

                word_offset = (loc * 3) % 32;
                word_to_add = (smallest_idx << word_offset);
                mat[dstaddr] += word_to_add;
            }

            k += 1;
        }
    }
}


void vecquant3appendvecVsparse_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  float zeropoint,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  VecQuant3AppendVecVSparse<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower,
    outlier_threshold_upper,
    width,
    fullwidth
  );

}



__global__ void VecQuant3AppendVecVSparse(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT3 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[8][PBLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 8;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 8; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower;
  float upper_threshold = outlier_threshold_upper;

  int num_outliers = 0;

  int smallest_idx = 0;
  if ((newvecval < lower_threshold) || (newvecval > upper_threshold)) {
    smallest_idx = 3; //zero-point
  } else {
    // find index of smallest entry in lut
    float prev_val = deq2[0][off];
    for (int val = 1; val < 8; val += 1) {
      if (deq2[val][off] < prev_val) {
        prev_val = deq2[val][off];
        smallest_idx = val;
      }
    }
  }

  // update mat entry using computed idx
  int loc = (threadIdx.x % 32);
  if (loc == 10) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 30);
    int word_to_add2 = (smallest_idx >> 2);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else if (loc == 21) {
    int row1 = packedoffset + (threadIdx.x / 32) * 3 + 1;
    int row2 = packedoffset + (threadIdx.x / 32) * 3 + 2;
    int i1 = fullwidth * row1 + width;
    int i2 = fullwidth * row2 + width;
    int word_to_add1 = (smallest_idx << 31);
    int word_to_add2 = (smallest_idx >> 1);

    atomicAdd(&mat[i1], word_to_add1);
    atomicAdd(&mat[i2], word_to_add2);

  } else {
    int row = packedoffset + (threadIdx.x / 32) * 3 + (loc / 11);
    int i = fullwidth * row + width;
    int word_offset = (loc * 3) % 32;
    int word_to_add = (smallest_idx << word_offset);
    atomicAdd(&mat[i], word_to_add);
  }
}


void vecquant3appendvecVsparseParallel_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0); // numheads
  int height = mat.size(1); // packed headdim
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (vseqlen, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  assert(lut_seqlen == fullwidth);

  // newvec shape
  int newvec_numheads = newvec.size(0);
  int newvec_height = newvec.size(1);
  int newvec_seqlen = newvec.size(2);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (newvec_seqlen + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3AppendVecVSparseParallel<<<blocks, threads>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    newvec_seqlen, // input seqlen
    fullwidth, // max seqlen
    newvec_height // headdim
  );
}


__global__ void VecQuant3AppendVecVSparseParallel(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth,
  int headdim
) {

    int headid = blockIdx.z;
    int headoffset = headdim * headid; // in terms of number of logical rows
    int packedheadoffset = (headoffset * BLOCKHEIGHT3) / BLOCKWIDTH;

    // src and dst indices
    int srcrow = headoffset + BLOCKWIDTH * blockIdx.x;
    int dstrow = packedheadoffset + BLOCKHEIGHT3 * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    //Modified dequant block
    __shared__ float deq2[8][BLOCKWIDTH];
    int off = threadIdx.x;
    int col_offset = col * 8;

    if (col < width) {
        for (int val = 0; val < 8; val += 1) {
          int lut_index = col_offset + val;
          deq2[val][off] = lookup_table[lut_index];
        }

        int k = 0;

        float threshold_lower = outlier_threshold_lower[col];
        float threshold_upper = outlier_threshold_upper[col];

        while (k < BLOCKWIDTH) {

            int srcaddr = (srcrow + k) * width + col;
            float newvecval = newvec[srcaddr];

            // find index of smallest entry in lut
            unsigned int smallest_idx = 0;
            float prev_val = fabsf(newvecval - deq2[0][off]);
            if ((newvecval < threshold_lower) || (newvecval > threshold_upper)) {
              smallest_idx = 3; // set to zero-point
            } else {
              for (int val = 1; val < 8; val += 1) {
                float sub = fabsf(newvecval - deq2[val][k]);
                if (sub < prev_val) {
                  prev_val = sub;
                  smallest_idx = val;
                }
              }
            }

            // compute dst addr
            int loc = (k % 32);
            int k_div_32 = (k / 32) * 3;
            int word_offset, word_to_add, dstaddr;
            if (loc == 10) {
                dstaddr = (dstrow + k_div_32) * fullwidth + col;
                word_to_add = (smallest_idx << 30);
                mat[dstaddr] += word_to_add;

                dstaddr = (dstrow + k_div_32 + 1) * fullwidth + col;
                word_to_add = (smallest_idx >> 2);
                mat[dstaddr] += word_to_add;
            } else if (loc == 21) {
                dstaddr = (dstrow + k_div_32 + 1) * fullwidth + col;
                word_to_add = (smallest_idx << 31);
                mat[dstaddr] += word_to_add;

                dstaddr = (dstrow + k_div_32 + 2) * fullwidth + col;
                word_to_add = (smallest_idx >> 1);
                mat[dstaddr] += word_to_add;
            } else {
                dstaddr = (dstrow + k_div_32 + (loc / 11)) * fullwidth + col;

                word_offset = (loc * 3) % 32;
                word_to_add = (smallest_idx << word_offset);
                mat[dstaddr] += word_to_add;
            }

            k += 1;
        }
    }
}

void vecquant2appendvecKsparse_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outliers_rescaled,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper,
  int kcachelen
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newveclen = newvec.size(0);
  assert (newveclen == headdim * numheads2); // for now only append one token

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  // TODO: modify this function to only subtract the outlier from the nearest signpost
  VecQuant2AppendVecKSparse<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    outliers_rescaled.data_ptr<float>(),
    width,
    fullwidth
  );
}

__global__ void VecQuant2AppendVecKSparse(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled, // cloned copy of newvec
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT2 * blockIdx.x;

  //Modified dequant block
  __shared__ float deq2[4][PBLOCKWIDTH];
  int off = threadIdx.x;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];
  int row_offset = offset * 4;

  // loop over LUT to find smallest entry
  for (int val = 0; val < 4; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower[offset];
  float upper_threshold = outlier_threshold_upper[offset];

  unsigned int smallest_idx = 0;

  float rangeval = (upper_threshold - lower_threshold)/2;
  float zeropoint = (upper_threshold + lower_threshold)/2;

  // TODO: could instead use delta from the signpost (instead of delta from the outlier threshold)
  //       -> the downside is this is harder to calibrate for
  outliers_rescaled[offset] = (newvecval - zeropoint) / rangeval;

  // find index of smallest entry in lut
  float prev_val = deq2[0][off];
  for (int val = 1; val < 4; val += 1) {
    if (deq2[val][off] < prev_val) {
      prev_val = deq2[val][off];
      smallest_idx = val;
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 16);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 16) * 2;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}

void vecquant2appendvecKsparseParallel_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outliers_rescaled,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper
) {

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0); // numheads
  int height = mat.size(1); // packed headdim
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (numheads == numheads2);

  int newvec_numheads = newvec.size(0);
  int newvec_height = newvec.size(1);
  int newvec_seqlen = newvec.size(2);
  assert (newvec_numheads == numheads2);
  assert (newvec_height == headdim);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (newvec_seqlen + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2AppendVecKSparseParallel<<<blocks, threads>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    outliers_rescaled.data_ptr<float>(),
    newvec_seqlen,
    fullwidth,
    headdim
  );
}

__global__ void VecQuant2AppendVecKSparseParallel(
           int* __restrict__ mat,
  const  float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const float* __restrict__ outlier_threshold_lower,
  const float* __restrict__ outlier_threshold_upper,
  float* __restrict__ outliers_rescaled,
  int width,
  int fullwidth,
  int headdim
) {
    int headid = blockIdx.z;
    int headoffset = headdim * headid; // in terms of number of logical rows
    int packedheadoffset = (headoffset * BLOCKHEIGHT2) / BLOCKWIDTH;

    // src and dst indices
    int srcrow = headoffset + BLOCKWIDTH * blockIdx.x;
    int dstrow = packedheadoffset + BLOCKHEIGHT2 * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    //Modified dequant block
    __shared__ float deq2[6][BLOCKWIDTH];
    int off = threadIdx.x;

    int lut_row = headoffset + BLOCKWIDTH * blockIdx.x + threadIdx.x;
    int row_offset = lut_row * 4;
    int offset = lut_row;

    for (int val = 0; val < 4; val += 1) {
      int lut_index = row_offset + val;
      deq2[val][off] = lookup_table[lut_index];
    }

    // outlier thresholds
    float lower_threshold = outlier_threshold_lower[lut_row];
    float upper_threshold = outlier_threshold_upper[lut_row];
    deq2[4][off] = (upper_threshold - lower_threshold)/2; // rangeval
    deq2[5][off] = (upper_threshold + lower_threshold)/2; // zeropoint

    if (col < width) {
        int k = 0;

        while (k < BLOCKWIDTH) {

            int srcaddr = (srcrow + k) * width + col;
            int k_div_16 = k / 16;
            int dstaddr = (dstrow + k_div_16) * fullwidth + col;
            float newvecval = newvec[srcaddr];
            outliers_rescaled[srcaddr] = (newvecval - deq2[5][k]) / deq2[4][k];

            // find index of smallest entry in lut
            unsigned int smallest_idx = 0;
            float prev_val = fabsf(newvecval - deq2[0][k]);
            for (int val = 1; val < 4; val += 1) {
              float sub = fabsf(newvecval - deq2[val][k]);
              if (sub < prev_val) {
                prev_val = sub;
                smallest_idx = val;
              }
            }

            // update mat entry using computed idx
            int word_offset = (k % 16) * 2;
            int word_to_add = (smallest_idx << word_offset);

            mat[dstaddr] += word_to_add;
            k += 1;
        }
    }
}

void vecquant2appendvecVsparseParallel_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  torch::Tensor outlier_threshold_lower,
  torch::Tensor outlier_threshold_upper
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0); // numheads
  int height = mat.size(1); // packed headdim
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (vseqlen, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  assert(lut_seqlen == fullwidth);

  // newvec shape
  int newvec_numheads = newvec.size(0);
  int newvec_height = newvec.size(1);
  int newvec_seqlen = newvec.size(2);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (newvec_seqlen + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2AppendVecVSparseParallel<<<blocks, threads>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower.data_ptr<float>(),
    outlier_threshold_upper.data_ptr<float>(),
    newvec_seqlen, // input seqlen
    fullwidth, // max seqlen
    newvec_height // headdim
  );
}


__global__ void VecQuant2AppendVecVSparseParallel(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  const  float* __restrict__ outlier_threshold_lower,
  const  float* __restrict__ outlier_threshold_upper,
  int width,
  int fullwidth,
  int headdim
) {

    int headid = blockIdx.z;
    int headoffset = headdim * headid; // in terms of number of logical rows
    int packedheadoffset = (headoffset * BLOCKHEIGHT2) / BLOCKWIDTH;

    // src and dst indices
    int srcrow = headoffset + BLOCKWIDTH * blockIdx.x;
    int dstrow = packedheadoffset + BLOCKHEIGHT2 * blockIdx.x;
    int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

    //Modified dequant block
    __shared__ float deq2[4][BLOCKWIDTH];
    int off = threadIdx.x;
    int col_offset = col * 4;

    if (col < width) {
        for (int val = 0; val < 4; val += 1) {
          int lut_index = col_offset + val;
          deq2[val][off] = lookup_table[lut_index];
        }

        int k = 0;

        float threshold_lower = outlier_threshold_lower[col];
        float threshold_upper = outlier_threshold_upper[col];

        while (k < BLOCKWIDTH) {

            int srcaddr = (srcrow + k) * width + col;
            int k_div_16 = k / 16;
            int dstaddr = (dstrow + k_div_16) * fullwidth + col;
            float newvecval = newvec[srcaddr];

            // find index of smallest entry in lut
            unsigned int smallest_idx = 0;
            float prev_val = fabsf(newvecval - deq2[0][off]);
            if ((newvecval < threshold_lower) || (newvecval > threshold_upper)) {
              smallest_idx = 1; // set to zero-point
            } else {
              for (int val = 1; val < 4; val += 1) {
                float sub = fabsf(newvecval - deq2[val][off]);
                if (sub < prev_val) {
                  prev_val = sub;
                  smallest_idx = val;
                }
              }
            }

            // update mat entry using computed idx
            int word_offset = (k % 16) * 2;
            int word_to_add = (smallest_idx << word_offset);

            mat[dstaddr] += word_to_add;
            k += 1;
        }
    }
}

void vecquant2appendvecVsparse_cuda(
  torch::Tensor mat,
  torch::Tensor lookup_table,
  torch::Tensor newvec,
  float zeropoint,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int vcachelen
) {

  // mat - kvcache - (num_heads, head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length

  // lookup table - (num_heads, head_dim, 16)
  int lut_seqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  int newveclen = newvec.size(0);

  int block_size = PBLOCKWIDTH;
  int num_blocks = (newveclen + PBLOCKWIDTH - 1) / PBLOCKWIDTH;

  VecQuant2AppendVecVSparse<<<num_blocks, block_size>>>(
    mat.data_ptr<int>(),
    lookup_table.data_ptr<float>(),
    newvec.data_ptr<float>(),
    outlier_threshold_lower,
    outlier_threshold_upper,
    width,
    fullwidth
  );

}



__global__ void VecQuant2AppendVecVSparse(
           int* __restrict__ mat,
         float* __restrict__ lookup_table,
  const  float* __restrict__ newvec,
  float outlier_threshold_lower,
  float outlier_threshold_upper,
  int width,
  int fullwidth
) {

  int packedoffset = PBLOCKHEIGHT2 * blockIdx.x;

  //Modified dequant block  (TODO change addressing logic)
  __shared__ float deq2[4][PBLOCKWIDTH];
  int off = threadIdx.x;
  int row_offset = width * 4;

  // get value of vec to pack
  int offset = PBLOCKWIDTH * blockIdx.x + threadIdx.x;
  float newvecval = newvec[offset];

  // loop over LUT to find smallest entry
  for (int val = 0; val < 4; val += 1) {
    int lut_index = row_offset + val;
    deq2[val][off] = fabsf(lookup_table[lut_index] - newvecval);
  }

  // check for outliers before packing
  float lower_threshold = outlier_threshold_lower;
  float upper_threshold = outlier_threshold_upper;

  int num_outliers = 0;

  int smallest_idx = 0;
  if ((newvecval < lower_threshold) || (newvecval > upper_threshold)) {
    smallest_idx = 1; //zero-point
  } else {
    // find index of smallest entry in lut
    float prev_val = deq2[0][off];
    for (int val = 1; val < 4; val += 1) {
      if (deq2[val][off] < prev_val) {
        prev_val = deq2[val][off];
        smallest_idx = val;
      }
    }
  }

  // update mat entry using computed idx
  int row = packedoffset + (threadIdx.x / 16);
  int i = fullwidth * row + width;
  int word_offset = (threadIdx.x % 16) * 2;
  int word_to_add = (smallest_idx << word_offset);
  atomicAdd(&mat[i], word_to_add);
}

__global__ void VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size,
    float rope_theta,
    int pos_offset
) {

  int headid = blockIdx.z;
  int headoffset = headdim * headid; // in terms of number of logical rows

  int packedheadoffset = (headoffset * BLOCKHEIGHT4) / BLOCKWIDTH; // in terms of packed words

  int row = packedheadoffset + BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  __shared__ float blockvec2[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[17][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = headoffset + BLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 16;

  float tmp5 = 0;

  for (int val = 0; val < 16; val += 1) { // TODO could be 17 instead
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index]; // illegal mem access here
  }

  int headdim2 = headdim/2;
  deq2[16][off] = powf ( rope_theta , (-2 * __int2float_rd(off % headdim2) / __int2float_rd(headdim)) );

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp;

  // for RoPE
  int pos = col + pos_offset;
  float tmp1, tmp2;
  float c, s;
  int k2;
  float tmp3;
  float theta = 0;
  float sign;

  for (int b = 0; b < batch_size; b++) {

    __syncthreads();
    i = fullwidth * row + col;

    int vec_batch_offset = b * headdim * numheads;
    int headdim2 = headdim/2;
    blockvec[threadIdx.x] = vec[vec_batch_offset + (row / BLOCKHEIGHT4) * BLOCKWIDTH + threadIdx.x];
    blockvec2[threadIdx.x] = vec[vec_batch_offset + (row / BLOCKHEIGHT4) * BLOCKWIDTH + (threadIdx.x+headdim2)%headdim];

    __syncthreads();

    k = 0;
    res = 0;

    // CHECK 1 for sequence length
    if (col < width) {

      while (k < BLOCKWIDTH) {
        tmp = as_unsigned(mat[i]);
        sign = (k<64) ? 1 : -1; // wouldn't work for cyclic

        tmp1 = deq2[(tmp >>  0) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  4) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  8) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  12) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  16) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  20) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  24) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  28) & 0xf][k];
        theta = deq2[16][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        i += fullwidth;
      }

      int mul_batch_offset = b * width * numheads;
      atomicAdd(&mul[mul_batch_offset + headid * width + col], res);
    }
  }
}

__global__ void VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size
) {

  int headid = blockIdx.z;
  int headoffset = headdim * headid; // in terms of number of logical rows

  int packedheadoffset = (headoffset * BLOCKHEIGHT4) / BLOCKWIDTH; // in terms of packed words

  int row = packedheadoffset + BLOCKHEIGHT4 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[16][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_col = col;
  int col_offset = lut_col * 16;

  // CHECK 1 for sequence length
  if (col < width) {
    for (int val = 0; val < 16; val += 1) {
      int lut_index = col_offset + val;
      deq2[val][off] = lookup_table[lut_index];
    }
  }

  const int blockwidth2 = BLOCKWIDTH / 2;
  __shared__ float resvec[blockwidth2][BLOCKWIDTH];

  for (int val = 0; val < blockwidth2; val += 1) {
    resvec[val][off] = 0;
  }

  __shared__ float tmpresvec[2][BLOCKWIDTH];
  for (int val = 0; val < 2; val += 1) {
    tmpresvec[val][off] = 0;
  }

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp;

  for (int b = 0; b < batch_size; b++) {

    __syncthreads();
    // fix this: Note that this is kind of backwards, and it should be * headdim here
    // instead of * width, but we already multiplied row by headdim instead of width
    i = fullwidth * row + col;
    k = 0;
    int vec_batch_offset = b * width * numheads;

    // CHECK 2 for sequence length
    if (col < width) {
      blockvec[threadIdx.x] = vec[vec_batch_offset + width * headid + col];
    }
    __syncthreads();

    int mul_batch_offset = b * headdim * numheads;
    int mul_head_offset = (row / BLOCKHEIGHT4) * BLOCKWIDTH;

    // CHECK 3 for sequence length
    float blockvec_offset = blockvec[off];
    if (col < width) {

      while (k < blockwidth2) {
        tmp = as_unsigned(mat[i]);

        resvec[k][off] = deq2[(tmp >>  0) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  4) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  8) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  12) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  16) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  20) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  24) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  28) & 0xf][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
      }
    }

    __syncthreads();

    // transpose and perform reduction
    if (off < blockwidth2) {
      res = 0;
      k = 0;
      while (k < blockwidth2) {
        res += resvec[off][(k+off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[0][off] = res;
    } else {
      res = 0;
      k = blockwidth2;
      int tmp_off = off - blockwidth2;
      while (k < BLOCKWIDTH) {
        res += resvec[tmp_off][(k+tmp_off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[1][tmp_off] = res;
    }

    // simple version - don't use all threads
    // k = 0;
    // res = 0;
    // if (off < blockwidth2) {
    //   while (k < BLOCKWIDTH) {
    //     res += resvec[off][(k+off)%BLOCKWIDTH];
    //     k += 1;
    //   }
    //   atomicAdd(&mul[mul_batch_offset + mul_head_offset + off], res);
    // }

    __syncthreads();
    k = 0;

    // CHECK 4 for sequence length
    if (col < width) {

      while (k < blockwidth2) {
        tmp = as_unsigned(mat[i]);

        resvec[k][off] = deq2[(tmp >>  0) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  4) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  8) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  12) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  16) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  20) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  24) & 0xf][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  28) & 0xf][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
      }

    }

    __syncthreads();

    // transpose and perform reduction - CHECK THAT THIS WORKS
    // TODO: could divide work among two threads to sum up row
    if (off < blockwidth2) {
      res = 0;
      k = 0;
      int tmp_off = off + blockwidth2;
      while (k < blockwidth2) {
        res += resvec[off][(k+tmp_off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[0][tmp_off] = res;
    } else {
      res = 0;
      k = blockwidth2;
      int tmp_off = off - blockwidth2;
      while (k < BLOCKWIDTH) {
        res += resvec[tmp_off][(k+off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[1][off] = res;
    }

    __syncthreads();

    atomicAdd(&mul[mul_batch_offset + mul_head_offset + off], tmpresvec[0][off] + tmpresvec[1][off]);

    // simple version - don't use all threads
    // k = 0;
    // res = 0;
    // if (off < blockwidth2) {
    //   while (k < BLOCKWIDTH) {
    //     res += resvec[off][(k+off)%BLOCKWIDTH];
    //     k += 1;
    //   }
    //   atomicAdd(&mul[mul_batch_offset + mul_head_offset + off + blockwidth2], res);
    // }

  }
}


// OPTIMIZED FUSED K KERNEL
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  float rope_theta,
  int pos_offset
  // torch::Tensor value
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size, rope_theta, pos_offset
  );

}

// KERNEL FOR V - FUSED
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)
  assert(vec_height == width);

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  assert (lutseqlen == fullwidth);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

}


// OPTIMIZED FUSED K KERNEL
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  torch::Tensor outliers,
  torch::Tensor outlier_indices,
  float rope_theta,
  int pos_offset
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height,
    width,
    fullwidth,
    headdim,
    numheads,
    batch_size,
    rope_theta,
    pos_offset
  );

  int seqlen = outliers.size(0); //full sequence length
  int num_outliers = outliers.size(1); // num outliers per token
  assert(seqlen == fullwidth);

  // TODO: need to make this support batching for sparse kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (kcachelen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_ROPE_BALANCED<<<num_blocks, block_size>>>(
    outliers.data<float>(),
    outlier_indices.data<int>(),
    vec.data<float>(),
    mul.data<float>(),
    kcachelen,
    fullwidth,
    numheads,
    headdim,
    num_outliers,
    rope_theta,
    pos_offset
  );
}

// KERNEL FOR V - FUSED
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen,
  torch::Tensor outliers,
  torch::Tensor outlier_indices
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

  int seqlen = outliers.size(0); //full sequence length
  int num_outliers = outliers.size(1); // num outliers per token
  assert(seqlen == fullwidth);

  // TODO: need to make this support batching for sparse kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (vcachelen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_BALANCED<<<num_blocks, block_size>>>(
    outliers.data<float>(),
    outlier_indices.data<int>(),
    vec.data<float>(),
    mul.data<float>(),
    vcachelen,
    fullwidth,
    numheads,
    headdim,
    num_outliers
  );
}

__global__ void VecQuant3MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size,
    float rope_theta,
    int pos_offset
) {

  int headid = blockIdx.z;
  int headoffset = headdim * headid; // in terms of number of logical rows

  int packedheadoffset = (headoffset * BLOCKHEIGHT3) / BLOCKWIDTH; // in terms of packed words

  int row = packedheadoffset + BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  __shared__ float blockvec2[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[9][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = headoffset + BLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 8;

  float tmp5 = 0;

  for (int val = 0; val < 8; val += 1) { // TODO could be 17 instead
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index]; // illegal mem access here
  }

  int headdim2 = headdim/2;
  deq2[8][off] = powf ( rope_theta , (-2 * __int2float_rd(off % headdim2) / __int2float_rd(headdim)) );

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp, tmp3, tmp4;

  // for RoPE
  int pos = col + pos_offset;
  float tmp1;
  float c, s;
  int k2;
  float theta = 0;
  float sign;

  for (int b = 0; b < batch_size; b++) {

    __syncthreads();
    i = fullwidth * row + col;

    int vec_batch_offset = b * headdim * numheads;
    int headdim2 = headdim/2;
    blockvec[threadIdx.x] = vec[vec_batch_offset + (row / BLOCKHEIGHT3) * BLOCKWIDTH + threadIdx.x];
    blockvec2[threadIdx.x] = vec[vec_batch_offset + (row / BLOCKHEIGHT3) * BLOCKWIDTH + (threadIdx.x+headdim2)%headdim];

    __syncthreads();

    k = 0;
    res = 0;

    // CHECK 1 for sequence length
    if (col < width) {

      int tmpflag = (headid == 0);

      while (k < BLOCKWIDTH) {
        tmp = as_unsigned(mat[i]);
        sign = (k<64) ? 1 : -1; // wouldn't work for cyclic

        tmp1 = deq2[(tmp >>  0) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  3) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  6) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  9) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  12) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  15) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  18) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  21) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  24) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  27) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        i += fullwidth;
        tmp3 = as_unsigned(mat[i]);
        tmp4 = ((tmp >>  30) & 0x3) + ((tmp3 & 0x1) << 2);
        tmp1 = deq2[tmp4 & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp = tmp3;

        tmp1 = deq2[(tmp >>  1) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  4) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  7) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  10) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  13) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  16) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  19) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  22) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  25) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  28) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        i += fullwidth;
        tmp3 = as_unsigned(mat[i]);
        tmp4 = ((tmp >>  31) & 0x1) + ((tmp3 & 0x3) << 1);
        tmp1 = deq2[tmp4 & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp = tmp3;

        tmp1 = deq2[(tmp >>  2) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  5) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  8) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  11) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  14) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  17) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  20) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  23) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  26) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  29) & 0x7][k];
        theta = deq2[8][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        i += fullwidth;

        tmpflag = 0;

      }

      int mul_batch_offset = b * width * numheads;
      atomicAdd(&mul[mul_batch_offset + headid * width + col], res);
    }
  }
}

__global__ void VecQuant3MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size
) {

  int headid = blockIdx.z;
  int headoffset = headdim * headid; // in terms of number of logical rows

  int packedheadoffset = (headoffset * BLOCKHEIGHT3) / BLOCKWIDTH; // in terms of packed words

  int row = packedheadoffset + BLOCKHEIGHT3 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[8][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_col = col;
  int col_offset = lut_col * 8;

  // CHECK 1 for sequence length
  if (col < width) {
    for (int val = 0; val < 8; val += 1) {
      int lut_index = col_offset + val;
      deq2[val][off] = lookup_table[lut_index];
    }
  }

  const int blockwidth2 = BLOCKWIDTH / 2;
  __shared__ float resvec[blockwidth2][BLOCKWIDTH];

  for (int val = 0; val < blockwidth2; val += 1) {
    resvec[val][off] = 0;
  }

  __shared__ float tmpresvec[2][BLOCKWIDTH];
  for (int val = 0; val < 2; val += 1) {
    tmpresvec[val][off] = 0;
  }

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp, tmp2, tmp3;

  for (int b = 0; b < batch_size; b++) {

    __syncthreads();
    // fix this: Note that this is kind of backwards, and it should be * headdim here
    // instead of * width, but we already multiplied row by headdim instead of width
    i = fullwidth * row + col;
    k = 0;
    int vec_batch_offset = b * width * numheads;

    // CHECK 2 for sequence length
    if (col < width) {
      blockvec[threadIdx.x] = vec[vec_batch_offset + width * headid + col];
    }
    __syncthreads();

    int mul_batch_offset = b * headdim * numheads;
    int mul_head_offset = (row / BLOCKHEIGHT3) * BLOCKWIDTH;

    // CHECK 3 for sequence length
    float blockvec_offset = blockvec[off];
    if (col < width) {

      while (k < blockwidth2) {
        tmp = as_unsigned(mat[i]);

        resvec[k][off] = deq2[(tmp >>  0) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  3) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  6) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  9) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  12) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  15) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  18) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  21) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  24) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  27) & 0x7][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
        tmp2 = as_unsigned(mat[i]);
        tmp3 = ((tmp >>  30) & 0x3) + ((tmp2 & 0x1) << 2);

        resvec[k][off] = deq2[tmp3][off] * blockvec_offset;
        k += 1;

        tmp = tmp2;

        resvec[k][off] = deq2[(tmp >>  1) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  4) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  7) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  10) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  13) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  16) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  19) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  22) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  25) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  28) & 0x7][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
        tmp2 = as_unsigned(mat[i]);
        tmp3 = ((tmp >>  31) & 0x1) + ((tmp2 & 0x3) << 1);

        resvec[k][off] = deq2[tmp3][off] * blockvec_offset;
        k += 1;

        tmp = tmp2;

        resvec[k][off] = deq2[(tmp >>  2) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  5) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  8) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  11) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  14) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  17) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  20) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  23) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  26) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  29) & 0x7][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
      }
    }

    __syncthreads();

    // transpose and perform reduction
    if (off < blockwidth2) {
      res = 0;
      k = 0;
      while (k < blockwidth2) {
        res += resvec[off][(k+off)%BLOCKWIDTH];
        k += 1;
      }
      // printf("res1: %f \n", res);
      tmpresvec[0][off] = res;
    } else {
      res = 0;
      k = blockwidth2;
      int tmp_off = off - blockwidth2;
      while (k < BLOCKWIDTH) {
        res += resvec[tmp_off][(k+tmp_off)%BLOCKWIDTH];
        k += 1;
      }
      // printf("res2: %f \n", res);
      tmpresvec[1][tmp_off] = res;
    }

    __syncthreads();
    k = 0;

    // CHECK 4 for sequence length
    if (col < width) {

      while (k < blockwidth2) {
        tmp = as_unsigned(mat[i]);

        resvec[k][off] = deq2[(tmp >>  0) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  3) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  6) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  9) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  12) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  15) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  18) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  21) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  24) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  27) & 0x7][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
        tmp2 = as_unsigned(mat[i]);
        tmp3 = ((tmp >>  30) & 0x3) + ((tmp2 & 0x1) << 2);

        resvec[k][off] = deq2[tmp3][off] * blockvec_offset;
        k += 1;

        tmp = tmp2;

        resvec[k][off] = deq2[(tmp >>  1) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  4) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  7) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  10) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  13) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  16) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  19) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  22) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  25) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  28) & 0x7][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
        tmp2 = as_unsigned(mat[i]);
        tmp3 = ((tmp >>  31) & 0x1) + ((tmp2 & 0x3) << 1);

        resvec[k][off] = deq2[tmp3][off] * blockvec_offset;
        k += 1;

        tmp = tmp2;

        resvec[k][off] = deq2[(tmp >>  2) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  5) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  8) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  11) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  14) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  17) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  20) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  23) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  26) & 0x7][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  29) & 0x7][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
      }

    }

    __syncthreads();

    // transpose and perform reduction - CHECK THAT THIS WORKS
    // TODO: could divide work among two threads to sum up row
    if (off < blockwidth2) {
      res = 0;
      k = 0;
      int tmp_off = off + blockwidth2;
      while (k < blockwidth2) {
        res += resvec[off][(k+tmp_off)%BLOCKWIDTH];
        k += 1;
      }
      // printf("res3: %f \n", res);
      tmpresvec[0][tmp_off] = res;
    } else {
      res = 0;
      k = blockwidth2;
      int tmp_off = off - blockwidth2;
      while (k < BLOCKWIDTH) {
        res += resvec[tmp_off][(k+off)%BLOCKWIDTH];
        k += 1;
      }
      // printf("res4: %f \n", res);
      tmpresvec[1][off] = res;
    }

    __syncthreads();

    atomicAdd(&mul[mul_batch_offset + mul_head_offset + off], tmpresvec[0][off] + tmpresvec[1][off]);

  }
}


// OPTIMIZED FUSED K KERNEL
void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  float rope_theta,
  int pos_offset
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size, rope_theta, pos_offset
  );

}

// KERNEL FOR V - FUSED
void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)
  assert(vec_height == width);

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  assert (lutseqlen == fullwidth);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

}


// OPTIMIZED FUSED K KERNEL
void vecquant3matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  torch::Tensor outliers,
  torch::Tensor outlier_indices,
  float rope_theta,
  int pos_offset
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height,
    width,
    fullwidth,
    headdim,
    numheads,
    batch_size,
    rope_theta,
    pos_offset
  );

  int seqlen = outliers.size(0); //full sequence length
  int num_outliers = outliers.size(1); // num outliers per token
  assert(seqlen == fullwidth);

  // TODO: need to make this support batching for sparse kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (kcachelen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_ROPE_BALANCED<<<num_blocks, block_size>>>(
    outliers.data<float>(),
    outlier_indices.data<int>(),
    vec.data<float>(),
    mul.data<float>(),
    kcachelen,
    fullwidth,
    numheads,
    headdim,
    num_outliers,
    rope_theta,
    pos_offset
  );
}

// KERNEL FOR V - FUSED
void vecquant3matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen,
  torch::Tensor outliers,
  torch::Tensor outlier_indices
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT3 - 1) / BLOCKHEIGHT3,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant3MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

  int seqlen = outliers.size(0); //full sequence length
  int num_outliers = outliers.size(1); // num outliers per token
  assert(seqlen == fullwidth);

  // TODO: need to make this support batching for sparse kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (vcachelen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_BALANCED<<<num_blocks, block_size>>>(
    outliers.data<float>(),
    outlier_indices.data<int>(),
    vec.data<float>(),
    mul.data<float>(),
    vcachelen,
    fullwidth,
    numheads,
    headdim,
    num_outliers
  );
}

__global__ void VecQuant2MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size,
    float rope_theta,
    int pos_offset
) {

  int headid = blockIdx.z;
  int headoffset = headdim * headid; // in terms of number of logical rows

  int packedheadoffset = (headoffset * BLOCKHEIGHT2) / BLOCKWIDTH; // in terms of packed words

  int row = packedheadoffset + BLOCKHEIGHT2 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];
  __shared__ float blockvec2[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[5][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_row = headoffset + BLOCKWIDTH * blockIdx.x + threadIdx.x;
  int row_offset = lut_row * 4;

  float tmp5 = 0;

  for (int val = 0; val < 4; val += 1) { // TODO could be 17 instead
    int lut_index = row_offset + val;
    deq2[val][off] = lookup_table[lut_index]; // illegal mem access here
  }

  int headdim2 = headdim/2;
  deq2[4][off] = powf ( rope_theta , (-2 * __int2float_rd(off % headdim2) / __int2float_rd(headdim)) );

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp;

  // for RoPE
  int pos = col + pos_offset;
  float tmp1, tmp2;
  float c, s;
  int k2;
  float tmp3;
  float theta = 0;
  float sign;

  for (int b = 0; b < batch_size; b++) {

    __syncthreads();
    i = fullwidth * row + col;

    int vec_batch_offset = b * headdim * numheads;
    int headdim2 = headdim/2;
    blockvec[threadIdx.x] = vec[vec_batch_offset + (row / BLOCKHEIGHT2) * BLOCKWIDTH + threadIdx.x];
    blockvec2[threadIdx.x] = vec[vec_batch_offset + (row / BLOCKHEIGHT2) * BLOCKWIDTH + (threadIdx.x+headdim2)%headdim];

    __syncthreads();

    k = 0;
    res = 0;

    // CHECK 1 for sequence length
    if (col < width) {

      while (k < BLOCKWIDTH) {
        tmp = as_unsigned(mat[i]);
        sign = (k<64) ? 1 : -1; // wouldn't work for cyclic

        tmp1 = deq2[(tmp >>  0) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  2) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  4) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  6) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  8) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  10) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  12) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  14) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  16) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  18) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  20) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  22) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  24) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  26) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  28) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        tmp1 = deq2[(tmp >>  30) & 0x3][k];
        theta = deq2[4][k];

        c = cosf(theta * pos);
        s = sinf(theta * pos);

        res += tmp1 * c * blockvec[k];
        res += sign * tmp1 * s * blockvec2[k];
        k += 1;

        i += fullwidth;
      }

      int mul_batch_offset = b * width * numheads;
      atomicAdd(&mul[mul_batch_offset + headid * width + col], res);
    }
  }
}

__global__ void VecQuant2MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt(
    const  float* __restrict__ vec,
    const    int* __restrict__ mat,
           float* __restrict__ mul,
    const  float* __restrict__ lookup_table,
    int height,
    int width,
    int fullwidth,
    int headdim,
    int numheads,
    int batch_size
) {

  int headid = blockIdx.z;
  int headoffset = headdim * headid; // in terms of number of logical rows

  int packedheadoffset = (headoffset * BLOCKHEIGHT2) / BLOCKWIDTH; // in terms of packed words

  int row = packedheadoffset + BLOCKHEIGHT2 * blockIdx.x;
  int col = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ float blockvec[BLOCKWIDTH];

  //Modified dequant block
  __shared__ float deq2[4][BLOCKWIDTH];
  int off = threadIdx.x;

  int lut_col = col;
  int col_offset = lut_col * 4;

  // CHECK 1 for sequence length
  if (col < width) {
    for (int val = 0; val < 4; val += 1) {
      int lut_index = col_offset + val;
      deq2[val][off] = lookup_table[lut_index];
    }
  }

  const int blockwidth2 = BLOCKWIDTH / 2;
  __shared__ float resvec[blockwidth2][BLOCKWIDTH];

  for (int val = 0; val < blockwidth2; val += 1) {
    resvec[val][off] = 0;
  }

  __shared__ float tmpresvec[2][BLOCKWIDTH];
  for (int val = 0; val < 2; val += 1) {
    tmpresvec[val][off] = 0;
  }

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp;

  for (int b = 0; b < batch_size; b++) {

    __syncthreads();
    // fix this: Note that this is kind of backwards, and it should be * headdim here
    // instead of * width, but we already multiplied row by headdim instead of width
    i = fullwidth * row + col;
    k = 0;
    int vec_batch_offset = b * width * numheads;

    // CHECK 2 for sequence length
    if (col < width) {
      blockvec[threadIdx.x] = vec[vec_batch_offset + width * headid + col];
    }
    __syncthreads();

    int mul_batch_offset = b * headdim * numheads;
    int mul_head_offset = (row / BLOCKHEIGHT2) * BLOCKWIDTH;

    // CHECK 3 for sequence length
    float blockvec_offset = blockvec[off];
    if (col < width) {

      while (k < blockwidth2) {
        tmp = as_unsigned(mat[i]);

        resvec[k][off] = deq2[(tmp >>  0) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  2) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  4) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  6) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  8) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  10) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  12) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  14) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  16) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  18) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  20) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  22) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  24) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  26) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  28) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  30) & 0x3][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
      }
    }

    __syncthreads();

    // transpose and perform reduction
    if (off < blockwidth2) {
      res = 0;
      k = 0;
      while (k < blockwidth2) {
        res += resvec[off][(k+off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[0][off] = res;
    } else {
      res = 0;
      k = blockwidth2;
      int tmp_off = off - blockwidth2;
      while (k < BLOCKWIDTH) {
        res += resvec[tmp_off][(k+tmp_off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[1][tmp_off] = res;
    }

    __syncthreads();
    k = 0;

    // CHECK 4 for sequence length
    if (col < width) {

      while (k < blockwidth2) {
        tmp = as_unsigned(mat[i]);

        resvec[k][off] = deq2[(tmp >>  0) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  2) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  4) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  6) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  8) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  10) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  12) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  14) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  16) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  18) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  20) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  22) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  24) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  26) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  28) & 0x3][off] * blockvec_offset;
        k += 1;

        resvec[k][off] = deq2[(tmp >>  30) & 0x3][off] * blockvec_offset;
        k += 1;

        i += fullwidth;
      }

    }

    __syncthreads();

    // transpose and perform reduction - CHECK THAT THIS WORKS
    // TODO: could divide work among two threads to sum up row
    if (off < blockwidth2) {
      res = 0;
      k = 0;
      int tmp_off = off + blockwidth2;
      while (k < blockwidth2) {
        res += resvec[off][(k+tmp_off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[0][tmp_off] = res;
    } else {
      res = 0;
      k = blockwidth2;
      int tmp_off = off - blockwidth2;
      while (k < BLOCKWIDTH) {
        res += resvec[tmp_off][(k+off)%BLOCKWIDTH];
        k += 1;
      }
      tmpresvec[1][off] = res;
    }

    __syncthreads();

    atomicAdd(&mul[mul_batch_offset + mul_head_offset + off], tmpresvec[0][off] + tmpresvec[1][off]);

  }
}


// OPTIMIZED FUSED K KERNEL
void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  float rope_theta,
  int pos_offset
  // torch::Tensor value
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size, rope_theta, pos_offset
  );

}

// KERNEL FOR V - FUSED
void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)
  assert(vec_height == width);

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);
  assert (lutseqlen == fullwidth);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

}


// OPTIMIZED FUSED K KERNEL
void vecquant2matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  torch::Tensor outliers,
  torch::Tensor outlier_indices,
  float rope_theta,
  int pos_offset
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height,
    width,
    fullwidth,
    headdim,
    numheads,
    batch_size,
    rope_theta,
    pos_offset
  );

  int seqlen = outliers.size(0); //full sequence length
  int num_outliers = outliers.size(1); // num outliers per token
  assert(seqlen == fullwidth);

  // TODO: need to make this support batching for sparse kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (kcachelen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_ROPE_BALANCED<<<num_blocks, block_size>>>(
    outliers.data<float>(),
    outlier_indices.data<int>(),
    vec.data<float>(),
    mul.data<float>(),
    kcachelen,
    fullwidth,
    numheads,
    headdim,
    num_outliers,
    rope_theta,
    pos_offset
  );
}

// KERNEL FOR V - FUSED
void vecquant2matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen,
  torch::Tensor outliers,
  torch::Tensor outlier_indices
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT2 - 1) / BLOCKHEIGHT2,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant2MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

  int seqlen = outliers.size(0); //full sequence length
  int num_outliers = outliers.size(1); // num outliers per token
  assert(seqlen == fullwidth);

  // TODO: need to make this support batching for sparse kernel
  int block_size = BLOCKWIDTH;
  int num_blocks = (vcachelen + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_BALANCED<<<num_blocks, block_size>>>(
    outliers.data<float>(),
    outlier_indices.data<int>(),
    vec.data<float>(),
    mul.data<float>(),
    vcachelen,
    fullwidth,
    numheads,
    headdim,
    num_outliers
  );
}

// OPTIMIZED FUSED K KERNEL
void vecquant4matmul_nuq_perchannel_transposed_rope_mha_batched_fused_opt2_orig_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int kcachelen,
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor startrows,
  torch::Tensor spmat,
  int num_rows,
  int num_threads,
  int nnz,
  float rope_theta,
  int pos_offset
) {

  // mul - out - (num_heads, qseqlen, kseqlen)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);

  // vec - in - (num_heads, qseqlen, head_dim)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2);
  assert (vbatch_size == batch_size);

  // mat - kvcache - (num_heads, head_dim, kseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim
  int width = kcachelen; // sequence length
  int fullwidth = mat.size(2); // max sequence length
  assert(width == mul_height);

  // lookup table - (num_heads, head_dim, 16)
  int numheads2 = lookup_table.size(0);
  int headdim = lookup_table.size(1);
  int lutlen = lookup_table.size(2);
  assert (headdim == vec_height);
  assert (numheads == numheads2);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);


  VecQuant4MatMulKernelNUQPerChannelTransposedRopeMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height,
    width,
    fullwidth,
    headdim,
    numheads,
    batch_size,
    rope_theta,
    pos_offset
  );

  // check if no nonzeros yet
  if (num_threads > 0) {

    // TODO: need to make this support batching for sparse kernel
    int block_size = BLOCKWIDTH;
    int num_blocks = (num_threads + BLOCKWIDTH - 1) / BLOCKWIDTH;

    SPMV_ATOMIC_CSR_ROPE_BALANCED<<<num_blocks, block_size>>>(
      rows.data<int>(),
      cols.data<int>(),
      startrows.data<int>(),
      spmat.data<float>(),
      vec.data<float>(),
      mul.data<float>(),
      num_rows,
      numheads,
      kcachelen,
      headdim,
      num_threads,
      nnz,
      rope_theta,
      pos_offset
    );

  }
}

// KERNEL FOR V - FUSED
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_orig_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor lookup_table,
  int vcachelen,
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor startcols,
  torch::Tensor spmat,
  int num_cols,
  int num_threads,
  int nnz
) {

  // mul - out - (score_seqlen, num_head, head_dim)
  int batch_size = mul.size(0);
  int mul_num_heads = mul.size(1);
  int mul_height = mul.size(2);
  int headdim = mul_height;

  // vec - in - (score_seqlen, num_head, vseqlen)
  int vbatch_size = vec.size(0);
  int num_vec_heads = vec.size(1);
  int vec_height = vec.size(2); // v seqlen

  // mat - kvcache - (num_heads, packed_head_dim, vseqlen)
  int numheads = mat.size(0);
  int height = mat.size(1); // headdim (packed)
  int width = vcachelen; // v sequence length
  int fullwidth = mat.size(2); // v sequence length (full max seqlen)

  // lookup table - (vseqlen, 16)
  int lutseqlen = lookup_table.size(0);
  int lutlen = lookup_table.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (numheads)
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant4MatMulKernelNUQPerChannelTransposedMHABatchedFusedOpt<<<blocks, threads>>>(
    vec.data_ptr<float>(),
    mat.data_ptr<int>(),
    mul.data_ptr<float>(),
    lookup_table.data_ptr<float>(),
    height, width, fullwidth, headdim, numheads, batch_size
  );

  // balanced
  int block_size = BLOCKWIDTH;
  int num_blocks = (num_threads + BLOCKWIDTH - 1) / BLOCKWIDTH;

  SPMV_ATOMIC_CSC_BALANCED<<<num_blocks, block_size>>>(
    rows.data<int>(),
    cols.data<int>(),
    startcols.data<int>(),
    spmat.data<float>(),
    vec.data<float>(),
    mul.data<float>(),
    num_cols,
    numheads,
    vcachelen,
    headdim,
    num_threads,
    nnz
  );
}
