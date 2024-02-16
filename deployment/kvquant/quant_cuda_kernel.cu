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
const int BLOCKHEIGHT3 =  12;
const int BLOCKHEIGHT4 =  16;

const int PBLOCKWIDTH  = 32;
const int PBLOCKHEIGHT4 =  4;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
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
    int nnz
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
    int batch_size
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

__global__ void VecQuant4AppendVecKSparse(
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

__global__ void VecQuant4AppendVecKSparse2(
  const  float* __restrict__ newvec,
  float* __restrict__ zeropoint,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count_per_block,
  int*  __restrict__ dst_indices,
  float*  __restrict__ dst_values,
  int num_blocks
);

__global__ void VecQuant4AppendVecVSparse(
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

__global__ void VecQuant4AppendVecVSparse2(
  const  float* __restrict__ newvec,
  float zeropoint,
  int*  __restrict__ mask,
  int*  __restrict__ outlier_count_per_block,
  int*  __restrict__ dst_indices,
  float*  __restrict__ dst_values,
  int num_blocks
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
    int nnz
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
              theta = powf ( 10000 , (-2 * __int2float_rd(channel_head_off % (headdim/2)) / headdim) );
              sign = (channel_head_off < (headdim/2)) ? 1 : -1;
              c = cosf(theta * row);
              s = sinf(theta * row);

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


std::vector<torch::Tensor> vecquant4appendvecKsparse_cuda(
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

  VecQuant4AppendVecKSparse<<<num_blocks, block_size>>>(
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

  VecQuant4AppendVecKSparse2<<<num_blocks, block_size>>>(
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

__global__ void VecQuant4AppendVecKSparse2(
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

__global__ void VecQuant4AppendVecKSparse(
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

std::vector<torch::Tensor> vecquant4appendvecVsparse_cuda(
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

  VecQuant4AppendVecVSparse<<<num_blocks, block_size>>>(
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

  VecQuant4AppendVecVSparse2<<<num_blocks, block_size>>>(
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

__global__ void VecQuant4AppendVecVSparse2(
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

__global__ void VecQuant4AppendVecVSparse(
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
    int batch_size
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
  deq2[16][off] = powf ( 10000 , (-2 * __int2float_rd(off % headdim2) / __int2float_rd(headdim)) );

  __syncthreads();

  int i;
  int k = 0;
  float res = 0;

  unsigned int tmp;

  // for RoPE
  int pos = col;
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
  int kcachelen
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
    height, width, fullwidth, headdim, numheads, batch_size
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
  torch::Tensor rows,
  torch::Tensor cols,
  torch::Tensor startrows,
  torch::Tensor spmat,
  int num_rows,
  int num_threads,
  int nnz
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
    batch_size
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
      nnz
    );

  }
}

// KERNEL FOR V - FUSED
void vecquant4matmul_nuq_perchannel_transposed_mha_batched_fused_opt2_cuda(
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
