/**
 * wmma_gemm.cu
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * Matrix multiply C = A * B on the GPU using Tensor Cores (WMMA). A and B are FP16,
 * C is FP32. Matrix sizes must be multiples of 16. Used for the capstone benchmark
 * on Ampere GPUs (e.g. RTX A2000).
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
#include <cassert>

#define WMMA_TILE_M 16
#define WMMA_TILE_N 16
#define WMMA_TILE_K 16

using namespace nvcuda;

// One warp computes one 16x16 block of C using WMMA tiles
__global__ void wmma_gemm_kernel(
  const half* __restrict__ A,
  const half* __restrict__ B,
  float* __restrict__ C,
  int M, int N, int K
) {
  int row = blockIdx.y * WMMA_TILE_M;
  int col = blockIdx.x * WMMA_TILE_N;
  if (row >= M || col >= N) return;

  wmma::fragment<wmma::matrix_a, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::row_major> a_frag;
  wmma::fragment<wmma::matrix_b, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, half, wmma::col_major> b_frag;
  wmma::fragment<wmma::accumulator, WMMA_TILE_M, WMMA_TILE_N, WMMA_TILE_K, float> acc_frag;
  wmma::fill_fragment(acc_frag, 0.0f);

  // Loop over K dimension in 16-wide chunks; each step does one 16x16x16 multiply-accumulate
  for (int k = 0; k < K; k += WMMA_TILE_K) {
    int kOff = k;
    if (row + WMMA_TILE_M <= M && k + WMMA_TILE_K <= K)
      wmma::load_matrix_sync(a_frag, A + row * K + kOff, K);
    else
      wmma::fill_fragment(a_frag, __float2half(0.0f));
    if (col + WMMA_TILE_N <= N && k + WMMA_TILE_K <= K)
      wmma::load_matrix_sync(b_frag, B + kOff * N + col, N);
    else
      wmma::fill_fragment(b_frag, __float2half(0.0f));
    wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
  }

  if (row + WMMA_TILE_M <= M && col + WMMA_TILE_N <= N)
    wmma::store_matrix_sync(C + row * N + col, acc_frag, N, wmma::mem_row_major);
}

static void launch_wmma_gemm(
  const half* d_A,
  const half* d_B,
  float* d_C,
  int M, int N, int K
) {
  assert(M % WMMA_TILE_M == 0 && N % WMMA_TILE_N == 0 && K % WMMA_TILE_K == 0);
  dim3 block(32, 1, 1);  // One warp per block (one 16x16 output tile per block)
  dim3 grid(
    (N + WMMA_TILE_N - 1) / WMMA_TILE_N,
    (M + WMMA_TILE_M - 1) / WMMA_TILE_M
  );
  wmma_gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
}

extern "C" {

void wmma_gemm_fp16(
  const void* d_A,
  const void* d_B,
  void* d_C,
  int M, int N, int K
) {
  launch_wmma_gemm(
    reinterpret_cast<const half*>(d_A),
    reinterpret_cast<const half*>(d_B),
    reinterpret_cast<float*>(d_C),
    M, N, K
  );
}

float wmma_gemm_fp16_timed(
  const void* d_A,
  const void* d_B,
  void* d_C,
  int M, int N, int K
) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);
  launch_wmma_gemm(
    reinterpret_cast<const half*>(d_A),
    reinterpret_cast<const half*>(d_B),
    reinterpret_cast<float*>(d_C),
    M, N, K
  );
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float ms = 0.0f;
  cudaEventElapsedTime(&ms, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

} // extern "C"
