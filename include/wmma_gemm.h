#ifndef WMMA_GEMM_H
#define WMMA_GEMM_H

/**
 * wmma_gemm.h
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * Declares FP16 matrix multiply (C = A * B) using Tensor Cores (WMMA) on the GPU.
 * Used for the capstone benchmark; matrix sizes must be multiples of 16.
 */

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * FP16 matrix multiply using WMMA (Tensor Cores) on GPU.
 * C = A * B, where A is (M x K), B is (K x N), C is (M x N).
 * M, N, K must be multiples of 16 (WMMA tile size).
 * A, B are in half precision (FP16); C can be FP32 or FP16.
 */
void wmma_gemm_fp16(
  const void* d_A,   /* row-major, M x K, half */
  const void* d_B,   /* row-major, K x N, half */
  void* d_C,         /* row-major, M x N, float (accumulator) or half */
  int M, int N, int K
);

/**
 * Same as above but returns wall-clock time (ms) for the kernel.
 */
float wmma_gemm_fp16_timed(
  const void* d_A,
  const void* d_B,
  void* d_C,
  int M, int N, int K
);

#ifdef __cplusplus
}
#endif

#endif /* WMMA_GEMM_H */
