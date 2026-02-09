/**
 * particle_filter.cu
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * CUDA particle filter: each particle has (x, y, vx, vy). We spread particles at init,
 * then each frame we predict (move by velocity + noise), weight from a 2D map, normalize,
 * and resample so likely particles are duplicated. The pipeline uses the mean state as
 * the track position.
 */

#include <cuda_runtime.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include "particle_filter_cuda.h"

#define PF_STATE_STRIDE 4  /* x, y, vx, vy */

// Spread particles in a rectangle with small random velocity
__global__ void init_kernel(
  float* state,
  float* weight,
  unsigned int n,
  float cx, float cy,
  float width, float height,
  float vel_std,
  unsigned long long seed
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long s = seed + i * 1103515245ULL + 12345ULL;
  float u1 = ((s >> 16) & 0x7FFF) / 32768.0f;
  float u2 = ((s >> 1)  & 0x7FFF) / 32768.0f;
  float u3 = ((s * 1103515245ULL + 12345ULL) >> 16 & 0x7FFF) / 32768.0f;
  float u4 = ((s * 1103515245ULL + 12345ULL) >> 1  & 0x7FFF) / 32768.0f;
  state[i * PF_STATE_STRIDE + 0] = cx + (u1 - 0.5f) * width;
  state[i * PF_STATE_STRIDE + 1] = cy + (u2 - 0.5f) * height;
  state[i * PF_STATE_STRIDE + 2] = (u3 - 0.5f) * 2.0f * vel_std;
  state[i * PF_STATE_STRIDE + 3] = (u4 - 0.5f) * 2.0f * vel_std;
  weight[i] = 1.0f / (float)n;
}

// Move each particle: position += velocity*dt + noise, velocity += noise
__global__ void predict_kernel(
  float* state,
  unsigned int n,
  float dt,
  float pos_std,
  float vel_std,
  unsigned long long seed
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  unsigned long long s = (seed + i * 31ULL) * 1103515245ULL + 12345ULL;
  float nx = ((s >> 16) & 0x7FFF) / 32768.0f;
  float ny = ((s >> 1) & 0x7FFF) / 32768.0f;
  float nvx = ((s * 1103515245ULL + 12345ULL) >> 16 & 0x7FFF) / 32768.0f;
  float nvy = ((s * 1103515245ULL + 12345ULL) >> 1  & 0x7FFF) / 32768.0f;
  float x  = state[i * PF_STATE_STRIDE + 0];
  float y  = state[i * PF_STATE_STRIDE + 1];
  float vx = state[i * PF_STATE_STRIDE + 2];
  float vy = state[i * PF_STATE_STRIDE + 3];
  state[i * PF_STATE_STRIDE + 0] = x + vx * dt + (nx - 0.5f) * 2.0f * pos_std;
  state[i * PF_STATE_STRIDE + 1] = y + vy * dt + (ny - 0.5f) * 2.0f * pos_std;
  state[i * PF_STATE_STRIDE + 2] = vx + (nvx - 0.5f) * 2.0f * vel_std;
  state[i * PF_STATE_STRIDE + 3] = vy + (nvy - 0.5f) * 2.0f * vel_std;
}

// Set weight from the observation map at particle (x, y); low map value = small weight
__global__ void weight_kernel(
  const float* state,
  float* weight,
  const float* map,
  unsigned int n,
  int map_w, int map_h
) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float x = state[i * PF_STATE_STRIDE + 0];
  float y = state[i * PF_STATE_STRIDE + 1];
  int ix = (int)(x + 0.5f);
  int iy = (int)(y + 0.5f);
  if (ix < 0) ix = 0; if (ix >= map_w) ix = map_w - 1;
  if (iy < 0) iy = 0; if (iy >= map_h) iy = map_h - 1;
  float w = map[iy * map_w + ix];
  weight[i] = (w > 1e-6f) ? w : 1e-6f;
}

// Multiply all weights by inv_sum so they add to 1
__global__ void normalize_kernel(float* weight, float inv_sum, unsigned int n) {
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) weight[i] *= inv_sum;
}

// Sum weights in each block for later normalization
__global__ void sum_weights_kernel(const float* weight, float* block_sum, unsigned int n) {
  __shared__ float sh[256];
  unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? weight[i] : 0.0f;
  sh[threadIdx.x] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) sh[threadIdx.x] += sh[threadIdx.x + s];
    __syncthreads();
  }
  if (threadIdx.x == 0) block_sum[blockIdx.x] = sh[0];
}

// Add all block sums into one total (one block)
__global__ void sum_blocks_kernel(const float* block_sum, float* total, int num_blocks) {
  __shared__ float sh[256];
  int i = threadIdx.x;
  sh[i] = (i < num_blocks) ? block_sum[i] : 0.0f;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (i < s) sh[i] += sh[i + s];
    __syncthreads();
  }
  if (i == 0) *total = sh[0];
}

// Exclusive prefix sum in shared memory (used to build CDF for resample)
__device__ void block_scan_exclusive(float* sh, int n) {
  __syncthreads();
  if (threadIdx.x == 0) {
    float sum = 0.0f;
    for (int i = 0; i < n; i++) {
      float v = sh[i];
      sh[i] = sum;
      sum += v;
    }
    sh[n] = sum;
  }
  __syncthreads();
}

// Build CDF from weights: each block scans 256 weights, then we combine block totals
__global__ void scan_weights_kernel(
  const float* weight,
  float* scan_out,
  float* block_total,
  unsigned int n
) {
  __shared__ float sh[260];
  unsigned int i = threadIdx.x;
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  sh[i] = (g < n) ? weight[g] : 0.0f;
  block_scan_exclusive(sh, (int)blockDim.x);
  if (g < n) scan_out[g] = sh[threadIdx.x];
  if (threadIdx.x == 0) block_total[blockIdx.x] = sh[256];
}

// Prefix sum of block totals so we can add them to each block's CDF
__global__ void scan_block_totals_kernel(
  const float* block_total,
  float* block_prefix_sum,
  int num_blocks
) {
  __shared__ float sh[256];
  int tid = threadIdx.x;
  sh[tid] = (tid < num_blocks) ? block_total[tid] : 0.0f;
  __syncthreads();
  if (tid == 0) {
    for (int d = 1; d < num_blocks; d++) sh[d] += sh[d - 1];
    block_prefix_sum[0] = 0.0f;
    for (int d = 1; d < num_blocks; d++) block_prefix_sum[d] = sh[d - 1];
  }
}

// Add block prefix to each block so CDF is global
__global__ void add_block_prefix_kernel(
  float* scan_out,
  const float* block_prefix_sum,
  unsigned int n
) {
  unsigned int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g < n && blockIdx.x > 0)
    scan_out[g] += block_prefix_sum[blockIdx.x];
}

// Resample: each new particle is a copy of an old one chosen by the CDF (likely particles copied more)
__global__ void resample_kernel(
  const float* state_in,
  const float* weight_in,
  const float* cdf,
  float* state_out,
  unsigned int n,
  unsigned long long seed
) {
  unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= n) return;
  float u = (j + 0.5f) / (float)n;
  // Find which old particle this new slot copies (binary search in CDF)
  int lo = 0, hi = (int)n;
  while (lo + 1 < hi) {
    int mid = (lo + hi) / 2;
    if (cdf[mid] <= u) lo = mid; else hi = mid;
  }
  int idx = (cdf[hi] <= u) ? hi : lo;
  if (idx >= (int)n) idx = (int)n - 1;
  state_out[j * PF_STATE_STRIDE + 0] = state_in[idx * PF_STATE_STRIDE + 0];
  state_out[j * PF_STATE_STRIDE + 1] = state_in[idx * PF_STATE_STRIDE + 1];
  state_out[j * PF_STATE_STRIDE + 2] = state_in[idx * PF_STATE_STRIDE + 2];
  state_out[j * PF_STATE_STRIDE + 3] = state_in[idx * PF_STATE_STRIDE + 3];
}

// Weighted average of (x, y, vx, vy) over all particles -> one mean state
__global__ void mean_state_kernel(
  const float* state,
  const float* weight,
  float* mean,
  unsigned int n
) {
  __shared__ float sh[4][256];
  int tid = threadIdx.x;
  float x = 0, y = 0, vx = 0, vy = 0;
  for (unsigned int i = tid; i < n; i += blockDim.x) {
    float w = weight[i];
    x  += state[i * 4 + 0] * w;
    y  += state[i * 4 + 1] * w;
    vx += state[i * 4 + 2] * w;
    vy += state[i * 4 + 3] * w;
  }
  sh[0][tid] = x;  sh[1][tid] = y;  sh[2][tid] = vx;  sh[3][tid] = vy;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sh[0][tid] += sh[0][tid + s];
      sh[1][tid] += sh[1][tid + s];
      sh[2][tid] += sh[2][tid + s];
      sh[3][tid] += sh[3][tid + s];
    }
    __syncthreads();
  }
  if (tid == 0) {
    mean[0] = sh[0][0]; mean[1] = sh[1][0]; mean[2] = sh[2][0]; mean[3] = sh[3][0];
  }
}

extern "C" {

// Allocate GPU memory for state and weights
void particle_filter_alloc(unsigned int num_particles, void** d_state, void** d_weight) {
  cudaMalloc(d_state, num_particles * PF_STATE_STRIDE * sizeof(float));
  cudaMalloc(d_weight, num_particles * sizeof(float));
}

void particle_filter_free(void* d_state, void* d_weight) {
  if (d_state) cudaFree(d_state);
  if (d_weight) cudaFree(d_weight);
}

// Set initial particle positions and equal weights
void particle_filter_init(
  void* d_state,
  void* d_weight,
  unsigned int num_particles,
  float center_x, float center_y,
  float width, float height,
  float init_velocity_std
) {
  init_kernel<<<(num_particles + 255) / 256, 256>>>(
    (float*)d_state, (float*)d_weight, num_particles,
    center_x, center_y, width, height, init_velocity_std,
    (unsigned long long)num_particles * 12345ULL
  );
}

// Move particles (constant velocity + noise)
void particle_filter_predict(
  void* d_state,
  unsigned int num_particles,
  float dt,
  float pos_noise_std,
  float vel_noise_std
) {
  predict_kernel<<<(num_particles + 255) / 256, 256>>>(
    (float*)d_state, num_particles, dt, pos_noise_std, vel_noise_std,
    (unsigned long long)num_particles * 54321ULL
  );
}

// Set weight from observation map at each particle position
void particle_filter_weight(
  const void* d_state,
  void* d_weight,
  const float* d_observation,
  unsigned int num_particles,
  int map_width, int map_height
) {
  weight_kernel<<<(num_particles + 255) / 256, 256>>>(
    (const float*)d_state, (float*)d_weight, d_observation,
    num_particles, map_width, map_height
  );
}

// Make weights sum to 1 (sum on GPU, then scale)
void particle_filter_normalize_weights(void* d_weight, unsigned int num_particles) {
  float* d_block_sum = nullptr;
  float* d_total = nullptr;
  int num_blocks = (num_particles + 255) / 256;
  cudaMalloc(&d_block_sum, sizeof(float) * num_blocks);
  cudaMalloc(&d_total, sizeof(float));
  sum_weights_kernel<<<num_blocks, 256>>>((const float*)d_weight, d_block_sum, num_particles);
  sum_blocks_kernel<<<1, 256>>>(d_block_sum, d_total, num_blocks);
  float h_total;
  cudaMemcpy(&h_total, d_total, sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_block_sum);
  cudaFree(d_total);
  float inv = (h_total > 1e-9f) ? (1.0f / h_total) : 1.0f;
  normalize_kernel<<<(num_particles + 255) / 256, 256>>>((float*)d_weight, inv, num_particles);
}

// Resample: build CDF from weights, then copy particles by CDF (writes to d_state_out then back to d_state)
void particle_filter_resample(
  const void* d_state,
  const void* d_weight,
  void* d_state_out,
  unsigned int num_particles
) {
  float* d_cdf = nullptr;
  float* d_block_total = nullptr;
  float* d_block_prefix_sum = nullptr;
  cudaMalloc(&d_cdf, (num_particles + 1) * sizeof(float));
  int num_blocks = (num_particles + 255) / 256;
  cudaMalloc(&d_block_total, num_blocks * sizeof(float));
  cudaMalloc(&d_block_prefix_sum, num_blocks * sizeof(float));
  scan_weights_kernel<<<num_blocks, 256>>>(
    (const float*)d_weight, d_cdf, d_block_total, num_particles
  );
  scan_block_totals_kernel<<<1, 256>>>(d_block_total, d_block_prefix_sum, num_blocks);
  add_block_prefix_kernel<<<num_blocks, 256>>>(d_cdf, d_block_prefix_sum, num_particles);
  cudaFree(d_block_total);
  cudaFree(d_block_prefix_sum);
  float one = 1.0f;
  cudaMemcpy(d_cdf + num_particles, &one, sizeof(float), cudaMemcpyHostToDevice);
  resample_kernel<<<(num_particles + 255) / 256, 256>>>(
    (const float*)d_state, (const float*)d_weight, d_cdf,
    (float*)d_state_out, num_particles,
    (unsigned long long)num_particles * 99999ULL
  );
  cudaFree(d_cdf);
  cudaMemcpy((void*)d_state, d_state_out, num_particles * PF_STATE_STRIDE * sizeof(float), cudaMemcpyDeviceToDevice);
}

// Compute weighted mean (x, y, vx, vy) and copy to host
void particle_filter_mean_state(
  const void* d_state,
  const void* d_weight,
  unsigned int num_particles,
  float* mean
) {
  float* d_mean = nullptr;
  cudaMalloc(&d_mean, 4 * sizeof(float));
  mean_state_kernel<<<1, 256>>>(
    (const float*)d_state, (const float*)d_weight, d_mean, num_particles
  );
  cudaMemcpy(mean, d_mean, 4 * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_mean);
}

} // extern "C"
