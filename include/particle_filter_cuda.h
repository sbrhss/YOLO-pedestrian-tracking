#ifndef PARTICLE_FILTER_CUDA_H
#define PARTICLE_FILTER_CUDA_H

/**
 * particle_filter_cuda.h
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * C API for the CUDA particle filter. Each particle has state (x, y, vx, vy).
 * Steps: alloc, init, then each frame: predict, weight from observation map,
 * normalize weights, resample, and optionally get mean state.
 */

#include <cstddef>

#ifdef __cplusplus
extern "C" {
#endif

/** Particle state: x, y, vx, vy (4 floats per particle) */
#define PF_STATE_DIM 4

/**
 * Allocate GPU buffers for particle filter. Call once.
 * state: [num_particles * PF_STATE_DIM] (x, y, vx, vy)
 * weight: [num_particles]
 */
void particle_filter_alloc(unsigned int num_particles, void** d_state, void** d_weight);

/**
 * Free GPU buffers.
 */
void particle_filter_free(void* d_state, void* d_weight);

/**
 * Initialize particles: spread uniformly in (x,y) and small random velocity.
 * width, height: image or region size for initial spread.
 */
void particle_filter_init(
  void* d_state,
  void* d_weight,
  unsigned int num_particles,
  float center_x, float center_y,
  float width, float height,
  float init_velocity_std
);

/**
 * Predict step: constant velocity model + Gaussian noise.
 * dt: time step.
 * process_noise_std: std for x,y and for vx,vy (e.g. 2.0f, 0.5f).
 */
void particle_filter_predict(
  void* d_state,
  unsigned int num_particles,
  float dt,
  float pos_noise_std,
  float vel_noise_std
);

/**
 * Weight step: likelihood from 2D score map (e.g. from detector).
 * d_observation: device pointer to float map, row-major, size map_width * map_height.
 * Map is sampled at particle (x,y) with bilinear or nearest.
 */
void particle_filter_weight(
  const void* d_state,
  void* d_weight,
  const float* d_observation,
  unsigned int num_particles,
  int map_width, int map_height
);

/**
 * Normalize weights (sum = 1) in-place.
 */
void particle_filter_normalize_weights(void* d_weight, unsigned int num_particles);

/**
 * Resample: stratified resampling using prefix sum. Reads d_state, d_weight; writes new d_state.
 * Optional: d_rand can be NULL (uses curand or simple RNG). For deterministic test, pass pre-filled random [0,1).
 */
void particle_filter_resample(
  const void* d_state,
  const void* d_weight,
  void* d_state_out,
  unsigned int num_particles
);

/**
 * Compute weighted mean state (x, y, vx, vy) into host array mean[4].
 */
void particle_filter_mean_state(
  const void* d_state,
  const void* d_weight,
  unsigned int num_particles,
  float* mean
);

#ifdef __cplusplus
}
#endif

#endif /* PARTICLE_FILTER_CUDA_H */
