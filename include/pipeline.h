#ifndef PIPELINE_H
#define PIPELINE_H

/**
 * pipeline.h
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * Declares the main tracking pipeline: run_pipeline (video or CSV/synthetic),
 * run_pipeline_images (folder of images in, folder of images out with tracking overlay),
 * and run_wmma_benchmark. PipelineConfig holds particle filter settings.
 */

#include <string>
#include <vector>
#include "detection_io.h"

namespace capstone {

struct PipelineConfig {
  unsigned int num_particles = 4096;
  float process_pos_noise   = 2.0f;
  float process_vel_noise   = 0.5f;
  float init_velocity_std   = 1.0f;
  float dt                  = 1.0f;
};

/**
 * Run tracking pipeline:
 * - Optional: read video frames (if HAVE_OPENCV and video_path not empty)
 * - For each frame: get detections (from file, from YOLO, or placeholder), build observation map,
 *   run particle filter (predict -> weight -> normalize -> resample), get mean state.
 * - Optionally write output video with track overlay.
 *
 * If yolo_model_path is non-empty and video_path is set, runs real-time: YOLO on each frame
 * and feeds detections directly to the particle filter (no CSV). Otherwise uses
 * detections_path (CSV) or synthetic detections.
 */
void run_pipeline(
  const std::string& detections_path,
  const std::string& video_path,
  const std::string& output_video_path,
  const std::string& yolo_model_path,
  int map_width,
  int map_height,
  const PipelineConfig& config
);

/**
 * Benchmark: WMMA GEMM only (no video). Reports GFLOPS.
 */
void run_wmma_benchmark(int M, int N, int K);

/**
 * Run YOLO detection only on a folder of images (no particle filter).
 * Reads images from images_dir, runs YOLO, draws detection boxes, saves to output_images_dir.
 * conf_threshold: lower (e.g. 0.25) detects more people, higher (e.g. 0.5) fewer false positives.
 */
void run_pipeline_images(
  const std::string& images_dir,
  const std::string& output_images_dir,
  const std::string& yolo_model_path,
  float conf_threshold = 0.25f
);

} // namespace capstone

#endif /* PIPELINE_H */
