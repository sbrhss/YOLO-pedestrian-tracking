/**
 * main.cpp
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * This is the program entry point. It reads command-line options (video path, YOLO model,
 * image folders, etc.) and then runs the right mode: WMMA benchmark, images pipeline
 * (YOLO + tracking on a folder of images), or video pipeline (YOLO + particle filter on video).
 * No window is shown in images mode; output is saved to a folder.
 */

#include "pipeline.h"
#include "wmma_gemm.h"
#include <cstdio>
#include <cstring>
#include <string>

#if HAVE_OPENCV
#include <opencv2/core.hpp>
#endif

int main(int argc, char** argv) {
  // Options we can set from the command line
  bool bench_wmma = false;
  std::string detections_path;
  std::string video_path;
  std::string output_video_path;
  std::string yolo_model_path;
  std::string images_dir;
  std::string output_images_dir;
  float conf_threshold = 0.25f;  // Lower value = detect more people (more false positives)
  int map_w = 640, map_h = 480;

  // Parse command line: --video, --yolo, --images, --out-images, --conf, etc.
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--bench-wmma") == 0) bench_wmma = true;
    else if (strcmp(argv[i], "--detections") == 0 && i + 1 < argc) detections_path = argv[++i];
    else if (strcmp(argv[i], "--video") == 0 && i + 1 < argc) video_path = argv[++i];
    else if (strcmp(argv[i], "--out-video") == 0 && i + 1 < argc) output_video_path = argv[++i];
    else if (strcmp(argv[i], "--yolo") == 0 && i + 1 < argc) yolo_model_path = argv[++i];
    else if (strcmp(argv[i], "--images") == 0 && i + 1 < argc) images_dir = argv[++i];
    else if (strcmp(argv[i], "--out-images") == 0 && i + 1 < argc) output_images_dir = argv[++i];
    else if (strcmp(argv[i], "--conf") == 0 && i + 1 < argc) conf_threshold = (float)atof(argv[++i]);
    else if (strcmp(argv[i], "--map") == 0 && i + 2 < argc) { map_w = atoi(argv[++i]); map_h = atoi(argv[++i]); }
  }

  // Run only the Tensor Core benchmark (no video)
  if (bench_wmma) {
    capstone::run_wmma_benchmark(256, 256, 256);
    capstone::run_wmma_benchmark(512, 512, 512);
    capstone::run_wmma_benchmark(1024, 1024, 1024);
    return 0;
  }

  // Particle filter settings (used for video and images pipeline)
  capstone::PipelineConfig config;
  config.num_particles = 4096;
  config.process_pos_noise = 2.0f;
  config.process_vel_noise = 0.5f;
  config.init_velocity_std = 1.0f;
  config.dt = 1.0f;

#if HAVE_OPENCV
  // Print whether OpenCV was built with CUDA (so we know if YOLO uses GPU)
  {
    std::string info = cv::getBuildInformation();
    printf("[OpenCV] Build info (CUDA/cuDNN):\n");
    bool found = false;
    for (size_t start = 0; start < info.size(); ) {
      size_t end = info.find('\n', start);
      if (end == std::string::npos) end = info.size();
      std::string line = info.substr(start, end - start);
      if (line.find("CUDA") != std::string::npos ||
          line.find("cuDNN") != std::string::npos ||
          line.find("NVIDIA") != std::string::npos ||
          line.find("GPU") != std::string::npos) {
        printf("  %s\n", line.c_str());
        found = true;
      }
      start = end + (end < info.size() ? 1 : 0);
    }
    if (!found)
      printf("  (no CUDA/cuDNN lines found - OpenCV may be built without GPU)\n");
  }
#endif

  // Mode: process a folder of images and save results (with tracking and ID on each image)
  if (!images_dir.empty() && !output_images_dir.empty() && !yolo_model_path.empty()) {
    printf("Images mode: YOLO + particle filter, save to folder.\n");
    printf("  Input:  %s\n", images_dir.c_str());
    printf("  Output: %s\n", output_images_dir.c_str());
    printf("  YOLO:   %s  (conf=%.2f, lower=more people)\n", yolo_model_path.c_str(), conf_threshold);
    capstone::run_pipeline_images(images_dir, output_images_dir, yolo_model_path, conf_threshold);
    printf("Done.\n");
    return 0;
  }

  printf("Running pipeline: map %dx%d, %u particles\n", map_w, map_h, config.num_particles);
  const char* det_src = !detections_path.empty() ? detections_path.c_str()
    : (!yolo_model_path.empty() && !video_path.empty()) ? "(YOLO real-time)" : "(synthetic)";
  printf("Detections: %s\n", det_src);
  printf("Video: %s\n", video_path.empty() ? "(none)" : video_path.c_str());
  printf("YOLO (real-time): %s\n", yolo_model_path.empty() ? "(none)" : yolo_model_path.c_str());

  // Run the main pipeline (video with YOLO + tracking, or CSV/synthetic)
  capstone::run_pipeline(detections_path, video_path, output_video_path, yolo_model_path, map_w, map_h, config);
  printf("Done.\n");
  return 0;
}
