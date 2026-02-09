#ifndef DETECTION_IO_H
#define DETECTION_IO_H

/**
 * detection_io.h
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * Defines the Detection struct (box + confidence) and functions to load detections
 * from CSV and to build a 2D observation map from detections for the particle filter.
 */

#include <vector>
#include <string>

namespace capstone {

struct Detection {
  float x;       // center or left
  float y;
  float w;
  float h;
  float confidence;
  int   class_id; // 0 = pedestrian, etc.
};

/** One frame's detections */
using DetectionsPerFrame = std::vector<Detection>;

/**
 * Load per-frame detections from a CSV:
 * frame_id,x,y,w,h,confidence,class_id
 * (or frame_id,x,y,w,h,confidence if class_id omitted)
 */
std::vector<DetectionsPerFrame> load_detections_csv(
  const std::string& path,
  int num_frames
);

/**
 * Build a 2D likelihood map (GPU or CPU) from detections.
 * Each detection draws a Gaussian blob at (x,y) with sigma ~ w/2.
 * Returns row-major float map of size width*height.
 */
std::vector<float> build_observation_map_cpu(
  int width, int height,
  const std::vector<Detection>& detections,
  float sigma_scale = 2.0f
);

} // namespace capstone

#endif /* DETECTION_IO_H */
