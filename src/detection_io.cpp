/**
 * detection_io.cpp
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * This file reads and writes detection data. It can load detections from a CSV file
 * (one line per detection: frame_id, x, y, w, h, confidence, class_id). It also builds
 * a 2D "observation map" from a list of detections: each detection adds a Gaussian blob
 * so the particle filter can use it as a likelihood map.
 */

#include "detection_io.h"
#include <fstream>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace capstone {

// Load all detections from a CSV; result is one vector of detections per frame.
std::vector<DetectionsPerFrame> load_detections_csv(
  const std::string& path,
  int num_frames
) {
  std::vector<DetectionsPerFrame> per_frame(num_frames);
  std::ifstream f(path);
  if (!f.is_open()) return per_frame;
  std::string line;
  while (std::getline(f, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream ss(line);
    int frame_id;
    Detection d;
    char comma;
    if (!(ss >> frame_id >> comma >> d.x >> comma >> d.y >> comma >> d.w >> comma >> d.h >> comma >> d.confidence)) continue;
    if (ss >> comma >> d.class_id) { /* optional */ } else d.class_id = 0;
    if (frame_id >= 0 && frame_id < num_frames)
      per_frame[frame_id].push_back(d);
  }
  return per_frame;
}

// Build a 2D map where each detection is a Gaussian blob; used by the particle filter.
std::vector<float> build_observation_map_cpu(
  int width, int height,
  const std::vector<Detection>& detections,
  float sigma_scale
) {
  std::vector<float> map((size_t)width * height, 0.0f);
  for (const auto& d : detections) {
    float sigma = std::max(d.w, d.h) * sigma_scale * 0.5f;
    if (sigma < 1.0f) sigma = 1.0f;
    int r = (int)(3.0f * sigma) + 1;  // Radius of the blob
    int cx = (int)(d.x + 0.5f);
    int cy = (int)(d.y + 0.5f);
    for (int dy = -r; dy <= r; dy++) {
      for (int dx = -r; dx <= r; dx++) {
        int x = cx + dx, y = cy + dy;
        if (x < 0 || x >= width || y < 0 || y >= height) continue;
        float g = std::exp(-(dx*dx + dy*dy) / (2.0f * sigma * sigma));
        map[(size_t)y * width + x] += d.confidence * g;
      }
    }
  }
  // Normalize so max value is 1
  float max_val = 0.0f;
  for (float v : map) max_val = std::max(max_val, v);
  if (max_val > 1e-6f)
    for (float& v : map) v /= max_val;
  return map;
}

} // namespace capstone
