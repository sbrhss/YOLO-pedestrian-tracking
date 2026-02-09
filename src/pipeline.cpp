/**
 * pipeline.cpp
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * This file runs the full tracking pipeline. For video: it runs YOLO on each frame,
 * matches detections to existing tracks (by predicted position), updates or creates tracks
 * with a CUDA particle filter per person, draws trajectory and ID on the frame, and can
 * write an output video. For a folder of images it does the same but saves each frame
 * to an output folder. It also supports CSV detections or synthetic data, and a WMMA
 * (Tensor Core) benchmark.
 */

#include "pipeline.h"
#include "particle_filter_cuda.h"
#include "wmma_gemm.h"
#include "detection_io.h"
#include "yolo_detector.h"
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstring>
#include <algorithm>
#include <sys/stat.h>

#if HAVE_OPENCV
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/imgcodecs.hpp>
#endif

namespace capstone {

#if HAVE_OPENCV
// One track = one person: particle filter state on GPU, mean position/velocity, trail for drawing.
struct Track {
  void* d_state = nullptr;
  void* d_weight = nullptr;
  void* d_state_aux = nullptr;
  float mean[4] = {0, 0, 0, 0};
  int id = 0;
  int missed = 0;
  int assigned_det = -1;  // index into dets_map for this frame
  std::vector<cv::Point2f> trail;  // trajectory in frame coords (stable ID -> stable color along path)
};
static const int kMaxTracks = 12;
static const int kMaxMissedFrames = 30;
static const float kAssocMaxDistMap = 80.f;
static const int kMaxTrailLen = 60;  // number of points to draw per trajectory
#endif

// Run particle filter only (no video): CSV or synthetic detections, print mean each 20 frames.
static void run_pipeline_no_video(
  const std::vector<DetectionsPerFrame>& detections_per_frame,
  int map_width, int map_height,
  const PipelineConfig& config
) {
  const unsigned int n = config.num_particles;
  void* d_state = nullptr;
  void* d_state_aux = nullptr;
  void* d_weight = nullptr;
  float* d_observation = nullptr;

  particle_filter_alloc(n, &d_state, &d_weight);
  cudaMalloc(&d_state_aux, n * PF_STATE_DIM * sizeof(float));
  cudaMalloc(&d_observation, (size_t)map_width * map_height * sizeof(float));

  float mean[4];
  int num_frames = (int)detections_per_frame.size();
  if (num_frames == 0) num_frames = 100;

  for (int f = 0; f < num_frames; f++) {
    float cx = map_width * 0.5f, cy = map_height * 0.5f;
    if (f == 0) {
      particle_filter_init(d_state, d_weight, n, cx, cy, (float)map_width * 0.4f, (float)map_height * 0.4f, config.init_velocity_std);
    } else {
      particle_filter_predict(d_state, n, config.dt, config.process_pos_noise, config.process_vel_noise);
      std::vector<Detection> dets = (f < (int)detections_per_frame.size()) ? detections_per_frame[f] : std::vector<Detection>();
      std::vector<float> obs_cpu = build_observation_map_cpu(map_width, map_height, dets, 2.0f);
      if (obs_cpu.empty()) obs_cpu.resize((size_t)map_width * map_height, 1e-6f);
      cudaMemcpy(d_observation, obs_cpu.data(), (size_t)map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);
      particle_filter_weight(d_state, d_weight, d_observation, n, map_width, map_height);
      particle_filter_normalize_weights(d_weight, n);
      particle_filter_resample(d_state, d_weight, d_state_aux, n);
    }
    particle_filter_mean_state(d_state, d_weight, n, mean);
    if (f % 20 == 0)
      printf("frame %d mean x=%.1f y=%.1f vx=%.2f vy=%.2f\n", f, mean[0], mean[1], mean[2], mean[3]);
  }

  cudaFree(d_observation);
  cudaFree(d_state_aux);
  particle_filter_free(d_state, d_weight);
}

void run_pipeline(
  const std::string& detections_path,
  const std::string& video_path,
  const std::string& output_video_path,
  const std::string& yolo_model_path,
  int map_width,
  int map_height,
  const PipelineConfig& config
) {
  int num_frames = 500;
  std::vector<DetectionsPerFrame> detections_per_frame;

#if HAVE_OPENCV
  // Real-time path: video + YOLO — run YOLO per frame and feed detections to particle filter
  if (!video_path.empty() && !yolo_model_path.empty()) {
    YOLODetector detector;
    if (!detector.load(yolo_model_path)) {
      std::fprintf(stderr, "Failed to load YOLO model: %s\n", yolo_model_path.c_str());
      return;
    }
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
      std::fprintf(stderr, "Cannot open video: %s\n", video_path.c_str());
      return;
    }
    cv::Mat frame;
    int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    // Internal map size for particle filter (smaller = faster; drawing still in frame coords)
    const int map_max = 640;
    if (w >= h) {
      map_width = map_max;
      map_height = (int)(map_max * (float)h / (float)w);
    } else {
      map_height = map_max;
      map_width = (int)(map_max * (float)w / (float)h);
    }
    if (map_width < 64) map_width = 64;
    if (map_height < 64) map_height = 64;
    const float scale_to_map_x = (float)map_width / (float)w;
    const float scale_to_map_y = (float)map_height / (float)h;
    const float scale_to_frame_x = (float)w / (float)map_width;
    const float scale_to_frame_y = (float)h / (float)map_height;
    cv::VideoWriter writer;
    if (!output_video_path.empty() && w > 0 && h > 0)
      writer.open(output_video_path, cv::VideoWriter::fourcc('M','J','P','G'), 15.0, cv::Size(w, h));
    const unsigned int n = config.num_particles;
    float* d_observation = nullptr;
    cudaMalloc(&d_observation, (size_t)map_width * map_height * sizeof(float));
    std::vector<Track> tracks;
    int next_track_id = 1;
    int frame_idx = 0;
    double t_prev = cv::getTickCount();
    float fps_smooth = 0.f;
    const char* win_name = "Pedestrian tracking (YOLO + particle filter)";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL);
    static const cv::Scalar kTrackColors[] = {
      cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255),
      cv::Scalar(0, 165, 255), cv::Scalar(255, 255, 0), cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
      cv::Scalar(0, 128, 255), cv::Scalar(128, 0, 255), cv::Scalar(192, 192, 192), cv::Scalar(0, 200, 200)
    };
    std::printf("Real-time multi-person: YOLO -> particle filter (video %dx%d, map %dx%d, max %d tracks). Press 'q' to quit.\n",
                w, h, map_width, map_height, kMaxTracks);
    while (cap.read(frame)) {
      double t_now = cv::getTickCount();
      double elapsed = (t_now - t_prev) / cv::getTickFrequency();
      t_prev = t_now;
      float fps = (elapsed > 1e-6) ? (float)(1.0 / elapsed) : 0.f;
      fps_smooth = (fps_smooth > 0.f) ? (0.9f * fps_smooth + 0.1f * fps) : fps;

      // Get person detections and convert to map coordinates for the particle filter
      std::vector<Detection> dets = detector.detect(frame, 0.5f, 0, 0.45f);
      std::vector<Detection> dets_map;
      for (const auto& d : dets) {
        Detection dm;
        dm.x = d.x * scale_to_map_x;
        dm.y = d.y * scale_to_map_y;
        dm.w = d.w * scale_to_map_x;
        dm.h = d.h * scale_to_map_y;
        dm.confidence = d.confidence;
        dm.class_id = d.class_id;
        dets_map.push_back(dm);
      }

      // Reset assignment for this frame
      for (auto& t : tracks) t.assigned_det = -1;
      std::vector<bool> det_used(dets_map.size(), false);

      // Association: use particle filter PREDICTED position (mean + velocity*dt) so ID follows trajectory
      // Global assignment: sort all (track, det) pairs by distance, assign closest pairs first to reduce ID swaps
      struct Pair { int ti; int dj; float d2; };
      std::vector<Pair> pairs;
      for (size_t ti = 0; ti < tracks.size(); ti++) {
        float pred_x = tracks[ti].mean[0] + tracks[ti].mean[2] * config.dt;
        float pred_y = tracks[ti].mean[1] + tracks[ti].mean[3] * config.dt;
        for (size_t j = 0; j < dets_map.size(); j++) {
          float dx = dets_map[j].x - pred_x, dy = dets_map[j].y - pred_y;
          float d2 = dx * dx + dy * dy;
          if (d2 < kAssocMaxDistMap * kAssocMaxDistMap) pairs.push_back({ (int)ti, (int)j, d2 });
        }
      }
      std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d2 < b.d2; });
      for (const auto& p : pairs) {
        if (tracks[p.ti].assigned_det >= 0 || det_used[p.dj]) continue;
        tracks[p.ti].assigned_det = p.dj;
        tracks[p.ti].missed = 0;
        det_used[p.dj] = true;
      }
      for (auto& t : tracks)
        if (t.assigned_det < 0) t.missed++;

      // Create new tracks for unassigned detections
      for (size_t j = 0; j < dets_map.size(); j++) {
        if (det_used[j] || (int)tracks.size() >= kMaxTracks) continue;
        Track t;
        t.id = next_track_id++;
        t.missed = 0;
        t.assigned_det = (int)j;
        particle_filter_alloc(n, &t.d_state, &t.d_weight);
        cudaMalloc(&t.d_state_aux, n * PF_STATE_DIM * sizeof(float));
        float cx = dets_map[j].x, cy = dets_map[j].y;
        float bw = std::max(dets_map[j].w, 20.f), bh = std::max(dets_map[j].h, 40.f);
        particle_filter_init(t.d_state, t.d_weight, n, cx, cy, bw, bh, config.init_velocity_std);
        particle_filter_mean_state(t.d_state, t.d_weight, n, t.mean);
        tracks.push_back(t);
      }

      // Update each track
      for (auto& t : tracks) {
        if (t.assigned_det >= 0) {
          std::vector<Detection> single = { dets_map[t.assigned_det] };
          std::vector<float> obs_cpu = build_observation_map_cpu(map_width, map_height, single, 2.0f);
          if (obs_cpu.empty()) obs_cpu.resize((size_t)map_width * map_height, 1e-6f);
          cudaMemcpy(d_observation, obs_cpu.data(), (size_t)map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);
          particle_filter_predict(t.d_state, n, config.dt, config.process_pos_noise, config.process_vel_noise);
          particle_filter_weight(t.d_state, t.d_weight, d_observation, n, map_width, map_height);
          particle_filter_normalize_weights(t.d_weight, n);
          particle_filter_resample(t.d_state, t.d_weight, t.d_state_aux, n);
        } else {
          std::vector<float> obs_cpu((size_t)map_width * map_height, 0.5f);
          cudaMemcpy(d_observation, obs_cpu.data(), (size_t)map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);
          particle_filter_predict(t.d_state, n, config.dt, config.process_pos_noise, config.process_vel_noise);
          particle_filter_weight(t.d_state, t.d_weight, d_observation, n, map_width, map_height);
          particle_filter_normalize_weights(t.d_weight, n);
          particle_filter_resample(t.d_state, t.d_weight, t.d_state_aux, n);
        }
        particle_filter_mean_state(t.d_state, t.d_weight, n, t.mean);
      }

      // Append current position to each track's trajectory (frame coords)
      for (auto& t : tracks) {
        float fx = t.mean[0] * scale_to_frame_x;
        float fy = t.mean[1] * scale_to_frame_y;
        t.trail.push_back(cv::Point2f(fx, fy));
        if ((int)t.trail.size() > kMaxTrailLen) t.trail.erase(t.trail.begin());
      }

      // Remove dead tracks (free GPU buffers first, then erase)
      for (auto& t : tracks) {
        if (t.missed > kMaxMissedFrames) {
          particle_filter_free(t.d_state, t.d_weight);
          cudaFree(t.d_state_aux);
          t.d_state = t.d_weight = t.d_state_aux = nullptr;
        }
      }
      tracks.erase(std::remove_if(tracks.begin(), tracks.end(),
        [](const Track& t) { return t.d_state == nullptr; }), tracks.end());

      // Draw YOLO detection boxes in light gray (no per-person color here; track ID gives stable color)
      for (const auto& d : dets) {
        int x1 = (int)(d.x - d.w * 0.5f), y1 = (int)(d.y - d.h * 0.5f);
        int x2 = (int)(d.x + d.w * 0.5f), y2 = (int)(d.y + d.h * 0.5f);
        cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(180, 180, 180), 1);
      }
      // Draw each track's trajectory then box with person ID
      const int track_box_w = 80, track_box_h = 180;
      const int track_dot_radius = 8;
      for (const auto& t : tracks) {
        cv::Scalar color = kTrackColors[(t.id - 1) % 12];
        if (t.trail.size() >= 2) {
          std::vector<std::vector<cv::Point>> contours(1);
          for (const auto& p : t.trail) contours[0].push_back(cv::Point((int)p.x, (int)p.y));
          cv::polylines(frame, contours, false, color, 2);
        }
        float track_cx = t.mean[0] * scale_to_frame_x;
        float track_cy = t.mean[1] * scale_to_frame_y;
        int cx = (int)track_cx, cy = (int)track_cy;
        int bx1 = cx - track_box_w / 2, by1 = cy - track_box_h / 2;
        int bx2 = cx + track_box_w / 2, by2 = cy + track_box_h / 2;
        cv::rectangle(frame, cv::Point(bx1, by1), cv::Point(bx2, by2), color, 2);
        char label[16];
        std::snprintf(label, sizeof(label), "ID %d", t.id);
        cv::putText(frame, label, cv::Point(bx1, by1 - 4), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
        cv::circle(frame, cv::Point(cx, cy), track_dot_radius, color, -1);
        cv::circle(frame, cv::Point(cx, cy), track_dot_radius, color, 2);
      }

      char fps_buf[64];
      std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f  |  Det: %zu  |  Tracks: %zu", fps_smooth, dets.size(), tracks.size());
      cv::putText(frame, fps_buf, cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);

      if (writer.isOpened()) writer.write(frame);
      cv::imshow(win_name, frame);
      if (cv::waitKey(1) == 'q') break;
      if (frame_idx % 30 == 0)
        std::printf("frame %d  detections %zu  tracks %zu\n", frame_idx, dets.size(), tracks.size());
      frame_idx++;
    }
    for (auto& t : tracks) {
      particle_filter_free(t.d_state, t.d_weight);
      cudaFree(t.d_state_aux);
    }
    cudaFree(d_observation);
    cv::destroyWindow(win_name);
    if (writer.isOpened()) writer.release();
    std::printf("Processed %d frames. Max tracks used: %d\n", frame_idx, next_track_id - 1);
    return;
  }
#endif

  // If no video+YOLO: use CSV file or synthetic moving detection
  if (!detections_path.empty()) {
    detections_per_frame = load_detections_csv(detections_path, num_frames);
    if (!detections_per_frame.empty())
      num_frames = (int)detections_per_frame.size();
  } else {
    detections_per_frame.resize((size_t)num_frames);
    for (int f = 0; f < num_frames; f++) {
      float t = f * 0.1f;
      Detection d;
      d.x = map_width * (0.5f + 0.2f * std::sin(t));
      d.y = map_height * (0.5f + 0.2f * std::cos(t * 0.7f));
      d.w = 20.0f; d.h = 40.0f; d.confidence = 0.9f; d.class_id = 0;
      detections_per_frame[f].push_back(d);
    }
  }

#if HAVE_OPENCV
  if (!video_path.empty()) {
    cv::VideoCapture cap(video_path);
    if (cap.isOpened()) {
      cv::Mat frame;
      int w = (int)cap.get(cv::CAP_PROP_FRAME_WIDTH);
      int h = (int)cap.get(cv::CAP_PROP_FRAME_HEIGHT);
      if (map_width <= 0) map_width = w;
      if (map_height <= 0) map_height = h;
      cv::VideoWriter writer;
      if (!output_video_path.empty() && w > 0 && h > 0)
        writer.open(output_video_path, cv::VideoWriter::fourcc('M','J','P','G'), 15.0, cv::Size(w, h));
      const unsigned int n = config.num_particles;
      void* d_state = nullptr, * d_state_aux = nullptr, * d_weight = nullptr;
      float* d_observation = nullptr;
      particle_filter_alloc(n, &d_state, &d_weight);
      cudaMalloc(&d_state_aux, n * PF_STATE_DIM * sizeof(float));
      cudaMalloc(&d_observation, (size_t)map_width * map_height * sizeof(float));
      int frame_idx = 0;
      float mean[4];
      while (cap.read(frame) && frame_idx < num_frames) {
        float cx = map_width * 0.5f, cy = map_height * 0.5f;
        if (frame_idx == 0) {
          particle_filter_init(d_state, d_weight, n, cx, cy, (float)map_width * 0.4f, (float)map_height * 0.4f, config.init_velocity_std);
        } else {
          particle_filter_predict(d_state, n, config.dt, config.process_pos_noise, config.process_vel_noise);
          std::vector<Detection> dets = (frame_idx < (int)detections_per_frame.size()) ? detections_per_frame[frame_idx] : std::vector<Detection>();
          std::vector<float> obs_cpu = build_observation_map_cpu(map_width, map_height, dets, 2.0f);
          if (obs_cpu.empty()) obs_cpu.resize((size_t)map_width * map_height, 1e-6f);
          cudaMemcpy(d_observation, obs_cpu.data(), (size_t)map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);
          particle_filter_weight(d_state, d_weight, d_observation, n, map_width, map_height);
          particle_filter_normalize_weights(d_weight, n);
          particle_filter_resample(d_state, d_weight, d_state_aux, n);
        }
        particle_filter_mean_state(d_state, d_weight, n, mean);
        cv::circle(frame, cv::Point((int)mean[0], (int)mean[1]), 15, cv::Scalar(0, 255, 0), 2);
        if (writer.isOpened()) writer.write(frame);
        frame_idx++;
      }
      cudaFree(d_observation);
      cudaFree(d_state_aux);
      particle_filter_free(d_state, d_weight);
      if (writer.isOpened()) writer.release();
      return;
    }
  }
#endif
  run_pipeline_no_video(detections_per_frame, map_width, map_height, config);
}

#if HAVE_OPENCV
void run_pipeline_images(
  const std::string& images_dir,
  const std::string& output_images_dir,
  const std::string& yolo_model_path,
  float conf_threshold
) {
  if (images_dir.empty() || output_images_dir.empty() || yolo_model_path.empty()) {
    std::fprintf(stderr, "run_pipeline_images: need images_dir, output_images_dir, and yolo_model_path\n");
    return;
  }
  YOLODetector detector;
  if (!detector.load(yolo_model_path)) {
    std::fprintf(stderr, "Failed to load YOLO model: %s\n", yolo_model_path.c_str());
    return;
  }
  // Find all images in the folder (PNG, JPG)
  std::vector<std::string> paths;
  cv::glob(images_dir + "/*.PNG", paths);
  std::vector<std::string> p2;
  cv::glob(images_dir + "/*.png", p2);
  paths.insert(paths.end(), p2.begin(), p2.end());
  p2.clear(); cv::glob(images_dir + "/*.jpg", p2);
  paths.insert(paths.end(), p2.begin(), p2.end());
  p2.clear(); cv::glob(images_dir + "/*.jpeg", p2);
  paths.insert(paths.end(), p2.begin(), p2.end());
  p2.clear(); cv::glob(images_dir + "/*.JPG", p2);
  paths.insert(paths.end(), p2.begin(), p2.end());
  std::sort(paths.begin(), paths.end());
  if (paths.empty()) {
    std::fprintf(stderr, "No images found in %s\n", images_dir.c_str());
    return;
  }
  mkdir(output_images_dir.c_str(), 0755);

  cv::Mat first = cv::imread(paths[0]);
  if (first.empty()) {
    std::fprintf(stderr, "Could not read first image\n");
    return;
  }
  int w = first.cols, h = first.rows;
  PipelineConfig config;
  config.num_particles = 4096;
  config.dt = 1.0f;
  const int map_max = 640;
  int map_width, map_height;
  if (w >= h) {
    map_width = map_max;
    map_height = (int)(map_max * (float)h / (float)w);
  } else {
    map_height = map_max;
    map_width = (int)(map_max * (float)w / (float)h);
  }
  if (map_width < 64) map_width = 64;
  if (map_height < 64) map_height = 64;
  const float scale_to_map_x = (float)map_width / (float)w;
  const float scale_to_map_y = (float)map_height / (float)h;
  const float scale_to_frame_x = (float)w / (float)map_width;
  const float scale_to_frame_y = (float)h / (float)map_height;

  const unsigned int n = config.num_particles;
  float* d_observation = nullptr;
  cudaMalloc(&d_observation, (size_t)map_width * map_height * sizeof(float));
  std::vector<Track> tracks;
  int next_track_id = 1;
  static const cv::Scalar kTrackColors[] = {
    cv::Scalar(0, 255, 0), cv::Scalar(255, 0, 0), cv::Scalar(0, 255, 255), cv::Scalar(255, 0, 255),
    cv::Scalar(0, 165, 255), cv::Scalar(255, 255, 0), cv::Scalar(128, 255, 0), cv::Scalar(255, 128, 0),
    cv::Scalar(0, 128, 255), cv::Scalar(128, 0, 255), cv::Scalar(192, 192, 192), cv::Scalar(0, 200, 200)
  };
  const int track_box_w = 80, track_box_h = 180, track_dot_radius = 8;

  std::printf("Images pipeline: YOLO + particle filter (ID, trajectory, box) on %zu images. Saving to %s\n", paths.size(), output_images_dir.c_str());
  const double t_start = cv::getTickCount();
  size_t processed_count = 0;
  for (size_t fi = 0; fi < paths.size(); fi++) {
    cv::Mat frame = cv::imread(paths[fi]);
    if (frame.empty()) continue;
    if (frame.cols != w || frame.rows != h) {
      cv::resize(frame, frame, cv::Size(w, h));
    }
    std::vector<Detection> dets = detector.detect(frame, conf_threshold, 0, 0.45f);
    std::vector<Detection> dets_map;
    for (const auto& d : dets) {
      Detection dm;
      dm.x = d.x * scale_to_map_x; dm.y = d.y * scale_to_map_y;
      dm.w = d.w * scale_to_map_x; dm.h = d.h * scale_to_map_y;
      dm.confidence = d.confidence; dm.class_id = d.class_id;
      dets_map.push_back(dm);
    }
    for (auto& t : tracks) t.assigned_det = -1;
    std::vector<bool> det_used(dets_map.size(), false);
    struct Pair { int ti; int dj; float d2; };
    std::vector<Pair> pairs;
    for (size_t ti = 0; ti < tracks.size(); ti++) {
      float pred_x = tracks[ti].mean[0] + tracks[ti].mean[2] * config.dt;
      float pred_y = tracks[ti].mean[1] + tracks[ti].mean[3] * config.dt;
      for (size_t j = 0; j < dets_map.size(); j++) {
        float dx = dets_map[j].x - pred_x, dy = dets_map[j].y - pred_y;
        float d2 = dx * dx + dy * dy;
        if (d2 < kAssocMaxDistMap * kAssocMaxDistMap) pairs.push_back({ (int)ti, (int)j, d2 });
      }
    }
    std::sort(pairs.begin(), pairs.end(), [](const Pair& a, const Pair& b) { return a.d2 < b.d2; });
    for (const auto& p : pairs) {
      if (tracks[p.ti].assigned_det >= 0 || det_used[p.dj]) continue;
      tracks[p.ti].assigned_det = p.dj;
      tracks[p.ti].missed = 0;
      det_used[p.dj] = true;
    }
    for (auto& t : tracks) if (t.assigned_det < 0) t.missed++;
    for (size_t j = 0; j < dets_map.size(); j++) {
      if (det_used[j] || (int)tracks.size() >= kMaxTracks) continue;
      Track t;
      t.id = next_track_id++;
      t.missed = 0;
      t.assigned_det = (int)j;
      particle_filter_alloc(n, &t.d_state, &t.d_weight);
      cudaMalloc(&t.d_state_aux, n * PF_STATE_DIM * sizeof(float));
      float cx = dets_map[j].x, cy = dets_map[j].y;
      float bw = std::max(dets_map[j].w, 20.f), bh = std::max(dets_map[j].h, 40.f);
      particle_filter_init(t.d_state, t.d_weight, n, cx, cy, bw, bh, config.init_velocity_std);
      particle_filter_mean_state(t.d_state, t.d_weight, n, t.mean);
      tracks.push_back(t);
    }
    for (auto& t : tracks) {
      if (t.assigned_det >= 0) {
        std::vector<Detection> single = { dets_map[t.assigned_det] };
        std::vector<float> obs_cpu = build_observation_map_cpu(map_width, map_height, single, 2.0f);
        if (obs_cpu.empty()) obs_cpu.resize((size_t)map_width * map_height, 1e-6f);
        cudaMemcpy(d_observation, obs_cpu.data(), (size_t)map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);
        particle_filter_predict(t.d_state, n, config.dt, config.process_pos_noise, config.process_vel_noise);
        particle_filter_weight(t.d_state, t.d_weight, d_observation, n, map_width, map_height);
        particle_filter_normalize_weights(t.d_weight, n);
        particle_filter_resample(t.d_state, t.d_weight, t.d_state_aux, n);
      } else {
        std::vector<float> obs_cpu((size_t)map_width * map_height, 0.5f);
        cudaMemcpy(d_observation, obs_cpu.data(), (size_t)map_width * map_height * sizeof(float), cudaMemcpyHostToDevice);
        particle_filter_predict(t.d_state, n, config.dt, config.process_pos_noise, config.process_vel_noise);
        particle_filter_weight(t.d_state, t.d_weight, d_observation, n, map_width, map_height);
        particle_filter_normalize_weights(t.d_weight, n);
        particle_filter_resample(t.d_state, t.d_weight, t.d_state_aux, n);
      }
      particle_filter_mean_state(t.d_state, t.d_weight, n, t.mean);
    }
    for (auto& t : tracks) {
      float fx = t.mean[0] * scale_to_frame_x, fy = t.mean[1] * scale_to_frame_y;
      t.trail.push_back(cv::Point2f(fx, fy));
      if ((int)t.trail.size() > kMaxTrailLen) t.trail.erase(t.trail.begin());
    }
    for (auto& t : tracks) {
      if (t.missed > kMaxMissedFrames) {
        particle_filter_free(t.d_state, t.d_weight);
        cudaFree(t.d_state_aux);
        t.d_state = t.d_weight = t.d_state_aux = nullptr;
      }
    }
    tracks.erase(std::remove_if(tracks.begin(), tracks.end(), [](const Track& t) { return t.d_state == nullptr; }), tracks.end());

    for (const auto& d : dets) {
      int x1 = (int)(d.x - d.w * 0.5f), y1 = (int)(d.y - d.h * 0.5f);
      int x2 = (int)(d.x + d.w * 0.5f), y2 = (int)(d.y + d.h * 0.5f);
      cv::rectangle(frame, cv::Point(x1, y1), cv::Point(x2, y2), cv::Scalar(180, 180, 180), 1);
    }
    for (const auto& t : tracks) {
      cv::Scalar color = kTrackColors[(t.id - 1) % 12];
      if (t.trail.size() >= 2) {
        std::vector<std::vector<cv::Point>> contours(1);
        for (const auto& p : t.trail) contours[0].push_back(cv::Point((int)p.x, (int)p.y));
        cv::polylines(frame, contours, false, color, 2);
      }
      float track_cx = t.mean[0] * scale_to_frame_x, track_cy = t.mean[1] * scale_to_frame_y;
      int cx = (int)track_cx, cy = (int)track_cy;
      int bx1 = cx - track_box_w / 2, by1 = cy - track_box_h / 2;
      int bx2 = cx + track_box_w / 2, by2 = cy + track_box_h / 2;
      cv::rectangle(frame, cv::Point(bx1, by1), cv::Point(bx2, by2), color, 2);
      char label[16];
      std::snprintf(label, sizeof(label), "ID %d", t.id);
      cv::putText(frame, label, cv::Point(bx1, by1 - 4), cv::FONT_HERSHEY_SIMPLEX, 0.7, color, 2);
      cv::circle(frame, cv::Point(cx, cy), track_dot_radius, color, -1);
      cv::circle(frame, cv::Point(cx, cy), track_dot_radius, color, 2);
    }
    const double t_now = cv::getTickCount();
    const double elapsed_sec = (t_now - t_start) / cv::getTickFrequency();
    const double fps = (elapsed_sec > 1e-6) ? (double)(fi + 1) / elapsed_sec : 0.0;
    char fps_buf[64];
    std::snprintf(fps_buf, sizeof(fps_buf), "FPS: %.1f", fps);
    cv::putText(frame, fps_buf, cv::Point(10, 35), cv::FONT_HERSHEY_SIMPLEX, 0.9, cv::Scalar(0, 255, 255), 2);
    size_t last_slash = paths[fi].find_last_of("/\\");
    std::string filename = (last_slash != std::string::npos) ? paths[fi].substr(last_slash + 1) : paths[fi];
    std::string out_path = output_images_dir + "/" + filename;
    if (!cv::imwrite(out_path, frame))
      std::fprintf(stderr, "Failed to write %s\n", out_path.c_str());
    processed_count = fi + 1;
    if ((fi + 1) % 10 == 0 || fi == paths.size() - 1)
      std::printf("  %zu/%zu saved  (FPS: %.1f)\n", fi + 1, paths.size(), fps);
  }
  for (auto& t : tracks) {
    particle_filter_free(t.d_state, t.d_weight);
    cudaFree(t.d_state_aux);
  }
  cudaFree(d_observation);
  const double t_end = cv::getTickCount();
  const double total_sec = (t_end - t_start) / cv::getTickFrequency();
  const double avg_fps = (total_sec > 1e-6 && processed_count > 0) ? (double)processed_count / total_sec : 0.0;
  std::printf("Saved %zu images with ID, trajectory, and track box to %s\n", paths.size(), output_images_dir.c_str());
  std::printf("Images mode FPS: %.2f  (processed %zu images in %.2f s)\n", avg_fps, processed_count, total_sec);
}
#else
void run_pipeline_images(const std::string&, const std::string&, const std::string&, float) {}
#endif

// Benchmark: run FP16 matrix multiply on GPU and report time and GFLOPS.
void run_wmma_benchmark(int M, int N, int K) {
  M = (M + 15) / 16 * 16;
  N = (N + 15) / 16 * 16;
  K = (K + 15) / 16 * 16;
  size_t szA = (size_t)M * K * sizeof(uint16_t);
  size_t szB = (size_t)K * N * sizeof(uint16_t);
  size_t szC = (size_t)M * N * sizeof(float);
  void* d_A = nullptr, * d_B = nullptr, * d_C = nullptr;
  cudaMalloc(&d_A, szA);
  cudaMalloc(&d_B, szB);
  cudaMalloc(&d_C, szC);
  cudaMemset(d_A, 0, szA);
  cudaMemset(d_B, 0, szB);
  wmma_gemm_fp16(d_A, d_B, d_C, M, N, K);
  cudaDeviceSynchronize();
  const int repeats = 20;
  float total_ms = 0.0f;
  for (int r = 0; r < repeats; r++)
    total_ms += wmma_gemm_fp16_timed(d_A, d_B, d_C, M, N, K);
  float ms = total_ms / (float)repeats;
  double flops = 2.0 * (double)M * (double)N * (double)K;
  double gflops = (ms > 1e-6f) ? (flops / (1e6 * (double)ms)) : 0.0;
  printf("WMMA GEMM %d x %d x %d: %.3f ms (avg %d runs), %.2f GFLOPS (FP16 Tensor Cores)\n", M, N, K, ms, repeats, gflops);
  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
}

} // namespace capstone
