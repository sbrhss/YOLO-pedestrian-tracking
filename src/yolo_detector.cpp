/**
 * yolo_detector.cpp
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * This file runs YOLO object detection using OpenCV's DNN module. It loads an ONNX model
 * (e.g. YOLOv8), runs it on each frame, and returns person detections as boxes (center x, y,
 * width, height) and confidence. It tries CUDA first for speed, then falls back to CPU.
 * NMS is used to remove overlapping boxes so we keep one box per person.
 */

#include "yolo_detector.h"
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <sys/stat.h>

#if HAVE_OPENCV
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>

namespace capstone {

// Load the ONNX model from disk; use CUDA if available.
bool YOLODetector::load(const std::string& model_path) {
  struct stat st;
  if (stat(model_path.c_str(), &st) != 0) {
    std::fprintf(stderr, "[YOLO] Model file not found: %s (use full path or run from project root, e.g. --yolo build/yolov8n.onnx)\n", model_path.c_str());
    return false;
  }
  if (!S_ISREG(st.st_mode)) {
    std::fprintf(stderr, "[YOLO] Not a regular file: %s\n", model_path.c_str());
    return false;
  }
  try {
    net_ = cv::dnn::readNetFromONNX(model_path);
    if (net_.empty()) {
      std::fprintf(stderr, "[YOLO] readNetFromONNX returned empty net for: %s\n", model_path.c_str());
      return false;
    }
#if defined(CV_VERSION_MAJOR) && CV_VERSION_MAJOR >= 4
    // Prefer GPU (CUDA) for faster inference; fall back to CPU if needed
    try {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
      std::fprintf(stderr, "[YOLO] OpenCV DNN backend: CUDA (GPU)\n");
    } catch (...) {
      net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
      net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
      std::fprintf(stderr, "[YOLO] OpenCV DNN backend: CPU (CUDA unavailable or failed)\n");
    }
#endif
    loaded_ = true;
    input_size_ = 640;
    return true;
  } catch (const cv::Exception& e) {
    std::fprintf(stderr, "[YOLO] OpenCV exception loading %s: %s\n", model_path.c_str(), e.what());
    return false;
  } catch (...) {
    std::fprintf(stderr, "[YOLO] Unknown exception loading: %s\n", model_path.c_str());
    return false;
  }
}

// Run YOLO on one frame; returns only "person" detections above conf_threshold.
std::vector<Detection> YOLODetector::detect(
  const cv::Mat& frame,
  float conf_threshold,
  int person_class_id,
  float nms_iou_threshold
) {
  if (!loaded_ || frame.empty()) return {};
  const int w = frame.cols;
  const int h = frame.rows;
  if (w <= 0 || h <= 0) return {};

  // Resize to model input size and run the network
  cv::Mat blob;
  cv::dnn::blobFromImage(frame, blob, 1.0 / 255.0, cv::Size(input_size_, input_size_), cv::Scalar(), true, false);
  net_.setInput(blob);
  std::vector<std::string> outNames = net_.getUnconnectedOutLayersNames();
  std::vector<cv::Mat> outs;
  net_.forward(outs, outNames);
  if (outs.empty()) return {};

  return parse_output(outs[0], w, h, conf_threshold, person_class_id, nms_iou_threshold);
}

// IoU = overlap area / union area; used by NMS to merge overlapping boxes.
float YOLODetector::detection_iou(const Detection& a, const Detection& b) {
  float ax1 = a.x - a.w * 0.5f, ay1 = a.y - a.h * 0.5f, ax2 = a.x + a.w * 0.5f, ay2 = a.y + a.h * 0.5f;
  float bx1 = b.x - b.w * 0.5f, by1 = b.y - b.h * 0.5f, bx2 = b.x + b.w * 0.5f, by2 = b.y + b.h * 0.5f;
  float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1), ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
  float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
  float inter = iw * ih;
  float area_a = a.w * a.h, area_b = b.w * b.h;
  float uni = area_a + area_b - inter;
  return (uni > 1e-6f) ? (inter / uni) : 0.f;
}

// Non-maximum suppression: keep the best box in each overlapping group.
void YOLODetector::nms(std::vector<Detection>& dets, float iou_threshold) {
  if (dets.empty() || iou_threshold <= 0.f) return;
  std::sort(dets.begin(), dets.end(), [](const Detection& a, const Detection& b) { return a.confidence > b.confidence; });
  for (size_t i = 0; i < dets.size(); i++) {
    for (size_t j = dets.size() - 1; j > i; j--) {
      if (detection_iou(dets[i], dets[j]) >= iou_threshold) dets.erase(dets.begin() + (std::ptrdiff_t)j);
    }
  }
}

// Turn raw YOLO output (1, 84, 8400) into list of person boxes in frame coordinates.
std::vector<Detection> YOLODetector::parse_output(
  const cv::Mat& output,
  int frame_width,
  int frame_height,
  float conf_threshold,
  int person_class_id,
  float nms_iou_threshold
) {
  std::vector<Detection> result;
  // YOLOv8: 4 box coords + 80 class scores, 8400 proposals
  if (output.dims < 2) return result;
  const int num_channels = output.size[1];
  const int num_proposals = output.dims >= 3 ? output.size[2] : output.size[0];
  const int num_classes = num_channels - 4;
  if (num_classes <= 0 || num_proposals <= 0) return result;

  const float scale_x = (float)frame_width / (float)input_size_;
  const float scale_y = (float)frame_height / (float)input_size_;

  for (int i = 0; i < num_proposals; i++) {
    float cx = 0.f, cy = 0.f, bw = 0.f, bh = 0.f;
    if (output.dims == 3) {
      cx = output.at<float>(0, 0, i);
      cy = output.at<float>(0, 1, i);
      bw = output.at<float>(0, 2, i);
      bh = output.at<float>(0, 3, i);
    } else {
      cx = output.at<float>(i, 0);
      cy = output.at<float>(i, 1);
      bw = output.at<float>(i, 2);
      bh = output.at<float>(i, 3);
    }
    int best_class = 0;
    float best_score = 0.f;
    for (int c = 0; c < num_classes; c++) {
      float s = output.dims == 3 ? output.at<float>(0, 4 + c, i) : output.at<float>(i, 4 + c);
      if (s > best_score) { best_score = s; best_class = c; }
    }
    // Keep only person class and above confidence
    if (best_class != person_class_id || best_score < conf_threshold) continue;

    Detection d;
    d.x = cx * scale_x;
    d.y = cy * scale_y;
    d.w = bw * scale_x;
    d.h = bh * scale_y;
    d.confidence = best_score;
    d.class_id = best_class;
    result.push_back(d);
  }

  if (nms_iou_threshold > 0.f) nms(result, nms_iou_threshold);
  return result;
}

} // namespace capstone

#endif /* HAVE_OPENCV */
