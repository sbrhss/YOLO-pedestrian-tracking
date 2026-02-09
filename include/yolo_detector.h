#ifndef YOLO_DETECTOR_H
#define YOLO_DETECTOR_H

/**
 * yolo_detector.h
 * Author: Saber Hosseini
 * Date: February 7, 2026
 *
 * Declares the YOLO detector class: load an ONNX model and run detection on frames.
 * Returns person boxes in the same format as the rest of the pipeline (center x, y, w, h, confidence).
 */

#include "detection_io.h"
#include <string>
#include <vector>

#if HAVE_OPENCV
#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#endif

namespace capstone {

/**
 * Real-time YOLO detector (OpenCV DNN backend).
 * Loads ONNX model (e.g. YOLOv8 from Ultralytics), runs inference on each frame,
 * returns detections in the same format as the particle filter pipeline (center x,y,w,h,confidence,class_id).
 * Use with run_pipeline_realtime() so YOLO runs per frame and feeds the particle filter automatically.
 */
class YOLODetector {
public:
  YOLODetector() = default;

#if HAVE_OPENCV
  /** Load ONNX model (e.g. yolov8n.onnx). Returns true on success. */
  bool load(const std::string& model_path);

  /**
   * Run detection on a frame. Returns detections (center x, y, w, h in frame coordinates).
   * Only returns boxes for class_id == person_class_id (default 0 = COCO person).
   * conf_threshold: keep detections with confidence >= this (lower = more people, more false positives).
   * nms_iou_threshold: NMS IoU threshold; overlapping boxes above this are merged (one per person). Use 0 to disable NMS.
   */
  std::vector<Detection> detect(
    const cv::Mat& frame,
    float conf_threshold = 0.5f,
    int person_class_id = 0,
    float nms_iou_threshold = 0.45f
  );

  /** Model input size (e.g. 640 for 640x640). */
  int input_size() const { return input_size_; }

  bool is_loaded() const { return loaded_; }
#else
  bool load(const std::string&) { return false; }
  std::vector<Detection> detect(const void* /*frame*/, float = 0.5f, int = 0, float = 0.45f) { return {}; }
  int input_size() const { return 640; }
  bool is_loaded() const { return false; }
#endif

private:
  bool loaded_ = false;
  int input_size_ = 640;

#if HAVE_OPENCV
  cv::dnn::Net net_;
  /** Parse DNN output (YOLOv8 style: 1 x 84 x 8400) and return person detections in frame coords; optional NMS. */
  std::vector<Detection> parse_output(
    const cv::Mat& output,
    int frame_width,
    int frame_height,
    float conf_threshold,
    int person_class_id,
    float nms_iou_threshold
  );
  static float detection_iou(const Detection& a, const Detection& b);
  static void nms(std::vector<Detection>& dets, float iou_threshold);
#endif
};

} // namespace capstone

#endif /* YOLO_DETECTOR_H */
