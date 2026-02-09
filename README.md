# GPU Capstone: Pedestrian Tracking with YOLO + Particle Filter (CUDA)

This project tracks pedestrians in video. It uses **YOLO** to detect people and a **particle filter on the GPU** to follow them over time. There is also a **custom kernel that uses Tensor Cores (WMMA)** for fast matrix math. It is written in **C/C++ and CUDA** and was made for the Coursera GPU Programming capstone. It runs on **NVIDIA RTX A2000 6GB** (or other GPUs with compute capability 8.6, Ampere).

## Demo: how the tracking looks

The GIF below shows the pipeline in action: YOLO detects people, the particle filter keeps stable IDs, and each person gets a colored box, trajectory trail, and label.

![Pedestrian tracking demo — YOLO + particle filter with stable IDs and trajectories](output_images/output_gif.gif)

## What the project does

1. **Detection**: Finds pedestrians in each frame. You can use YOLO inside the program (with an ONNX model) or give detections from a CSV file (e.g. from YOLO or TensorRT run separately).
2. **Observation map**: From the detections we build a 2D map that says “how likely is a person here?”. The particle filter uses this to weight the particles.
3. **Particle filter (on GPU)**: For each frame we: predict where particles move, weight them using the map, normalize, then resample. All this runs on the GPU in CUDA.
4. **WMMA benchmark**: A separate benchmark runs a matrix multiply that uses Tensor Cores (FP16). You can see the speed in GFLOPS.

## Main parts of the code

| Part | What it does |
|------|----------------|
| **WMMA GEMM** (`cuda/wmma_gemm.cu`) | Matrix multiply in FP16 using Tensor Cores (16×16×16 tiles). |
| **Particle filter** (`cuda/particle_filter.cu`) | Predict step, weight from the 2D map, normalize weights, then resample (with scan). |
| **Detection I/O** (`src/detection_io.cpp`) | Reads detections from a CSV file and builds the 2D likelihood map (Gaussian blobs). |
| **YOLO detector** (`src/yolo_detector.cpp`) | Loads a YOLOv8 ONNX model with OpenCV DNN and runs it on each frame. Returns person detections for the particle filter. |
| **Pipeline** (`src/pipeline.cpp`) | Puts everything together: read video, get detections (from YOLO or CSV), build map, run particle filter, draw result. |

## What you need to build

- **CUDA Toolkit** (version 11 or newer)
- **CMake** (3.18 or newer)
- **C++17**
- **OpenCV** (optional but recommended for video and for YOLO). If OpenCV is not found, the project still builds but video and YOLO are not available.

## How to build

```bash
cd /path/to/gpu_capstone_project
cmake -B build -DCMAKE_CUDA_ARCHITECTURES=86
cmake --build build
```

Use `86` for RTX A2000 (Ampere). For another GPU, change the number to your GPU’s compute capability.

## How to run

**1. Only benchmark Tensor Cores (WMMA):**
```bash
./build/PedestrianTrackingCUDA --bench-wmma
```

**2. Tracking with fake detections (no CSV, no video file):**
```bash
./build/PedestrianTrackingCUDA --map 640 480
```

**3. Tracking with detections from a CSV file:**
```bash
./build/PedestrianTrackingCUDA --detections detections.csv --map 640 480
```

**4. With a video file and detections from CSV:**
```bash
./build/PedestrianTrackingCUDA --video input.mp4 --out-video out.avi --detections detections.csv
```

**5. YOLO on each frame (no CSV):** The program runs YOLO on every frame and gives the detections to the particle filter.
```bash
./build/PedestrianTrackingCUDA --video input.mp4 --out-video out.avi --yolo yolov8n.onnx
```
For webcam use: `--video 0 --yolo yolov8n.onnx`

### How to get a YOLO ONNX model

Install Ultralytics and export YOLOv8 to ONNX:

```bash
pip install ultralytics
python -c "from ultralytics import YOLO; m=YOLO('yolov8n.pt'); m.export(format='onnx')"
```

You get `yolov8n.onnx`. Put it in the project folder or give the full path to `--yolo`. The program uses OpenCV DNN (on GPU if OpenCV was built with CUDA/cuDNN).

### CSV format for detections

One line per detection:

```
frame_id,x,y,w,h,confidence,class_id
0,320,240,30,60,0.9,0
1,322,238,30,60,0.85,0
```

You can create this file by running YOLO or TensorRT on your video in another script, then use `--detections` in this program.

### Pedestrian tracking images (sample data)

A set of pedestrian tracking images suitable for testing the pipeline (e.g. with `--images` and `--out-images`) can be found here:

- **People Tracking dataset**: https://www.kaggle.com/datasets/trainingdatapro/people-tracking

Download the dataset from Kaggle, extract the images into a folder, then run for example:

```bash
./build/PedestrianTrackingCUDA --images /path/to/people-tracking/images --out-images /path/to/output --yolo yolov8n.onnx --conf 0.25
```

## Tensor Cores (WMMA) and YOLO

- **In this project**: The **WMMA GEMM** kernel is our own code that uses Tensor Cores (FP16). Run `--bench-wmma` to see the speed.
- **YOLO in the pipeline**: When you use `--yolo model.onnx --video ...`, YOLO runs inside the same program on each frame and its detections go directly to the particle filter (via OpenCV DNN).
- **If you use TensorRT**: You can run YOLO with TensorRT yourself, write the detections to a CSV, then run this program with `--detections`. That way YOLO uses Tensor Cores in TensorRT, and this program does the particle filter and WMMA benchmark.

## Folder structure

```
gpu_capstone_project/
  CMakeLists.txt
  README.md
  run_with_cuda.sh          (helper script to set CUDA lib path and run)
  include/
    wmma_gemm.h
    particle_filter_cuda.h
    detection_io.h
    yolo_detector.h
    pipeline.h
  src/
    main.cpp
    pipeline.cpp
    detection_io.cpp
    yolo_detector.cpp
  cuda/
    wmma_gemm.cu
    particle_filter.cu
```

## Coursera capstone (rubric)

- **Design**: Pipeline with detection (YOLO or CSV), observation map, particle filter (predict → weight → normalize → resample), and WMMA benchmark.
- **Implementation**: C/C++ and CUDA (particle filter + WMMA FP16 GEMM).
- **Optimization**: Tensor Cores in WMMA; coalesced memory and block reduction in the particle filter.
- **Evaluation**: Use `--bench-wmma` for GFLOPS; use `--detections` or `--yolo` with video to check tracking.
- **Documentation**: This README and comments in the code.

## Future improvements

Possible directions to extend the project:

- **Stronger detector**: Use a larger or more accurate YOLO (e.g. YOLOv8m/v8l, YOLOv9) or switch to TensorRT for faster and more stable inference on GPU.
- **Better tracking**: Replace or augment the particle filter with a Kalman filter, or adopt a tracking-by-detection method (e.g. SORT, DeepSORT) with optional re-identification for more stable IDs across occlusions.
- **Particle filter upgrades**: Use bilinear sampling when reading the observation map instead of nearest-neighbor; tune number of particles and noise per scene; add simple occlusion handling.
- **Performance**: Batch or overlap YOLO and particle filter on GPU; support multi-GPU for very long videos or high resolution.
- **Features**: Count people in a region, export trajectories to CSV, or add a simple activity cue (e.g. standing vs walking) from track velocity.

## License

For educational use (Coursera capstone).
