#!/usr/bin/env bash
# Set CUDA 12 library path for OpenCV DNN (libcublasLt.so.12, etc.).
# Use: ./run_with_cuda.sh --video input.mp4 --yolo build/yolov8n.onnx

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUDA_PATHS=""
for cuda in /usr/local/cuda-12.2 /usr/local/cuda-12.1 /usr/local/cuda-12 /usr/local/cuda; do
  if [ -d "$cuda/lib64" ]; then
    CUDA_PATHS="$cuda/lib64${CUDA_PATHS:+:$CUDA_PATHS}"
    break
  fi
done
if [ -n "$CUDA_PATHS" ]; then
  export LD_LIBRARY_PATH="$CUDA_PATHS${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi
exec "$SCRIPT_DIR/build/PedestrianTrackingCUDA" "$@"
