cmake_minimum_required(VERSION 3.18)
project(PedestrianTrackingCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# RTX A2000: compute capability 8.6 (Ampere)
set(CMAKE_CUDA_ARCHITECTURES "86")

# FP16 and Tensor Core (WMMA) support
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -use_fast_math")

find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)

# Headers and sources
set(PROJECT_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)
set(PROJECT_SOURCES
  src/main.cpp
  src/pipeline.cpp
  src/detection_io.cpp
  src/yolo_detector.cpp
)
set(PROJECT_CUDA_SOURCES
  cuda/wmma_gemm.cu
  cuda/particle_filter.cu
)

add_executable(${PROJECT_NAME} ${PROJECT_SOURCES} ${PROJECT_CUDA_SOURCES})
target_include_directories(${PROJECT_NAME} PRIVATE ${PROJECT_INCLUDE_DIR} ${CUDAToolkit_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE CUDA::cudart)

if(OpenCV_FOUND)
  target_include_directories(${PROJECT_NAME} PRIVATE ${OpenCV_INCLUDE_DIRS})
  target_link_libraries(${PROJECT_NAME} PRIVATE ${OpenCV_LIBS})
  target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_OPENCV=1)
else()
  target_compile_definitions(${PROJECT_NAME} PRIVATE HAVE_OPENCV=0)
endif()

# Allow WMMA (Tensor Cores) - requires sm_70+
set_target_properties(${PROJECT_NAME} PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
)
