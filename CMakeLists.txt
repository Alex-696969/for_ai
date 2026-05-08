cmake_minimum_required(VERSION 3.16)

project(resnet50_tensorrt_cpp_demo)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 请根据自己的实际安装路径修改
set(TENSORRT_ROOT "D:/MySoftWare/TensorRT/TensorRT-10.12.0.36")
set(OpenCV_DIR "C:/opencv/install/x64/vc17/lib")
set(CUDA_ROOT "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.9/")

find_package(OpenCV REQUIRED)

include_directories(
    ${TENSORRT_ROOT}/include
    ${CUDA_ROOT}/include
    ${OpenCV_INCLUDE_DIRS}
)

link_directories(
    ${TENSORRT_ROOT}/lib
    ${CUDA_ROOT}/lib/x64
)

add_executable(
    infer_resnet50_trt_cpp
    infer_resnet50_trt_cpp.cpp
)

target_link_libraries(
    infer_resnet50_trt_cpp
    PRIVATE
    nvinfer_10
    cudart
    ${OpenCV_LIBS}
)