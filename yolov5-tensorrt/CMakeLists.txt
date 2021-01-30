cmake_minimum_required(VERSION 3.10)

project(yolov5-tensorrt)

add_definitions(-std=c++11)

option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_BUILD_TYPE Debug)

include_directories(${PROJECT_SOURCE_DIR}/include)

# include and link dirs of cuda and tensorrt, you need adapt them if yours are different
# cuda
include_directories("/usr/local/cuda-10.2/include")
link_directories("/usr/local/cuda-10.2/lib64")
find_package(CUDA REQUIRED)
# tensorrt
include_directories("/home/deploy/software/TensorRT-7.0.0.11/include") # h文件
link_directories("/home/deploy/software/TensorRT-7.0.0.11/lib") # so文件

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -Wall -Ofast -Wfatal-errors -D_MWAITXINTRIN_H_INCLUDED")

cuda_add_library(myplugins SHARED ${PROJECT_SOURCE_DIR}/yololayer.cu)
target_link_libraries(myplugins nvinfer cudart)

# opencv
set(OpenCV_DIR "/home/deploy/software/opencv-4.5.1/build/")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(${CMAKE_PROJECT_NAME} ${PROJECT_SOURCE_DIR}/calibrator.cpp ${PROJECT_SOURCE_DIR}/yolov5.cpp)
target_link_libraries(${CMAKE_PROJECT_NAME} nvinfer cudart myplugins ${OpenCV_LIBS})

add_definitions(-O2 -pthread)

# 重新build执行
# TARGET：定义了与构建指定相关联的新命令。当已经存在是，相应的command将不再执行。
# PRE_BUILD：在目标中执行任何其他规则之前运行。
# POST_BUILD：在目标中的所有其他规则都已执行后运行。
add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
  		           PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/weights ${CMAKE_CURRENT_BINARY_DIR}/weights)