﻿# CMakeList.txt : CMake project for libtorch-yolov5, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.10)

project ("libtorch-yolov5-openvino")

#set(CMAKE_BUILD_TYPE "release")

#openvino OpenVINO推理引擎的头文件
set(InferenceEngine_DIR "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\inference_engine\\share")
set(ngraph_DIR "C:\\Program Files (x86)\\Intel\\openvino_2021.2.185\\deployment_tools\\ngraph\\cmake")
find_package(ngraph REQUIRED)
find_package(InferenceEngine 2.1 REQUIRED)

# opencv
set(OpenCV_DIR "C:\\Program Files\\opencv\\build")
find_package(OpenCV REQUIRED)

# libtorch
set(Torch_DIR "C:\\Users\\zengxh\\Documents\\software-install\\libtorch-cpu-debug")
find_package(Torch PATHS ${Torch_DIR} NO_DEFAULT REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
if (Torch_FOUND)
    message(STATUS "Torch library found!")
    message(STATUS "    include path: ${TORCH_INCLUDE_DIRS}" \n)
else ()
    message(FATAL_ERROR "Could not locate Torch" \n)
endif()

include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCE_FILES src/*.cpp)

# Add source to this project's executable.
add_executable (${CMAKE_PROJECT_NAME} ${SOURCE_FILES})

#target_include_directories(${CMAKE_PROJECT_NAME} PUBLIC ${InferenceEngine_INCLUDE_DIRS})

# TODO: Add tests and install targets if needed.
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES} ${InferenceEngine_LIBRARIES} ${ngraph_LIBRARIES})

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:${CMAKE_PROJECT_NAME}>)
endif (MSVC)

# 重新build执行
# TARGET：定义了与构建指定相关联的新命令。当已经存在是，相应的command将不再执行。
# PRE_BUILD：在目标中执行任何其他规则之前运行。
# POST_BUILD：在目标中的所有其他规则都已执行后运行。
add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
  		           PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/weights ${CMAKE_CURRENT_BINARY_DIR}/weights)
add_custom_command(TARGET ${CMAKE_PROJECT_NAME}
  		           PRE_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_directory ${CMAKE_CURRENT_SOURCE_DIR}/images ${CMAKE_CURRENT_BINARY_DIR}/images)