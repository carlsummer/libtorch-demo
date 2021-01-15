# CMakeList.txt: Cuda-libtorch 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("Cuda-libtorch")

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH "C:\\Users\\zengxh\\Documents\\software-install\\libtorch-cpu")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# 将源代码添加到此项目的可执行文件。
add_executable (Cuda-libtorch "Cuda-libtorch.cpp" "Cuda-libtorch.h")

# TODO: 如有需要，请添加测试并安装目标。

target_link_libraries(Cuda-libtorch "${TORCH_LIBRARIES}")
set_property(TARGET Cuda-libtorch PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET Cuda-libtorch
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:Cuda-libtorch>)
endif (MSVC)