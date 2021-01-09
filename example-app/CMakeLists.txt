# CMakeList.txt: torchdemo 的 CMake 项目，在此处包括源代码并定义
# 项目特定的逻辑。
#
cmake_minimum_required (VERSION 3.8)

project ("example-app")



set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_PREFIX_PATH "F:\\Program Files\\libtorch\\libtorch-gpu-release")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
# 将源代码添加到此项目的可执行文件。
add_executable (example-app "torchdemo.cpp" "torchdemo.h")

# TODO: 如有需要，请添加测试并安装目标。
target_link_libraries(example-app "${TORCH_LIBRARIES}")
set_property(TARGET example-app PROPERTY CXX_STANDARD 14)

# The following code block is suggested to be used on Windows.
# According to https://github.com/pytorch/pytorch/issues/25457,
# the DLLs need to be copied to avoid memory errors.
if (MSVC)
   file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
   add_custom_command(TARGET example-app
                      POST_BUILD
                      COMMAND ${CMAKE_COMMAND} -E copy_if_different
                      ${TORCH_DLLS}
                      $<TARGET_FILE_DIR:example-app>)
endif (MSVC)