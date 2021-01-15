# CMakeList.txt : CMake project for libtorch-yolov5, include source and define
# project specific logic here.
#
cmake_minimum_required (VERSION 3.8)

project ("libtorch-yolov5")

set(CMAKE_BUILD_TYPE "Release")

# opencv
set(OpenCV_DIR "C:\\Program Files\\opencv\\build")
find_package(OpenCV REQUIRED)

# libtorch
set(Torch_DIR "C:\\Users\\zengxh\\Documents\\software-install\\libtorch-cpu")
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

# TODO: Add tests and install targets if needed.
target_link_libraries(${CMAKE_PROJECT_NAME} ${OpenCV_LIBS} ${TORCH_LIBRARIES})

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

