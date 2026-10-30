cmake_minimum_required(VERSION 3.24)
project(inference LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT CMAKE_CUDA_ARCHITECTURES)
  message(FATAL_ERROR "Set -DCMAKE_CUDA_ARCHITECTURES to the detected GPU architecture")
endif()

add_library(inference SHARED src/kernels.cu src/engine.cu)
target_include_directories(inference PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
