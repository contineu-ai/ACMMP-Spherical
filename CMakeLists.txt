cmake_minimum_required(VERSION 2.8)
project(ACMMP LANGUAGES CXX CUDA)

# ─── Suppress FindCUDA deprecation warning ───────────────────────────
if(POLICY CMP0146)
  cmake_policy(SET CMP0146 OLD)
endif()

# ─── Dependencies ────────────────────────────────────────────────────
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda-12.6")
find_package(CUDA 12.6 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

find_package(OpenMP)
if(OPENMP_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ─── Includes ────────────────────────────────────────────────────────
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(.)

# ─── NVCC Flags ──────────────────────────────────────────────────────
list(APPEND CUDA_NVCC_FLAGS
     -O3 --use_fast_math --maxrregcount=128 --ptxas-options=-v -std=c++11 --compiler-options -Wall
     -gencode arch=compute_52,code=sm_52
     -gencode arch=compute_86,code=sm_86
     -gencode arch=compute_89,code=sm_89)

# ─── Host Compiler Flags ─────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX)
  add_definitions(-std=c++11 -pthread -Wall -Wextra -pedantic -Wno-unused-function -Wno-switch)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()

# ─── Build the Executable ────────────────────────────────────────────
cuda_add_executable(ACMMP
  main.h
  ACMMP.h
  ACMMP.cpp
  ACMMP.cu
  main.cpp
)

# ─── Link Libraries (plain signature!) ──────────────────────────────
target_link_libraries(ACMMP
  ${OpenCV_LIBS}
  Eigen3::Eigen
  ${CUDA_LIBRARIES}
)

if(OPENMP_FOUND)
  target_link_libraries(ACMMP ${OpenMP_CXX_FLAGS})
endif()
