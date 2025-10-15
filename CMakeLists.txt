cmake_minimum_required(VERSION 3.18)
project(ACMMP LANGUAGES CXX CUDA)

# ─── Dependencies ────────────────────────────────────────────────────
find_package(CUDAToolkit 12.6 REQUIRED)
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

# ─── CUDA Settings ───────────────────────────────────────────────────
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_ARCHITECTURES "52;86;89")

# ─── Host Compiler Flags ─────────────────────────────────────────────
if(CMAKE_COMPILER_IS_GNUCXX)
    add_definitions(-std=c++11 -pthread -Wall -Wextra -pedantic -Wno-unused-function -Wno-switch)
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_C_FLAGS_RELEASE} -O3 -ffast-math -march=native")
endif()

# ─── Build the Executable ────────────────────────────────────────────
add_executable(ACMMP
    main.h
    ACMMP.h
    ACMMP.cpp
    ACMMP.cu
    main.cpp
)

# Set CUDA flags
target_compile_options(ACMMP PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        -O3
        --use_fast_math
        --maxrregcount=128
        --ptxas-options=-v
    >
)

# ─── Link Libraries ──────────────────────────────────────────────────
target_link_libraries(ACMMP
    ${OpenCV_LIBS}
    Eigen3::Eigen
    CUDA::cudart
)

if(OPENMP_FOUND)
    target_link_libraries(ACMMP OpenMP::OpenMP_CXX)
endif()