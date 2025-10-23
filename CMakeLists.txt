cmake_minimum_required(VERSION 3.8)
project(ACMMP LANGUAGES CXX CUDA)

# ─── CUDA Architecture Settings ─────────────────────────────────────
set(CMAKE_CUDA_ARCHITECTURES  86 89)

# ─── C++ Standard ───────────────────────────────────────────────────
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ─── Find Packages ──────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenMP)

# ─── Include Directories ────────────────────────────────────────────
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(.)

# ─── Compiler Flags ─────────────────────────────────────────────────
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math --maxrregcount=128")

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -g -G")
endif()

# ─── Source Files ───────────────────────────────────────────────────
set(CUDA_SOURCES
    ACMMP.cu
    FusionGPU.cu
    SphericalLUT_MultiRes.cu
    BatchACMMP.cu
    precompute.cu  
)

set(CXX_SOURCES
    main.cpp
    ACMMP.cpp
    
)

set(HEADERS
    main.h
    ACMMP.h
    ACMMP_device.cuh
    FusionGPU.h
    SphericalLUT_MultiRes.h
    BatchACMMP.h
)

# ─── Create Executable ──────────────────────────────────────────────
add_executable(ACMMP 
    ${CXX_SOURCES}
    ${CUDA_SOURCES}
    ${HEADERS}
)

# ─── Set Properties ─────────────────────────────────────────────────
set_property(TARGET ACMMP PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

# ─── Link Libraries ─────────────────────────────────────────────────
target_link_libraries(ACMMP 
    ${OpenCV_LIBS}
    Eigen3::Eigen
    CUDA::cudart
    CUDA::curand
)

if(OpenMP_CXX_FOUND)
    target_link_libraries(ACMMP OpenMP::OpenMP_CXX)
endif()

# ─── Print Configuration ────────────────────────────────────────────
message(STATUS "CUDA Version: ${CUDAToolkit_VERSION}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "OpenCV Version: ${OpenCV_VERSION}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")