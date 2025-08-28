#include "SphericalLUT_MultiRes.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <cmath>
#define FLT_MAX 100

// Global instance
SphericalLUTManager* g_lut_manager = nullptr;

// Device constant memory for quick LUT access
__constant__ SphericalLUT* d_lut_array_const[10];  // Support up to 10 resolutions
__constant__ int d_num_luts;

// Optimized kernel with reduced register usage and better memory coalescing
__device__ void InitializeLUTKernel_Optimized(float3* dir_vectors, 
                                              float* sin_lat, float* cos_lat,
                                              float* sin_lon, float* cos_lon,
                                              float* lon_values, float* lat_values,
                                              int width, int height, 
                                              float cx, float cy)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int idx = y * width + x;
    
    // Use more precise constants and faster operations
    const float inv_width = __fdividef(1.0f, static_cast<float>(width));
    const float inv_height = __fdividef(1.0f, static_cast<float>(height));
    
    // Compute longitude and latitude for this pixel
    float lon = ((static_cast<float>(x) - cx) * inv_width) * 2.0f * CUDART_PI_F;
    float lat = -((static_cast<float>(y) - cy) * inv_height) * CUDART_PI_F;
    
    // Store raw values
    lon_values[idx] = lon;
    lat_values[idx] = lat;
    
    // Compute trigonometric values using single call
    float sin_lat_val, cos_lat_val, sin_lon_val, cos_lon_val;
    __sincosf(lat, &sin_lat_val, &cos_lat_val);
    __sincosf(lon, &sin_lon_val, &cos_lon_val);
    
    sin_lat[idx] = sin_lat_val;
    cos_lat[idx] = cos_lat_val;
    sin_lon[idx] = sin_lon_val;
    cos_lon[idx] = cos_lon_val;
    
    // Compute direction vector with FMA operations
    dir_vectors[idx] = make_float3(
        __fmul_rn(cos_lat_val, sin_lon_val),
        -sin_lat_val,
        __fmul_rn(cos_lat_val, cos_lon_val)
    );
}

// Original kernel wrapper that calls optimized version
__global__ void InitializeLUTKernel(float3* dir_vectors, 
                                    float* sin_lat, float* cos_lat,
                                    float* sin_lon, float* cos_lon,
                                    float* lon_values, float* lat_values,
                                    int width, int height, 
                                    float cx, float cy)
{
    InitializeLUTKernel_Optimized(dir_vectors, sin_lat, cos_lat, sin_lon, cos_lon,
                                  lon_values, lat_values, width, height, cx, cy);
}

// Constructor with optimizations
SphericalLUTManager::SphericalLUTManager() : d_lut_array(nullptr), d_lut_count(nullptr) {
    // Allocate device array for LUT pointers
    cudaMalloc(&d_lut_array, MAX_RESOLUTIONS * sizeof(SphericalLUT*));
    cudaMalloc(&d_lut_count, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_lut_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    // Pre-allocate memory pool for frequent allocations (optional optimization)
    memory_pool_size = 256 * 1024 * 1024; // 256MB pool
    cudaMalloc(&memory_pool, memory_pool_size);
    memory_pool_offset = 0;
}

// Destructor with proper texture cleanup
SphericalLUTManager::~SphericalLUTManager() {
    FreeAllLUTs();
    if (d_lut_array) cudaFree(d_lut_array);
    if (d_lut_count) cudaFree(d_lut_count);
    if (memory_pool) cudaFree(memory_pool);
}

// Enhanced LUT creation with memory pool optimization
SphericalLUT* SphericalLUTManager::CreateLUT(int width, int height, float cx, float cy) {
    SphericalLUT* lut = new SphericalLUT;
    lut->width = width;
    lut->height = height;
    lut->cx = cx;
    lut->cy = cy;
    
    size_t pixel_count = width * height;
    size_t float_size = pixel_count * sizeof(float);
    size_t float3_size = pixel_count * sizeof(float3);
    
    // Try to use memory pool for better performance
    size_t total_size = float3_size + 6 * float_size;
    bool use_pool = (memory_pool_offset + total_size <= memory_pool_size);
    
    if (use_pool) {
        // Use memory pool
        char* pool_ptr = static_cast<char*>(memory_pool) + memory_pool_offset;
        lut->d_dir_vectors = reinterpret_cast<float3*>(pool_ptr);
        lut->d_sin_lat = reinterpret_cast<float*>(pool_ptr + float3_size);
        lut->d_cos_lat = reinterpret_cast<float*>(pool_ptr + float3_size + float_size);
        lut->d_sin_lon = reinterpret_cast<float*>(pool_ptr + float3_size + 2 * float_size);
        lut->d_cos_lon = reinterpret_cast<float*>(pool_ptr + float3_size + 3 * float_size);
        lut->d_lon_values = reinterpret_cast<float*>(pool_ptr + float3_size + 4 * float_size);
        lut->d_lat_values = reinterpret_cast<float*>(pool_ptr + float3_size + 5 * float_size);
        
        memory_pool_offset += total_size;
        lut->uses_pool = true;
    } else {
        // Allocate individual memory blocks
        cudaMalloc(&lut->d_dir_vectors, float3_size);
        cudaMalloc(&lut->d_sin_lat, float_size);
        cudaMalloc(&lut->d_cos_lat, float_size);
        cudaMalloc(&lut->d_sin_lon, float_size);
        cudaMalloc(&lut->d_cos_lon, float_size);
        cudaMalloc(&lut->d_lon_values, float_size);
        cudaMalloc(&lut->d_lat_values, float_size);
        lut->uses_pool = false;
    }
    
    // Calculate memory size
    lut->memory_size = total_size;
    
    // Use optimized grid dimensions for better occupancy
    dim3 block(32, 8);  // Better memory coalescing
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    // Initialize on GPU with optimized kernel
    InitializeLUTKernel<<<grid, block>>>(
        lut->d_dir_vectors,
        lut->d_sin_lat,
        lut->d_cos_lat,
        lut->d_sin_lon,
        lut->d_cos_lon,
        lut->d_lon_values,
        lut->d_lat_values,
        width, height, cx, cy
    );
    
    cudaDeviceSynchronize();
    
    std::cout << "Created optimized LUT for resolution " << width << "x" << height 
              << " with center (" << cx << ", " << cy << ")"
              << " - Memory: " << lut->memory_size / (1024.0 * 1024.0) << " MB";
    
    if (use_pool) {
        std::cout << " (using memory pool)";
    }
    std::cout << std::endl;
    
    return lut;
}

// Enhanced LUT freeing with memory pool awareness
void SphericalLUTManager::FreeLUT(SphericalLUT* lut) {
    if (!lut) return;
    
    // Free memory (skip if using pool)
    if (!lut->uses_pool) {
        cudaFree(lut->d_dir_vectors);
        cudaFree(lut->d_sin_lat);
        cudaFree(lut->d_cos_lat);
        cudaFree(lut->d_sin_lon);
        cudaFree(lut->d_cos_lon);
        cudaFree(lut->d_lon_values);
        cudaFree(lut->d_lat_values);
    }
    
    delete lut;
}

// Enhanced get-or-create with memory pool optimization
SphericalLUT* SphericalLUTManager::GetOrCreateLUT(int width, int height, float cx, float cy) {
    ResolutionKey key = {width, height, cx, cy};
    
    // Check if LUT already exists
    auto it = lut_map.find(key);
    if (it != lut_map.end()) {
        return it->second;
    }
    
    // Check if we've reached max resolutions
    if (lut_map.size() >= MAX_RESOLUTIONS) {
        std::cerr << "Warning: Maximum number of LUT resolutions reached. Using closest match." << std::endl;
        return FindClosestLUT(width, height, cx, cy);
    }
    
    // Create new LUT
    SphericalLUT* lut = CreateLUT(width, height, cx, cy);
    lut_map[key] = lut;
    device_luts.push_back(lut);
    
    // Update device array
    UpdateDeviceArray();
    
    return lut;
}

// Original FindClosestLUT remains the same
SphericalLUT* SphericalLUTManager::FindClosestLUT(int width, int height, float cx, float cy) {
    if (lut_map.empty()) return nullptr;
    
    SphericalLUT* closest = nullptr;
    float min_distance = FLT_MAX;
    
    for (const auto& pair : lut_map) {
        const ResolutionKey& key = pair.first;
        float distance = std::abs(key.width - width) + std::abs(key.height - height) +
                        std::abs(key.cx - cx) + std::abs(key.cy - cy);
        
        if (distance < min_distance) {
            min_distance = distance;
            closest = pair.second;
        }
    }
    
    if (closest) {
        std::cout << "Using closest LUT: " << closest->width << "x" << closest->height 
                  << " for requested " << width << "x" << height << std::endl;
    }
    
    return closest;
}

// Enhanced multi-scale initialization with memory pool optimization
void SphericalLUTManager::InitializeMultiScaleLUTs(int base_width, int base_height, 
                                                   float base_cx, float base_cy, 
                                                   int num_scales) {
    std::cout << "Initializing " << num_scales << " scale LUTs with memory pool optimization..." << std::endl;
    
    // Create LUTs in order of frequency (largest first for texture binding)
    for (int scale = 0; scale < num_scales; ++scale) {
        int scale_factor = 1 << scale;  // 2^scale
        int width = base_width / scale_factor;
        int height = base_height / scale_factor;
        float cx = base_cx / scale_factor;
        float cy = base_cy / scale_factor;
        
        // Skip if resolution becomes too small
        if (width < 32 || height < 32) break;
        
        GetOrCreateLUT(width, height, cx, cy);
    }
    
    std::cout << "Total LUTs created: " << lut_map.size() 
              << ", Memory usage: " << GetTotalMemoryUsage() / (1024.0 * 1024.0) << " MB" << std::endl;
}

// Optimized device array update with better memory patterns
void SphericalLUTManager::UpdateDeviceArray() {
    if (device_luts.empty()) return;
    
    // Create host array of device LUT pointers
    SphericalLUT** h_lut_array = new SphericalLUT*[device_luts.size()];
    for (size_t i = 0; i < device_luts.size(); ++i) {
        h_lut_array[i] = device_luts[i];
    }
    
    // Use asynchronous copy for better performance
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_lut_array, h_lut_array, device_luts.size() * sizeof(SphericalLUT*), 
                    cudaMemcpyHostToDevice, stream);
    
    // Update count
    int count = device_luts.size();
    cudaMemcpyAsync(d_lut_count, &count, sizeof(int), cudaMemcpyHostToDevice, stream);
    
    // Also update constant memory if size permits
    if (device_luts.size() <= 10) {
        cudaMemcpyToSymbolAsync(d_lut_array_const, h_lut_array, 
                               device_luts.size() * sizeof(SphericalLUT*), 0, 
                               cudaMemcpyHostToDevice, stream);
        cudaMemcpyToSymbolAsync(d_num_luts, &count, sizeof(int), 0,
                               cudaMemcpyHostToDevice, stream);
    }
    
    cudaStreamSynchronize(stream);
    cudaStreamDestroy(stream);
    
    delete[] h_lut_array;
}

// Original GetLUTByIndex remains the same
SphericalLUT* SphericalLUTManager::GetLUTByIndex(int idx) {
    if (idx >= 0 && idx < (int)device_luts.size()) {
        return device_luts[idx];
    }
    return nullptr;
}

// Enhanced cleanup with memory pool reset
void SphericalLUTManager::FreeAllLUTs() {
    for (auto& pair : lut_map) {
        FreeLUT(pair.second);
    }
    lut_map.clear();
    device_luts.clear();
    
    // Reset memory pool
    memory_pool_offset = 0;
}

// Original GetTotalMemoryUsage remains the same
size_t SphericalLUTManager::GetTotalMemoryUsage() const {
    size_t total = 0;
    for (const auto& pair : lut_map) {
        total += pair.second->memory_size;
    }
    return total;
}

// Global functions remain the same
void InitializeLUTManager() {
    if (g_lut_manager == nullptr) {
        g_lut_manager = new SphericalLUTManager();
        std::cout << "Optimized LUT Manager initialized" << std::endl;
    }
}

void FreeLUTManager() {
    if (g_lut_manager != nullptr) {
        delete g_lut_manager;
        g_lut_manager = nullptr;
        std::cout << "Optimized LUT Manager freed" << std::endl;
    }
}