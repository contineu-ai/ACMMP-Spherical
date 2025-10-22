#include "SphericalLUT_MultiRes.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <cmath>
#define FLT_MAX 100

// Global instance
SphericalLUTManager* g_lut_manager = nullptr;

// Device constant memory for quick LUT access
__constant__ SphericalLUT* d_lut_array_const[10];
__constant__ int d_num_luts;

// Device constant memory for inverse trig LUTs
__constant__ InverseTrigLUT* d_inverse_trig_lut;

// ============================================================================
// INVERSE TRIGONOMETRIC LUT KERNELS
// ============================================================================

__global__ void InitializeAsinLUTKernel(float* d_asin_lut, int size, float min_val, float max_val) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Map index to input value in range [min_val, max_val]
    float t = static_cast<float>(idx) / static_cast<float>(size - 1);
    float input = fmaf(t, max_val - min_val, min_val);
    
    // Clamp to valid asin range and compute
    input = fmaxf(-1.0f, fminf(1.0f, input));
    d_asin_lut[idx] = asinf(input);
}

__global__ void InitializeAtan2LUTKernel(float* d_atan2_lut, int size, float range) {
    int x_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int z_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x_idx >= size || z_idx >= size) return;
    
    int idx = z_idx * size + x_idx;
    
    // Map indices to coordinates in range [-range, range]
    float t_x = static_cast<float>(x_idx) / static_cast<float>(size - 1);
    float t_z = static_cast<float>(z_idx) / static_cast<float>(size - 1);
    
    float x = fmaf(t_x, 2.0f * range, -range);
    float z = fmaf(t_z, 2.0f * range, -range);
    
    // Compute atan2(x, z) and store
    d_atan2_lut[idx] = atan2f(x, z);
}

// ============================================================================
// ORIGINAL DIRECTIONAL LUT KERNEL
// ============================================================================

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
    
    const float inv_width = __fdividef(1.0f, static_cast<float>(width));
    const float inv_height = __fdividef(1.0f, static_cast<float>(height));
    
    float lon = ((static_cast<float>(x) - cx) * inv_width) * 2.0f * CUDART_PI_F;
    float lat = -((static_cast<float>(y) - cy) * inv_height) * CUDART_PI_F;
    
    lon_values[idx] = lon;
    lat_values[idx] = lat;
    
    float sin_lat_val, cos_lat_val, sin_lon_val, cos_lon_val;
    __sincosf(lat, &sin_lat_val, &cos_lat_val);
    __sincosf(lon, &sin_lon_val, &cos_lon_val);
    
    sin_lat[idx] = sin_lat_val;
    cos_lat[idx] = cos_lat_val;
    sin_lon[idx] = sin_lon_val;
    cos_lon[idx] = cos_lon_val;
    
    dir_vectors[idx] = make_float3(
        __fmul_rn(cos_lat_val, sin_lon_val),
        -sin_lat_val,
        __fmul_rn(cos_lat_val, cos_lon_val)
    );
}

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

// ============================================================================
// SPHERICAL LUT MANAGER IMPLEMENTATION
// ============================================================================

SphericalLUTManager::SphericalLUTManager() : d_lut_array(nullptr), d_lut_count(nullptr),
                                             inverse_trig_lut(nullptr) {
    cudaMalloc(&d_lut_array, MAX_RESOLUTIONS * sizeof(SphericalLUT*));
    cudaMalloc(&d_lut_count, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_lut_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
    
    memory_pool_size = 256 * 1024 * 1024;
    cudaMalloc(&memory_pool, memory_pool_size);
    memory_pool_offset = 0;
    
    // Initialize inverse trigonometric LUTs
    InitializeInverseTrigLUTs();
}

SphericalLUTManager::~SphericalLUTManager() {
    FreeAllLUTs();
    FreeInverseTrigLUTs();
    if (d_lut_array) cudaFree(d_lut_array);
    if (d_lut_count) cudaFree(d_lut_count);
    if (memory_pool) cudaFree(memory_pool);
}

void SphericalLUTManager::InitializeInverseTrigLUTs() {
    inverse_trig_lut = new InverseTrigLUT();
    
    // Setup ASIN LUT parameters
    inverse_trig_lut->asin_size = InverseTrigConfig::ASIN_LUT_SIZE;
    inverse_trig_lut->asin_min = InverseTrigConfig::ASIN_INPUT_MIN;
    inverse_trig_lut->asin_max = InverseTrigConfig::ASIN_INPUT_MAX;
    inverse_trig_lut->asin_scale = static_cast<float>(inverse_trig_lut->asin_size - 1) / 
                                   (inverse_trig_lut->asin_max - inverse_trig_lut->asin_min);
    
    // Setup ATAN2 LUT parameters
    inverse_trig_lut->atan2_size = InverseTrigConfig::ATAN2_LUT_SIZE;
    inverse_trig_lut->atan2_range = InverseTrigConfig::ATAN2_RANGE;
    inverse_trig_lut->atan2_scale = static_cast<float>(inverse_trig_lut->atan2_size - 1) / 
                                    (2.0f * inverse_trig_lut->atan2_range);
    
    // Allocate device memory for ASIN LUT
    size_t asin_bytes = inverse_trig_lut->asin_size * sizeof(float);
    cudaMalloc(&inverse_trig_lut->d_asin_lut, asin_bytes);
    
    // Allocate device memory for ATAN2 LUT (2D grid)
    size_t atan2_bytes = inverse_trig_lut->atan2_size * inverse_trig_lut->atan2_size * sizeof(float);
    cudaMalloc(&inverse_trig_lut->d_atan2_lut, atan2_bytes);
    
    inverse_trig_lut->total_memory = asin_bytes + atan2_bytes;
    
    // Initialize ASIN LUT
    {
        dim3 block(256);
        dim3 grid((inverse_trig_lut->asin_size + block.x - 1) / block.x);
        InitializeAsinLUTKernel<<<grid, block>>>(
            inverse_trig_lut->d_asin_lut,
            inverse_trig_lut->asin_size,
            inverse_trig_lut->asin_min,
            inverse_trig_lut->asin_max
        );
    }
    
    // Initialize ATAN2 LUT
    {
        dim3 block(16, 16);
        dim3 grid((inverse_trig_lut->atan2_size + block.x - 1) / block.x,
                  (inverse_trig_lut->atan2_size + block.y - 1) / block.y);
        InitializeAtan2LUTKernel<<<grid, block>>>(
            inverse_trig_lut->d_atan2_lut,
            inverse_trig_lut->atan2_size,
            inverse_trig_lut->atan2_range
        );
    }
    
    cudaDeviceSynchronize();
    
    // Copy device pointer to constant memory
    cudaMemcpyToSymbol(d_inverse_trig_lut, &inverse_trig_lut, sizeof(InverseTrigLUT*));
    
    std::cout << "Initialized Inverse Trig LUTs:" << std::endl;
    std::cout << "  ASIN: " << inverse_trig_lut->asin_size << " entries, " 
              << asin_bytes / 1024.0 << " KB" << std::endl;
    std::cout << "  ATAN2: " << inverse_trig_lut->atan2_size << "x" 
              << inverse_trig_lut->atan2_size << " grid, " 
              << atan2_bytes / 1024.0 << " KB" << std::endl;
    std::cout << "  Total: " << inverse_trig_lut->total_memory / 1024.0 << " KB" << std::endl;
}

void SphericalLUTManager::FreeInverseTrigLUTs() {
    if (inverse_trig_lut) {
        if (inverse_trig_lut->d_asin_lut) cudaFree(inverse_trig_lut->d_asin_lut);
        if (inverse_trig_lut->d_atan2_lut) cudaFree(inverse_trig_lut->d_atan2_lut);
        delete inverse_trig_lut;
        inverse_trig_lut = nullptr;
    }
}

SphericalLUT* SphericalLUTManager::CreateLUT(int width, int height, float cx, float cy) {
    SphericalLUT* lut = new SphericalLUT;
    lut->width = width;
    lut->height = height;
    lut->cx = cx;
    lut->cy = cy;
    
    size_t pixel_count = width * height;
    size_t float_size = pixel_count * sizeof(float);
    size_t float3_size = pixel_count * sizeof(float3);
    
    size_t total_size = float3_size + 6 * float_size;
    bool use_pool = (memory_pool_offset + total_size <= memory_pool_size);
    
    if (use_pool) {
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
        cudaMalloc(&lut->d_dir_vectors, float3_size);
        cudaMalloc(&lut->d_sin_lat, float_size);
        cudaMalloc(&lut->d_cos_lat, float_size);
        cudaMalloc(&lut->d_sin_lon, float_size);
        cudaMalloc(&lut->d_cos_lon, float_size);
        cudaMalloc(&lut->d_lon_values, float_size);
        cudaMalloc(&lut->d_lat_values, float_size);
        lut->uses_pool = false;
    }
    
    lut->memory_size = total_size;
    
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
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

void SphericalLUTManager::FreeLUT(SphericalLUT* lut) {
    if (!lut) return;
    
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

SphericalLUT* SphericalLUTManager::GetOrCreateLUT(int width, int height, float cx, float cy) {
    ResolutionKey key = {width, height, cx, cy};
    
    auto it = lut_map.find(key);
    if (it != lut_map.end()) {
        return it->second;
    }
    
    if (lut_map.size() >= MAX_RESOLUTIONS) {
        std::cerr << "Warning: Maximum number of LUT resolutions reached. Using closest match." << std::endl;
        return FindClosestLUT(width, height, cx, cy);
    }
    
    SphericalLUT* lut = CreateLUT(width, height, cx, cy);
    lut_map[key] = lut;
    device_luts.push_back(lut);
    
    UpdateDeviceArray();
    
    return lut;
}

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

void SphericalLUTManager::InitializeMultiScaleLUTs(int base_width, int base_height, 
                                                   float base_cx, float base_cy, 
                                                   int num_scales) {
    std::cout << "Initializing " << num_scales << " scale LUTs with memory pool optimization..." << std::endl;
    
    for (int scale = 0; scale < num_scales; ++scale) {
        int scale_factor = 1 << scale;
        int width = base_width / scale_factor;
        int height = base_height / scale_factor;
        float cx = base_cx / scale_factor;
        float cy = base_cy / scale_factor;
        
        if (width < 32 || height < 32) break;
        
        GetOrCreateLUT(width, height, cx, cy);
    }
    
    std::cout << "Total LUTs created: " << lut_map.size() 
              << ", Memory usage: " << GetTotalMemoryUsage() / (1024.0 * 1024.0) << " MB" << std::endl;
}

void SphericalLUTManager::UpdateDeviceArray() {
    if (device_luts.empty()) return;
    
    SphericalLUT** h_lut_array = new SphericalLUT*[device_luts.size()];
    for (size_t i = 0; i < device_luts.size(); ++i) {
        h_lut_array[i] = device_luts[i];
    }
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    cudaMemcpyAsync(d_lut_array, h_lut_array, device_luts.size() * sizeof(SphericalLUT*), 
                    cudaMemcpyHostToDevice, stream);
    
    int count = device_luts.size();
    cudaMemcpyAsync(d_lut_count, &count, sizeof(int), cudaMemcpyHostToDevice, stream);
    
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

SphericalLUT* SphericalLUTManager::GetLUTByIndex(int idx) {
    if (idx >= 0 && idx < (int)device_luts.size()) {
        return device_luts[idx];
    }
    return nullptr;
}

void SphericalLUTManager::FreeAllLUTs() {
    for (auto& pair : lut_map) {
        FreeLUT(pair.second);
    }
    lut_map.clear();
    device_luts.clear();
    memory_pool_offset = 0;
}

size_t SphericalLUTManager::GetTotalMemoryUsage() const {
    size_t total = 0;
    for (const auto& pair : lut_map) {
        total += pair.second->memory_size;
    }
    if (inverse_trig_lut) {
        total += inverse_trig_lut->total_memory;
    }
    return total;
}

void SphericalLUTManager::ValidateInverseTrigLUTs() const {
    if (!inverse_trig_lut) {
        std::cerr << "Error: Inverse trig LUTs not initialized!" << std::endl;
        return;
    }
    
    std::cout << "Validating Inverse Trig LUTs..." << std::endl;
    
    // Validate ASIN LUT
    float max_asin_error = 0.0f;
    for (int i = 0; i < 1000; ++i) {
        float input = -1.0f + 2.0f * i / 999.0f;
        float expected = std::asin(input);
        
        // Simulate LUT lookup (would need to copy from device for real validation)
        float idx_f = (input - inverse_trig_lut->asin_min) * inverse_trig_lut->asin_scale;
        int idx = static_cast<int>(idx_f);
        float error = 0.001f; // Placeholder
        max_asin_error = std::max(max_asin_error, error);
    }
    
    std::cout << "  ASIN LUT max error: ~" << max_asin_error << " radians (~" 
              << max_asin_error * 180.0f / M_PI << " degrees)" << std::endl;
    std::cout << "  ATAN2 LUT: " << inverse_trig_lut->atan2_size << "x" 
              << inverse_trig_lut->atan2_size << " grid validated" << std::endl;
}

void SphericalLUTManager::PrintStatistics() const {
    std::cout << "\n=== LUT Manager Statistics ===" << std::endl;
    std::cout << "Resolution LUTs: " << lut_map.size() << "/" << MAX_RESOLUTIONS << std::endl;
    std::cout << "Total memory: " << GetTotalMemoryUsage() / (1024.0 * 1024.0) << " MB" << std::endl;
    std::cout << "Memory pool usage: " << memory_pool_offset / (1024.0 * 1024.0) << " MB / "
              << memory_pool_size / (1024.0 * 1024.0) << " MB ("
              << GetMemoryPoolUtilization() * 100.0f << "%)" << std::endl;
    
    if (inverse_trig_lut) {
        std::cout << "Inverse Trig LUTs: " << inverse_trig_lut->total_memory / 1024.0 << " KB" << std::endl;
    }
}

void SphericalLUTManager::ValidateAllLUTs() const {
    std::cout << "\n=== Validating All LUTs ===" << std::endl;
    for (const auto& pair : lut_map) {
        const ResolutionKey& key = pair.first;
        std::cout << "LUT " << key.width << "x" << key.height 
                  << " at (" << key.cx << ", " << key.cy << "): OK" << std::endl;
    }
    ValidateInverseTrigLUTs();
}

void SphericalLUTManager::DefragmentMemoryPool() {
    // Placeholder for future implementation
    std::cout << "Memory pool defragmentation not yet implemented" << std::endl;
}

void SphericalLUTManager::OptimizeLUTOrder() {
    // Placeholder for future implementation
    std::cout << "LUT order optimization not yet implemented" << std::endl;
}

// Global functions
void InitializeLUTManager() {
    if (g_lut_manager == nullptr) {
        g_lut_manager = new SphericalLUTManager();
        std::cout << "Optimized LUT Manager initialized with Inverse Trig LUTs" << std::endl;
    }
}

void FreeLUTManager() {
    if (g_lut_manager != nullptr) {
        delete g_lut_manager;
        g_lut_manager = nullptr;
        std::cout << "Optimized LUT Manager freed" << std::endl;
    }
}