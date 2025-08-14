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

// Kernel to initialize lookup tables on GPU
__global__ void InitializeLUTKernel(float3* dir_vectors, 
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
    
    // Compute longitude and latitude for this pixel
    float lon = ((float)x - cx) / (float)width * 2.0f * CUDART_PI_F;
    float lat = -((float)y - cy) / (float)height * CUDART_PI_F;
    
    // Store raw values
    lon_values[idx] = lon;
    lat_values[idx] = lat;
    
    // Compute and store trigonometric values
    float sin_lat_val, cos_lat_val, sin_lon_val, cos_lon_val;
    __sincosf(lat, &sin_lat_val, &cos_lat_val);
    __sincosf(lon, &sin_lon_val, &cos_lon_val);
    
    sin_lat[idx] = sin_lat_val;
    cos_lat[idx] = cos_lat_val;
    sin_lon[idx] = sin_lon_val;
    cos_lon[idx] = cos_lon_val;
    
    // Compute and store direction vector
    dir_vectors[idx] = make_float3(
        cos_lat_val * sin_lon_val,
        -sin_lat_val,
        cos_lat_val * cos_lon_val
    );
}

// Constructor
SphericalLUTManager::SphericalLUTManager() : d_lut_array(nullptr), d_lut_count(nullptr) {
    // Allocate device array for LUT pointers
    cudaMalloc(&d_lut_array, MAX_RESOLUTIONS * sizeof(SphericalLUT*));
    cudaMalloc(&d_lut_count, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_lut_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

// Destructor
SphericalLUTManager::~SphericalLUTManager() {
    FreeAllLUTs();
    if (d_lut_array) cudaFree(d_lut_array);
    if (d_lut_count) cudaFree(d_lut_count);
}

// Create a new LUT for specific resolution
SphericalLUT* SphericalLUTManager::CreateLUT(int width, int height, float cx, float cy) {
    SphericalLUT* lut = new SphericalLUT;
    lut->width = width;
    lut->height = height;
    lut->cx = cx;
    lut->cy = cy;
    
    size_t pixel_count = width * height;
    size_t float_size = pixel_count * sizeof(float);
    size_t float3_size = pixel_count * sizeof(float3);
    
    // Allocate device memory
    cudaMalloc(&lut->d_dir_vectors, float3_size);
    cudaMalloc(&lut->d_sin_lat, float_size);
    cudaMalloc(&lut->d_cos_lat, float_size);
    cudaMalloc(&lut->d_sin_lon, float_size);
    cudaMalloc(&lut->d_cos_lon, float_size);
    cudaMalloc(&lut->d_lon_values, float_size);
    cudaMalloc(&lut->d_lat_values, float_size);
    
    // Calculate memory size
    lut->memory_size = float3_size + 6 * float_size;
    
    // Initialize on GPU
    dim3 block(16, 16);
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
    
    std::cout << "Created LUT for resolution " << width << "x" << height 
              << " with center (" << cx << ", " << cy << ")"
              << " - Memory: " << lut->memory_size / (1024.0 * 1024.0) << " MB" << std::endl;
    
    return lut;
}

// Free a LUT
void SphericalLUTManager::FreeLUT(SphericalLUT* lut) {
    if (!lut) return;
    
    cudaFree(lut->d_dir_vectors);
    cudaFree(lut->d_sin_lat);
    cudaFree(lut->d_cos_lat);
    cudaFree(lut->d_sin_lon);
    cudaFree(lut->d_cos_lon);
    cudaFree(lut->d_lon_values);
    cudaFree(lut->d_lat_values);
    
    delete lut;
}

// Get or create LUT for specific resolution
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

// Find closest LUT if exact match not found
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

// Initialize LUTs for multiple scales
void SphericalLUTManager::InitializeMultiScaleLUTs(int base_width, int base_height, 
                                                   float base_cx, float base_cy, 
                                                   int num_scales) {
    std::cout << "Initializing " << num_scales << " scale LUTs..." << std::endl;
    
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

// Update device array of LUT pointers
void SphericalLUTManager::UpdateDeviceArray() {
    if (device_luts.empty()) return;
    
    // Create host array of device LUT pointers
    SphericalLUT** h_lut_array = new SphericalLUT*[device_luts.size()];
    for (size_t i = 0; i < device_luts.size(); ++i) {
        h_lut_array[i] = device_luts[i];
    }
    
    // Copy to device
    cudaMemcpy(d_lut_array, h_lut_array, device_luts.size() * sizeof(SphericalLUT*), 
               cudaMemcpyHostToDevice);
    
    // Update count
    int count = device_luts.size();
    cudaMemcpy(d_lut_count, &count, sizeof(int), cudaMemcpyHostToDevice);
    
    // Also update constant memory if size permits
    if (device_luts.size() <= 10) {
        cudaMemcpyToSymbol(d_lut_array_const, h_lut_array, 
                          device_luts.size() * sizeof(SphericalLUT*));
        cudaMemcpyToSymbol(d_num_luts, &count, sizeof(int));
    }
    
    delete[] h_lut_array;
}

// Get LUT by index
SphericalLUT* SphericalLUTManager::GetLUTByIndex(int idx) {
    if (idx >= 0 && idx < (int)device_luts.size()) {
        return device_luts[idx];
    }
    return nullptr;
}

// Free all LUTs
void SphericalLUTManager::FreeAllLUTs() {
    for (auto& pair : lut_map) {
        FreeLUT(pair.second);
    }
    lut_map.clear();
    device_luts.clear();
}

// Get total memory usage
size_t SphericalLUTManager::GetTotalMemoryUsage() const {
    size_t total = 0;
    for (const auto& pair : lut_map) {
        total += pair.second->memory_size;
    }
    return total;
}

// Global functions
void InitializeLUTManager() {
    if (g_lut_manager == nullptr) {
        g_lut_manager = new SphericalLUTManager();
        std::cout << "LUT Manager initialized" << std::endl;
    }
}

void FreeLUTManager() {
    if (g_lut_manager != nullptr) {
        delete g_lut_manager;
        g_lut_manager = nullptr;
        std::cout << "LUT Manager freed" << std::endl;
    }
}