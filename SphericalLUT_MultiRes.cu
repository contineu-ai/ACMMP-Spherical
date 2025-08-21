// SphericalLUT_MultiRes.cu - Enhanced version
#include "SphericalLUT_MultiRes.h"
#include <cuda_runtime.h>
#include <math_constants.h>
#include <iostream>
#include <cmath>
#define FLT_MAX 100
SphericalLUTManager* g_lut_manager = nullptr;

// Device constant memory for quick LUT access
__constant__ SphericalLUT* d_lut_array_const[20];
__constant__ int d_num_luts;


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

// Kernel to initialize enhanced lookup tables on GPU
__global__ void InitializeLUTKernel(
    float3* dir_vectors, 
    float* sin_lat, float* cos_lat,
    float* sin_lon, float* cos_lon,
    float* depth_multipliers,
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
    
    // Compute and store trigonometric values
    float sin_lat_val, cos_lat_val, sin_lon_val, cos_lon_val;
    __sincosf(lat, &sin_lat_val, &cos_lat_val);
    __sincosf(lon, &sin_lon_val, &cos_lon_val);
    
    sin_lat[idx] = sin_lat_val;
    cos_lat[idx] = cos_lat_val;
    sin_lon[idx] = sin_lon_val;
    cos_lon[idx] = cos_lon_val;
    
    // Compute and store direction vector
    float3 dir;
    dir.x = cos_lat_val * sin_lon_val;
    dir.y = -sin_lat_val;
    dir.z = cos_lat_val * cos_lon_val;
    dir_vectors[idx] = dir;
    
    // Store depth multiplier (inverse of direction magnitude for plane computations)
    float dir_mag = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
    depth_multipliers[idx] = (dir_mag > 1e-6f) ? (1.0f / dir_mag) : 0.0f;
}

// NEW: Kernel to initialize inverse projection table
__global__ void InitializeInverseLUTKernel(
    float* inv_x_table,
    float* inv_y_table,
    int inv_width, int inv_height,
    int img_width, int img_height,
    float cx, float cy)
{
    int lon_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int lat_idx = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (lon_idx >= inv_width || lat_idx >= inv_height) return;
    
    int idx = lat_idx * inv_width + lon_idx;
    
    // Map indices to angles
    float lon = (float)lon_idx / (float)inv_width * 2.0f * CUDART_PI_F - CUDART_PI_F;
    float lat = (float)lat_idx / (float)inv_height * CUDART_PI_F - CUDART_PI_F/2.0f;
    
    // Convert to pixel coordinates
    float x = (lon / (2.0f * CUDART_PI_F)) * (float)img_width + cx;
    float y = (-lat / CUDART_PI_F) * (float)img_height + cy;
    
    inv_x_table[idx] = x;
    inv_y_table[idx] = y;
}

// Create enhanced LUT
SphericalLUT* SphericalLUTManager::CreateLUT(int width, int height, float cx, float cy) {
    SphericalLUT* lut = new SphericalLUT;
    lut->width = width;
    lut->height = height;
    lut->cx = cx;
    lut->cy = cy;
    
    size_t pixel_count = width * height;
    size_t float_size = pixel_count * sizeof(float);
    size_t float3_size = pixel_count * sizeof(float3);
    
    // Allocate forward projection tables
    cudaMalloc(&lut->d_dir_vectors, float3_size);
    cudaMalloc(&lut->d_sin_lat, float_size);
    cudaMalloc(&lut->d_cos_lat, float_size);
    cudaMalloc(&lut->d_sin_lon, float_size);
    cudaMalloc(&lut->d_cos_lon, float_size);
    cudaMalloc(&lut->d_depth_multipliers, float_size);
    
    // Allocate inverse projection tables (360x180 resolution for good accuracy)
    lut->inv_width = 720;   // 0.5 degree resolution
    lut->inv_height = 360;  // 0.5 degree resolution
    lut->inv_table_size = lut->inv_width * lut->inv_height;
    cudaMalloc(&lut->d_inv_x_table, lut->inv_table_size * sizeof(float));
    cudaMalloc(&lut->d_inv_y_table, lut->inv_table_size * sizeof(float));
    
    // Calculate memory size
    lut->memory_size = float3_size + 6 * float_size + 2 * lut->inv_table_size * sizeof(float);
    
    // Initialize forward projection LUT
    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    InitializeLUTKernel<<<grid, block>>>(
        lut->d_dir_vectors,
        lut->d_sin_lat,
        lut->d_cos_lat,
        lut->d_sin_lon,
        lut->d_cos_lon,
        lut->d_depth_multipliers,
        width, height, cx, cy
    );
    
    // Initialize inverse projection LUT
    dim3 inv_grid((lut->inv_width + block.x - 1) / block.x,
                  (lut->inv_height + block.y - 1) / block.y);
    
    InitializeInverseLUTKernel<<<inv_grid, block>>>(
        lut->d_inv_x_table,
        lut->d_inv_y_table,
        lut->inv_width, lut->inv_height,
        width, height, cx, cy
    );
    
    cudaDeviceSynchronize();
    
    std::cout << "Created enhanced LUT for " << width << "x" << height 
              << " (cx=" << cx << ", cy=" << cy << ")"
              << " - Memory: " << lut->memory_size / (1024.0 * 1024.0) << " MB" << std::endl;
    
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

// Rest of the manager implementation remains similar...
SphericalLUTManager::SphericalLUTManager() : d_lut_array(nullptr), d_lut_count(nullptr) {
    cudaMalloc(&d_lut_array, MAX_RESOLUTIONS * sizeof(SphericalLUT*));
    cudaMalloc(&d_lut_count, sizeof(int));
    int zero = 0;
    cudaMemcpy(d_lut_count, &zero, sizeof(int), cudaMemcpyHostToDevice);
}

SphericalLUTManager::~SphericalLUTManager() {
    FreeAllLUTs();
    if (d_lut_array) cudaFree(d_lut_array);
    if (d_lut_count) cudaFree(d_lut_count);
}

void SphericalLUTManager::FreeLUT(SphericalLUT* lut) {
    if (!lut) return;
    
    cudaFree(lut->d_dir_vectors);
    cudaFree(lut->d_sin_lat);
    cudaFree(lut->d_cos_lat);
    cudaFree(lut->d_sin_lon);
    cudaFree(lut->d_cos_lon);
    cudaFree(lut->d_depth_multipliers);
    cudaFree(lut->d_inv_x_table);
    cudaFree(lut->d_inv_y_table);
    
    delete lut;
}


