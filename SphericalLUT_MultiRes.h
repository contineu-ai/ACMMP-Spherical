// SphericalLUT_MultiRes.h - Enhanced version
#ifndef SPHERICAL_LUT_MULTIRESO_H
#define SPHERICAL_LUT_MULTIRESO_H

#include <cuda_runtime.h>
#include <vector_types.h>
#include <map>
#include <vector>

// Resolution key for LUT lookup
struct ResolutionKey {
    int width;
    int height;
    float cx;
    float cy;
    
    bool operator<(const ResolutionKey& other) const {
        if (width != other.width) return width < other.width;
        if (height != other.height) return height < other.height;
        if (cx != other.cx) return cx < other.cx;
        return cy < other.cy;
    }
};

// Enhanced LUT with both forward and inverse projections
struct SphericalLUT {
    // Forward projection tables (pixel to direction)
    float3* d_dir_vectors;      // Direction vectors for each pixel
    float* d_sin_lat;           // Precomputed sin(latitude)
    float* d_cos_lat;           // Precomputed cos(latitude)
    float* d_sin_lon;           // Precomputed sin(longitude)
    float* d_cos_lon;           // Precomputed cos(longitude)
    
    // NEW: Inverse projection tables (direction to pixel)
    float* d_inv_x_table;       // Precomputed x pixel coordinates
    float* d_inv_y_table;       // Precomputed y pixel coordinates
    int inv_table_size;         // Size of inverse table (typically 360x180)
    int inv_width;              // Width of inverse table
    int inv_height;             // Height of inverse table
    
    // NEW: Optimized depth computation tables
    float* d_depth_multipliers; // For plane hypothesis depth computation
    
    // Camera parameters
    int width;
    int height;
    float cx;  // Principal point x
    float cy;  // Principal point y
    
    // Size in bytes
    size_t memory_size;
};

class SphericalLUTManager {
private:
    std::map<ResolutionKey, SphericalLUT*> lut_map;
    std::vector<SphericalLUT*> device_luts;
    SphericalLUT** d_lut_array;
    int* d_lut_count;
    static const int MAX_RESOLUTIONS = 20;
    
public:
    SphericalLUTManager();
    ~SphericalLUTManager();
    
    SphericalLUT* GetOrCreateLUT(int width, int height, float cx, float cy);
    SphericalLUT* FindClosestLUT(int width, int height, float cx, float cy);
    void InitializeMultiScaleLUTs(int base_width, int base_height, 
                                  float base_cx, float base_cy, int num_scales);
    
    SphericalLUT** GetDeviceLUTArray() { return d_lut_array; }
    SphericalLUT* GetLUTByIndex(int idx);
    void FreeAllLUTs();
    size_t GetTotalMemoryUsage() const;
    
private:
    SphericalLUT* CreateLUT(int width, int height, float cx, float cy);
    void FreeLUT(SphericalLUT* lut);
    void UpdateDeviceArray();
};

extern SphericalLUTManager* g_lut_manager;
void InitializeLUTManager();
void FreeLUTManager();

#endif