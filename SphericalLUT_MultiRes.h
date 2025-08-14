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
    
    bool operator==(const ResolutionKey& other) const {
        return width == other.width && height == other.height &&
               std::abs(cx - other.cx) < 0.01f && std::abs(cy - other.cy) < 0.01f;
    }
};

// Single resolution LUT
struct SphericalLUT {
    // Forward projection tables (pixel to direction)
    float3* d_dir_vectors;      // Direction vectors for each pixel
    float* d_sin_lat;           // Precomputed sin(latitude) for each pixel
    float* d_cos_lat;           // Precomputed cos(latitude) for each pixel
    float* d_sin_lon;           // Precomputed sin(longitude) for each pixel
    float* d_cos_lon;           // Precomputed cos(longitude) for each pixel
    float* d_lon_values;        // Longitude values for each pixel
    float* d_lat_values;        // Latitude values for each pixel
    
    // Camera parameters
    int width;
    int height;
    float cx;  // Principal point x
    float cy;  // Principal point y
    
    // Size in bytes
    size_t memory_size;
};

// Multi-resolution LUT manager
class SphericalLUTManager {
private:
    std::map<ResolutionKey, SphericalLUT*> lut_map;
    std::vector<SphericalLUT*> device_luts;
    
    // Device array of LUT pointers for kernel access
    SphericalLUT** d_lut_array;
    int* d_lut_count;
    
    // Maximum number of resolutions to support
    static const int MAX_RESOLUTIONS = 10;
    
public:
    SphericalLUTManager();
    ~SphericalLUTManager();
    
    // Initialize or get LUT for specific resolution
    SphericalLUT* GetOrCreateLUT(int width, int height, float cx, float cy);
    
    // Find closest LUT if exact match not found
    SphericalLUT* FindClosestLUT(int width, int height, float cx, float cy);
    
    // Initialize LUTs for all expected resolutions
    void InitializeMultiScaleLUTs(int base_width, int base_height, 
                                  float base_cx, float base_cy, 
                                  int num_scales);
    
    // Get device LUT array for kernel access
    SphericalLUT** GetDeviceLUTArray() { return d_lut_array; }
    
    // Get specific LUT by index
    SphericalLUT* GetLUTByIndex(int idx);
    
    // Clean up all LUTs
    void FreeAllLUTs();
    
    // Get memory usage
    size_t GetTotalMemoryUsage() const;
    
private:
    SphericalLUT* CreateLUT(int width, int height, float cx, float cy);
    void FreeLUT(SphericalLUT* lut);
    void UpdateDeviceArray();
};

// Global LUT manager instance
extern SphericalLUTManager* g_lut_manager;

// Initialize global LUT manager
void InitializeLUTManager();

// Free global LUT manager
void FreeLUTManager();

#endif // SPHERICAL_LUT_MULTIRESO_H