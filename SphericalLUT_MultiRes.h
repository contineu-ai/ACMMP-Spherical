#ifndef SPHERICAL_LUT_MULTIRES_H
#define SPHERICAL_LUT_MULTIRES_H

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

struct Camera;

namespace InverseTrigConfig {
    constexpr int ASIN_LUT_SIZE = 8192;
    constexpr int ATAN2_LUT_SIZE = 512;
    constexpr float ASIN_INPUT_MIN = -1.0f;
    constexpr float ASIN_INPUT_MAX = 1.0f;
    constexpr float ATAN2_RANGE = 10.0f;
}

struct InverseTrigLUT {
    float* d_asin_lut;
    int asin_size;
    float asin_min;
    float asin_max;
    float asin_scale;

    float* d_atan2_lut;
    int atan2_size;
    float atan2_range;
    float atan2_scale;

    size_t total_memory;

    InverseTrigLUT() : d_asin_lut(nullptr), asin_size(0),
                       asin_min(0.0f), asin_max(0.0f), asin_scale(0.0f),
                       d_atan2_lut(nullptr), atan2_size(0),
                       atan2_range(0.0f), atan2_scale(0.0f),
                       total_memory(0) {}
};

struct SphericalLUT {
    int width, height;
    float cx, cy;

    float3* d_dir_vectors;
    float* d_sin_lat;
    float* d_cos_lat;
    float* d_sin_lon;
    float* d_cos_lon;
    float* d_lon_values;
    float* d_lat_values;

    size_t memory_size;
    bool uses_pool;

    SphericalLUT() : width(0), height(0), cx(0.0f), cy(0.0f),
                     d_dir_vectors(nullptr), d_sin_lat(nullptr), d_cos_lat(nullptr),
                     d_sin_lon(nullptr), d_cos_lon(nullptr),
                     d_lon_values(nullptr), d_lat_values(nullptr),
                     memory_size(0), uses_pool(false) {}
};

struct ResolutionKey {
    int width, height;
    float cx, cy;

    bool operator==(const ResolutionKey& other) const {
        return width == other.width && height == other.height &&
               fabsf(cx - other.cx) < 0.01f && fabsf(cy - other.cy) < 0.01f;
    }

    bool operator<(const ResolutionKey& other) const {
        if (width != other.width) return width < other.width;
        if (height != other.height) return height < other.height;
        if (fabsf(cx - other.cx) > 0.01f) return cx < other.cx;
        return cy < other.cy;
    }
};

struct ResolutionKeyHash {
    size_t operator()(const ResolutionKey& key) const {
        size_t h1 = std::hash<int>{}(key.width);
        size_t h2 = std::hash<int>{}(key.height);
        size_t h3 = std::hash<int>{}(static_cast<int>(key.cx * 100));
        size_t h4 = std::hash<int>{}(static_cast<int>(key.cy * 100));
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

class SphericalLUTManager {
public:
    static constexpr int MAX_RESOLUTIONS = 10;

private:
    std::unordered_map<ResolutionKey, SphericalLUT*, ResolutionKeyHash> lut_map;
    std::vector<SphericalLUT*> device_luts;

    SphericalLUT** d_lut_array;
    int* d_lut_count;

    void* memory_pool;
    size_t memory_pool_size;
    size_t memory_pool_offset;

    InverseTrigLUT* inverse_trig_lut;

    SphericalLUT* CreateLUT(int width, int height, float cx, float cy);
    void FreeLUT(SphericalLUT* lut);
    void UpdateDeviceArray();
    void InitializeInverseTrigLUTs();
    void FreeInverseTrigLUTs();

public:
    SphericalLUTManager();
    ~SphericalLUTManager();

    SphericalLUT* GetOrCreateLUT(int width, int height, float cx, float cy);
    SphericalLUT* FindClosestLUT(int width, int height, float cx, float cy);
    void FreeAllLUTs();
    size_t GetTotalMemoryUsage() const;
};

void InitializeLUTManager();
void FreeLUTManager();
extern SphericalLUTManager* g_lut_manager;

#endif // SPHERICAL_LUT_MULTIRES_H
