#ifndef SPHERICAL_LUT_MULTIRES_H
#define SPHERICAL_LUT_MULTIRES_H

#include <vector>
#include <unordered_map>
#include <cstdint>
#include <cmath>
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

// Forward declarations
struct Camera;

// Inverse trigonometric LUT configuration
namespace InverseTrigConfig {
    constexpr int ASIN_LUT_SIZE = 8192;        // 8K resolution for asin
    constexpr int ATAN2_LUT_SIZE = 512;        // 512x512 grid for atan2
    constexpr float ASIN_INPUT_MIN = -1.0f;
    constexpr float ASIN_INPUT_MAX = 1.0f;
    constexpr float ATAN2_RANGE = 10.0f;       // Cover [-10, 10] for x and z
}

// Inverse trigonometric lookup tables (shared across all resolution LUTs)
struct InverseTrigLUT {
    // ASIN lookup table (1D)
    float* d_asin_lut;          // Device memory
    int asin_size;
    float asin_min;
    float asin_max;
    float asin_scale;           // For fast indexing
    
    // ATAN2 lookup table (2D grid)
    float* d_atan2_lut;         // Device memory
    int atan2_size;             // Grid dimension (size x size)
    float atan2_range;          // Input range [-range, range]
    float atan2_scale;          // For fast indexing
    
    // Memory tracking
    size_t total_memory;
    
    InverseTrigLUT() : d_asin_lut(nullptr), d_atan2_lut(nullptr),
                       asin_size(0), atan2_size(0),
                       asin_min(0.0f), asin_max(0.0f),
                       asin_scale(0.0f), atan2_range(0.0f),
                       atan2_scale(0.0f), total_memory(0) {}
};

// Enhanced SphericalLUT structure with memory pool support
struct SphericalLUT {
    int width, height;
    float cx, cy;
    
    // Device memory pointers
    float3* d_dir_vectors;
    float* d_sin_lat;
    float* d_cos_lat;
    float* d_sin_lon;
    float* d_cos_lon;
    float* d_lon_values;
    float* d_lat_values;
    
    // Memory management
    size_t memory_size;
    bool uses_pool;
    
    SphericalLUT() : width(0), height(0), cx(0.0f), cy(0.0f),
                     d_dir_vectors(nullptr), d_sin_lat(nullptr), d_cos_lat(nullptr),
                     d_sin_lon(nullptr), d_cos_lon(nullptr), 
                     d_lon_values(nullptr), d_lat_values(nullptr),
                     memory_size(0), uses_pool(false) {}
};

// Resolution key for hash map
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
    
    bool operator!=(const ResolutionKey& other) const {
        return !(*this == other);
    }
    
    bool operator<=(const ResolutionKey& other) const {
        return *this < other || *this == other;
    }
    
    bool operator>(const ResolutionKey& other) const {
        return other < *this;
    }
    
    bool operator>=(const ResolutionKey& other) const {
        return !(*this < other);
    }
};

// Custom hash function for ResolutionKey
struct ResolutionKeyHash {
    size_t operator()(const ResolutionKey& key) const {
        size_t h1 = std::hash<int>{}(key.width);
        size_t h2 = std::hash<int>{}(key.height);
        size_t h3 = std::hash<int>{}(static_cast<int>(key.cx * 100));
        size_t h4 = std::hash<int>{}(static_cast<int>(key.cy * 100));
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

// Enhanced LUT Manager with inverse trig LUTs
class SphericalLUTManager {
public:
    static constexpr int MAX_RESOLUTIONS = 10;
    static constexpr int MAX_TEXTURE_LUTS = 5;
    
private:
    // LUT storage
    std::unordered_map<ResolutionKey, SphericalLUT*, ResolutionKeyHash> lut_map;
    std::vector<SphericalLUT*> device_luts;
    
    // Device memory for LUT array
    SphericalLUT** d_lut_array;
    int* d_lut_count;
    
    // Memory pool for efficient allocation
    void* memory_pool;
    size_t memory_pool_size;
    size_t memory_pool_offset;
    
    // Inverse trigonometric LUTs (shared across all resolutions)
    InverseTrigLUT* inverse_trig_lut;
    
    // Private methods
    SphericalLUT* CreateLUT(int width, int height, float cx, float cy);
    void FreeLUT(SphericalLUT* lut);
    void UpdateDeviceArray();
    
    // Inverse trig LUT methods
    void InitializeInverseTrigLUTs();
    void FreeInverseTrigLUTs();
    
public:
    // Constructor and destructor
    SphericalLUTManager();
    ~SphericalLUTManager();
    
    // Main interface
    SphericalLUT* GetOrCreateLUT(int width, int height, float cx, float cy);
    SphericalLUT* FindClosestLUT(int width, int height, float cx, float cy);
    SphericalLUT* GetLUTByIndex(int idx);
    
    // Inverse trig LUT access
    InverseTrigLUT* GetInverseTrigLUT() const { return inverse_trig_lut; }
    
    // Batch operations
    void InitializeMultiScaleLUTs(int base_width, int base_height, 
                                  float base_cx, float base_cy, 
                                  int num_scales = 4);
    
    // Memory management
    void FreeAllLUTs();
    size_t GetTotalMemoryUsage() const;
    
    // Statistics
    size_t GetLUTCount() const { return lut_map.size(); }
    bool IsUsingMemoryPool() const { return memory_pool != nullptr; }
    size_t GetMemoryPoolUsage() const { return memory_pool_offset; }
    float GetMemoryPoolUtilization() const { 
        return memory_pool_size > 0 ? static_cast<float>(memory_pool_offset) / memory_pool_size : 0.0f; 
    }
    
    // Performance optimization methods
    void DefragmentMemoryPool();
    void OptimizeLUTOrder();
    
    // Debug and profiling
    void PrintStatistics() const;
    void ValidateAllLUTs() const;
    void ValidateInverseTrigLUTs() const;
};

// Global manager functions
void InitializeLUTManager();
void FreeLUTManager();
extern SphericalLUTManager* g_lut_manager;

// Inline helper functions
inline SphericalLUT* GetLUTForResolution(int width, int height, float cx, float cy) {
    return g_lut_manager ? g_lut_manager->GetOrCreateLUT(width, height, cx, cy) : nullptr;
}

inline bool IsLUTManagerInitialized() {
    return g_lut_manager != nullptr;
}

inline InverseTrigLUT* GetGlobalInverseTrigLUT() {
    return g_lut_manager ? g_lut_manager->GetInverseTrigLUT() : nullptr;
}

// Performance monitoring structure
struct LUTPerformanceStats {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t direct_accesses;
    uint64_t fallback_computations;
    uint64_t inverse_trig_lut_hits;
    uint64_t inverse_trig_direct_calls;
    double total_lookup_time;
    
    LUTPerformanceStats() : cache_hits(0), cache_misses(0), direct_accesses(0), 
                           fallback_computations(0), inverse_trig_lut_hits(0),
                           inverse_trig_direct_calls(0), total_lookup_time(0.0) {}
    
    double GetCacheHitRate() const {
        uint64_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
    }
    
    double GetDirectAccessRate() const {
        uint64_t total = direct_accesses + fallback_computations;
        return total > 0 ? static_cast<double>(direct_accesses) / total : 0.0;
    }
    
    double GetInverseTrigLUTRate() const {
        uint64_t total = inverse_trig_lut_hits + inverse_trig_direct_calls;
        return total > 0 ? static_cast<double>(inverse_trig_lut_hits) / total : 0.0;
    }
};

// Global performance statistics
extern LUTPerformanceStats g_lut_stats;

// Utility macros for performance measurement
#ifdef DEBUG_LUT_PERFORMANCE
#define LUT_PERF_START() auto start_time = std::chrono::high_resolution_clock::now()
#define LUT_PERF_END(counter) do { \
    auto end_time = std::chrono::high_resolution_clock::now(); \
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time); \
    g_lut_stats.total_lookup_time += duration.count() * 1e-6; \
    g_lut_stats.counter++; \
} while(0)
#else
#define LUT_PERF_START()
#define LUT_PERF_END(counter)
#endif

// Configuration constants
namespace LUTConfig {
    // Memory management
    constexpr size_t DEFAULT_MEMORY_POOL_SIZE = 256 * 1024 * 1024; // 256MB
    constexpr size_t MIN_LUT_SIZE = 32 * 32;
    constexpr size_t MAX_LUT_SIZE = 8192 * 8192;
    
    // Performance tuning
    constexpr int OPTIMAL_BLOCK_SIZE_X = 32;
    constexpr int OPTIMAL_BLOCK_SIZE_Y = 8;
    constexpr float EXACT_MATCH_TOLERANCE = 0.1f;
    
    // Memory access optimization
    constexpr bool USE_MEMORY_POOL = true;
    constexpr bool ENABLE_PREFETCHING = true;
    
    // Fallback thresholds
    constexpr float MAX_ACCEPTABLE_ERROR = 1e-6f;
    constexpr int MAX_SEARCH_ITERATIONS = 100;
    
    // Inverse trig optimization
    constexpr bool USE_INVERSE_TRIG_LUTS = true;
    constexpr bool VALIDATE_INVERSE_TRIG = false; // Set to true for accuracy testing
}

#endif // SPHERICAL_LUT_MULTIRES_H