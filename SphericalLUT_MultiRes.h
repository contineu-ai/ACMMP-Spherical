#ifndef SPHERICAL_LUT_MULTIRES_H
#define SPHERICAL_LUT_MULTIRES_H

#include <vector>
#include <unordered_map>
#include <cstdint>  // For uint64_t
#include <cmath>    // For fabsf
#include <cuda_runtime.h>
#include <cuda_texture_types.h>

// Forward declarations
struct Camera;

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
    bool uses_pool;  // Flag to indicate if using memory pool
    
    // Constructor
    SphericalLUT() : width(0), height(0), cx(0.0f), cy(0.0f),
                     d_dir_vectors(nullptr), d_sin_lat(nullptr), d_cos_lat(nullptr),
                     d_sin_lon(nullptr), d_cos_lon(nullptr), 
                     d_lon_values(nullptr), d_lat_values(nullptr),
                     memory_size(0), uses_pool(false) {}
};

// Resolution key for hash map with better hashing and comparison operators
struct ResolutionKey {
    int width, height;
    float cx, cy;
    
    bool operator==(const ResolutionKey& other) const {
        return width == other.width && height == other.height && 
               fabsf(cx - other.cx) < 0.01f && fabsf(cy - other.cy) < 0.01f;
    }
    
    // Comparison operator for std::map/std::set (strict weak ordering)
    bool operator<(const ResolutionKey& other) const {
        if (width != other.width) return width < other.width;
        if (height != other.height) return height < other.height;
        if (fabsf(cx - other.cx) > 0.01f) return cx < other.cx;
        return cy < other.cy;
    }
    
    // Additional comparison operators for completeness
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
        // Better hash combining to reduce collisions
        size_t h1 = std::hash<int>{}(key.width);
        size_t h2 = std::hash<int>{}(key.height);
        size_t h3 = std::hash<int>{}(static_cast<int>(key.cx * 100));
        size_t h4 = std::hash<int>{}(static_cast<int>(key.cy * 100));
        
        // Improved hash combination
        return h1 ^ (h2 << 1) ^ (h3 << 2) ^ (h4 << 3);
    }
};

// Enhanced LUT Manager with memory pool and texture support
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
    
    // Private methods
    SphericalLUT* CreateLUT(int width, int height, float cx, float cy);
    void FreeLUT(SphericalLUT* lut);
    void UpdateDeviceArray();
    
public:
    // Constructor and destructor
    SphericalLUTManager();
    ~SphericalLUTManager();
    
    // Main interface
    SphericalLUT* GetOrCreateLUT(int width, int height, float cx, float cy);
    SphericalLUT* FindClosestLUT(int width, int height, float cx, float cy);
    SphericalLUT* GetLUTByIndex(int idx);
    
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
    void OptimizeLUTOrder(); // Reorder LUTs based on access frequency
    
    // Debug and profiling
    void PrintStatistics() const;
    void ValidateAllLUTs() const;
};

// Global manager functions
void InitializeLUTManager();
void FreeLUTManager();
extern SphericalLUTManager* g_lut_manager;

// Inline helper functions for better performance
inline SphericalLUT* GetLUTForResolution(int width, int height, float cx, float cy) {
    return g_lut_manager ? g_lut_manager->GetOrCreateLUT(width, height, cx, cy) : nullptr;
}

inline bool IsLUTManagerInitialized() {
    return g_lut_manager != nullptr;
}

// Performance monitoring structure
struct LUTPerformanceStats {
    uint64_t cache_hits;
    uint64_t cache_misses;
    uint64_t direct_accesses;
    uint64_t fallback_computations;
    double total_lookup_time;
    
    LUTPerformanceStats() : cache_hits(0), cache_misses(0), direct_accesses(0), 
                           fallback_computations(0), total_lookup_time(0.0) {}
    
    double GetCacheHitRate() const {
        uint64_t total = cache_hits + cache_misses;
        return total > 0 ? static_cast<double>(cache_hits) / total : 0.0;
    }
    
    double GetDirectAccessRate() const {
        uint64_t total = direct_accesses + fallback_computations;
        return total > 0 ? static_cast<double>(direct_accesses) / total : 0.0;
    }
};

// Global performance statistics (optional, for debugging)
extern LUTPerformanceStats g_lut_stats;

// Utility macros for performance measurement (debug builds only)
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

// Configuration constants for fine-tuning
namespace LUTConfig {
    // Memory management
    constexpr size_t DEFAULT_MEMORY_POOL_SIZE = 256 * 1024 * 1024; // 256MB
    constexpr size_t MIN_LUT_SIZE = 32 * 32; // Minimum LUT resolution
    constexpr size_t MAX_LUT_SIZE = 8192 * 8192; // Maximum LUT resolution
    
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
}

#endif // SPHERICAL_LUT_MULTIRES_H