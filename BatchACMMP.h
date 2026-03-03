// BatchACMMP.h - Fixed compilation issues
#ifndef BATCHACMMP_H
#define BATCHACMMP_H

#include "ACMMP.h"
#include <vector>
#include <string>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <list>
#include <functional>
#include <cuda_runtime.h>
#include "main.h"

// CUDA error checking macro
#define CUDA_CHECK(call) do { \
    cudaError_t error = call; \
    if (error != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(error) << std::endl; \
        exit(1); \
    } \
} while(0)

// Forward declarations
struct Problem;

// ── LRU Image Cache ──────────────────────────────────────────────────────────
struct ImageCacheEntry {
    cv::Mat image_float;   // grayscale float32 full-res
    Camera camera;
    cv::Mat mask_float;    // normalized 0-1, empty if no mask
    bool has_mask = false;
};

class ImageCache {
public:
    ImageCache(size_t max_entries, const std::string& dense_folder, bool has_masks);
    std::shared_ptr<const ImageCacheEntry> get(int image_id);  // thread-safe
    size_t hits() const { std::lock_guard<std::mutex> lk(mutex_); return hits_; }
    size_t misses() const { std::lock_guard<std::mutex> lk(mutex_); return misses_; }
private:
    mutable std::mutex mutex_;
    size_t max_entries_;
    std::string dense_folder_;
    bool has_masks_;
    std::unordered_map<int, std::shared_ptr<ImageCacheEntry>> cache_;
    std::list<int> lru_order_;
    size_t hits_ = 0;
    size_t misses_ = 0;
    void evict_if_needed();  // under lock
    std::shared_ptr<ImageCacheEntry> load(int image_id);
};

// ── ProblemGPUResources ──────────────────────────────────────────────────────
class ProblemGPUResources {
public:
    ProblemGPUResources();
    ~ProblemGPUResources();

    // NEW METHOD: Handles one-time allocation of all necessary GPU memory.
    void allocate(int max_width, int max_height, int max_images);

    // Handles freeing all allocated GPU memory.
    void cleanup();

    cudaStream_t stream = nullptr;
    int stream_id = -1;

    // === GPU DEVICE MEMORY (Owned by this object) ===
    // CUDA arrays for 2D texture memory
    cudaArray* cuArray[MAX_IMAGES];
    cudaArray* cuDepthArray[MAX_IMAGES];
    
    // Pointers to device memory buffers
    Camera* cameras_cuda = nullptr;
    cudaTextureObjects* texture_objects_cuda = nullptr;
    cudaTextureObjects* texture_depths_cuda = nullptr;
    float4* plane_hypotheses_cuda = nullptr;
    float4* scaled_plane_hypotheses_cuda = nullptr;
    float* costs_cuda = nullptr;
    float* pre_costs_cuda = nullptr;
    RNGState* rand_states_cuda = nullptr;
    unsigned int* selected_views_cuda = nullptr;
    float* depths_cuda = nullptr;
    float4* prior_planes_cuda = nullptr;
    unsigned int* plane_masks_cuda = nullptr;
    uint8_t* ref_mask_cuda = nullptr;  // Item 4: 1=masked, 0=valid

    // Item 5: Batch planar prior buffers
    int2* support_points_cuda = nullptr;
    int* num_support_points_cuda = nullptr;   // device atomic counter
    int2* triangle_vertices_cuda = nullptr;   // packed [v1,v2,v3,...], 3 per triangle

    // Pinned host counterparts for planar prior
    int2* support_points_pinned = nullptr;
    int* num_support_points_pinned = nullptr;

    // === HOST-SIDE HELPERS (Owned by this object) ===
    // Host-side structs that hold the CUDA texture object handles.
    // These need to be persistent to be copied to texture_objects_cuda.
    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;

    // Pinned host memory for fast, asynchronous DMA transfers
    float4* planes_host_pinned = nullptr;
    float* costs_host_pinned = nullptr;

    // Texture reuse tracking
    int allocated_images = 0;       // how many array slots were allocated
    bool textures_created = false;  // texture objects created once in allocate()
};

// Structure to hold completed results for disk writing
struct CompletedResult {
    int problem_idx;
    Problem problem;
    cv::Mat_<float> depths;
    cv::Mat_<cv::Vec3f> normals;
    cv::Mat_<float> costs;
    bool geom_consistency;
    
    CompletedResult() = default;
    CompletedResult(int idx, const Problem& prob, 
                   cv::Mat_<float> d, cv::Mat_<cv::Vec3f> n, cv::Mat_<float> c, bool geom)
        : problem_idx(idx), problem(prob), depths(std::move(d)), 
          normals(std::move(n)), costs(std::move(c)), geom_consistency(geom) {}
    
    // Move constructor
    CompletedResult(CompletedResult&& other) noexcept
        : problem_idx(other.problem_idx), problem(std::move(other.problem)),
          depths(std::move(other.depths)), normals(std::move(other.normals)),
          costs(std::move(other.costs)), geom_consistency(other.geom_consistency) {}
    
    // Move assignment
    CompletedResult& operator=(CompletedResult&& other) noexcept {
        if (this != &other) {
            problem_idx = other.problem_idx;
            problem = std::move(other.problem);
            depths = std::move(other.depths);
            normals = std::move(other.normals);
            costs = std::move(other.costs);
            geom_consistency = other.geom_consistency;
        }
        return *this;
    }
    
    // Delete copy constructor and assignment to force move semantics
    CompletedResult(const CompletedResult&) = delete;
    CompletedResult& operator=(const CompletedResult&) = delete;
};

class BatchACMMP {
public:
    BatchACMMP(const std::string& dense_folder_, 
               const std::vector<Problem>& problems,
               bool geom_consistency_,
               bool planar_prior_,
               bool hierarchy_,
               bool multi_geometry_,size_t mask_disk_queue_size_ = 400);
               
    
    ~BatchACMMP();
    
    void processAllProblems();
    void processBatch(const std::vector<int>& idxs);
    void waitForGPUCompletion();
    void waitForDiskCompletion();
    void waitForCompletion();
    void allocate(int max_width, int max_height, int max_images);
    void cleanup();
    // Memory and progress monitoring
    size_t getPeakMemoryUsage() const;
    size_t getCurrentMemoryUsage() const;
    size_t getActiveGPUProblems() const;
    size_t getPendingDiskWrites() const;
    size_t getCompletedDiskWrites() const;

    size_t getCompletedGPUProblems() const {
        return gpu_completed_.load();
    }
    
    // Improved completion check without deadlock risk
    bool isComplete() const {
        return disk_completed_.load() >= problems_enqueued_.load() &&
               active_gpu_problems_.load() == 0;
    }
    std::condition_variable disk_queue_space_cv_;      
private:
    // Configuration
    std::string dense_folder;
    std::vector<Problem> all_problems;
    bool geom_consistency;
    bool planar_prior;
    bool hierarchy;
    bool multi_geometry;
    size_t mask_disk_queue_size; 
    // GPU processing resources
    size_t max_concurrent_problems;
    size_t num_disk_writers;
    std::vector<cudaStream_t> streams;
    std::vector<std::unique_ptr<ProblemGPUResources>> resource_pool;
    
    // GPU resource management
    std::queue<ProblemGPUResources*> available_resources;
    std::mutex resource_mutex_;
    std::condition_variable resource_cv_;
    
    // GPU work queue
    std::queue<int> gpu_work_queue_;
    std::mutex gpu_queue_mutex_;
    std::condition_variable gpu_queue_cv_;
    
    // Disk I/O queue
    std::queue<CompletedResult> disk_write_queue_;
    mutable std::mutex disk_queue_mutex_; // Make mutable for const methods
    std::condition_variable disk_queue_cv_;
    
    // Thread management
    std::vector<std::thread> gpu_worker_threads;
    std::vector<std::thread> disk_writer_threads;
    std::atomic<bool> stopping_gpu_{false};
    std::atomic<bool> stopping_disk_{false};
    
    // Progress tracking
    std::atomic<int> problems_enqueued_{0};
    std::atomic<int> gpu_completed_{0};
    std::atomic<int> disk_completed_{0};
    
    // Memory monitoring
    mutable std::mutex memory_mutex_;
    size_t peak_memory_usage_ = 0;
    std::atomic<size_t> active_gpu_problems_{0};
    
    // Memory estimation
    size_t available_gpu_memory;
    size_t memory_per_problem;

    // Image/camera/mask cache
    std::unique_ptr<ImageCache> image_cache_;
    
    // Internal methods
    void initializeResourcePool();
    void initializeDiskWriters();
    ProblemGPUResources* acquireResources();
    void releaseResources(ProblemGPUResources* r);
    
    void gpuWorkerFunction();
    void diskWriterFunction();
    void processProblemOnStream(int problem_idx, ProblemGPUResources* resources);
    void writeProblemToDisk(CompletedResult&& result);
    
    size_t estimateMemoryPerProblem(const Problem& problem);
    size_t getAvailableGPUMemory();
    size_t getSystemRAM();
    size_t getProcessMemoryUsage() const;
};


#endif // BATCHACMMP_H

