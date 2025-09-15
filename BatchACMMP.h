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
#include <functional>
#include <cuda_runtime.h>

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
struct ProblemGPUResources;

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
               bool multi_geometry_);
    
    ~BatchACMMP();

    void processAllProblems();
    void processBatch(const std::vector<int>& idxs);
    void waitForGPUCompletion();
    void waitForDiskCompletion();
    void waitForCompletion();
    
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

private:
    // Configuration
    std::string dense_folder;
    std::vector<Problem> all_problems;
    bool geom_consistency;
    bool planar_prior;
    bool hierarchy;
    bool multi_geometry;

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

// GPU resource management struct
struct ProblemGPUResources {
    static constexpr int MAX_IMAGES_PER_PROBLEM = 32;
    
    int stream_id = -1;
    cudaStream_t stream = nullptr;
    
    cudaArray* cuArray[MAX_IMAGES_PER_PROBLEM];
    cudaArray* cuDepthArray[MAX_IMAGES_PER_PROBLEM];
    
    void* cameras_cuda = nullptr;
    void* texture_objects_cuda = nullptr;
    void* texture_depths_cuda = nullptr;
    void* plane_hypotheses_cuda = nullptr;
    void* scaled_plane_hypotheses_cuda = nullptr;
    void* costs_cuda = nullptr;
    void* pre_costs_cuda = nullptr;
    void* rand_states_cuda = nullptr;
    void* selected_views_cuda = nullptr;
    void* depths_cuda = nullptr;
    void* prior_planes_cuda = nullptr;
    void* plane_masks_cuda = nullptr;
    
    void* planes_host_pinned = nullptr;
    void* costs_host_pinned = nullptr;
    
    ProblemGPUResources();
    ~ProblemGPUResources();
    void cleanup();
};

#endif // BATCHACMMP_H

