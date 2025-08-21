// BatchACMMP.h - Updated header with streaming I/O
#ifndef BATCH_ACMMP_H
#define BATCH_ACMMP_H

#include "main.h"
#include "ACMMP.h"
#include <cuda_runtime.h>

#include <vector>
#include <memory>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <unordered_map>
#include <iomanip>
#include <string>
#include <sstream>
#include <cassert>
#include <functional>
#include <sys/stat.h>
#include <sys/types.h>
#include <cerrno>
#include <cstring>

// ---- small CUDA helper ----
#ifndef CUDA_CHECK
#define CUDA_CHECK(call) do { \
  cudaError_t err__ = (call); \
  if (err__ != cudaSuccess) { \
    fprintf(stderr, "[CUDA] %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err__)); \
    abort(); \
  } \
} while(0)
#endif

// ---- Directory creation helper (compatible with older C++ standards) ----
static inline void ensure_directory_exists(const std::string& path) {
    if (::mkdir(path.c_str(), 0777) != 0) {
        if (errno != EEXIST) {
            std::cerr << "mkdir(" << path << ") failed: " << std::strerror(errno) << "\n";
        }
    }
}

// Create directories recursively (similar to mkdir -p)
static inline void create_directories_recursive(const std::string& path) {
    if (path.empty()) return;
    
    size_t pos = 0;
    std::string dir;
    
    // Handle absolute paths starting with '/'
    if (path[0] == '/') {
        pos = 1;
        dir = "/";
    }
    
    while ((pos = path.find('/', pos)) != std::string::npos) {
        dir = path.substr(0, pos++);
        if (!dir.empty()) {
            ensure_directory_exists(dir);
        }
    }
    
    // Create the final directory
    ensure_directory_exists(path);
}

// Callback for immediate result processing
using ResultCallback = std::function<void(int problem_idx, 
                                         const cv::Mat_<float>& depths,
                                         const cv::Mat_<cv::Vec3f>& normals,
                                         const cv::Mat_<float>& costs)>;

// Lightweight result struct for immediate processing
struct ProcessedResult {
    int problem_idx;
    int width, height;
    std::vector<float4> planes;
    std::vector<float> costs;
};

// Forward declarations
class ACMMP;
struct Problem;
struct PatchMatchParams;

struct ProblemGPUResources {
    ProblemGPUResources();
    ~ProblemGPUResources();

    int    problem_id = -1;
    int    stream_id  = -1;
    cudaStream_t stream = nullptr;

    // Problem-specific device buffers
    Camera*              cameras_cuda            = nullptr;
    cudaArray*           cuArray[MAX_IMAGES];
    cudaArray*           cuDepthArray[MAX_IMAGES];
    cudaTextureObject_t* texture_objects_cuda    = nullptr;
    cudaTextureObject_t* texture_depths_cuda     = nullptr;
    float4*              plane_hypotheses_cuda   = nullptr;
    float4*              scaled_plane_hypotheses_cuda = nullptr;
    float*               costs_cuda              = nullptr;
    float*               pre_costs_cuda          = nullptr;
    curandState*         rand_states_cuda        = nullptr;
    unsigned int*        selected_views_cuda     = nullptr;
    float*               depths_cuda             = nullptr;
    float4*              prior_planes_cuda       = nullptr;
    unsigned int*        plane_masks_cuda        = nullptr;

    // Pinned host buffers for faster transfers
    float4* planes_host_pinned = nullptr;
    float*  costs_host_pinned  = nullptr;
    size_t  host_pitch_elems   = 0;

    int width  = 0;
    int height = 0;
    int num_images = 0;

    void cleanup();
};

class BatchACMMP {
public:
    BatchACMMP(const std::string& dense_folder, 
               const std::vector<Problem>& problems,
               bool geom_consistency = false,
               bool planar_prior = false,
               bool hierarchy = false,
               bool multi_geometry = false);
    ~BatchACMMP();

    void setResultCallback(ResultCallback callback) { result_callback_ = callback; }
    void setMaxConcurrentProblems(size_t max_problems);
    void processBatch(const std::vector<int>& problem_indices);
    void processAllProblems();
    void waitForCompletion();

private:
    // Config
    size_t max_concurrent_problems = 1;
    
    // CUDA streams
    std::vector<cudaStream_t> streams;

    // Resource pool
    std::vector<std::unique_ptr<ProblemGPUResources>> resource_pool;
    std::queue<ProblemGPUResources*> available_resources;
    std::mutex resource_mutex_;
    std::condition_variable resource_cv_;

    // Work queue
    std::queue<int> problem_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // I/O queue for asynchronous writing
    std::queue<ProcessedResult> io_queue_;
    std::mutex io_mutex_;
    std::condition_variable io_cv_;
    std::vector<std::thread> io_threads_;
    std::atomic<bool> io_stopping_{false};

    // Progress tracking
    std::atomic<int> problems_enqueued_{0};
    std::atomic<int> problems_completed_{0};
    std::atomic<int> problems_written_{0};
    std::atomic<bool> stopping_{false};

    // Result callback
    ResultCallback result_callback_;

    // Shared params
    PatchMatchParams params;
    std::string dense_folder;
    std::vector<Problem> all_problems;
    bool geom_consistency;
    bool planar_prior;
    bool hierarchy;
    bool multi_geometry;

    // Workers
    std::vector<std::thread> worker_threads;

    // Memory estimates
    size_t available_gpu_memory = 0;
    size_t memory_per_problem   = 0;

    // Private helpers
    size_t estimateMemoryPerProblem(const Problem& problem);
    size_t getAvailableGPUMemory();
    void   initializeResourcePool();
    void   initializeIOThreads(size_t num_io_threads = 2);
    ProblemGPUResources* acquireResources();
    void   releaseResources(ProblemGPUResources* resources);
    void   workerFunction();
    void   ioWorkerFunction();
    void   processProblemOnStream(int problem_idx, ProblemGPUResources* resources);
    void   enqueueResult(ProcessedResult&& result);
};

#endif // BATCH_ACMMP_H

