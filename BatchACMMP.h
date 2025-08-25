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

struct StoredResult {
    int width = 0, height = 0;
    std::vector<float4> planes;  // per-pixel: (nx, ny, nz, depth)
    std::vector<float>  costs;   // per-pixel
};

// Forward declarations
class ACMMP;
struct Problem;
struct PatchMatchParams;

struct ProblemGPUResources {
    ProblemGPUResources();   // zero-init everything
    ~ProblemGPUResources();  // calls cleanup()

    int    problem_id = -1;
    int    stream_id  = -1;
    cudaStream_t stream = nullptr;

    // Problem-specific (optional) device buffers if you wire ACMMP to reuse:
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

    // Optional pinned host buffers for faster D2H copies (if ACMMP exposes device pointers)
    float4* planes_host_pinned = nullptr;
    float*  costs_host_pinned  = nullptr;
    size_t  host_pitch_elems   = 0;

    // metadata
    int width  = 0;
    int height = 0;
    int num_images = 0;

    void cleanup();
};

class BatchACMMP {
public:
    BatchACMMP(const std::string& dense_folder_, 
               const std::vector<Problem>& problems,
               bool geom_consistency_,
               bool planar_prior_,
               bool hierarchy_,
               bool multi_geometry_ = false);
    ~BatchACMMP();

    void setMaxConcurrentProblems(size_t max_problems);
    void processBatch(const std::vector<int>& problem_indices);
    void processAllProblems();
    void waitForCompletion();

    // Extract cached results
    void extractResults(int problem_idx, cv::Mat_<float>& depths, 
                        cv::Mat_<cv::Vec3f>& normals,
                        cv::Mat_<float>& costs);

private:
    // config
    size_t max_concurrent_problems = 1;

    // CUDA streams
    std::vector<cudaStream_t> streams;

    // Results cache
    std::unordered_map<int, StoredResult> results_;
    std::mutex results_mutex_;

    // Resource pool
    std::vector<std::unique_ptr<ProblemGPUResources>> resource_pool;
    std::queue<ProblemGPUResources*> available_resources;
    std::mutex resource_mutex_;
    std::condition_variable resource_cv_;

    // Work queue
    std::queue<int> problem_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;

    // Progress
    std::atomic<int> problems_enqueued_{0};
    std::atomic<int> problems_completed_{0};

    std::atomic<bool> stopping_{false};

    // Shared params (stored for completeness)
    PatchMatchParams params;
    std::string dense_folder;
    std::vector<Problem> all_problems;
    bool geom_consistency;
    bool planar_prior;
    bool hierarchy;
    bool multi_geometry;

    // Workers
    std::vector<std::thread> worker_threads;

    // mem estimates
    size_t available_gpu_memory = 0;
    size_t memory_per_problem   = 0;

    // private helpers
    size_t estimateMemoryPerProblem(const Problem& problem);
    size_t getAvailableGPUMemory();
    void   initializeResourcePool();
    ProblemGPUResources* acquireResources();
    void   releaseResources(ProblemGPUResources* resources);
    void   workerFunction();
    void   processProblemOnStream(int problem_idx, ProblemGPUResources* resources);
};

#endif // BATCH_ACMMP_H
