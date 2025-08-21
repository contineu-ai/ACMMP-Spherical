#include "BatchACMMP.h"
#include "ACMMP_device.cuh"

#include <iostream>
#include <algorithm>
#include <chrono>

// ---------- ProblemGPUResources impl ----------
ProblemGPUResources::ProblemGPUResources() {
    for (int i = 0; i < MAX_IMAGES; ++i) {
        cuArray[i] = nullptr;
        cuDepthArray[i] = nullptr;
    }
}

ProblemGPUResources::~ProblemGPUResources() { cleanup(); }

void ProblemGPUResources::cleanup() {
    // Free CUDA arrays
    for (int i = 0; i < MAX_IMAGES; ++i) {
        if (cuArray[i])     { cudaFreeArray(cuArray[i]); cuArray[i] = nullptr; }
        if (cuDepthArray[i]){ cudaFreeArray(cuDepthArray[i]); cuDepthArray[i] = nullptr; }
    }

    // Free device memory
    if (cameras_cuda)             { CUDA_CHECK(cudaFree(cameras_cuda)); cameras_cuda = nullptr; }
    if (texture_objects_cuda)     { CUDA_CHECK(cudaFree(texture_objects_cuda)); texture_objects_cuda = nullptr; }
    if (texture_depths_cuda)      { CUDA_CHECK(cudaFree(texture_depths_cuda)); texture_depths_cuda = nullptr; }
    if (plane_hypotheses_cuda)    { CUDA_CHECK(cudaFree(plane_hypotheses_cuda)); plane_hypotheses_cuda = nullptr; }
    if (scaled_plane_hypotheses_cuda){ CUDA_CHECK(cudaFree(scaled_plane_hypotheses_cuda)); scaled_plane_hypotheses_cuda = nullptr; }
    if (costs_cuda)               { CUDA_CHECK(cudaFree(costs_cuda)); costs_cuda = nullptr; }
    if (pre_costs_cuda)           { CUDA_CHECK(cudaFree(pre_costs_cuda)); pre_costs_cuda = nullptr; }
    if (rand_states_cuda)         { CUDA_CHECK(cudaFree(rand_states_cuda)); rand_states_cuda = nullptr; }
    if (selected_views_cuda)      { CUDA_CHECK(cudaFree(selected_views_cuda)); selected_views_cuda = nullptr; }
    if (depths_cuda)              { CUDA_CHECK(cudaFree(depths_cuda)); depths_cuda = nullptr; }
    if (prior_planes_cuda)        { CUDA_CHECK(cudaFree(prior_planes_cuda)); prior_planes_cuda = nullptr; }
    if (plane_masks_cuda)         { CUDA_CHECK(cudaFree(plane_masks_cuda)); plane_masks_cuda = nullptr; }

    if (planes_host_pinned) { CUDA_CHECK(cudaFreeHost(planes_host_pinned)); planes_host_pinned = nullptr; }
    if (costs_host_pinned)  { CUDA_CHECK(cudaFreeHost(costs_host_pinned));  costs_host_pinned  = nullptr; }
}

// ---------- BatchACMMP impl ----------

BatchACMMP::BatchACMMP(const std::string& dense_folder_, 
                       const std::vector<Problem>& problems,
                       bool geom_consistency_,
                       bool planar_prior_,
                       bool hierarchy_,
                       bool multi_geometry_)
    : dense_folder(dense_folder_), all_problems(problems),
      geom_consistency(geom_consistency_), planar_prior(planar_prior_),
      hierarchy(hierarchy_), multi_geometry(multi_geometry_)
{
    // Device props
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    int concurrentKernels = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentKernels, cudaDevAttrConcurrentKernels, 0));

    available_gpu_memory = getAvailableGPUMemory();
    memory_per_problem   = problems.empty() ? (size_t)500 * 1024 * 1024
                                            : estimateMemoryPerProblem(problems[0]);

    const size_t usable = size_t(double(available_gpu_memory) * 0.8);
    size_t by_mem = std::max<size_t>(1, usable / std::max<size_t>(memory_per_problem, 1));
    size_t by_sm = (prop.multiProcessorCount > 0) ? std::max<size_t>(1, prop.multiProcessorCount / 2) : 8;
    size_t cap   = concurrentKernels ? 16 : 1;

    max_concurrent_problems = std::min<size_t>(std::min(by_mem, by_sm), cap);
    if (max_concurrent_problems == 0) max_concurrent_problems = 1;

    std::cout << "[BatchACMMP] GPU '" << prop.name << "', SMs=" << prop.multiProcessorCount
              << ", mem_free=" << (available_gpu_memory/(1024*1024)) << "MB, est/problem="
              << (memory_per_problem/(1024*1024)) << "MB, streams=" << max_concurrent_problems
              << std::endl;

    initializeResourcePool();
    initializeIOThreads();

    // Launch worker threads
    worker_threads.reserve(max_concurrent_problems);
    for (size_t i = 0; i < max_concurrent_problems; ++i) {
        worker_threads.emplace_back(&BatchACMMP::workerFunction, this);
    }
}

BatchACMMP::~BatchACMMP() {
    // Stop processing
    stopping_.store(true);
    queue_cv_.notify_all();
    resource_cv_.notify_all();

    // Stop I/O
    io_stopping_.store(true);
    io_cv_.notify_all();

    // Join all threads
    for (auto& t : worker_threads) {
        if (t.joinable()) t.join();
    }
    for (auto& t : io_threads_) {
        if (t.joinable()) t.join();
    }

    // Clean up resources
    for (auto& res : resource_pool) res.reset();
    for (auto& s : streams) {
        if (s) CUDA_CHECK(cudaStreamDestroy(s));
    }
}

void BatchACMMP::initializeIOThreads(size_t num_io_threads) {
    io_threads_.reserve(num_io_threads);
    for (size_t i = 0; i < num_io_threads; ++i) {
        io_threads_.emplace_back(&BatchACMMP::ioWorkerFunction, this);
    }
}

void BatchACMMP::ioWorkerFunction() {
    while (!io_stopping_.load()) {
        ProcessedResult result;
        bool has_work = false;

        // Wait for I/O work
        {
            std::unique_lock<std::mutex> lk(io_mutex_);
            io_cv_.wait(lk, [&]{
                return io_stopping_.load() || !io_queue_.empty();
            });
            if (io_stopping_.load() && io_queue_.empty()) break;
            
            if (!io_queue_.empty()) {
                result = std::move(io_queue_.front());
                io_queue_.pop();
                has_work = true;
            }
        }

        if (has_work) {
            // Process result using callback or default file writing
            const Problem& problem = all_problems[result.problem_idx];
            
            // Convert to OpenCV format
            cv::Mat_<float> depths(result.height, result.width, 0.0f);
            cv::Mat_<cv::Vec3f> normals(result.height, result.width, cv::Vec3f(0,0,0));
            cv::Mat_<float> costs(result.height, result.width, 0.0f);

            const int W = result.width, H = result.height;
            for (int y = 0; y < H; ++y) {
                float* drow = depths.ptr<float>(y);
                cv::Vec3f* nrow = normals.ptr<cv::Vec3f>(y);
                float* crow = costs.ptr<float>(y);
                for (int x = 0; x < W; ++x) {
                    const int i = y * W + x;
                    const float4 ph = result.planes[i];
                    drow[x] = ph.w;
                    nrow[x] = cv::Vec3f(ph.x, ph.y, ph.z);
                    crow[x] = result.costs[i];
                }
            }

            // Use callback if provided, otherwise default file writing
            if (result_callback_) {
                result_callback_(result.problem_idx, depths, normals, costs);
            } else {
                // Default file writing
                std::stringstream result_path;
                result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) 
                           << std::setfill('0') << problem.ref_image_id;
                std::string result_folder = result_path.str();
                
                // Ensure directory exists
                create_directories_recursive(result_folder);
                
                std::string suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
                std::string depth_path = result_folder + suffix;
                std::string normal_path = result_folder + "/normals.dmb";
                std::string cost_path = result_folder + "/costs.dmb";
                
                writeDepthDmb(depth_path, depths);
                writeNormalDmb(normal_path, normals);
                writeDepthDmb(cost_path, costs);
            }

            int written = problems_written_.fetch_add(1) + 1;
            if (written % 10 == 0) {
                std::cout << "[I/O] Written " << written << "/" << problems_enqueued_.load() << " results\n";
            }
        }
    }
}

void BatchACMMP::enqueueResult(ProcessedResult&& result) {
    {
        std::lock_guard<std::mutex> lk(io_mutex_);
        io_queue_.push(std::move(result));
    }
    io_cv_.notify_one();
}

size_t BatchACMMP::estimateMemoryPerProblem(const Problem& problem) {
    // Read camera to get dims
    std::stringstream cam_path;
    cam_path << dense_folder << "/cams/" << std::setw(8) << std::setfill('0')
             << problem.ref_image_id << "_cam.txt";
    Camera cam = ReadCamera(cam_path.str());

    const size_t W = cam.width, H = cam.height;
    const size_t N = 1 + problem.src_image_ids.size();

    const size_t bytes_image   = W * H * sizeof(float);
    const size_t bytes_plane4  = W * H * sizeof(float4);
    const size_t bytes_float   = W * H * sizeof(float);

    size_t textures = N * bytes_image + N * bytes_float;
    size_t working  = 2*bytes_plane4 + 2*bytes_float + bytes_float;
    size_t misc     = W * H * (sizeof(curandState) + sizeof(unsigned int));

    return (textures + working + misc) + (64 * 1024 * 1024);
}

size_t BatchACMMP::getAvailableGPUMemory() {
    size_t free_mem=0, total=0;
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total));
    return free_mem;
}

void BatchACMMP::initializeResourcePool() {
    streams.resize(max_concurrent_problems);
    resource_pool.resize(max_concurrent_problems);

    int prio_low=0, prio_high=0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));
    
    for (size_t i = 0; i < max_concurrent_problems; ++i) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, prio_high));

        std::unique_ptr<ProblemGPUResources> res(new ProblemGPUResources());
        res->stream_id = (int)i;
        res->stream    = streams[i];

        available_resources.push(res.get());
        resource_pool[i] = std::move(res);
    }
}

ProblemGPUResources* BatchACMMP::acquireResources() {
    std::unique_lock<std::mutex> lk(resource_mutex_);
    resource_cv_.wait(lk, [&]{
        return !available_resources.empty() || stopping_.load();
    });
    if (stopping_.load()) return nullptr;
    auto* r = available_resources.front();
    available_resources.pop();
    return r;
}

void BatchACMMP::releaseResources(ProblemGPUResources* r) {
    if (!r) return;
    {
        std::lock_guard<std::mutex> lk(resource_mutex_);
        available_resources.push(r);
    }
    resource_cv_.notify_one();
}

void BatchACMMP::processAllProblems() {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        for (int i = 0; i < (int)all_problems.size(); ++i) {
            problem_queue_.push(i);
        }
        problems_enqueued_.store((int)all_problems.size());
    }
    queue_cv_.notify_all();
    std::cout << "[BatchACMMP] Enqueued " << all_problems.size()
              << " problems across " << max_concurrent_problems << " streams.\n";
}

void BatchACMMP::processBatch(const std::vector<int>& idxs) {
    {
        std::lock_guard<std::mutex> lk(queue_mutex_);
        for (int i : idxs) problem_queue_.push(i);
        problems_enqueued_.fetch_add((int)idxs.size());
    }
    queue_cv_.notify_all();
}

void BatchACMMP::workerFunction() {
    while (!stopping_.load()) {
        int idx = -1;

        // Wait for work
        {
            std::unique_lock<std::mutex> lk(queue_mutex_);
            queue_cv_.wait(lk, [&]{
                return stopping_.load() || !problem_queue_.empty();
            });
            if (stopping_.load()) break;
            idx = problem_queue_.front();
            problem_queue_.pop();
        }

        auto* res = acquireResources();
        if (!res) break;

        processProblemOnStream(idx, res);
        releaseResources(res);

        int done = problems_completed_.fetch_add(1) + 1;
        if (done == problems_enqueued_.load()) {
            queue_cv_.notify_all();
        }
    }
}

void BatchACMMP::processProblemOnStream(int problem_idx, ProblemGPUResources* resources) {
    const Problem& problem = all_problems[problem_idx];
    cudaStream_t stream = resources->stream;

    std::cout << "[S" << resources->stream_id << "] Problem " << problem_idx
              << " (ref " << problem.ref_image_id << ")\n";

    ACMMP acmmp;
    if (geom_consistency) acmmp.SetGeomConsistencyParams(multi_geometry);
    if (hierarchy)        acmmp.SetHierarchyParams();

    acmmp.SetStream(stream);
    acmmp.InuputInitialization(dense_folder, all_problems, problem_idx);
    acmmp.CudaSpaceInitialization(dense_folder, problem);
    acmmp.RunPatchMatch();

    // Synchronize stream for this specific problem
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int width  = acmmp.GetReferenceImageWidth();
    const int height = acmmp.GetReferenceImageHeight();

    // Create result and immediately enqueue for I/O
    ProcessedResult result;
    result.problem_idx = problem_idx;
    result.width = width;
    result.height = height;
    result.planes.resize(size_t(width) * size_t(height));
    result.costs.resize(size_t(width) * size_t(height));

    // Extract results efficiently
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int c = y * width + x;
            result.planes[c] = acmmp.GetPlaneHypothesis(c);
            result.costs[c]  = acmmp.GetCost(c);
        }
    }

    // Immediately enqueue for I/O processing
    enqueueResult(std::move(result));
}

void BatchACMMP::waitForCompletion() {
    // Wait for processing to complete
    std::unique_lock<std::mutex> lk(queue_mutex_);
    queue_cv_.wait(lk, [&]{
        return problems_completed_.load() >= problems_enqueued_.load();
    });
    
    // Wait for I/O to complete
    while (problems_written_.load() < problems_enqueued_.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }
    
    // Final stream sync for safety
    for (auto& s : streams) CUDA_CHECK(cudaStreamSynchronize(s));
    
    std::cout << "[BatchACMMP] All " << problems_enqueued_.load() 
              << " problems processed and written.\n";
}