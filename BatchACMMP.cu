#include "BatchACMMP.h"
#include "ACMMP_device.cuh"

#include <iostream>
#include <algorithm>
#include <chrono>

// ---------- ProblemGPUResources impl ----------
ProblemGPUResources::ProblemGPUResources() {
    // zero-init arrays
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

    // We don't destroy the stream here; BatchACMMP owns it.
}

// ---------- BatchACMMP impl ----------

BatchACMMP::BatchACMMP(const std::string& dense_folder_, 
                       const std::vector<Problem>& problems,
                       bool geom_consistency_,
                       bool planar_prior_,
                       bool hierarchy_)
    : dense_folder(dense_folder_), all_problems(problems),
      geom_consistency(geom_consistency_), planar_prior(planar_prior_),
      hierarchy(hierarchy_), multi_geometry(false)
{
    // Device props
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));

    // concurrent kernels?
    int concurrentKernels = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&concurrentKernels, cudaDevAttrConcurrentKernels, 0));

    available_gpu_memory = getAvailableGPUMemory();
    memory_per_problem   = problems.empty() ? (size_t)500 * 1024 * 1024
                                            : estimateMemoryPerProblem(problems[0]);

    // Reserve 20% headroom
    const size_t usable = size_t(double(available_gpu_memory) * 0.8);
    size_t by_mem = std::max<size_t>(1, usable / std::max<size_t>(memory_per_problem, 1));

    // A practical cap: streams ~ SM count/2 if kernels are heavy, otherwise 8-12 is typical.
    size_t by_sm = (prop.multiProcessorCount > 0) ? std::max<size_t>(1, prop.multiProcessorCount / 2) : 8;
    size_t cap   = concurrentKernels ? 16 : 1; // if device can't overlap, force 1

    max_concurrent_problems = std::min<size_t>(std::min(by_mem, by_sm), cap);
    if (max_concurrent_problems == 0) max_concurrent_problems = 1;

    std::cout << "[BatchACMMP] GPU '" << prop.name << "', SMs=" << prop.multiProcessorCount
              << ", mem_free=" << (available_gpu_memory/(1024*1024)) << "MB, est/problem="
              << (memory_per_problem/(1024*1024)) << "MB, streams=" << max_concurrent_problems
              << std::endl;

    initializeResourcePool();

    // Launch exactly one worker per stream (more just burns CPU)
    worker_threads.reserve(max_concurrent_problems);
    for (size_t i = 0; i < max_concurrent_problems; ++i) {
        worker_threads.emplace_back(&BatchACMMP::workerFunction, this);
    }
}

BatchACMMP::~BatchACMMP() {
    // Stop accepting work
    stopping_.store(true);
    queue_cv_.notify_all();
    resource_cv_.notify_all();

    for (auto& t : worker_threads) {
        if (t.joinable()) t.join();
    }

    // Clean up resources (streams last)
    for (auto& res : resource_pool) res.reset();

    for (auto& s : streams) {
        if (s) CUDA_CHECK(cudaStreamDestroy(s));
    }
}

void BatchACMMP::setMaxConcurrentProblems(size_t max_problems) {
    // No dynamic shrink/grow implemented for simplicity.
    // You can wire this to rebuild the pool if needed.
    (void)max_problems;
}

size_t BatchACMMP::estimateMemoryPerProblem(const Problem& problem) {
    // Read camera to get dims
    std::stringstream cam_path;
    cam_path << dense_folder << "/cams/" << std::setw(8) << std::setfill('0')
             << problem.ref_image_id << "_cam.txt";
    Camera cam = ReadCamera(cam_path.str());

    const size_t W = cam.width, H = cam.height;
    const size_t N = 1 + problem.src_image_ids.size();

    // Be a bit more explicit with channels; assume float textures, and a couple of aux buffers.
    const size_t bytes_image   = W * H * sizeof(float);           // grayscale (adjust if RGBA)
    const size_t bytes_plane4  = W * H * sizeof(float4);
    const size_t bytes_float   = W * H * sizeof(float);

    size_t textures = N * bytes_image               // images
                    + N * bytes_float;              // depths (if used)

    size_t working  = 2*bytes_plane4                // hypotheses + scaled
                    + 2*bytes_float                 // cost + pre_cost
                    + bytes_float;                  // depths
    size_t misc     = W * H * (sizeof(curandState) + sizeof(unsigned int)); // RNG + selected_views

    // Round up a little
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
    // prio_high is numerically highest priority (smallest value).
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

        // wait for work
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
        if (!res) break; // stopping

        processProblemOnStream(idx, res);
        releaseResources(res);

        int done = problems_completed_.fetch_add(1) + 1;
        if (done == problems_enqueued_.load()) {
            // Wake up waiters
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

    // *** CRITICAL: make ACMMP use our stream (see tiny hook below) ***
    acmmp.SetStream(stream);

    acmmp.InuputInitialization(dense_folder, all_problems, problem_idx);
    acmmp.CudaSpaceInitialization(dense_folder, problem);
    acmmp.RunPatchMatch();   // should enqueue work on 'stream'

    // Ensure device work for this problem is done before reading results
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int width  = acmmp.GetReferenceImageWidth();
    const int height = acmmp.GetReferenceImageHeight();

    std::vector<float4> planes(size_t(width) * size_t(height));
    std::vector<float>  costs(size_t(width) * size_t(height));

    // If your ACMMP has bulk getters, prefer those; otherwise fall back to per-pixel.
    // (Consider adding acmmp.CopyOutputsToHost(planes.data(), costs.data()) for speed.)
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const int c = y * width + x;
            planes[c] = acmmp.GetPlaneHypothesis(c);
            costs[c]  = acmmp.GetCost(c);
        }
    }

    {
        std::lock_guard<std::mutex> lk(results_mutex_);
        StoredResult& slot = results_[problem_idx];
        slot.width  = width;
        slot.height = height;
        slot.planes = std::move(planes);
        slot.costs  = std::move(costs);
    }
}

void BatchACMMP::waitForCompletion() {
    std::unique_lock<std::mutex> lk(queue_mutex_);
    queue_cv_.wait(lk, [&]{
        return problems_completed_.load() >= problems_enqueued_.load();
    });
    // Stream sync for safety
    for (auto& s : streams) CUDA_CHECK(cudaStreamSynchronize(s));
}

void BatchACMMP::extractResults(int problem_idx,
                                cv::Mat_<float>& depths,
                                cv::Mat_<cv::Vec3f>& normals,
                                cv::Mat_<float>& costs) {
    StoredResult res;
    {
        std::lock_guard<std::mutex> lk(results_mutex_);
        auto it = results_.find(problem_idx);
        if (it == results_.end()) {
            depths.release(); normals.release(); costs.release();
            std::cerr << "[BatchACMMP] No result for problem " << problem_idx << "\n";
            return;
        }
        res = it->second; // copy metadata and vectors
    }

    depths  = cv::Mat_<float>(res.height, res.width, 0.0f);
    normals = cv::Mat_<cv::Vec3f>(res.height, res.width, cv::Vec3f(0,0,0));
    costs   = cv::Mat_<float>(res.height, res.width, 0.0f);

    const int W = res.width, H = res.height;
    for (int y = 0; y < H; ++y) {
        float* drow = depths.ptr<float>(y);
        cv::Vec3f* nrow = normals.ptr<cv::Vec3f>(y);
        float* crow = costs.ptr<float>(y);
        for (int x = 0; x < W; ++x) {
            const int i = y * W + x;
            const float4 ph = res.planes[i];
            drow[x] = ph.w;
            nrow[x] = cv::Vec3f(ph.x, ph.y, ph.z);
            crow[x] = res.costs[i];
        }
    }
}
