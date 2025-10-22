// ========================================
// BatchACMMP.cu - Implementation with parallel disk I/O
// ========================================

#include "BatchACMMP.h"
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

void checkCudaLimits() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("Max Texture 2D: %dx%d\n", prop.maxTexture2D[0], prop.maxTexture2D[1]);
    printf("Max Texture 2D Layered: %dx%dx%d\n", 
           prop.maxTexture2DLayered[0], prop.maxTexture2DLayered[1], prop.maxTexture2DLayered[2]);
    printf("Max Surface 2D: %dx%d\n", prop.maxSurface2D[0], prop.maxSurface2D[1]);
    printf("Max Grid Size: %dx%dx%d\n", 
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    
    // Critical: texture reference limit
    printf("Max Textures per kernel: %d\n", prop.maxTexture1DLinear);
    printf("Total constant memory: %zu\n", prop.totalConstMem);
}

// ProblemGPUResources implementation
ProblemGPUResources::ProblemGPUResources() {
    for (int i = 0; i < MAX_IMAGES; ++i) {
        cuArray[i] = nullptr;
        cuDepthArray[i] = nullptr;
    }
}

void ProblemGPUResources::allocate(int max_width, int max_height, int max_images) {
    // This function is called once per resource object when the pool is initialized.
    
    // Allocate arrays for images and depths using the maximum possible dimensions.
    for (int i = 0; i < max_images; ++i) {
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        CUDA_CHECK(cudaMallocArray(&cuArray[i], &channelDesc, max_width, max_height));
        CUDA_CHECK(cudaMallocArray(&cuDepthArray[i], &channelDesc, max_width, max_height));
    }

    // Allocate all other required device memory buffers.
    CUDA_CHECK(cudaMalloc(&cameras_cuda, sizeof(Camera) * max_images));
    CUDA_CHECK(cudaMalloc(&texture_objects_cuda, sizeof(cudaTextureObjects)));
    CUDA_CHECK(cudaMalloc(&texture_depths_cuda, sizeof(cudaTextureObjects)));
    CUDA_CHECK(cudaMalloc(&plane_hypotheses_cuda, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&scaled_plane_hypotheses_cuda, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&costs_cuda, sizeof(float) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&pre_costs_cuda, sizeof(float) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&rand_states_cuda, sizeof(curandState) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&selected_views_cuda, sizeof(unsigned int) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&depths_cuda, sizeof(float) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&prior_planes_cuda, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&plane_masks_cuda, sizeof(unsigned int) * max_width * max_height));

    // Allocate pinned host memory for high-speed asynchronous transfers.
    CUDA_CHECK(cudaMallocHost(&planes_host_pinned, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMallocHost(&costs_host_pinned, sizeof(float) * max_width * max_height));
}

void BatchACMMP::initializeResourcePool() {
    // Determine the maximum resource dimensions needed for any problem in the batch.
    // This ensures all our pooled resources are large enough.
    int max_width = 0, max_height = 0, max_images = 0;
    for (const auto& p : all_problems) {
        // A robust way to get dimensions would be to read the camera file for each problem.
        // For simplicity, we use a fixed upper bound, but reading the files is better.
        // This is a placeholder; you should replace it with actual dimension fetching logic
        // if your image sizes vary significantly.
        std::stringstream cam_path;
        cam_path << dense_folder << "/cams/" << std::setw(8) << std::setfill('0') << p.ref_image_id << "_cam.txt";
        Camera cam = ReadCamera(cam_path.str());
        
        max_width = std::max(max_width, (int)cam.width);
        max_height = std::max(max_height, (int)cam.height);
        max_images = std::max(max_images, (int)(1 + p.src_image_ids.size()));
    }
    // Clamp max_images to the maximum supported by the static array.
    max_images = std::min(max_images, MAX_IMAGES);
    
    std::cout << "[BatchACMMP] Allocating resources for max dimensions: " 
              << max_width << "x" << max_height << " with up to " << max_images << " images." << std::endl;

    streams.resize(max_concurrent_problems);
    resource_pool.resize(max_concurrent_problems);

    int prio_low=0, prio_high=0;
    CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));
    
    for (size_t i = 0; i < max_concurrent_problems; ++i) {
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, prio_high));

        std::unique_ptr<ProblemGPUResources> res(new ProblemGPUResources());
        res->stream_id = (int)i;
        res->stream = streams[i];

        // Allocate the GPU memory for this resource object.
        res->allocate(max_width, max_height, max_images);

        available_resources.push(res.get());
        resource_pool[i] = std::move(res);
    }
    
    // Launch GPU worker threads.
    gpu_worker_threads.reserve(max_concurrent_problems);
    for (size_t i = 0; i < max_concurrent_problems; ++i) {
        gpu_worker_threads.emplace_back(&BatchACMMP::gpuWorkerFunction, this);
    }
    
    std::cout << "[BatchACMMP] Created " << max_concurrent_problems << " GPU worker threads" << std::endl;
}


ProblemGPUResources::~ProblemGPUResources() { 
    cleanup(); 
}

// In BatchACMMP.cu, replace the entire cleanup function with this one.

void ProblemGPUResources::cleanup() {
    // Don't synchronize the stream here - it's owned by BatchACMMP
    // Just clean up the resources allocated by this object
    
    for (int i = 0; i < MAX_IMAGES; ++i) {
        if (cuArray[i]) { 
            cudaFreeArray(cuArray[i]); 
            cuArray[i] = nullptr; 
        }
        if (cuDepthArray[i]) { 
            cudaFreeArray(cuDepthArray[i]); 
            cuDepthArray[i] = nullptr; 
        }
    }

    // C++11 COMPATIBLE FIX: Define the lambda to take void*&
    auto safeFree = [](void*& ptr, const char* name) {
        if (ptr) {
            cudaError_t err = cudaFree(ptr);
            if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
                // Don't print errors during shutdown
            }
            ptr = nullptr;
        }
    };

    // C++11 COMPATIBLE FIX: Add a (void*&) cast to every call
    safeFree((void*&)cameras_cuda, "cameras_cuda");
    safeFree((void*&)texture_objects_cuda, "texture_objects_cuda");
    safeFree((void*&)texture_depths_cuda, "texture_depths_cuda");
    safeFree((void*&)plane_hypotheses_cuda, "plane_hypotheses_cuda");
    safeFree((void*&)scaled_plane_hypotheses_cuda, "scaled_plane_hypotheses_cuda");
    safeFree((void*&)costs_cuda, "costs_cuda");
    safeFree((void*&)pre_costs_cuda, "pre_costs_cuda");
    safeFree((void*&)rand_states_cuda, "rand_states_cuda");
    safeFree((void*&)selected_views_cuda, "selected_views_cuda");
    safeFree((void*&)depths_cuda, "depths_cuda");
    safeFree((void*&)prior_planes_cuda, "prior_planes_cuda");
    safeFree((void*&)plane_masks_cuda, "plane_masks_cuda");

    if (planes_host_pinned) { 
        cudaFreeHost(planes_host_pinned); 
        planes_host_pinned = nullptr; 
    }
    if (costs_host_pinned) { 
        cudaFreeHost(costs_host_pinned); 
        costs_host_pinned = nullptr; 
    }
    
    // Clear the stream reference (don't destroy it - BatchACMMP owns it)
    stream = nullptr;
}

// BatchACMMP implementation
BatchACMMP::BatchACMMP(const std::string& dense_folder_, 
                       const std::vector<Problem>& problems,
                       bool geom_consistency_,
                       bool planar_prior_,
                       bool hierarchy_,
                       bool multi_geometry_,
                       size_t mask_disk_queue_size_)
    : dense_folder(dense_folder_), all_problems(problems),
      geom_consistency(geom_consistency_), planar_prior(planar_prior_),
      hierarchy(hierarchy_), multi_geometry(multi_geometry_),mask_disk_queue_size(mask_disk_queue_size_) 
{
    // Device properties
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    checkCudaLimits();
    available_gpu_memory = getAvailableGPUMemory();
    memory_per_problem = problems.empty() ? (size_t)500 * 1024 * 1024
                                          : estimateMemoryPerProblem(problems[0]);

    // Calculate optimal GPU concurrency
    size_t usable_gpu = size_t(double(available_gpu_memory) * 0.75);
    size_t by_gpu_mem = std::max<size_t>(1, usable_gpu / memory_per_problem);
    
    // Hardware-based limits
    size_t hardware_threads = std::thread::hardware_concurrency();
    size_t by_sm = (prop.multiProcessorCount >= 20) ? 8 :
                   (prop.multiProcessorCount >= 10) ? 6 :
                   (prop.multiProcessorCount >= 5)  ? 4 : 2;
    
    max_concurrent_problems = std::min({by_gpu_mem, by_sm, size_t(12)});
    max_concurrent_problems = std::max<size_t>(1, 12);
    
    // Separate disk writer threads - optimize for disk I/O
    num_disk_writers = std::min<size_t>(4, std::max<size_t>(2, hardware_threads / 4));
    
    std::cout << "[BatchACMMP] Configuration:" << std::endl;
    std::cout << "  GPU Streams: " << max_concurrent_problems << std::endl;
    std::cout << "  Disk Writers: " << num_disk_writers << std::endl;
    std::cout << "  GPU Memory: " << (available_gpu_memory/(1024*1024)) << "MB free, "
              << (memory_per_problem/(1024*1024)) << "MB/problem" << std::endl;

    initializeResourcePool();
    initializeDiskWriters();
}

// ========================================
// Fixed BatchACMMP destructor in BatchACMMP.cu
// ========================================

BatchACMMP::~BatchACMMP() {
    // Step 1: Signal all threads to stop
    stopping_gpu_.store(true);
    stopping_disk_.store(true);
    
    // Step 2: Wake up all waiting threads
    gpu_queue_cv_.notify_all();
    disk_queue_cv_.notify_all();
    disk_queue_space_cv_.notify_all();  
    resource_cv_.notify_all();
    
    // Step 3: Join worker threads
    for (auto& t : gpu_worker_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    for (auto& t : disk_writer_threads) {
        if (t.joinable()) {
            t.join();
        }
    }
    
    // Step 4: Clean up GPU resources FIRST (they may reference streams)
    for (auto& res : resource_pool) {
        if (res && res->stream) {
            cudaStreamSynchronize(res->stream);  // sync per stream
            res->cleanup();
        }
    }

    // Step 5: NOW destroy the streams
    for (auto& s : streams) {
        if (s) {
            cudaStreamDestroy(s);
            s = nullptr;
        }
    }
        
    // Step 7: Final device synchronization
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess && err != cudaErrorCudartUnloading) {
        // Ignore errors during shutdown
    }
    
    std::cout << "[BatchACMMP] Shutdown complete. Peak memory: " 
              << getPeakMemoryUsage() << "MB" << std::endl;
}

size_t BatchACMMP::getSystemRAM() {
    std::ifstream meminfo("/proc/meminfo");
    std::string line;
    while (std::getline(meminfo, line)) {
        if (line.substr(0, 8) == "MemTotal") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            return std::stoull(value) * 1024;
        }
    }
    return 16ULL * 1024 * 1024 * 1024; // Default 16GB
}

size_t BatchACMMP::estimateMemoryPerProblem(const Problem& problem) {
    std::stringstream cam_path;
    cam_path << dense_folder << "/cams/" << std::setw(8) << std::setfill('0')
             << problem.ref_image_id << "_cam.txt";
    Camera cam = ReadCamera(cam_path.str());

    const size_t W = cam.width, H = cam.height;
    const size_t N = 1 + problem.src_image_ids.size();

    size_t textures = N * W * H * (sizeof(float) + sizeof(float)); // images + depths
    size_t working = W * H * (2*sizeof(float4) + 3*sizeof(float)); // hypotheses + costs
    size_t misc = W * H * (sizeof(curandState) + sizeof(unsigned int));
    
    return (textures + working + misc) * 130 / 100; // 30% overhead
}

size_t BatchACMMP::getAvailableGPUMemory() {
    size_t free_mem=0, total=0;
    cudaDeviceSynchronize();
    CUDA_CHECK(cudaMemGetInfo(&free_mem, &total));
    return free_mem;
}

size_t BatchACMMP::getProcessMemoryUsage() const {
    std::ifstream status("/proc/self/status");
    std::string line;
    while (std::getline(status, line)) {
        if (line.substr(0, 6) == "VmRSS:") {
            std::istringstream iss(line);
            std::string key, value, unit;
            iss >> key >> value >> unit;
            return std::stoull(value) * 1024;
        }
    }
    return 0;
}

void BatchACMMP::initializeDiskWriters() {
    // Launch disk writer threads
    disk_writer_threads.reserve(num_disk_writers);
    for (size_t i = 0; i < num_disk_writers; ++i) {
        disk_writer_threads.emplace_back(&BatchACMMP::diskWriterFunction, this);
    }
    
    std::cout << "[BatchACMMP] Created " << num_disk_writers << " disk writer threads" << std::endl;
}

ProblemGPUResources* BatchACMMP::acquireResources() {
    std::unique_lock<std::mutex> lk(resource_mutex_);
    resource_cv_.wait(lk, [&]{
        return !available_resources.empty() || stopping_gpu_.load();
    });
    if (stopping_gpu_.load()) return nullptr;
    
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
        std::lock_guard<std::mutex> lk(gpu_queue_mutex_);
        for (int i = 0; i < (int)all_problems.size(); ++i) {
            gpu_work_queue_.push(i);
        }
        problems_enqueued_.store((int)all_problems.size());
    }
    gpu_queue_cv_.notify_all();
    
    std::cout << "[BatchACMMP] Enqueued " << all_problems.size() << " problems" << std::endl;
    std::cout << "  GPU Processing: " << max_concurrent_problems << " parallel streams" << std::endl;
    std::cout << "  Disk Writing: " << num_disk_writers << " parallel writers" << std::endl;
}

void BatchACMMP::gpuWorkerFunction() {
    while (!stopping_gpu_.load()) {
        int idx = -1;

        // Get work from GPU queue
        {
            std::unique_lock<std::mutex> lk(gpu_queue_mutex_);
            gpu_queue_cv_.wait(lk, [&]{
                return stopping_gpu_.load() || !gpu_work_queue_.empty();
            });
            if (stopping_gpu_.load()) break;
            if (gpu_work_queue_.empty()) continue;
            
            idx = gpu_work_queue_.front();
            gpu_work_queue_.pop();
        }

        auto* res = acquireResources();
        if (!res) break;

        try {
            processProblemOnStream(idx, res);
        } catch (const std::exception& e) {
            std::cerr << "[GPU Worker] Exception processing problem " << idx << ": " << e.what() << std::endl;
        }
        
        releaseResources(res);

        int done = gpu_completed_.fetch_add(1) + 1;
        if (done % 50 == 0 || done == problems_enqueued_.load()) {
            std::cout << "[GPU Progress] " << done << "/" << problems_enqueued_.load() 
                      << " (" << (100 * done / problems_enqueued_.load()) << "%) - "
                      << "Disk pending: " << getPendingDiskWrites() << std::endl;
        }
    }
}

void BatchACMMP::diskWriterFunction() {
    while (!stopping_disk_.load()) {
        CompletedResult result;
        bool queue_was_full = false;
        
        // Get completed result from queue
        {
            std::unique_lock<std::mutex> lk(disk_queue_mutex_);
            disk_queue_cv_.wait(lk, [&]{
                return stopping_disk_.load() || !disk_write_queue_.empty();
            });
            if (stopping_disk_.load() && disk_write_queue_.empty()) break;
            if (disk_write_queue_.empty()) continue;
            
            queue_was_full = (disk_write_queue_.size() >= mask_disk_queue_size);
            result = std::move(disk_write_queue_.front());
            disk_write_queue_.pop();
        }
        
        // Signal that space is available (OUTSIDE the lock!)
        if (queue_was_full) {
            disk_queue_space_cv_.notify_all();
        }
        
        try {
            writeProblemToDisk(std::move(result));
        } catch (const std::exception& e) {
            std::cerr << "[Disk Writer] Exception writing problem: " << e.what() << std::endl;
        }
        
        int done = disk_completed_.fetch_add(1) + 1;
        if (done % 100 == 0 || done == problems_enqueued_.load()) {
            std::cout << "[Disk Progress] Saved " << done << "/" << problems_enqueued_.load() 
                      << " (" << (100 * done / problems_enqueued_.load()) << "%)" << std::endl;
        }
    }
}

void BatchACMMP::processProblemOnStream(int problem_idx, ProblemGPUResources* resources) {
    const Problem& problem = all_problems[problem_idx];
    cudaStream_t stream = resources->stream;
    
    // Set the device explicitly for this thread.
    cudaSetDevice(0);
    
    active_gpu_problems_.fetch_add(1);
    
    try {
        // cudaStreamSynchronize(stream);
        // Ensure the stream is valid before creating ACMMP.
        cudaError_t stream_check = cudaStreamQuery(stream);
        if (stream_check == cudaErrorInvalidResourceHandle) {
            std::cerr << "Invalid stream for problem " << problem_idx << ", creating new stream" << std::endl;
            cudaStreamCreate(&stream);
            resources->stream = stream;
        }
        
        // Process in an isolated scope to manage ACMMP's lifetime.
        {
            ACMMP acmmp;
            if (geom_consistency) acmmp.SetGeomConsistencyParams(multi_geometry);
            if (hierarchy) acmmp.SetHierarchyParams();

            // Set the stream for the ACMMP object to use for async operations.
            acmmp.SetStream(stream);
            
            // Initialize host-side data (reading images from disk).
            acmmp.InuputInitialization(dense_folder, all_problems, problem_idx);
            
            // Initialize CUDA space by copying data to the pre-allocated GPU buffers.
            acmmp.CudaSpaceInitialization(dense_folder, problem, resources);
            
            // Ensure everything is set up before running.
            cudaError_t pre_run_check = cudaGetLastError();
            if (pre_run_check != cudaSuccess) {
                std::cerr << "Pre-run error: " << cudaGetErrorString(pre_run_check) << std::endl;
                throw std::runtime_error("CUDA setup failed");
            }
            
            // Run the main algorithm, using the provided GPU resources.
            acmmp.RunPatchMatch(resources);
            
            
            // Extract results...
            const int width = acmmp.GetReferenceImageWidth();
            const int height = acmmp.GetReferenceImageHeight();

            cv::Mat_<float> depths(height, width);
            cv::Mat_<cv::Vec3f> normals(height, width);
            cv::Mat_<float> costs(height, width);

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int c = y * width + x;
                    const float4 plane_hypothesis = acmmp.GetPlaneHypothesis(c);
                    depths(y, x) = plane_hypothesis.w;
                    normals(y, x) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                    costs(y, x) = acmmp.GetCost(c);
                }
            }

            // Queue the results for asynchronous disk writing with backpressure
            {
                std::unique_lock<std::mutex> lk(disk_queue_mutex_);
                // Wait if queue is full - this provides backpressure
                disk_queue_space_cv_.wait(lk, [&]{
                    return disk_write_queue_.size() < mask_disk_queue_size || stopping_disk_.load();
                });
                
                size_t queue_size = disk_write_queue_.size();
                disk_write_queue_.emplace(problem_idx, problem, 
                                         std::move(depths), std::move(normals), 
                                         std::move(costs), geom_consistency);
                
                // Warn if queue is getting large
                // if (queue_size > mask_disk_queue_size * 3 / 4) {  // 75%
                //     std::cout << "[Backpressure] Disk queue at " << queue_size << "/" 
                //               << mask_disk_queue_size << " - GPU throttled" << std::endl;
                // }
            }
            disk_queue_cv_.notify_one();
        } // ACMMP destructor is called here, freeing only its HOST memory.
        
        // Final sync after ACMMP is destroyed.
        // cudaStreamSynchronize(stream);
        
    } catch (const std::exception& e) {
        std::cerr << "[Problem " << problem_idx << "] Exception: " << e.what() << std::endl;
        cudaGetLastError(); // Clear any pending errors.
        throw;
    }
    
    active_gpu_problems_.fetch_sub(1);
}

void BatchACMMP::writeProblemToDisk(CompletedResult&& result) {
    // Create result folder
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP/2333_" << std::setw(8) 
                << std::setfill('0') << result.problem.ref_image_id;
    std::string result_folder = result_path.str();
    
    // Create directory (mkdir is thread-safe on most systems)
    makeDir(result_folder);
    
    // Write files
    std::string suffix = result.geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    
    writeDepthDmb(depth_path, result.depths);
    writeNormalDmb(normal_path, result.normals);
    writeDepthDmb(cost_path, result.costs);
}

void BatchACMMP::waitForGPUCompletion() {
    std::unique_lock<std::mutex> lk(gpu_queue_mutex_);
    gpu_queue_cv_.wait(lk, [&]{
        return gpu_completed_.load() >= problems_enqueued_.load();
    });
    
    for (auto& s : streams) {
        if (s) CUDA_CHECK(cudaStreamSynchronize(s));
    }
    // cudaDeviceSynchronize();
    
    std::cout << "[BatchACMMP] GPU processing complete!" << std::endl;
}

void BatchACMMP::waitForDiskCompletion() {
    std::unique_lock<std::mutex> lk(disk_queue_mutex_);
    disk_queue_cv_.wait(lk, [&]{
        return disk_completed_.load() >= problems_enqueued_.load();
    });
    
    std::cout << "[BatchACMMP] Disk writing complete!" << std::endl;
}

void BatchACMMP::waitForCompletion() {
    waitForGPUCompletion();
    
    size_t pending = getPendingDiskWrites();
    if (pending > 0) {
        std::cout << "[BatchACMMP] GPU complete. Flushing remaining " 
                  << pending << " results to disk..." << std::endl;
    }
    
    waitForDiskCompletion();
    
    // Verify all problems were written
    int total = problems_enqueued_.load();
    int written = disk_completed_.load();
    if (written == total) {
        std::cout << "[BatchACMMP] ✓ All " << total << " problems written successfully!" << std::endl;
    } else {
        std::cerr << "[BatchACMMP] ✗ WARNING: Only " << written << "/" << total 
                  << " problems written!" << std::endl;
    }
}

size_t BatchACMMP::getPeakMemoryUsage() const {
    std::lock_guard<std::mutex> lk(memory_mutex_);
    return peak_memory_usage_ / (1024 * 1024);
}

size_t BatchACMMP::getCurrentMemoryUsage() const {
    return getProcessMemoryUsage() / (1024 * 1024);
}

size_t BatchACMMP::getActiveGPUProblems() const {
    return active_gpu_problems_.load();
}

size_t BatchACMMP::getPendingDiskWrites() const {
    std::lock_guard<std::mutex> lk(disk_queue_mutex_);
    return disk_write_queue_.size();
}

size_t BatchACMMP::getCompletedDiskWrites() const {
    return disk_completed_.load();
}

// ======================================== 