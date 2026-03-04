// ========================================
// BatchACMMP.cu - Implementation with parallel disk I/O
// ========================================

#include "BatchACMMP.h"
#include "SphericalLUT_MultiRes.h"
#include "CompressedDMB.h"
#include <sys/stat.h>
#include <iostream>
#include <algorithm>
#include <chrono>
#include <fstream>
#include <sstream>
#include <iomanip>

// Forward declarations for GPU kernels defined in ACMMP.cu
__global__ void ExtractSupportPointsKernel(
    const float4* plane_hypotheses, const float* costs,
    int width, int height,
    int2* support_points_out, int* num_points_out, int max_points);

__global__ void RasterizeTrianglesKernel(
    const int2* triangle_vertices, const int* num_triangles_ptr,
    int max_triangles, unsigned int* plane_masks, int width, int height);

__global__ void ComputePriorPlanesKernel(
    const float4* plane_hypotheses, const Camera* cameras,
    const int2* triangle_vertices, unsigned int* plane_masks,
    float4* prior_planes, float depth_min, float depth_max,
    int width, int height);

__global__ void ExtractGridSupportPointsKernel(
    const float4* plane_hypotheses, const float* costs,
    int width, int height, int block_size,
    int2* support_points_out, int grid_w, int grid_h);

__global__ void GridTriangulationKernel(
    const int2* support_points, const int num_support_points,
    const float* costs, int width, int height, int block_size,
    int2* triangle_vertices, int* num_triangles, int max_triangles);

// ── ImageCache implementation ──────────────────────────────────────────────
ImageCache::ImageCache(size_t max_entries, const std::string& dense_folder, bool has_masks)
    : max_entries_(max_entries), dense_folder_(dense_folder), has_masks_(has_masks) {}

std::shared_ptr<const ImageCacheEntry> ImageCache::get(int image_id) {
    std::lock_guard<std::mutex> lk(mutex_);
    auto it = cache_.find(image_id);
    if (it != cache_.end()) {
        // Move to front of LRU
        lru_order_.remove(image_id);
        lru_order_.push_front(image_id);
        hits_++;
        return it->second;
    }
    misses_++;
    evict_if_needed();
    auto entry = load(image_id);
    cache_[image_id] = entry;
    lru_order_.push_front(image_id);
    return entry;
}

void ImageCache::evict_if_needed() {
    while (cache_.size() >= max_entries_ && !lru_order_.empty()) {
        int victim = lru_order_.back();
        lru_order_.pop_back();
        cache_.erase(victim);
    }
}

std::shared_ptr<ImageCacheEntry> ImageCache::load(int image_id) {
    auto entry = std::make_shared<ImageCacheEntry>();

    // Load image
    std::stringstream image_path;
    image_path << dense_folder_ << "/images/" << std::setw(8) << std::setfill('0') << image_id << ".png";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    image_uint.convertTo(entry->image_float, CV_32FC1);

    // Load camera
    std::stringstream cam_path;
    cam_path << dense_folder_ << "/cams/" << std::setw(8) << std::setfill('0') << image_id << "_cam.txt";
    entry->camera = ReadCamera(cam_path.str());
    entry->camera.height = entry->image_float.rows;
    entry->camera.width = entry->image_float.cols;

    // Load mask
    if (has_masks_) {
        std::stringstream mask_path;
        mask_path << dense_folder_ << "/masks/" << std::setw(8) << std::setfill('0') << image_id << ".png";
        cv::Mat mask_img = cv::imread(mask_path.str(), cv::IMREAD_GRAYSCALE);
        if (!mask_img.empty()) {
            mask_img.convertTo(entry->mask_float, CV_32FC1, 1.0 / 255.0);
            entry->has_mask = true;
        }
    }
    return entry;
}

// ── End ImageCache ────────────────────────────────────────────────────────

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
        texture_objects_host.images[i] = 0;
        texture_depths_host.images[i] = 0;
    }
}

void ProblemGPUResources::allocate(int max_width, int max_height, int max_images) {
    // This function is called once per resource object when the pool is initialized.
    allocated_images = max_images;

    // Allocate arrays for images (FP16 for 2x texture cache) and depths (FP32 for precision).
    for (int i = 0; i < max_images; ++i) {
        cudaChannelFormatDesc imageChannelDesc = cudaCreateChannelDesc(16, 0, 0, 0, cudaChannelFormatKindFloat);
        CUDA_CHECK(cudaMallocArray(&cuArray[i], &imageChannelDesc, max_width, max_height));
        cudaChannelFormatDesc depthChannelDesc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
        CUDA_CHECK(cudaMallocArray(&cuDepthArray[i], &depthChannelDesc, max_width, max_height));
    }

    // Allocate all other required device memory buffers.
    CUDA_CHECK(cudaMalloc(&cameras_cuda, sizeof(Camera) * max_images));
    CUDA_CHECK(cudaMalloc(&texture_objects_cuda, sizeof(cudaTextureObjects)));
    CUDA_CHECK(cudaMalloc(&texture_depths_cuda, sizeof(cudaTextureObjects)));
    CUDA_CHECK(cudaMalloc(&plane_hypotheses_cuda, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&scaled_plane_hypotheses_cuda, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&costs_cuda, sizeof(float) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&pre_costs_cuda, sizeof(float) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&rand_states_cuda, sizeof(RNGState) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&selected_views_cuda, sizeof(unsigned int) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&depths_cuda, sizeof(float) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&prior_planes_cuda, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMalloc(&plane_masks_cuda, sizeof(unsigned int) * max_width * max_height));

    // Item 4: GPU early masking buffer
    CUDA_CHECK(cudaMalloc(&ref_mask_cuda, sizeof(uint8_t) * max_width * max_height));

    // Item 5: Batch planar prior buffers
    const int max_support_points = ((max_width + 4) / 5) * ((max_height + 4) / 5);
    const int max_triangles = max_support_points * 2;
    CUDA_CHECK(cudaMalloc(&support_points_cuda, sizeof(int2) * max_support_points));
    CUDA_CHECK(cudaMalloc(&num_support_points_cuda, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&triangle_vertices_cuda, sizeof(int2) * max_triangles * 3));
    CUDA_CHECK(cudaMallocHost(&support_points_pinned, sizeof(int2) * max_support_points));
    CUDA_CHECK(cudaMallocHost(&num_support_points_pinned, sizeof(int)));

    // Allocate pinned host memory for high-speed asynchronous transfers.
    CUDA_CHECK(cudaMallocHost(&planes_host_pinned, sizeof(float4) * max_width * max_height));
    CUDA_CHECK(cudaMallocHost(&costs_host_pinned, sizeof(float) * max_width * max_height));

    // Item 2: Create texture objects once for all allocated slots
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(cudaTextureDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModeLinear;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = 0;

    for (int i = 0; i < max_images; i++) {
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType = cudaResourceTypeArray;

        resDesc.res.array.array = cuArray[i];
        CUDA_CHECK(cudaCreateTextureObject(&texture_objects_host.images[i], &resDesc, &texDesc, NULL));

        resDesc.res.array.array = cuDepthArray[i];
        CUDA_CHECK(cudaCreateTextureObject(&texture_depths_host.images[i], &resDesc, &texDesc, NULL));
    }

    // Upload texture handles to GPU (one-time)
    CUDA_CHECK(cudaMemcpy(texture_objects_cuda, &texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(texture_depths_cuda, &texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice));
    textures_created = true;
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

    // Distribute resources across all GPUs in round-robin fashion
    for (size_t i = 0; i < max_concurrent_problems; ++i) {
        int gpu_id = static_cast<int>(i) % num_gpus;
        CUDA_CHECK(cudaSetDevice(gpu_id));

        int prio_low=0, prio_high=0;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&prio_low, &prio_high));
        CUDA_CHECK(cudaStreamCreateWithPriority(&streams[i], cudaStreamNonBlocking, prio_high));

        std::unique_ptr<ProblemGPUResources> res(new ProblemGPUResources());
        res->stream_id = (int)i;
        res->stream = streams[i];
        res->device_id = gpu_id;

        // Allocate the GPU memory on this device.
        res->allocate(max_width, max_height, max_images);

        available_resources.push(res.get());
        resource_pool[i] = std::move(res);
    }

    // Restore to device 0
    CUDA_CHECK(cudaSetDevice(0));

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
    
    // IMPORTANT: Destroy texture objects BEFORE freeing their backing arrays
    for (int i = 0; i < MAX_IMAGES; ++i) {
        if (texture_objects_host.images[i] != 0) {
            cudaDestroyTextureObject(texture_objects_host.images[i]);
            texture_objects_host.images[i] = 0;
        }
        if (texture_depths_host.images[i] != 0) {
            cudaDestroyTextureObject(texture_depths_host.images[i]);
            texture_depths_host.images[i] = 0;
        }
    }
    
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
    safeFree((void*&)ref_mask_cuda, "ref_mask_cuda");
    safeFree((void*&)support_points_cuda, "support_points_cuda");
    safeFree((void*&)num_support_points_cuda, "num_support_points_cuda");
    safeFree((void*&)triangle_vertices_cuda, "triangle_vertices_cuda");

    if (support_points_pinned) {
        cudaFreeHost(support_points_pinned);
        support_points_pinned = nullptr;
    }
    if (num_support_points_pinned) {
        cudaFreeHost(num_support_points_pinned);
        num_support_points_pinned = nullptr;
    }
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
    // Multi-GPU detection
    CUDA_CHECK(cudaGetDeviceCount(&num_gpus));
    num_gpus = std::max(1, num_gpus);
    std::cout << "[BatchACMMP] Detected " << num_gpus << " GPU(s)" << std::endl;

    // Use device 0 for initial sizing
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    checkCudaLimits();
    available_gpu_memory = getAvailableGPUMemory();
    memory_per_problem = problems.empty() ? (size_t)500 * 1024 * 1024
                                          : estimateMemoryPerProblem(problems[0]);

    // Calculate per-GPU concurrency, then multiply by num_gpus
    size_t usable_gpu = size_t(double(available_gpu_memory) * 0.75);
    size_t by_gpu_mem = std::max<size_t>(1, usable_gpu / memory_per_problem);

    // Hardware-based limits: ~6 SMs per concurrent problem for good occupancy
    size_t hardware_threads = std::thread::hardware_concurrency();
    size_t by_sm = std::max<size_t>(2, prop.multiProcessorCount / 6);

    size_t per_gpu_concurrent = std::min({by_gpu_mem, by_sm, size_t(24)});
    per_gpu_concurrent = std::max<size_t>(1, per_gpu_concurrent);
    max_concurrent_problems = per_gpu_concurrent * num_gpus;

    // Separate disk writer threads - optimize for disk I/O
    num_disk_writers = std::min<size_t>(4, std::max<size_t>(2, hardware_threads / 4));

    std::cout << "[BatchACMMP] Configuration:" << std::endl;
    std::cout << "  GPUs: " << num_gpus << " (" << per_gpu_concurrent << " streams each)" << std::endl;
    std::cout << "  GPU Streams (total): " << max_concurrent_problems << std::endl;
    std::cout << "  Disk Writers: " << num_disk_writers << std::endl;
    std::cout << "  GPU Memory (device 0): " << (available_gpu_memory/(1024*1024)) << "MB free, "
              << (memory_per_problem/(1024*1024)) << "MB/problem" << std::endl;

    // Replicate LUTs to all GPU devices (device 0 already has them from main)
    if (num_gpus > 1) {
        ReplicateLUTsToAllDevices(num_gpus);
    }

    // Initialize image cache: detect mask folder
    std::string mask_dir = dense_folder + "/masks";
    struct stat mask_stat;
    bool masks_exist = (stat(mask_dir.c_str(), &mask_stat) == 0 && S_ISDIR(mask_stat.st_mode));
    image_cache_ = std::unique_ptr<ImageCache>(new ImageCache(50, dense_folder, masks_exist));
    std::cout << "[BatchACMMP] Image cache: max 50 entries, masks=" << (masks_exist ? "yes" : "no") << std::endl;

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
            cudaSetDevice(res->device_id);
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
    
    if (image_cache_) {
        std::cout << "[BatchACMMP] Image cache: " << image_cache_->hits() << " hits, "
                  << image_cache_->misses() << " misses" << std::endl;
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

    size_t textures = N * W * H * (sizeof(uint16_t) + sizeof(float)); // images (FP16) + depths (FP32)
    size_t working = W * H * (2*sizeof(float4) + 3*sizeof(float)); // hypotheses + costs
    size_t misc = W * H * (sizeof(RNGState) + sizeof(unsigned int));
    size_t masks = W * H * sizeof(uint8_t);                           // ref_mask_cuda
    size_t prior = W * H * (sizeof(float4) + sizeof(unsigned int));   // prior_planes + plane_masks
    size_t max_sp = ((W + 4) / 5) * ((H + 4) / 5);
    size_t support = max_sp * sizeof(int2) + sizeof(int);             // support_points + counter
    size_t triangles = max_sp * 2 * 3 * sizeof(int2);                // triangle_vertices
    size_t pinned = W * H * (sizeof(float4) + sizeof(float))         // planes + costs
                  + max_sp * sizeof(int2) + sizeof(int);              // support points pinned

    return (textures + working + misc + masks + prior + support + triangles + pinned) * 130 / 100;
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
    
    // Set the device for this resource's GPU (multi-GPU support).
    cudaSetDevice(resources->device_id);
    
    active_gpu_problems_.fetch_add(1);
    
    try {
        cudaError_t stream_check = cudaStreamQuery(stream);
        if (stream_check == cudaErrorInvalidResourceHandle) {
            std::cerr << "Invalid stream for problem " << problem_idx << ", creating new stream" << std::endl;
            cudaStreamCreate(&stream);
            resources->stream = stream;
        }
        
        {
            ACMMP acmmp;
            if (geom_consistency) acmmp.SetGeomConsistencyParams(multi_geometry);
            if (hierarchy) acmmp.SetHierarchyParams();
            acmmp.SetBatchMode();

            acmmp.SetStream(stream);
            acmmp.InputInitialization(dense_folder, all_problems, problem_idx, *image_cache_);
            acmmp.CudaSpaceInitialization(dense_folder, problem, resources);

            cudaError_t pre_run_check = cudaGetLastError();
            if (pre_run_check != cudaSuccess) {
                std::cerr << "Pre-run error: " << cudaGetErrorString(pre_run_check) << std::endl;
                throw std::runtime_error("CUDA setup failed");
            }

            const int width = acmmp.GetReferenceImageWidth();
            const int height = acmmp.GetReferenceImageHeight();

            if (planar_prior && !geom_consistency) {
                // ═══ PASS 1: initial depth estimates (2 iters suffice for rough estimates) ═══
                acmmp.SetMaxIterations(2);
                acmmp.RunPatchMatch(resources, /*skip_host_download=*/true);
                acmmp.SetMaxIterations(4);  // restore default

                // GPU-only planar prior pipeline (no CPU sync point)
                const int block_size = 5;
                const int grid_w = (width + block_size - 1) / block_size;
                const int grid_h = (height + block_size - 1) / block_size;
                const int num_grid_pts = grid_w * grid_h;
                const int max_tris = num_grid_pts * 2;

                // Step 1: Extract grid-ordered support points (GPU)
                {
                    dim3 eg_block(16, 16);
                    dim3 eg_grid((grid_w + eg_block.x - 1) / eg_block.x,
                                 (grid_h + eg_block.y - 1) / eg_block.y);
                    ExtractGridSupportPointsKernel<<<eg_grid, eg_block, 0, stream>>>(
                        resources->plane_hypotheses_cuda, resources->costs_cuda,
                        width, height, block_size,
                        resources->support_points_cuda, grid_w, grid_h);
                }

                // Step 2: Grid triangulation (GPU) — no D→H→D round-trip
                CUDA_CHECK(cudaMemsetAsync(resources->num_support_points_cuda, 0, sizeof(int), stream));
                {
                    dim3 gt_block(16, 16);
                    dim3 gt_grid((grid_w - 1 + gt_block.x - 1) / gt_block.x,
                                 (grid_h - 1 + gt_block.y - 1) / gt_block.y);
                    GridTriangulationKernel<<<gt_grid, gt_block, 0, stream>>>(
                        resources->support_points_cuda, num_grid_pts,
                        resources->costs_cuda, width, height, block_size,
                        resources->triangle_vertices_cuda, resources->num_support_points_cuda, max_tris);
                }

                // Steps 3-5: Rasterize + compute priors without host sync
                // Launch with max_tris upper bound; kernel reads actual count from device ptr
                CUDA_CHECK(cudaMemsetAsync(resources->plane_masks_cuda, 0,
                                           sizeof(unsigned int) * width * height, stream));
                RasterizeTrianglesKernel<<<(max_tris + 255) / 256, 256, 0, stream>>>(
                    resources->triangle_vertices_cuda, resources->num_support_points_cuda,
                    max_tris, resources->plane_masks_cuda, width, height);

                // Compute prior planes from triangulated depths (GPU)
                float dmin = acmmp.GetMinDepth();
                float dmax = acmmp.GetMaxDepth();
                {
                    dim3 pp_block(16, 16);
                    dim3 pp_grid((width + pp_block.x - 1) / pp_block.x,
                                 (height + pp_block.y - 1) / pp_block.y);
                    ComputePriorPlanesKernel<<<pp_grid, pp_block, 0, stream>>>(
                        resources->plane_hypotheses_cuda, resources->cameras_cuda,
                        resources->triangle_vertices_cuda, resources->plane_masks_cuda,
                        resources->prior_planes_cuda,
                        dmin, dmax, width, height);
                }

                // ═══ PASS 2: prior-assisted PatchMatch (3 iters with prior guidance) ═══
                acmmp.SetPlanarPriorParams();
                acmmp.SetMaxIterations(3);
                acmmp.RunPatchMatch(resources, /*skip_host_download=*/false);
            } else {
                acmmp.RunPatchMatch(resources);
            }

            cv::Mat_<float> depths(height, width);
            cv::Mat_<cv::Vec3f> normals(height, width);
            cv::Mat_<float> costs(height, width);

            // Item 3: Read directly from pinned buffers (no intermediate copy)
            const float4* planes_pinned = resources->planes_host_pinned;
            const float* costs_pinned = resources->costs_host_pinned;

            for (int y = 0; y < height; ++y) {
                for (int x = 0; x < width; ++x) {
                    const int c = y * width + x;
                    const float4 ph = planes_pinned[c];
                    depths(y, x) = ph.w;
                    normals(y, x) = cv::Vec3f(ph.x, ph.y, ph.z);
                    costs(y, x) = costs_pinned[c];
                }
            }


            // Queue for disk writing (compressed format will be used)
            {
                std::unique_lock<std::mutex> lk(disk_queue_mutex_);
                disk_queue_space_cv_.wait(lk, [&]{
                    return disk_write_queue_.size() < mask_disk_queue_size || stopping_disk_.load();
                });
                
                disk_write_queue_.emplace(problem_idx, problem, 
                                         std::move(depths), std::move(normals), 
                                         std::move(costs), geom_consistency);
            }
            disk_queue_cv_.notify_one();
        }
        
    } catch (const std::exception& e) {
        std::cerr << "[Problem " << problem_idx << "] Exception: " << e.what() << std::endl;
        cudaGetLastError();
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
    
    // Use compressed format (.cdmb) instead of raw (.dmb)
    // This reduces file sizes by ~75-90%
    std::string depth_suffix = result.geom_consistency ? "/depths_geom.cdmb" : "/depths.cdmb";
    std::string depth_path = result_folder + depth_suffix;
    std::string normal_path = result_folder + "/normals.cdmb";
    std::string cost_path = result_folder + "/costs.cdmb";
    
    // Write compressed files
    CompressedDMB::writeDepthCompressed(depth_path, result.depths);
    CompressedDMB::writeNormalCompressed(normal_path, result.normals);
    CompressedDMB::writeCostCompressed(cost_path, result.costs);
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