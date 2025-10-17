// FULLY C++11 COMPATIBLE VERSION - All modern C++ features removed
// FIXED: Mutex and atomic copy issues resolved

#include "ACMMP.h"
#include "ACMMP_device.cuh"
#include "FusionGPU.h"

// CUDA headers must come first
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_constants.h>

// C++ headers (C++11 compatible only)
#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <condition_variable>
#include <list>

// Enhanced CUDA error checking macro
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA operation failed"); \
    } \
} while(0)
#endif

// Safe arithmetic operations
template<typename T>
bool safe_multiply(T a, T b, T& result) {
    if (a == 0 || b == 0) {
        result = 0;
        return true;
    }
    if (a > std::numeric_limits<T>::max() / b) {
        return false;
    }
    result = a * b;
    return true;
}

// CUDA device function for efficient binary search
__device__ int find_problem_id(int global_idx, int* problem_offsets, int num_problems) {
    int low = 0, high = num_problems;
    while (low < high) {
        int mid = low + (high - low) / 2;
        if (global_idx >= problem_offsets[mid]) {
            low = mid + 1;
        } else {
            high = mid;
        }
    }
    return low - 1;
}

// Lookup table structure for fast image ID to index mapping
struct ImageLookupTables {
    std::unordered_map<int, int> image_to_camera_idx;
    std::unordered_map<int, int> image_to_texture_idx;
    
    // GPU arrays for kernel access
    int* d_image_to_camera_map;
    int* d_image_to_texture_map;
    int* d_image_ids;
    int num_images;
    int max_image_id;
    
    ImageLookupTables() : d_image_to_camera_map(nullptr), d_image_to_texture_map(nullptr), 
                         d_image_ids(nullptr), num_images(0), max_image_id(0) {}
    
    ~ImageLookupTables() {
        cleanup();
    }
    
    void cleanup() {
        if (d_image_to_camera_map) {
            cudaFree(d_image_to_camera_map);
            d_image_to_camera_map = nullptr;
        }
        if (d_image_to_texture_map) {
            cudaFree(d_image_to_texture_map);
            d_image_to_texture_map = nullptr;
        }
        if (d_image_ids) {
            cudaFree(d_image_ids);
            d_image_ids = nullptr;
        }
    }
    
    void buildTables(const std::vector<int>& camera_image_ids, 
                    const std::vector<int>& texture_image_ids) {
        cleanup();
        
        // Build camera mapping
        for (size_t i = 0; i < camera_image_ids.size(); ++i) {
            image_to_camera_idx[camera_image_ids[i]] = static_cast<int>(i);
            max_image_id = std::max(max_image_id, camera_image_ids[i]);
        }
        
        // Build texture mapping
        for (size_t i = 0; i < texture_image_ids.size(); ++i) {
            image_to_texture_idx[texture_image_ids[i]] = static_cast<int>(i);
            max_image_id = std::max(max_image_id, texture_image_ids[i]);
        }
        
        // Create GPU lookup arrays
        std::vector<int> camera_map(max_image_id + 1, -1);
        std::vector<int> texture_map(max_image_id + 1, -1);
        
        for (const auto& pair : image_to_camera_idx) {
            camera_map[pair.first] = pair.second;
        }
        for (const auto& pair : image_to_texture_idx) {
            texture_map[pair.first] = pair.second;
        }
        
        // Copy to GPU
        size_t map_size = (max_image_id + 1) * sizeof(int);
        CUDA_SAFE_CALL(cudaMalloc(&d_image_to_camera_map, map_size));
        CUDA_SAFE_CALL(cudaMalloc(&d_image_to_texture_map, map_size));
        
        CUDA_SAFE_CALL(cudaMemcpy(d_image_to_camera_map, camera_map.data(), 
                                  map_size, cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(d_image_to_texture_map, texture_map.data(), 
                                  map_size, cudaMemcpyHostToDevice));
        
        std::cout << "[LookupTables] Built lookup tables for " << camera_image_ids.size() 
                  << " cameras, " << texture_image_ids.size() << " textures, max_id=" 
                  << max_image_id << std::endl;
    }
};

// FIXED: C++11 compatible thread pool with proper mutex handling
class EfficientThreadPool {
private:
    std::vector<std::thread> workers;
    std::vector<std::queue<std::function<void()>>> task_queues;
    // FIX: Use vector of unique_ptr to avoid mutex copy issues
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes;
    
    // Shared condition variable for efficient sleeping
    std::mutex pool_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<size_t> next_queue;

public:
    EfficientThreadPool(size_t threads = std::thread::hardware_concurrency()) 
        : stop(false), next_queue(0) {
        if (threads == 0) threads = 4;
        
        task_queues.resize(threads);
        // FIX: Initialize unique_ptr mutexes
        queue_mutexes.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            queue_mutexes.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
        }
        
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, threads] {
                while (!stop.load()) {
                    std::function<void()> task;
                    bool found_task = false;
                    
                    // Try to get task from own queue first
                    {
                        std::unique_lock<std::mutex> lock(*queue_mutexes[i]);
                        if (!task_queues[i].empty()) {
                            task = std::move(task_queues[i].front());
                            task_queues[i].pop();
                            found_task = true;
                        }
                    }
                    
                    // Work stealing from other queues
                    if (!found_task) {
                        for (size_t j = 0; j < threads && !found_task; ++j) {
                            size_t queue_idx = (i + j) % threads;
                            std::unique_lock<std::mutex> lock(*queue_mutexes[queue_idx]);
                            if (!task_queues[queue_idx].empty()) {
                                task = std::move(task_queues[queue_idx].front());
                                task_queues[queue_idx].pop();
                                found_task = true;
                            }
                        }
                    }
                    
                    if (found_task) {
                        task();
                    } else {
                        // Efficient waiting instead of busy-waiting
                        std::unique_lock<std::mutex> lock(pool_mutex);
                        condition.wait(lock, [this] { return stop.load() || hasAnyTasks(); });
                    }
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        
        size_t queue_idx = next_queue.fetch_add(1) % task_queues.size();
        
        {
            std::unique_lock<std::mutex> lock(*queue_mutexes[queue_idx]);
            if (stop.load()) {
                throw std::runtime_error("enqueue on stopped ThreadPool");
            }
            task_queues[queue_idx].emplace([task]() { (*task)(); });
        }
        
        // Notify waiting threads
        condition.notify_one();
        
        return res;
    }

    ~EfficientThreadPool() {
        stop.store(true);
        condition.notify_all();
        for (std::thread &worker : workers) {
            if (worker.joinable()) {
                worker.join();
            }
        }
    }

private:
    bool hasAnyTasks() {
        for (size_t i = 0; i < task_queues.size(); ++i) {
            std::unique_lock<std::mutex> lock(*queue_mutexes[i]);
            if (!task_queues[i].empty()) {
                return true;
            }
        }
        return false;
    }
};

// Thread-safe persistent GPU buffer manager
class PersistentGPUBuffers {
private:
    // Texture arrays for batch processing
    cudaTextureObject_t* depth_textures_buffer;
    cudaTextureObject_t* normal_textures_buffer;
    cudaTextureObject_t* image_textures_buffer;
    int* texture_image_ids_buffer;
    
    // Problem data buffers
    int* ref_image_ids_buffer;
    int* all_src_image_ids_buffer;
    int* src_counts_buffer;
    int* src_offsets_buffer;
    int* problem_offsets_buffer;
    int* widths_buffer;
    int* heights_buffer;
    
    // Output buffers
    PointList* output_points_buffer;
    int* valid_flags_buffer;
    
    // Buffer sizes
    size_t max_textures;
    size_t max_problems;
    size_t max_src_images;
    size_t max_pixels;
    
    std::atomic<bool> buffers_allocated;
    mutable std::mutex access_mutex;

    void safeCleanup() {
        std::unique_lock<std::mutex> lock(access_mutex);
        if (!buffers_allocated.load()) return;
        
        // Synchronize all CUDA operations before cleanup
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        if (depth_textures_buffer) { cudaFree(depth_textures_buffer); depth_textures_buffer = nullptr; }
        if (normal_textures_buffer) { cudaFree(normal_textures_buffer); normal_textures_buffer = nullptr; }
        if (image_textures_buffer) { cudaFree(image_textures_buffer); image_textures_buffer = nullptr; }
        if (texture_image_ids_buffer) { cudaFree(texture_image_ids_buffer); texture_image_ids_buffer = nullptr; }
        if (ref_image_ids_buffer) { cudaFree(ref_image_ids_buffer); ref_image_ids_buffer = nullptr; }
        if (all_src_image_ids_buffer) { cudaFree(all_src_image_ids_buffer); all_src_image_ids_buffer = nullptr; }
        if (src_counts_buffer) { cudaFree(src_counts_buffer); src_counts_buffer = nullptr; }
        if (src_offsets_buffer) { cudaFree(src_offsets_buffer); src_offsets_buffer = nullptr; }
        if (problem_offsets_buffer) { cudaFree(problem_offsets_buffer); problem_offsets_buffer = nullptr; }
        if (widths_buffer) { cudaFree(widths_buffer); widths_buffer = nullptr; }
        if (heights_buffer) { cudaFree(heights_buffer); heights_buffer = nullptr; }
        if (output_points_buffer) { cudaFree(output_points_buffer); output_points_buffer = nullptr; }
        if (valid_flags_buffer) { cudaFree(valid_flags_buffer); valid_flags_buffer = nullptr; }
        
        buffers_allocated.store(false);
    }

public:
    PersistentGPUBuffers() : buffers_allocated(false) {
        // Initialize all pointers to nullptr
        depth_textures_buffer = nullptr;
        normal_textures_buffer = nullptr;
        image_textures_buffer = nullptr;
        texture_image_ids_buffer = nullptr;
        ref_image_ids_buffer = nullptr;
        all_src_image_ids_buffer = nullptr;
        src_counts_buffer = nullptr;
        src_offsets_buffer = nullptr;
        problem_offsets_buffer = nullptr;
        widths_buffer = nullptr;
        heights_buffer = nullptr;
        output_points_buffer = nullptr;
        valid_flags_buffer = nullptr;
    }
    
    void allocateBuffers(size_t est_max_textures, size_t est_max_problems, 
                        size_t est_max_src_images, size_t est_max_pixels) {
        std::unique_lock<std::mutex> lock(access_mutex);
        
        if (buffers_allocated.load()) return;
        
        // Safe arithmetic with overflow checking
        size_t safe_max_textures, safe_max_problems, safe_max_src_images, safe_max_pixels;
        
        if (!safe_multiply(est_max_textures, static_cast<size_t>(2), safe_max_textures) ||
            !safe_multiply(est_max_problems, static_cast<size_t>(2), safe_max_problems) ||
            !safe_multiply(est_max_src_images, static_cast<size_t>(2), safe_max_src_images) ||
            !safe_multiply(est_max_pixels, static_cast<size_t>(2), safe_max_pixels)) {
            throw std::overflow_error("Buffer size calculation overflow");
        }
        
        max_textures = safe_max_textures;
        max_problems = safe_max_problems;
        max_src_images = safe_max_src_images;
        max_pixels = safe_max_pixels;
        
        // Check available GPU memory
        size_t free_mem, total_mem;
        CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
        
        size_t total_required = 
            max_textures * (3 * sizeof(cudaTextureObject_t) + sizeof(int)) +
            max_problems * 6 * sizeof(int) +
            max_src_images * sizeof(int) +
            max_pixels * (sizeof(PointList) + sizeof(int));
        
        if (total_required > free_mem * 0.8) {
            throw std::runtime_error("Insufficient GPU memory for requested buffer sizes");
        }
        
        std::cout << "[PersistentGPU] Allocating buffers for max: " 
                  << max_textures << " textures, " << max_problems << " problems, "
                  << max_pixels << " pixels (" << (total_required / (1024*1024)) << " MB)" << std::endl;
        
        try {
            // Allocate all buffers with error checking
            CUDA_SAFE_CALL(cudaMalloc(&depth_textures_buffer, max_textures * sizeof(cudaTextureObject_t)));
            CUDA_SAFE_CALL(cudaMalloc(&normal_textures_buffer, max_textures * sizeof(cudaTextureObject_t)));
            CUDA_SAFE_CALL(cudaMalloc(&image_textures_buffer, max_textures * sizeof(cudaTextureObject_t)));
            CUDA_SAFE_CALL(cudaMalloc(&texture_image_ids_buffer, max_textures * sizeof(int)));
            
            CUDA_SAFE_CALL(cudaMalloc(&ref_image_ids_buffer, max_problems * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&all_src_image_ids_buffer, max_src_images * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&src_counts_buffer, max_problems * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&src_offsets_buffer, max_problems * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&problem_offsets_buffer, max_problems * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&widths_buffer, max_problems * sizeof(int)));
            CUDA_SAFE_CALL(cudaMalloc(&heights_buffer, max_problems * sizeof(int)));
            
            CUDA_SAFE_CALL(cudaMalloc(&output_points_buffer, max_pixels * sizeof(PointList)));
            CUDA_SAFE_CALL(cudaMalloc(&valid_flags_buffer, max_pixels * sizeof(int)));
            
            buffers_allocated.store(true);
            
            CUDA_SAFE_CALL(cudaMemGetInfo(&free_mem, &total_mem));
            std::cout << "[PersistentGPU] Buffers allocated successfully. GPU memory: " 
                      << free_mem / (1024*1024) << " MB free / " 
                      << total_mem / (1024*1024) << " MB total" << std::endl;
        } catch (...) {
            safeCleanup();
            throw;
        }
    }
    
    ~PersistentGPUBuffers() {
        safeCleanup();
    }
    
    // Thread-safe buffer getters with bounds checking
    template<typename T>
    T* getBufferSafe(T* buffer, size_t needed, size_t max_size, const char* name) {
        std::unique_lock<std::mutex> lock(access_mutex);
        if (!buffers_allocated.load()) {
            throw std::runtime_error("Buffers not allocated");
        }
        if (needed > max_size) {
            throw std::runtime_error(std::string(name) + " buffer size exceeded: needed " + 
                                   std::to_string(needed) + ", max " + std::to_string(max_size));
        }
        return buffer;
    }
    
    cudaTextureObject_t* getDepthTexturesBuffer(size_t needed) {
        return getBufferSafe(depth_textures_buffer, needed, max_textures, "Depth textures");
    }
    
    cudaTextureObject_t* getNormalTexturesBuffer(size_t needed) {
        return getBufferSafe(normal_textures_buffer, needed, max_textures, "Normal textures");
    }
    
    cudaTextureObject_t* getImageTexturesBuffer(size_t needed) {
        return getBufferSafe(image_textures_buffer, needed, max_textures, "Image textures");
    }
    
    int* getTextureImageIdsBuffer(size_t needed) {
        return getBufferSafe(texture_image_ids_buffer, needed, max_textures, "Texture image IDs");
    }
    
    int* getRefImageIdsBuffer(size_t needed) {
        return getBufferSafe(ref_image_ids_buffer, needed, max_problems, "Reference image IDs");
    }
    
    int* getAllSrcImageIdsBuffer(size_t needed) {
        return getBufferSafe(all_src_image_ids_buffer, needed, max_src_images, "Source image IDs");
    }
    
    int* getSrcCountsBuffer(size_t needed) {
        return getBufferSafe(src_counts_buffer, needed, max_problems, "Source counts");
    }
    
    int* getSrcOffsetsBuffer(size_t needed) {
        return getBufferSafe(src_offsets_buffer, needed, max_problems, "Source offsets");
    }
    
    int* getProblemOffsetsBuffer(size_t needed) {
        return getBufferSafe(problem_offsets_buffer, needed, max_problems, "Problem offsets");
    }
    
    int* getWidthsBuffer(size_t needed) {
        return getBufferSafe(widths_buffer, needed, max_problems, "Widths");
    }
    
    int* getHeightsBuffer(size_t needed) {
        return getBufferSafe(heights_buffer, needed, max_problems, "Heights");
    }
    
    PointList* getOutputPointsBuffer(size_t needed) {
        return getBufferSafe(output_points_buffer, needed, max_pixels, "Output points");
    }
    
    int* getValidFlagsBuffer(size_t needed) {
        return getBufferSafe(valid_flags_buffer, needed, max_pixels, "Valid flags");
    }
};

// C++11 compatible image data structure
struct ImageData {
    Camera camera;
    cv::Mat_<float> depth;
    cv::Mat_<cv::Vec3f> normal;
    cv::Mat image;
    std::atomic<bool> valid;
    std::chrono::steady_clock::time_point last_accessed;
    
    ImageData() : valid(false), last_accessed(std::chrono::steady_clock::now()) {}
};

// C++11 compatible optimized data loader
class OptimizedDataLoader {
private:
    std::string dense_folder;
    std::string img_folder; 
    std::string cam_folder;
    bool geom_consistency;
    
    // Thread-safe LRU cache
    std::unordered_map<int, std::shared_ptr<ImageData>> cache;
    std::list<int> lru_order;
    std::unordered_map<int, std::list<int>::iterator> lru_map;
    
    size_t max_cache_size;
    mutable std::mutex cache_mutex;
    
    // Thread pool for parallel I/O
    std::unique_ptr<EfficientThreadPool> thread_pool;
    
    // Thread-safe LRU update
    void updateLRU(int image_id) {
        // Must be called with lock already held
        auto lru_it = lru_map.find(image_id);
        if (lru_it != lru_map.end()) {
            auto list_it = lru_it->second;
            lru_order.erase(list_it);
            lru_map.erase(lru_it);
        }
        
        lru_order.push_front(image_id);
        lru_map[image_id] = lru_order.begin();
    }
    
    void trimCache() {
        // Must be called with lock already held
        while (cache.size() > max_cache_size && !lru_order.empty()) {
            int oldest = lru_order.back();
            lru_order.pop_back();
            
            auto lru_it = lru_map.find(oldest);
            if (lru_it != lru_map.end()) {
                lru_map.erase(lru_it);
            }
            
            auto cache_it = cache.find(oldest);
            if (cache_it != cache.end()) {
                cache.erase(cache_it);
            }
        }
    }
    
    std::shared_ptr<ImageData> loadImageDataSync(int image_id) {
        auto data = std::make_shared<ImageData>();
        data->valid.store(false);
        
        try {
            // Load camera
            char buf[512];
            int ret = snprintf(buf, sizeof(buf), "%s/%08d_cam.txt", cam_folder.c_str(), image_id);
            if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
                return data;
            }
            
            data->camera = ReadCamera(std::string(buf));
            if (data->camera.width <= 0 || data->camera.height <= 0) {
                return data;
            }
            
            // Load depth
            std::string depth_suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
            ret = snprintf(buf, sizeof(buf), "%s/ACMMP/2333_%08d%s", 
                          dense_folder.c_str(), image_id, depth_suffix.c_str());
            if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
                return data;
            }
            
            if (readDepthDmb(std::string(buf), data->depth) != 0 || 
                data->depth.cols <= 0 || data->depth.rows <= 0) {
                return data;
            }
            
            // Load normal
            ret = snprintf(buf, sizeof(buf), "%s/ACMMP/2333_%08d/normals.dmb", 
                          dense_folder.c_str(), image_id);
            if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
                return data;
            }
            
            if (readNormalDmb(std::string(buf), data->normal) != 0 || 
                data->normal.cols <= 0 || data->normal.rows <= 0) {
                return data;
            }
            
            // Load image
            ret = snprintf(buf, sizeof(buf), "%s/%08d.jpg", img_folder.c_str(), image_id);
            if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
                return data;
            }
            
            data->image = cv::imread(std::string(buf), cv::IMREAD_COLOR);
            if (data->image.empty()) {
                return data;
            }
            
            // Rescale image to match depth
            cv::Mat_<cv::Vec3b> img_color;
            if (data->image.channels() == 3) {
                img_color = cv::Mat_<cv::Vec3b>(data->image);
            } else {
                cv::cvtColor(data->image, img_color, cv::COLOR_GRAY2BGR);
            }
            cv::Mat_<cv::Vec3b> scaled_color;
            RescaleImageAndCamera(img_color, scaled_color, data->depth, data->camera);
            data->image = cv::Mat(scaled_color);
            
            data->valid.store(true);
            data->last_accessed = std::chrono::steady_clock::now();
            
        } catch (const std::exception& e) {
            std::cerr << "Error loading image " << image_id << ": " << e.what() << std::endl;
            return data;
        } catch (...) {
            std::cerr << "Unknown error loading image " << image_id << std::endl;
            return data;
        }
        
        return data;
    }
    
public:
    OptimizedDataLoader(const std::string& folder, bool geom = false, size_t cache_size = 100) 
        : dense_folder(folder), geom_consistency(geom), max_cache_size(cache_size) {
        img_folder = folder + "/images";
        cam_folder = folder + "/cams";
        
        unsigned int hw_threads = std::thread::hardware_concurrency();
        // C++11 compatible min/max instead of std::clamp
        size_t io_threads = std::min(static_cast<size_t>(16), std::max(static_cast<size_t>(4), static_cast<size_t>(hw_threads)));
        
        // C++11 compatible unique_ptr construction
        thread_pool.reset(new EfficientThreadPool(io_threads));
        
        std::cout << "[OptimizedLoader] Using " << io_threads << " threads for parallel I/O" << std::endl;
    }
    
    void preloadChunkParallel(const std::vector<int>& image_ids) {
        if (image_ids.empty()) return;
        
        std::vector<std::future<std::shared_ptr<ImageData>>> futures;
        std::vector<int> images_to_load;
        
        std::cout << "  Preloading " << image_ids.size() << " images in parallel..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Check cache and prepare loading list
        {
            std::unique_lock<std::mutex> lock(cache_mutex);
            for (int image_id : image_ids) {
                if (cache.find(image_id) == cache.end()) {
                    images_to_load.push_back(image_id);
                }
            }
        }
        
        // Launch parallel loading tasks
        for (int image_id : images_to_load) {
            auto future = thread_pool->enqueue([this, image_id]() {
                return loadImageDataSync(image_id);
            });
            futures.push_back(std::move(future));
        }
        
        // Collect results and store in cache
        size_t loaded_count = 0;
        {
            std::unique_lock<std::mutex> lock(cache_mutex);
            for (size_t i = 0; i < futures.size(); ++i) {
                int image_id = images_to_load[i];
                auto data = futures[i].get();
                
                if (data && data->valid.load()) {
                    cache[image_id] = data;
                    updateLRU(image_id);
                    loaded_count++;
                }
            }
            trimCache();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "  Loaded " << loaded_count << "/" << images_to_load.size() 
                  << " new images in " << duration.count() << " ms" << std::endl;
    }
    
    // C++11 compatible lock strategy using standard mutex
    bool getData(int image_id, Camera& cam, cv::Mat_<float>& depth, 
                cv::Mat_<cv::Vec3f>& normal, cv::Mat& image) {
        std::unique_lock<std::mutex> lock(cache_mutex);
        
        auto it = cache.find(image_id);
        if (it == cache.end() || !it->second->valid.load()) {
            return false;
        }
        
        auto data = it->second;
        cam = data->camera;
        depth = data->depth;
        normal = data->normal;
        image = data->image;
        
        updateLRU(image_id);
        data->last_accessed = std::chrono::steady_clock::now();
        
        return true;
    }
    
    size_t getCacheSize() const {
        std::unique_lock<std::mutex> lock(cache_mutex);
        return cache.size();
    }
    
    void clearCache() {
        std::unique_lock<std::mutex> lock(cache_mutex);
        cache.clear();
        lru_order.clear();
        lru_map.clear();
    }
};

// FIXED: Optimized texture manager with proper synchronization
class OptimizedTextureManager {
private:
    struct TextureData {
        cudaArray* depth_array;
        cudaArray* normal_array;
        cudaArray* image_array;
        cudaTextureObject_t depth_texture;
        cudaTextureObject_t normal_texture;
        cudaTextureObject_t image_texture;
        bool is_valid;  // FIX: Changed from atomic<bool> to regular bool
        
        TextureData() : depth_array(nullptr), normal_array(nullptr), image_array(nullptr),
                       depth_texture(0), normal_texture(0), image_texture(0), is_valid(false) {}
        
        void cleanup() {
            // Synchronize before cleanup
            cudaDeviceSynchronize();
            
            if (depth_texture != 0) {
                cudaDestroyTextureObject(depth_texture);
                depth_texture = 0;
            }
            if (normal_texture != 0) {
                cudaDestroyTextureObject(normal_texture);
                normal_texture = 0;
            }
            if (image_texture != 0) {
                cudaDestroyTextureObject(image_texture);
                image_texture = 0;
            }
            
            if (depth_array != nullptr) {
                cudaFreeArray(depth_array);
                depth_array = nullptr;
            }
            if (normal_array != nullptr) {
                cudaFreeArray(normal_array);
                normal_array = nullptr;
            }
            if (image_array != nullptr) {
                cudaFreeArray(image_array);
                image_array = nullptr;
            }
            
            is_valid = false;
        }
    };
    
    // FIX: Use vector of unique_ptr to avoid copy issues
    std::vector<std::unique_ptr<TextureData>> textures;
    std::vector<int> current_image_ids;
    std::atomic<bool> loaded;
    mutable std::mutex texture_mutex;
    
    // Use multiple streams for parallel texture creation
    static const int num_streams = 4;
    cudaStream_t streams[num_streams];
    
    void release() {
        std::lock_guard<std::mutex> lock(texture_mutex);
        
        if (!loaded.load() && textures.empty()) {
            return;
        }
        
        // Synchronize all streams before cleanup
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();
        
        // Clean up all textures
        for (auto& tex : textures) {
            if (tex) {
                tex->cleanup();
            }
        }
        
        textures.clear();
        current_image_ids.clear();
        loaded.store(false);
    }

public:
    OptimizedTextureManager() : loaded(false) {
        for (int i = 0; i < num_streams; ++i) {
            CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
        }
    }
    
    ~OptimizedTextureManager() {
        release();
        for (int i = 0; i < num_streams; ++i) {
            if (streams[i] != 0) {
                cudaStreamDestroy(streams[i]);
            }
        }
    }
    
    bool loadChunk(const std::vector<int>& image_ids, OptimizedDataLoader& loader) {
        release();  // Clean up previous chunk
        
        if (image_ids.empty()) {
            return false;
        }
        
        std::lock_guard<std::mutex> lock(texture_mutex);
        
        current_image_ids = image_ids;
        textures.clear();
        textures.reserve(image_ids.size());
        
        // FIX: Initialize unique_ptrs
        for (size_t i = 0; i < image_ids.size(); ++i) {
            textures.push_back(std::unique_ptr<TextureData>(new TextureData()));
        }
        
        std::atomic<size_t> successful_textures(0);
        std::vector<std::future<bool>> texture_futures;
        
        // Process textures in parallel
        for (size_t i = 0; i < image_ids.size(); ++i) {
            auto future = std::async(std::launch::async, [this, i, &image_ids, &loader, &successful_textures]() {
                try {
                    int image_id = image_ids[i];
                    cudaStream_t stream = streams[i % num_streams];
                    
                    Camera cam;
                    cv::Mat_<float> depth;
                    cv::Mat_<cv::Vec3f> normal;
                    cv::Mat image;
                    
                    if (!loader.getData(image_id, cam, depth, normal, image)) {
                        return false;
                    }
                    
                    TextureData& tex = *textures[i];
                    
                    // Create depth texture
                    cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc<float>();
                    CUDA_SAFE_CALL(cudaMallocArray(&tex.depth_array, &depth_desc, cam.width, cam.height));
                    CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(tex.depth_array, 0, 0, 
                                                           depth.ptr<float>(), depth.step[0], 
                                                           cam.width * sizeof(float), cam.height, 
                                                           cudaMemcpyHostToDevice, stream));
                    
                    cudaResourceDesc depth_res_desc = {};
                    depth_res_desc.resType = cudaResourceTypeArray;
                    depth_res_desc.res.array.array = tex.depth_array;
                    
                    cudaTextureDesc depth_tex_desc = {};
                    depth_tex_desc.addressMode[0] = cudaAddressModeWrap;
                    depth_tex_desc.addressMode[1] = cudaAddressModeClamp;
                    depth_tex_desc.filterMode = cudaFilterModePoint;
                    depth_tex_desc.readMode = cudaReadModeElementType;
                    depth_tex_desc.normalizedCoords = false;
                    
                    CUDA_SAFE_CALL(cudaCreateTextureObject(&tex.depth_texture, &depth_res_desc, &depth_tex_desc, NULL));
                    
                    // Create normal texture
                    cv::Mat normal_rgba = cv::Mat(normal.rows, normal.cols, CV_32FC4, cv::Scalar(0.0f, 0.0f, 0.0f, 0.0f));
                    cv::Mat src_mats[1] = {normal};
                    cv::Mat dst_mats[1] = {normal_rgba};
                    int from_to[6] = {0, 0, 1, 1, 2, 2};
                    cv::mixChannels(src_mats, 1, dst_mats, 1, from_to, 3);
                    
                    cudaChannelFormatDesc normal_desc = cudaCreateChannelDesc<float4>();
                    CUDA_SAFE_CALL(cudaMallocArray(&tex.normal_array, &normal_desc, cam.width, cam.height));
                    CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(tex.normal_array, 0, 0, 
                                                           normal_rgba.ptr<float>(), normal_rgba.step[0], 
                                                           cam.width * sizeof(float4), cam.height, 
                                                           cudaMemcpyHostToDevice, stream));
                    
                    cudaResourceDesc normal_res_desc = {};
                    normal_res_desc.resType = cudaResourceTypeArray;
                    normal_res_desc.res.array.array = tex.normal_array;
                    
                    cudaTextureDesc normal_tex_desc = {};
                    normal_tex_desc.addressMode[0] = cudaAddressModeWrap;
                    normal_tex_desc.addressMode[1] = cudaAddressModeClamp;
                    normal_tex_desc.filterMode = cudaFilterModePoint;
                    normal_tex_desc.readMode = cudaReadModeElementType;
                    normal_tex_desc.normalizedCoords = false;
                    
                    CUDA_SAFE_CALL(cudaCreateTextureObject(&tex.normal_texture, &normal_res_desc, &normal_tex_desc, NULL));
                    
                    // Create image texture
                    cv::Mat rgba, rgba_float;
                    cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
                    rgba.convertTo(rgba_float, CV_32FC4, 1.0/255.0);
                    
                    cudaChannelFormatDesc image_desc = cudaCreateChannelDesc<float4>();
                    CUDA_SAFE_CALL(cudaMallocArray(&tex.image_array, &image_desc, cam.width, cam.height));
                    CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(tex.image_array, 0, 0, 
                                                           rgba_float.ptr<float>(), rgba_float.step[0], 
                                                           cam.width * sizeof(float4), cam.height, 
                                                           cudaMemcpyHostToDevice, stream));
                    
                    cudaResourceDesc image_res_desc = {};
                    image_res_desc.resType = cudaResourceTypeArray;
                    image_res_desc.res.array.array = tex.image_array;
                    
                    cudaTextureDesc image_tex_desc = {};
                    image_tex_desc.addressMode[0] = cudaAddressModeWrap;
                    image_tex_desc.addressMode[1] = cudaAddressModeClamp;
                    image_tex_desc.filterMode = cudaFilterModeLinear;
                    image_tex_desc.readMode = cudaReadModeElementType;
                    image_tex_desc.normalizedCoords = false;
                    
                    CUDA_SAFE_CALL(cudaCreateTextureObject(&tex.image_texture, &image_res_desc, &image_tex_desc, NULL));
                    
                    // Synchronize stream before marking as valid
                    CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
                    
                    tex.is_valid = true;  // FIX: Direct assignment instead of atomic store
                    successful_textures.fetch_add(1);
                    
                    return true;
                } catch (const std::exception& e) {
                    std::cerr << "    Failed to create texture for image " << image_ids[i] << ": " << e.what() << std::endl;
                    textures[i]->cleanup();
                    return false;
                }
            });
            
            texture_futures.push_back(std::move(future));
        }
        
        // Wait for all texture creation to complete
        for (auto& future : texture_futures) {
            future.get();
        }
        
        // Final synchronization
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        size_t final_count = successful_textures.load();
        std::cout << "    Successfully loaded " << final_count << "/" << image_ids.size() << " textures" << std::endl;
        
        loaded.store(final_count > 0);
        return loaded.load();
    }
    
    std::vector<cudaTextureObject_t> getDepthTextures() const {
        std::lock_guard<std::mutex> lock(texture_mutex);
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex && tex->is_valid && tex->depth_texture != 0) {
                result.push_back(tex->depth_texture);
            }
        }
        return result;
    }
    
    std::vector<cudaTextureObject_t> getNormalTextures() const {
        std::lock_guard<std::mutex> lock(texture_mutex);
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex && tex->is_valid && tex->normal_texture != 0) {
                result.push_back(tex->normal_texture);
            }
        }
        return result;
    }
    
    std::vector<cudaTextureObject_t> getImageTextures() const {
        std::lock_guard<std::mutex> lock(texture_mutex);
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex && tex->is_valid && tex->image_texture != 0) {
                result.push_back(tex->image_texture);
            }
        }
        return result;
    }
    
    std::vector<int> getValidImageIds() const {
        std::lock_guard<std::mutex> lock(texture_mutex);
        std::vector<int> result;
        for (size_t i = 0; i < textures.size() && i < current_image_ids.size(); ++i) {
            if (textures[i] && textures[i]->is_valid) {
                result.push_back(current_image_ids[i]);
            }
        }
        return result;
    }
};

// CORRECTED: Optimized kernel with binary search instead of linear search
__global__ void CorrectedChunkBatchKernel(
    cudaTextureObject_t* depth_textures,
    cudaTextureObject_t* normal_textures, 
    cudaTextureObject_t* image_textures,
    int* texture_image_ids,
    int num_textures,
    Camera* cameras,
    int* camera_image_ids,
    int num_cameras,
    int* ref_image_ids,
    int* src_image_ids,
    int* src_counts,
    int* src_offsets,
    int* problem_offsets,
    int* widths,
    int* heights,
    PointList* output_points,
    int* valid_flags,
    int num_problems_in_chunk,
    int* image_to_camera_map,
    int* image_to_texture_map,
    int max_image_id
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIXED: Use binary search instead of linear search
    int problem_id = find_problem_id(global_idx, problem_offsets, num_problems_in_chunk);
    
    if (problem_id < 0 || problem_id >= num_problems_in_chunk) return;
    
    int local_idx = global_idx - problem_offsets[problem_id];
    int width = widths[problem_id];
    int height = heights[problem_id];
    
    if (local_idx >= width * height) return;
    
    int c = local_idx % width;
    int r = local_idx / width;
    
    int ref_image_id = ref_image_ids[problem_id];
    
    // O(1) lookup instead of linear search
    if (ref_image_id < 0 || ref_image_id > max_image_id) {
        valid_flags[global_idx] = 0;
        return;
    }
    
    int ref_cam_idx = image_to_camera_map[ref_image_id];
    int ref_tex_idx = image_to_texture_map[ref_image_id];
    
    if (ref_cam_idx < 0 || ref_tex_idx < 0 || ref_cam_idx >= num_cameras || ref_tex_idx >= num_textures) {
        valid_flags[global_idx] = 0;
        return;
    }
    
    const Camera& ref_cam = cameras[ref_cam_idx];
    
    // Sample reference depth
    float ref_depth = tex2D<float>(depth_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    
    if (ref_depth <= 0.0f) {
        valid_flags[global_idx] = 0;
        return;
    }

    // Get 3D point in world coordinates
    float3 PointX = Get3DPointonWorld_cu(static_cast<float>(c), static_cast<float>(r), ref_depth, ref_cam);
    
    // Sample reference normal and color
    float4 ref_normal_tex = tex2D<float4>(normal_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    float3 ref_normal = make_float3(ref_normal_tex.x, ref_normal_tex.y, ref_normal_tex.z);
    
    float4 ref_color = tex2D<float4>(image_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    
    // Initialize sums for weighted averaging
    float3 point_sum = PointX;
    float3 normal_sum = ref_normal;
    float color_sum[3] = {
        ref_color.z * 255.0f,  // R
        ref_color.y * 255.0f,  // G
        ref_color.x * 255.0f   // B
    };
    int num_consistent = 1;
    float confidence_sum = 1.0f;
    
    // Check source images for this problem
    int src_start = src_offsets[problem_id];
    int src_count = src_counts[problem_id];
    
    for (int j = 0; j < src_count; ++j) {
        int src_image_id = src_image_ids[src_start + j];
        
        // O(1) lookup for source indices
        if (src_image_id < 0 || src_image_id > max_image_id) continue;
        
        int src_cam_idx = image_to_camera_map[src_image_id];
        int src_tex_idx = image_to_texture_map[src_image_id];
        
        if (src_cam_idx < 0 || src_tex_idx < 0 || src_cam_idx >= num_cameras || src_tex_idx >= num_textures) continue;
        
        const Camera& src_cam = cameras[src_cam_idx];
        
        // Project and check consistency
        float2 proj_point;
        float proj_depth_in_src;
        ProjectonCamera_cu(PointX, src_cam, proj_point, proj_depth_in_src);
        
        int src_c = static_cast<int>(proj_point.x + 0.5f);
        int src_r = static_cast<int>(proj_point.y + 0.5f);
        
        if (src_c < 0 || src_c >= src_cam.width || src_r < 0 || src_r >= src_cam.height) 
            continue;
        
        float src_depth = tex2D<float>(depth_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
        if (src_depth <= 0.0f) continue;
        
        float3 PointX_src = Get3DPointonWorld_cu(static_cast<float>(src_c), static_cast<float>(src_r), src_depth, src_cam);
        
        float2 reproj_point_in_ref;
        float dummy_depth;
        ProjectonCamera_cu(PointX_src, ref_cam, reproj_point_in_ref, dummy_depth);
        
        float reproj_error = hypotf(c - reproj_point_in_ref.x, r - reproj_point_in_ref.y);
        float relative_depth_diff = fabsf(proj_depth_in_src - src_depth) / src_depth;
        
        float4 src_normal_tex = tex2D<float4>(normal_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
        float3 src_normal = make_float3(src_normal_tex.x, src_normal_tex.y, src_normal_tex.z);
        
        float dot_product = ref_normal.x * src_normal.x + ref_normal.y * src_normal.y + ref_normal.z * src_normal.z;
        dot_product = fmaxf(-1.0f, fminf(1.0f, dot_product));
        float angle = acosf(dot_product);

        float adaptive_reproj_threshold = GetAdaptiveReprojectionThreshold(ref_cam, c, r, 3.0f);
        float adaptive_reproj_threshold_src = GetAdaptiveReprojectionThreshold(src_cam, src_c, src_r, 3.0f);
        float combined_threshold = fmaxf(adaptive_reproj_threshold, adaptive_reproj_threshold_src);        

        float adaptive_depth_threshold = fmaxf(
            GetAdaptiveDepthThreshold(src_depth, src_cam),
            GetAdaptiveDepthThreshold(ref_depth, ref_cam)
        );

        if (reproj_error < combined_threshold && 
            relative_depth_diff < adaptive_depth_threshold && 
            angle < 0.12f) {
            
            float confidence = CalculateGeometricConfidence(
                ref_cam, src_cam, PointX, c, r, src_c, src_r,
                reproj_error, relative_depth_diff, angle
            );

            point_sum.x += PointX_src.x * confidence;
            point_sum.y += PointX_src.y * confidence;
            point_sum.z += PointX_src.z * confidence;

            normal_sum.x += src_normal.x * confidence;
            normal_sum.y += src_normal.y * confidence;
            normal_sum.z += src_normal.z * confidence;
            
            float4 src_color = tex2D<float4>(image_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
            color_sum[0] += src_color.z * 255.0f * confidence;
            color_sum[1] += src_color.y * 255.0f * confidence;
            color_sum[2] += src_color.x * 255.0f * confidence;

            confidence_sum += confidence;
            num_consistent++;
        }
    }
    
    if (num_consistent >= 5) {
        PointList final_point;
        
        final_point.coord = make_float3(
            point_sum.x / confidence_sum,
            point_sum.y / confidence_sum,
            point_sum.z / confidence_sum
        );
        
        float3 avg_normal = make_float3(
            normal_sum.x / confidence_sum,
            normal_sum.y / confidence_sum,
            normal_sum.z / confidence_sum
        );
        float normal_length = hypotf(hypotf(avg_normal.x, avg_normal.y), avg_normal.z);
        if (normal_length > 0.0f) {
            avg_normal.x /= normal_length;
            avg_normal.y /= normal_length;
            avg_normal.z /= normal_length;
        }
        final_point.normal = avg_normal;
        
        final_point.color = make_float3(
            color_sum[0] / confidence_sum,
            color_sum[1] / confidence_sum,
            color_sum[2] / confidence_sum
        );
        
        output_points[global_idx] = final_point;
        valid_flags[global_idx] = 1;
    } else {
        valid_flags[global_idx] = 0;
    }
}

// Smart chunking strategy
std::vector<std::vector<size_t>> createSmartChunks(const std::vector<Problem>& problems, 
                                                   size_t max_images_per_chunk) {
    std::vector<std::vector<size_t>> chunks;
    std::vector<size_t> current_chunk;
    std::unordered_set<int> current_images;
    
    for (size_t i = 0; i < problems.size(); ++i) {
        std::unordered_set<int> problem_images;
        problem_images.insert(problems[i].ref_image_id);
        for (int src_id : problems[i].src_image_ids) {
            problem_images.insert(src_id);
        }
        
        // Check if adding this problem would exceed image limit
        std::unordered_set<int> combined_images = current_images;
        combined_images.insert(problem_images.begin(), problem_images.end());
        
        if (combined_images.size() > max_images_per_chunk && !current_chunk.empty()) {
            // Start new chunk
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_images.clear();
        }
        
        // Add problem to current chunk
        current_chunk.push_back(i);
        current_images.insert(problem_images.begin(), problem_images.end());
    }
    
    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }
    
    return chunks;
}

// FULLY C++11 COMPATIBLE: Main fusion function with all fixes applied
void RunFusionCuda(const std::string &dense_folder,
                           const std::vector<Problem> &problems,
                           bool geom_consistency,
                           size_t max_images_per_chunk)
{
    std::cout << "[C++11 Compatible Fusion] Starting with " << problems.size() << " problems..." << std::endl;
    
    // Input validation
    if (problems.empty()) {
        std::cerr << "No problems to process" << std::endl;
        return;
    }
    
    if (max_images_per_chunk == 0) {
        max_images_per_chunk = 50;  // Default
    }
    
    // Estimate buffer sizes with safety margins
    size_t est_max_textures = max_images_per_chunk;
    size_t est_max_problems = 0;
    size_t est_max_src_images = 0;
    size_t est_max_pixels = 0;
    
    auto chunks = createSmartChunks(problems, max_images_per_chunk);
    
    for (const auto& chunk : chunks) {
        est_max_problems = std::max(est_max_problems, chunk.size());
        size_t chunk_src_images = 0;
        size_t chunk_pixels = 0;
        
        for (size_t prob_idx : chunk) {
            if (prob_idx >= problems.size()) continue;
            
            const Problem& problem = problems[prob_idx];
            chunk_src_images += problem.src_image_ids.size();
            
            // Conservative pixel estimate
            chunk_pixels += 3200 * 1600;
        }
        
        est_max_src_images = std::max(est_max_src_images, chunk_src_images);
        est_max_pixels = std::max(est_max_pixels, chunk_pixels);
    }
    
    std::cout << "[C++11 Compatible Fusion] Created " << chunks.size() << " chunks" << std::endl;
    std::cout << "[C++11 Compatible Fusion] Estimated max: " << est_max_textures << " textures, " 
              << est_max_problems << " problems, " << est_max_pixels << " pixels" << std::endl;
    
    // Initialize optimized managers
    OptimizedDataLoader loader(dense_folder, geom_consistency, 200);
    OptimizedTextureManager texture_manager;
    PersistentGPUBuffers gpu_buffers;
    ImageLookupTables lookup_tables;
    
    try {
        // Allocate persistent GPU buffers once
        gpu_buffers.allocateBuffers(est_max_textures, est_max_problems, est_max_src_images, est_max_pixels);
        
        // Pre-load all cameras (they're small and needed for lookup tables)
        std::unordered_set<int> all_image_ids;
        for (const auto& problem : problems) {
            all_image_ids.insert(problem.ref_image_id);
            for (int src_id : problem.src_image_ids) {
                all_image_ids.insert(src_id);
            }
        }
        
        std::vector<Camera> all_cameras;
        std::vector<int> camera_image_ids;
        
        std::cout << "[FusionFix] Loading cameras with correct resolutions..." << std::endl;
        
        for (int image_id : all_image_ids) {
            try {
                // 1. Load camera from file
                char cam_buf[512];
                int ret = snprintf(cam_buf, sizeof(cam_buf), "%s/cams/%08d_cam.txt", 
                                dense_folder.c_str(), image_id);
                if (ret < 0 || ret >= static_cast<int>(sizeof(cam_buf))) continue;
                
                Camera cam = ReadCamera(std::string(cam_buf));
                if (cam.width <= 0 || cam.height <= 0) continue;
                
                // 2. Load depth map to get TRUE resolution
                std::string depth_suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
                char depth_buf[512];
                ret = snprintf(depth_buf, sizeof(depth_buf), "%s/ACMMP/2333_%08d%s", 
                            dense_folder.c_str(), image_id, depth_suffix.c_str());
                if (ret < 0 || ret >= static_cast<int>(sizeof(depth_buf))) continue;
                
                cv::Mat_<float> depth;
                if (readDepthDmb(std::string(depth_buf), depth) != 0) continue;
                if (depth.cols <= 0 || depth.rows <= 0) continue;
                
                // 3. Load image to calculate scale
                char img_buf[512];
                ret = snprintf(img_buf, sizeof(img_buf), "%s/images/%08d.jpg", 
                            dense_folder.c_str(), image_id);
                if (ret < 0 || ret >= static_cast<int>(sizeof(img_buf))) continue;
                
                cv::Mat image = cv::imread(std::string(img_buf), cv::IMREAD_COLOR);
                if (image.empty()) continue;
                
                // 4. Calculate scale factors
                float scale_x = (float)depth.cols / (float)image.cols;
                float scale_y = (float)depth.rows / (float)image.rows;
                
                // 5. Adjust camera to match depth resolution
                cam.width = depth.cols;
                cam.height = depth.rows;
                
                if (cam.model == SPHERE) {
                    cam.params[1] *= scale_x;  // cx
                    cam.params[2] *= scale_y;  // cy
                } else {  // PINHOLE
                    cam.K[0] *= scale_x;  // fx
                    cam.K[2] *= scale_x;  // cx
                    cam.K[4] *= scale_y;  // fy
                    cam.K[5] *= scale_y;  // cy
                }
                
                // 6. Add correctly adjusted camera
                all_cameras.push_back(cam);  // ← NOW CORRECT!
                camera_image_ids.push_back(image_id);
                
            } catch (...) {
                continue;
            }
        }
    
    std::cout << "[FusionFix] Loaded " << all_cameras.size() 
              << " cameras with correct resolutions" << std::endl;
        
        if (all_cameras.empty()) {
            throw std::runtime_error("No valid cameras found");
        }
        
        // Copy cameras to GPU once
        Camera* cameras_cuda = nullptr;
        int* camera_image_ids_cuda = nullptr;
        
        CUDA_SAFE_CALL(cudaMalloc(&cameras_cuda, all_cameras.size() * sizeof(Camera)));
        CUDA_SAFE_CALL(cudaMalloc(&camera_image_ids_cuda, camera_image_ids.size() * sizeof(int)));
        CUDA_SAFE_CALL(cudaMemcpy(cameras_cuda, all_cameras.data(), 
                                   all_cameras.size() * sizeof(Camera), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(camera_image_ids_cuda, camera_image_ids.data(), 
                                   camera_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        std::vector<PointList> all_points;
        auto total_start = std::chrono::high_resolution_clock::now();
        
        // Process each chunk with corrected pipeline
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            const auto& chunk = chunks[chunk_idx];
            
            std::cout << "[C++11 Compatible Fusion] Processing chunk " << (chunk_idx + 1) << "/" << chunks.size() 
                      << " (" << chunk.size() << " problems)" << std::endl;
            
            // Get unique images for this chunk
            std::unordered_set<int> chunk_images;
            for (size_t prob_idx : chunk) {
                if (prob_idx >= problems.size()) continue;
                
                const Problem& problem = problems[prob_idx];
                chunk_images.insert(problem.ref_image_id);
                for (int src_id : problem.src_image_ids) {
                    chunk_images.insert(src_id);
                }
            }
            
            std::vector<int> chunk_image_ids(chunk_images.begin(), chunk_images.end());
            std::cout << "  Chunk images: " << chunk_image_ids.size() << std::endl;
            
            // Parallel preload chunk data
            loader.preloadChunkParallel(chunk_image_ids);
            
            // Load chunk textures
            if (!texture_manager.loadChunk(chunk_image_ids, loader)) {
                std::cerr << "Warning: Failed to load textures for chunk " << chunk_idx << std::endl;
                continue;
            }
            
            // Build lookup tables for this chunk
            const auto& texture_image_ids = texture_manager.getValidImageIds();
            lookup_tables.buildTables(camera_image_ids, texture_image_ids);
            
            // Prepare batch data for this chunk
            std::vector<int> ref_image_ids;
            std::vector<int> all_src_image_ids;
            std::vector<int> src_counts;
            std::vector<int> src_offsets;
            std::vector<int> problem_offsets;
            std::vector<int> widths;
            std::vector<int> heights;
            
            int total_pixels = 0;
            int src_offset = 0;
            
            for (size_t i = 0; i < chunk.size(); ++i) {
                size_t prob_idx = chunk[i];
                if (prob_idx >= problems.size()) continue;
                
                const Problem& problem = problems[prob_idx];
                
                ref_image_ids.push_back(problem.ref_image_id);
                
                Camera ref_cam;
                cv::Mat_<float> depth;
                cv::Mat_<cv::Vec3f> normal;
                cv::Mat image;
                if (!loader.getData(problem.ref_image_id, ref_cam, depth, normal, image)) {
                    continue;
                }
                
                widths.push_back(ref_cam.width);
                heights.push_back(ref_cam.height);
                problem_offsets.push_back(total_pixels);
                
                size_t problem_pixels;
                if (!safe_multiply(static_cast<size_t>(ref_cam.width), static_cast<size_t>(ref_cam.height), problem_pixels)) {
                    std::cerr << "Pixel count overflow for problem " << prob_idx << std::endl;
                    continue;
                }
                total_pixels += problem_pixels;
                
                src_offsets.push_back(src_offset);
                src_counts.push_back(problem.src_image_ids.size());
                
                for (int src_id : problem.src_image_ids) {
                    all_src_image_ids.push_back(src_id);
                }
                src_offset += problem.src_image_ids.size();
            }
            
            if (total_pixels == 0) continue;
            
            // Get texture data
            const auto& depth_textures = texture_manager.getDepthTextures();
            const auto& normal_textures = texture_manager.getNormalTextures();
            const auto& image_textures = texture_manager.getImageTextures();
            
            if (depth_textures.empty() || normal_textures.empty() || image_textures.empty()) {
                std::cerr << "Warning: No valid textures loaded for chunk " << chunk_idx << std::endl;
                continue;
            }
            
            // Use persistent buffers with bounds checking
            auto depth_textures_cuda = gpu_buffers.getDepthTexturesBuffer(depth_textures.size());
            auto normal_textures_cuda = gpu_buffers.getNormalTexturesBuffer(normal_textures.size());
            auto image_textures_cuda = gpu_buffers.getImageTexturesBuffer(image_textures.size());
            auto texture_image_ids_cuda = gpu_buffers.getTextureImageIdsBuffer(texture_image_ids.size());
            auto ref_image_ids_cuda = gpu_buffers.getRefImageIdsBuffer(ref_image_ids.size());
            auto all_src_image_ids_cuda = gpu_buffers.getAllSrcImageIdsBuffer(all_src_image_ids.size());
            auto src_counts_cuda = gpu_buffers.getSrcCountsBuffer(src_counts.size());
            auto src_offsets_cuda = gpu_buffers.getSrcOffsetsBuffer(src_offsets.size());
            auto problem_offsets_cuda = gpu_buffers.getProblemOffsetsBuffer(problem_offsets.size());
            auto widths_cuda = gpu_buffers.getWidthsBuffer(widths.size());
            auto heights_cuda = gpu_buffers.getHeightsBuffer(heights.size());
            auto output_points_cuda = gpu_buffers.getOutputPointsBuffer(total_pixels);
            auto valid_flags_cuda = gpu_buffers.getValidFlagsBuffer(total_pixels);
            
            // Batch copy data to GPU using async operations
            cudaStream_t copy_stream;
            CUDA_SAFE_CALL(cudaStreamCreate(&copy_stream));
            
            CUDA_SAFE_CALL(cudaMemcpyAsync(depth_textures_cuda, depth_textures.data(), 
                                          depth_textures.size() * sizeof(cudaTextureObject_t), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(normal_textures_cuda, normal_textures.data(), 
                                          normal_textures.size() * sizeof(cudaTextureObject_t), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(image_textures_cuda, image_textures.data(), 
                                          image_textures.size() * sizeof(cudaTextureObject_t), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(texture_image_ids_cuda, texture_image_ids.data(), 
                                          texture_image_ids.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            
            CUDA_SAFE_CALL(cudaMemcpyAsync(ref_image_ids_cuda, ref_image_ids.data(), 
                                          ref_image_ids.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(all_src_image_ids_cuda, all_src_image_ids.data(), 
                                          all_src_image_ids.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(src_counts_cuda, src_counts.data(), 
                                          src_counts.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(src_offsets_cuda, src_offsets.data(), 
                                          src_offsets.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(problem_offsets_cuda, problem_offsets.data(), 
                                          problem_offsets.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(widths_cuda, widths.data(), 
                                          widths.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemcpyAsync(heights_cuda, heights.data(), 
                                          heights.size() * sizeof(int), 
                                          cudaMemcpyHostToDevice, copy_stream));
            CUDA_SAFE_CALL(cudaMemsetAsync(valid_flags_cuda, 0, total_pixels * sizeof(int), copy_stream));
            
            // Wait for copy to complete
            CUDA_SAFE_CALL(cudaStreamSynchronize(copy_stream));
            CUDA_SAFE_CALL(cudaStreamDestroy(copy_stream));
            
            // Launch corrected kernel with binary search optimization
            int block_size = 256;
            int grid_size = (total_pixels + block_size - 1) / block_size;
            
            auto chunk_start = std::chrono::high_resolution_clock::now();
            
            CorrectedChunkBatchKernel<<<grid_size, block_size>>>(
                depth_textures_cuda,
                normal_textures_cuda,
                image_textures_cuda,
                texture_image_ids_cuda,
                (int)texture_image_ids.size(),
                cameras_cuda,
                camera_image_ids_cuda,
                (int)all_cameras.size(),
                ref_image_ids_cuda,
                all_src_image_ids_cuda,
                src_counts_cuda,
                src_offsets_cuda,
                problem_offsets_cuda,
                widths_cuda,
                heights_cuda,
                output_points_cuda,
                valid_flags_cuda,
                (int)chunk.size(),
                lookup_tables.d_image_to_camera_map,
                lookup_tables.d_image_to_texture_map,
                lookup_tables.max_image_id
            );
            
            CUDA_SAFE_CALL(cudaDeviceSynchronize());
            
            auto chunk_end = std::chrono::high_resolution_clock::now();
            auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(chunk_end - chunk_start);
            
            // Copy results back
            std::vector<PointList> chunk_points(total_pixels);
            std::vector<int> valid_flags_host(total_pixels);
            
            CUDA_SAFE_CALL(cudaMemcpy(chunk_points.data(), output_points_cuda, 
                                      total_pixels * sizeof(PointList), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(valid_flags_host.data(), valid_flags_cuda, 
                                      total_pixels * sizeof(int), cudaMemcpyDeviceToHost));
            
            // Collect valid points
            size_t chunk_valid_count = 0;
            for (int i = 0; i < total_pixels; ++i) {
                if (valid_flags_host[i]) {
                    all_points.push_back(chunk_points[i]);
                    chunk_valid_count++;
                }
            }
            
            std::cout << "  Chunk " << (chunk_idx + 1) << ": " << chunk_valid_count 
                      << " points in " << chunk_duration.count() << " ms" << std::endl;
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
        
        std::cout << "[C++11 Compatible Fusion] Generated " << all_points.size() << " points total" << std::endl;
        std::cout << "[C++11 Compatible Fusion] Total time: " << total_duration.count() << " seconds" << std::endl;
        
        // Write output
        std::string output_path = dense_folder + "/ACMMP/ACMM_model_cpp11_compatible.ply";
        StoreColorPlyFileBinaryPointCloud(output_path, all_points);
        
        std::cout << "[C++11 Compatible Fusion] Complete! Output written to: " << output_path << std::endl;
        std::cout << "[C++11 Compatible Fusion] Final cache size: " << loader.getCacheSize() << " images" << std::endl;
        
        // Cleanup
        if (cameras_cuda) cudaFree(cameras_cuda);
        if (camera_image_ids_cuda) cudaFree(camera_image_ids_cuda);
        
    } catch (const std::exception& e) {
        std::cerr << "[C++11 Compatible Fusion] Error: " << e.what() << std::endl;
        throw;
    }
}