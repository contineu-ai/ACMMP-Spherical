#include "ACMMP.h"
#include "ACMMP_device.cuh"  // Include the device functions header
#include "FusionGPU.h"       // Include our own header

// CUDA headers must come first
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_constants.h>

// C++ headers
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

// CUDA error checking macro
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// Thread pool for parallel file I/O
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads = std::thread::hardware_concurrency()) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
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
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }
};

// Persistent GPU buffer manager to avoid allocation overhead
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
    
    bool buffers_allocated;

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
        if (buffers_allocated) return;
        
        // Add 50% buffer to avoid reallocations
        max_textures = est_max_textures * 1.5;
        max_problems = est_max_problems * 1.5;
        max_src_images = est_max_src_images * 1.5;
        max_pixels = est_max_pixels * 1.5;
        
        std::cout << "[PersistentGPU] Allocating buffers for max: " 
                  << max_textures << " textures, " << max_problems << " problems, "
                  << max_pixels << " pixels" << std::endl;
        
        // Allocate all buffers once
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
        
        buffers_allocated = true;
        
        // Check allocated memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "[PersistentGPU] Buffers allocated. GPU memory: " 
                  << free_mem / (1024*1024) << " MB free / " 
                  << total_mem / (1024*1024) << " MB total" << std::endl;
    }
    
    ~PersistentGPUBuffers() {
        if (buffers_allocated) {
            cudaFree(depth_textures_buffer);
            cudaFree(normal_textures_buffer);
            cudaFree(image_textures_buffer);
            cudaFree(texture_image_ids_buffer);
            cudaFree(ref_image_ids_buffer);
            cudaFree(all_src_image_ids_buffer);
            cudaFree(src_counts_buffer);
            cudaFree(src_offsets_buffer);
            cudaFree(problem_offsets_buffer);
            cudaFree(widths_buffer);
            cudaFree(heights_buffer);
            cudaFree(output_points_buffer);
            cudaFree(valid_flags_buffer);
        }
    }
    
    // Get buffers with size checking
    cudaTextureObject_t* getDepthTexturesBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return depth_textures_buffer;
    }
    
    cudaTextureObject_t* getNormalTexturesBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return normal_textures_buffer;
    }
    
    cudaTextureObject_t* getImageTexturesBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return image_textures_buffer;
    }
    
    int* getTextureImageIdsBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return texture_image_ids_buffer;
    }
    
    int* getRefImageIdsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return ref_image_ids_buffer;
    }
    
    int* getAllSrcImageIdsBuffer(size_t needed) {
        if (needed > max_src_images) {
            throw std::runtime_error("Source images buffer size exceeded");
        }
        return all_src_image_ids_buffer;
    }
    
    int* getSrcCountsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return src_counts_buffer;
    }
    
    int* getSrcOffsetsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return src_offsets_buffer;
    }
    
    int* getProblemOffsetsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return problem_offsets_buffer;
    }
    
    int* getWidthsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return widths_buffer;
    }
    
    int* getHeightsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return heights_buffer;
    }
    
    PointList* getOutputPointsBuffer(size_t needed) {
        if (needed > max_pixels) {
            throw std::runtime_error("Output buffer size exceeded");
        }
        return output_points_buffer;
    }
    
    int* getValidFlagsBuffer(size_t needed) {
        if (needed > max_pixels) {
            throw std::runtime_error("Valid flags buffer size exceeded");
        }
        return valid_flags_buffer;
    }
};

// Image data structure for efficient caching
struct ImageData {
    Camera camera;
    cv::Mat_<float> depth;
    cv::Mat_<cv::Vec3f> normal;
    cv::Mat image;
    bool valid;
    std::chrono::steady_clock::time_point last_accessed;
    
    ImageData() : valid(false) {}
};

// Optimized data loader with parallel I/O and LRU cache
class OptimizedDataLoader {
private:
    std::string dense_folder;
    std::string img_folder; 
    std::string cam_folder;
    bool geom_consistency;
    
    // LRU cache with thread safety
    std::unordered_map<int, std::shared_ptr<ImageData>> cache;
    std::list<int> lru_order;
    std::unordered_map<int, std::list<int>::iterator> lru_map;
    
    size_t max_cache_size;
    mutable std::mutex cache_mutex;
    
    // Thread pool for parallel I/O
    std::unique_ptr<ThreadPool> thread_pool;
    
    void updateLRU(int image_id) {
        auto it = lru_map.find(image_id);
        if (it != lru_map.end()) {
            lru_order.erase(it->second);
        }
        lru_order.push_front(image_id);
        lru_map[image_id] = lru_order.begin();
    }
    
    void trimCache() {
        while (cache.size() > max_cache_size && !lru_order.empty()) {
            int oldest = lru_order.back();
            lru_order.pop_back();
            lru_map.erase(oldest);
            cache.erase(oldest);
        }
    }
    
    std::shared_ptr<ImageData> loadImageDataSync(int image_id) {
        auto data = std::make_shared<ImageData>();
        data->valid = false;
        
        // Load camera
        char buf[256];
        sprintf(buf, "%s/%08d_cam.txt", cam_folder.c_str(), image_id);
        try {
            data->camera = ReadCamera(std::string(buf));
            if (data->camera.width <= 0 || data->camera.height <= 0) {
                return data;
            }
        } catch (...) {
            return data;
        }
        
        // Load depth
        std::string depth_suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
        sprintf(buf, "%s/ACMMP/2333_%08d%s", dense_folder.c_str(), image_id, depth_suffix.c_str());
        if (readDepthDmb(std::string(buf), data->depth) != 0 || 
            data->depth.cols <= 0 || data->depth.rows <= 0) {
            return data;
        }
        
        // Load normal
        sprintf(buf, "%s/ACMMP/2333_%08d/normals.dmb", dense_folder.c_str(), image_id);
        if (readNormalDmb(std::string(buf), data->normal) != 0 || 
            data->normal.cols <= 0 || data->normal.rows <= 0) {
            return data;
        }
        
        // Load image
        sprintf(buf, "%s/%08d.jpg", img_folder.c_str(), image_id);
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
        
        data->valid = true;
        data->last_accessed = std::chrono::steady_clock::now();
        return data;
    }
    
public:
    OptimizedDataLoader(const std::string& folder, bool geom = false, size_t cache_size = 100) 
        : dense_folder(folder), geom_consistency(geom), max_cache_size(cache_size) {
        img_folder = folder + "/images";
        cam_folder = folder + "/cams";
        
        // Create thread pool with optimal number of threads for I/O
        unsigned int hw_threads = std::thread::hardware_concurrency();
        size_t io_threads = 16;
        if (hw_threads >= 4) {
            io_threads = std::min(static_cast<size_t>(16), static_cast<size_t>(hw_threads));
        } else {
            io_threads = 4;
        }
        thread_pool = std::unique_ptr<ThreadPool>(new ThreadPool(io_threads));
        
        std::cout << "[OptimizedLoader] Using " << io_threads << " threads for parallel I/O" << std::endl;
    }
    
    // Parallel preload for a chunk of images
    void preloadChunkParallel(const std::vector<int>& image_ids) {
        std::vector<std::future<std::shared_ptr<ImageData>>> futures;
        std::mutex results_mutex;
        
        std::cout << "  Preloading " << image_ids.size() << " images in parallel..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch parallel loading tasks
        for (int image_id : image_ids) {
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (cache.find(image_id) != cache.end()) {
                    // Already cached, skip
                    continue;
                }
            }
            
            // Submit loading task to thread pool
            auto future = thread_pool->enqueue([this, image_id]() {
                return loadImageDataSync(image_id);
            });
            futures.push_back(std::move(future));
        }
        
        // Collect results and store in cache
        size_t loaded_count = 0;
        for (size_t i = 0; i < futures.size(); ++i) {
            int image_id = image_ids[i];
            auto data = futures[i].get();
            
            if (data && data->valid) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache[image_id] = data;
                updateLRU(image_id);
                loaded_count++;
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            trimCache();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "  Loaded " << loaded_count << "/" << image_ids.size() 
                  << " images in " << duration.count() << " ms" << std::endl;
    }
    
    bool getData(int image_id, Camera& cam, cv::Mat_<float>& depth, 
                cv::Mat_<cv::Vec3f>& normal, cv::Mat& image) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = cache.find(image_id);
        if (it == cache.end() || !it->second->valid) {
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
        std::lock_guard<std::mutex> lock(cache_mutex);
        return cache.size();
    }
    
    void clearCache() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.clear();
        lru_order.clear();
        lru_map.clear();
    }
};

// Optimized texture manager with reduced allocation overhead
class OptimizedTextureManager {
private:
    struct TextureData {
        cudaArray* depth_array;
        cudaArray* normal_array;
        cudaArray* image_array;
        cudaTextureObject_t depth_texture;
        cudaTextureObject_t normal_texture;
        cudaTextureObject_t image_texture;
        bool is_valid;
        
        TextureData() : depth_array(nullptr), normal_array(nullptr), image_array(nullptr),
                       depth_texture(0), normal_texture(0), image_texture(0), is_valid(false) {}
        
        void cleanup() {
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
    
    std::vector<TextureData> textures;
    std::vector<int> current_image_ids;
    bool loaded = false;
    
    // Use multiple streams for parallel texture creation
    static const int num_streams = 4;
    cudaStream_t streams[num_streams];
    
    void release() {
        if (!loaded && textures.empty()) {
            return;
        }
        
        // Synchronize all streams
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();
        
        // Clean up all textures
        for (auto& tex : textures) {
            tex.cleanup();
        }
        
        textures.clear();
        current_image_ids.clear();
        loaded = false;
    }

public:
    OptimizedTextureManager() {
        for (int i = 0; i < num_streams; ++i) {
            CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
        }
    }
    
    ~OptimizedTextureManager() {
        release();
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
    }
    
    bool loadChunk(const std::vector<int>& image_ids, OptimizedDataLoader& loader) {
        release();  // Clean up previous chunk
        
        if (image_ids.empty()) {
            return false;
        }
        
        current_image_ids = image_ids;
        textures.resize(image_ids.size());
        
        bool overall_success = true;
        size_t successful_textures = 0;
        
        // Process textures in parallel using streams
        for (size_t i = 0; i < image_ids.size(); ++i) {
            int image_id = image_ids[i];
            cudaStream_t stream = streams[i % num_streams];
            
            Camera cam;
            cv::Mat_<float> depth;
            cv::Mat_<cv::Vec3f> normal;
            cv::Mat image;
            
            if (!loader.getData(image_id, cam, depth, normal, image)) {
                continue;
            }
            
            TextureData& tex = textures[i];
            bool texture_success = true;
            
            // Create textures with error handling
            try {
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
                
                tex.is_valid = true;
                successful_textures++;
                
            } catch (const std::exception& e) {
                std::cerr << "    Failed to create texture for image " << image_id << ": " << e.what() << std::endl;
                tex.cleanup();
                texture_success = false;
                overall_success = false;
            }
        }
        
        // Sync all streams
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        std::cout << "    Successfully loaded " << successful_textures << "/" << image_ids.size() << " textures" << std::endl;
        
        loaded = (successful_textures > 0);
        return loaded;
    }
    
    std::vector<cudaTextureObject_t> getDepthTextures() const {
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex.is_valid && tex.depth_texture != 0) {
                result.push_back(tex.depth_texture);
            }
        }
        return result;
    }
    
    std::vector<cudaTextureObject_t> getNormalTextures() const {
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex.is_valid && tex.normal_texture != 0) {
                result.push_back(tex.normal_texture);
            }
        }
        return result;
    }
    
    std::vector<cudaTextureObject_t> getImageTextures() const {
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex.is_valid && tex.image_texture != 0) {
                result.push_back(tex.image_texture);
            }
        }
        return result;
    }
    
    std::vector<int> getValidImageIds() const {
        std::vector<int> result;
        for (size_t i = 0; i < textures.size() && i < current_image_ids.size(); ++i) {
            if (textures[i].is_valid) {
                result.push_back(current_image_ids[i]);
            }
        }
        return result;
    }
};

// Fast batch kernel 
__global__ void ChunkBatchKernel(
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
    int num_problems_in_chunk
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find which problem this thread belongs to
    int problem_id = 0;
    while (problem_id < num_problems_in_chunk - 1 && global_idx >= problem_offsets[problem_id + 1]) {
        problem_id++;
    }
    
    if (problem_id >= num_problems_in_chunk) return;
    
    int local_idx = global_idx - problem_offsets[problem_id];
    int width = widths[problem_id];
    int height = heights[problem_id];
    
    if (local_idx >= width * height) return;
    
    int c = local_idx % width;
    int r = local_idx / width;
    
    int ref_image_id = ref_image_ids[problem_id];
    
    // Find reference camera and texture
    int ref_cam_idx = -1;
    int ref_tex_idx = -1;
    
    for (int i = 0; i < num_cameras; ++i) {
        if (camera_image_ids[i] == ref_image_id) {
            ref_cam_idx = i;
            break;
        }
    }
    
    for (int i = 0; i < num_textures; ++i) {
        if (texture_image_ids[i] == ref_image_id) {
            ref_tex_idx = i;
            break;
        }
    }
    
    if (ref_cam_idx == -1 || ref_tex_idx == -1) {
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
    
    // Initialize sums for averaging
    float3 point_sum = PointX;
    float3 normal_sum = ref_normal;
    float color_sum[3] = {
        ref_color.z * 255.0f,  // R
        ref_color.y * 255.0f,  // G
        ref_color.x * 255.0f   // B
    };
    int num_consistent = 1;
    
    // Check source images for this problem
    int src_start = src_offsets[problem_id];
    int src_count = src_counts[problem_id];
    
    for (int j = 0; j < src_count; ++j) {
        int src_image_id = src_image_ids[src_start + j];
        
        // Find source camera and texture
        int src_cam_idx = -1;
        int src_tex_idx = -1;
        
        for (int i = 0; i < num_cameras; ++i) {
            if (camera_image_ids[i] == src_image_id) {
                src_cam_idx = i;
                break;
            }
        }
        
        for (int i = 0; i < num_textures; ++i) {
            if (texture_image_ids[i] == src_image_id) {
                src_tex_idx = i;
                break;
            }
        }
        
        if (src_cam_idx == -1 || src_tex_idx == -1) continue;
        
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
        
        if (reproj_error < 1.0 && relative_depth_diff < 0.005f && angle < 0.05f) {
            point_sum.x += PointX_src.x;
            point_sum.y += PointX_src.y;
            point_sum.z += PointX_src.z;
            
            normal_sum.x += src_normal.x;
            normal_sum.y += src_normal.y;
            normal_sum.z += src_normal.z;
            
            float4 src_color = tex2D<float4>(image_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
            color_sum[0] += src_color.z * 255.0f;
            color_sum[1] += src_color.y * 255.0f;
            color_sum[2] += src_color.x * 255.0f;
            
            num_consistent++;
        }
    }
    
    if (num_consistent >= 3) {
        PointList final_point;
        
        final_point.coord = make_float3(
            point_sum.x / num_consistent,
            point_sum.y / num_consistent,
            point_sum.z / num_consistent
        );
        
        float3 avg_normal = make_float3(
            normal_sum.x / num_consistent,
            normal_sum.y / num_consistent,
            normal_sum.z / num_consistent
        );
        float normal_length = hypotf(hypotf(avg_normal.x, avg_normal.y), avg_normal.z);
        if (normal_length > 0.0f) {
            avg_normal.x /= normal_length;
            avg_normal.y /= normal_length;
            avg_normal.z /= normal_length;
        }
        final_point.normal = avg_normal;
        
        final_point.color = make_float3(
            color_sum[0] / num_consistent,
            color_sum[1] / num_consistent,
            color_sum[2] / num_consistent
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

// Main optimized fusion function
void RunFusionCuda(const std::string &dense_folder,
                           const std::vector<Problem> &problems,
                           bool geom_consistency,
                           size_t max_images_per_chunk)
{
    std::cout << "[Optimized Fusion] Starting with " << problems.size() << " problems..." << std::endl;
    
    // Estimate buffer sizes
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
            const Problem& problem = problems[prob_idx];
            chunk_src_images += problem.src_image_ids.size();
            
            // Estimate pixels (assume average image size)
            chunk_pixels += 3200 * 1600;  // Conservative estimate
        }
        
        est_max_src_images = std::max(est_max_src_images, chunk_src_images);
        est_max_pixels = std::max(est_max_pixels, chunk_pixels);
    }
    
    std::cout << "[Optimized Fusion] Created " << chunks.size() << " chunks" << std::endl;
    std::cout << "[Optimized Fusion] Estimated max: " << est_max_textures << " textures, " 
              << est_max_problems << " problems, " << est_max_pixels << " pixels" << std::endl;
    
    // Initialize optimized managers
    OptimizedDataLoader loader(dense_folder, geom_consistency, 200);
    OptimizedTextureManager texture_manager;
    PersistentGPUBuffers gpu_buffers;
    
    // Allocate persistent GPU buffers once
    gpu_buffers.allocateBuffers(est_max_textures, est_max_problems, est_max_src_images, est_max_pixels);
    
    // Pre-load all cameras (they're small)
    std::unordered_set<int> all_image_ids;
    for (const auto& problem : problems) {
        all_image_ids.insert(problem.ref_image_id);
        for (int src_id : problem.src_image_ids) {
            all_image_ids.insert(src_id);
        }
    }
    
    std::vector<Camera> all_cameras;
    std::vector<int> camera_image_ids;
    for (int image_id : all_image_ids) {
        char buf[256];
        sprintf(buf, "%s/cams/%08d_cam.txt", dense_folder.c_str(), image_id);
        try {
            Camera cam = ReadCamera(std::string(buf));
            if (cam.width > 0 && cam.height > 0) {
                all_cameras.push_back(cam);
                camera_image_ids.push_back(image_id);
            }
        } catch (...) {
            continue;
        }
    }
    
    // Copy cameras to GPU once
    Camera* cameras_cuda;
    int* camera_image_ids_cuda;
    CUDA_SAFE_CALL(cudaMalloc(&cameras_cuda, all_cameras.size() * sizeof(Camera)));
    CUDA_SAFE_CALL(cudaMalloc(&camera_image_ids_cuda, camera_image_ids.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(cameras_cuda, all_cameras.data(), 
                               all_cameras.size() * sizeof(Camera), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(camera_image_ids_cuda, camera_image_ids.data(), 
                               camera_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    std::vector<PointList> all_points;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Process each chunk with optimized pipeline
    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        const auto& chunk = chunks[chunk_idx];
        
        std::cout << "[Optimized Fusion] Processing chunk " << (chunk_idx + 1) << "/" << chunks.size() 
                  << " (" << chunk.size() << " problems)" << std::endl;
        
        // Get unique images for this chunk
        std::unordered_set<int> chunk_images;
        for (size_t prob_idx : chunk) {
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
            total_pixels += ref_cam.width * ref_cam.height;
            
            src_offsets.push_back(src_offset);
            src_counts.push_back(problem.src_image_ids.size());
            
            for (int src_id : problem.src_image_ids) {
                all_src_image_ids.push_back(src_id);
            }
            src_offset += problem.src_image_ids.size();
        }
        
        if (total_pixels == 0) continue;
        
        // Use persistent GPU buffers instead of allocating new ones
        const auto& depth_textures = texture_manager.getDepthTextures();
        const auto& normal_textures = texture_manager.getNormalTextures();
        const auto& image_textures = texture_manager.getImageTextures();
        const auto& texture_image_ids = texture_manager.getValidImageIds();
        
        if (depth_textures.empty() || normal_textures.empty() || image_textures.empty()) {
            std::cerr << "Warning: No valid textures loaded for chunk " << chunk_idx << std::endl;
            continue;
        }
        
        // Get reusable buffers
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
        
        // Copy data to GPU (reusing buffers)
        CUDA_SAFE_CALL(cudaMemcpy(depth_textures_cuda, depth_textures.data(), depth_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(normal_textures_cuda, normal_textures.data(), normal_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(image_textures_cuda, image_textures.data(), image_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(texture_image_ids_cuda, texture_image_ids.data(), texture_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ref_image_ids_cuda, ref_image_ids.data(), ref_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(all_src_image_ids_cuda, all_src_image_ids.data(), all_src_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(src_counts_cuda, src_counts.data(), src_counts.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(src_offsets_cuda, src_offsets.data(), src_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(problem_offsets_cuda, problem_offsets.data(), problem_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(widths_cuda, widths.data(), widths.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(heights_cuda, heights.data(), heights.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(valid_flags_cuda, 0, total_pixels * sizeof(int)));
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (total_pixels + block_size - 1) / block_size;
        
        auto chunk_start = std::chrono::high_resolution_clock::now();
        
        ChunkBatchKernel<<<grid_size, block_size>>>(
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
            (int)chunk.size()
        );
        
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        auto chunk_end = std::chrono::high_resolution_clock::now();
        auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(chunk_end - chunk_start);
        
        // Copy results back
        std::vector<PointList> chunk_points(total_pixels);
        std::vector<int> valid_flags_host(total_pixels);
        
        CUDA_SAFE_CALL(cudaMemcpy(chunk_points.data(), output_points_cuda, total_pixels * sizeof(PointList), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(valid_flags_host.data(), valid_flags_cuda, total_pixels * sizeof(int), cudaMemcpyDeviceToHost));
        
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
    
    std::cout << "[Optimized Fusion] Generated " << all_points.size() << " points total" << std::endl;
    std::cout << "[Optimized Fusion] Total time: " << total_duration.count() << " seconds" << std::endl;
    
    // Write output
    std::string output_path = dense_folder + "/ACMMP/ACMM_model_optimized.ply";
    StoreColorPlyFileBinaryPointCloud(output_path, all_points);
    
    std::cout << "[Optimized Fusion] Complete! Output written to: " << output_path << std::endl;
    std::cout << "[Optimized Fusion] Final cache size: " << loader.getCacheSize() << " images" << std::endl;
    
    // Cleanup
    cudaFree(cameras_cuda);
    cudaFree(camera_image_ids_cuda);
}