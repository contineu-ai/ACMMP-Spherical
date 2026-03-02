// FULLY C++11 COMPATIBLE VERSION - Memory Optimized
// FIXED: Memory leaks and accumulation issues
// FIXED: Limited async concurrency to prevent memory spikes
// FIXED: Streaming point output to disk
// FIXED: Reduced cache size and explicit deallocation

#include "ACMMP.h"
#include "ACMMP_device.cuh"
#include "FusionGPU.h"

#include "CompressedDMB.h"
#include <cuda_runtime.h>
#include <cuda.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <math_constants.h>

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
#include <sstream>
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

#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        throw std::runtime_error("CUDA operation failed"); \
    } \
} while(0)
#endif

// ============================================================================
// SECTION 1: MEMORY MONITORING
// ============================================================================

class MemoryMonitor {
public:
    static void logUsage(const std::string& stage) {
        std::ifstream status("/proc/self/status");
        std::string line;
        size_t vmrss = 0, vmhwm = 0;
        
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line.substr(6));
                iss >> vmrss;
            } else if (line.substr(0, 6) == "VmHWM:") {
                std::istringstream iss(line.substr(6));
                iss >> vmhwm;
            }
        }
        
        std::cout << "[Memory] " << stage << " - Current: " << (vmrss / 1024) 
                  << " MB, Peak: " << (vmhwm / 1024) << " MB" << std::endl;
    }
    
    static size_t getCurrentUsageMB() {
        std::ifstream status("/proc/self/status");
        std::string line;
        while (std::getline(status, line)) {
            if (line.substr(0, 6) == "VmRSS:") {
                std::istringstream iss(line.substr(6));
                size_t kb;
                iss >> kb;
                return kb / 1024;
            }
        }
        return 0;
    }
};

// ============================================================================
// SECTION 2: RAII WRAPPERS FOR MEMORY SAFETY
// ============================================================================

template<typename T>
class CudaMemoryGuard {
private:
    T* ptr;
    std::string name;
public:
    CudaMemoryGuard(const std::string& debug_name = "unnamed") : ptr(nullptr), name(debug_name) {}
    
    ~CudaMemoryGuard() { 
        if (ptr) { 
            cudaFree(ptr); 
            ptr = nullptr;
        } 
    }
    
    void alloc(size_t count) {
        if (ptr) {
            cudaFree(ptr);
            ptr = nullptr;
        }
        CUDA_SAFE_CALL(cudaMalloc(&ptr, count * sizeof(T)));
    }
    
    T* get() { return ptr; }
    const T* get() const { return ptr; }
    
    T* release() {
        T* tmp = ptr;
        ptr = nullptr;
        return tmp;
    }
    
    bool isAllocated() const { return ptr != nullptr; }
    
    CudaMemoryGuard(const CudaMemoryGuard&) = delete;
    CudaMemoryGuard& operator=(const CudaMemoryGuard&) = delete;
    
    CudaMemoryGuard(CudaMemoryGuard&& other) : ptr(other.ptr), name(std::move(other.name)) { 
        other.ptr = nullptr; 
    }
    CudaMemoryGuard& operator=(CudaMemoryGuard&& other) {
        if (this != &other) {
            if (ptr) cudaFree(ptr);
            ptr = other.ptr;
            name = std::move(other.name);
            other.ptr = nullptr;
        }
        return *this;
    }
};

class CudaStreamGuard {
private:
    cudaStream_t stream;
    bool valid;
public:
    CudaStreamGuard() : stream(0), valid(false) {
        cudaError_t err = cudaStreamCreate(&stream);
        if (err == cudaSuccess) {
            valid = true;
        } else {
            std::cerr << "[WARNING] Failed to create CUDA stream: " 
                      << cudaGetErrorString(err) << std::endl;
        }
    }
    
    ~CudaStreamGuard() {
        if (valid && stream != 0) {
            cudaStreamSynchronize(stream);
            cudaStreamDestroy(stream);
        }
    }
    
    cudaStream_t get() { return stream; }
    bool isValid() const { return valid; }
    
    void synchronize() {
        if (valid) {
            CUDA_SAFE_CALL(cudaStreamSynchronize(stream));
        }
    }
    
    CudaStreamGuard(const CudaStreamGuard&) = delete;
    CudaStreamGuard& operator=(const CudaStreamGuard&) = delete;
};

// ============================================================================
// SECTION 3: LOGGING AND FAILURE TRACKING
// ============================================================================

enum class LogLevel { DEBUG, INFO, WARNING, ERROR };

class FusionLogger {
private:
    static LogLevel min_level;
    
public:
    static void setLevel(LogLevel level) { min_level = level; }
    
    static void log(LogLevel level, const std::string& component, const std::string& message) {
        if (level < min_level) return;
        
        const char* level_str;
        switch (level) {
            case LogLevel::DEBUG:   level_str = "DEBUG"; break;
            case LogLevel::INFO:    level_str = "INFO"; break;
            case LogLevel::WARNING: level_str = "WARNING"; break;
            case LogLevel::ERROR:   level_str = "ERROR"; break;
            default: level_str = "UNKNOWN";
        }
        
        std::cerr << "[" << level_str << "][" << component << "] " << message << std::endl;
    }
    
    static void debug(const std::string& component, const std::string& message) {
        log(LogLevel::DEBUG, component, message);
    }
    
    static void info(const std::string& component, const std::string& message) {
        log(LogLevel::INFO, component, message);
    }
    
    static void warning(const std::string& component, const std::string& message) {
        log(LogLevel::WARNING, component, message);
    }
    
    static void error(const std::string& component, const std::string& message) {
        log(LogLevel::ERROR, component, message);
    }
};

LogLevel FusionLogger::min_level = LogLevel::INFO;

struct ImageLoadFailure {
    int image_id;
    std::string stage;
    std::string reason;
    std::chrono::steady_clock::time_point timestamp;
    
    ImageLoadFailure(int id, const std::string& s, const std::string& r)
        : image_id(id), stage(s), reason(r), timestamp(std::chrono::steady_clock::now()) {}
};

struct ChunkFailure {
    size_t chunk_idx;
    std::string stage;
    std::string reason;
    std::vector<int> affected_image_ids;
    size_t problems_in_chunk;
    
    ChunkFailure(size_t idx, const std::string& s, const std::string& r)
        : chunk_idx(idx), stage(s), reason(r), problems_in_chunk(0) {}
};

class FailureTracker {
private:
    std::vector<ImageLoadFailure> image_failures;
    std::vector<ChunkFailure> chunk_failures;
    mutable std::mutex tracker_mutex;
    
public:
    void recordImageFailure(int image_id, const std::string& stage, const std::string& reason) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        image_failures.emplace_back(image_id, stage, reason);
        
        FusionLogger::warning("ImageLoad", 
            "Image " + std::to_string(image_id) + " failed at stage '" + stage + "': " + reason);
    }
    
    void recordChunkFailure(size_t chunk_idx, const std::string& stage, const std::string& reason,
                           const std::vector<int>& affected_images = std::vector<int>(), 
                           size_t num_problems = 0) {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        ChunkFailure failure(chunk_idx, stage, reason);
        failure.affected_image_ids = affected_images;
        failure.problems_in_chunk = num_problems;
        chunk_failures.push_back(failure);
        
        std::string msg = "Chunk " + std::to_string(chunk_idx) + " failed at stage '" + stage + "': " + reason;
        if (num_problems > 0) {
            msg += " (affected " + std::to_string(num_problems) + " problems)";
        }
        FusionLogger::error("ChunkProcess", msg);
    }
    
    void printSummary() const {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        
        if (image_failures.empty() && chunk_failures.empty()) {
            FusionLogger::info("Summary", "No failures recorded - all processing successful!");
            return;
        }
        
        std::cout << "\n" << std::string(60, '=') << std::endl;
        std::cout << "FAILURE SUMMARY" << std::endl;
        std::cout << std::string(60, '=') << std::endl;
        
        if (!image_failures.empty()) {
            std::map<std::string, std::vector<int>> failures_by_stage;
            std::map<std::string, std::vector<std::string>> reasons_by_stage;
            
            for (const auto& f : image_failures) {
                failures_by_stage[f.stage].push_back(f.image_id);
                reasons_by_stage[f.stage].push_back(f.reason);
            }
            
            std::cout << "\nImage Loading Failures (" << image_failures.size() << " total):" << std::endl;
            for (const auto& pair : failures_by_stage) {
                std::cout << "  Stage '" << pair.first << "': " << pair.second.size() << " images" << std::endl;
                
                std::cout << "    IDs: ";
                size_t show_count = std::min(pair.second.size(), size_t(10));
                for (size_t i = 0; i < show_count; ++i) {
                    std::cout << pair.second[i];
                    if (i < show_count - 1) std::cout << ", ";
                }
                if (pair.second.size() > 10) {
                    std::cout << " ... and " << (pair.second.size() - 10) << " more";
                }
                std::cout << std::endl;
                
                if (!reasons_by_stage[pair.first].empty()) {
                    std::cout << "    Example reason: " << reasons_by_stage[pair.first][0] << std::endl;
                }
            }
        }
        
        if (!chunk_failures.empty()) {
            std::cout << "\nChunk Processing Failures (" << chunk_failures.size() << " total):" << std::endl;
            for (const auto& f : chunk_failures) {
                std::cout << "  Chunk " << f.chunk_idx << " at stage '" << f.stage << "': " 
                          << f.reason << std::endl;
                if (f.problems_in_chunk > 0) {
                    std::cout << "    Affected problems: " << f.problems_in_chunk << std::endl;
                }
            }
        }
        
        std::cout << std::string(60, '=') << std::endl << std::endl;
    }
    
    size_t getImageFailureCount() const {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        return image_failures.size();
    }
    
    size_t getChunkFailureCount() const {
        std::lock_guard<std::mutex> lock(tracker_mutex);
        return chunk_failures.size();
    }
};

// ============================================================================
// SECTION 4: SAFE ARITHMETIC AND UTILITY FUNCTIONS
// ============================================================================

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

// ============================================================================
// SECTION 5: STREAMING POINT WRITER
// ============================================================================

class StreamingPointWriter {
private:
    std::string temp_path;
    std::ofstream temp_stream;
    size_t point_count;
    mutable std::mutex write_mutex;
    
public:
    StreamingPointWriter(const std::string& output_folder) 
        : point_count(0) {
        temp_path = output_folder + "/ACMMP/points_temp.bin";
        temp_stream.open(temp_path, std::ios::binary | std::ios::trunc);
        if (!temp_stream.is_open()) {
            throw std::runtime_error("Failed to open temp file: " + temp_path);
        }
        FusionLogger::info("StreamWriter", "Opened temp file: " + temp_path);
    }
    
    ~StreamingPointWriter() {
        if (temp_stream.is_open()) {
            temp_stream.close();
        }
        // Clean up temp file if it exists
        std::remove(temp_path.c_str());
    }
    
    void writePoints(const std::vector<PointList>& points, const std::vector<int>& valid_flags) {
        std::lock_guard<std::mutex> lock(write_mutex);
        
        for (size_t i = 0; i < points.size(); ++i) {
            if (valid_flags[i]) {
                temp_stream.write(reinterpret_cast<const char*>(&points[i]), sizeof(PointList));
                point_count++;
            }
        }
        temp_stream.flush();
    }
    
    size_t getPointCount() const {
        std::lock_guard<std::mutex> lock(write_mutex);
        return point_count;
    }
    
    void finalize(const std::string& output_path) {
        std::lock_guard<std::mutex> lock(write_mutex);
        
        temp_stream.close();
        
        FusionLogger::info("StreamWriter", "Finalizing " + std::to_string(point_count) + " points to PLY");
        
        // Read back and write PLY
        std::ifstream temp_read(temp_path, std::ios::binary);
        if (!temp_read.is_open()) {
            throw std::runtime_error("Failed to reopen temp file for reading");
        }
        
        std::ofstream ply_file(output_path, std::ios::binary);
        if (!ply_file.is_open()) {
            throw std::runtime_error("Failed to open output PLY file: " + output_path);
        }
        
        // Write PLY header
        ply_file << "ply\n";
        ply_file << "format binary_little_endian 1.0\n";
        ply_file << "element vertex " << point_count << "\n";
        ply_file << "property float x\n";
        ply_file << "property float y\n";
        ply_file << "property float z\n";
        ply_file << "property float nx\n";
        ply_file << "property float ny\n";
        ply_file << "property float nz\n";
        ply_file << "property uchar red\n";
        ply_file << "property uchar green\n";
        ply_file << "property uchar blue\n";
        ply_file << "end_header\n";
        
        // Stream points from temp file to PLY
        const size_t BATCH_SIZE = 100000;
        std::vector<PointList> batch(BATCH_SIZE);
        size_t written = 0;
        
        while (written < point_count) {
            size_t to_read = std::min(BATCH_SIZE, point_count - written);
            temp_read.read(reinterpret_cast<char*>(batch.data()), to_read * sizeof(PointList));
            
            for (size_t i = 0; i < to_read; ++i) {
                const PointList& p = batch[i];
                
                ply_file.write(reinterpret_cast<const char*>(&p.coord.x), sizeof(float));
                ply_file.write(reinterpret_cast<const char*>(&p.coord.y), sizeof(float));
                ply_file.write(reinterpret_cast<const char*>(&p.coord.z), sizeof(float));
                ply_file.write(reinterpret_cast<const char*>(&p.normal.x), sizeof(float));
                ply_file.write(reinterpret_cast<const char*>(&p.normal.y), sizeof(float));
                ply_file.write(reinterpret_cast<const char*>(&p.normal.z), sizeof(float));
                
                unsigned char r = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, p.color.x)));
                unsigned char g = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, p.color.y)));
                unsigned char b = static_cast<unsigned char>(std::min(255.0f, std::max(0.0f, p.color.z)));
                
                ply_file.write(reinterpret_cast<const char*>(&r), 1);
                ply_file.write(reinterpret_cast<const char*>(&g), 1);
                ply_file.write(reinterpret_cast<const char*>(&b), 1);
            }
            
            written += to_read;
        }
        
        temp_read.close();
        ply_file.close();
        
        // Remove temp file
        std::remove(temp_path.c_str());
        
        FusionLogger::info("StreamWriter", "Wrote " + std::to_string(point_count) + " points to " + output_path);
    }
};

// ============================================================================
// SECTION 6: LOOKUP TABLES WITH EXCEPTION SAFETY
// ============================================================================

struct ImageLookupTables {
    std::unordered_map<int, int> image_to_camera_idx;
    std::unordered_map<int, int> image_to_texture_idx;
    
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
        int* temp_camera_map = nullptr;
        int* temp_texture_map = nullptr;
        
        try {
            cleanup();
            
            max_image_id = 0;
            image_to_camera_idx.clear();
            image_to_texture_idx.clear();
            
            for (size_t i = 0; i < camera_image_ids.size(); ++i) {
                image_to_camera_idx[camera_image_ids[i]] = static_cast<int>(i);
                max_image_id = std::max(max_image_id, camera_image_ids[i]);
            }
            
            for (size_t i = 0; i < texture_image_ids.size(); ++i) {
                image_to_texture_idx[texture_image_ids[i]] = static_cast<int>(i);
                max_image_id = std::max(max_image_id, texture_image_ids[i]);
            }
            
            std::vector<int> camera_map(max_image_id + 1, -1);
            std::vector<int> texture_map(max_image_id + 1, -1);
            
            for (const auto& pair : image_to_camera_idx) {
                camera_map[pair.first] = pair.second;
            }
            for (const auto& pair : image_to_texture_idx) {
                texture_map[pair.first] = pair.second;
            }
            
            size_t map_size = (max_image_id + 1) * sizeof(int);
            CUDA_SAFE_CALL(cudaMalloc(&temp_camera_map, map_size));
            CUDA_SAFE_CALL(cudaMalloc(&temp_texture_map, map_size));
            
            CUDA_SAFE_CALL(cudaMemcpy(temp_camera_map, camera_map.data(), 
                                      map_size, cudaMemcpyHostToDevice));
            CUDA_SAFE_CALL(cudaMemcpy(temp_texture_map, texture_map.data(), 
                                      map_size, cudaMemcpyHostToDevice));
            
            d_image_to_camera_map = temp_camera_map;
            d_image_to_texture_map = temp_texture_map;
            temp_camera_map = nullptr;
            temp_texture_map = nullptr;
            
            std::cout << "[LookupTables] Built lookup tables for " << camera_image_ids.size() 
                      << " cameras, " << texture_image_ids.size() << " textures, max_id=" 
                      << max_image_id << std::endl;
                      
        } catch (...) {
            if (temp_camera_map) cudaFree(temp_camera_map);
            if (temp_texture_map) cudaFree(temp_texture_map);
            throw;
        }
    }
};

// ============================================================================
// SECTION 7: THREAD POOL (C++11 COMPATIBLE)
// ============================================================================

class EfficientThreadPool {
private:
    std::vector<std::thread> workers;
    std::vector<std::queue<std::function<void()>>> task_queues;
    std::vector<std::unique_ptr<std::mutex>> queue_mutexes;
    
    std::mutex pool_mutex;
    std::condition_variable condition;
    std::atomic<bool> stop;
    std::atomic<size_t> next_queue;

public:
    EfficientThreadPool(size_t threads = std::thread::hardware_concurrency()) 
        : stop(false), next_queue(0) {
        if (threads == 0) threads = 4;
        
        task_queues.resize(threads);
        queue_mutexes.reserve(threads);
        for (size_t i = 0; i < threads; ++i) {
            queue_mutexes.push_back(std::unique_ptr<std::mutex>(new std::mutex()));
        }
        
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this, i, threads] {
                while (!stop.load()) {
                    std::function<void()> task;
                    bool found_task = false;
                    
                    {
                        std::unique_lock<std::mutex> lock(*queue_mutexes[i]);
                        if (!task_queues[i].empty()) {
                            task = std::move(task_queues[i].front());
                            task_queues[i].pop();
                            found_task = true;
                        }
                    }
                    
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
                        std::unique_lock<std::mutex> lock(pool_mutex);
                        condition.wait_for(lock, std::chrono::milliseconds(10), 
                            [this] { return stop.load() || hasAnyTasks(); });
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

// ============================================================================
// SECTION 8: PERSISTENT GPU BUFFERS
// ============================================================================

class PersistentGPUBuffers {
private:
    cudaTextureObject_t* depth_textures_buffer;
    cudaTextureObject_t* normal_textures_buffer;
    cudaTextureObject_t* image_textures_buffer;
    int* texture_image_ids_buffer;
    
    int* ref_image_ids_buffer;
    int* all_src_image_ids_buffer;
    int* src_counts_buffer;
    int* src_offsets_buffer;
    int* problem_offsets_buffer;
    int* widths_buffer;
    int* heights_buffer;
    
    PointList* output_points_buffer;
    int* valid_flags_buffer;
    
    size_t max_textures;
    size_t max_problems;
    size_t max_src_images;
    size_t max_pixels;
    
    std::atomic<bool> buffers_allocated;
    mutable std::mutex access_mutex;

    void safeCleanup() {
        std::unique_lock<std::mutex> lock(access_mutex);
        if (!buffers_allocated.load()) return;
        
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
        max_textures = 0;
        max_problems = 0;
        max_src_images = 0;
        max_pixels = 0;
    }
    
    void allocateBuffers(size_t est_max_textures, size_t est_max_problems, 
                        size_t est_max_src_images, size_t est_max_pixels) {
        std::unique_lock<std::mutex> lock(access_mutex);
        
        if (buffers_allocated.load()) return;
        
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
    
    template<typename T>
    T* getBufferSafe(T* buffer, size_t needed, size_t max_size, const char* name) {
        std::unique_lock<std::mutex> lock(access_mutex);
        if (!buffers_allocated.load()) {
            throw std::runtime_error("Buffers not allocated");
        }
        if (needed > max_size) {
            std::ostringstream oss;
            oss << name << " buffer size exceeded: needed " << needed << ", max " << max_size;
            throw std::runtime_error(oss.str());
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

// ============================================================================
// SECTION 9: IMAGE DATA AND OPTIMIZED DATA LOADER
// ============================================================================

struct ImageData {
    Camera camera;
    cv::Mat_<float> depth;
    cv::Mat_<cv::Vec3f> normal;
    cv::Mat image;
    cv::Mat_<float> mask;  // Mask image: 0=masked, 1=valid
    bool has_mask = false;  // Flag indicating if mask was loaded
    std::atomic<bool> valid;
    std::chrono::steady_clock::time_point last_accessed;
    
    ImageData() : has_mask(false), valid(false), last_accessed(std::chrono::steady_clock::now()) {}
};


class OptimizedDataLoader {
private:
    std::string dense_folder;
    std::string img_folder; 
    std::string cam_folder;
    bool geom_consistency;
    
    std::unordered_map<int, std::shared_ptr<ImageData>> cache;
    std::list<int> lru_order;
    std::unordered_map<int, std::list<int>::iterator> lru_map;
    
    size_t max_cache_size;
    mutable std::mutex cache_mutex;
    
    std::unique_ptr<EfficientThreadPool> thread_pool;
    FailureTracker* tracker;
    
    void updateLRU(int image_id) {
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
        
        char buf[512];
        int ret;
        
        // Stage 1: Load Camera
        ret = snprintf(buf, sizeof(buf), "%s/%08d_cam.txt", cam_folder.c_str(), image_id);
        if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
            if (tracker) tracker->recordImageFailure(image_id, "camera", "Path buffer overflow");
            return data;
        }
        
        std::string cam_path(buf);
        std::ifstream cam_check(cam_path);
        if (!cam_check.good()) {
            if (tracker) tracker->recordImageFailure(image_id, "camera", "File not found: " + cam_path);
            return data;
        }
        cam_check.close();
        
        try {
            data->camera = ReadCamera(cam_path);
        } catch (const std::exception& e) {
            if (tracker) tracker->recordImageFailure(image_id, "camera", "Parse error: " + std::string(e.what()));
            return data;
        }
        
        if (data->camera.width <= 0 || data->camera.height <= 0) {
            if (tracker) tracker->recordImageFailure(image_id, "camera", 
                "Invalid dimensions: " + std::to_string(data->camera.width) + "x" + std::to_string(data->camera.height));
            return data;
        }
        
        // Stage 2: Load Depth Map
        std::string depth_suffix = geom_consistency ? "/depths_geom" : "/depths";
        bool depth_loaded = false;
        std::string depth_error;
        
        ret = snprintf(buf, sizeof(buf), "%s/ACMMP/2333_%08d%s.cdmb", 
                      dense_folder.c_str(), image_id, depth_suffix.c_str());
        if (ret >= 0 && ret < static_cast<int>(sizeof(buf))) {
            std::string cdmb_path(buf);
            std::ifstream cdmb_check(cdmb_path);
            if (cdmb_check.good()) {
                cdmb_check.close();
                if (CompressedDMB::readDepthCompressed(cdmb_path, data->depth) == 0) {
                    if (data->depth.cols > 0 && data->depth.rows > 0) {
                        depth_loaded = true;
                    } else {
                        depth_error = "Compressed depth has invalid dimensions";
                    }
                } else {
                    depth_error = "Failed to decompress .cdmb file";
                }
            }
        }
        
        if (!depth_loaded) {
            ret = snprintf(buf, sizeof(buf), "%s/ACMMP/2333_%08d%s.dmb", 
                          dense_folder.c_str(), image_id, depth_suffix.c_str());
            if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
                if (tracker) tracker->recordImageFailure(image_id, "depth", "Path buffer overflow");
                return data;
            }
            
            std::string dmb_path(buf);
            std::ifstream dmb_check(dmb_path);
            if (!dmb_check.good()) {
                std::string msg = "File not found: " + dmb_path;
                if (!depth_error.empty()) msg += " (cdmb error: " + depth_error + ")";
                if (tracker) tracker->recordImageFailure(image_id, "depth", msg);
                return data;
            }
            dmb_check.close();
            
            int read_result = readDepthDmb(dmb_path, data->depth);
            if (read_result != 0) {
                if (tracker) tracker->recordImageFailure(image_id, "depth", 
                    "Read error code: " + std::to_string(read_result));
                return data;
            }
        }
        
        if (data->depth.cols <= 0 || data->depth.rows <= 0) {
            if (tracker) tracker->recordImageFailure(image_id, "depth", 
                "Invalid dimensions: " + std::to_string(data->depth.cols) + "x" + std::to_string(data->depth.rows));
            return data;
        }
        
        // Stage 3: Load Normal Map
        bool normal_loaded = false;
        std::string normal_error;
        
        ret = snprintf(buf, sizeof(buf), "%s/ACMMP/2333_%08d/normals.cdmb", dense_folder.c_str(), image_id);
        if (ret >= 0 && ret < static_cast<int>(sizeof(buf))) {
            std::string cdmb_path(buf);
            std::ifstream cdmb_check(cdmb_path);
            if (cdmb_check.good()) {
                cdmb_check.close();
                if (CompressedDMB::readNormalCompressed(cdmb_path, data->normal) == 0) {
                    if (data->normal.cols > 0 && data->normal.rows > 0) {
                        normal_loaded = true;
                    } else {
                        normal_error = "Compressed normal has invalid dimensions";
                    }
                } else {
                    normal_error = "Failed to decompress normal .cdmb";
                }
            }
        }
        
        if (!normal_loaded) {
            ret = snprintf(buf, sizeof(buf), "%s/ACMMP/2333_%08d/normals.dmb", dense_folder.c_str(), image_id);
            if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
                if (tracker) tracker->recordImageFailure(image_id, "normal", "Path buffer overflow");
                return data;
            }
            
            std::string dmb_path(buf);
            std::ifstream dmb_check(dmb_path);
            if (!dmb_check.good()) {
                std::string msg = "File not found: " + dmb_path;
                if (!normal_error.empty()) msg += " (cdmb error: " + normal_error + ")";
                if (tracker) tracker->recordImageFailure(image_id, "normal", msg);
                return data;
            }
            dmb_check.close();
            
            int read_result = readNormalDmb(dmb_path, data->normal);
            if (read_result != 0) {
                if (tracker) tracker->recordImageFailure(image_id, "normal", 
                    "Read error code: " + std::to_string(read_result));
                return data;
            }
        }
        
        if (data->normal.cols <= 0 || data->normal.rows <= 0) {
            if (tracker) tracker->recordImageFailure(image_id, "normal", 
                "Invalid dimensions: " + std::to_string(data->normal.cols) + "x" + std::to_string(data->normal.rows));
            return data;
        }
        
        if (data->depth.cols != data->normal.cols || data->depth.rows != data->normal.rows) {
            if (tracker) tracker->recordImageFailure(image_id, "normal", 
                "Dimension mismatch with depth: depth=" + std::to_string(data->depth.cols) + "x" + 
                std::to_string(data->depth.rows) + ", normal=" + std::to_string(data->normal.cols) + 
                "x" + std::to_string(data->normal.rows));
            return data;
        }
        
        // Stage 4: Load Image
        ret = snprintf(buf, sizeof(buf), "%s/%08d.png", img_folder.c_str(), image_id);
        if (ret < 0 || ret >= static_cast<int>(sizeof(buf))) {
            if (tracker) tracker->recordImageFailure(image_id, "image", "Path buffer overflow");
            return data;
        }
        
        std::string img_path(buf);
        data->image = cv::imread(img_path, cv::IMREAD_COLOR);
        
        if (data->image.empty()) {
            ret = snprintf(buf, sizeof(buf), "%s/%08d.jpg", img_folder.c_str(), image_id);
            if (ret >= 0 && ret < static_cast<int>(sizeof(buf))) {
                data->image = cv::imread(std::string(buf), cv::IMREAD_COLOR);
            }
            
            if (data->image.empty()) {
                if (tracker) tracker->recordImageFailure(image_id, "image", 
                    "Failed to load (tried .png and .jpg): " + img_path);
                return data;
            }
        }
        
        // Stage 4.5: Load Mask (optional) - mask folder is at same level as images
        std::string mask_folder = dense_folder + "/masks";
        ret = snprintf(buf, sizeof(buf), "%s/%08d.png", mask_folder.c_str(), image_id);
        if (ret >= 0 && ret < static_cast<int>(sizeof(buf))) {
            std::string mask_path(buf);
            cv::Mat mask_img = cv::imread(mask_path, cv::IMREAD_GRAYSCALE);
            if (!mask_img.empty()) {
                mask_img.convertTo(data->mask, CV_32FC1, 1.0 / 255.0);  // Normalize to 0-1
                data->has_mask = true;
            }
        }
        
        // Stage 5: Rescale Image (and Mask)
        try {
            cv::Mat_<cv::Vec3b> img_color;
            if (data->image.channels() == 3) {
                img_color = cv::Mat_<cv::Vec3b>(data->image);
            } else if (data->image.channels() == 1) {
                cv::cvtColor(data->image, img_color, cv::COLOR_GRAY2BGR);
            } else {
                if (tracker) tracker->recordImageFailure(image_id, "image", 
                    "Unexpected channels: " + std::to_string(data->image.channels()));
                return data;
            }
            
            int old_width = img_color.cols;
            int old_height = img_color.rows;
            cv::Mat_<cv::Vec3b> scaled_color;
            RescaleImageAndCamera(img_color, scaled_color, data->depth, data->camera);
            data->image = cv::Mat(scaled_color);
            
            // Rescale mask if present
            if (data->has_mask && !data->mask.empty()) {
                if (scaled_color.cols != old_width || scaled_color.rows != old_height) {
                    cv::Mat scaled_mask;
                    cv::resize(data->mask, scaled_mask, cv::Size(scaled_color.cols, scaled_color.rows), 
                               0, 0, cv::INTER_NEAREST);
                    data->mask = scaled_mask;
                }
            }
            
        } catch (const std::exception& e) {
            if (tracker) tracker->recordImageFailure(image_id, "rescale", 
                "Rescale failed: " + std::string(e.what()));
            return data;
        }

        
        data->valid.store(true);
        data->last_accessed = std::chrono::steady_clock::now();
        
        return data;
    }

public:
    // REDUCED cache size from 200 to 50
    OptimizedDataLoader(const std::string& folder, bool geom = false, size_t cache_size = 50) 
        : dense_folder(folder), geom_consistency(geom), max_cache_size(cache_size), tracker(nullptr) {
        img_folder = folder + "/images";
        cam_folder = folder + "/cams";
        
        unsigned int hw_threads = std::thread::hardware_concurrency();
        size_t io_threads = std::min(static_cast<size_t>(8), std::max(static_cast<size_t>(4), static_cast<size_t>(hw_threads / 2)));
        
        thread_pool.reset(new EfficientThreadPool(io_threads));
        
        std::cout << "[OptimizedLoader] Using " << io_threads << " threads, cache size " << cache_size << std::endl;
    }
    
    void setFailureTracker(FailureTracker* t) {
        tracker = t;
    }
    
    void preloadChunkParallel(const std::vector<int>& image_ids) {
        if (image_ids.empty()) return;
        
        std::vector<std::future<std::shared_ptr<ImageData>>> futures;
        std::vector<int> images_to_load;
        
        std::cout << "  Preloading " << image_ids.size() << " images in parallel..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        {
            std::unique_lock<std::mutex> lock(cache_mutex);
            for (int image_id : image_ids) {
                if (cache.find(image_id) == cache.end()) {
                    images_to_load.push_back(image_id);
                }
            }
        }
        
        for (int image_id : images_to_load) {
            auto future = thread_pool->enqueue([this, image_id]() {
                return loadImageDataSync(image_id);
            });
            futures.push_back(std::move(future));
        }
        
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
        
        if (loaded_count < images_to_load.size()) {
            FusionLogger::warning("DataLoader", 
                "Failed to load " + std::to_string(images_to_load.size() - loaded_count) + " images");
        }
    }
    
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
    
    // Force eviction of specific images
    void evictImages(const std::vector<int>& image_ids) {
        std::unique_lock<std::mutex> lock(cache_mutex);
        for (int id : image_ids) {
            auto cache_it = cache.find(id);
            if (cache_it != cache.end()) {
                cache.erase(cache_it);
            }
            auto lru_it = lru_map.find(id);
            if (lru_it != lru_map.end()) {
                lru_order.erase(lru_it->second);
                lru_map.erase(lru_it);
            }
        }
    }
};

// ============================================================================
// SECTION 10: TEXTURE MANAGER (WITH LIMITED CONCURRENCY)
// ============================================================================

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
            cudaDeviceSynchronize();
            
            if (depth_texture != 0) { cudaDestroyTextureObject(depth_texture); depth_texture = 0; }
            if (normal_texture != 0) { cudaDestroyTextureObject(normal_texture); normal_texture = 0; }
            if (image_texture != 0) { cudaDestroyTextureObject(image_texture); image_texture = 0; }
            
            if (depth_array != nullptr) { cudaFreeArray(depth_array); depth_array = nullptr; }
            if (normal_array != nullptr) { cudaFreeArray(normal_array); normal_array = nullptr; }
            if (image_array != nullptr) { cudaFreeArray(image_array); image_array = nullptr; }
            
            is_valid = false;
        }
    };
    
    std::vector<std::unique_ptr<TextureData>> textures;
    std::vector<int> current_image_ids;
    std::atomic<bool> loaded;
    mutable std::mutex texture_mutex;
    
    static const int num_streams = 4;
    cudaStream_t streams[num_streams];
    
    FailureTracker* tracker;
    
    // CRITICAL: Limit concurrent texture loading to prevent memory spikes
    static const size_t MAX_CONCURRENT_TEXTURE_LOADS = 4;
    
    void release() {
        std::lock_guard<std::mutex> lock(texture_mutex);
        
        if (!loaded.load() && textures.empty()) {
            return;
        }
        
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();
        
        for (auto& tex : textures) {
            if (tex) {
                tex->cleanup();
            }
        }
        
        textures.clear();
        current_image_ids.clear();
        loaded.store(false);
    }
    
    // Single texture load function
    bool loadSingleTexture(size_t idx, int image_id, OptimizedDataLoader& loader, 
                           std::atomic<size_t>& successful_textures) {
        cudaStream_t stream = streams[idx % num_streams];
        
        Camera cam;
        cv::Mat_<float> depth;
        cv::Mat_<cv::Vec3f> normal;
        cv::Mat image;
        
        if (!loader.getData(image_id, cam, depth, normal, image)) {
            if (tracker) tracker->recordImageFailure(image_id, "texture_data", 
                "Failed to retrieve cached data");
            return false;
        }
        
        TextureData& tex = *textures[idx];
        
        try {
            // Create depth texture
            cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc<float>();
            cudaError_t err = cudaMallocArray(&tex.depth_array, &depth_desc, cam.width, cam.height);
            if (err != cudaSuccess) {
                if (tracker) tracker->recordImageFailure(image_id, "texture_depth_alloc", 
                    cudaGetErrorString(err));
                return false;
            }
            
            err = cudaMemcpy2DToArrayAsync(tex.depth_array, 0, 0, 
                                           depth.ptr<float>(), depth.step[0], 
                                           cam.width * sizeof(float), cam.height, 
                                           cudaMemcpyHostToDevice, stream);
            if (err != cudaSuccess) {
                if (tracker) tracker->recordImageFailure(image_id, "texture_depth_copy", 
                    cudaGetErrorString(err));
                tex.cleanup();
                return false;
            }
            
            cudaResourceDesc depth_res_desc = {};
            depth_res_desc.resType = cudaResourceTypeArray;
            depth_res_desc.res.array.array = tex.depth_array;
            
            cudaTextureDesc depth_tex_desc = {};
            depth_tex_desc.addressMode[0] = cudaAddressModeClamp;
            depth_tex_desc.addressMode[1] = cudaAddressModeClamp;
            depth_tex_desc.filterMode = cudaFilterModePoint;
            depth_tex_desc.readMode = cudaReadModeElementType;
            depth_tex_desc.normalizedCoords = false;
            
            err = cudaCreateTextureObject(&tex.depth_texture, &depth_res_desc, &depth_tex_desc, NULL);
            if (err != cudaSuccess) {
                if (tracker) tracker->recordImageFailure(image_id, "texture_depth_create", 
                    cudaGetErrorString(err));
                tex.cleanup();
                return false;
            }
            
            // Create normal texture - use scoped allocation to ensure cleanup
            {
                cv::Mat normal_rgba = cv::Mat(normal.rows, normal.cols, CV_32FC4, 
                                              cv::Scalar(0.0f, 0.0f, 0.0f, 0.0f));
                cv::Mat src_mats[1] = {normal};
                cv::Mat dst_mats[1] = {normal_rgba};
                int from_to[6] = {0, 0, 1, 1, 2, 2};
                cv::mixChannels(src_mats, 1, dst_mats, 1, from_to, 3);
                
                cudaChannelFormatDesc normal_desc = cudaCreateChannelDesc<float4>();
                err = cudaMallocArray(&tex.normal_array, &normal_desc, cam.width, cam.height);
                if (err != cudaSuccess) {
                    if (tracker) tracker->recordImageFailure(image_id, "texture_normal_alloc", 
                        cudaGetErrorString(err));
                    tex.cleanup();
                    return false;
                }
                
                err = cudaMemcpy2DToArrayAsync(tex.normal_array, 0, 0, 
                                               normal_rgba.ptr<float>(), normal_rgba.step[0], 
                                               cam.width * sizeof(float4), cam.height, 
                                               cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    if (tracker) tracker->recordImageFailure(image_id, "texture_normal_copy", 
                        cudaGetErrorString(err));
                    tex.cleanup();
                    return false;
                }
            } // normal_rgba goes out of scope here, freeing memory
            
            cudaResourceDesc normal_res_desc = {};
            normal_res_desc.resType = cudaResourceTypeArray;
            normal_res_desc.res.array.array = tex.normal_array;
            
            cudaTextureDesc normal_tex_desc = {};
            normal_tex_desc.addressMode[0] = cudaAddressModeClamp;
            normal_tex_desc.addressMode[1] = cudaAddressModeClamp;
            normal_tex_desc.filterMode = cudaFilterModePoint;
            normal_tex_desc.readMode = cudaReadModeElementType;
            normal_tex_desc.normalizedCoords = false;
            
            err = cudaCreateTextureObject(&tex.normal_texture, &normal_res_desc, &normal_tex_desc, NULL);
            if (err != cudaSuccess) {
                if (tracker) tracker->recordImageFailure(image_id, "texture_normal_create", 
                    cudaGetErrorString(err));
                tex.cleanup();
                return false;
            }
            
            // Create image texture - use scoped allocation
            {
                cv::Mat rgba, rgba_float;
                cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
                rgba.convertTo(rgba_float, CV_32FC4, 1.0/255.0);
                rgba.release(); // Explicitly release intermediate
                
                cudaChannelFormatDesc image_desc = cudaCreateChannelDesc<float4>();
                err = cudaMallocArray(&tex.image_array, &image_desc, cam.width, cam.height);
                if (err != cudaSuccess) {
                    if (tracker) tracker->recordImageFailure(image_id, "texture_image_alloc", 
                        cudaGetErrorString(err));
                    tex.cleanup();
                    return false;
                }
                
                err = cudaMemcpy2DToArrayAsync(tex.image_array, 0, 0, 
                                               rgba_float.ptr<float>(), rgba_float.step[0], 
                                               cam.width * sizeof(float4), cam.height, 
                                               cudaMemcpyHostToDevice, stream);
                if (err != cudaSuccess) {
                    if (tracker) tracker->recordImageFailure(image_id, "texture_image_copy", 
                        cudaGetErrorString(err));
                    tex.cleanup();
                    return false;
                }
            } // rgba_float goes out of scope here
            
            cudaResourceDesc image_res_desc = {};
            image_res_desc.resType = cudaResourceTypeArray;
            image_res_desc.res.array.array = tex.image_array;
            
            cudaTextureDesc image_tex_desc = {};
            image_tex_desc.addressMode[0] = cudaAddressModeClamp;
            image_tex_desc.addressMode[1] = cudaAddressModeClamp;
            image_tex_desc.filterMode = cudaFilterModeLinear;
            image_tex_desc.readMode = cudaReadModeElementType;
            image_tex_desc.normalizedCoords = false;
            
            err = cudaCreateTextureObject(&tex.image_texture, &image_res_desc, &image_tex_desc, NULL);
            if (err != cudaSuccess) {
                if (tracker) tracker->recordImageFailure(image_id, "texture_image_create", 
                    cudaGetErrorString(err));
                tex.cleanup();
                return false;
            }
            
            err = cudaStreamSynchronize(stream);
            if (err != cudaSuccess) {
                if (tracker) tracker->recordImageFailure(image_id, "texture_sync", 
                    cudaGetErrorString(err));
                tex.cleanup();
                return false;
            }
            
            tex.is_valid = true;
            successful_textures.fetch_add(1);
            return true;
            
        } catch (const std::exception& e) {
            if (tracker) tracker->recordImageFailure(image_id, "texture_exception", e.what());
            tex.cleanup();
            return false;
        } catch (...) {
            if (tracker) tracker->recordImageFailure(image_id, "texture_exception", "Unknown exception");
            tex.cleanup();
            return false;
        }
    }

public:
    OptimizedTextureManager() : loaded(false), tracker(nullptr) {
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
    
    void setFailureTracker(FailureTracker* t) {
        tracker = t;
    }
    
    bool loadChunk(const std::vector<int>& image_ids, OptimizedDataLoader& loader, size_t chunk_idx = 0) {
        release();
        
        if (image_ids.empty()) {
            if (tracker) tracker->recordChunkFailure(chunk_idx, "texture_init", "Empty image list");
            return false;
        }
        
        std::lock_guard<std::mutex> lock(texture_mutex);
        
        current_image_ids = image_ids;
        textures.clear();
        textures.reserve(image_ids.size());
        
        for (size_t i = 0; i < image_ids.size(); ++i) {
            textures.push_back(std::unique_ptr<TextureData>(new TextureData()));
        }
        
        std::atomic<size_t> successful_textures(0);
        
        // CRITICAL FIX: Process in batches to limit memory usage
        for (size_t batch_start = 0; batch_start < image_ids.size(); batch_start += MAX_CONCURRENT_TEXTURE_LOADS) {
            size_t batch_end = std::min(batch_start + MAX_CONCURRENT_TEXTURE_LOADS, image_ids.size());
            
            std::vector<std::future<bool>> batch_futures;
            
            for (size_t i = batch_start; i < batch_end; ++i) {
                auto future = std::async(std::launch::async, 
                    [this, i, &image_ids, &loader, &successful_textures]() {
                        return loadSingleTexture(i, image_ids[i], loader, successful_textures);
                    });
                batch_futures.push_back(std::move(future));
            }
            
            // Wait for this batch to complete before starting next
            for (auto& future : batch_futures) {
                future.get();
            }
            
            // Sync streams after each batch
            for (int s = 0; s < num_streams; ++s) {
                cudaStreamSynchronize(streams[s]);
            }
        }
        
        size_t final_count = successful_textures.load();
        size_t failed_count = image_ids.size() - final_count;
        
        std::cout << "    Successfully loaded " << final_count << "/" << image_ids.size() << " textures" << std::endl;
        
        if (failed_count > 0) {
            FusionLogger::warning("TextureManager", 
                "Chunk " + std::to_string(chunk_idx) + ": " + std::to_string(failed_count) + 
                "/" + std::to_string(image_ids.size()) + " textures failed");
        }
        
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

// ============================================================================
// SECTION 11: CUDA KERNEL
// ============================================================================

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
    int max_image_id,
    float pole_exclusion_degrees = 10.0f
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    int problem_id = find_problem_id(global_idx, problem_offsets, num_problems_in_chunk);
    
    if (problem_id < 0 || problem_id >= num_problems_in_chunk) return;
    
    int local_idx = global_idx - problem_offsets[problem_id];
    int width = widths[problem_id];
    int height = heights[problem_id];
    
    if (local_idx >= width * height) return;
    
    int c = local_idx % width;
    int r = local_idx / width;
    
    int ref_image_id = ref_image_ids[problem_id];
    
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
    
    if (IsNearPole(ref_cam, c, r, pole_exclusion_degrees)) {
        valid_flags[global_idx] = 0;
        return;
    }
    
    float ref_depth = tex2D<float>(depth_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    
    if (ref_depth <= 0.0f) {
        valid_flags[global_idx] = 0;
        return;
    }

    float3 PointX = Get3DPointonWorld_cu(static_cast<float>(c), static_cast<float>(r), ref_depth, ref_cam);
    
    float4 ref_normal_tex = tex2D<float4>(normal_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    float3 ref_normal = make_float3(ref_normal_tex.x, ref_normal_tex.y, ref_normal_tex.z);
    
    float4 ref_color = tex2D<float4>(image_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    
    const float black_threshold = 0.01f;
    if (ref_color.x < black_threshold && 
        ref_color.y < black_threshold && 
        ref_color.z < black_threshold) {
        valid_flags[global_idx] = 0;
        return;
    }
    
    float3 point_sum = PointX;
    float3 normal_sum = ref_normal;
    float color_sum[3] = {
        ref_color.z * 255.0f,
        ref_color.y * 255.0f,
        ref_color.x * 255.0f
    };
    int num_consistent = 1;
    float confidence_sum = 1.0f;
    
    int src_start = src_offsets[problem_id];
    int src_count = src_counts[problem_id];
    
    for (int j = 0; j < src_count; ++j) {
        int src_image_id = src_image_ids[src_start + j];
        
        if (src_image_id < 0 || src_image_id > max_image_id) continue;
        
        int src_cam_idx = image_to_camera_map[src_image_id];
        int src_tex_idx = image_to_texture_map[src_image_id];
        
        if (src_cam_idx < 0 || src_tex_idx < 0 || src_cam_idx >= num_cameras || src_tex_idx >= num_textures) continue;
        
        const Camera& src_cam = cameras[src_cam_idx];
        
        float2 proj_point;
        float proj_depth_in_src;
        ProjectonCamera_cu(PointX, src_cam, proj_point, proj_depth_in_src);
        
        int src_c = static_cast<int>(proj_point.x + 0.5f);
        int src_r = static_cast<int>(proj_point.y + 0.5f);

        // Wrap x-coordinate for spherical source cameras
        if (src_cam.model == SPHERE) {
            src_c = ((src_c % src_cam.width) + src_cam.width) % src_cam.width;
        }

        if (src_c < 0 || src_c >= src_cam.width || src_r < 0 || src_r >= src_cam.height)
            continue;
        
        if (IsNearPole(src_cam, src_c, src_r, pole_exclusion_degrees)) {
            continue;
        }
        
        float src_depth = tex2D<float>(depth_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
        if (src_depth <= 0.0f) continue;
        
        float3 PointX_src = Get3DPointonWorld_cu(static_cast<float>(src_c), static_cast<float>(src_r), src_depth, src_cam);
        
        float2 reproj_point_in_ref;
        float dummy_depth;
        ProjectonCamera_cu(PointX_src, ref_cam, reproj_point_in_ref, dummy_depth);
        
        float diff_x = c - reproj_point_in_ref.x;
        if (ref_cam.model == SPHERE) {
            const float w = static_cast<float>(ref_cam.width);
            if (diff_x > w * 0.5f) diff_x -= w;
            else if (diff_x < -w * 0.5f) diff_x += w;
        }
        float reproj_error = hypotf(diff_x, r - reproj_point_in_ref.y);
        float relative_depth_diff = fabsf(proj_depth_in_src - src_depth) / src_depth;
        
        float4 src_normal_tex = tex2D<float4>(normal_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
        float3 src_normal = make_float3(src_normal_tex.x, src_normal_tex.y, src_normal_tex.z);
        
        float dot_product = ref_normal.x * src_normal.x + ref_normal.y * src_normal.y + ref_normal.z * src_normal.z;
        dot_product = fmaxf(-1.0f, fminf(1.0f, dot_product));
        float angle = acosf(dot_product);

        float adaptive_reproj_threshold = GetAdaptiveReprojectionThreshold(ref_cam, c, r, 2.0f);
        float adaptive_reproj_threshold_src = GetAdaptiveReprojectionThreshold(src_cam, src_c, src_r, 3.0f);
        float combined_threshold = fmaxf(adaptive_reproj_threshold, adaptive_reproj_threshold_src);        

        float adaptive_depth_threshold = fmaxf(
            GetAdaptiveDepthThreshold(src_depth, src_cam, 0.03f),
            GetAdaptiveDepthThreshold(ref_depth, ref_cam, 0.03f)
        );
        
        float adaptive_normal_threshold = 0.3f;
        if (IsNearPole(ref_cam, c, r, 20.0f)) {
            adaptive_normal_threshold = 0.45f;
        }
        
        if (reproj_error < combined_threshold && 
            relative_depth_diff < adaptive_depth_threshold && 
            angle < adaptive_normal_threshold) {
            
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

// ============================================================================
// SECTION 12: CHUNKING STRATEGY
// ============================================================================

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
        
        std::unordered_set<int> combined_images = current_images;
        combined_images.insert(problem_images.begin(), problem_images.end());
        
        if (combined_images.size() > max_images_per_chunk && !current_chunk.empty()) {
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_images.clear();
        }
        
        current_chunk.push_back(i);
        current_images.insert(problem_images.begin(), problem_images.end());
    }
    
    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }
    
    return chunks;
}

// ============================================================================
// SECTION 13: MAIN FUSION FUNCTION
// ============================================================================

void RunFusionCuda(const std::string &dense_folder,
                   const std::vector<Problem> &problems,
                   bool geom_consistency,
                   size_t max_images_per_chunk)
{
    FusionLogger::info("Fusion", "Starting with " + std::to_string(problems.size()) + " problems...");
    MemoryMonitor::logUsage("Start");
    
    FailureTracker tracker;
    
    if (problems.empty()) {
        FusionLogger::error("Fusion", "No problems to process");
        return;
    }
    
    if (max_images_per_chunk == 0) {
        max_images_per_chunk = 50;
    }
    
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
            chunk_pixels += 3200 * 1600;
        }
        
        est_max_src_images = std::max(est_max_src_images, chunk_src_images);
        est_max_pixels = std::max(est_max_pixels, chunk_pixels);
    }
    
    FusionLogger::info("Fusion", "Created " + std::to_string(chunks.size()) + " chunks");
    FusionLogger::info("Fusion", "Estimated max: " + std::to_string(est_max_textures) + " textures, " 
                      + std::to_string(est_max_problems) + " problems, " 
                      + std::to_string(est_max_pixels) + " pixels");
    
    // REDUCED cache size from 200 to 50
    OptimizedDataLoader loader(dense_folder, geom_consistency, 50);
    loader.setFailureTracker(&tracker);
    
    OptimizedTextureManager texture_manager;
    texture_manager.setFailureTracker(&tracker);
    
    PersistentGPUBuffers gpu_buffers;
    ImageLookupTables lookup_tables;
    
    CudaMemoryGuard<Camera> cameras_cuda_guard("cameras");
    CudaMemoryGuard<int> camera_image_ids_cuda_guard("camera_ids");
    
    // Initialize streaming point writer
    std::unique_ptr<StreamingPointWriter> point_writer;
    
    try {
        point_writer.reset(new StreamingPointWriter(dense_folder));
        
        gpu_buffers.allocateBuffers(est_max_textures, est_max_problems, est_max_src_images, est_max_pixels);
        
        std::unordered_set<int> all_image_ids;
        for (const auto& problem : problems) {
            all_image_ids.insert(problem.ref_image_id);
            for (int src_id : problem.src_image_ids) {
                all_image_ids.insert(src_id);
            }
        }
        
        std::vector<Camera> all_cameras;
        std::vector<int> camera_image_ids;
        
        FusionLogger::info("CameraLoad", "Loading " + std::to_string(all_image_ids.size()) + " cameras...");
        
        for (int image_id : all_image_ids) {
            try {
                char cam_buf[512];
                int ret = snprintf(cam_buf, sizeof(cam_buf), "%s/cams/%08d_cam.txt", 
                                dense_folder.c_str(), image_id);
                if (ret < 0 || ret >= static_cast<int>(sizeof(cam_buf))) {
                    tracker.recordImageFailure(image_id, "camera_path", "Path buffer overflow");
                    continue;
                }
                
                Camera cam = ReadCamera(std::string(cam_buf));
                if (cam.width <= 0 || cam.height <= 0) {
                    tracker.recordImageFailure(image_id, "camera_dims", 
                        "Invalid dimensions: " + std::to_string(cam.width) + "x" + std::to_string(cam.height));
                    continue;
                }
                
                std::string depth_suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
                char depth_buf[512];
                ret = snprintf(depth_buf, sizeof(depth_buf), "%s/ACMMP/2333_%08d%s", 
                            dense_folder.c_str(), image_id, depth_suffix.c_str());
                if (ret < 0 || ret >= static_cast<int>(sizeof(depth_buf))) continue;
                
                cv::Mat_<float> depth;
                if (readDepthDmb(std::string(depth_buf), depth) != 0) continue;
                if (depth.cols <= 0 || depth.rows <= 0) continue;
                
                char img_buf[512];
                ret = snprintf(img_buf, sizeof(img_buf), "%s/images/%08d.png", 
                            dense_folder.c_str(), image_id);
                if (ret < 0 || ret >= static_cast<int>(sizeof(img_buf))) continue;
                
                cv::Mat image = cv::imread(std::string(img_buf), cv::IMREAD_COLOR);
                if (image.empty()) continue;
                
                float scale_x = (float)depth.cols / (float)image.cols;
                float scale_y = (float)depth.rows / (float)image.rows;
                
                cam.width = depth.cols;
                cam.height = depth.rows;
                
                if (cam.model == SPHERE) {
                    cam.params[1] *= scale_x;
                    cam.params[2] *= scale_y;
                } else {
                    cam.K[0] *= scale_x;
                    cam.K[2] *= scale_x;
                    cam.K[4] *= scale_y;
                    cam.K[5] *= scale_y;
                }
                
                all_cameras.push_back(cam);
                camera_image_ids.push_back(image_id);
                
            } catch (const std::exception& e) {
                tracker.recordImageFailure(image_id, "camera_exception", e.what());
                continue;
            }
        }
        
        FusionLogger::info("CameraLoad", "Loaded " + std::to_string(all_cameras.size()) + 
                          "/" + std::to_string(all_image_ids.size()) + " cameras successfully");
        
        MemoryMonitor::logUsage("After camera load");
        
        if (all_cameras.empty()) {
            FusionLogger::error("Fusion", "No valid cameras found - cannot proceed");
            tracker.printSummary();
            return;
        }
        
        cameras_cuda_guard.alloc(all_cameras.size());
        camera_image_ids_cuda_guard.alloc(camera_image_ids.size());
        
        CUDA_SAFE_CALL(cudaMemcpy(cameras_cuda_guard.get(), all_cameras.data(), 
                                   all_cameras.size() * sizeof(Camera), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(camera_image_ids_cuda_guard.get(), camera_image_ids.data(), 
                                   camera_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        
        auto total_start = std::chrono::high_resolution_clock::now();
        
        for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
            const auto& chunk = chunks[chunk_idx];
            
            FusionLogger::info("Chunk", "Processing chunk " + std::to_string(chunk_idx + 1) + 
                              "/" + std::to_string(chunks.size()) + 
                              " (" + std::to_string(chunk.size()) + " problems)");
            
            MemoryMonitor::logUsage("Chunk " + std::to_string(chunk_idx) + " start");
            
            std::unordered_set<int> chunk_images;
            for (size_t prob_idx : chunk) {
                if (prob_idx >= problems.size()) {
                    FusionLogger::warning("Chunk", "Invalid problem index: " + std::to_string(prob_idx));
                    continue;
                }
                const Problem& problem = problems[prob_idx];
                chunk_images.insert(problem.ref_image_id);
                for (int src_id : problem.src_image_ids) {
                    chunk_images.insert(src_id);
                }
            }
            
            std::vector<int> chunk_image_ids(chunk_images.begin(), chunk_images.end());
            FusionLogger::info("Chunk", "  Unique images: " + std::to_string(chunk_image_ids.size()));
            
            loader.preloadChunkParallel(chunk_image_ids);
            
            MemoryMonitor::logUsage("Chunk " + std::to_string(chunk_idx) + " after preload");
            
            if (!texture_manager.loadChunk(chunk_image_ids, loader, chunk_idx)) {
                tracker.recordChunkFailure(chunk_idx, "texture_load", 
                    "No textures successfully loaded", chunk_image_ids, chunk.size());
                continue;
            }
            
            MemoryMonitor::logUsage("Chunk " + std::to_string(chunk_idx) + " after texture load");
            
            const auto& texture_image_ids = texture_manager.getValidImageIds();
            if (texture_image_ids.empty()) {
                tracker.recordChunkFailure(chunk_idx, "lookup_tables", 
                    "No valid texture image IDs", chunk_image_ids, chunk.size());
                continue;
            }
            
            lookup_tables.buildTables(camera_image_ids, texture_image_ids);
            
            std::vector<int> ref_image_ids;
            std::vector<int> all_src_image_ids;
            std::vector<int> src_counts;
            std::vector<int> src_offsets;
            std::vector<int> problem_offsets;
            std::vector<int> widths;
            std::vector<int> heights;
            
            int total_pixels = 0;
            int src_offset = 0;
            int skipped_problems = 0;
            
            for (size_t i = 0; i < chunk.size(); ++i) {
                size_t prob_idx = chunk[i];
                if (prob_idx >= problems.size()) continue;
                
                const Problem& problem = problems[prob_idx];
                
                Camera ref_cam;
                cv::Mat_<float> depth;
                cv::Mat_<cv::Vec3f> normal;
                cv::Mat image;
                
                if (!loader.getData(problem.ref_image_id, ref_cam, depth, normal, image)) {
                    FusionLogger::warning("Chunk", "  Problem " + std::to_string(prob_idx) + 
                        ": Failed to get data for ref image " + std::to_string(problem.ref_image_id));
                    skipped_problems++;
                    continue;
                }
                
                ref_image_ids.push_back(problem.ref_image_id);
                widths.push_back(ref_cam.width);
                heights.push_back(ref_cam.height);
                problem_offsets.push_back(total_pixels);
                
                size_t problem_pixels;
                if (!safe_multiply(static_cast<size_t>(ref_cam.width), 
                                   static_cast<size_t>(ref_cam.height), problem_pixels)) {
                    FusionLogger::warning("Chunk", "  Problem " + std::to_string(prob_idx) + 
                        ": Pixel count overflow");
                    skipped_problems++;
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
            
            if (skipped_problems > 0) {
                FusionLogger::warning("Chunk", "  Skipped " + std::to_string(skipped_problems) + 
                    "/" + std::to_string(chunk.size()) + " problems due to data issues");
            }
            
            if (total_pixels == 0) {
                tracker.recordChunkFailure(chunk_idx, "no_pixels", 
                    "No valid pixels to process", chunk_image_ids, chunk.size());
                continue;
            }
            
            const auto& depth_textures = texture_manager.getDepthTextures();
            const auto& normal_textures = texture_manager.getNormalTextures();
            const auto& image_textures = texture_manager.getImageTextures();
            
            if (depth_textures.empty() || normal_textures.empty() || image_textures.empty()) {
                tracker.recordChunkFailure(chunk_idx, "no_textures", 
                    "depth=" + std::to_string(depth_textures.size()) + 
                    " normal=" + std::to_string(normal_textures.size()) + 
                    " image=" + std::to_string(image_textures.size()),
                    chunk_image_ids, chunk.size());
                continue;
            }
            
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
            
            {
                CudaStreamGuard copy_stream;
                if (!copy_stream.isValid()) {
                    tracker.recordChunkFailure(chunk_idx, "stream_create", 
                        "Failed to create CUDA stream", chunk_image_ids, chunk.size());
                    continue;
                }
                
                CUDA_SAFE_CALL(cudaMemcpyAsync(depth_textures_cuda, depth_textures.data(), 
                                              depth_textures.size() * sizeof(cudaTextureObject_t), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(normal_textures_cuda, normal_textures.data(), 
                                              normal_textures.size() * sizeof(cudaTextureObject_t), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(image_textures_cuda, image_textures.data(), 
                                              image_textures.size() * sizeof(cudaTextureObject_t), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(texture_image_ids_cuda, texture_image_ids.data(), 
                                              texture_image_ids.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                
                CUDA_SAFE_CALL(cudaMemcpyAsync(ref_image_ids_cuda, ref_image_ids.data(), 
                                              ref_image_ids.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(all_src_image_ids_cuda, all_src_image_ids.data(), 
                                              all_src_image_ids.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(src_counts_cuda, src_counts.data(), 
                                              src_counts.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(src_offsets_cuda, src_offsets.data(), 
                                              src_offsets.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(problem_offsets_cuda, problem_offsets.data(), 
                                              problem_offsets.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(widths_cuda, widths.data(), 
                                              widths.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemcpyAsync(heights_cuda, heights.data(), 
                                              heights.size() * sizeof(int), 
                                              cudaMemcpyHostToDevice, copy_stream.get()));
                CUDA_SAFE_CALL(cudaMemsetAsync(valid_flags_cuda, 0, total_pixels * sizeof(int), copy_stream.get()));
                
                copy_stream.synchronize();
            }
            
            int block_size = 256;
            int grid_size = (total_pixels + block_size - 1) / block_size;
            
            auto chunk_start = std::chrono::high_resolution_clock::now();
            
            float pole_exclusion_degrees = 10.0f;
            
            CorrectedChunkBatchKernel<<<grid_size, block_size>>>(
                depth_textures_cuda,
                normal_textures_cuda,
                image_textures_cuda,
                texture_image_ids_cuda,
                (int)texture_image_ids.size(),
                cameras_cuda_guard.get(),
                camera_image_ids_cuda_guard.get(),
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
                lookup_tables.max_image_id,
                pole_exclusion_degrees
            );
            
            cudaError_t kernel_err = cudaGetLastError();
            if (kernel_err != cudaSuccess) {
                tracker.recordChunkFailure(chunk_idx, "kernel_launch", 
                    std::string("Kernel launch failed: ") + cudaGetErrorString(kernel_err),
                    chunk_image_ids, chunk.size());
                continue;
            }
            
            kernel_err = cudaDeviceSynchronize();
            if (kernel_err != cudaSuccess) {
                tracker.recordChunkFailure(chunk_idx, "kernel_sync", 
                    std::string("Kernel sync failed: ") + cudaGetErrorString(kernel_err),
                    chunk_image_ids, chunk.size());
                continue;
            }
            
            auto chunk_end = std::chrono::high_resolution_clock::now();
            auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(chunk_end - chunk_start);
            
            // Copy results and stream to disk immediately
            std::vector<PointList> chunk_points(total_pixels);
            std::vector<int> valid_flags_host(total_pixels);
            
            CUDA_SAFE_CALL(cudaMemcpy(chunk_points.data(), output_points_cuda, 
                                      total_pixels * sizeof(PointList), cudaMemcpyDeviceToHost));
            CUDA_SAFE_CALL(cudaMemcpy(valid_flags_host.data(), valid_flags_cuda, 
                                      total_pixels * sizeof(int), cudaMemcpyDeviceToHost));
            
            // Stream points to disk instead of accumulating in memory
            size_t chunk_valid_count = 0;
            for (int i = 0; i < total_pixels; ++i) {
                if (valid_flags_host[i]) chunk_valid_count++;
            }
            
            point_writer->writePoints(chunk_points, valid_flags_host);
            
            // CRITICAL: Force deallocation of chunk vectors
            std::vector<PointList>().swap(chunk_points);
            std::vector<int>().swap(valid_flags_host);
            
            float valid_ratio = (float)chunk_valid_count / total_pixels * 100.0f;
            
            FusionLogger::info("Chunk", "  Result: " + std::to_string(chunk_valid_count) + 
                " points (" + std::to_string(valid_ratio).substr(0, 5) + "%) in " + 
                std::to_string(chunk_duration.count()) + " ms");
            
            if (valid_ratio < 1.0f) {
                FusionLogger::warning("Chunk", "  Very low valid point ratio - possible data issue");
            }
            
            MemoryMonitor::logUsage("Chunk " + std::to_string(chunk_idx) + " end");
        }
        
        auto total_end = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
        
        size_t total_points = point_writer->getPointCount();
        FusionLogger::info("Fusion", "Generated " + std::to_string(total_points) + 
                          " points total in " + std::to_string(total_duration.count()) + " seconds");
        
        // Finalize output
        std::string output_path = dense_folder + "/ACMMP/ACMM_model.ply";
        point_writer->finalize(output_path);
        
        FusionLogger::info("Fusion", "Output written to: " + output_path);
        FusionLogger::info("Fusion", "Final cache size: " + std::to_string(loader.getCacheSize()) + " images");
        
        MemoryMonitor::logUsage("Final");
        tracker.printSummary();
        
    } catch (const std::exception& e) {
        FusionLogger::error("Fusion", std::string("Fatal error: ") + e.what());
        MemoryMonitor::logUsage("Error state");
        tracker.printSummary();
        throw;
    }
}