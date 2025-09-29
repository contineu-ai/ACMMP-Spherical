#ifndef ASYNC_IMAGE_LOADER_H
#define ASYNC_IMAGE_LOADER_H

#include <opencv2/opencv.hpp>
#include <deque>
#include <list>
#include <future>
#include <atomic>
#include <condition_variable>
#include <unordered_map>
#include <memory>
#include <thread>
#include <iostream>
#include <iomanip>
#include <string>
#include <vector>
#include <algorithm> // For std::min


int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);

Camera ReadCamera(const std::string &cam_path);


// Data structure to hold preloaded problem data
struct PreloadedProblemData {
    int problem_idx;
    std::vector<cv::Mat> images_float;
    std::vector<Camera> cameras;
    std::vector<cv::Mat> depths;
    bool ready = false;
    std::exception_ptr exception = nullptr;
};

// Main async loader class - Corrected and Optimized
class AsyncImageLoader {
private:
    // Holds the loaded data and an iterator to its position in the LRU list
    struct CacheEntry {
        std::shared_ptr<PreloadedProblemData> data;
        std::list<int>::iterator lru_iterator;
    };

    // Configuration
    const std::string dense_folder_;
    const std::vector<Problem>& all_problems_;
    const bool geom_consistency_;
    const bool multi_geometry_;
    const size_t max_preload_queue_size_;
    const size_t num_loader_threads_;

    // Thread management
    std::vector<std::thread> loader_threads_;
    std::atomic<bool> stopping_{false};

    // Queue management
    std::deque<int> load_queue_;
    mutable std::mutex load_queue_mutex_;
    std::condition_variable load_queue_cv_;

    // Preloaded data storage & LRU cache
    mutable std::mutex cache_mutex_;
    std::unordered_map<int, CacheEntry> cache_;
    std::list<int> lru_list_; // Used for O(1) LRU updates
    std::condition_variable preloaded_cv_;

    // Memory management
    std::atomic<size_t> current_memory_usage_{0};
    const size_t max_memory_usage_;

    // Statistics
    std::atomic<int> total_loaded_{0};
    std::atomic<int> cache_hits_{0};
    std::atomic<int> cache_misses_{0};

public:
    AsyncImageLoader(const std::string& dense_folder,
                       const std::vector<Problem>& problems,
                       bool geom_consistency,
                       bool multi_geometry,
                       size_t max_preload_queue_size = 20,
                       size_t num_loader_threads = 4,
                       size_t max_memory_gb = 8)
        : dense_folder_(dense_folder),
          all_problems_(problems),
          geom_consistency_(geom_consistency),
          multi_geometry_(multi_geometry),
          max_preload_queue_size_(max_preload_queue_size),
          num_loader_threads_(std::min(num_loader_threads, static_cast<size_t>(std::thread::hardware_concurrency()))),
          max_memory_usage_(max_memory_gb * 1024ULL * 1024ULL * 1024ULL) {

        for (size_t i = 0; i < num_loader_threads_; ++i) {
            loader_threads_.emplace_back(&AsyncImageLoader::loaderThreadFunc, this);
        }
        preloadInitialBatch();
    }

    ~AsyncImageLoader() {
        stopping_.store(true);
        load_queue_cv_.notify_all();
        preloaded_cv_.notify_all();

        for (auto& t : loader_threads_) {
            if (t.joinable()) {
                t.join();
            }
        }
    }

    // Main interface to get data, blocks if not ready
    std::shared_ptr<PreloadedProblemData> getProblemData(int problem_idx) {
        std::unique_lock<std::mutex> lock(cache_mutex_);

        while (true) {
            auto it = cache_.find(problem_idx);

            // Case 1: Cache Hit (data is ready)
            if (it != cache_.end() && it->second.data->ready) {
                cache_hits_++;
                
                // Update LRU
                lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
                
                auto data = it->second.data;
                lock.unlock();
                
                // Trigger preload for next items
                triggerPreloadNext(problem_idx);
                
                if (data->exception) {
                    std::rethrow_exception(data->exception);
                }
                return data;
            }

            // Case 2: Being loaded (placeholder exists)
            if (it != cache_.end() && !it->second.data->ready) {
                // Wait for loading to complete
                preloaded_cv_.wait(lock, [this, problem_idx] {
                    if (stopping_.load()) return true;
                    auto find_it = cache_.find(problem_idx);
                    return find_it != cache_.end() && find_it->second.data->ready;
                });
                
                if (stopping_.load()) {
                    throw std::runtime_error("Loader is stopping while waiting for data.");
                }
                continue;
            }

            // Case 3: Cache Miss - need to load
            cache_misses_++;
            lock.unlock();

            // Add to queue for urgent loading
            {
                std::lock_guard<std::mutex> q_lock(load_queue_mutex_);
                // Check if already in queue
                auto queue_it = std::find(load_queue_.begin(), load_queue_.end(), problem_idx);
                if (queue_it == load_queue_.end()) {
                    load_queue_.push_front(problem_idx);
                }
            }
            load_queue_cv_.notify_one();

            // Re-lock and wait
            lock.lock();
        }
    }

    // Print cache and memory statistics
    void printStats() const {
        double hit_rate = (cache_hits_.load() + cache_misses_.load() == 0) ? 0.0 :
                          (100.0 * cache_hits_.load()) / (cache_hits_.load() + cache_misses_.load());
        
        size_t q_size;
        {
            std::lock_guard<std::mutex> q_lock(load_queue_mutex_);
            q_size = load_queue_.size();
        }

        std::cout << "[AsyncLoader Stats] Loaded: " << total_loaded_.load()
                  << ", Hits: " << cache_hits_.load()
                  << ", Misses: " << cache_misses_.load()
                  << ", Hit Rate: " << std::fixed << std::setprecision(2) << hit_rate << "%"
                  << ", Memory: " << (current_memory_usage_.load() / (1024*1024)) << "MB / "
                  << (max_memory_usage_ / (1024*1024)) << "MB"
                  << ", Load Q Size: " << q_size
                  << std::endl;
    }

private:
    // The main function for each loader thread
// Fixed loader thread function
    void loaderThreadFunc() {
        while (!stopping_.load()) {
            int problem_idx = -1;
            {
                std::unique_lock<std::mutex> lock(load_queue_mutex_);
                load_queue_cv_.wait(lock, [this] {
                    return !load_queue_.empty() || stopping_.load();
                });

                if (stopping_.load()) break;
                if (load_queue_.empty()) continue;

                problem_idx = load_queue_.front();
                load_queue_.pop_front();
            }

            // Check if already loading/loaded
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = cache_.find(problem_idx);
                if (it != cache_.end() && it->second.data->ready) {
                    continue; // Already loaded
                }
                if (it == cache_.end()) {
                    // Insert placeholder only if not in cache
                    auto placeholder_data = std::make_shared<PreloadedProblemData>();
                    placeholder_data->ready = false;
                    placeholder_data->problem_idx = problem_idx;
                    lru_list_.push_front(problem_idx);
                    cache_[problem_idx] = {placeholder_data, lru_list_.begin()};
                }
            }

            // Load data from disk
            std::shared_ptr<PreloadedProblemData> data;
            try {
                data = loadProblemData(problem_idx);
            } catch (...) {
                data = std::make_shared<PreloadedProblemData>();
                data->problem_idx = problem_idx;
                data->ready = true;
                data->exception = std::current_exception();
            }

            // Replace placeholder with actual data
            {
                std::lock_guard<std::mutex> lock(cache_mutex_);
                auto it = cache_.find(problem_idx);
                if (it != cache_.end()) {
                    // Calculate size difference for memory tracking
                    size_t old_size = estimateDataSize(it->second.data);
                    size_t new_size = estimateDataSize(data);
                    
                    // Update the data
                    it->second.data = data;
                    
                    // Update memory usage (subtract old, add new)
                    if (current_memory_usage_ >= old_size) {
                        current_memory_usage_ -= old_size;
                    } else {
                        current_memory_usage_ = 0;
                    }
                    current_memory_usage_ += new_size;
                    
                    // Evict if necessary
                    while (current_memory_usage_ > max_memory_usage_ && lru_list_.size() > 1) {
                        evictOldest();
                    }
                }
            }

            total_loaded_++;
            preloaded_cv_.notify_all();
        }
    }

    // Fixed eviction to ensure no orphaned LRU entries
    void evictOldest() {
        // Must be called with cache_mutex_ locked
        while (!lru_list_.empty()) {
            int oldest_idx = lru_list_.back();
            auto it = cache_.find(oldest_idx);
            
            // Check if this entry is ready (not a placeholder being loaded)
            if (it != cache_.end() && it->second.data->ready) {
                // Safe to evict
                size_t data_size = estimateDataSize(it->second.data);
                if (current_memory_usage_ >= data_size) {
                    current_memory_usage_ -= data_size;
                } else {
                    current_memory_usage_ = 0;
                }
                
                lru_list_.pop_back();
                cache_.erase(it);
                break;
            } else {
                // Skip placeholders or missing entries
                lru_list_.pop_back();
                if (it != cache_.end()) {
                    // Remove orphaned cache entry
                    cache_.erase(it);
                }
            }
        }
    }
    // Moves an accessed item to the front of the LRU list in O(1)
    void updateLRU(std::unordered_map<int, CacheEntry>::iterator& it) {
        // Must be called with cache_mutex_ locked
        lru_list_.splice(lru_list_.begin(), lru_list_, it->second.lru_iterator);
    }
    
    // Queues the initial batch of problems to load
    void preloadInitialBatch() {
        std::vector<int> initial_problems;
        for (size_t i = 0; i < std::min(max_preload_queue_size_, all_problems_.size()); ++i) {
            initial_problems.push_back(i);
        }
        preloadProblems(initial_problems);
    }

    // Predictively queues up subsequent problems for loading
    void triggerPreloadNext(int current_idx) {
        std::vector<int> next_problems_to_queue;
        {
            std::lock_guard<std::mutex> lock(load_queue_mutex_);
            
            if (load_queue_.size() >= max_preload_queue_size_) {
                return; // Don't add more if queue is already full
            }
            
            for (size_t i = 1; i <= max_preload_queue_size_ && (current_idx + i < all_problems_.size()); ++i) {
                int next_idx = current_idx + i;
                
                bool in_cache;
                {
                    std::lock_guard<std::mutex> cache_lock(cache_mutex_);
                    in_cache = cache_.count(next_idx);
                }
                if (in_cache) continue;

                bool in_queue = false;
                for(const auto& item : load_queue_) if (item == next_idx) in_queue = true;
                if (in_queue) continue;
                
                next_problems_to_queue.push_back(next_idx);
                if(load_queue_.size() + next_problems_to_queue.size() >= max_preload_queue_size_) break;
            }
        }
        
        if (!next_problems_to_queue.empty()) {
            preloadProblems(next_problems_to_queue);
        }
    }

    // Adds a list of problems to the back of the loading queue
    void preloadProblems(const std::vector<int>& problem_indices) {
        std::lock_guard<std::mutex> lock(load_queue_mutex_);
        for (int idx : problem_indices) {
            load_queue_.push_back(idx);
        }
        load_queue_cv_.notify_all(); // Wake up all loaders
    }

    // Loads a single problem's data from disk
    std::shared_ptr<PreloadedProblemData> loadProblemData(int problem_idx) {
        const Problem& problem = all_problems_[problem_idx];
        auto data = std::make_shared<PreloadedProblemData>();
        data->problem_idx = problem_idx;

        std::string image_folder = dense_folder_ + "/images";
        std::string cam_folder = dense_folder_ + "/cams";

        // Load reference image
        {
            std::stringstream image_path;
            image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
            cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
            if (image_uint.empty()) {
                throw std::runtime_error("Failed to load image: " + image_path.str());
            }
            cv::Mat image_float;
            image_uint.convertTo(image_float, CV_32FC1);
            data->images_float.push_back(image_float);
        }

        // Load reference camera
        {
            std::stringstream cam_path;
            cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << "_cam.txt";
            Camera camera = ReadCamera(cam_path.str());
            camera.height = data->images_float[0].rows;
            camera.width = data->images_float[0].cols;
            data->cameras.push_back(camera);
        }

        // Load source images and cameras
        for (size_t i = 0; i < problem.src_image_ids.size(); ++i) {
            { // Load image
                std::stringstream image_path;
                image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << ".jpg";
                cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
                if (image_uint.empty()) {
                    throw std::runtime_error("Failed to load image: " + image_path.str());
                }
                cv::Mat image_float;
                image_uint.convertTo(image_float, CV_32FC1);
                data->images_float.push_back(image_float);
            }
            { // Load camera
                std::stringstream cam_path;
                cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << "_cam.txt";
                Camera camera = ReadCamera(cam_path.str());
                camera.height = data->images_float.back().rows;
                camera.width = data->images_float.back().cols;
                data->cameras.push_back(camera);
            }
        }

        // Scale images and cameras
        int max_image_size = problem.cur_image_size;
        for (size_t i = 0; i < data->images_float.size(); ++i) {
            if (data->images_float[i].cols <= max_image_size && data->images_float[i].rows <= max_image_size) {
                continue;
            }

            const float factor_x = static_cast<float>(max_image_size) / data->images_float[i].cols;
            const float factor_y = static_cast<float>(max_image_size) / data->images_float[i].rows;
            const float factor = std::min(factor_x, factor_y);
            const int new_cols = std::round(data->images_float[i].cols * factor);
            const int new_rows = std::round(data->images_float[i].rows * factor);

            cv::Mat scaled_image;
            cv::resize(data->images_float[i], scaled_image, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
            data->images_float[i] = scaled_image;

            const float scale_x = new_cols / static_cast<float>(data->cameras[i].width);
            const float scale_y = new_rows / static_cast<float>(data->cameras[i].height);

            if (data->cameras[i].model == SPHERE) {
                data->cameras[i].params[1] *= scale_x;
                data->cameras[i].params[2] *= scale_y;
            } else {
                data->cameras[i].K[0] *= scale_x;
                data->cameras[i].K[2] *= scale_x;
                data->cameras[i].K[4] *= scale_y;
                data->cameras[i].K[5] *= scale_y;
            }
            data->cameras[i].height = new_rows;
            data->cameras[i].width = new_cols;
        }

        // Load depth maps if using geometric consistency
        if (geom_consistency_) {
            std::stringstream result_path;
            result_path << dense_folder_ << "/ACMMP/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
            std::string suffix = multi_geometry_ ? "/depths_geom.dmb" : "/depths.dmb";
            std::string depth_path = result_path.str() + suffix;
            
            cv::Mat_<float> ref_depth;
            if (readDepthDmb(depth_path, ref_depth) == 0) {
                data->depths.push_back(ref_depth);
                for (size_t i = 0; i < problem.src_image_ids.size(); ++i) {
                    std::stringstream src_result_path;
                    src_result_path << dense_folder_ << "/ACMMP/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
                    std::string src_depth_path = src_result_path.str() + suffix;
                    cv::Mat_<float> src_depth;
                    if (readDepthDmb(src_depth_path, src_depth) == 0) {
                        data->depths.push_back(src_depth);
                    }
                }
            }
        }

        data->ready = true;
        return data;
    }

    // Estimates the memory size of loaded data
    size_t estimateDataSize(const std::shared_ptr<PreloadedProblemData>& data) const {
        if (!data) return 0;
        size_t size = 0;
        for (const auto& img : data->images_float) {
            size += img.total() * img.elemSize();
        }
        for (const auto& depth : data->depths) {
            size += depth.total() * depth.elemSize();
        }
        size += data->cameras.size() * sizeof(Camera);
        return size;
    }
};

#endif // ASYNC_IMAGE_LOADER_H