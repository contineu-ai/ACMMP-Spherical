#include "main.h"
#include "ACMMP.h"
#include "FusionGPU.h"
#include "SphericalLUT_MultiRes.h"  // Add multi-resolution LUT support
#include <chrono>
#include <omp.h>
#include <atomic>   // Add this for atomic operations
#include "BatchACMMP.h"
#include <cstring>  // For strcmp

void makeDir(const std::string& path) {
    if (mkdir(path.c_str(), 0777) && errno != EEXIST) {
        std::cerr << "Error creating directory: " << path << std::endl;
    }
}

void printUsage(const char* program_name) {
    std::cout << "USAGE: " << program_name << " dense_folder [OPTIONS]" << std::endl;
    std::cout << "OPTIONS:" << std::endl;
    std::cout << "  --no-batch              Disable batch processing (use sequential processing)" << std::endl;
    std::cout << "  --batch                 Enable batch processing (default)" << std::endl;
    std::cout << "  --help                  Show this help message" << std::endl;
    std::cout << std::endl;
    std::cout << "EXAMPLES:" << std::endl;
    std::cout << "  " << program_name << " /path/to/dense/folder                    # Use batch processing (default)" << std::endl;
    std::cout << "  " << program_name << " /path/to/dense/folder --no-batch        # Use sequential processing" << std::endl;
}

void ProcessProblem(const std::string &dense_folder, const std::vector<Problem> &problems, 
                   const int idx, bool geom_consistency, bool planar_prior, 
                   bool hierarchy, bool multi_geometrty)
{
    ProblemGPUResources resources;
    const Problem problem = problems[idx];
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') 
              << problem.ref_image_id << "..." << std::endl;
    cudaSetDevice(0);
    
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) 
                << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    ACMMP acmmp;
    if (geom_consistency) {
        acmmp.SetGeomConsistencyParams(multi_geometrty);
    }
    if (hierarchy) {
        acmmp.SetHierarchyParams();
    }

    acmmp.InuputInitialization(dense_folder, problems, idx);
    acmmp.CudaSpaceInitialization(dense_folder, problem,&resources);
    acmmp.RunPatchMatch(&resources);

    const int width = acmmp.GetReferenceImageWidth();
    const int height = acmmp.GetReferenceImageHeight();

    cv::Mat_<float> depths = cv::Mat::zeros(height, width, CV_32FC1);
    cv::Mat_<cv::Vec3f> normals = cv::Mat::zeros(height, width, CV_32FC3);
    cv::Mat_<float> costs = cv::Mat::zeros(height, width, CV_32FC1);

    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            int center = row * width + col;
            float4 plane_hypothesis = acmmp.GetPlaneHypothesis(center);
            depths(row, col) = plane_hypothesis.w;
            normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
            costs(row, col) = acmmp.GetCost(center);
        }
    }

// ── Planar prior: timed + tile-parallel rasterization (no thinning) ──────────
if (planar_prior) {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::time_point a, clock::time_point b) {
        return std::chrono::duration_cast<std::chrono::milliseconds>(b - a).count();
    };

    // Tune tile size for your CPU cache; 64 or 128 are good starts.
    constexpr int TILE = 128;

    auto T_block_start = clock::now();
    std::cout << "Run Planar Prior Assisted PatchMatch MVS (fast) ..." << std::endl;

    auto T0 = clock::now();
    acmmp.SetPlanarPriorParams();
    auto T1 = clock::now();
    std::cout << "  [timing] SetPlanarPriorParams: " << ms(T0, T1) << " ms\n";

    const cv::Rect imageRC(0, 0, width, height);

    // 1) Support points + Delaunay (unchanged)
    T0 = clock::now();
    std::vector<cv::Point> support2DPoints;
    acmmp.GetSupportPoints(support2DPoints);
    auto Tsp = clock::now();

    const auto& triangles = acmmp.DelaunayTriangulation(imageRC, support2DPoints);
    auto Tdl = clock::now();

    std::cout << "  [timing] GetSupportPoints (" << support2DPoints.size() << " pts): "
              << ms(T0, Tsp) << " ms\n";
    std::cout << "  [timing] DelaunayTriangulation (" << triangles.size() << " tris): "
              << ms(Tsp, Tdl) << " ms\n";

    // ═══════════════════════════════════════════════════════════════════════════
    // 2) OPTIMIZED Plane params per triangle (parallel with atomic counting)
    // ═══════════════════════════════════════════════════════════════════════════
    auto Tpp0 = clock::now();
    std::vector<float4>  planeParams_tri(triangles.size());
    std::vector<uint8_t> tri_valid(triangles.size(), 0);
    
    // Use atomic counter for thread-safe valid triangle counting
    std::atomic<size_t> valid_tris_atomic{0};
    
    // Get optimal number of threads and chunk size
    const int num_threads = omp_get_max_threads();
    const size_t chunk_size = std::max(size_t(1), triangles.size() / (num_threads * 4));
    
    // Parallel processing with dynamic scheduling for better load balancing
    #pragma omp parallel for schedule(dynamic, chunk_size)
    for (size_t i = 0; i < triangles.size(); ++i) {
        const auto& tr = triangles[i];
        
        // Quick bounds check first (cheapest operation)
        if (!(imageRC.contains(tr.pt1) && imageRC.contains(tr.pt2) && imageRC.contains(tr.pt3))) {
            tri_valid[i] = 0;
            planeParams_tri[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            continue;
        }
        
        // Compute plane parameters using optimized function
        planeParams_tri[i] = acmmp.GetPriorPlaneParams(tr, depths);
        
        // Check if plane computation was successful
        bool is_valid = (planeParams_tri[i].x != 0.0f || planeParams_tri[i].y != 0.0f || 
                        planeParams_tri[i].z != 0.0f);
        tri_valid[i] = is_valid ? 1 : 0;
        
        // Thread-safe increment of valid triangle counter
        if (is_valid) {
            valid_tris_atomic.fetch_add(1, std::memory_order_relaxed);
        }
    }
    
    size_t valid_tris = valid_tris_atomic.load();
    auto Tpp1 = clock::now();
    std::cout << "  [timing] PlaneParams (" << valid_tris << " valid tris): "
              << ms(Tpp0, Tpp1) << " ms [OPTIMIZED]\n";

    // 3) Build triangle -> tile map (serial, cheap)
    auto Tmap0 = clock::now();
    const int nx = (width  + TILE - 1) / TILE;
    const int ny = (height + TILE - 1) / TILE;
    const int num_tiles = nx * ny;

    auto clampi = [](int v, int lo, int hi_excl) { return std::max(lo, std::min(v, hi_excl)); };

    std::vector<std::vector<int>> tile_tris(num_tiles);
    tile_tris.shrink_to_fit();
    for (int i = 0; i < (int)triangles.size(); ++i) {
        if (!tri_valid[i]) continue;
        const auto& tr = triangles[i];
        int minx = std::min({tr.pt1.x, tr.pt2.x, tr.pt3.x});
        int maxx = std::max({tr.pt1.x, tr.pt2.x, tr.pt3.x});
        int miny = std::min({tr.pt1.y, tr.pt2.y, tr.pt3.y});
        int maxy = std::max({tr.pt1.y, tr.pt2.y, tr.pt3.y});
        // Clip to image (exclusive upper bound for tile indexing)
        minx = clampi(minx, 0, width);
        maxx = clampi(maxx, 0, width  - 1);
        miny = clampi(miny, 0, height);
        maxy = clampi(maxy, 0, height - 1);
        if (minx > maxx || miny > maxy) continue;

        int tx0 = minx / TILE;
        int tx1 = maxx / TILE;
        int ty0 = miny / TILE;
        int ty1 = maxy / TILE;

        for (int ty = ty0; ty <= ty1; ++ty) {
            for (int tx = tx0; tx <= tx1; ++tx) {
                tile_tris[ty * nx + tx].push_back(i);
            }
        }
    }
    auto Tmap1 = clock::now();
    std::cout << "  [timing] Tile map build (" << nx << "x" << ny << " tiles): "
              << ms(Tmap0, Tmap1) << " ms\n";

    // 4) Parallel rasterization per tile ROI (no overlap between threads)
    auto Trast0 = clock::now();
    cv::Mat mask_tri_i(height, width, CV_32SC1, cv::Scalar(0));  // id+1, 0=bg

    cv::parallel_for_(cv::Range(0, num_tiles), [&](const cv::Range& r) {
        for (int tid = r.start; tid < r.end; ++tid) {
            const int ty = tid / nx;
            const int tx = tid % nx;
            const int x0 = tx * TILE;
            const int y0 = ty * TILE;
            const int x1 = std::min(x0 + TILE, width);
            const int y1 = std::min(y0 + TILE, height);
            const cv::Rect tileR(x0, y0, x1 - x0, y1 - y0);

            cv::Mat tileView = mask_tri_i(tileR);  // disjoint memory per tile

            // Draw all intersecting triangles into this ROI
            const auto& lst = tile_tris[tid];
            for (int idx : lst) {
                if (!tri_valid[idx]) continue;
                const auto& tr = triangles[(size_t)idx];

                // Shift points to tile-local coordinates
                cv::Point pts_local[3] = {
                    tr.pt1 - tileR.tl(),
                    tr.pt2 - tileR.tl(),
                    tr.pt3 - tileR.tl()
                };
                // Safe: OpenCV clips polygon to ROI. Exclusive tile bounds avoid races.
                cv::fillConvexPoly(tileView, pts_local, 3, int(idx + 1), cv::LINE_8, 0);
            }
        }
    });
    auto Trast1 = clock::now();
    std::cout << "  [timing] Rasterize (tile-parallel): " << ms(Trast0, Trast1) << " ms\n";

    // Labeled coverage stats
    auto Tcnt0 = clock::now();
    cv::Mat1b nzmask;
    cv::compare(mask_tri_i, 0, nzmask, cv::CMP_NE);
    int labeled_px = cv::countNonZero(nzmask);
    auto Tcnt1 = clock::now();
    std::cout << "  [stats ] Labeled pixels: " << labeled_px << " ("
              << std::fixed << std::setprecision(2)
              << (100.0 * double(labeled_px) / double(width * height)) << "%), "
              << "counted in " << ms(Tcnt0, Tcnt1) << " ms\n";

    // 5) Depth-from-plane (serial; do NOT call ACMMP from multiple threads)
    auto Td0 = clock::now();
    cv::Mat priordepths(height, width, CV_32FC1, cv::Scalar(0.0f));
    const float dmin = acmmp.GetMinDepth();
    const float dmax = acmmp.GetMaxDepth();
    for (int y = 0; y < height; ++y) {
        const int*  lab = mask_tri_i.ptr<int>(y);
        float*      out = priordepths.ptr<float>(y);
        for (int x = 0; x < width; ++x) {
            const int id = lab[x];
            if (id <= 0) continue;
            const size_t tix = size_t(id - 1);
            if (!tri_valid[tix]) continue;
            const float4& n4 = planeParams_tri[tix];
            const float d = acmmp.GetDepthFromPlaneParam(n4, x, y);
            if (d >= dmin && d <= dmax) out[x] = d;
        }
    }
    auto Td1 = clock::now();
    const double depth_ms = ms(Td0, Td1);
    const double mpix = labeled_px / 1e6;
    const double mpix_per_s = (depth_ms > 0.0) ? (mpix / (depth_ms / 1000.0)) : 0.0;
    std::cout << "  [timing] Depth-from-plane fill: " << depth_ms << " ms  ("
              << std::fixed << std::setprecision(2) << mpix_per_s << " MPix/s)\n";

    // 6) CUDA prior init + PatchMatch (sync for real timings)
    auto Tc0 = clock::now();
    cv::Mat mask_tri_f; mask_tri_i.convertTo(mask_tri_f, CV_32F);
    acmmp.CudaPlanarPriorInitialization(planeParams_tri, mask_tri_f);
    cudaDeviceSynchronize();
    auto Tc1 = clock::now();
    std::cout << "  [timing] CudaPlanarPriorInitialization: " << ms(Tc0, Tc1) << " ms\n";

    auto Tp0 = clock::now();
    acmmp.RunPatchMatch(&resources);
    cudaDeviceSynchronize();
    auto Tp1 = clock::now();
    std::cout << "  [timing] RunPatchMatch (prior-assisted): " << ms(Tp0, Tp1) << " ms\n";

    // 7) Extract outputs
    auto Te0 = clock::now();
    for (int col = 0; col < width; ++col) {
        for (int row = 0; row < height; ++row) {
            const int center = row * width + col;
            const float4 ph = acmmp.GetPlaneHypothesis(center);
            depths(row, col)  = ph.w;
            normals(row, col) = cv::Vec3f(ph.x, ph.y, ph.z);
            costs(row, col)   = acmmp.GetCost(center);
        }
    }
    auto Te1 = clock::now();
    std::cout << "  [timing] Extract outputs: " << ms(Te0, Te1) << " ms\n";

    auto T_block_end = clock::now();
    std::cout << "  [timing] Planar-prior TOTAL: " << ms(T_block_start, T_block_end) << " ms\n";
}

    std::string suffix = "/depths.dmb";
    if (geom_consistency) {
        suffix = "/depths_geom.dmb";
    }
    std::string depth_path = result_folder + suffix;
    std::string normal_path = result_folder + "/normals.dmb";
    std::string cost_path = result_folder + "/costs.dmb";
    writeDepthDmb(depth_path, depths);
    writeNormalDmb(normal_path, normals);
    writeDepthDmb(cost_path, costs);
    std::cout << "Processing image " << std::setw(8) << std::setfill('0') 
              << problem.ref_image_id << " done!" << std::endl;
}


// Process problems using the original sequential method
void ProcessProblemsSequential(const std::string &dense_folder,
                              const std::vector<Problem> &problems,
                              bool geom_consistency,
                              bool planar_prior,
                              bool hierarchy,
                              bool multi_geometry) {
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Sequential Processing" << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (size_t i = 0; i < problems.size(); ++i) {
        ProcessProblem(dense_folder, problems, i, geom_consistency, planar_prior, hierarchy, multi_geometry);
    }
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Sequential Processing Complete!" << std::endl;
    std::cout << "Total time: " << duration.count() << " seconds" << std::endl;
    std::cout << "========================================\n" << std::endl;
}

// Wrapper function that chooses between batch and sequential processing
void ProcessProblemsWithMode(const std::string &dense_folder,
                            std::vector<Problem> &problems,
                            bool geom_consistency,
                            bool planar_prior,
                            bool hierarchy,
                            bool multi_geometry ,
                            bool use_batching ) {
    
    if (use_batching) {
        std::cout << "Processing mode: BATCH (parallel GPU streams)" << std::endl;
        ProcessProblemsInParallel(dense_folder, problems, geom_consistency, 
                                 planar_prior, hierarchy, multi_geometry);
    } else {
        std::cout << "Processing mode: SEQUENTIAL (one problem at a time)" << std::endl;
        ProcessProblemsSequential(dense_folder, problems, geom_consistency, 
                                 planar_prior, hierarchy, multi_geometry);
    }
}

void GenerateSampleList(const std::string &dense_folder, std::vector<Problem> &problems)
{
    std::string cluster_list_path = dense_folder + std::string("/pair.txt");
    problems.clear();

    std::ifstream file(cluster_list_path);
    int num_images;
    file >> num_images;

    for (int i = 0; i < num_images; ++i) {
        Problem problem;
        problem.src_image_ids.clear();
        file >> problem.ref_image_id;

        int num_src_images;
        file >> num_src_images;
        for (int j = 0; j < num_src_images; ++j) {
            int id;
            float score;
            file >> id >> score;
            if (score <= 0.0f) {
                continue;
            }
            problem.src_image_ids.push_back(id);
        }
        problems.push_back(problem);
    }
}

// Initialize LUTs for all expected resolutions based on multi-scale settings
void InitializeLUTsForAllResolutions(const std::string &dense_folder, 
                                     const std::vector<Problem> &problems,
                                     int max_num_downscale)
{
    if (problems.empty()) {
        std::cerr << "No problems to process, cannot initialize LUTs" << std::endl;
        return;
    }
    
    // Initialize the LUT manager
    InitializeLUTManager();
    
    // Read the first camera to get base resolution and principal point
    std::string cam_folder = dense_folder + std::string("/cams");
    std::stringstream cam_path;
    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') 
             << problems[0].ref_image_id << "_cam.txt";
    
    Camera first_camera = ReadCamera(cam_path.str());
    
    // Get the maximum image size from the first problem
    std::string image_folder = dense_folder + std::string("/images");
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') 
               << problems[0].ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    
    int base_width = image_uint.cols;
    int base_height = image_uint.rows;
    float base_cx = first_camera.params[1];
    float base_cy = first_camera.params[2];
    
    std::cout << "Base resolution: " << base_width << "x" << base_height 
              << " with principal point (" << base_cx << ", " << base_cy << ")" << std::endl;
    
    // Calculate all possible resolutions based on downscaling
    std::set<ResolutionKey> unique_resolutions;
    
    // Add base resolution
    unique_resolutions.insert({base_width, base_height, base_cx, base_cy});
    
    // Process all problems to find all unique resolutions
    for (const auto& problem : problems) {
        // Get image size for this problem
        std::stringstream img_path;
        img_path << image_folder << "/" << std::setw(8) << std::setfill('0') 
                 << problem.ref_image_id << ".jpg";
        cv::Mat_<uint8_t> img = cv::imread(img_path.str(), cv::IMREAD_GRAYSCALE);
        
        int width = img.cols;
        int height = img.rows;
        
        // Calculate scaled versions based on max_image_size
        if (problem.max_image_size > 0) {
            if (width > problem.max_image_size || height > problem.max_image_size) {
                const float factor_x = static_cast<float>(problem.max_image_size) / width;
                const float factor_y = static_cast<float>(problem.max_image_size) / height;
                const float factor = std::min(factor_x, factor_y);
                
                int new_width = std::round(width * factor);
                int new_height = std::round(height * factor);
                float new_cx = base_cx * factor;
                float new_cy = base_cy * factor;
                
                unique_resolutions.insert({new_width, new_height, new_cx, new_cy});
            }
        }
        
        // Add pyramid levels
        for (int scale = 0; scale <= max_num_downscale; ++scale) {
            int scale_factor = 1 << scale;  // 2^scale
            int scaled_width = width / scale_factor;
            int scaled_height = height / scale_factor;
            
            if (scaled_width < 32 || scaled_height < 32) break;
            
            float scaled_cx = base_cx * scaled_width / base_width;
            float scaled_cy = base_cy * scaled_height / base_height;
            
            unique_resolutions.insert({scaled_width, scaled_height, scaled_cx, scaled_cy});
        }
    }
    
    std::cout << "Found " << unique_resolutions.size() << " unique resolutions" << std::endl;
    
    // Initialize LUTs for all unique resolutions
    for (const auto& res : unique_resolutions) {
        g_lut_manager->GetOrCreateLUT(res.width, res.height, res.cx, res.cy);
    }
    
    std::cout << "Initialized " << unique_resolutions.size() << " LUTs" << std::endl;
    std::cout << "Total memory usage: " 
              << g_lut_manager->GetTotalMemoryUsage() / (1024.0 * 1024.0) << " MB" << std::endl;
}

int ComputeMultiScaleSettings(const std::string &dense_folder, std::vector<Problem> &problems)
{
    int max_num_downscale = 2;
    int size_bound = 800;
    PatchMatchParams pmp;
    std::string image_folder = dense_folder + std::string("/images");

    size_t num_images = problems.size();

    for (size_t i = 0; i < num_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') 
                   << problems[i].ref_image_id << ".jpg";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);

        int rows = image_uint.rows;
        int cols = image_uint.cols;
        int max_size = std::max(rows, cols);
        if (max_size > pmp.max_image_size) {
            max_size = pmp.max_image_size;
        }
        problems[i].max_image_size = max_size;

        int k = 0;
        while (max_size > size_bound) {
            max_size /= 2;
            k++;
        }

        if (k > max_num_downscale) {
            max_num_downscale = k;
        }

        problems[i].num_downscale = k;
    }

    return max_num_downscale;
}

void JointBilateralUpsampling(const std::string &dense_folder, const Problem &problem, int acmmp_size)
{
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    std::string depth_path = result_folder + "/depths_geom.dmb";
    cv::Mat_<float> ref_depth;
    readDepthDmb(depth_path, ref_depth);

    std::string image_folder = dense_folder + std::string("/images");
    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".jpg";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    const float factor_x = static_cast<float>(acmmp_size) / image_float.cols;
    const float factor_y = static_cast<float>(acmmp_size) / image_float.rows;
    const float factor = std::min(factor_x, factor_y);

    const int new_cols = std::round(image_float.cols * factor);
    const int new_rows = std::round(image_float.rows * factor);
    cv::Mat scaled_image_float;
    cv::resize(image_float, scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);

    // std::cout << "Run JBU for image " << problem.ref_image_id <<  ".jpg" << std::endl;
    RunJBU(scaled_image_float, ref_depth, dense_folder, problem);
}

void ProcessProblemsInParallel(const std::string &dense_folder, 
                               std::vector<Problem> &problems,
                               bool geom_consistency,
                               bool planar_prior,
                               bool hierarchy,
                               bool multi_geometry) {
    
    std::cout << "\n========================================" << std::endl;
    std::cout << "Starting Parallel GPU + Disk Processing" << std::endl;
    std::cout << "Mode: " << (geom_consistency ? "GEOM" : "PLANAR") 
              << (hierarchy ? "+HIERARCHY" : "") 
              << (multi_geometry ? "+MULTI_GEOM" : "") << std::endl;
    std::cout << "========================================" << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    // Ensure output directory exists
    std::string output_dir = dense_folder + "/ACMMP";
    makeDir(output_dir);
    
    // Create batch processor with scope management
    {
        BatchACMMP batch_processor(dense_folder, problems,
                                  geom_consistency, planar_prior, hierarchy, 
                                  multi_geometry);
        
        // Start processing
        batch_processor.processAllProblems();
        
        // Simple polling-based wait - more reliable than complex synchronization
        bool done = false;
        auto last_report = std::chrono::steady_clock::now();
        
        while (!done) {
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
            
            size_t gpu_completed = batch_processor.getCompletedGPUProblems();
            size_t disk_completed = batch_processor.getCompletedDiskWrites();
            size_t total = problems.size();
            
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(now - last_report).count() >= 60) {
                std::cout << "[Progress] GPU: " << gpu_completed << "/" << total 
                          << ", Disk: " << disk_completed << "/" << total
                          << ", Active GPU: " << batch_processor.getActiveGPUProblems()
                          << ", Pending Disk: " << batch_processor.getPendingDiskWrites() << std::endl;
                last_report = now;
            }
            
            // Check if everything is complete
            if (disk_completed >= total && 
                batch_processor.getActiveGPUProblems() == 0 &&
                batch_processor.getPendingDiskWrites() == 0) {
                done = true;
            }
        }
        
        std::cout << "[BatchACMMP] All processing complete!" << std::endl;
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "Parallel Processing Complete!" << std::endl;
        std::cout << "Total time: " << total_duration.count() << " seconds" << std::endl;
        std::cout << "Peak RAM usage: " << batch_processor.getPeakMemoryUsage() << " MB" << std::endl;
        std::cout << "GPU throughput: " << (problems.size() * 60.0 / total_duration.count()) << " problems/minute" << std::endl;
        std::cout << "========================================\n" << std::endl;
        
        // batch_processor destructor will handle cleanup when going out of scope
    }
}


int main(int argc, char** argv)
{
    if (argc < 2) {
        printUsage(argv[0]);
        return -1;
    }

    std::string dense_folder = argv[1];
    bool use_batching = true;  // Default to batching enabled

    // Parse command line arguments
    for (int i = 2; i < argc; ++i) {
        if (strcmp(argv[i], "--no-batch") == 0) {
            use_batching = false;
        } else if (strcmp(argv[i], "--batch") == 0) {
            use_batching = true;
        } else if (strcmp(argv[i], "--help") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown option: " << argv[i] << std::endl;
            printUsage(argv[0]);
            return -1;
        }
    }

    std::cout << "=== ACMMP Multi-View Stereo Processing ===" << std::endl;
    std::cout << "Dense folder: " << dense_folder << std::endl;
    std::cout << "Batch processing: " << (use_batching ? "ENABLED" : "DISABLED") << std::endl;
    std::cout << "==========================================" << std::endl;

    std::vector<Problem> problems;
    GenerateSampleList(dense_folder, problems);

    std::string output_folder = dense_folder + std::string("/ACMMP");
    mkdir(output_folder.c_str(), 0777);

    size_t num_images = problems.size();
    std::cout << "There are " << num_images << " problems needed to be processed!" << std::endl;

    // Compute multi-scale settings
    int max_num_downscale = ComputeMultiScaleSettings(dense_folder, problems);
    
    // Initialize LUTs for all expected resolutions
    InitializeLUTsForAllResolutions(dense_folder, problems, max_num_downscale);

    int flag = 0;
    int geom_iterations = 2;
    bool geom_consistency = false;
    bool planar_prior = false;
    bool hierarchy = false;
    bool multi_geometry = false;
    
    while (max_num_downscale >= 0) {
        std::cout << "Scale: " << max_num_downscale << std::endl;

        for (size_t i = 0; i < num_images; ++i) {
            if (problems[i].num_downscale >= 0) {
                problems[i].cur_image_size = problems[i].max_image_size / 
                                            pow(2, problems[i].num_downscale);
                problems[i].num_downscale--;
            }
        }

        if (flag == 0) {
            flag = 1;
            std::cout << "Scale: " << max_num_downscale << std::endl;
            // Phase 1: Planar prior processing
            geom_consistency = false;
            planar_prior = true;
            ProcessProblemsWithMode(dense_folder, problems, geom_consistency, 
                                   planar_prior, hierarchy, false, use_batching);
            
            // Phase 2: Geometric consistency processing
            geom_consistency = true;
            planar_prior = false;
            std::cout << "Scale: " << max_num_downscale << std::endl;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                multi_geometry = (geom_iter > 0);
                ProcessProblemsWithMode(dense_folder, problems, geom_consistency, 
                                       planar_prior, hierarchy, multi_geometry, use_batching);
            }
        }
        else {
            // Joint Bilateral Upsampling phase
            std::cout << "Scale: " << max_num_downscale << std::endl;
            for (size_t i = 0; i < num_images; ++i) {
               JointBilateralUpsampling(dense_folder, problems[i], problems[i].cur_image_size);
            }

            // Phase 3: Hierarchy processing
            hierarchy = true;
            geom_consistency = false;
            planar_prior = true;
            std::cout << "Scale: " << max_num_downscale << std::endl;
            ProcessProblemsWithMode(dense_folder, problems, geom_consistency, 
                                   planar_prior, hierarchy, false, use_batching);
            
            // Phase 4: Final geometric consistency
            hierarchy = false;
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                std::cout << "Scale: " << max_num_downscale << std::endl;
                multi_geometry = (geom_iter > 0);
                ProcessProblemsWithMode(dense_folder, problems, geom_consistency, 
                                       planar_prior, hierarchy, multi_geometry, use_batching);
            }
        }

        max_num_downscale--;
    }

    geom_consistency = true;
    RunFusionCuda(dense_folder, problems, geom_consistency);

    // Clean up LUT manager
    FreeLUTManager();

    return 0;
}