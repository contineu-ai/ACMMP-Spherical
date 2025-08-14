#include "main.h"
#include "ACMMP.h"
#include "FusionGPU.h"
#include "SphericalLUT_MultiRes.h"  // Add multi-resolution LUT support

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
    int max_num_downscale = -1;
    int size_bound = 350;
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

void ProcessProblem(const std::string &dense_folder, const std::vector<Problem> &problems, 
                   const int idx, bool geom_consistency, bool planar_prior, 
                   bool hierarchy, bool multi_geometrty=false)
{
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
    acmmp.CudaSpaceInitialization(dense_folder, problem);
    acmmp.RunPatchMatch();

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

    if (planar_prior) {
        std::cout << "Run Planar Prior Assisted PatchMatch MVS ..." << std::endl;
        acmmp.SetPlanarPriorParams();

        const cv::Rect imageRC(0, 0, width, height);
        std::vector<cv::Point> support2DPoints;

        acmmp.GetSupportPoints(support2DPoints);
        const auto triangles = acmmp.DelaunayTriangulation(imageRC, support2DPoints);
        cv::Mat refImage = acmmp.GetReferenceImage().clone();
        std::vector<cv::Mat> mbgr(3);
        mbgr[0] = refImage.clone();
        mbgr[1] = refImage.clone();
        mbgr[2] = refImage.clone();
        cv::Mat srcImage;
        cv::merge(mbgr, srcImage);
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && 
                imageRC.contains(triangle.pt3)) {
                cv::line(srcImage, triangle.pt1, triangle.pt2, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt1, triangle.pt3, cv::Scalar(0, 0, 255));
                cv::line(srcImage, triangle.pt2, triangle.pt3, cv::Scalar(0, 0, 255));
            }
        }
        std::string triangulation_path = result_folder + "/triangulation.png";
        cv::imwrite(triangulation_path, srcImage);

        cv::Mat_<float> mask_tri = cv::Mat::zeros(height, width, CV_32FC1);
        std::vector<float4> planeParams_tri;
        planeParams_tri.clear();

        uint32_t idx = 0;
        for (const auto triangle : triangles) {
            if (imageRC.contains(triangle.pt1) && imageRC.contains(triangle.pt2) && 
                imageRC.contains(triangle.pt3)) {
                float L01 = sqrt(pow(triangle.pt1.x - triangle.pt2.x, 2) + 
                                pow(triangle.pt1.y - triangle.pt2.y, 2));
                float L02 = sqrt(pow(triangle.pt1.x - triangle.pt3.x, 2) + 
                                pow(triangle.pt1.y - triangle.pt3.y, 2));
                float L12 = sqrt(pow(triangle.pt2.x - triangle.pt3.x, 2) + 
                                pow(triangle.pt2.y - triangle.pt3.y, 2));

                float max_edge_length = std::max(L01, std::max(L02, L12));
                float step = 1.0 / max_edge_length;

                for (float p = 0; p < 1.0; p += step) {
                    for (float q = 0; q < 1.0 - p; q += step) {
                        int x = p * triangle.pt1.x + q * triangle.pt2.x + 
                               (1.0 - p - q) * triangle.pt3.x;
                        int y = p * triangle.pt1.y + q * triangle.pt2.y + 
                               (1.0 - p - q) * triangle.pt3.y;
                        mask_tri(y, x) = idx + 1.0;
                    }
                }

                float4 n4 = acmmp.GetPriorPlaneParams(triangle, depths);
                planeParams_tri.push_back(n4);
                idx++;
            }
        }

        cv::Mat_<float> priordepths = cv::Mat::zeros(height, width, CV_32FC1);
        for (int i = 0; i < width; ++i) {
            for (int j = 0; j < height; ++j) {
                if (mask_tri(j, i) > 0) {
                    float d = acmmp.GetDepthFromPlaneParam(planeParams_tri[mask_tri(j, i) - 1], i, j);
                    if (d <= acmmp.GetMaxDepth() && d >= acmmp.GetMinDepth()) {
                        priordepths(j, i) = d;
                    }
                    else {
                        mask_tri(j, i) = 0;
                    }
                }
            }
        }

        acmmp.CudaPlanarPriorInitialization(planeParams_tri, mask_tri);
        acmmp.RunPatchMatch();

        for (int col = 0; col < width; ++col) {
            for (int row = 0; row < height; ++row) {
                int center = row * width + col;
                float4 plane_hypothesis = acmmp.GetPlaneHypothesis(center);
                depths(row, col) = plane_hypothesis.w;
                normals(row, col) = cv::Vec3f(plane_hypothesis.x, plane_hypothesis.y, plane_hypothesis.z);
                costs(row, col) = acmmp.GetCost(center);
            }
        }
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

    std::cout << "Run JBU for image " << problem.ref_image_id <<  ".jpg" << std::endl;
    RunJBU(scaled_image_float, ref_depth, dense_folder, problem );
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cout << "USAGE: ACMMP dense_folder" << std::endl;
        return -1;
    }

    std::string dense_folder = argv[1];
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
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i) {
                ProcessProblem(dense_folder, problems, i, geom_consistency, planar_prior, hierarchy);
            }
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                multi_geometry = (geom_iter > 0);
                for (size_t i = 0; i < num_images; ++i) {
                    ProcessProblem(dense_folder, problems, i, geom_consistency, 
                                 planar_prior, hierarchy, multi_geometry);
                }
            }
        }
        else {
            for (size_t i = 0; i < num_images; ++i) {
               JointBilateralUpsampling(dense_folder, problems[i], problems[i].cur_image_size);
            }

            hierarchy = true;
            geom_consistency = false;
            planar_prior = true;
            for (size_t i = 0; i < num_images; ++i) {
                ProcessProblem(dense_folder, problems, i, geom_consistency, 
                             planar_prior, hierarchy);
            }
            hierarchy = false;
            geom_consistency = true;
            planar_prior = false;
            for (int geom_iter = 0; geom_iter < geom_iterations; ++geom_iter) {
                multi_geometry = (geom_iter > 0);
                for (size_t i = 0; i < num_images; ++i) {
                    ProcessProblem(dense_folder, problems, i, geom_consistency, 
                                 planar_prior, hierarchy, multi_geometry);
                }
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