#ifndef MAIN_H_
#define MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

// Standard C++ includes
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include <iomanip>
#include <set>
#include <chrono>
#include <cstring>
#include <cerrno>

// System includes
#include <sys/stat.h>  // mkdir
#include <sys/types.h> // mkdir

#define MAX_IMAGES 256
#define JBU_NUM 2

enum CameraModel {
    PINHOLE = 0,
    SPHERE = 11 // Using 11 to match your comment, but any distinct int is fine
};

struct Camera {
    // Use the enum instead of std::string
    CameraModel model;
    
    // Use a fixed-size C-style array for parameters.
    // Make it large enough for any model. Pinhole uses 9 (K), Sphere uses 3 (f, cx, cy).
    // Let's keep K as it is and add sphere_params explicitly.
    float params[4]; // For SPHERE: [f, cx, cy, unused]
    float R[9];
    float t[3];
    float K[9];
    int width, height;
    float depth_min, depth_max;
};

struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    int max_image_size = 3200;
    int num_downscale = 0;
    int cur_image_size = 3200;
};

struct Triangle {
    cv::Point pt1, pt2, pt3;
    
    // Constructor that takes Point objects directly
    Triangle(const cv::Point& pt1, const cv::Point& pt2, const cv::Point& pt3) 
        : pt1(pt1), pt2(pt2), pt3(pt3) {}
    
    // Constructor that takes Point pointers (for backward compatibility if needed)
    Triangle(const cv::Point* pt1, const cv::Point* pt2, const cv::Point* pt3) 
        : pt1(*pt1), pt2(*pt2), pt3(*pt3) {}
};
struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};



// Function declarations
void makeDir(const std::string& path);
void printUsage(const char* program_name);

void GenerateSampleList(const std::string& dense_folder, std::vector<Problem>& problems);
void InitializeLUTsForAllResolutions(const std::string& dense_folder, 
                                     const std::vector<Problem>& problems,
                                     int max_num_downscale);
int ComputeMultiScaleSettings(const std::string& dense_folder, std::vector<Problem>& problems);

void ProcessProblem(const std::string& dense_folder, 
                   const std::vector<Problem>& problems, 
                   const int idx, 
                   bool geom_consistency, 
                   bool planar_prior, 
                   bool hierarchy, 
                   bool multi_geometry = false);

void ProcessProblemsInParallel(const std::string& dense_folder, 
                               std::vector<Problem>& problems,
                               bool geom_consistency,
                               bool planar_prior,
                               bool hierarchy,
                               bool multi_geometry = false);

void ProcessProblemsSequential(const std::string& dense_folder,
                              const std::vector<Problem>& problems,
                              bool geom_consistency,
                              bool planar_prior,
                              bool hierarchy,
                              bool multi_geometry = false);

void ProcessProblemsWithMode(const std::string& dense_folder,
                            std::vector<Problem>& problems,
                            bool geom_consistency,
                            bool planar_prior,
                            bool hierarchy,
                            bool multi_geometry = false,
                            bool use_batching = true);

void JointBilateralUpsampling(const std::string& dense_folder, 
                             const Problem& problem, 
                             int acmmp_size);




extern void InitializeLUTManager();
extern void FreeLUTManager();



#endif // MAIN_H_