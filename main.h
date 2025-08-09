#ifndef _MAIN_H_
#define _MAIN_H_

#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/opencv.hpp"

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <cuda_texture_types.h>
#include <curand_kernel.h>
#include <vector_types.h>

#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <map>
#include <memory>
#include "iomanip"

#include <sys/stat.h> // mkdir
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
  float params[4];       // For SPHERE: [f, cx, cy, unused]

  float R[9];
  float t[3];
  float K[9];
  int   width, height;
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
    Triangle (const cv::Point _pt1, const cv::Point _pt2, const cv::Point _pt3) : pt1(_pt1) , pt2(_pt2), pt3(_pt3) {}
};

struct PointList {
    float3 coord;
    float3 normal;
    float3 color;
};

#endif // _MAIN_H_
