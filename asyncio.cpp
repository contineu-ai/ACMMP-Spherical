// asyncio.cpp

// System I/O and standard library headers
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <cstdio>

// OpenCV headers
#include <opencv2/opencv.hpp>

// Your custom headers that define `Camera` and other data structures
// The previous error messages point to this being `main.h`
#include "main.h"
Camera ReadCamera(const std::string &cam_path)
{
    Camera camera;
    std::ifstream file(cam_path);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open camera file: " << cam_path << std::endl;
        return camera;
    }
    std::string token;

    file >> token; // Consume "extrinsic"
    for (int i = 0; i < 3; ++i) {
        file >> camera.R[3 * i + 0] >> camera.R[3 * i + 1] >> camera.R[3 * i + 2] >> camera.t[i];
    }
    
    // --- FIX 1: Consume the extra "0.0 0.0 0.0 1.0" line ---
    // This is the primary parsing bug.
    float dummy_float;
    for (int i = 0; i < 4; ++i) {
        file >> dummy_float;
    }

    // Now the file stream is positioned correctly.
    file >> token; // Consume "intrinsic" header
    file >> token; // This token should be the model name or K[0]

    if (token == "SPHERE") {
        camera.model = ::SPHERE;
        
        // Use clear variable names
        float file_f, file_cx, file_cy;
        file >> file_f >> file_cx >> file_cy;

        // Store the intrinsic parameters. Note that 'f' is not used in the new math,
        // but we store it anyway.
        camera.params[0] = file_f;
        camera.params[1] = file_cx; 
        camera.params[2] = file_cy; 
        camera.width = file_cx*2;
        camera.height = file_cy*2;
        float depth_min, depth_interval;
        int   n_depth_planes;
        float depth_max;

        file >> depth_min >> depth_interval >> n_depth_planes >> depth_max;

        camera.depth_min  = depth_min;
        camera.depth_max  = depth_max;
        
    } else {
        // This is the PINHOLE camera format
        camera.model = ::PINHOLE;
        
        // The token was the first value of the K matrix.
        camera.K[0] = std::stof(token);
        file >> camera.K[1] >> camera.K[2];
        file >> camera.K[3] >> camera.K[4] >> camera.K[5];
        file >> camera.K[6] >> camera.K[7] >> camera.K[8];

        float dummy1, dummy2;
        file >> camera.depth_min >> camera.depth_max >> dummy1 >> dummy2;
    }

    return camera;
}


int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage){
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    depth = cv::Mat::zeros(h,w,CV_32F);
    fread(depth.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return 0;
}

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1;
    int32_t h = depth.rows;
    int32_t w = depth.cols;
    int32_t nb = 1;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)depth.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}

int readNormalDmb (const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    FILE *inimage;
    inimage = fopen(file_path.c_str(), "rb");
    if (!inimage) {
        std::cout << "Error opening file " << file_path << std::endl;
        return -1;
    }

    int32_t type, h, w, nb;

    type = -1;

    fread(&type,sizeof(int32_t),1,inimage);
    fread(&h,sizeof(int32_t),1,inimage);
    fread(&w,sizeof(int32_t),1,inimage);
    fread(&nb,sizeof(int32_t),1,inimage);

    if (type != 1) {
        fclose(inimage);
        return -1;
    }

    int32_t dataSize = h*w*nb;

    normal = cv::Mat::zeros(h,w,CV_32FC3);
    fread(normal.data,sizeof(float),dataSize,inimage);

    fclose(inimage);
    return 0;
}

int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
{
    FILE *outimage;
    outimage = fopen(file_path.c_str(), "wb");
    if (!outimage) {
        std::cout << "Error opening file " << file_path << std::endl;
    }

    int32_t type = 1; //float
    int32_t h = normal.rows;
    int32_t w = normal.cols;
    int32_t nb = 3;

    fwrite(&type,sizeof(int32_t),1,outimage);
    fwrite(&h,sizeof(int32_t),1,outimage);
    fwrite(&w,sizeof(int32_t),1,outimage);
    fwrite(&nb,sizeof(int32_t),1,outimage);

    float* data = (float*)normal.data;

    int32_t datasize = w*h*nb;
    fwrite(data,sizeof(float),datasize,outimage);

    fclose(outimage);
    return 0;
}
