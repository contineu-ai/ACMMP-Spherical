#include "ACMMP.h"
#include "BatchACMMP.h"
#include <cmath>

#include <cstdarg>

void StringAppendV(std::string* dst, const char* format, va_list ap) {
  // First try with a small fixed size buffer.
  static const int kFixedBufferSize = 1024;
  char fixed_buffer[kFixedBufferSize];

  // It is possible for methods that use a va_list to invalidate
  // the data in it upon use.  The fix is to make a copy
  // of the structure before using it and use that copy instead.
  va_list backup_ap;
  va_copy(backup_ap, ap);
  int result = vsnprintf(fixed_buffer, kFixedBufferSize, format, backup_ap);
  va_end(backup_ap);

  if (result < kFixedBufferSize) {
    if (result >= 0) {
      // Normal case - everything fits.
      dst->append(fixed_buffer, result);
      return;
    }

#ifdef _MSC_VER
    // Error or MSVC running out of space.  MSVC 8.0 and higher
    // can be asked about space needed with the special idiom below:
    va_copy(backup_ap, ap);
    result = vsnprintf(nullptr, 0, format, backup_ap);
    va_end(backup_ap);
#endif

    if (result < 0) {
      // Just an error.
      return;
    }
  }

  // Increase the buffer size to the size requested by vsnprintf,
  // plus one for the closing \0.
  const int variable_buffer_size = result + 1;
//   std::unique_ptr<char> variable_buffer(new char[variable_buffer_size]);
  std::unique_ptr<char[]> variable_buffer(new char[variable_buffer_size]);

  // Restore the va_list before we use it again.
  va_copy(backup_ap, ap);
  result =
      vsnprintf(variable_buffer.get(), variable_buffer_size, format, backup_ap);
  va_end(backup_ap);

  if (result >= 0 && result < variable_buffer_size) {
    dst->append(variable_buffer.get(), result);
  }
}

std::string StringPrintf(const char* format, ...) {
  va_list ap;
  va_start(ap, format);
  std::string result;
  StringAppendV(&result, format, ap);
  va_end(ap);
  return result;
}

void CudaSafeCall(const cudaError_t error, const std::string& file,
                  const int line) {
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("%s in %s at line %i", cudaGetErrorString(error),
                              file.c_str(), line)
              << std::endl;
    exit(EXIT_FAILURE);
  }
}

void CudaCheckError(const char* file, const int line) {
  cudaError error = cudaGetLastError();
  if (error != cudaSuccess) {
    std::cerr << StringPrintf("cudaCheckError() failed at %s:%i : %s", file,
                              line, cudaGetErrorString(error))
              << std::endl;
    exit(EXIT_FAILURE);
  }

  // More careful checking. However, this will affect performance.
  // Comment away if needed.
  error = cudaDeviceSynchronize();
  if (cudaSuccess != error) {
    std::cerr << StringPrintf("cudaCheckError() with sync failed at %s:%i : %s",
                              file, line, cudaGetErrorString(error))
              << std::endl;
    std::cerr
        << "This error is likely caused by the graphics card timeout "
           "detection mechanism of your operating system. Please refer to "
           "the FAQ in the documentation on how to solve this problem."
        << std::endl;
    exit(EXIT_FAILURE);
  }
}

ACMMP::ACMMP() {}

ACMMP::~ACMMP()
{
    // Free only the host-side memory that was allocated by this ACMMP instance.
    delete[] plane_hypotheses_host;
    delete[] costs_host;
    
    // Note: The original destructor had other host allocations to free.
    // Ensure you free any other host memory allocated with 'new' here.
    if (params.hierarchy) {
        delete[] scaled_plane_hypotheses_host; // Assuming these were created in this class
        delete[] pre_costs_host;
    }

    if (params.planar_prior) {
        delete[] prior_planes_host;
        delete[] plane_masks_host;
        cudaFree(prior_planes_cuda);
        cudaFree(plane_masks_cuda);
    }

}

#include <string> // For std::stof

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
    
    // Skip extrinsic padding line
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
// In file: ACMH.cpp
// Replace the entire function with this one.

void RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera)
{
    const int cols = depth.cols;
    const int rows = depth.rows;

    camera.width = cols;
    camera.height = rows;

    // If the image already matches the depth map size, we don't need to resize.
    // It's now safe to return early.
    if (cols == src.cols && rows == src.rows) {
        dst = src.clone();
        return;
    }

    // If resizing is needed, proceed.
    const float scale_x = cols / static_cast<float>(src.cols);
    const float scale_y = rows / static_cast<float>(src.rows);

    cv::resize(src, dst, cv::Size(cols,rows), 0, 0, cv::INTER_LINEAR);

    // Also scale the intrinsic parameters.
    if (camera.model == SPHERE) {
        camera.params[1] *= scale_x;   // cx
        camera.params[2] *= scale_y;   // cy
    } else { // PINHOLE
        camera.K[0] *= scale_x;  camera.K[2] *= scale_x;
        camera.K[4] *= scale_y;  camera.K[5] *= scale_y;
    }
}

float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera)
{
    // 1) Camera-frame 3D from pixel
    float3 Xc;
    if (camera.model == SPHERE) {
        // ERP -> unit ray (lon, lat), then scale by 'depth' (Euclidean along the ray)
        const float PI  = 3.14159265358979323846f;
        const float lon = ( (float)x - camera.params[1] ) / (float)camera.width  * (2.0f * PI);
        const float lat = -( (float)y - camera.params[2] ) / (float)camera.height * PI;

        const float cos_lat = std::cos(lat);
        Xc.x =  cos_lat * std::sin(lon) * depth;
        Xc.y = -std::sin(lat) * depth;
        Xc.z =  cos_lat * std::cos(lon) * depth;
    } else { // PINHOLE
        // Standard pinhole back-projection with depth = Z
        Xc.x = depth * ( (float)x - camera.K[2] ) / camera.K[0];
        Xc.y = depth * ( (float)y - camera.K[5] ) / camera.K[4];
        Xc.z = depth;
    }

    // 2) cam->world: Xw = R^T * Xc + C, with C = -R^T * t
    const float Cx = -(camera.R[0]*camera.t[0] + camera.R[1]*camera.t[1] + camera.R[2]*camera.t[2]);
    const float Cy = -(camera.R[3]*camera.t[0] + camera.R[4]*camera.t[1] + camera.R[5]*camera.t[2]);
    const float Cz = -(camera.R[6]*camera.t[0] + camera.R[7]*camera.t[1] + camera.R[8]*camera.t[2]);

    float3 Xw;
    Xw.x = camera.R[0]*Xc.x + camera.R[3]*Xc.y + camera.R[6]*Xc.z + Cx; // R^T * Xc + C
    Xw.y = camera.R[1]*Xc.x + camera.R[4]*Xc.y + camera.R[7]*Xc.z + Cy;
    Xw.z = camera.R[2]*Xc.x + camera.R[5]*Xc.y + camera.R[8]*Xc.z + Cz;

    return Xw;
}


float3 Get3DPointonRefCam(const int x, const int y, const float depth, const Camera camera)
 {
    // 1) Camera‐space point
    float3 point_cam;
    if (camera.model == SPHERE) {
        // Spherical projection using principal point
        float lon = (static_cast<float>(x) - camera.params[1]) / static_cast<float>(camera.width) * 2.0f * M_PI;
        float lat = -(static_cast<float>(y) - camera.params[2]) / static_cast<float>(camera.height) * M_PI;
        
        point_cam = make_float3(
            std::cos(lat) * std::sin(lon) * depth,
                           -std::sin(lat) * depth,
            std::cos(lat) * std::cos(lon) * depth
        );

    } else { // pin-hole
        point_cam = make_float3(
          depth * (static_cast<float>(x) - camera.K[2]) / camera.K[0],
          depth * (static_cast<float>(y) - camera.K[5]) / camera.K[4],
          depth
        );
    }

    return point_cam;
}   


void ProjectonCamera(const float3 PointX,
                     const Camera camera,
                     float2 &point,
                     float &depth) {
    // 1) Transform into camera frame (This part was correct)
    float3 tmp;
    tmp.x = camera.R[0]*PointX.x + camera.R[1]*PointX.y + camera.R[2]*PointX.z + camera.t[0];
    tmp.y = camera.R[3]*PointX.x + camera.R[4]*PointX.y + camera.R[5]*PointX.z + camera.t[1];
    tmp.z = camera.R[6]*PointX.x + camera.R[7]*PointX.y + camera.R[8]*PointX.z + camera.t[2];

    if (camera.model == SPHERE) {
        depth = std::sqrt(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
        if (depth < 1e-6) {
            point.x = camera.params[1];
            point.y = camera.params[2];
            return;
        }

        // Spherical back-projection
        float latitude  = -std::asin(tmp.y / depth);
        float longitude =  std::atan2(tmp.x, tmp.z);
        
        point.x = (longitude / (2.0f * M_PI)) * static_cast<float>(camera.width) + camera.params[1];
        point.y = (-latitude / M_PI) * static_cast<float>(camera.height) + camera.params[2];

    } else { // pin-hole
        depth = tmp.z;
        if (depth < 1e-6) {
            point.x = camera.K[2];
            point.y = camera.K[5];
            return;
        }
        point.x = (camera.K[0]*tmp.x + camera.K[1]*tmp.y + camera.K[2]*tmp.z) / depth;
        point.y = (camera.K[3]*tmp.x + camera.K[4]*tmp.y + camera.K[5]*tmp.z) / depth;
    }
}

float GetAngle( const cv::Vec3f &v1, const cv::Vec3f &v2 )
{
    float dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
    dot_product = std::fmax(-1.0f, std::fmin(1.0f, dot_product));
    float angle = acosf(dot_product);

    return angle;
}


#include "CompressedDMB.h"

// ============================================================================
// REPLACE: readDepthDmb - Now supports both formats
// ============================================================================

int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth)
{
    // Try compressed format first
    if (CompressedDMB::isCompressedFormat(file_path)) {
        return CompressedDMB::readDepthCompressed(file_path, depth);
    }
    
    // Also check if .cdmb version exists when .dmb is requested
    if (file_path.length() > 4 && file_path.substr(file_path.length() - 4) == ".dmb") {
        std::string cdmb_path = file_path.substr(0, file_path.length() - 4) + ".cdmb";
        FILE* test = fopen(cdmb_path.c_str(), "rb");
        if (test) {
            fclose(test);
            return CompressedDMB::readDepthCompressed(cdmb_path, depth);
        }
    }
    
    // Fall back to original DMB format
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

// ============================================================================
// REPLACE: writeDepthDmb - Now writes compressed format
// ============================================================================

int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth)
{
    // Convert .dmb extension to .cdmb for compressed output
    std::string output_path = file_path;
    if (file_path.length() > 4 && file_path.substr(file_path.length() - 4) == ".dmb") {
        output_path = file_path.substr(0, file_path.length() - 4) + ".cdmb";
    } else if (file_path.length() <= 5 || file_path.substr(file_path.length() - 5) != ".cdmb") {
        output_path = file_path + ".cdmb";
    }
    
    return CompressedDMB::writeDepthCompressed(output_path, depth);
}

// ============================================================================
// REPLACE: readNormalDmb - Now supports both formats
// ============================================================================

int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal)
{
    // Try compressed format first
    if (CompressedDMB::isCompressedFormat(file_path)) {
        return CompressedDMB::readNormalCompressed(file_path, normal);
    }
    
    // Also check if .cdmb version exists when .dmb is requested
    if (file_path.length() > 4 && file_path.substr(file_path.length() - 4) == ".dmb") {
        std::string cdmb_path = file_path.substr(0, file_path.length() - 4) + ".cdmb";
        FILE* test = fopen(cdmb_path.c_str(), "rb");
        if (test) {
            fclose(test);
            return CompressedDMB::readNormalCompressed(cdmb_path, normal);
        }
    }
    
    // Fall back to original DMB format
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

// ============================================================================
// REPLACE: writeNormalDmb - Now writes compressed format
// ============================================================================

int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal)
{
    // Convert .dmb extension to .cdmb for compressed output
    std::string output_path = file_path;
    if (file_path.length() > 4 && file_path.substr(file_path.length() - 4) == ".dmb") {
        output_path = file_path.substr(0, file_path.length() - 4) + ".cdmb";
    } else if (file_path.length() <= 5 || file_path.substr(file_path.length() - 5) != ".cdmb") {
        output_path = file_path + ".cdmb";
    }
    
    return CompressedDMB::writeNormalCompressed(output_path, normal);
}

// ============================================================================
// NEW FUNCTION: writeCostDmb - For cost maps (was using writeDepthDmb before)
// ============================================================================

int writeCostDmb(const std::string file_path, const cv::Mat_<float> cost)
{
    // Convert .dmb extension to .cdmb for compressed output
    std::string output_path = file_path;
    if (file_path.length() > 4 && file_path.substr(file_path.length() - 4) == ".dmb") {
        output_path = file_path.substr(0, file_path.length() - 4) + ".cdmb";
    } else if (file_path.length() <= 5 || file_path.substr(file_path.length() - 5) != ".cdmb") {
        output_path = file_path + ".cdmb";
    }
    
    return CompressedDMB::writeCostCompressed(output_path, cost);
}

// ============================================================================
// NEW FUNCTION: readCostDmb - For cost maps
// ============================================================================

int readCostDmb(const std::string file_path, cv::Mat_<float> &cost)
{
    // Try compressed format first
    if (CompressedDMB::isCompressedFormat(file_path)) {
        return CompressedDMB::readCostCompressed(file_path, cost);
    }
    
    // Also check if .cdmb version exists when .dmb is requested
    if (file_path.length() > 4 && file_path.substr(file_path.length() - 4) == ".dmb") {
        std::string cdmb_path = file_path.substr(0, file_path.length() - 4) + ".cdmb";
        FILE* test = fopen(cdmb_path.c_str(), "rb");
        if (test) {
            fclose(test);
            return CompressedDMB::readCostCompressed(cdmb_path, cost);
        }
    }
    
    // Fall back to reading as depth (original behavior - costs used depth format)
    return readDepthDmb(file_path, cost);
}



void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc)
{
    std::cout << "store 3D points to ply file" << std::endl;

    FILE *outputPly;
    outputPly=fopen(plyFilePath.c_str(), "wb");

    /*write header*/
    fprintf(outputPly, "ply\n");
    fprintf(outputPly, "format binary_little_endian 1.0\n");
    fprintf(outputPly, "element vertex %d\n",pc.size());
    fprintf(outputPly, "property float x\n");
    fprintf(outputPly, "property float y\n");
    fprintf(outputPly, "property float z\n");
    fprintf(outputPly, "property float nx\n");
    fprintf(outputPly, "property float ny\n");
    fprintf(outputPly, "property float nz\n");
    fprintf(outputPly, "property uchar red\n");
    fprintf(outputPly, "property uchar green\n");
    fprintf(outputPly, "property uchar blue\n");
    fprintf(outputPly, "end_header\n");

    //write data
#pragma omp parallel for
    for(size_t i = 0; i < pc.size(); i++) {
        const PointList &p = pc[i];
        float3 X = p.coord;
        const float3 normal = p.normal;
        const float3 color = p.color;
        const char b_color = (int)color.x;
        const char g_color = (int)color.y;
        const char r_color = (int)color.z;

        if(!(X.x < FLT_MAX && X.x > -FLT_MAX) || !(X.y < FLT_MAX && X.y > -FLT_MAX) || !(X.z < FLT_MAX && X.z >= -FLT_MAX)){
            X.x = 0.0f;
            X.y = 0.0f;
            X.z = 0.0f;
        }
#pragma omp critical
        {
            fwrite(&X.x,      sizeof(X.x), 1, outputPly);
            fwrite(&X.y,      sizeof(X.y), 1, outputPly);
            fwrite(&X.z,      sizeof(X.z), 1, outputPly);
            fwrite(&normal.x, sizeof(normal.x), 1, outputPly);
            fwrite(&normal.y, sizeof(normal.y), 1, outputPly);
            fwrite(&normal.z, sizeof(normal.z), 1, outputPly);
            fwrite(&r_color,  sizeof(char), 1, outputPly);
            fwrite(&g_color,  sizeof(char), 1, outputPly);
            fwrite(&b_color,  sizeof(char), 1, outputPly);
        }

    }
    fclose(outputPly);
}

static float GetDisparity(const Camera &camera, const int2 &p, const float &depth)
{
    if (camera.model == SPHERE)
        return depth;   // already radial distance
    float point3D[3];
    point3D[0] = depth * (p.x - camera.K[2]) / camera.K[0];
    point3D[1] = depth * (p.y - camera.K[5]) / camera.K[4];
    point3D[2] = depth;

    return std::sqrt(point3D[0] * point3D[0] + point3D[1] * point3D[1] + point3D[2] * point3D[2]);
}

void ACMMP::SetGeomConsistencyParams(bool multi_geometry=false)
{
    params.geom_consistency = true;
    params.max_iterations = 2;
    if (multi_geometry) {
       params.multi_geometry = true;
    }
}

void ACMMP::SetHierarchyParams()
{
    params.hierarchy = true;
}

void ACMMP::SetPlanarPriorParams()
{
    params.planar_prior = true;
}

static int readDepthAuto(const std::string& base_path, cv::Mat_<float>& depth) {
    // Try compressed format first (.cdmb)
    std::string cdmb_path = base_path;
    if (cdmb_path.length() > 4 && cdmb_path.substr(cdmb_path.length() - 4) == ".dmb") {
        cdmb_path = cdmb_path.substr(0, cdmb_path.length() - 4) + ".cdmb";
    }
    
    if (CompressedDMB::readDepthCompressed(cdmb_path, depth) == 0) {
        return 0;
    }
    
    // Fall back to original format (.dmb)
    return readDepthDmb(base_path, depth);
}

static int readNormalAuto(const std::string& base_path, cv::Mat_<cv::Vec3f>& normal) {
    // Try compressed format first (.cdmb)
    std::string cdmb_path = base_path;
    if (cdmb_path.length() > 4 && cdmb_path.substr(cdmb_path.length() - 4) == ".dmb") {
        cdmb_path = cdmb_path.substr(0, cdmb_path.length() - 4) + ".cdmb";
    }
    
    if (CompressedDMB::readNormalCompressed(cdmb_path, normal) == 0) {
        return 0;
    }
    
    // Fall back to original format (.dmb)
    return readNormalDmb(base_path, normal);
}

static int readCostAuto(const std::string& base_path, cv::Mat_<float>& cost) {
    // Try compressed format first (.cdmb)
    std::string cdmb_path = base_path;
    if (cdmb_path.length() > 4 && cdmb_path.substr(cdmb_path.length() - 4) == ".dmb") {
        cdmb_path = cdmb_path.substr(0, cdmb_path.length() - 4) + ".cdmb";
    }
    
    if (CompressedDMB::readCostCompressed(cdmb_path, cost) == 0) {
        return 0;
    }
    
    // Fall back to reading as depth (original format used depth I/O for costs)
    return readDepthDmb(base_path, cost);
}

void ACMMP::InputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx)
{
    images.clear();
    cameras.clear();
    masks.clear();  // Clear masks
    has_masks_ = false;  // Reset mask flag
    const Problem problem = problems[idx];

    std::string image_folder = dense_folder + std::string("/images");
    std::string cam_folder = dense_folder + std::string("/cams");
    std::string mask_folder = dense_folder + std::string("/masks");  // Mask folder
    
    // Check if mask folder exists
    struct stat mask_stat;
    bool mask_folder_exists = (stat(mask_folder.c_str(), &mask_stat) == 0 && S_ISDIR(mask_stat.st_mode));

    std::stringstream image_path;
    image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".png";
    cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
    cv::Mat image_float;
    image_uint.convertTo(image_float, CV_32FC1);
    images.push_back(image_float);
    
    // Load reference mask if available
    if (mask_folder_exists) {
        std::stringstream mask_path;
        mask_path << mask_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << ".png";
        cv::Mat mask_img = cv::imread(mask_path.str(), cv::IMREAD_GRAYSCALE);
        if (!mask_img.empty()) {
            cv::Mat mask_float;
            mask_img.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);  // Normalize to 0-1
            masks.push_back(mask_float);
            has_masks_ = true;
        } else {
            // Create all-ones mask if file not found
            cv::Mat mask_float = cv::Mat::ones(image_float.rows, image_float.cols, CV_32FC1);
            masks.push_back(mask_float);
        }
    }
    
    std::stringstream cam_path;

    cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.ref_image_id << "_cam.txt";
    Camera camera = ReadCamera(cam_path.str());
    camera.height = image_float.rows;
    camera.width = image_float.cols;
    cameras.push_back(camera);

    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        std::stringstream image_path;
        image_path << image_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << ".png";
        cv::Mat_<uint8_t> image_uint = cv::imread(image_path.str(), cv::IMREAD_GRAYSCALE);
        cv::Mat image_float;
        image_uint.convertTo(image_float, CV_32FC1);
        images.push_back(image_float);
        
        // Load source mask if available
        if (mask_folder_exists) {
            std::stringstream mask_path;
            mask_path << mask_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << ".png";
            cv::Mat mask_img = cv::imread(mask_path.str(), cv::IMREAD_GRAYSCALE);
            if (!mask_img.empty()) {
                cv::Mat mask_float;
                mask_img.convertTo(mask_float, CV_32FC1, 1.0 / 255.0);
                masks.push_back(mask_float);
            } else {
                cv::Mat mask_float = cv::Mat::ones(image_float.rows, image_float.cols, CV_32FC1);
                masks.push_back(mask_float);
            }
        }
        
        std::stringstream cam_path;
        cam_path << cam_folder << "/" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i] << "_cam.txt";
        Camera camera = ReadCamera(cam_path.str());
        camera.height = image_float.rows;
        camera.width = image_float.cols;
        cameras.push_back(camera);
    }


    // Fix C: Build image_id → problem_index map for safe source image scaling lookup
    std::unordered_map<int, int> id_to_idx;
    for (size_t pi = 0; pi < problems.size(); pi++)
        id_to_idx[problems[pi].ref_image_id] = (int)pi;

    // Scale cameras and images (and masks if present)
    int max_image_size = problems[idx].cur_image_size;
    for (size_t i = 0; i < images.size(); ++i) {
        if (i > 0) {
            int src_id = problem.src_image_ids[i - 1];
            auto it = id_to_idx.find(src_id);
            max_image_size = (it != id_to_idx.end()) ? problems[it->second].cur_image_size : problems[idx].cur_image_size;
        }

        if (images[i].cols <= max_image_size && images[i].rows <= max_image_size) {
            continue;
        }

        const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        cv::Mat_<float> scaled_image_float;
        cv::resize(images[i], scaled_image_float, cv::Size(new_cols,new_rows), 0, 0, cv::INTER_LINEAR);
        images[i] = scaled_image_float.clone();

        // Scale mask if present (use INTER_NEAREST to preserve binary values)
        if (has_masks_ && i < masks.size()) {
            cv::Mat scaled_mask;
            cv::resize(masks[i], scaled_mask, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_NEAREST);
            masks[i] = scaled_mask.clone();
        }

        if (cameras[i].model == SPHERE) {
            cameras[i].params[1] *= scale_x;
            cameras[i].params[2] *= scale_y;
        } else {
            cameras[i].K[0] *= scale_x;
            cameras[i].K[2] *= scale_x;
            cameras[i].K[4] *= scale_y;
            cameras[i].K[5] *= scale_y;
        }
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }


    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    params.num_images = (int)images.size();
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

    if (params.geom_consistency) {
        depths.clear();

        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        cv::Mat_<float> ref_depth;

        // Use auto-detection function
        readDepthAuto(depth_path, ref_depth);
        depths.push_back(ref_depth);

        size_t num_src_images = problem.src_image_ids.size();
        for (size_t i = 0; i < num_src_images; ++i) {
            std::stringstream result_path;
            result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string result_folder = result_path.str();
            std::string depth_path = result_folder + suffix;
            cv::Mat_<float> depth;

            // Use auto-detection function
            readDepthAuto(depth_path, depth);
            depths.push_back(depth);
        }
    }
}


void ACMMP::InputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx, ImageCache& cache)
{
    images.clear();
    cameras.clear();
    masks.clear();
    has_masks_ = false;
    const Problem problem = problems[idx];

    // Build id → problem_index map for source image scaling
    std::unordered_map<int, int> id_to_idx;
    for (size_t pi = 0; pi < problems.size(); pi++)
        id_to_idx[problems[pi].ref_image_id] = (int)pi;

    // Load ref image from cache
    auto ref_entry = cache.get(problem.ref_image_id);
    images.push_back(ref_entry->image_float.clone());
    cameras.push_back(ref_entry->camera);
    if (ref_entry->has_mask) {
        masks.push_back(ref_entry->mask_float.clone());
        has_masks_ = true;
    }

    // Load source images from cache
    size_t num_src_images = problem.src_image_ids.size();
    for (size_t i = 0; i < num_src_images; ++i) {
        auto src_entry = cache.get(problem.src_image_ids[i]);
        images.push_back(src_entry->image_float.clone());
        if (src_entry->has_mask) {
            masks.push_back(src_entry->mask_float.clone());
        } else if (has_masks_) {
            cv::Mat mask_float = cv::Mat::ones(src_entry->image_float.rows, src_entry->image_float.cols, CV_32FC1);
            masks.push_back(mask_float);
        }
        cameras.push_back(src_entry->camera);
    }

    // Scale cameras and images (same logic as original)
    int max_image_size = problems[idx].cur_image_size;
    for (size_t i = 0; i < images.size(); ++i) {
        if (i > 0) {
            int src_id = problem.src_image_ids[i - 1];
            auto it = id_to_idx.find(src_id);
            max_image_size = (it != id_to_idx.end()) ? problems[it->second].cur_image_size : problems[idx].cur_image_size;
        }

        if (images[i].cols <= max_image_size && images[i].rows <= max_image_size) {
            continue;
        }

        const float factor_x = static_cast<float>(max_image_size) / images[i].cols;
        const float factor_y = static_cast<float>(max_image_size) / images[i].rows;
        const float factor = std::min(factor_x, factor_y);

        const int new_cols = std::round(images[i].cols * factor);
        const int new_rows = std::round(images[i].rows * factor);

        const float scale_x = new_cols / static_cast<float>(images[i].cols);
        const float scale_y = new_rows / static_cast<float>(images[i].rows);

        cv::Mat_<float> scaled_image_float;
        cv::resize(images[i], scaled_image_float, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_LINEAR);
        images[i] = scaled_image_float.clone();

        if (has_masks_ && i < masks.size()) {
            cv::Mat scaled_mask;
            cv::resize(masks[i], scaled_mask, cv::Size(new_cols, new_rows), 0, 0, cv::INTER_NEAREST);
            masks[i] = scaled_mask.clone();
        }

        if (cameras[i].model == SPHERE) {
            cameras[i].params[1] *= scale_x;
            cameras[i].params[2] *= scale_y;
        } else {
            cameras[i].K[0] *= scale_x;
            cameras[i].K[2] *= scale_x;
            cameras[i].K[4] *= scale_y;
            cameras[i].K[5] *= scale_y;
        }
        cameras[i].height = scaled_image_float.rows;
        cameras[i].width = scaled_image_float.cols;
    }

    params.depth_min = cameras[0].depth_min * 0.6f;
    params.depth_max = cameras[0].depth_max * 1.2f;
    params.num_images = (int)images.size();
    params.disparity_min = cameras[0].K[0] * params.baseline / params.depth_max;
    params.disparity_max = cameras[0].K[0] * params.baseline / params.depth_min;

    if (params.geom_consistency) {
        depths.clear();

        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        cv::Mat_<float> ref_depth;
        readDepthAuto(depth_path, ref_depth);
        depths.push_back(ref_depth);

        for (size_t i = 0; i < num_src_images; ++i) {
            std::stringstream src_result_path;
            src_result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.src_image_ids[i];
            std::string src_result_folder = src_result_path.str();
            std::string src_depth_path = src_result_folder + suffix;
            cv::Mat_<float> depth;
            readDepthAuto(src_depth_path, depth);
            depths.push_back(depth);
        }
    }
}

void ACMMP::CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem, ProblemGPUResources* res)
{
    num_images = (int)images.size();
    cudaStream_t s = stream_ ? stream_ : 0;

    // Guard: num_images must not exceed allocated slots
    if (res->allocated_images > 0 && num_images > res->allocated_images) {
        std::cerr << "ERROR: num_images (" << num_images << ") > allocated_images ("
                  << res->allocated_images << "). Clamping." << std::endl;
        num_images = res->allocated_images;
    }

    for (int i = 0; i < num_images; ++i) {
        int rows = images[i].rows;
        int cols = images[i].cols;

        // Always copy image data to array
        CUDA_CHECK(cudaMemcpy2DToArrayAsync(res->cuArray[i], 0, 0, images[i].ptr<float>(), images[i].step[0], cols * sizeof(float), rows, cudaMemcpyHostToDevice, s));

        // Skip texture creation if already created in allocate() (batch path)
        if (!res->textures_created) {
            struct cudaResourceDesc resDesc;
            memset(&resDesc, 0, sizeof(cudaResourceDesc));
            resDesc.resType = cudaResourceTypeArray;
            resDesc.res.array.array = res->cuArray[i];

            struct cudaTextureDesc texDesc;
            memset(&texDesc, 0, sizeof(cudaTextureDesc));
            texDesc.addressMode[0] = cudaAddressModeClamp;
            texDesc.addressMode[1] = cudaAddressModeClamp;
            texDesc.filterMode = cudaFilterModeLinear;
            texDesc.readMode = cudaReadModeElementType;
            texDesc.normalizedCoords = 0;

            if (res->texture_objects_host.images[i] != 0) {
                cudaDestroyTextureObject(res->texture_objects_host.images[i]);
            }
            CUDA_CHECK(cudaCreateTextureObject(&(res->texture_objects_host.images[i]), &resDesc, &texDesc, NULL));
        }
    }

    // Upload texture handles only if sequential path created them
    if (!res->textures_created) {
        CUDA_CHECK(cudaMemcpyAsync(res->texture_objects_cuda, &res->texture_objects_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice, s));
    }
    // ALWAYS upload cameras (per-problem data)
    CUDA_CHECK(cudaMemcpyAsync(res->cameras_cuda, &cameras[0], sizeof(Camera) * num_images, cudaMemcpyHostToDevice, s));

    // Item 4: Upload reference mask to GPU
    if (has_masks_ && !masks.empty() && res->ref_mask_cuda) {
        const int w = cameras[0].width, h = cameras[0].height;
        std::vector<uint8_t> mask_u8(w * h);
        for (int y = 0; y < h; y++)
            for (int x = 0; x < w; x++)
                mask_u8[y * w + x] = (masks[0].at<float>(y, x) > 0.5f) ? 1 : 0;
        CUDA_CHECK(cudaMemcpyAsync(res->ref_mask_cuda, mask_u8.data(),
            w * h, cudaMemcpyHostToDevice, s));
        params.has_mask = true;
    } else {
        params.has_mask = false;
    }

    if (!batch_mode_) {
        plane_hypotheses_host = new float4[cameras[0].height * cameras[0].width];
        costs_host = new float[cameras[0].height * cameras[0].width];
    }

    if (params.geom_consistency) {
        for (int i = 0; i < num_images; ++i) {
            int rows = depths[i].rows;
            int cols = depths[i].cols;

            CUDA_CHECK(cudaMemcpy2DToArrayAsync(res->cuDepthArray[i], 0, 0, depths[i].ptr<float>(), depths[i].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice, s));

            if (!res->textures_created) {
                struct cudaResourceDesc resDesc;
                memset(&resDesc, 0, sizeof(cudaResourceDesc));
                resDesc.resType = cudaResourceTypeArray;
                resDesc.res.array.array = res->cuDepthArray[i];

                struct cudaTextureDesc texDesc;
                memset(&texDesc, 0, sizeof(cudaTextureDesc));
                texDesc.addressMode[0] = cudaAddressModeClamp;
                texDesc.addressMode[1] = cudaAddressModeClamp;
                texDesc.filterMode = cudaFilterModeLinear;
                texDesc.readMode  = cudaReadModeElementType;
                texDesc.normalizedCoords = 0;

                if (res->texture_depths_host.images[i] != 0) {
                    cudaDestroyTextureObject(res->texture_depths_host.images[i]);
                }
                CUDA_CHECK(cudaCreateTextureObject(&(res->texture_depths_host.images[i]), &resDesc, &texDesc, NULL));
            }
        }
        if (!res->textures_created) {
            CUDA_CHECK(cudaMemcpyAsync(res->texture_depths_cuda, &res->texture_depths_host, sizeof(cudaTextureObjects), cudaMemcpyHostToDevice, s));
        }

        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;

        std::string result_folder = result_path.str();
        std::string suffix = "/depths.dmb";
        if (params.multi_geometry) {
            suffix = "/depths_geom.dmb";
        }
        std::string depth_path = result_folder + suffix;
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";
        
        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        
        // Use auto-detection functions that try compressed first
        readDepthAuto(depth_path, ref_depth);
        readNormalAuto(normal_path, ref_normal);
        readCostAuto(cost_path, ref_cost);
        
        int width = ref_depth.cols;
        int height = ref_depth.rows;
        std::vector<float4> initial_planes(width * height);
        std::vector<float> initial_costs(width * height);
        
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                int center = y * width + x;
                initial_planes[center].x = ref_normal(y, x)[0];
                initial_planes[center].y = ref_normal(y, x)[1];
                initial_planes[center].z = ref_normal(y, x)[2];
                initial_planes[center].w = ref_depth(y, x);
                initial_costs[center] = ref_cost(y, x);
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(res->plane_hypotheses_cuda, initial_planes.data(), sizeof(float4) * width * height, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(res->costs_cuda, initial_costs.data(), sizeof(float) * width * height, cudaMemcpyHostToDevice, s));
    }

    if (params.hierarchy) {
        std::stringstream result_path;
        result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
        std::string result_folder = result_path.str();
        std::string depth_path = result_folder + "/depths.dmb";
        std::string normal_path = result_folder + "/normals.dmb";
        std::string cost_path = result_folder + "/costs.dmb";

        cv::Mat_<float> ref_depth;
        cv::Mat_<cv::Vec3f> ref_normal;
        cv::Mat_<float> ref_cost;
        
        // Use auto-detection functions that try compressed first
        readDepthAuto(depth_path, ref_depth);
        readNormalAuto(normal_path, ref_normal);
        readCostAuto(cost_path, ref_cost);
        
        int width = ref_normal.cols;
        int height = ref_normal.rows;
        
        scaled_plane_hypotheses_host = new float4[height * width];
        pre_costs_host = new float[height * width];

        if (width != images[0].cols || height != images[0].rows) {
            params.upsample = true;
            params.scaled_cols = width;
            params.scaled_rows = height;
        }
        else {
            params.upsample = false;
        }

        for (int row = 0; row < height; ++row) {
            for (int col = 0; col < width; ++col) {
                int center = row * width + col;
                float4 plane_hypothesis;
                plane_hypothesis.x = ref_normal(row, col)[0];
                plane_hypothesis.y = ref_normal(row, col)[1];
                plane_hypothesis.z = ref_normal(row, col)[2];
                plane_hypothesis.w = params.upsample ? ref_cost(row, col) : ref_depth(row, col);
                scaled_plane_hypotheses_host[center] = plane_hypothesis;
            }
        }

        std::vector<float4> initial_planes_for_current_res(cameras[0].width * cameras[0].height);
        for (int row = 0; row < cameras[0].height; ++row) {
            for (int col = 0; col < cameras[0].width; ++col) {
                int center = row * cameras[0].width + col;
                initial_planes_for_current_res[center].w = ref_depth(row, col);
            }
        }
        
        CUDA_CHECK(cudaMemcpyAsync(res->scaled_plane_hypotheses_cuda, scaled_plane_hypotheses_host, sizeof(float4) * height * width, cudaMemcpyHostToDevice, s));
        CUDA_CHECK(cudaMemcpyAsync(res->plane_hypotheses_cuda, initial_planes_for_current_res.data(), sizeof(float4) * cameras[0].width * cameras[0].height, cudaMemcpyHostToDevice, s));
    }
}

void ACMMP::CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks, ProblemGPUResources* res)
{
    const int w = cameras[0].width;
    const int h = cameras[0].height;
    const int n = w * h;
    cudaStream_t s = stream_ ? stream_ : 0;

    prior_planes_host = new float4[n];
    plane_masks_host = new unsigned int[n];

    for (int i = 0; i < w; ++i) {
        for (int j = 0; j < h; ++j) {
            int center = j * w + i;
            plane_masks_host[center] = (unsigned int)masks(j, i);
            if (masks(j, i) > 0) {
                prior_planes_host[center] = PlaneParams[masks(j, i) - 1];
            }
        }
    }

    // Fix B: Write to resource pool buffers if available, else class-local
    if (res) {
        cudaMemcpyAsync(res->prior_planes_cuda, prior_planes_host, sizeof(float4) * n, cudaMemcpyHostToDevice, s);
        cudaMemcpyAsync(res->plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * n, cudaMemcpyHostToDevice, s);
    } else {
        cudaMalloc((void**)&prior_planes_cuda, sizeof(float4) * n);
        cudaMalloc((void**)&plane_masks_cuda, sizeof(unsigned int) * n);
        cudaMemcpy(prior_planes_cuda, prior_planes_host, sizeof(float4) * n, cudaMemcpyHostToDevice);
        cudaMemcpy(plane_masks_cuda, plane_masks_host, sizeof(unsigned int) * n, cudaMemcpyHostToDevice);
    }
}

int ACMMP::GetReferenceImageWidth()
{
    return cameras[0].width;
}

int ACMMP::GetReferenceImageHeight()
{
    return cameras[0].height;
}

cv::Mat ACMMP::GetReferenceImage()
{
    return images[0];
}

cv::Mat ACMMP::GetReferenceMask()
{
    if (has_masks_ && !masks.empty()) {
        return masks[0];
    }
    // Return all-ones mask if no masks available
    return cv::Mat::ones(cameras[0].height, cameras[0].width, CV_32FC1);
}

float4 ACMMP::GetPlaneHypothesis(const int index)
{
    return plane_hypotheses_host[index];
}

float ACMMP::GetCost(const int index)
{
    return costs_host[index];
}

float ACMMP::GetMinDepth()
{
    return params.depth_min;
}

float ACMMP::GetMaxDepth()
{
    return params.depth_max;
}

void ACMMP::GetSupportPoints(std::vector<cv::Point>& support2DPoints)
{
    support2DPoints.clear();
    const int step_size = 5;
    const int width = GetReferenceImageWidth();
    const int height = GetReferenceImageHeight();
    for (int col = 0; col < width; col += step_size) {
        for (int row = 0; row < height; row += step_size) {
            float min_cost = 2.0f;
            cv::Point temp_point;
            int c_bound = std::min(width, col + step_size);
            int r_bound = std::min(height, row + step_size);
            for (int c = col; c < c_bound; ++c) {
                for (int r = row; r < r_bound; ++r) {
                    int center = r * width + c;
                    if (GetCost(center) < 2.0f && min_cost > GetCost(center)) {
                        temp_point = cv::Point(c, r);
                        min_cost = GetCost(center);
                    }
                }
            }
            if (min_cost < 0.1f) {
                support2DPoints.push_back(temp_point);
            }
        }
    }
}

std::vector<Triangle> ACMMP::DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points)
{
    if (points.empty()) {
        return std::vector<Triangle>();
    }

    std::vector<Triangle> results;

    std::vector<cv::Vec6f> temp_results;
    cv::Subdiv2D subdiv2d(boundRC);
    for (const auto point : points) {
        subdiv2d.insert(cv::Point2f((float)point.x, (float)point.y));
    }
    subdiv2d.getTriangleList(temp_results);

    for (const auto temp_vec : temp_results) {
        cv::Point pt1((int)temp_vec[0], (int)temp_vec[1]);
        cv::Point pt2((int)temp_vec[2], (int)temp_vec[3]);
        cv::Point pt3((int)temp_vec[4], (int)temp_vec[5]);
        results.push_back(Triangle(pt1, pt2, pt3));
    }
    return results;
}

float4 ACMMP::GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths)
{
    cv::Mat A(3, 4, CV_32FC1);
    cv::Mat B(4, 1, CV_32FC1);

    float3 ptX1 = Get3DPointonRefCam(triangle.pt1.x, triangle.pt1.y, depths(triangle.pt1.y, triangle.pt1.x), cameras[0]);
    float3 ptX2 = Get3DPointonRefCam(triangle.pt2.x, triangle.pt2.y, depths(triangle.pt2.y, triangle.pt2.x), cameras[0]);
    float3 ptX3 = Get3DPointonRefCam(triangle.pt3.x, triangle.pt3.y, depths(triangle.pt3.y, triangle.pt3.x), cameras[0]);

    A.at<float>(0, 0) = ptX1.x;
    A.at<float>(0, 1) = ptX1.y;
    A.at<float>(0, 2) = ptX1.z;
    A.at<float>(0, 3) = 1.0;
    A.at<float>(1, 0) = ptX2.x;
    A.at<float>(1, 1) = ptX2.y;
    A.at<float>(1, 2) = ptX2.z;
    A.at<float>(1, 3) = 1.0;
    A.at<float>(2, 0) = ptX3.x;
    A.at<float>(2, 1) = ptX3.y;
    A.at<float>(2, 2) = ptX3.z;
    A.at<float>(2, 3) = 1.0;
    cv::SVD::solveZ(A, B);
    float4 n4 = make_float4(B.at<float>(0, 0), B.at<float>(1, 0), B.at<float>(2, 0), B.at<float>(3, 0));
    float3 n = make_float3(n4.x, n4.y, n4.z);
    float  nn = std::sqrt(n.x*n.x + n.y*n.y + n.z*n.z);
    if (nn > 1e-12f) { n.x/=nn; n.y/=nn; n.z/=nn; n4.w/=nn; }
    n4.x = n.x; n4.y = n.y; n4.z = n.z;

    // Ensure positive depth along the triangle centroid ray
    const int cx = (triangle.pt1.x + triangle.pt2.x + triangle.pt3.x) / 3;
    const int cy = (triangle.pt1.y + triangle.pt2.y + triangle.pt3.y) / 3;

    float lon = (float(cx) - cameras[0].params[1]) / float(cameras[0].width)  * 2.0f * M_PI;
    float lat = -(float(cy) - cameras[0].params[2]) / float(cameras[0].height) * M_PI;
    float3 dir = make_float3(std::cos(lat)*std::sin(lon),
                            -std::sin(lat),
                            std::cos(lat)*std::cos(lon));
    const float denom = n4.x*dir.x + n4.y*dir.y + n4.z*dir.z;
    if (-n4.w / denom < 0.0f) { n4.x = -n4.x; n4.y = -n4.y; n4.z = -n4.z; n4.w = -n4.w; }

    return n4;}

    
float ACMMP::GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y)
{
    if (cameras[0].model == SPHERE) {
        // Convert pixel to ray direction
        float lon = (static_cast<float>(x) - cameras[0].params[1]) / static_cast<float>(cameras[0].width) * 2.0f * M_PI;
        float lat = -(static_cast<float>(y) - cameras[0].params[2]) / static_cast<float>(cameras[0].height) * M_PI;
        
        float3 dir = make_float3(
            std::cos(lat) * std::sin(lon),
            -std::sin(lat),
            std::cos(lat) * std::cos(lon)
        );
        
        // Compute intersection with plane
        float denom = plane_hypothesis.x * dir.x + plane_hypothesis.y * dir.y + plane_hypothesis.z * dir.z;
        return (std::abs(denom) < 1e-6f) ? 1e6f : (-plane_hypothesis.w / denom);
    } else {
        // Original pinhole logic
        return -plane_hypothesis.w * cameras[0].K[0] / ((x - cameras[0].K[2]) * plane_hypothesis.x + (cameras[0].K[0] / cameras[0].K[4]) * (y - cameras[0].K[5]) * plane_hypothesis.y + cameras[0].K[0] * plane_hypothesis.z);
    }
}

 void JBUAddImageToTextureFloatGray ( std::vector<cv::Mat_<float>>  &imgs, cudaTextureObject_t texs[], cudaArray *cuArray[], const int &numSelViews, cudaStream_t s = 0)
{
    for (int i=0; i<numSelViews; i++) {
        int index = i;
        int rows = imgs[index].rows;
        int cols = imgs[index].cols;
        // Create channel with floating point type
        cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc (32, 0, 0,  0, cudaChannelFormatKindFloat);
        // Allocate array with correct size and number of channels
        cudaMallocArray(&cuArray[i], &channelDesc, cols, rows);
        cudaMemcpy2DToArrayAsync(cuArray[i], 0, 0, imgs[index].ptr<float>(), imgs[index].step[0], cols*sizeof(float), rows, cudaMemcpyHostToDevice, s);

        // Specify texture
        struct cudaResourceDesc resDesc;
        memset(&resDesc, 0, sizeof(cudaResourceDesc));
        resDesc.resType         = cudaResourceTypeArray;
        resDesc.res.array.array = cuArray[i];

        // Specify texture object parameters
        struct cudaTextureDesc texDesc;
        memset(&texDesc, 0, sizeof(cudaTextureDesc));
        texDesc.addressMode[0]   = cudaAddressModeWrap;
        texDesc.addressMode[1]   = cudaAddressModeClamp;
        texDesc.filterMode       = cudaFilterModeLinear;
        texDesc.readMode         = cudaReadModeElementType;
        texDesc.normalizedCoords = 0;

        // Create texture object
        cudaCreateTextureObject(&(texs[i]), &resDesc, &texDesc, NULL);
    }
    return;
}

 JBU::JBU(){}

 JBU::~JBU()
 {
     free(depth_h);

     cudaFree(depth_d);
     cudaFree(jp_d);
     cudaFree(jt_d);
 }

 void JBU::InitializeParameters(int n)
 {
     depth_h = (float*)malloc(sizeof(float) * n);

     cudaMalloc ((void**)&depth_d,  sizeof(float) * n);

     cudaMalloc((void**)&jp_d, sizeof(JBUParameters) * 1);
     cudaMemcpyAsync(jp_d, &jp_h, sizeof(JBUParameters) * 1, cudaMemcpyHostToDevice, stream_);

     cudaMalloc((void**)&jt_d, sizeof(JBUTexObj) * 1);
     cudaMemcpyAsync(jt_d, &jt_h, sizeof(JBUTexObj) * 1, cudaMemcpyHostToDevice, stream_);
 }

void RunJBU(const cv::Mat_<float>  &scaled_image_float, const cv::Mat_<float> &src_depthmap, const std::string &dense_folder , const Problem &problem)           
{
    uint32_t rows = scaled_image_float.rows;
    uint32_t cols = scaled_image_float.cols;
    int Imagescale = std::max(scaled_image_float.rows / src_depthmap.rows, scaled_image_float.cols / src_depthmap.cols);

    if (Imagescale == 1) {
        std::cout << "Image.rows = Depthmap.rows" << std::endl;
        return;
    }

    std::vector<cv::Mat_<float> > imgs(JBU_NUM);
    imgs[0] = scaled_image_float.clone();
    imgs[1] = src_depthmap.clone();

    JBU jbu;
    jbu.jp_h.height = rows;
    jbu.jp_h.width = cols;
    jbu.jp_h.s_height = src_depthmap.rows;
    jbu.jp_h.s_width = src_depthmap.cols;
    jbu.jp_h.Imagescale = Imagescale;
    // Create per-call non-blocking stream
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

    JBUAddImageToTextureFloatGray(imgs, jbu.jt_h.imgs, jbu.cuArray, JBU_NUM, s);

    // SetStream BEFORE InitializeParameters (it uses stream_ for async copies)
    jbu.SetStream(s);
    jbu.InitializeParameters(rows * cols);
    jbu.CudaRun();

    cv::Mat_<float> depthmap = cv::Mat::zeros( rows, cols, CV_32FC1 );

    for (uint32_t i = 0; i < cols; ++i) {
        for(uint32_t j = 0; j < rows; ++j) {
            int center = i + cols * j;
            depthmap (j, i) = jbu.depth_h[center];
        }
    }

    cv::Mat_<float> disp0 = depthmap.clone();
    std::stringstream result_path;
    result_path << dense_folder << "/ACMMP" << "/2333_" << std::setw(8) << std::setfill('0') << problem.ref_image_id;
    std::string result_folder = result_path.str();
    mkdir(result_folder.c_str(), 0777);

    // Use compressed format for output
    std::string depth_path = result_folder + "/depths.cdmb";
    CompressedDMB::writeDepthCompressed(depth_path, disp0);

    for (int i=0; i < JBU_NUM; i++) {
        CUDA_SAFE_CALL( cudaDestroyTextureObject(jbu.jt_h.imgs[i]) );
        CUDA_SAFE_CALL( cudaFreeArray(jbu.cuArray[i]) );
    }
    cudaStreamSynchronize(s);
    cudaStreamDestroy(s);
}