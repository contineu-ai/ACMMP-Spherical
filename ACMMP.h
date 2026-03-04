#ifndef _ACMMP_H_
#define _ACMMP_H_

#include "main.h"
class ProblemGPUResources;
int readDepthDmb(const std::string file_path, cv::Mat_<float> &depth);
int readNormalDmb(const std::string file_path, cv::Mat_<cv::Vec3f> &normal);
int writeDepthDmb(const std::string file_path, const cv::Mat_<float> depth);
int writeNormalDmb(const std::string file_path, const cv::Mat_<cv::Vec3f> normal);
int readCostDmb(const std::string file_path, cv::Mat_<float> &cost);
int writeCostDmb(const std::string file_path, const cv::Mat_<float> cost);

// APD binary mat I/O
bool ReadBinMat(const std::string &mat_path, cv::Mat &mat);
bool WriteBinMat(const std::string &mat_path, const cv::Mat &mat);

Camera ReadCamera(const std::string &cam_path);
void  RescaleImageAndCamera(cv::Mat_<cv::Vec3b> &src, cv::Mat_<cv::Vec3b> &dst, cv::Mat_<float> &depth, Camera &camera);
float3 Get3DPointonWorld(const int x, const int y, const float depth, const Camera camera);
void ProjectonCamera(const float3 PointX, const Camera camera, float2 &point, float &depth);
float GetAngle(const cv::Vec3f &v1, const cv::Vec3f &v2);
void StoreColorPlyFileBinaryPointCloud (const std::string &plyFilePath, const std::vector<PointList> &pc);

#define CUDA_SAFE_CALL(error) CudaSafeCall(error, __FILE__, __LINE__)
#define CUDA_CHECK_ERROR() CudaCheckError(__FILE__, __LINE__)

void CudaSafeCall(const cudaError_t error, const std::string& file, const int line);
void CudaCheckError(const char* file, const int line);

struct cudaTextureObjects {
    cudaTextureObject_t images[MAX_IMAGES];
};

// Lightweight counter-based RNG state (4 bytes vs curandState's 48 bytes).
// Uses Philox4x32-10 as a hash function: hash(counter++, pixel_key) → random uint32.
struct RNGState {
    unsigned int counter;
};

// APD data pass helper — unified GPU data access for APD kernels
struct DataPassHelper {
    int width;
    int height;
    int ref_index;
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_depths_cuda;
    Camera *cameras_cuda;
    float4 *plane_hypotheses_cuda;
    RNGState *rand_states_cuda;       // ACMMP lightweight RNG (not curandState)
    unsigned int *selected_views_cuda;
    short2 *neighbours_cuda;
    int *neighbours_map_cuda;
    uchar *weak_info_cuda;
    float *costs_cuda;
    PatchMatchParams *params;
    float4 *fit_plane_hypotheses_cuda;
    uchar *weak_reliable_cuda;
    uchar *view_weight_cuda;
    int view_weight_stride;           // Number of images allocated per pixel in view_weight_cuda
    short2 *weak_nearest_strong;
    uint8_t *ref_mask_cuda;           // Pole/invalid mask
};

struct PatchMatchParams {
    int max_iterations = 3;
    int patch_size = 5;
    int num_images = 5;
    int max_image_size = 3200;
    int radius_increment = 2;
    float sigma_spatial = 5.0f;
    float sigma_color = 3.0f;
    int top_k = 4;
    float baseline = 0.54f;
    float depth_min = 0.0f;
    float depth_max = 1.0f;
    float disparity_min = 0.0f;
    float disparity_max = 1.0f;

    float scaled_cols;
    float scaled_rows;

    bool geom_consistency = false;
    bool planar_prior = false;
    bool multi_geometry = false;
    bool hierarchy = false;
    bool upsample = false;
    bool has_mask = false;

    // APD-specific fields
    int strong_radius = 5;
    int strong_increment = 2;
    int weak_radius = 5;
    int weak_increment = 5;
    bool use_APD = true;
    int weak_peak_radius = 2;
    int rotate_time = 4;
    float ransac_threshold = 0.005f;
    float geom_factor = 0.2f;
    RunState state = FIRST_INIT;
};

class ACMMP {
public:
    ACMMP();
    ~ACMMP();
    void SetStream(cudaStream_t s) { stream_ = s; }
    cudaStream_t GetStream() const { return stream_; }

    // Original ACMMP methods
    void InputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx);
    void InputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx, class ImageCache& cache);
    void CudaSpaceInitialization(const std::string &dense_folder, const Problem &problem, ProblemGPUResources* res);
    void RunPatchMatch(ProblemGPUResources* res, bool skip_host_download = false);
    void Colmap2MVS(const std::string &dense_folder, std::vector<Problem> &problems);
    void SetGeomConsistencyParams(bool multi_geometry);
    void SetPlanarPriorParams();
    void SetHierarchyParams();

    // APD methods
    void APDInputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx);
    void APDInputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx, class ImageCache& cache);
    void APDCudaSpaceInitialization(const std::string &dense_folder, const Problem &problem, ProblemGPUResources* res);
    void RunAPDPatchMatch(ProblemGPUResources* res, bool skip_host_download = false);
    void SetAPDParams(const PatchMatchParams &apd_params);

    int GetReferenceImageWidth();
    int GetReferenceImageHeight();
    cv::Mat GetReferenceImage();
    cv::Mat GetReferenceMask();
    bool HasMasks() const { return has_masks_; }
    void SetBatchMode() { batch_mode_ = true; }
    float4 GetPlaneHypothesis(const int index);
    float GetCost(const int index);
    void GetSupportPoints(std::vector<cv::Point>& support2DPoints);
    std::vector<Triangle> DelaunayTriangulation(const cv::Rect boundRC, const std::vector<cv::Point>& points);
    float4 GetPriorPlaneParams(const Triangle triangle, const cv::Mat_<float> depths);
    float GetDepthFromPlaneParam(const float4 plane_hypothesis, const int x, const int y);
    float GetMinDepth();
    float GetMaxDepth();
    void CudaPlanarPriorInitialization(const std::vector<float4> &PlaneParams, const cv::Mat_<float> &masks, ProblemGPUResources* res = nullptr);

    // APD host-side data accessors
    cv::Mat GetWeakInfo() const { return weak_info_host; }
    cv::Mat GetSelectedViews() const { return selected_views_host; }
    int GetWeakCount() const { return weak_count; }

private:
    cudaStream_t stream_ = 0;
    bool batch_mode_ = false;
    int num_images;
    std::vector<cv::Mat> images;
    std::vector<cv::Mat> depths;
    std::vector<cv::Mat> masks;
    bool has_masks_ = false;
    std::vector<Camera> cameras;

    cudaTextureObjects texture_objects_host;
    cudaTextureObjects texture_depths_host;
    float4 *plane_hypotheses_host = nullptr;
    float4 *scaled_plane_hypotheses_host = nullptr;
    float *costs_host = nullptr;
    float *pre_costs_host = nullptr;
    float4 *prior_planes_host = nullptr;
    unsigned int *plane_masks_host = nullptr;
    PatchMatchParams params;

    Camera *cameras_cuda;
    cudaArray *cuArray[MAX_IMAGES];
    cudaArray *cuDepthArray[MAX_IMAGES];
    cudaArray *cuMaskArray[MAX_IMAGES];
    cudaTextureObjects *texture_objects_cuda;
    cudaTextureObjects *texture_masks_cuda;
    cudaTextureObjects *texture_depths_cuda;
    float4 *plane_hypotheses_cuda;
    float4 *scaled_plane_hypotheses_cuda;
    float *costs_cuda;
    float *pre_costs_cuda;
    RNGState *rand_states_cuda;
    unsigned int *selected_views_cuda;
    float *depths_cuda;
    float4 *prior_planes_cuda = nullptr;
    unsigned int *plane_masks_cuda = nullptr;

    // APD host-side data (not GPU buffers — those live in ProblemGPUResources)
    cv::Mat weak_info_host;
    cv::Mat neighbours_map_host;
    cv::Mat selected_views_host;
    int weak_count = 0;
};

struct TexObj {
    cudaTextureObject_t imgs[MAX_IMAGES];
};

// Host-callable APD kernel launcher
void RunAPDPatchMatch_GPU(DataPassHelper *helper_cuda, int width, int height,
                          const PatchMatchParams &params, cudaStream_t stream);

#endif // _ACMMP_H_
