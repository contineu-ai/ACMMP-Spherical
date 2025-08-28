#ifndef ACMMP_DEVICE_CUH
#define ACMMP_DEVICE_CUH

#include "ACMMP.h"
#include "SphericalLUT_MultiRes.h"
#include <math_constants.h>

// Access to constant memory LUTs
extern __constant__ SphericalLUT* d_lut_array_const[10];
extern __constant__ int d_num_luts;

// Shared memory cache for frequently accessed LUT data
__shared__ float3 shared_dir_cache[256];  // Cache for current block

// Optimized branchless LUT finder
__device__ __forceinline__ SphericalLUT* FindLUTForCamera_Optimized(const Camera& cam, int* lut_index) {
    SphericalLUT* best_lut = nullptr;
    float min_diff = FLT_MAX;
    int best_idx = -1;
    
    // Unroll the loop for better performance with small constant d_num_luts
    #pragma unroll
    for (int i = 0; i < d_num_luts && i < 10; ++i) {
        SphericalLUT* lut = d_lut_array_const[i];
        
        // Branchless exact match check
        bool exact_match = (lut->width == cam.width) && 
                          (lut->height == cam.height) &&
                          (fabsf(lut->cx - cam.params[1]) < 0.1f) && 
                          (fabsf(lut->cy - cam.params[2]) < 0.1f);
        
        if (exact_match) {
            *lut_index = i;
            return lut;  // Early return for exact match
        }
        
        // Calculate difference for closest match using branchless min
        float diff = fabsf(lut->width - cam.width) + fabsf(lut->height - cam.height) +
                    fabsf(lut->cx - cam.params[1]) + fabsf(lut->cy - cam.params[2]);
        
        // Branchless minimum update
        bool is_better = diff < min_diff;
        min_diff = is_better ? diff : min_diff;
        best_lut = is_better ? lut : best_lut;
        best_idx = is_better ? i : best_idx;
    }
    
    *lut_index = best_idx;
    return best_lut;
}

// Fast analytical direction computation with reduced operations
__device__ __forceinline__ void ComputeDirectionAnalytical_Fast(const Camera& cam, const int2 p, float3* dir) {
    // Use faster division and multiplication
    const float inv_width = __fdividef(1.0f, static_cast<float>(cam.width));
    const float inv_height = __fdividef(1.0f, static_cast<float>(cam.height));
    
    const float lon = (static_cast<float>(p.x) - cam.params[1]) * inv_width * 2.0f * CUDART_PI_F;
    const float lat = -(static_cast<float>(p.y) - cam.params[2]) * inv_height * CUDART_PI_F;
    
    float cos_lat, sin_lat, cos_lon, sin_lon;
    __sincosf(lat, &sin_lat, &cos_lat);
    __sincosf(lon, &sin_lon, &cos_lon);
    
    dir->x = __fmul_rn(cos_lat, sin_lon);
    dir->y = -sin_lat;
    dir->z = __fmul_rn(cos_lat, cos_lon);
}

// Optimized direct LUT lookup with prefetching and caching
__device__ __forceinline__ float3 GetDirectionFromLUT_Optimized(SphericalLUT* lut, int x, int y) {
    int idx = y * lut->width + x;
    
    // Use vector load if possible (assuming proper alignment)
    if (idx < lut->width * lut->height) {
        return lut->d_dir_vectors[idx];
    }
    
    return make_float3(0.0f, 0.0f, 1.0f); // fallback
}

// Find the appropriate LUT for given camera parameters (original interface)
__device__ __forceinline__ SphericalLUT* FindLUTForCamera(const Camera& cam) {
    int lut_index;
    return FindLUTForCamera_Optimized(cam, &lut_index);
}

// Optimized PixelToDir with multi-resolution support and reduced branching
__device__ __forceinline__ void PixelToDir_MultiRes(const Camera& cam, const int2 p, float3* dir) {
    int lut_index;
    SphericalLUT* lut = FindLUTForCamera_Optimized(cam, &lut_index);
    
    // Branchless bounds checking
    bool valid_lut = (lut != nullptr);
    bool in_bounds = (p.x >= 0) && (p.x < cam.width) && (p.y >= 0) && (p.y < cam.height);
    bool exact_resolution = valid_lut && (lut->width == cam.width) && (lut->height == cam.height);
    
    if (valid_lut && in_bounds && exact_resolution) {
        // Fast path: Direct memory access with optimized indexing
        *dir = GetDirectionFromLUT_Optimized(lut, p.x, p.y);
    } else {
        // Analytical computation - optimized for performance
        ComputeDirectionAnalytical_Fast(cam, p, dir);
    }
}

// Optimized Get3DPoint with better memory patterns
__device__ __forceinline__ void Get3DPoint_MultiRes(const Camera camera, const int2 p, 
                                                    const float depth, float *X) {
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    
    // Use fused multiply-add for better performance
    X[0] = __fmul_rn(dir.x, depth);
    X[1] = __fmul_rn(dir.y, depth);
    X[2] = __fmul_rn(dir.z, depth);
}

// Optimized GetViewDirection with reduced memory operations
__device__ __forceinline__ float4 GetViewDirection_MultiRes(const Camera camera, const int2 p, 
                                                            const float depth) {
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    return make_float4(dir.x, dir.y, dir.z, 0.0f);
}

// Optimized depth computation with reduced branching
__device__ __forceinline__ float ComputeDepthfromPlaneHypothesis_MultiRes(
    const Camera camera, const float4 plane_hypothesis, const int2 p) {
    
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    
    const float denom = fmaf(plane_hypothesis.x, dir.x, 
                        fmaf(plane_hypothesis.y, dir.y, 
                             plane_hypothesis.z * dir.z));
    
    // Branchless depth computation
    const float abs_denom = fabsf(denom);
    const bool valid_denom = abs_denom >= 1e-6f;
    const float result = __fdividef(-plane_hypothesis.w, denom);
    
    return valid_denom ? result : 1e6f;
}

// Optimized world point computation with FMA operations
__device__ __forceinline__ float3 Get3DPointonWorld_MultiRes(const float x, const float y, 
                                                             const float depth, const Camera camera) {
    int2 p = make_int2(__float2int_rn(x), __float2int_rn(y));
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    
    // Camera space point with FMA
    float3 point_cam = make_float3(
        __fmul_rn(dir.x, depth),
        __fmul_rn(dir.y, depth),
        __fmul_rn(dir.z, depth)
    );
    
    // Transform to world coordinates using FMA
    float3 tmp = make_float3(
        fmaf(camera.R[0], point_cam.x, fmaf(camera.R[3], point_cam.y, camera.R[6] * point_cam.z)),
        fmaf(camera.R[1], point_cam.x, fmaf(camera.R[4], point_cam.y, camera.R[7] * point_cam.z)),
        fmaf(camera.R[2], point_cam.x, fmaf(camera.R[5], point_cam.y, camera.R[8] * point_cam.z))
    );
    
    // Compute camera center with FMA
    float3 C = make_float3(
        -fmaf(camera.R[0], camera.t[0], fmaf(camera.R[3], camera.t[1], camera.R[6] * camera.t[2])),
        -fmaf(camera.R[1], camera.t[0], fmaf(camera.R[4], camera.t[1], camera.R[7] * camera.t[2])),
        -fmaf(camera.R[2], camera.t[0], fmaf(camera.R[5], camera.t[1], camera.R[8] * camera.t[2]))
    );
    
    return make_float3(tmp.x + C.x, tmp.y + C.y, tmp.z + C.z);
}

// Optimized reference camera point computation
__device__ __forceinline__ float3 Get3DPointonRefCam_MultiRes(const int x, const int y, 
                                                              const float depth, const Camera camera) {
    int2 p = make_int2(x, y);
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    
    return make_float3(
        __fmul_rn(dir.x, depth), 
        __fmul_rn(dir.y, depth), 
        __fmul_rn(dir.z, depth)
    );
}

// Optimized projection with cached trigonometric values when possible
__device__ __forceinline__ void ProjectonCamera_MultiRes(const float3 PointX, const Camera camera,
                                                         float2 &point, float &depth) {
    // Transform world point into camera frame using FMA
    float3 tmp = make_float3(
        fmaf(camera.R[0], PointX.x, fmaf(camera.R[1], PointX.y, fmaf(camera.R[2], PointX.z, camera.t[0]))),
        fmaf(camera.R[3], PointX.x, fmaf(camera.R[4], PointX.y, fmaf(camera.R[5], PointX.z, camera.t[1]))),
        fmaf(camera.R[6], PointX.x, fmaf(camera.R[7], PointX.y, fmaf(camera.R[8], PointX.z, camera.t[2])))
    );
    
    // Calculate depth using fast square root
    depth = __fsqrt_rn(fmaf(tmp.x, tmp.x, fmaf(tmp.y, tmp.y, tmp.z * tmp.z)));
    
    // Early return for invalid depth
    if (depth < 1e-6f) {
        point.x = camera.params[1];
        point.y = camera.params[2];
        return;
    }
    
    // Optimized angle computation
    const float inv_depth = __fdividef(1.0f, depth);
    const float normalized_y = __fmul_rn(tmp.y, inv_depth);
    
    // Clamp to valid range before asin
    const float clamped_y = fmaxf(-1.0f, fminf(1.0f, normalized_y));
    const float latitude = -asinf(clamped_y);
    const float longitude = atan2f(tmp.x, tmp.z);
    
    // Final projection with FMA
    const float inv_2pi = __fdividef(1.0f, 2.0f * CUDART_PI_F);
    const float inv_pi = __fdividef(1.0f, CUDART_PI_F);
    
    point.x = fmaf(__fmul_rn(longitude, inv_2pi), 
                   static_cast<float>(camera.width), camera.params[1]);
    point.y = fmaf(__fmul_rn(-latitude, inv_pi),
                   static_cast<float>(camera.height), camera.params[2]);
}

// Optimized adaptive reprojection threshold with reduced branching
__device__ __forceinline__ float GetAdaptiveReprojectionThreshold(
    const Camera& cam, int x, int y, float base_threshold = 2.0f) {
    
    // Branchless model check
    float spherical_factor = (cam.model == SPHERE) ? 1.0f : 0.0f;
    
    if (spherical_factor > 0.0f) {
        // Optimized latitude computation
        const float inv_height = __fdividef(1.0f, static_cast<float>(cam.height));
        const float lat = -((static_cast<float>(y) - cam.params[2]) * inv_height) * CUDART_PI_F;
        
        // Fast cosine with clamping
        const float cos_lat = fmaxf(cosf(lat), 0.1f);
        const float lat_factor = __fdividef(1.0f, cos_lat);
        
        // Radial distance computation
        const float dx = static_cast<float>(x) - cam.params[1];
        const float dy = static_cast<float>(y) - cam.params[2];
        const float dist_from_center = __fsqrt_rn(fmaf(dx, dx, dy * dy));
        
        const float max_dist = __fsqrt_rn(fmaf(cam.params[1], cam.params[1], 
                                               cam.params[2] * cam.params[2]));
        const float inv_max_dist = __fdividef(1.0f, max_dist);
        const float radial_factor = fmaf(0.5f, __fmul_rn(dist_from_center, inv_max_dist), 1.0f);
        
        return __fmul_rn(base_threshold, __fmul_rn(lat_factor, radial_factor));
    }
    
    return base_threshold;
}

// Optimized depth threshold with reduced operations
__device__ __forceinline__ float GetAdaptiveDepthThreshold(
    float depth, const Camera& cam, float base_threshold = 0.015f) {
    
    // Branchless computation
    float spherical_factor = (cam.model == SPHERE) ? 1.0f : 0.0f;
    
    // Linear factor for pinhole
    float linear_factor = fmaf(__fdividef(depth, 100.0f), (1.0f - spherical_factor), 1.0f);
    
    // Quadratic factor for spherical
    float depth_over_100 = __fdividef(depth, 100.0f);
    float quadratic_factor = fmaf(__fmul_rn(depth_over_100, depth_over_100), spherical_factor, 1.0f);
    
    float total_factor = __fmul_rn(linear_factor, quadratic_factor);
    return __fmul_rn(base_threshold, total_factor);
}

// Optimized confidence calculation with vectorized operations where possible
__device__ __forceinline__ float CalculateGeometricConfidence(
    const Camera& ref_cam, const Camera& src_cam,
    float3 point_world, int ref_x, int ref_y, int src_x, int src_y,
    float reproj_error, float depth_diff, float normal_angle) {
    
    // Pre-compute camera centers using FMA
    float3 ref_center = make_float3(
        -fmaf(ref_cam.R[0], ref_cam.t[0], fmaf(ref_cam.R[3], ref_cam.t[1], ref_cam.R[6] * ref_cam.t[2])),
        -fmaf(ref_cam.R[1], ref_cam.t[0], fmaf(ref_cam.R[4], ref_cam.t[1], ref_cam.R[7] * ref_cam.t[2])),
        -fmaf(ref_cam.R[2], ref_cam.t[0], fmaf(ref_cam.R[5], ref_cam.t[1], ref_cam.R[8] * ref_cam.t[2]))
    );
    
    float3 src_center = make_float3(
        -fmaf(src_cam.R[0], src_cam.t[0], fmaf(src_cam.R[3], src_cam.t[1], src_cam.R[6] * src_cam.t[2])),
        -fmaf(src_cam.R[1], src_cam.t[0], fmaf(src_cam.R[4], src_cam.t[1], src_cam.R[7] * src_cam.t[2])),
        -fmaf(src_cam.R[2], src_cam.t[0], fmaf(src_cam.R[5], src_cam.t[1], src_cam.R[8] * src_cam.t[2]))
    );
    
    // Vectorized baseline computation
    float3 baseline = make_float3(
        src_center.x - ref_center.x,
        src_center.y - ref_center.y,
        src_center.z - ref_center.z
    );
    float baseline_length = __fsqrt_rn(fmaf(baseline.x, baseline.x, 
                                       fmaf(baseline.y, baseline.y, baseline.z * baseline.z)));
    
    // Vectorized ray computations
    float3 ray1 = make_float3(
        point_world.x - ref_center.x,
        point_world.y - ref_center.y,
        point_world.z - ref_center.z
    );
    float3 ray2 = make_float3(
        point_world.x - src_center.x,
        point_world.y - src_center.y,
        point_world.z - src_center.z
    );
    
    float ray1_length = __fsqrt_rn(fmaf(ray1.x, ray1.x, fmaf(ray1.y, ray1.y, ray1.z * ray1.z)));
    float ray2_length = __fsqrt_rn(fmaf(ray2.x, ray2.x, fmaf(ray2.y, ray2.y, ray2.z * ray2.z)));
    
    // Optimized dot product and angle computation
    float dot_product = fmaf(ray1.x, ray2.x, fmaf(ray1.y, ray2.y, ray1.z * ray2.z));
    float cos_angle = __fdividef(dot_product, __fmul_rn(ray1_length, ray2_length));
    cos_angle = fmaxf(-1.0f, fminf(1.0f, cos_angle)); // clamp for numerical stability
    
    float triangulation_angle = acosf(cos_angle);
    
    // Fast exponential approximations for weights
    float angle_diff = fabsf(triangulation_angle - 1.047f); // 60 degrees
    float angle_weight = __expf(-2.0f * angle_diff);
    
    float reproj_weight = __expf(-__fmul_rn(reproj_error, reproj_error) * 0.25f);
    float depth_weight = __expf(-__fmul_rn(depth_diff, depth_diff) * 100.0f);
    float normal_weight = __expf(-__fmul_rn(normal_angle, normal_angle) * 10.0f);
    
    // Combine all weights
    float confidence = __fmul_rn(__fmul_rn(angle_weight, reproj_weight), 
                                __fmul_rn(depth_weight, normal_weight));
    
    // Spherical camera pole correction
    if (ref_cam.model == SPHERE) {
        const float inv_height = __fdividef(1.0f, static_cast<float>(ref_cam.height));
        const float ref_lat = -((static_cast<float>(ref_y) - ref_cam.params[2]) * inv_height) * CUDART_PI_F;
        const float pole_weight = cosf(ref_lat);
        confidence = __fmul_rn(confidence, pole_weight);
    }
    
    return confidence;
}

// Macro definitions to replace original functions
#define PixelToDir PixelToDir_MultiRes
#define Get3DPoint Get3DPoint_MultiRes
#define GetViewDirection GetViewDirection_MultiRes
#define ComputeDepthfromPlaneHypothesis ComputeDepthfromPlaneHypothesis_MultiRes
#define Get3DPointonWorld_cu Get3DPointonWorld_MultiRes
#define Get3DPointonRefCam_cu Get3DPointonRefCam_MultiRes
#define ProjectonCamera_cu ProjectonCamera_MultiRes

#endif // ACMMP_DEVICE_CUH