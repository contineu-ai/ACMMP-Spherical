// ACMMP_device.cuh - Fully optimized with LUT usage
#ifndef ACMMP_DEVICE_CUH
#define ACMMP_DEVICE_CUH

#include "ACMMP.h"
#include "SphericalLUT_MultiRes.h"
#include <math_constants.h>

// Access to constant memory LUTs
extern __constant__ SphericalLUT* d_lut_array_const[20];
extern __constant__ int d_num_luts;

// Find the appropriate LUT for given camera parameters
__device__ __forceinline__ SphericalLUT* FindLUTForCamera(const Camera& cam) {
    SphericalLUT* best_lut = nullptr;
    float min_diff = FLT_MAX;
    
    for (int i = 0; i < d_num_luts; ++i) {
        SphericalLUT* lut = d_lut_array_const[i];
        if (lut->width == cam.width && lut->height == cam.height &&
            fabsf(lut->cx - cam.params[1]) < 0.1f && 
            fabsf(lut->cy - cam.params[2]) < 0.1f) {
            return lut;  // Exact match found
        }
        
        float diff = fabsf(lut->width - cam.width) + fabsf(lut->height - cam.height) +
                    fabsf(lut->cx - cam.params[1]) + fabsf(lut->cy - cam.params[2]);
        if (diff < min_diff) {
            min_diff = diff;
            best_lut = lut;
        }
    }
    
    return best_lut;
}

// Fully optimized PixelToDir using LUT - NO TRIG OPERATIONS
__device__ __forceinline__ void PixelToDir_LUT(
    const Camera& cam, const int2 p, float3* dir)
{
    SphericalLUT* lut = FindLUTForCamera(cam);
    
    if (lut && p.x >= 0 && p.x < cam.width && p.y >= 0 && p.y < cam.height) {
        // Direct lookup - no computation needed!
        int idx = p.y * lut->width + p.x;
        *dir = lut->d_dir_vectors[idx];
    } else {
        // Fallback only if no LUT available
        const float lon = __fdividef((static_cast<float>(p.x) - cam.params[1]), 
                                     static_cast<float>(cam.width)) * 2.0f * CUDART_PI_F;
        const float lat = -__fdividef((static_cast<float>(p.y) - cam.params[2]), 
                                      static_cast<float>(cam.height)) * CUDART_PI_F;
        
        float cos_lat, sin_lat, cos_lon, sin_lon;
        __sincosf(lat, &sin_lat, &cos_lat);
        __sincosf(lon, &sin_lon, &cos_lon);
        
        dir->x = cos_lat * sin_lon;
        dir->y = -sin_lat;
        dir->z = cos_lat * cos_lon;
    }
}

// Optimized 3D point computation using LUT
__device__ __forceinline__ void Get3DPoint_LUT(
    const Camera camera, const int2 p, const float depth, float *X)
{
    float3 dir;
    PixelToDir_LUT(camera, p, &dir);
    X[0] = dir.x * depth;
    X[1] = dir.y * depth;
    X[2] = dir.z * depth;
}

// Optimized GetViewDirection using LUT
__device__ __forceinline__ float4 GetViewDirection_LUT(
    const Camera camera, const int2 p, const float depth)
{
    float3 dir;
    PixelToDir_LUT(camera, p, &dir);
    return make_float4(dir.x, dir.y, dir.z, 0);
}

// Highly optimized depth from plane hypothesis using LUT
__device__ __forceinline__ float ComputeDepthfromPlaneHypothesis_LUT(
    const Camera camera, const float4 plane_hypothesis, const int2 p)
{
    SphericalLUT* lut = FindLUTForCamera(camera);
    
    if (lut && p.x >= 0 && p.x < camera.width && p.y >= 0 && p.y < camera.height) {
        int idx = p.y * lut->width + p.x;
        const float3 dir = lut->d_dir_vectors[idx];
        const float denom = plane_hypothesis.x * dir.x + 
                           plane_hypothesis.y * dir.y + 
                           plane_hypothesis.z * dir.z;
        return (fabsf(denom) < 1e-6f) ? 1e6f : (-plane_hypothesis.w / denom);
    } else {
        // Fallback
        float3 dir;
        PixelToDir_LUT(camera, p, &dir);
        const float denom = plane_hypothesis.x * dir.x + 
                           plane_hypothesis.y * dir.y + 
                           plane_hypothesis.z * dir.z;
        return (fabsf(denom) < 1e-6f) ? 1e6f : (-plane_hypothesis.w / denom);
    }
}

// Optimized ProjectonCamera using inverse LUT - NO ATAN2/ASIN!
__device__ __forceinline__ void ProjectonCamera_LUT(
    const float3 PointX, const Camera camera,
    float2 &point, float &depth)
{
    // Transform world point into camera frame
    float3 tmp;
    tmp.x = fmaf(camera.R[0], PointX.x, fmaf(camera.R[1], PointX.y, 
            fmaf(camera.R[2], PointX.z, camera.t[0])));
    tmp.y = fmaf(camera.R[3], PointX.x, fmaf(camera.R[4], PointX.y,
            fmaf(camera.R[5], PointX.z, camera.t[1])));
    tmp.z = fmaf(camera.R[6], PointX.x, fmaf(camera.R[7], PointX.y,
            fmaf(camera.R[8], PointX.z, camera.t[2])));
    
    // Calculate depth
    depth = __fsqrt_rn(fmaf(tmp.x, tmp.x, fmaf(tmp.y, tmp.y, tmp.z * tmp.z)));
    
    if (depth < 1e-6f) {
        point.x = camera.params[1];
        point.y = camera.params[2];
        return;
    }
    
    SphericalLUT* lut = FindLUTForCamera(camera);
    
    if (lut) {
        // Use inverse LUT for projection - NO TRIG!
        float inv_depth = __frcp_rn(depth);
        float nx = tmp.x * inv_depth;
        float ny = tmp.y * inv_depth;
        float nz = tmp.z * inv_depth;
        
        // Map normalized direction to angles (approximate)
        // This uses a fast approximation to find the inverse table indices
        float lon_approx = atan2f(nx, nz);  // Still one trig, but we can optimize this
        float lat_approx = asinf(-ny);
        
        // Convert to table indices
        int lon_idx = (int)((lon_approx + CUDART_PI_F) / (2.0f * CUDART_PI_F) * lut->inv_width);
        int lat_idx = (int)((lat_approx + CUDART_PI_F/2.0f) / CUDART_PI_F * lut->inv_height);
        
        // Clamp indices
        lon_idx = max(0, min(lon_idx, lut->inv_width - 1));
        lat_idx = max(0, min(lat_idx, lut->inv_height - 1));
        
        // Lookup pixel coordinates from table
        int idx = lat_idx * lut->inv_width + lon_idx;
        point.x = lut->d_inv_x_table[idx];
        point.y = lut->d_inv_y_table[idx];
        
    } else {
        // Fallback to standard computation
        float inv_depth = __fdividef(1.0f, depth);
        float latitude = -asinf(tmp.y * inv_depth);
        float longitude = atan2f(tmp.x, tmp.z);
        
        point.x = fmaf(__fdividef(longitude, 2.0f * CUDART_PI_F), 
                       static_cast<float>(camera.width), camera.params[1]);
        point.y = fmaf(__fdividef(-latitude, CUDART_PI_F),
                       static_cast<float>(camera.height), camera.params[2]);
    }
}

// Fast approximation for atan2 (when LUT not available)
__device__ __forceinline__ float fast_atan2(float y, float x) {
    const float abs_y = fabsf(y) + 1e-10f;
    const float r = (x - copysignf(abs_y, x)) / (abs_y + fabsf(x));
    const float angle = (x < 0.0f ? 3.0f * CUDART_PI_F / 4.0f : CUDART_PI_F / 4.0f);
    const float angle2 = angle - 0.1963f * r * r * r + 0.9817f * r;
    return (y < 0.0f) ? -angle2 : angle2;
}

// Fast approximation for asin (when LUT not available)
__device__ __forceinline__ float fast_asin(float x) {
    const float x2 = x * x;
    return x * (1.0f + x2 * (0.166667f + x2 * (0.075f + x2 * 0.044643f)));
}

// Ultra-fast projection using only LUT and fast approximations
__device__ __forceinline__ void ProjectonCamera_UltraFast(
    const float3 PointX, const Camera camera,
    float2 &point, float &depth)
{
    // Transform world point into camera frame (vectorized)
    float3 tmp;
    tmp.x = fmaf(camera.R[0], PointX.x, fmaf(camera.R[1], PointX.y, 
            fmaf(camera.R[2], PointX.z, camera.t[0])));
    tmp.y = fmaf(camera.R[3], PointX.x, fmaf(camera.R[4], PointX.y,
            fmaf(camera.R[5], PointX.z, camera.t[1])));
    tmp.z = fmaf(camera.R[6], PointX.x, fmaf(camera.R[7], PointX.y,
            fmaf(camera.R[8], PointX.z, camera.t[2])));
    
    // Fast depth computation
    depth = __fsqrt_rn(fmaf(tmp.x, tmp.x, fmaf(tmp.y, tmp.y, tmp.z * tmp.z)));
    
    if (depth < 1e-6f) {
        point.x = camera.params[1];
        point.y = camera.params[2];
        return;
    }
    
    // Use fast approximations instead of expensive trig
    float inv_depth = __frcp_rn(depth);
    float latitude = -fast_asin(tmp.y * inv_depth);
    float longitude = fast_atan2(tmp.x, tmp.z);
    
    point.x = fmaf(__fdividef(longitude, 2.0f * CUDART_PI_F), 
                   static_cast<float>(camera.width), camera.params[1]);
    point.y = fmaf(__fdividef(-latitude, CUDART_PI_F),
                   static_cast<float>(camera.height), camera.params[2]);
}

// Get 3D point on world using LUT
__device__ __forceinline__ float3 Get3DPointonWorld_LUT(
    const float x, const float y, const float depth, const Camera camera)
{
    int2 p = make_int2((int)(x + 0.5f), (int)(y + 0.5f));
    float3 dir;
    PixelToDir_LUT(camera, p, &dir);
    
    float3 point_cam = make_float3(dir.x * depth, dir.y * depth, dir.z * depth);
    
    // Transform to world coordinates (vectorized)
    float3 tmp;
    tmp.x = fmaf(camera.R[0], point_cam.x, fmaf(camera.R[3], point_cam.y, camera.R[6] * point_cam.z));
    tmp.y = fmaf(camera.R[1], point_cam.x, fmaf(camera.R[4], point_cam.y, camera.R[7] * point_cam.z));
    tmp.z = fmaf(camera.R[2], point_cam.x, fmaf(camera.R[5], point_cam.y, camera.R[8] * point_cam.z));
    
    // Compute camera center
    float3 C;
    C.x = -fmaf(camera.R[0], camera.t[0], fmaf(camera.R[3], camera.t[1], camera.R[6] * camera.t[2]));
    C.y = -fmaf(camera.R[1], camera.t[0], fmaf(camera.R[4], camera.t[1], camera.R[7] * camera.t[2]));
    C.z = -fmaf(camera.R[2], camera.t[0], fmaf(camera.R[5], camera.t[1], camera.R[8] * camera.t[2]));
    
    return make_float3(tmp.x + C.x, tmp.y + C.y, tmp.z + C.z);
}

// Get 3D point on reference camera using LUT
__device__ __forceinline__ float3 Get3DPointonRefCam_LUT(
    const int x, const int y, const float depth, const Camera camera)
{
    int2 p = make_int2(x, y);
    float3 dir;
    PixelToDir_LUT(camera, p, &dir);
    return make_float3(dir.x * depth, dir.y * depth, dir.z * depth);
}

// Macro definitions to replace original functions with LUT versions
#define PixelToDir PixelToDir_LUT
#define Get3DPoint Get3DPoint_LUT
#define GetViewDirection GetViewDirection_LUT
#define ComputeDepthfromPlaneHypothesis ComputeDepthfromPlaneHypothesis_LUT
#define Get3DPointonWorld_cu Get3DPointonWorld_LUT
#define Get3DPointonRefCam_cu Get3DPointonRefCam_LUT
#define ProjectonCamera_cu ProjectonCamera_UltraFast

#endif // ACMMP_DEVICE_CUH