#ifndef ACMMP_DEVICE_CUH
#define ACMMP_DEVICE_CUH

#include "ACMMP.h"
#include <math_constants.h>

// Optimized spherical-only device functions with fast math

__device__ __forceinline__ float3 Get3DPointonWorld_cu(const float x,
                                                        const float y, 
                                                        const float depth,
                                                        const Camera camera)
{
    // Direct spherical projection with fast math
    const float lon = __fdividef((x - camera.params[1]), static_cast<float>(camera.width)) * 2.0f * CUDART_PI_F;
    const float lat = -__fdividef((y - camera.params[2]), static_cast<float>(camera.height)) * CUDART_PI_F;
    
    float cos_lat, sin_lat, cos_lon, sin_lon;
    __sincosf(lat, &sin_lat, &cos_lat);
    __sincosf(lon, &sin_lon, &cos_lon);
    
    float3 point_cam;
    point_cam.x = cos_lat * sin_lon * depth;
    point_cam.y = -sin_lat * depth;
    point_cam.z = cos_lat * cos_lon * depth;
    
    // Optimized matrix multiplication using FMA
    float3 tmp;
    tmp.x = fmaf(camera.R[0], point_cam.x, fmaf(camera.R[3], point_cam.y, camera.R[6] * point_cam.z));
    tmp.y = fmaf(camera.R[1], point_cam.x, fmaf(camera.R[4], point_cam.y, camera.R[7] * point_cam.z));
    tmp.z = fmaf(camera.R[2], point_cam.x, fmaf(camera.R[5], point_cam.y, camera.R[8] * point_cam.z));
    
    // Compute camera center efficiently
    float3 C;
    C.x = -fmaf(camera.R[0], camera.t[0], fmaf(camera.R[3], camera.t[1], camera.R[6] * camera.t[2]));
    C.y = -fmaf(camera.R[1], camera.t[0], fmaf(camera.R[4], camera.t[1], camera.R[7] * camera.t[2]));
    C.z = -fmaf(camera.R[2], camera.t[0], fmaf(camera.R[5], camera.t[1], camera.R[8] * camera.t[2]));
    
    return make_float3(tmp.x + C.x, tmp.y + C.y, tmp.z + C.z);
}

__device__ __forceinline__ void ProjectonCamera_cu(const float3 PointX,
                                                   const Camera camera,
                                                   float2 &point,
                                                   float &depth)
{
    // Transform world point into camera frame using FMA
    float3 tmp;
    tmp.x = fmaf(camera.R[0], PointX.x, fmaf(camera.R[1], PointX.y, 
            fmaf(camera.R[2], PointX.z, camera.t[0])));
    tmp.y = fmaf(camera.R[3], PointX.x, fmaf(camera.R[4], PointX.y,
            fmaf(camera.R[5], PointX.z, camera.t[1])));
    tmp.z = fmaf(camera.R[6], PointX.x, fmaf(camera.R[7], PointX.y,
            fmaf(camera.R[8], PointX.z, camera.t[2])));
    
    // Fast spherical projection
    depth = __fsqrt_rn(fmaf(tmp.x, tmp.x, fmaf(tmp.y, tmp.y, tmp.z * tmp.z)));
    
    if (depth < 1e-6f) {
        point.x = camera.params[1];
        point.y = camera.params[2];
        return;
    }
    
    // Use fast inverse and trig functions
    float inv_depth = __fdividef(1.0f, depth);
    float latitude = -asinf(tmp.y * inv_depth);
    float longitude = atan2f(tmp.x, tmp.z);
    
    point.x = fmaf(__fdividef(longitude, 2.0f * CUDART_PI_F), 
                   static_cast<float>(camera.width), camera.params[1]);
    point.y = fmaf(__fdividef(-latitude, CUDART_PI_F),
                   static_cast<float>(camera.height), camera.params[2]);
}

// Optimized version with pre-computed values
__device__ __forceinline__ void ProjectonCamera_cu_optimized(const float3 PointX,
                                                             const Camera camera,
                                                             float2 &point,
                                                             float &depth)
{
    // Transform with vectorized operations
    float3 tmp;
    tmp.x = fmaf(camera.R[0], PointX.x, fmaf(camera.R[1], PointX.y,
            fmaf(camera.R[2], PointX.z, camera.t[0])));
    tmp.y = fmaf(camera.R[3], PointX.x, fmaf(camera.R[4], PointX.y,
            fmaf(camera.R[5], PointX.z, camera.t[1])));
    tmp.z = fmaf(camera.R[6], PointX.x, fmaf(camera.R[7], PointX.y,
            fmaf(camera.R[8], PointX.z, camera.t[2])));
    
    // Use fast reciprocal square root for normalization
    float depth_sq = fmaf(tmp.x, tmp.x, fmaf(tmp.y, tmp.y, tmp.z * tmp.z));
    depth = __fsqrt_rn(depth_sq);
    
    if (depth < 1e-6f) {
        point.x = camera.params[1];
        point.y = camera.params[2];
        return;
    }
    
    // Fast inverse
    float inv_depth = __frcp_rn(depth);
    
    // Fast approximation for small angles
    float y_norm = tmp.y * inv_depth;
    float latitude;
    if (fabsf(y_norm) < 0.5f) {
        // Taylor series approximation for asin
        float y2 = y_norm * y_norm;
        latitude = -y_norm * fmaf(y2, fmaf(y2, 0.075f, 0.16666667f), 1.0f);
    } else {
        latitude = -asinf(y_norm);
    }
    
    float longitude = atan2f(tmp.x, tmp.z);
    
    // Pre-computed constants
    const float inv_2pi = 0.159154943f; // 1/(2*pi)
    const float inv_pi = 0.318309886f;  // 1/pi
    
    point.x = fmaf(longitude * inv_2pi, static_cast<float>(camera.width), camera.params[1]);
    point.y = fmaf(-latitude * inv_pi, static_cast<float>(camera.height), camera.params[2]);
}

// Fast 3D point computation for spherical cameras
__device__ __forceinline__ float3 Get3DPointonRefCam_cu(const int x, const int y, 
                                                        const float depth,
                                                        const Camera camera)
{
    const float lon = __fdividef((static_cast<float>(x) - camera.params[1]), 
                                 static_cast<float>(camera.width)) * 2.0f * CUDART_PI_F;
    const float lat = -__fdividef((static_cast<float>(y) - camera.params[2]),
                                  static_cast<float>(camera.height)) * CUDART_PI_F;
    
    float cos_lat, sin_lat, cos_lon, sin_lon;
    __sincosf(lat, &sin_lat, &cos_lat);
    __sincosf(lon, &sin_lon, &cos_lon);
    
    return make_float3(cos_lat * sin_lon * depth,
                      -sin_lat * depth,
                      cos_lat * cos_lon * depth);
}

// Compute depth from plane hypothesis - optimized for spherical
__device__ __forceinline__ float ComputeDepthfromPlaneHypothesis_cu(
    const Camera camera, const float4 plane_hypothesis, const int2 p)
{
    const float lon = __fdividef((static_cast<float>(p.x) - camera.params[1]),
                                 static_cast<float>(camera.width)) * 2.0f * CUDART_PI_F;
    const float lat = -__fdividef((static_cast<float>(p.y) - camera.params[2]),
                                  static_cast<float>(camera.height)) * CUDART_PI_F;
    
    float cos_lat, sin_lat, cos_lon, sin_lon;
    __sincosf(lat, &sin_lat, &cos_lat);
    __sincosf(lon, &sin_lon, &cos_lon);
    
    float3 dir = make_float3(cos_lat * sin_lon, -sin_lat, cos_lat * cos_lon);
    
    const float denom = fmaf(plane_hypothesis.x, dir.x,
                            fmaf(plane_hypothesis.y, dir.y,
                                 plane_hypothesis.z * dir.z));
    
    return (fabsf(denom) < 1e-6f) ? 1e6f : __fdividef(-plane_hypothesis.w, denom);
}

// Fast normal transformation
__device__ __forceinline__ float4 TransformNormal_cu(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = fmaf(camera.R[0], plane_hypothesis.x,
                               fmaf(camera.R[3], plane_hypothesis.y,
                                    camera.R[6] * plane_hypothesis.z));
    transformed_normal.y = fmaf(camera.R[1], plane_hypothesis.x,
                               fmaf(camera.R[4], plane_hypothesis.y,
                                    camera.R[7] * plane_hypothesis.z));
    transformed_normal.z = fmaf(camera.R[2], plane_hypothesis.x,
                               fmaf(camera.R[5], plane_hypothesis.y,
                                    camera.R[8] * plane_hypothesis.z));
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

// Warp-level primitives for cost aggregation
__device__ __forceinline__ float warpReduceSum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float warpReduceMin(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fminf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

#endif // ACMMP_DEVICE_CUH