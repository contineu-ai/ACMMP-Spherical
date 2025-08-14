#ifndef ACMMP_DEVICE_CUH
#define ACMMP_DEVICE_CUH

#include "ACMMP.h"
#include "SphericalLUT_MultiRes.h"
#include <math_constants.h>

// Access to constant memory LUTs
extern __constant__ SphericalLUT* d_lut_array_const[10];
extern __constant__ int d_num_luts;

// Find the appropriate LUT for given camera parameters
__device__ __forceinline__ SphericalLUT* FindLUTForCamera(const Camera& cam) {
    // Search through available LUTs for best match
    SphericalLUT* best_lut = nullptr;
    float min_diff = FLT_MAX;
    
    for (int i = 0; i < d_num_luts; ++i) {
        SphericalLUT* lut = d_lut_array_const[i];
        if (lut->width == cam.width && lut->height == cam.height &&
            fabsf(lut->cx - cam.params[1]) < 0.1f && 
            fabsf(lut->cy - cam.params[2]) < 0.1f) {
            return lut;  // Exact match found
        }
        
        // Calculate difference for closest match
        float diff = fabsf(lut->width - cam.width) + fabsf(lut->height - cam.height) +
                    fabsf(lut->cx - cam.params[1]) + fabsf(lut->cy - cam.params[2]);
        if (diff < min_diff) {
            min_diff = diff;
            best_lut = lut;
        }
    }
    
    return best_lut;
}

// Optimized PixelToDir with multi-resolution support
__device__ __forceinline__ void PixelToDir_MultiRes(const Camera& cam, const int2 p, float3* dir) {
    // Find appropriate LUT
    SphericalLUT* lut = FindLUTForCamera(cam);
    
    // If no LUT found or out of bounds, fall back to calculation
    if (!lut || p.x < 0 || p.x >= cam.width || p.y < 0 || p.y >= cam.height) {
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
        return;
    }
    
    // Check if we need to scale coordinates for LUT access
    if (lut->width == cam.width && lut->height == cam.height) {
        // Direct lookup
        int idx = p.y * lut->width + p.x;
        *dir = lut->d_dir_vectors[idx];
    } else {
        // Need to interpolate or scale - for now, fall back to calculation
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

// Get 3D point with multi-resolution LUT
__device__ __forceinline__ void Get3DPoint_MultiRes(const Camera camera, const int2 p, 
                                                    const float depth, float *X) {
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    X[0] = dir.x * depth;
    X[1] = dir.y * depth;
    X[2] = dir.z * depth;
}

// Get view direction with multi-resolution LUT
__device__ __forceinline__ float4 GetViewDirection_MultiRes(const Camera camera, const int2 p, 
                                                            const float depth) {
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    return make_float4(dir.x, dir.y, dir.z, 0);
}

// Compute depth from plane hypothesis with multi-resolution LUT
__device__ __forceinline__ float ComputeDepthfromPlaneHypothesis_MultiRes(
    const Camera camera, const float4 plane_hypothesis, const int2 p) {
    
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    const float denom = plane_hypothesis.x*dir.x + plane_hypothesis.y*dir.y + plane_hypothesis.z*dir.z;
    return (fabsf(denom) < 1e-6f) ? 1e6f : (-plane_hypothesis.w / denom);
}

// Get 3D point on world with multi-resolution LUT
__device__ __forceinline__ float3 Get3DPointonWorld_MultiRes(const float x, const float y, 
                                                             const float depth, const Camera camera) {
    int2 p = make_int2((int)(x + 0.5f), (int)(y + 0.5f));
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    
    float3 point_cam;
    point_cam.x = dir.x * depth;
    point_cam.y = dir.y * depth;
    point_cam.z = dir.z * depth;
    
    // Transform to world coordinates
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

// Get 3D point on reference camera with multi-resolution LUT
__device__ __forceinline__ float3 Get3DPointonRefCam_MultiRes(const int x, const int y, 
                                                              const float depth, const Camera camera) {
    int2 p = make_int2(x, y);
    float3 dir;
    PixelToDir_MultiRes(camera, p, &dir);
    return make_float3(dir.x * depth, dir.y * depth, dir.z * depth);
}

// ProjectonCamera with optimizations (still needs atan2/asin)
__device__ __forceinline__ void ProjectonCamera_MultiRes(const float3 PointX, const Camera camera,
                                                         float2 &point, float &depth) {
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
    
    // Still need to compute angles for inverse projection
    float inv_depth = __fdividef(1.0f, depth);
    float latitude = -asinf(tmp.y * inv_depth);
    float longitude = atan2f(tmp.x, tmp.z);
    
    point.x = fmaf(__fdividef(longitude, 2.0f * CUDART_PI_F), 
                   static_cast<float>(camera.width), camera.params[1]);
    point.y = fmaf(__fdividef(-latitude, CUDART_PI_F),
                   static_cast<float>(camera.height), camera.params[2]);
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