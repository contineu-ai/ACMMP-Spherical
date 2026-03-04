#ifndef APD_DEVICE_CUH
#define APD_DEVICE_CUH

// Include ACMMP spherical device functions (LUT-accelerated projection)
#include "ACMMP_device.cuh"

// UNDEF the compatibility macros — APD code must use explicit _APD wrappers
// that dispatch between pinhole and spherical based on camera model
#undef Get3DPoint
#undef GetViewDirection
#undef ComputeDepthfromPlaneHypothesis
#undef Get3DPointonWorld_cu
#undef Get3DPointonRefCam_cu
#undef ProjectonCamera_cu
#undef PixelToDir

// ============================================================================
// APD DISPATCH WRAPPERS — route to spherical or pinhole based on cam.model
// ============================================================================

// 4a. Get3DPoint_APD: backproject pixel to camera-frame 3D point
__device__ __forceinline__ void Get3DPoint_APD(const Camera& cam, const int2 p, float depth, float *X) {
    if (cam.model == SPHERE) {
        Get3DPoint_MultiRes(cam, p, depth, X);
    } else {
        X[0] = depth * (p.x - cam.K[2]) / cam.K[0];
        X[1] = depth * (p.y - cam.K[5]) / cam.K[4];
        X[2] = depth;
    }
}

__device__ __forceinline__ void Get3DPoint_APD(const Camera& cam, const short2 p, float depth, float *X) {
    Get3DPoint_APD(cam, make_int2(p.x, p.y), depth, X);
}

// 4b. GetViewDirection_APD
__device__ __forceinline__ float4 GetViewDirection_APD(const Camera& cam, const int2 p, float depth) {
    if (cam.model == SPHERE) {
        return GetViewDirection_MultiRes(cam, p, depth);
    } else {
        float X[3];
        X[0] = depth * (p.x - cam.K[2]) / cam.K[0];
        X[1] = depth * (p.y - cam.K[5]) / cam.K[4];
        X[2] = depth;
        float norm = sqrtf(X[0]*X[0] + X[1]*X[1] + X[2]*X[2]);
        float4 dir;
        dir.x = X[0] / norm;
        dir.y = X[1] / norm;
        dir.z = X[2] / norm;
        dir.w = 0;
        return dir;
    }
}

// 4c. ComputeDepthfromPlaneHypothesis_APD
__device__ __forceinline__ float ComputeDepthfromPlaneHypothesis_APD(
    const Camera& cam, const float4 plane_hypothesis, const int2 p) {
    if (cam.model == SPHERE) {
        return ComputeDepthfromPlaneHypothesis_MultiRes(cam, plane_hypothesis, p);
    } else {
        return -plane_hypothesis.w * cam.K[0] /
               ((p.x - cam.K[2]) * plane_hypothesis.x +
                (cam.K[0] / cam.K[4]) * (p.y - cam.K[5]) * plane_hypothesis.y +
                cam.K[0] * plane_hypothesis.z);
    }
}

// 4d. GetDistance2Origin_APD
__device__ __forceinline__ float GetDistance2Origin_APD(const Camera& cam, const int2 p, float depth, const float4 normal) {
    float X[3];
    Get3DPoint_APD(cam, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

// 4e. Get3DPointonWorld_APD
__device__ __forceinline__ float3 Get3DPointonWorld_APD(float x, float y, float depth, const Camera& cam) {
    if (cam.model == SPHERE) {
        return Get3DPointonWorld_MultiRes(x, y, depth, cam);
    } else {
        float3 pointX;
        pointX.x = depth * (x - cam.K[2]) / cam.K[0];
        pointX.y = depth * (y - cam.K[5]) / cam.K[4];
        pointX.z = depth;

        float3 tmpX;
        tmpX.x = cam.R[0] * pointX.x + cam.R[3] * pointX.y + cam.R[6] * pointX.z;
        tmpX.y = cam.R[1] * pointX.x + cam.R[4] * pointX.y + cam.R[7] * pointX.z;
        tmpX.z = cam.R[2] * pointX.x + cam.R[5] * pointX.y + cam.R[8] * pointX.z;

        pointX.x = tmpX.x + cam.c[0];
        pointX.y = tmpX.y + cam.c[1];
        pointX.z = tmpX.z + cam.c[2];
        return pointX;
    }
}

// 4e2. ProjectonCamera_APD
__device__ __forceinline__ void ProjectonCamera_APD(const float3 PointX, const Camera& cam, float2 &point, float &depth) {
    if (cam.model == SPHERE) {
        ProjectonCamera_MultiRes(PointX, cam, point, depth);
    } else {
        float3 tmp;
        tmp.x = cam.R[0] * PointX.x + cam.R[1] * PointX.y + cam.R[2] * PointX.z + cam.t[0];
        tmp.y = cam.R[3] * PointX.x + cam.R[4] * PointX.y + cam.R[5] * PointX.z + cam.t[1];
        tmp.z = cam.R[6] * PointX.x + cam.R[7] * PointX.y + cam.R[8] * PointX.z + cam.t[2];
        depth = cam.K[6] * tmp.x + cam.K[7] * tmp.y + cam.K[8] * tmp.z;
        point.x = (cam.K[0] * tmp.x + cam.K[1] * tmp.y + cam.K[2] * tmp.z) / depth;
        point.y = (cam.K[3] * tmp.x + cam.K[4] * tmp.y + cam.K[5] * tmp.z) / depth;
    }
}

// 4f. ComputeCorrespondingPoint_APD — per-pixel reprojection
__device__ __forceinline__ float2 ComputeCorrespondingPoint_APD(
    const Camera& ref_cam, const Camera& src_cam,
    const float4& plane, const int2 p, const float *H_pinhole) {
    if (ref_cam.model == SPHERE) {
        float depth = ComputeDepthfromPlaneHypothesis_APD(ref_cam, plane, p);
        float3 pw = Get3DPointonWorld_APD((float)p.x, (float)p.y, depth, ref_cam);
        float2 sp; float sd;
        ProjectonCamera_APD(pw, src_cam, sp, sd);
        return sp;
    } else {
        float3 pt;
        pt.x = H_pinhole[0]*p.x + H_pinhole[1]*p.y + H_pinhole[2];
        pt.y = H_pinhole[3]*p.x + H_pinhole[4]*p.y + H_pinhole[5];
        pt.z = H_pinhole[6]*p.x + H_pinhole[7]*p.y + H_pinhole[8];
        return make_float2(pt.x/pt.z, pt.y/pt.z);
    }
}

// Overload for short2 neighbor points
__device__ __forceinline__ float2 ComputeCorrespondingPoint_APD(
    const Camera& ref_cam, const Camera& src_cam,
    const float4& plane, const short2 p, const float *H_pinhole) {
    return ComputeCorrespondingPoint_APD(ref_cam, src_cam, plane, make_int2(p.x, p.y), H_pinhole);
}

// 4g. X-wrap helper for spherical boundary handling
__device__ __forceinline__ bool CheckAndWrapSourcePoint(float2& src_pt, const Camera& cam) {
    if (cam.model == SPHERE) {
        float w = (float)cam.width;
        float h = (float)cam.height;
        src_pt.x = fmodf(src_pt.x, w);
        if (src_pt.x < 0.f) src_pt.x += w;
        if (src_pt.y < 0.f || src_pt.y >= h) return false;
        return true;
    } else {
        return (src_pt.x >= 0.f && src_pt.x < (float)cam.width &&
                src_pt.y >= 0.f && src_pt.y < (float)cam.height);
    }
}

// 4h. Disparity conversion for spherical cameras
__device__ __forceinline__ float DepthToDisparity_APD(const Camera& cam, float baseline, float depth) {
    if (cam.model == SPHERE) {
        float pixels_per_radian = (float)cam.width / (2.0f * CUDART_PI_F);
        return pixels_per_radian * baseline / depth;
    } else {
        return cam.K[0] * baseline / depth;
    }
}

__device__ __forceinline__ float DisparityToDepth_APD(const Camera& cam, float baseline, float disparity) {
    if (cam.model == SPHERE) {
        float pixels_per_radian = (float)cam.width / (2.0f * CUDART_PI_F);
        return pixels_per_radian * baseline / disparity;
    } else {
        return cam.K[0] * baseline / disparity;
    }
}

#endif // APD_DEVICE_CUH
