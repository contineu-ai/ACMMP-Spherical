#ifndef ACMMP_DEVICE_CUH
#define ACMMP_DEVICE_CUH

#include "ACMMP.h"
#include <math_constants.h>  // for CUDART_PI_F

// Move these device functions to header as inline functions
__device__ __forceinline__ float3 Get3DPointonWorld_cu(const float x,
                                                        const float y,
                                                        const float depth,
                                                        const Camera camera)
{
    float3 point_cam;
    if (camera.model == SPHERE) {
        // --- FIX 5: Use cx and cy for spherical projection in CUDA ---
        const float lon = (x - camera.params[1]) / static_cast<float>(camera.width) * 2.0f * CUDART_PI_F;
        const float lat = -(y - camera.params[2]) / static_cast<float>(camera.height) * CUDART_PI_F;
        point_cam.x = cosf(lat) * sinf(lon) * depth;
        point_cam.y = -sinf(lat)           * depth;
        point_cam.z = cosf(lat) * cosf(lon) * depth;
    } else {
        point_cam.x = depth * (x - camera.K[2]) / camera.K[0];
        point_cam.y = depth * (y - camera.K[5]) / camera.K[4];
        point_cam.z = depth;
    }

    // 2) rotate into world coords
    float3 tmp;
    tmp.x = camera.R[0]*point_cam.x + camera.R[3]*point_cam.y + camera.R[6]*point_cam.z;
    tmp.y = camera.R[1]*point_cam.x + camera.R[4]*point_cam.y + camera.R[7]*point_cam.z;
    tmp.z = camera.R[2]*point_cam.x + camera.R[5]*point_cam.y + camera.R[8]*point_cam.z;

    // 3) translate by camera center
    float3 C;
    C.x = -(camera.R[0]*camera.t[0] + camera.R[3]*camera.t[1] + camera.R[6]*camera.t[2]);
    C.y = -(camera.R[1]*camera.t[0] + camera.R[4]*camera.t[1] + camera.R[7]*camera.t[2]);
    C.z = -(camera.R[2]*camera.t[0] + camera.R[5]*camera.t[1] + camera.R[8]*camera.t[2]);

    // 4) final world point
    return make_float3(tmp.x + C.x,
                       tmp.y + C.y,
                       tmp.z + C.z);
}

__device__ __forceinline__ void ProjectonCamera_cu(const float3 PointX,
                                                    const Camera camera,
                                                    float2 &point,
                                                    float &depth)
{
    // 1) Transform world point into camera frame
    float3 tmp;
    tmp.x = camera.R[0] * PointX.x + camera.R[1] * PointX.y +
            camera.R[2] * PointX.z + camera.t[0];
    tmp.y = camera.R[3] * PointX.x + camera.R[4] * PointX.y +
            camera.R[5] * PointX.z + camera.t[1];
    tmp.z = camera.R[6] * PointX.x + camera.R[7] * PointX.y +
            camera.R[8] * PointX.z + camera.t[2];

    if (camera.model == SPHERE) {
        depth = sqrtf(tmp.x*tmp.x + tmp.y*tmp.y + tmp.z*tmp.z);
        if (depth < 1e-6f) {
            point.x = camera.params[1];
            point.y = camera.params[2];
            return;
        }

        // --- FIX 6: Use cx and cy for spherical back-projection in CUDA ---
        float latitude  = -asinf(tmp.y / depth);
        float longitude =  atan2f(tmp.x, tmp.z);

        point.x = (longitude / (2.0f * CUDART_PI_F)) * static_cast<float>(camera.width) + camera.params[1];
        point.y = (-latitude / CUDART_PI_F) * static_cast<float>(camera.height) + camera.params[2];

    }
    else {
        // 2b) Pinhole: depth is z-coordinate
        depth = tmp.z;

        // 3b) Standard intrinsics: (fx X + skew Y + cx Z)/Z
        point.x = (camera.K[0] * tmp.x +
                   camera.K[1] * tmp.y +
                   camera.K[2] * tmp.z) / depth;
        point.y = (camera.K[3] * tmp.x +
                   camera.K[4] * tmp.y +
                   camera.K[5] * tmp.z) / depth;
    }
}

#endif // ACMMP_DEVICE_CUH