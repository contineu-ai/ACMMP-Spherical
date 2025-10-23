#include "ACMMP.h"
#include "BatchACMMP.h"
#include "main.h"
#include "ACMMP_device.cuh"
#include <cuda_runtime.h>
#include <iostream>

// ============================================================================
// KERNEL TO COMPUTE PRECOMPUTED RAY TRANSFORMATIONS
// ============================================================================

__global__ void ComputePrecomputedRayTransformsKernel(
    PrecomputedRayData* precomp_data,
    const Camera* cameras,
    int ref_idx,
    int width,
    int height,
    int num_src_images)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    const Camera& ref_cam = cameras[ref_idx];
    
    // Get ray direction in reference camera
    int2 p = make_int2(x, y);
    float3 r_ref;
    PixelToDir_MultiRes(ref_cam, p, &r_ref);
    
    // Compute reference camera center: C_ref = -R_ref^T * t_ref (FIXED)
    float3 C_ref = make_float3(
        -(ref_cam.R[0] * ref_cam.t[0] + ref_cam.R[3] * ref_cam.t[1] + ref_cam.R[6] * ref_cam.t[2]),
        -(ref_cam.R[1] * ref_cam.t[0] + ref_cam.R[4] * ref_cam.t[1] + ref_cam.R[7] * ref_cam.t[2]),
        -(ref_cam.R[2] * ref_cam.t[0] + ref_cam.R[5] * ref_cam.t[1] + ref_cam.R[8] * ref_cam.t[2])
    );
    
    // Precompute for each source camera
    for (int src_idx = 0; src_idx < num_src_images; ++src_idx) {
        const Camera& src_cam = cameras[ref_idx + 1 + src_idx];
        
        int idx = ((y * width + x) * num_src_images) + src_idx;
        
        // Compute ray_src = R_src * R_ref^T * r_ref
        // First: R_ref^T * r_ref (CORRECT)
        float3 tmp = make_float3(
            ref_cam.R[0] * r_ref.x + ref_cam.R[3] * r_ref.y + ref_cam.R[6] * r_ref.z,
            ref_cam.R[1] * r_ref.x + ref_cam.R[4] * r_ref.y + ref_cam.R[7] * r_ref.z,
            ref_cam.R[2] * r_ref.x + ref_cam.R[5] * r_ref.y + ref_cam.R[8] * r_ref.z
        );
        
        // Then: R_src * tmp (CORRECT)
        precomp_data[idx].ray_src = make_float3(
            src_cam.R[0] * tmp.x + src_cam.R[1] * tmp.y + src_cam.R[2] * tmp.z,
            src_cam.R[3] * tmp.x + src_cam.R[4] * tmp.y + src_cam.R[5] * tmp.z,
            src_cam.R[6] * tmp.x + src_cam.R[7] * tmp.y + src_cam.R[8] * tmp.z
        );
        
        // Compute C_src = -R_src^T * t_src (FIXED)
        float3 C_src = make_float3(
            -(src_cam.R[0] * src_cam.t[0] + src_cam.R[3] * src_cam.t[1] + src_cam.R[6] * src_cam.t[2]),
            -(src_cam.R[1] * src_cam.t[0] + src_cam.R[4] * src_cam.t[1] + src_cam.R[7] * src_cam.t[2]),
            -(src_cam.R[2] * src_cam.t[0] + src_cam.R[5] * src_cam.t[1] + src_cam.R[8] * src_cam.t[2])
        );
        
        // Baseline vector: C_ref - C_src (CORRECT)
        float3 baseline_vec = make_float3(
            C_ref.x - C_src.x,
            C_ref.y - C_src.y,
            C_ref.z - C_src.z
        );
        
        // Transform to source camera frame: R_src * baseline_vec (CORRECT)
        precomp_data[idx].baseline = make_float3(
            src_cam.R[0] * baseline_vec.x + src_cam.R[1] * baseline_vec.y + src_cam.R[2] * baseline_vec.z,
            src_cam.R[3] * baseline_vec.x + src_cam.R[4] * baseline_vec.y + src_cam.R[5] * baseline_vec.z,
            src_cam.R[6] * baseline_vec.x + src_cam.R[7] * baseline_vec.y + src_cam.R[8] * baseline_vec.z
        );
    }
}
// ============================================================================
// HOST WRAPPER FUNCTION
// ============================================================================

void InitializePrecomputedTransforms(ProblemGPUResources* res,
                                     int num_images,
                                     int width,
                                     int height,
                                     cudaStream_t stream)
{
    if (!res->precomp_data_cuda || !res->cameras_cuda) {
        std::cerr << "Error: Precomputed data or cameras not allocated!" << std::endl;
        return;
    }
    
    int num_src_images = num_images - 1;
    
    dim3 block(32, 8);
    dim3 grid((width + block.x - 1) / block.x, 
              (height + block.y - 1) / block.y);
    
    ComputePrecomputedRayTransformsKernel<<<grid, block, 0, stream>>>(
        res->precomp_data_cuda,
        res->cameras_cuda,
        0,  // ref_idx is always 0
        width,
        height,
        num_src_images
    );
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA kernel launch error in InitializePrecomputedTransforms: " 
                  << cudaGetErrorString(err) << std::endl;
    }
    
    // Optional: wait for completion if debugging
    // cudaStreamSynchronize(stream);
}