#include "ACMMP.h"
#include <math_constants.h>  // for CUDART_PI_F
#include <memory>
// these two get used before their definitions, so forward‐declare them
__device__ float3 Get3DPointonWorld_cu(const float x,
                                       const float y,
                                       const float depth,
                                       const Camera camera);
__device__ void ProjectonCamera_cu(const float3 PointX,
                                   const Camera camera,
                                   float2 &point,
                                   float &depth);

__device__ __forceinline__ float SampleDepthInv(curandState* rs, float dmin, float dmax) {
    dmin = fmaxf(dmin, 1e-6f);
    dmax = fmaxf(dmax, dmin + 1e-6f);
    const float inv_min = 1.0f / dmax;
    const float inv_max = 1.0f / dmin;
    const float u = curand_uniform(rs);           // (0,1]
    const float inv = inv_min + u * (inv_max - inv_min);
    return 1.0f / inv;
}

#define mul4(v,k) { \
    v->x = v->x * k; \
    v->y = v->y * k; \
    v->z = v->z * k;\
}

#define vecdiv4(v,k) { \
    v->x = v->x / k; \
    v->y = v->y / k; \
    v->z = v->z / k;\
}

__device__  void sort_small(float *d, const int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j-1]; j--)
            d[j] = d[j-1];
        d[j] = tmp;
    }
}

__device__ void sort_small_weighted(float *d, float *w, int n)
{
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        float tmp_w = w[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--) {
            d[j] = d[j - 1];
            w[j] = w[j - 1];
        }
        d[j] = tmp;
        w[j] = tmp_w;
    }
}

__device__ int FindMinCostIndex(const float *costs, const int n)
{
    float min_cost = costs[0];
    int min_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] <= min_cost) {
            min_cost = costs[idx];
            min_cost_idx = idx;
        }
    }
    return min_cost_idx;
}

__device__ int FindMaxCostIndex(const float *costs, const int n)
{
    float max_cost = costs[0];
    int max_cost_idx = 0;
    for (int idx = 1; idx < n; ++idx) {
        if (costs[idx] >= max_cost) {
            max_cost = costs[idx];
            max_cost_idx = idx;
        }
    }
    return max_cost_idx;
}

__device__  void setBit(unsigned int &input, const unsigned int n)
{
    input |= (unsigned int)(1 << n);
}

__device__  int isSet(unsigned int input, const unsigned int n)
{
    return (input >> n) & 1;
}

__device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result)
{
  result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
  result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
  result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

__device__ float Vec3DotVec3(const float4 vec1, const float4 vec2)
{
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

__device__ void NormalizeVec3 (float4 *vec)
{
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf (normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

__device__ inline void PixelToDir(const Camera& cam, const int2 p, float3* dir)
{
    if (cam.model == PINHOLE) {
        dir->x =  (static_cast<float>(p.x) - cam.K[2]) / cam.K[0];
        dir->y =  (static_cast<float>(p.y) - cam.K[5]) / cam.K[4];
        dir->z =  1.f;
        NormalizeVec3(reinterpret_cast<float4*>(dir));
    } else { // SPHERE
        // --- FIX 4: Use cx and cy for spherical projection in CUDA ---
        const float lon = (static_cast<float>(p.x) - cam.params[1]) / static_cast<float>(cam.width) * 2.0f * CUDART_PI_F;
        const float lat = -(static_cast<float>(p.y) - cam.params[2]) / static_cast<float>(cam.height) * CUDART_PI_F;
        dir->x =  cosf(lat) * sinf(lon);
        dir->y = -sinf(lat);
        dir->z =  cosf(lat) * cosf(lon);
    }
}


__device__ void TransformPDFToCDF(float* probs, const int num_probs)
{
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        prob_sum += probs[i];
    }
    const float inv_prob_sum = 1.0f / prob_sum;

    float cum_prob = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        const float prob = probs[i] * inv_prob_sum;
        cum_prob += prob;
        probs[i] = cum_prob;
    }
}

__device__ void Get3DPoint(const Camera camera, const int2 p, const float depth, float *X)
{
    float3 dir;  PixelToDir(camera, p, &dir);
    X[0] = dir.x * depth;
    X[1] = dir.y * depth;
    X[2] = dir.z * depth;
}

__device__ float4 GetViewDirection(const Camera camera, const int2 p, const float depth)
{
    float3 dir;  PixelToDir(camera, p, &dir);
    return make_float4(dir.x, dir.y, dir.z, 0);
}


__device__ float GetDistance2Origin(const Camera camera, const int2 p, const float depth, const float4 normal)
{
    float X[3];
    Get3DPoint(camera, p, depth, X);
    return -(normal.x * X[0] + normal.y * X[1] + normal.z * X[2]);
}

__device__   float SpatialGauss(float x1, float y1, float x2, float y2, float sigma, float mu = 0.0)
{
    float dis = pow(x1 - x2, 2) + pow(y1 - y2, 2) - mu;
    return exp(-1.0 * dis / (2 * sigma * sigma));
}

__device__  float RangeGauss(float x, float sigma, float mu = 0.0)
{
    float x_p = x - mu;
    return exp(-1.0 * (x_p * x_p) / (2 * sigma * sigma));
}

__device__ float ComputeDepthfromPlaneHypothesis(const Camera camera, const float4 plane_hypothesis, const int2 p)
{
    float3 dir;  PixelToDir(camera, p, &dir);
    // The plane is passed in as 'plane_hypothesis'
    const float denom = plane_hypothesis.x*dir.x + plane_hypothesis.y*dir.y + plane_hypothesis.z*dir.z;
    return (fabsf(denom) < 1e-6f) ? 1e6f : (-plane_hypothesis.w / denom);    
}
__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, curandState *rand_state, const float depth)
{
    float4 normal;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * curand_uniform(rand_state) -1.0f;
        q2 = 2.0f * curand_uniform(rand_state) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrt(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;

    float4 view_direction = GetViewDirection(camera, p, depth);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = - normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, curandState *rand_state, const float perturbation)
{
    float4 view_direction = GetViewDirection(camera, p, 1.0f);

    const float a1 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a2 = (curand_uniform(rand_state) - 0.5f) * perturbation;
    const float a3 = (curand_uniform(rand_state) - 0.5f) * perturbation;

    const float sin_a1 = sin(a1);
    const float sin_a2 = sin(a2);
    const float sin_a3 = sin(a3);
    const float cos_a1 = cos(a1);
    const float cos_a2 = cos(a2);
    const float cos_a3 = cos(a3);

    float R[9];
    R[0] = cos_a2 * cos_a3;
    R[1] = cos_a3 * sin_a1 * sin_a2 - cos_a1 * sin_a3;
    R[2] = sin_a1 * sin_a3 + cos_a1 * cos_a3 * sin_a2;
    R[3] = cos_a2 * sin_a3;
    R[4] = cos_a1 * cos_a3 + sin_a1 * sin_a2 * sin_a3;
    R[5] = cos_a1 * sin_a2 * sin_a3 - cos_a3 * sin_a1;
    R[6] = -sin_a2;
    R[7] = cos_a2 * sin_a1;
    R[8] = cos_a1 * cos_a2;

    float4 normal_perturbed;
    Mat33DotVec3(R, normal, &normal_perturbed);

    if (Vec3DotVec3(normal_perturbed, view_direction) >= 0.0f) {
        normal_perturbed = normal;
    }

    NormalizeVec3(&normal_perturbed);
    return normal_perturbed;
}

__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, curandState *rand_state, const float depth_min, const float depth_max)
{
    float depth = curand_uniform(rand_state) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, depth);
    plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 GeneratePertubedPlaneHypothesis(const Camera camera, const int2 p,
                                                  curandState *rand_state, const float perturbation,
                                                  const float4 plane_hypothesis_now,
                                                  const float depth_now,
                                                  const float depth_min, const float depth_max)
{
    // Local window intersected with global bounds
    float lo = fmaxf((1.0f - perturbation) * depth_now, depth_min);
    float hi = fminf((1.0f + perturbation) * depth_now, depth_max);
    if (!(hi > lo)) { lo = depth_min; hi = depth_max; }

    float4 best_ph   = plane_hypothesis_now;
    float  best_depth = depth_now;

    // Try a bounded number of candidates; occasionally jitter the normal
    for (int k = 0; k < 64; ++k) {
        const float cand_depth = SampleDepthInv(rand_state, lo, hi);

        // Every 8th try: perturb the normal a bit, otherwise keep current normal
        float4 n_try = ((k % 8) == 0)
            ? GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, 0.1f * CUDART_PI_F)
            : plane_hypothesis_now;

        float4 ph = n_try;
        ph.w = GetDistance2Origin(camera, p, cand_depth, n_try);

        const float test = ComputeDepthfromPlaneHypothesis(camera, ph, p);
        if (test >= depth_min && test <= depth_max && test < 1e6f) {
            best_ph    = ph;
            best_depth = test;
            break;
        }
    }

    // Final slight normal jitter around the accepted one
    float4 out = GeneratePerturbedNormal(camera, p, best_ph, rand_state, perturbation * CUDART_PI_F);
    out.w = GetDistance2Origin(camera, p, best_depth, out);
    return out;
}


__device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis)
{
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

__device__ float ComputeBilateralWeight(const float x_dist, const float y_dist, const float pix, const float center_pix, const float sigma_spatial, const float sigma_color)
{
    const float spatial_dist = sqrt(x_dist * x_dist + y_dist * y_dist);
    const float color_dist = fabs(pix - center_pix);
    return exp(-spatial_dist / (2.0f * sigma_spatial* sigma_spatial) - color_dist / (2.0f * sigma_color * sigma_color));
}

__device__ float ComputeBilateralNCC(
    const cudaTextureObject_t ref_image,
    const Camera ref_camera,
    const cudaTextureObject_t src_image,
    const Camera src_camera,
    const int2 p,
    const float4 plane_hypothesis,
    const PatchMatchParams params)
{
    const float cost_max = 2.0f;
    const int radius = params.patch_size / 2;

    // 1) Compute the 3D point of the reference center under this plane:
    float depth_ref = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
    float3 Pw_center = Get3DPointonWorld_cu(p.x, p.y, depth_ref, ref_camera);

    // 2) Project that center into the source to check validity:
    float2 pt_center; float dummy_depth;
    ProjectonCamera_cu(Pw_center, src_camera, pt_center, dummy_depth);
    if (pt_center.x < 0.0f || pt_center.x >= src_camera.width ||
        pt_center.y < 0.0f || pt_center.y >= src_camera.height) {
        return cost_max;
    }

    // 3) Now accumulate bilateral‐weighted NCC over the patch:
    float cost = 0.0f;
    const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
    float sum_ref = 0, sum_ref_ref = 0;
    float sum_src = 0, sum_src_src = 0;
    float sum_ref_src = 0, sum_bw = 0;

    for (int i = -radius; i <= radius; i += params.radius_increment) {
        for (int j = -radius; j <= radius; j += params.radius_increment) {
            int2 ref_pt = make_int2(p.x + i, p.y + j);

            // sample reference
            float ref_pix = tex2D<float>(ref_image,
                                         ref_pt.x + 0.5f,
                                         ref_pt.y + 0.5f);
            // compute 3D under same plane
            float depth_n = ComputeDepthfromPlaneHypothesis(ref_camera,
                                                            plane_hypothesis,
                                                            ref_pt);
            float3 Pw_n = Get3DPointonWorld_cu(ref_pt.x,
                                               ref_pt.y,
                                               depth_n,
                                               ref_camera);

            // project into source
            float2 src_pt; float src_d;
            ProjectonCamera_cu(Pw_n, src_camera, src_pt, src_d);
            // out‐of‐bounds?
            if (src_pt.x < 0.0f || src_pt.x >= src_camera.width ||
                src_pt.y < 0.0f || src_pt.y >= src_camera.height) {
                continue;
            }
            float src_pix = tex2D<float>(src_image,
                                         src_pt.x + 0.5f,
                                         src_pt.y + 0.5f);

            // bilateral weight
            float w = ComputeBilateralWeight(i, j,
                                             ref_pix,
                                             ref_center_pix,
                                             params.sigma_spatial,
                                             params.sigma_color);

            // accumulate
            sum_bw      += w;
            sum_ref     += w * ref_pix;
            sum_ref_ref += w * ref_pix * ref_pix;
            sum_src     += w * src_pix;
            sum_src_src += w * src_pix * src_pix;
            sum_ref_src += w * ref_pix * src_pix;
        }
    }

    if (sum_bw < 1e-6f) return cost_max;
    float inv_bw = 1.0f / sum_bw;
    sum_ref     *= inv_bw;
    sum_ref_ref *= inv_bw;
    sum_src     *= inv_bw;
    sum_src_src *= inv_bw;
    sum_ref_src *= inv_bw;

    float var_ref = sum_ref_ref - sum_ref * sum_ref;
    float var_src = sum_src_src - sum_src * sum_src;
    const float kMinVar = 1e-5f;
    if (var_ref < kMinVar || var_src < kMinVar) {
        return cost_max;
    }
    float covar    = sum_ref_src - sum_ref * sum_src;
    float denom    = sqrtf(var_ref * var_src);
    float ncc_cost = 1.0f - covar / denom;
    ncc_cost = fmaxf(0.0f, fminf(cost_max, ncc_cost));
    return ncc_cost;
}



__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params)
{
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int cost_count = 0;
    int num_valid_views = 0;

    for (int i = 1; i < params.num_images; ++i) {
        float c = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max) {
            num_valid_views++;
        }
    }

    sort_small(cost_vector, cost_count);
    *selected_views = 0;

    int top_k = min(num_valid_views, params.top_k);
    if (top_k > 0) {
        float cost = 0.0f;
        for (int i = 0; i < top_k; ++i) {
            cost += cost_vector[i];
        }
        float cost_threshold = cost_vector[top_k - 1];
        for (int i = 0; i < params.num_images - 1; ++i) {
            if (cost_vector_copy[i] <= cost_threshold) {
                setBit(*selected_views, i);
            }
        }
        return cost / top_k;
    } else {
        return cost_max;
    }
}

__device__ void ComputeMultiViewCostVector(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, float *cost_vector, const PatchMatchParams params)
{
    for (int i = 1; i < params.num_images; ++i) {
        cost_vector[i - 1] = ComputeBilateralNCC(images[0], cameras[0], images[i], cameras[i], p, plane_hypothesis, params);
    }
}

__device__ float3 Get3DPointonWorld_cu(const float x,
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

__device__ void ProjectonCamera_cu(const float3 PointX,
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

__device__ float ComputeGeomConsistencyCost(const cudaTextureObject_t depth_image, const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, const int2 p)
{
    const float max_cost = 3.0f;

    float depth = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
    float3 forward_point = Get3DPointonWorld_cu(p.x, p.y, depth, ref_camera);

    float2 src_pt;
    float src_d;
    ProjectonCamera_cu(forward_point, src_camera, src_pt, src_d);
    const float src_depth = tex2D<float>(depth_image,  (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);

    if (src_depth == 0.0f) {
        return max_cost;
    }

    float3 src_3D_pt = Get3DPointonWorld_cu(src_pt.x, src_pt.y, src_depth, src_camera);

    float2 backward_point;
    float ref_d;
    ProjectonCamera_cu(src_3D_pt, ref_camera, backward_point, ref_d);

    const float diff_col = p.x - backward_point.x;
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses,  float4 *scaled_plane_hypotheses, float *costs,  float *pre_costs,  curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int width = cameras[0].width;
    int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    curand_init(clock64(), p.y, p.x, &rand_states[center]);

    if (!params.geom_consistency && !params.hierarchy ) {
        plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], params.depth_min, params.depth_max);
        costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
    }
    else if (params.planar_prior) {
        if (plane_masks[center] > 0 && costs[center] >= 0.1f) {
            float perturbation = 0.02f;

            float4 plane_hypothesis = prior_planes[center];
            float depth_perturbed = plane_hypothesis.w;
            const float depth_min_perturbed = (1 - 3 * perturbation) * depth_perturbed;
            const float depth_max_perturbed = (1 + 3 * perturbation) * depth_perturbed;
            depth_perturbed = curand_uniform(&rand_states[center]) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
            float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, plane_hypothesis, &rand_states[center], 3 * perturbation * M_PI);
            plane_hypothesis_perturbed.w = depth_perturbed;
            plane_hypotheses[center] = plane_hypothesis_perturbed;
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
        else {
            float4 plane_hypothesis = plane_hypotheses[center];
            float depth = plane_hypothesis.w;
            plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
            plane_hypotheses[center] = plane_hypothesis;
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
        }
    }
    else {
        if(params.upsample) {
            const float scale = 1.0 * params.scaled_cols / width;
            const float sigmad = 0.50;
            const float sigmar = 25.5;
            const int Imagescale = max(width / params.scaled_cols , height / params.scaled_rows);
            const int WinWidth =Imagescale * Imagescale + 1;
            int num_neighbors = WinWidth / 2;

            const float o_y = p.y * scale;
            const float o_x = p.x * scale;
            const float refPix = tex2D<float>(texture_objects[0].images[0], p.x + 0.5f, p.y + 0.5f);
            int r_y = 0;
            int r_ys = 0;
            int r_x = 0;
            int r_xs = 0;
            float sgauss = 0.0, rgauss = 0.0, totalgauss = 0.0;
            float c_total_val = 0.0, normalizing_factor = 0.0;
            float  srcPix = 0, neighborPix = 0;
            float4 srcNorm;
            float4 n_total_val;
            n_total_val.x = 0; n_total_val.y = 0; n_total_val.z = 0; n_total_val.w = 0;
     \
            for (int j = -num_neighbors; j <= num_neighbors; ++j) {
                // source
                r_y = o_y + j;
                r_y = (r_y > 0 ? (r_y < params.scaled_rows ? r_y : params.scaled_rows - 1) : 0) ;
                // reference
                r_ys = p.y + j;
                for (int i = -num_neighbors; i <= num_neighbors; ++i) {
                    // source
                    r_x = o_x + i;
                    r_x = (r_x > 0 ? (r_x < params.scaled_cols? r_x : params.scaled_cols - 1) : 0);
                    const int s_center = r_y*params.scaled_cols+r_x;
                    if (s_center >=  params.scaled_rows * params.scaled_cols) {
                        printf("Illegal: %d, %d, %f, %f (%d, %d)\n", r_x, r_y, o_x, o_y, params.scaled_cols,  params.scaled_rows);
                    }
                    srcPix = scaled_plane_hypotheses[s_center].w;
                    srcNorm = scaled_plane_hypotheses[s_center];
                    // refIm
                    r_xs = p.x + i;
                    neighborPix = tex2D<float>(texture_objects[0].images[0], r_xs + 0.5f, r_ys + 0.5f);

                    sgauss = SpatialGauss(o_x, o_y, r_x, r_y, sigmad);
                    rgauss = RangeGauss(fabs(refPix - neighborPix), sigmar);
                    totalgauss = sgauss * rgauss;
                    normalizing_factor += totalgauss;
                    c_total_val += srcPix * totalgauss;
                    mul4((&srcNorm), totalgauss);
                    n_total_val.x  = n_total_val.x + srcNorm.x;
                    n_total_val.y  = n_total_val.y + srcNorm.y;
                    n_total_val.z  = n_total_val.z + srcNorm.z;
                }
            }
            costs[center] = c_total_val / normalizing_factor;
            vecdiv4((&n_total_val), normalizing_factor);
            NormalizeVec3(&n_total_val);

             costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
            pre_costs[center] = costs[center];

            float4 plane_hypothesis = n_total_val;
            plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
            float depth = plane_hypotheses[center].w;
            plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
            plane_hypotheses[center] = plane_hypothesis;
            costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
         }
         else {
             float4 plane_hypothesis;
             if (params.hierarchy) {
                 plane_hypothesis = scaled_plane_hypotheses[center];
             }
             else {
                 plane_hypothesis = plane_hypotheses[center];
             }
             plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
             float depth = plane_hypothesis.w;
             plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depth, plane_hypothesis);
             plane_hypotheses[center] = plane_hypothesis;
             costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
         }
    }
}

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images,
                                          const cudaTextureObject_t *depth_images,
                                          const Camera *cameras,
                                          float4 *plane_hypothesis,
                                          float *depth,
                                          float *cost,
                                          curandState *rand_state,
                                          const float *view_weights,
                                          const float weight_norm,
                                          float4 *prior_planes,
                                          unsigned int *plane_masks,
                                          float *restricted_cost,
                                          const int2 p,
                                          const PatchMatchParams params)
{
    // Early exit if no views were selected
    if (weight_norm <= 0.0f) return;

    const float perturbation = 0.02f;
    const int center = p.y * cameras[0].width + p.x;

    // ACMMP's prior parameters (preserved)
    const float gamma = 0.5f;
    const float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
    const float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
    const float angle_sigma = CUDART_PI_F * (5.0f / 180.0f);  // Using CUDART_PI_F for consistency
    const float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
    const float beta = 0.18f;

    // 1) Random candidate depth generation (improved with ACMH approach)
    float depth_rand;
    float4 plane_hypothesis_rand;
    
    if (params.planar_prior && plane_masks[center] > 0) {
        // Use prior-based sampling
        float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
        depth_rand = SampleDepthInv(rand_state, 
                                   fmaxf(depth_prior - 3 * depth_sigma, params.depth_min),
                                   fminf(depth_prior + 3 * depth_sigma, params.depth_max));
        plane_hypothesis_rand = GeneratePerturbedNormal(cameras[0], p, prior_planes[center], rand_state, angle_sigma);
    } else {
        // Standard random sampling
        depth_rand = SampleDepthInv(rand_state, params.depth_min, params.depth_max);
        plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, depth_rand);
    }

    // 2) Local window around current depth (ACMH's bounded + healed approach)
    float lo = fmaxf((1.0f - perturbation) * (*depth), params.depth_min);
    float hi = fminf((1.0f + perturbation) * (*depth), params.depth_max);
    if (!(hi > lo)) { 
        lo = params.depth_min; 
        hi = params.depth_max; 
    }

    float depth_perturbed = *depth;
    bool ok = false;
    for (int k = 0; k < 32; ++k) {
        float cand = SampleDepthInv(rand_state, lo, hi);
        if (cand >= params.depth_min && cand <= params.depth_max) {
            depth_perturbed = cand;
            ok = true;
            break;
        }
    }
    if (!ok) {
        depth_perturbed = fminf(fmaxf(*depth, params.depth_min), params.depth_max);
    }

    // 3) Slightly perturbed normal around current one
    float4 plane_hypothesis_perturbed = 
        GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, perturbation * CUDART_PI_F);

    // 4) Evaluate candidates
    const int num_planes = 5;
    float  depths[num_planes]  = { depth_rand, *depth, depth_rand, *depth, depth_perturbed };
    float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand,
                                   plane_hypothesis_rand, plane_hypothesis_perturbed,
                                   *plane_hypothesis };

    for (int i = 0; i < num_planes; ++i) {
        float cost_vector[32] = { 2.0f };
        float4 temp_plane_hypothesis = normals[i];
        temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths[i], temp_plane_hypothesis);

        // Compute multi-view photo-consistency costs
        ComputeMultiViewCostVector(images, cameras, p, temp_plane_hypothesis, cost_vector, params);

        // Aggregate with view weights (+ optional geom consistency)
        float temp_cost = 0.0f;
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0.0f) {
                if (params.geom_consistency) {
                    temp_cost += view_weights[j] * (cost_vector[j] +
                                  0.1f * ComputeGeomConsistencyCost(depth_images[j+1],
                                                                    cameras[0], cameras[j+1],
                                                                    temp_plane_hypothesis, p));
                } else {
                    temp_cost += view_weights[j] * cost_vector[j];
                }
            }
        }
        if (weight_norm > 0.0f) {
            temp_cost /= weight_norm;  // Safe divide
        }

        // Validate depth
        const float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
        if (depth_before < params.depth_min || depth_before > params.depth_max || depth_before >= 1e6f) {
            continue;  // Skip invalid depths
        }

        // Accept based on prior availability (ACMMP's strength preserved)
        if (params.planar_prior && plane_masks[center] > 0) {
            // Prior-based acceptance
            float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
            float depth_diff = depths[i] - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
            angle_cos = fminf(fmaxf(angle_cos, -1.0f), 1.0f);  // Clamp for numerical stability
            float angle_diff = acosf(angle_cos);
            
            float prior = gamma + expf(-depth_diff * depth_diff / two_depth_sigma_squared) * 
                                 expf(-angle_diff * angle_diff / two_angle_sigma_squared);
            float restricted_temp_cost = expf(-temp_cost * temp_cost / beta) * prior;
            
            if (restricted_temp_cost > *restricted_cost) {
                *depth = depth_before;
                *plane_hypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
                *restricted_cost = restricted_temp_cost;
            }
        } else {
            // Standard acceptance (ACMH's clean approach)
            if (temp_cost < *cost) {
                *depth = depth_before;
                *plane_hypothesis = temp_plane_hypothesis;
                *cost = temp_cost;
            }
        }
    }
}

__device__ void CheckerboardPropagation(const cudaTextureObject_t *images, const cudaTextureObject_t *depths, const Camera *cameras, float4 *plane_hypotheses, float *costs, float *pre_costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const int2 p, const PatchMatchParams params, const int iter)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    int left_near = center - 1;
    int left_far = center - 3;
    int right_near = center + 1;
    int right_far = center + 3;
    int up_near = center - width;
    int up_far = center - 3 * width;
    int down_near = center + width;
    int down_far = center + 3 * width;

    // Adaptive Checkerboard Sampling
    float cost_array[8][32] = {2.0f};
    // 0 -- up_near, 1 -- up_far, 2 -- down_near, 3 -- down_far, 4 -- left_near, 5 -- left_far, 6 -- right_near, 7 -- right_far
    bool flag[8] = {false};
    int num_valid_pixels = 0;

    float costMin;
    int costMinPoint;

    // up_far
    if (p.y > 2) {
        flag[1] = true;
        num_valid_pixels++;
        costMin = costs[up_far];
        costMinPoint = up_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int pointTemp = up_far - 2 * i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_far], cost_array[1], params);
    }

    // dwon_far
    if (p.y < height - 3) {
        flag[3] = true;
        num_valid_pixels++;
        costMin = costs[down_far];
        costMinPoint = down_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y < height - 3 - 2 * i) {
                int pointTemp = down_far + 2 * i * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_far], cost_array[3], params);
    }

    // left_far
    if (p.x > 2) {
        flag[5] = true;
        num_valid_pixels++;
        costMin = costs[left_far];
        costMinPoint = left_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int pointTemp = left_far - 2 * i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_far], cost_array[5], params);
    }

    // right_far
    if (p.x < width - 3) {
        flag[7] = true;
        num_valid_pixels++;
        costMin = costs[right_far];
        costMinPoint = right_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x < width - 3 - 2 * i) {
                int pointTemp = right_far + 2 * i;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_far = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_far], cost_array[7], params);
    }

    // up_near
    if (p.y > 0) {
        flag[0] = true;
        num_valid_pixels++;
        costMin = costs[up_near];
        costMinPoint = up_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y > 1 + i && p.x > i) {
                int pointTemp = up_near - (1 + i) * width - (1+i);
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y > 1 + i && p.x < width - 1 - i) {
                int pointTemp = up_near - (1 + i) * width + (1+i);
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        up_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[up_near], cost_array[0], params);
    }

    // down_near
    if (p.y < height - 1) {
        flag[2] = true;
        num_valid_pixels++;
        costMin = costs[down_near];
        costMinPoint = down_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y < height - 2 - i && p.x > i) {
                int pointTemp = down_near + (1 + i) * width - (1+i);
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.y < height - 2 - i && p.x < width - 1 - i) {
                int pointTemp = down_near + (1 + i) * width + (1+i);
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        down_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[down_near], cost_array[2], params);
    }

    // left_near
    if (p.x > 0) {
        flag[4] = true;
        num_valid_pixels++;
        costMin = costs[left_near];
        costMinPoint = left_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x > 1 + i && p.y > i) {
                int pointTemp = left_near - (1 + i) - (1+i) * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x > 1 + i && p.y < height - 1 - i) {
                int pointTemp = left_near - (1 + i) + (1+i) * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        left_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[left_near], cost_array[4], params);
    }

    // right_near
    if (p.x < width - 1) {
        flag[6] = true;
        num_valid_pixels++;
        costMin = costs[right_near];
        costMinPoint = right_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x < width - 2 - i && p.y > i) {
                int pointTemp = right_near + (1 + i) - (1+i) * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
            if (p.x < width - 2 - i && p.y < height - 1- i) {
                int pointTemp = right_near + (1 + i) + (1+i) * width;
                if (costs[pointTemp] < costMin) {
                    costMin = costs[pointTemp];
                    costMinPoint = pointTemp;
                }
            }
        }
        right_near = costMinPoint;
        ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[right_near], cost_array[6], params);
    }
    const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // Multi-hypothesis Joint View Selection
    float view_weights[32] = {0.0f};
    float view_selection_priors[32] = {0.0f};
    int neighbor_positions[4] = {center - width, center + width, center - 1, center + 1};
    for (int i = 0; i < 4; ++i) {
        if (flag[2 * i]) {
            for (int j = 0; j < params.num_images - 1; ++j) {
                if (isSet(selected_views[neighbor_positions[i]], j) == 1) {
                    view_selection_priors[j] += 0.9f;
                } else {
                    view_selection_priors[j] += 0.1f;
                }
            }
        }
    }

    float sampling_probs[32] = {0.0f};
    float cost_threshold = 0.8 * expf((iter) * (iter) / (-90.0f));
    for (int i = 0; i < params.num_images - 1; i++) {
        float count = 0;
        int count_false = 0;
        float tmpw = 0;
        for (int j = 0; j < 8; j++) {
            if (cost_array[j][i] < cost_threshold) {
                tmpw += expf(cost_array[j][i] * cost_array[j][i] / (-0.18f));
                count++;
            }
            if (cost_array[j][i] > 1.2f) {
                count_false++;
            }
        }
        if (count > 2 && count_false < 3) {
            sampling_probs[i] = tmpw / count;
        }
        else if (count_false < 3) {
            sampling_probs[i] = expf(cost_threshold * cost_threshold / (-0.32f));
        }
        sampling_probs[i] = sampling_probs[i] * view_selection_priors[i];
    }

    TransformPDFToCDF(sampling_probs, params.num_images - 1);
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = curand_uniform(&rand_states[center]) - FLT_EPSILON;

        for (int image_id = 0; image_id < params.num_images - 1; ++image_id) {
            const float prob = sampling_probs[image_id];
            if (prob > rand_prob) {
                view_weights[image_id] += 1.0f;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    int num_selected_view = 0;
    float weight_norm = 0;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            setBit(temp_selected_views, i);
            weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    float final_costs[8] = {0.0f};
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < params.num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params.geom_consistency) {
                    if (flag[i]) {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.2f * ComputeGeomConsistencyCost(depths[j+1], cameras[0], cameras[j+1], plane_hypotheses[positions[i]], p));
                    }
                    else {
                        final_costs[i] += view_weights[j] * (cost_array[i][j] + 0.1f * 3.0f);
                    }
                }
                else {
                    final_costs[i] += view_weights[j] * cost_array[i][j];
                }
            }
        }
        final_costs[i] /= weight_norm;
    }

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    float cost_vector_now[32] = {2.0f};
    ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[center], cost_vector_now, params);
    float cost_now = 0.0f;
    for (int i = 0; i < params.num_images - 1; ++i) {
        if (params.geom_consistency) {
            cost_now += view_weights[i] * (cost_vector_now[i] + 0.2f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p));
        }
        else {
            cost_now += view_weights[i] * cost_vector_now[i];
        }
    }
    cost_now /= weight_norm;
    costs[center] = cost_now;
    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    float restricted_cost = 0.0f;
    if (params.planar_prior) {
        float restricted_final_costs[8] = {0.0f};
        float gamma = 0.5f;
        float depth_sigma = (params.depth_max - params.depth_min) / 64.0f;
        float two_depth_sigma_squared = 2 * depth_sigma * depth_sigma;
        float angle_sigma = M_PI * (5.0f / 180.0f);
        float two_angle_sigma_squared = 2 * angle_sigma * angle_sigma;
        float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
        float beta = 0.18f;

        if (plane_masks[center] > 0) {
            for (int i = 0; i < 8; i++) {
                if (flag[i]) {
                    float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[i]], p);
                    float depth_diff = depth_now - depth_prior;
                    float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[positions[i]]);
                    float angle_diff = acos(angle_cos);
                    float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
                    restricted_final_costs[i] = exp(-final_costs[i] * final_costs[i] / beta) * prior;
                }
            }
            const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

            float restricted_cost_now = 0.0f;
            float depth_now = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
            float depth_diff = depth_now - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[center]);
            float angle_diff = acos(angle_cos);
            float prior = gamma + exp(- depth_diff * depth_diff / two_depth_sigma_squared) * exp(- angle_diff * angle_diff / two_angle_sigma_squared);
            restricted_cost_now = exp(-cost_now * cost_now / beta) * prior;

            if (flag[max_cost_idx]) {
                float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[max_cost_idx]], p);

                if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_final_costs[max_cost_idx] > restricted_cost_now) {
                    depth_now   = depth_before;
                    plane_hypotheses[center] = plane_hypotheses[positions[max_cost_idx]];
                    costs[center] = final_costs[max_cost_idx];
                    restricted_cost = restricted_final_costs[max_cost_idx];
                    selected_views[center] = temp_selected_views;
                }
            }
        }
        else if (flag[min_cost_idx]) {
            float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

            if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
                depth_now = depth_before;
                plane_hypotheses[center] = plane_hypotheses[positions[min_cost_idx]];
                costs[center] = final_costs[min_cost_idx];
            }
        }
    }

    float4 plane_hypotheses_now;
    if (!params.planar_prior && flag[min_cost_idx]) {
        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);

        if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
            depth_now = depth_before;
           plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
           cost_now = final_costs[min_cost_idx];
           selected_views[center] = temp_selected_views;
        }
    }

    PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], view_weights, weight_norm, prior_planes, plane_masks, &restricted_cost, p, params);

    if (params.hierarchy) {
        if (cost_now < pre_costs[center] - 0.1f) {
            costs[center] = cost_now;
            plane_hypotheses[center] = plane_hypotheses_now;
        }
    }
    else {
        costs[center] = cost_now;
        plane_hypotheses[center] = plane_hypotheses_now;
    }
}

__global__ void BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs,  float *pre_costs,  curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs,  rand_states, selected_views, prior_planes, plane_masks, p, params, iter);
}

__global__ void RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs,  float *pre_costs, curandState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const PatchMatchParams params, const int iter)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardPropagation(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs, rand_states, selected_views, prior_planes, plane_masks, p, params, iter);
}

__global__ void GetDepthandNormal(Camera *cameras, float4 *plane_hypotheses, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const int2 p)
{
    int width = cameras[0].width;
    int height = cameras[0].height;
    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;

    float filter[21];
    int index = 0;

    filter[index++] = plane_hypotheses[center].w;

    // Left
    const int left = center - 1;
    const int leftleft = center - 3;

    // Up
    const int up = center - width;
    const int upup = center - 3 * width;

    // Down
    const int down = center + width;
    const int downdown = center + 3 * width;

    // Right
    const int right = center + 1;
    const int rightright = center + 3;

    if (costs[center] < 0.001f) {
        return;
    }

    if (p.y>0) {
        filter[index++] = plane_hypotheses[up].w;
    }
    if (p.y>2) {
        filter[index++] = plane_hypotheses[upup].w;
    }
    if (p.y>4) {
        filter[index++] = plane_hypotheses[upup-width*2].w;
    }
    if (p.y<height-1) {
        filter[index++] = plane_hypotheses[down].w;
    }
    if (p.y<height-3) {
        filter[index++] = plane_hypotheses[downdown].w;
    }
    if (p.y<height-5) {
        filter[index++] = plane_hypotheses[downdown+width*2].w;
    }
    if (p.x>0) {
        filter[index++] = plane_hypotheses[left].w;
    }
    if (p.x>2) {
        filter[index++] = plane_hypotheses[leftleft].w;
    }
    if (p.x>4) {
        filter[index++] = plane_hypotheses[leftleft-2].w;
    }
    if (p.x<width-1) {
        filter[index++] = plane_hypotheses[right].w;
    }
    if (p.x<width-3) {
        filter[index++] = plane_hypotheses[rightright].w;
    }
    if (p.x<width-5) {
        filter[index++] = plane_hypotheses[rightright+2].w;
    }
    if (p.y>0 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[up+2].w;
    }
    if (p.y< height-1 &&
        p.x<width-2) {
        filter[index++] = plane_hypotheses[down+2].w;
    }
    if (p.y>0 &&
        p.x>1)
    {
        filter[index++] = plane_hypotheses[up-2].w;
    }
    if (p.y<height-1 &&
        p.x>1) {
        filter[index++] = plane_hypotheses[down-2].w;
    }
    if (p.x>0 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[left  - width*2].w;
    }
    if (p.x<width-1 &&
        p.y>2)
    {
        filter[index++] = plane_hypotheses[right - width*2].w;
    }
    if (p.x>0 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[left  + width*2].w;
    }
    if (p.x<width-1 &&
        p.y<height-2) {
        filter[index++] = plane_hypotheses[right + width*2].w;
    }

    sort_small(filter,index);
    int median_index = index / 2;
    if (index % 2 == 0) {
        plane_hypotheses[center].w = (filter[median_index-1] + filter[median_index]) / 2;
    } else {
        plane_hypotheses[center].w = filter[median_index];
    }
}

__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardFilter(cameras, plane_hypotheses, costs, p);
}

void ACMMP::RunPatchMatch()
{
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    int BLOCK_W = 32;
    int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_size_randinit;
    grid_size_randinit.x = (width + 16 - 1) / 16;
    grid_size_randinit.y= (height + 16 - 1) / 16;
    grid_size_randinit.z = 1;
    dim3 block_size_randinit;
    block_size_randinit.x = 16;
    block_size_randinit.y = 16;
    block_size_randinit.z = 1;

    dim3 grid_size_checkerboard;
    grid_size_checkerboard.x = (width + BLOCK_W - 1) / BLOCK_W;
    grid_size_checkerboard.y= ( (height / 2) + BLOCK_H - 1) / BLOCK_H;
    grid_size_checkerboard.z = 1;
    dim3 block_size_checkerboard;
    block_size_checkerboard.x = BLOCK_W;
    block_size_checkerboard.y = BLOCK_H;
    block_size_checkerboard.z = 1;

    int max_iterations = params.max_iterations;

    RandomInitialization<<<grid_size_randinit, block_size_randinit>>>(texture_objects_cuda, cameras_cuda, plane_hypotheses_cuda, scaled_plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    for (int i = 0; i < max_iterations; ++i) {
        BlackPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, pre_costs_cuda, rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        RedPixelUpdate<<<grid_size_checkerboard, block_size_checkerboard>>>(texture_objects_cuda, texture_depths_cuda, cameras_cuda, plane_hypotheses_cuda, costs_cuda, pre_costs_cuda,rand_states_cuda, selected_views_cuda, prior_planes_cuda, plane_masks_cuda, params, i);
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        printf("iteration: %d\n", i);
    }

    GetDepthandNormal<<<grid_size_randinit, block_size_randinit>>>(cameras_cuda, plane_hypotheses_cuda, params);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    BlackPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
    RedPixelFilter<<<grid_size_checkerboard, block_size_checkerboard>>>(cameras_cuda, plane_hypotheses_cuda, costs_cuda);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());

    cudaMemcpy(plane_hypotheses_host, plane_hypotheses_cuda, sizeof(float4) * width * height, cudaMemcpyDeviceToHost);
    cudaMemcpy(costs_host, costs_cuda, sizeof(float) * width * height, cudaMemcpyDeviceToHost);
    CUDA_SAFE_CALL(cudaDeviceSynchronize());
}

__global__ void JBU_cu(JBUParameters *jp, JBUTexObj *jt, float *depth)
{
    const int2 p = make_int2 ( blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y );
    const int rows = jp[0].height;
    const int cols = jp[0].width;
    const int center = p.y * cols + p.x;

    if (p.x >= cols) {
        return;
    }
    if (p.y >= rows) {
        return;
    }

    const float scale  = 1.0 * jp[0].s_width / jp[0].width;
    const float sigmad = 0.50;
    const float sigmar = 25.5;
    const int WinWidth = jp[0].Imagescale * jp[0].Imagescale + 1;
    int num_neighbors = WinWidth / 2;

    const float o_y = p.y * scale;
    const float o_x = p.x * scale;
    const float refPix = tex2D<float>(jt[0].imgs[0], p.x + 0.5f, p.y + 0.5f);
    int r_y = 0;
    int r_ys = 0;
    int r_x = 0;
    int r_xs = 0;
    float sgauss = 0.0, rgauss = 0.0, totalgauss = 0.0;
    float total_val = 0.0, normalizing_factor = 0.0;
    float  srcPix = 0, neighborPix = 0;

    for (int j = -num_neighbors; j <= num_neighbors; ++j) {
        // source
        r_y = o_y + j;
        r_y = (r_y > 0 ? (r_y < jp[0].s_height ? r_y :jp[0].s_height - 1) : 0) ;
        // reference
        r_ys = p.y + j;
        r_ys = (r_ys > 0 ? (r_ys < jp[0].height ? r_ys :jp[0].height - 1) : 0) ;
        for (int i = -num_neighbors; i <= num_neighbors; ++i) {
            // source
            r_x = o_x + i;
            r_x = (r_x > 0 ? (r_x < jp[0].s_width ? r_x : jp[0].s_width - 1) : 0);
           srcPix = tex2D<float>(jt[0].imgs[1], r_x + 0.5f, r_y + 0.5f);
            // refIm
            r_xs = p.x + i;
            r_xs = (r_xs > 0 ? (r_xs < jp[0].width ? r_xs :jp[0].width - 1) : 0) ;
            neighborPix = tex2D<float>(jt[0].imgs[0], r_xs + 0.5f, r_ys + 0.5f);

            sgauss = SpatialGauss(o_x, o_y, r_x, r_y, sigmad);
            rgauss = RangeGauss(fabs(refPix - neighborPix), sigmar);
            totalgauss = sgauss * rgauss;
            normalizing_factor += totalgauss;
            total_val += srcPix * totalgauss;
        }
    }

    depth[center] = total_val / normalizing_factor;

}
void JBU::CudaRun()
{
    int rows = jp_h.height;
    int cols = jp_h.width;

    dim3 grid_size_initrand;
    grid_size_initrand.x= (cols + 16 - 1) / 16;
    grid_size_initrand.y=(rows + 16 - 1) / 16;
    grid_size_initrand.z= 1;
    dim3 block_size_initrand;
    block_size_initrand.x = 16;
    block_size_initrand.y = 16;
    block_size_initrand.z = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    cudaDeviceSynchronize();
    JBU_cu<<< grid_size_initrand, block_size_initrand>>>(jp_d, jt_d, depth_d);
    cudaDeviceSynchronize();

    cudaMemcpy(depth_h, depth_d, sizeof(float) * rows * cols, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Total time needed for computation: %f seconds\n", milliseconds / 1000.f);
}

#include <memory>
#include <unordered_set>
#include <unordered_map>
#include <queue>
#include <map>
#include <set>
#include <vector>
#include <string>
#include <iostream>
#include <fstream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <thread>
#include <future>
#include <atomic>
#include <chrono>
#include <mutex>
#include <algorithm>
#include <cstdio>
#include <cuda_runtime.h>
#include <cuda.h>
#include <condition_variable>
#include <list>

// Thread pool for parallel file I/O
class ThreadPool {
private:
    std::vector<std::thread> workers;
    std::queue<std::function<void()>> tasks;
    std::mutex queue_mutex;
    std::condition_variable condition;
    bool stop;

public:
    ThreadPool(size_t threads = std::thread::hardware_concurrency()) : stop(false) {
        for (size_t i = 0; i < threads; ++i) {
            workers.emplace_back([this] {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock<std::mutex> lock(this->queue_mutex);
                        this->condition.wait(lock, [this] { return this->stop || !this->tasks.empty(); });
                        if (this->stop && this->tasks.empty())
                            return;
                        task = std::move(this->tasks.front());
                        this->tasks.pop();
                    }
                    task();
                }
            });
        }
    }

    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type> {
        using return_type = typename std::result_of<F(Args...)>::type;
        
        auto task = std::make_shared<std::packaged_task<return_type()>>(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );
        
        std::future<return_type> res = task->get_future();
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            if (stop)
                throw std::runtime_error("enqueue on stopped ThreadPool");
            tasks.emplace([task]() { (*task)(); });
        }
        condition.notify_one();
        return res;
    }

    ~ThreadPool() {
        {
            std::unique_lock<std::mutex> lock(queue_mutex);
            stop = true;
        }
        condition.notify_all();
        for (std::thread &worker : workers)
            worker.join();
    }
};

// Persistent GPU buffer manager to avoid allocation overhead
class PersistentGPUBuffers {
private:
    // Texture arrays for batch processing
    cudaTextureObject_t* depth_textures_buffer;
    cudaTextureObject_t* normal_textures_buffer;
    cudaTextureObject_t* image_textures_buffer;
    int* texture_image_ids_buffer;
    
    // Problem data buffers
    int* ref_image_ids_buffer;
    int* all_src_image_ids_buffer;
    int* src_counts_buffer;
    int* src_offsets_buffer;
    int* problem_offsets_buffer;
    int* widths_buffer;
    int* heights_buffer;
    
    // Output buffers
    PointList* output_points_buffer;
    int* valid_flags_buffer;
    
    // Buffer sizes
    size_t max_textures;
    size_t max_problems;
    size_t max_src_images;
    size_t max_pixels;
    
    bool buffers_allocated;

public:
    PersistentGPUBuffers() : buffers_allocated(false) {
        // Initialize all pointers to nullptr
        depth_textures_buffer = nullptr;
        normal_textures_buffer = nullptr;
        image_textures_buffer = nullptr;
        texture_image_ids_buffer = nullptr;
        ref_image_ids_buffer = nullptr;
        all_src_image_ids_buffer = nullptr;
        src_counts_buffer = nullptr;
        src_offsets_buffer = nullptr;
        problem_offsets_buffer = nullptr;
        widths_buffer = nullptr;
        heights_buffer = nullptr;
        output_points_buffer = nullptr;
        valid_flags_buffer = nullptr;
    }
    
    void allocateBuffers(size_t est_max_textures, size_t est_max_problems, 
                        size_t est_max_src_images, size_t est_max_pixels) {
        if (buffers_allocated) return;
        
        // Add 50% buffer to avoid reallocations
        max_textures = est_max_textures * 1.5;
        max_problems = est_max_problems * 1.5;
        max_src_images = est_max_src_images * 1.5;
        max_pixels = est_max_pixels * 1.5;
        
        std::cout << "[PersistentGPU] Allocating buffers for max: " 
                  << max_textures << " textures, " << max_problems << " problems, "
                  << max_pixels << " pixels" << std::endl;
        
        // Allocate all buffers once
        CUDA_SAFE_CALL(cudaMalloc(&depth_textures_buffer, max_textures * sizeof(cudaTextureObject_t)));
        CUDA_SAFE_CALL(cudaMalloc(&normal_textures_buffer, max_textures * sizeof(cudaTextureObject_t)));
        CUDA_SAFE_CALL(cudaMalloc(&image_textures_buffer, max_textures * sizeof(cudaTextureObject_t)));
        CUDA_SAFE_CALL(cudaMalloc(&texture_image_ids_buffer, max_textures * sizeof(int)));
        
        CUDA_SAFE_CALL(cudaMalloc(&ref_image_ids_buffer, max_problems * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&all_src_image_ids_buffer, max_src_images * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&src_counts_buffer, max_problems * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&src_offsets_buffer, max_problems * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&problem_offsets_buffer, max_problems * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&widths_buffer, max_problems * sizeof(int)));
        CUDA_SAFE_CALL(cudaMalloc(&heights_buffer, max_problems * sizeof(int)));
        
        CUDA_SAFE_CALL(cudaMalloc(&output_points_buffer, max_pixels * sizeof(PointList)));
        CUDA_SAFE_CALL(cudaMalloc(&valid_flags_buffer, max_pixels * sizeof(int)));
        
        buffers_allocated = true;
        
        // Check allocated memory
        size_t free_mem, total_mem;
        cudaMemGetInfo(&free_mem, &total_mem);
        std::cout << "[PersistentGPU] Buffers allocated. GPU memory: " 
                  << free_mem / (1024*1024) << " MB free / " 
                  << total_mem / (1024*1024) << " MB total" << std::endl;
    }
    
    ~PersistentGPUBuffers() {
        if (buffers_allocated) {
            cudaFree(depth_textures_buffer);
            cudaFree(normal_textures_buffer);
            cudaFree(image_textures_buffer);
            cudaFree(texture_image_ids_buffer);
            cudaFree(ref_image_ids_buffer);
            cudaFree(all_src_image_ids_buffer);
            cudaFree(src_counts_buffer);
            cudaFree(src_offsets_buffer);
            cudaFree(problem_offsets_buffer);
            cudaFree(widths_buffer);
            cudaFree(heights_buffer);
            cudaFree(output_points_buffer);
            cudaFree(valid_flags_buffer);
        }
    }
    
    // Get buffers with size checking
    cudaTextureObject_t* getDepthTexturesBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return depth_textures_buffer;
    }
    
    cudaTextureObject_t* getNormalTexturesBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return normal_textures_buffer;
    }
    
    cudaTextureObject_t* getImageTexturesBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return image_textures_buffer;
    }
    
    int* getTextureImageIdsBuffer(size_t needed) {
        if (needed > max_textures) {
            throw std::runtime_error("Texture buffer size exceeded");
        }
        return texture_image_ids_buffer;
    }
    
    int* getRefImageIdsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return ref_image_ids_buffer;
    }
    
    int* getAllSrcImageIdsBuffer(size_t needed) {
        if (needed > max_src_images) {
            throw std::runtime_error("Source images buffer size exceeded");
        }
        return all_src_image_ids_buffer;
    }
    
    int* getSrcCountsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return src_counts_buffer;
    }
    
    int* getSrcOffsetsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return src_offsets_buffer;
    }
    
    int* getProblemOffsetsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return problem_offsets_buffer;
    }
    
    int* getWidthsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return widths_buffer;
    }
    
    int* getHeightsBuffer(size_t needed) {
        if (needed > max_problems) {
            throw std::runtime_error("Problem buffer size exceeded");
        }
        return heights_buffer;
    }
    
    PointList* getOutputPointsBuffer(size_t needed) {
        if (needed > max_pixels) {
            throw std::runtime_error("Output buffer size exceeded");
        }
        return output_points_buffer;
    }
    
    int* getValidFlagsBuffer(size_t needed) {
        if (needed > max_pixels) {
            throw std::runtime_error("Valid flags buffer size exceeded");
        }
        return valid_flags_buffer;
    }
};

// Image data structure for efficient caching
struct ImageData {
    Camera camera;
    cv::Mat_<float> depth;
    cv::Mat_<cv::Vec3f> normal;
    cv::Mat image;
    bool valid;
    std::chrono::steady_clock::time_point last_accessed;
    
    ImageData() : valid(false) {}
};

// Optimized data loader with parallel I/O and LRU cache
class OptimizedDataLoader {
private:
    std::string dense_folder;
    std::string img_folder; 
    std::string cam_folder;
    bool geom_consistency;
    
    // LRU cache with thread safety
    std::unordered_map<int, std::shared_ptr<ImageData>> cache;
    std::list<int> lru_order;
    std::unordered_map<int, std::list<int>::iterator> lru_map;
    
    size_t max_cache_size;
    mutable std::mutex cache_mutex;
    
    // Thread pool for parallel I/O
    std::unique_ptr<ThreadPool> thread_pool;
    
    void updateLRU(int image_id) {
        auto it = lru_map.find(image_id);
        if (it != lru_map.end()) {
            lru_order.erase(it->second);
        }
        lru_order.push_front(image_id);
        lru_map[image_id] = lru_order.begin();
    }
    
    void trimCache() {
        while (cache.size() > max_cache_size && !lru_order.empty()) {
            int oldest = lru_order.back();
            lru_order.pop_back();
            lru_map.erase(oldest);
            cache.erase(oldest);
        }
    }
    
    std::shared_ptr<ImageData> loadImageDataSync(int image_id) {
        auto data = std::make_shared<ImageData>();
        data->valid = false;
        
        // Load camera
        char buf[256];
        sprintf(buf, "%s/%08d_cam.txt", cam_folder.c_str(), image_id);
        try {
            data->camera = ReadCamera(std::string(buf));
            if (data->camera.width <= 0 || data->camera.height <= 0) {
                return data;
            }
        } catch (...) {
            return data;
        }
        
        // Load depth
        std::string depth_suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
        sprintf(buf, "%s/ACMMP/2333_%08d%s", dense_folder.c_str(), image_id, depth_suffix.c_str());
        if (readDepthDmb(std::string(buf), data->depth) != 0 || 
            data->depth.cols <= 0 || data->depth.rows <= 0) {
            return data;
        }
        
        // Load normal
        sprintf(buf, "%s/ACMMP/2333_%08d/normals.dmb", dense_folder.c_str(), image_id);
        if (readNormalDmb(std::string(buf), data->normal) != 0 || 
            data->normal.cols <= 0 || data->normal.rows <= 0) {
            return data;
        }
        
        // Load image
        sprintf(buf, "%s/%08d.jpg", img_folder.c_str(), image_id);
        data->image = cv::imread(std::string(buf), cv::IMREAD_COLOR);
        if (data->image.empty()) {
            return data;
        }
        
        // Rescale image to match depth
        cv::Mat_<cv::Vec3b> img_color;
        if (data->image.channels() == 3) {
            img_color = cv::Mat_<cv::Vec3b>(data->image);
        } else {
            cv::cvtColor(data->image, img_color, cv::COLOR_GRAY2BGR);
        }
        cv::Mat_<cv::Vec3b> scaled_color;
        RescaleImageAndCamera(img_color, scaled_color, data->depth, data->camera);
        data->image = cv::Mat(scaled_color);
        
        data->valid = true;
        data->last_accessed = std::chrono::steady_clock::now();
        return data;
    }
    
public:
    OptimizedDataLoader(const std::string& folder, bool geom = false, size_t cache_size = 100) 
        : dense_folder(folder), geom_consistency(geom), max_cache_size(cache_size) {
        img_folder = folder + "/images";
        cam_folder = folder + "/cams";
        
        // Create thread pool with optimal number of threads for I/O
        unsigned int hw_threads = std::thread::hardware_concurrency();
        size_t io_threads = 16;
        if (hw_threads >= 4) {
            io_threads = std::min(static_cast<size_t>(16), static_cast<size_t>(hw_threads));
        } else {
            io_threads = 4;
        }
        thread_pool = std::unique_ptr<ThreadPool>(new ThreadPool(io_threads));
        
        std::cout << "[OptimizedLoader] Using " << io_threads << " threads for parallel I/O" << std::endl;
    }
    
    // Parallel preload for a chunk of images
    void preloadChunkParallel(const std::vector<int>& image_ids) {
        std::vector<std::future<std::shared_ptr<ImageData>>> futures;
        std::mutex results_mutex;
        
        std::cout << "  Preloading " << image_ids.size() << " images in parallel..." << std::endl;
        auto start_time = std::chrono::high_resolution_clock::now();
        
        // Launch parallel loading tasks
        for (int image_id : image_ids) {
            {
                std::lock_guard<std::mutex> lock(cache_mutex);
                if (cache.find(image_id) != cache.end()) {
                    // Already cached, skip
                    continue;
                }
            }
            
            // Submit loading task to thread pool
            auto future = thread_pool->enqueue([this, image_id]() {
                return loadImageDataSync(image_id);
            });
            futures.push_back(std::move(future));
        }
        
        // Collect results and store in cache
        size_t loaded_count = 0;
        for (size_t i = 0; i < futures.size(); ++i) {
            int image_id = image_ids[i];
            auto data = futures[i].get();
            
            if (data && data->valid) {
                std::lock_guard<std::mutex> lock(cache_mutex);
                cache[image_id] = data;
                updateLRU(image_id);
                loaded_count++;
            }
        }
        
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            trimCache();
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
        
        std::cout << "  Loaded " << loaded_count << "/" << image_ids.size() 
                  << " images in " << duration.count() << " ms" << std::endl;
    }
    
    bool getData(int image_id, Camera& cam, cv::Mat_<float>& depth, 
                cv::Mat_<cv::Vec3f>& normal, cv::Mat& image) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        
        auto it = cache.find(image_id);
        if (it == cache.end() || !it->second->valid) {
            return false;
        }
        
        auto data = it->second;
        cam = data->camera;
        depth = data->depth;
        normal = data->normal;
        image = data->image;
        
        updateLRU(image_id);
        data->last_accessed = std::chrono::steady_clock::now();
        
        return true;
    }
    
    size_t getCacheSize() const {
        std::lock_guard<std::mutex> lock(cache_mutex);
        return cache.size();
    }
    
    void clearCache() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        cache.clear();
        lru_order.clear();
        lru_map.clear();
    }
};

// Optimized texture manager with reduced allocation overhead
class OptimizedTextureManager {
private:
    struct TextureData {
        cudaArray* depth_array;
        cudaArray* normal_array;
        cudaArray* image_array;
        cudaTextureObject_t depth_texture;
        cudaTextureObject_t normal_texture;
        cudaTextureObject_t image_texture;
        bool is_valid;
        
        TextureData() : depth_array(nullptr), normal_array(nullptr), image_array(nullptr),
                       depth_texture(0), normal_texture(0), image_texture(0), is_valid(false) {}
        
        void cleanup() {
            if (depth_texture != 0) {
                cudaDestroyTextureObject(depth_texture);
                depth_texture = 0;
            }
            if (normal_texture != 0) {
                cudaDestroyTextureObject(normal_texture);
                normal_texture = 0;
            }
            if (image_texture != 0) {
                cudaDestroyTextureObject(image_texture);
                image_texture = 0;
            }
            
            if (depth_array != nullptr) {
                cudaFreeArray(depth_array);
                depth_array = nullptr;
            }
            if (normal_array != nullptr) {
                cudaFreeArray(normal_array);
                normal_array = nullptr;
            }
            if (image_array != nullptr) {
                cudaFreeArray(image_array);
                image_array = nullptr;
            }
            
            is_valid = false;
        }
    };
    
    std::vector<TextureData> textures;
    std::vector<int> current_image_ids;
    bool loaded = false;
    
    // Use multiple streams for parallel texture creation
    static const int num_streams = 4;
    cudaStream_t streams[num_streams];
    
    void release() {
        if (!loaded && textures.empty()) {
            return;
        }
        
        // Synchronize all streams
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        cudaDeviceSynchronize();
        
        // Clean up all textures
        for (auto& tex : textures) {
            tex.cleanup();
        }
        
        textures.clear();
        current_image_ids.clear();
        loaded = false;
    }

public:
    OptimizedTextureManager() {
        for (int i = 0; i < num_streams; ++i) {
            CUDA_SAFE_CALL(cudaStreamCreate(&streams[i]));
        }
    }
    
    ~OptimizedTextureManager() {
        release();
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamDestroy(streams[i]);
        }
    }
    
    bool loadChunk(const std::vector<int>& image_ids, OptimizedDataLoader& loader) {
        release();  // Clean up previous chunk
        
        if (image_ids.empty()) {
            return false;
        }
        
        current_image_ids = image_ids;
        textures.resize(image_ids.size());
        
        bool overall_success = true;
        size_t successful_textures = 0;
        
        // Process textures in parallel using streams
        for (size_t i = 0; i < image_ids.size(); ++i) {
            int image_id = image_ids[i];
            cudaStream_t stream = streams[i % num_streams];
            
            Camera cam;
            cv::Mat_<float> depth;
            cv::Mat_<cv::Vec3f> normal;
            cv::Mat image;
            
            if (!loader.getData(image_id, cam, depth, normal, image)) {
                continue;
            }
            
            TextureData& tex = textures[i];
            bool texture_success = true;
            
            // Create textures with error handling
            try {
                // Create depth texture
                cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc<float>();
                CUDA_SAFE_CALL(cudaMallocArray(&tex.depth_array, &depth_desc, cam.width, cam.height));
                CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(tex.depth_array, 0, 0, 
                                                       depth.ptr<float>(), depth.step[0], 
                                                       cam.width * sizeof(float), cam.height, 
                                                       cudaMemcpyHostToDevice, stream));
                
                cudaResourceDesc depth_res_desc = {};
                depth_res_desc.resType = cudaResourceTypeArray;
                depth_res_desc.res.array.array = tex.depth_array;
                
                cudaTextureDesc depth_tex_desc = {};
                depth_tex_desc.addressMode[0] = cudaAddressModeWrap;
                depth_tex_desc.addressMode[1] = cudaAddressModeClamp;
                depth_tex_desc.filterMode = cudaFilterModePoint;
                depth_tex_desc.readMode = cudaReadModeElementType;
                depth_tex_desc.normalizedCoords = false;
                
                CUDA_SAFE_CALL(cudaCreateTextureObject(&tex.depth_texture, &depth_res_desc, &depth_tex_desc, NULL));
                
                // Create normal texture
                cv::Mat normal_rgba = cv::Mat(normal.rows, normal.cols, CV_32FC4, cv::Scalar(0.0f, 0.0f, 0.0f, 0.0f));
                cv::Mat src_mats[1] = {normal};
                cv::Mat dst_mats[1] = {normal_rgba};
                int from_to[6] = {0, 0, 1, 1, 2, 2};
                cv::mixChannels(src_mats, 1, dst_mats, 1, from_to, 3);
                
                cudaChannelFormatDesc normal_desc = cudaCreateChannelDesc<float4>();
                CUDA_SAFE_CALL(cudaMallocArray(&tex.normal_array, &normal_desc, cam.width, cam.height));
                CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(tex.normal_array, 0, 0, 
                                                       normal_rgba.ptr<float>(), normal_rgba.step[0], 
                                                       cam.width * sizeof(float4), cam.height, 
                                                       cudaMemcpyHostToDevice, stream));
                
                cudaResourceDesc normal_res_desc = {};
                normal_res_desc.resType = cudaResourceTypeArray;
                normal_res_desc.res.array.array = tex.normal_array;
                
                cudaTextureDesc normal_tex_desc = {};
                normal_tex_desc.addressMode[0] = cudaAddressModeWrap;
                normal_tex_desc.addressMode[1] = cudaAddressModeClamp;
                normal_tex_desc.filterMode = cudaFilterModePoint;
                normal_tex_desc.readMode = cudaReadModeElementType;
                normal_tex_desc.normalizedCoords = false;
                
                CUDA_SAFE_CALL(cudaCreateTextureObject(&tex.normal_texture, &normal_res_desc, &normal_tex_desc, NULL));
                
                // Create image texture
                cv::Mat rgba, rgba_float;
                cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
                rgba.convertTo(rgba_float, CV_32FC4, 1.0/255.0);
                
                cudaChannelFormatDesc image_desc = cudaCreateChannelDesc<float4>();
                CUDA_SAFE_CALL(cudaMallocArray(&tex.image_array, &image_desc, cam.width, cam.height));
                CUDA_SAFE_CALL(cudaMemcpy2DToArrayAsync(tex.image_array, 0, 0, 
                                                       rgba_float.ptr<float>(), rgba_float.step[0], 
                                                       cam.width * sizeof(float4), cam.height, 
                                                       cudaMemcpyHostToDevice, stream));
                
                cudaResourceDesc image_res_desc = {};
                image_res_desc.resType = cudaResourceTypeArray;
                image_res_desc.res.array.array = tex.image_array;
                
                cudaTextureDesc image_tex_desc = {};
                image_tex_desc.addressMode[0] = cudaAddressModeWrap;
                image_tex_desc.addressMode[1] = cudaAddressModeClamp;
                image_tex_desc.filterMode = cudaFilterModeLinear;
                image_tex_desc.readMode = cudaReadModeElementType;
                image_tex_desc.normalizedCoords = false;
                
                CUDA_SAFE_CALL(cudaCreateTextureObject(&tex.image_texture, &image_res_desc, &image_tex_desc, NULL));
                
                tex.is_valid = true;
                successful_textures++;
                
            } catch (const std::exception& e) {
                std::cerr << "    Failed to create texture for image " << image_id << ": " << e.what() << std::endl;
                tex.cleanup();
                texture_success = false;
                overall_success = false;
            }
        }
        
        // Sync all streams
        for (int i = 0; i < num_streams; ++i) {
            cudaStreamSynchronize(streams[i]);
        }
        
        std::cout << "    Successfully loaded " << successful_textures << "/" << image_ids.size() << " textures" << std::endl;
        
        loaded = (successful_textures > 0);
        return loaded;
    }
    
    std::vector<cudaTextureObject_t> getDepthTextures() const {
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex.is_valid && tex.depth_texture != 0) {
                result.push_back(tex.depth_texture);
            }
        }
        return result;
    }
    
    std::vector<cudaTextureObject_t> getNormalTextures() const {
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex.is_valid && tex.normal_texture != 0) {
                result.push_back(tex.normal_texture);
            }
        }
        return result;
    }
    
    std::vector<cudaTextureObject_t> getImageTextures() const {
        std::vector<cudaTextureObject_t> result;
        for (const auto& tex : textures) {
            if (tex.is_valid && tex.image_texture != 0) {
                result.push_back(tex.image_texture);
            }
        }
        return result;
    }
    
    std::vector<int> getValidImageIds() const {
        std::vector<int> result;
        for (size_t i = 0; i < textures.size() && i < current_image_ids.size(); ++i) {
            if (textures[i].is_valid) {
                result.push_back(current_image_ids[i]);
            }
        }
        return result;
    }
};

// Fast batch kernel (unchanged from original)
__global__ void ChunkBatchKernel(
    cudaTextureObject_t* depth_textures,
    cudaTextureObject_t* normal_textures, 
    cudaTextureObject_t* image_textures,
    int* texture_image_ids,
    int num_textures,
    Camera* cameras,
    int* camera_image_ids,
    int num_cameras,
    int* ref_image_ids,
    int* src_image_ids,
    int* src_counts,
    int* src_offsets,
    int* problem_offsets,
    int* widths,
    int* heights,
    PointList* output_points,
    int* valid_flags,
    int num_problems_in_chunk
) {
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Find which problem this thread belongs to
    int problem_id = 0;
    while (problem_id < num_problems_in_chunk - 1 && global_idx >= problem_offsets[problem_id + 1]) {
        problem_id++;
    }
    
    if (problem_id >= num_problems_in_chunk) return;
    
    int local_idx = global_idx - problem_offsets[problem_id];
    int width = widths[problem_id];
    int height = heights[problem_id];
    
    if (local_idx >= width * height) return;
    
    int c = local_idx % width;
    int r = local_idx / width;
    
    int ref_image_id = ref_image_ids[problem_id];
    
    // Find reference camera and texture
    int ref_cam_idx = -1;
    int ref_tex_idx = -1;
    
    for (int i = 0; i < num_cameras; ++i) {
        if (camera_image_ids[i] == ref_image_id) {
            ref_cam_idx = i;
            break;
        }
    }
    
    for (int i = 0; i < num_textures; ++i) {
        if (texture_image_ids[i] == ref_image_id) {
            ref_tex_idx = i;
            break;
        }
    }
    
    if (ref_cam_idx == -1 || ref_tex_idx == -1) {
        valid_flags[global_idx] = 0;
        return;
    }
    
    const Camera& ref_cam = cameras[ref_cam_idx];
    
    // Sample reference depth
    float ref_depth = tex2D<float>(depth_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    
    if (ref_depth <= 0.0f) {
        valid_flags[global_idx] = 0;
        return;
    }

    // Get 3D point in world coordinates
    float3 PointX = Get3DPointonWorld_cu(static_cast<float>(c), static_cast<float>(r), ref_depth, ref_cam);
    
    // Sample reference normal and color
    float4 ref_normal_tex = tex2D<float4>(normal_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    float3 ref_normal = make_float3(ref_normal_tex.x, ref_normal_tex.y, ref_normal_tex.z);
    
    float4 ref_color = tex2D<float4>(image_textures[ref_tex_idx], c + 0.5f, r + 0.5f);
    
    // Initialize sums for averaging
    float3 point_sum = PointX;
    float3 normal_sum = ref_normal;
    float color_sum[3] = {
        ref_color.z * 255.0f,  // R
        ref_color.y * 255.0f,  // G
        ref_color.x * 255.0f   // B
    };
    int num_consistent = 1;
    
    // Check source images for this problem
    int src_start = src_offsets[problem_id];
    int src_count = src_counts[problem_id];
    
    for (int j = 0; j < src_count; ++j) {
        int src_image_id = src_image_ids[src_start + j];
        
        // Find source camera and texture
        int src_cam_idx = -1;
        int src_tex_idx = -1;
        
        for (int i = 0; i < num_cameras; ++i) {
            if (camera_image_ids[i] == src_image_id) {
                src_cam_idx = i;
                break;
            }
        }
        
        for (int i = 0; i < num_textures; ++i) {
            if (texture_image_ids[i] == src_image_id) {
                src_tex_idx = i;
                break;
            }
        }
        
        if (src_cam_idx == -1 || src_tex_idx == -1) continue;
        
        const Camera& src_cam = cameras[src_cam_idx];
        
        // Project and check consistency
        float2 proj_point;
        float proj_depth_in_src;
        ProjectonCamera_cu(PointX, src_cam, proj_point, proj_depth_in_src);
        
        int src_c = static_cast<int>(proj_point.x + 0.5f);
        int src_r = static_cast<int>(proj_point.y + 0.5f);
        
        if (src_c < 0 || src_c >= src_cam.width || src_r < 0 || src_r >= src_cam.height) 
            continue;
        
        float src_depth = tex2D<float>(depth_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
        if (src_depth <= 0.0f) continue;
        
        float3 PointX_src = Get3DPointonWorld_cu(static_cast<float>(src_c), static_cast<float>(src_r), src_depth, src_cam);
        
        float2 reproj_point_in_ref;
        float dummy_depth;
        ProjectonCamera_cu(PointX_src, ref_cam, reproj_point_in_ref, dummy_depth);
        
        float reproj_error = hypotf(c - reproj_point_in_ref.x, r - reproj_point_in_ref.y);
        float relative_depth_diff = fabsf(proj_depth_in_src - src_depth) / src_depth;
        
        float4 src_normal_tex = tex2D<float4>(normal_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
        float3 src_normal = make_float3(src_normal_tex.x, src_normal_tex.y, src_normal_tex.z);
        
        float dot_product = ref_normal.x * src_normal.x + ref_normal.y * src_normal.y + ref_normal.z * src_normal.z;
        dot_product = fmaxf(-1.0f, fminf(1.0f, dot_product));
        float angle = acosf(dot_product);
        
        if (reproj_error < 1.0 && relative_depth_diff < 0.005f && angle < 0.05f) {
            point_sum.x += PointX_src.x;
            point_sum.y += PointX_src.y;
            point_sum.z += PointX_src.z;
            
            normal_sum.x += src_normal.x;
            normal_sum.y += src_normal.y;
            normal_sum.z += src_normal.z;
            
            float4 src_color = tex2D<float4>(image_textures[src_tex_idx], src_c + 0.5f, src_r + 0.5f);
            color_sum[0] += src_color.z * 255.0f;
            color_sum[1] += src_color.y * 255.0f;
            color_sum[2] += src_color.x * 255.0f;
            
            num_consistent++;
        }
    }
    
    if (num_consistent >= 3) {
        PointList final_point;
        
        final_point.coord = make_float3(
            point_sum.x / num_consistent,
            point_sum.y / num_consistent,
            point_sum.z / num_consistent
        );
        
        float3 avg_normal = make_float3(
            normal_sum.x / num_consistent,
            normal_sum.y / num_consistent,
            normal_sum.z / num_consistent
        );
        float normal_length = hypotf(hypotf(avg_normal.x, avg_normal.y), avg_normal.z);
        if (normal_length > 0.0f) {
            avg_normal.x /= normal_length;
            avg_normal.y /= normal_length;
            avg_normal.z /= normal_length;
        }
        final_point.normal = avg_normal;
        
        final_point.color = make_float3(
            color_sum[0] / num_consistent,
            color_sum[1] / num_consistent,
            color_sum[2] / num_consistent
        );
        
        output_points[global_idx] = final_point;
        valid_flags[global_idx] = 1;
    } else {
        valid_flags[global_idx] = 0;
    }
}

// Smart chunking strategy
std::vector<std::vector<size_t>> createSmartChunks(const std::vector<Problem>& problems, 
                                                   size_t max_images_per_chunk) {
    std::vector<std::vector<size_t>> chunks;
    std::vector<size_t> current_chunk;
    std::unordered_set<int> current_images;
    
    for (size_t i = 0; i < problems.size(); ++i) {
        std::unordered_set<int> problem_images;
        problem_images.insert(problems[i].ref_image_id);
        for (int src_id : problems[i].src_image_ids) {
            problem_images.insert(src_id);
        }
        
        // Check if adding this problem would exceed image limit
        std::unordered_set<int> combined_images = current_images;
        combined_images.insert(problem_images.begin(), problem_images.end());
        
        if (combined_images.size() > max_images_per_chunk && !current_chunk.empty()) {
            // Start new chunk
            chunks.push_back(current_chunk);
            current_chunk.clear();
            current_images.clear();
        }
        
        // Add problem to current chunk
        current_chunk.push_back(i);
        current_images.insert(problem_images.begin(), problem_images.end());
    }
    
    if (!current_chunk.empty()) {
        chunks.push_back(current_chunk);
    }
    
    return chunks;
}

// Main optimized fusion function
void RunFusionCuda(const std::string &dense_folder,
                           const std::vector<Problem> &problems,
                           bool geom_consistency,
                           size_t max_images_per_chunk)
{
    std::cout << "[Optimized Fusion] Starting with " << problems.size() << " problems..." << std::endl;
    
    // Estimate buffer sizes
    size_t est_max_textures = max_images_per_chunk;
    size_t est_max_problems = 0;
    size_t est_max_src_images = 0;
    size_t est_max_pixels = 0;
    
    auto chunks = createSmartChunks(problems, max_images_per_chunk);
    
    for (const auto& chunk : chunks) {
        est_max_problems = std::max(est_max_problems, chunk.size());
        size_t chunk_src_images = 0;
        size_t chunk_pixels = 0;
        
        for (size_t prob_idx : chunk) {
            const Problem& problem = problems[prob_idx];
            chunk_src_images += problem.src_image_ids.size();
            
            // Estimate pixels (assume average image size)
            chunk_pixels += 3200 * 1600;  // Conservative estimate
        }
        
        est_max_src_images = std::max(est_max_src_images, chunk_src_images);
        est_max_pixels = std::max(est_max_pixels, chunk_pixels);
    }
    
    std::cout << "[Optimized Fusion] Created " << chunks.size() << " chunks" << std::endl;
    std::cout << "[Optimized Fusion] Estimated max: " << est_max_textures << " textures, " 
              << est_max_problems << " problems, " << est_max_pixels << " pixels" << std::endl;
    
    // Initialize optimized managers
    OptimizedDataLoader loader(dense_folder, geom_consistency, 200);
    OptimizedTextureManager texture_manager;
    PersistentGPUBuffers gpu_buffers;
    
    // Allocate persistent GPU buffers once
    gpu_buffers.allocateBuffers(est_max_textures, est_max_problems, est_max_src_images, est_max_pixels);
    
    // Pre-load all cameras (they're small)
    std::unordered_set<int> all_image_ids;
    for (const auto& problem : problems) {
        all_image_ids.insert(problem.ref_image_id);
        for (int src_id : problem.src_image_ids) {
            all_image_ids.insert(src_id);
        }
    }
    
    std::vector<Camera> all_cameras;
    std::vector<int> camera_image_ids;
    for (int image_id : all_image_ids) {
        char buf[256];
        sprintf(buf, "%s/cams/%08d_cam.txt", dense_folder.c_str(), image_id);
        try {
            Camera cam = ReadCamera(std::string(buf));
            if (cam.width > 0 && cam.height > 0) {
                all_cameras.push_back(cam);
                camera_image_ids.push_back(image_id);
            }
        } catch (...) {
            continue;
        }
    }
    
    // Copy cameras to GPU once
    Camera* cameras_cuda;
    int* camera_image_ids_cuda;
    CUDA_SAFE_CALL(cudaMalloc(&cameras_cuda, all_cameras.size() * sizeof(Camera)));
    CUDA_SAFE_CALL(cudaMalloc(&camera_image_ids_cuda, camera_image_ids.size() * sizeof(int)));
    CUDA_SAFE_CALL(cudaMemcpy(cameras_cuda, all_cameras.data(), 
                               all_cameras.size() * sizeof(Camera), cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(camera_image_ids_cuda, camera_image_ids.data(), 
                               camera_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
    
    std::vector<PointList> all_points;
    auto total_start = std::chrono::high_resolution_clock::now();
    
    // Process each chunk with optimized pipeline
    for (size_t chunk_idx = 0; chunk_idx < chunks.size(); ++chunk_idx) {
        const auto& chunk = chunks[chunk_idx];
        
        std::cout << "[Optimized Fusion] Processing chunk " << (chunk_idx + 1) << "/" << chunks.size() 
                  << " (" << chunk.size() << " problems)" << std::endl;
        
        // Get unique images for this chunk
        std::unordered_set<int> chunk_images;
        for (size_t prob_idx : chunk) {
            const Problem& problem = problems[prob_idx];
            chunk_images.insert(problem.ref_image_id);
            for (int src_id : problem.src_image_ids) {
                chunk_images.insert(src_id);
            }
        }
        
        std::vector<int> chunk_image_ids(chunk_images.begin(), chunk_images.end());
        std::cout << "  Chunk images: " << chunk_image_ids.size() << std::endl;
        
        // Parallel preload chunk data
        loader.preloadChunkParallel(chunk_image_ids);
        
        // Load chunk textures
        if (!texture_manager.loadChunk(chunk_image_ids, loader)) {
            std::cerr << "Warning: Failed to load textures for chunk " << chunk_idx << std::endl;
            continue;
        }
        
        // Prepare batch data for this chunk
        std::vector<int> ref_image_ids;
        std::vector<int> all_src_image_ids;
        std::vector<int> src_counts;
        std::vector<int> src_offsets;
        std::vector<int> problem_offsets;
        std::vector<int> widths;
        std::vector<int> heights;
        
        int total_pixels = 0;
        int src_offset = 0;
        
        for (size_t i = 0; i < chunk.size(); ++i) {
            size_t prob_idx = chunk[i];
            const Problem& problem = problems[prob_idx];
            
            ref_image_ids.push_back(problem.ref_image_id);
            
            Camera ref_cam;
            cv::Mat_<float> depth;
            cv::Mat_<cv::Vec3f> normal;
            cv::Mat image;
            if (!loader.getData(problem.ref_image_id, ref_cam, depth, normal, image)) {
                continue;
            }
            
            widths.push_back(ref_cam.width);
            heights.push_back(ref_cam.height);
            problem_offsets.push_back(total_pixels);
            total_pixels += ref_cam.width * ref_cam.height;
            
            src_offsets.push_back(src_offset);
            src_counts.push_back(problem.src_image_ids.size());
            
            for (int src_id : problem.src_image_ids) {
                all_src_image_ids.push_back(src_id);
            }
            src_offset += problem.src_image_ids.size();
        }
        
        if (total_pixels == 0) continue;
        
        // Use persistent GPU buffers instead of allocating new ones
        const auto& depth_textures = texture_manager.getDepthTextures();
        const auto& normal_textures = texture_manager.getNormalTextures();
        const auto& image_textures = texture_manager.getImageTextures();
        const auto& texture_image_ids = texture_manager.getValidImageIds();
        
        if (depth_textures.empty() || normal_textures.empty() || image_textures.empty()) {
            std::cerr << "Warning: No valid textures loaded for chunk " << chunk_idx << std::endl;
            continue;
        }
        
        // Get reusable buffers
        auto depth_textures_cuda = gpu_buffers.getDepthTexturesBuffer(depth_textures.size());
        auto normal_textures_cuda = gpu_buffers.getNormalTexturesBuffer(normal_textures.size());
        auto image_textures_cuda = gpu_buffers.getImageTexturesBuffer(image_textures.size());
        auto texture_image_ids_cuda = gpu_buffers.getTextureImageIdsBuffer(texture_image_ids.size());
        auto ref_image_ids_cuda = gpu_buffers.getRefImageIdsBuffer(ref_image_ids.size());
        auto all_src_image_ids_cuda = gpu_buffers.getAllSrcImageIdsBuffer(all_src_image_ids.size());
        auto src_counts_cuda = gpu_buffers.getSrcCountsBuffer(src_counts.size());
        auto src_offsets_cuda = gpu_buffers.getSrcOffsetsBuffer(src_offsets.size());
        auto problem_offsets_cuda = gpu_buffers.getProblemOffsetsBuffer(problem_offsets.size());
        auto widths_cuda = gpu_buffers.getWidthsBuffer(widths.size());
        auto heights_cuda = gpu_buffers.getHeightsBuffer(heights.size());
        auto output_points_cuda = gpu_buffers.getOutputPointsBuffer(total_pixels);
        auto valid_flags_cuda = gpu_buffers.getValidFlagsBuffer(total_pixels);
        
        // Copy data to GPU (reusing buffers)
        CUDA_SAFE_CALL(cudaMemcpy(depth_textures_cuda, depth_textures.data(), depth_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(normal_textures_cuda, normal_textures.data(), normal_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(image_textures_cuda, image_textures.data(), image_textures.size() * sizeof(cudaTextureObject_t), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(texture_image_ids_cuda, texture_image_ids.data(), texture_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(ref_image_ids_cuda, ref_image_ids.data(), ref_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(all_src_image_ids_cuda, all_src_image_ids.data(), all_src_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(src_counts_cuda, src_counts.data(), src_counts.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(src_offsets_cuda, src_offsets.data(), src_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(problem_offsets_cuda, problem_offsets.data(), problem_offsets.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(widths_cuda, widths.data(), widths.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemcpy(heights_cuda, heights.data(), heights.size() * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_SAFE_CALL(cudaMemset(valid_flags_cuda, 0, total_pixels * sizeof(int)));
        
        // Launch kernel
        int block_size = 256;
        int grid_size = (total_pixels + block_size - 1) / block_size;
        
        auto chunk_start = std::chrono::high_resolution_clock::now();
        
        ChunkBatchKernel<<<grid_size, block_size>>>(
            depth_textures_cuda,
            normal_textures_cuda,
            image_textures_cuda,
            texture_image_ids_cuda,
            (int)texture_image_ids.size(),
            cameras_cuda,
            camera_image_ids_cuda,
            (int)all_cameras.size(),
            ref_image_ids_cuda,
            all_src_image_ids_cuda,
            src_counts_cuda,
            src_offsets_cuda,
            problem_offsets_cuda,
            widths_cuda,
            heights_cuda,
            output_points_cuda,
            valid_flags_cuda,
            (int)chunk.size()
        );
        
        CUDA_SAFE_CALL(cudaDeviceSynchronize());
        
        auto chunk_end = std::chrono::high_resolution_clock::now();
        auto chunk_duration = std::chrono::duration_cast<std::chrono::milliseconds>(chunk_end - chunk_start);
        
        // Copy results back
        std::vector<PointList> chunk_points(total_pixels);
        std::vector<int> valid_flags_host(total_pixels);
        
        CUDA_SAFE_CALL(cudaMemcpy(chunk_points.data(), output_points_cuda, total_pixels * sizeof(PointList), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(valid_flags_host.data(), valid_flags_cuda, total_pixels * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Collect valid points
        size_t chunk_valid_count = 0;
        for (int i = 0; i < total_pixels; ++i) {
            if (valid_flags_host[i]) {
                all_points.push_back(chunk_points[i]);
                chunk_valid_count++;
            }
        }
        
        std::cout << "  Chunk " << (chunk_idx + 1) << ": " << chunk_valid_count 
                  << " points in " << chunk_duration.count() << " ms" << std::endl;
    }
    
    auto total_end = std::chrono::high_resolution_clock::now();
    auto total_duration = std::chrono::duration_cast<std::chrono::seconds>(total_end - total_start);
    
    std::cout << "[Optimized Fusion] Generated " << all_points.size() << " points total" << std::endl;
    std::cout << "[Optimized Fusion] Total time: " << total_duration.count() << " seconds" << std::endl;
    
    // Write output
    std::string output_path = dense_folder + "/ACMMP/ACMM_model_optimized.ply";
    StoreColorPlyFileBinaryPointCloud(output_path, all_points);
    
    std::cout << "[Optimized Fusion] Complete! Output written to: " << output_path << std::endl;
    std::cout << "[Optimized Fusion] Final cache size: " << loader.getCacheSize() << " images" << std::endl;
    
    // Cleanup
    cudaFree(cameras_cuda);
    cudaFree(camera_image_ids_cuda);
}