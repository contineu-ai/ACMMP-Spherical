#include "ACMMP.h"
#include "ACMMP_device.cuh"  // Include the device functions header
#include <math_constants.h>  // for CUDART_PI_F
#include <memory>
// extern __constant__ SphericalLUT* d_lut_array_const[10];
// extern __constant__ int d_num_luts;

__device__ __forceinline__ float SampleDepthInv(curandState* rs, float dmin, float dmax) {
    dmin = fmaxf(dmin, 1e-6f);
    dmax = fmaxf(dmax, dmin + 1e-6f);
    const float inv_min = __fdividef(1.0f, dmax);
    const float inv_max = __fdividef(1.0f, dmin);
    const float u = curand_uniform(rs);
    const float inv = fmaf(u, inv_max - inv_min, inv_min);
    return __fdividef(1.0f, inv);
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