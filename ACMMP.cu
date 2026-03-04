#include "ACMMP.h"
#include "BatchACMMP.h"
#include "ACMMP_device.cuh"  // Include the device functions header
#include <math_constants.h>  // for CUDART_PI_F
#include <memory>


// Lightweight counter-based RNG using Philox4x32-10 as a hash function.
// Each call increments the counter and produces a uniform float in (0,1].
// pixel_key should be a unique value per pixel (e.g., p.y * width + p.x).
__device__ __forceinline__ float rng_uniform(RNGState* state, unsigned int pixel_key) {
    uint4 ctr = make_uint4(state->counter++, pixel_key, 0, 0);
    uint2 key = make_uint2(0xD2511F53u, 0xCD9E8D57u);
    uint4 result = curand_Philox4x32_10(ctr, key);
    // Map to (0, 1] float with 24-bit mantissa precision
    return (result.x >> 8) * (1.0f / 16777215.0f) + (1.0f / 16777216.0f);
}

__device__ __forceinline__ float SampleDepthInv(RNGState* rs, unsigned int pixel_key, float dmin, float dmax) {
    dmin = fmaxf(dmin, 1e-6f);
    dmax = fmaxf(dmax, dmin + 1e-6f);
    const float inv_min = __fdividef(1.0f, dmax);
    const float inv_max = __fdividef(1.0f, dmin);
    const float u = rng_uniform(rs, pixel_key);
    const float inv = fmaf(u, inv_max - inv_min, inv_min);
    return __fdividef(1.0f, inv);
}

__device__ __forceinline__ void ScaleVec3(float4* v, float k) {
    v->x *= k;
    v->y *= k;
    v->z *= k;
}

__device__ __forceinline__ void DivVec3(float4* v, float k) {
    v->x /= k;
    v->y /= k;
    v->z /= k;
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

__device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, RNGState *rand_state, unsigned int pixel_key, const float depth)
{
    float4 normal;
    float q1 = 1.0f;
    float q2 = 1.0f;
    float s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * rng_uniform(rand_state, pixel_key) -1.0f;
        q2 = 2.0f * rng_uniform(rand_state, pixel_key) - 1.0f;
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

__device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, RNGState *rand_state, unsigned int pixel_key, const float perturbation)
{
    float4 view_direction = GetViewDirection(camera, p, 1.0f);

    const float a1 = (rng_uniform(rand_state, pixel_key) - 0.5f) * perturbation;
    const float a2 = (rng_uniform(rand_state, pixel_key) - 0.5f) * perturbation;
    const float a3 = (rng_uniform(rand_state, pixel_key) - 0.5f) * perturbation;

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

__device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, RNGState *rand_state, unsigned int pixel_key, const float depth_min, const float depth_max)
{
    float depth = rng_uniform(rand_state, pixel_key) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, pixel_key, depth);
    plane_hypothesis.w = GetDistance2Origin(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

__device__ float4 GeneratePerturbedPlaneHypothesis(const Camera camera, const int2 p,
                                                  RNGState *rand_state, unsigned int pixel_key, const float perturbation,
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
        const float cand_depth = SampleDepthInv(rand_state, pixel_key, lo, hi);

        // Every 8th try: perturb the normal a bit, otherwise keep current normal
        float4 n_try = ((k % 8) == 0)
            ? GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, pixel_key, 0.1f * CUDART_PI_F)
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
    float4 out = GeneratePerturbedNormal(camera, p, best_ph, rand_state, pixel_key, perturbation * CUDART_PI_F);
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

__device__ __forceinline__ float ComputeBilateralWeight(
    const float spatial_dist_sq, 
    const float color_dist, 
    const float inv_sigma_spatial_sq, 
    const float inv_sigma_color_sq)
{
    return expf(-spatial_dist_sq * inv_sigma_spatial_sq - color_dist * inv_sigma_color_sq);
}

// Precompute reference patch for bilateral NCC — source-independent, computed once.
__device__ void PrecomputeBilateralPatch(
    const cudaTextureObject_t ref_image, const Camera& ref_camera,
    const int2 p, const float4& plane_hypothesis, const PatchMatchParams& params,
    BilateralPatch& patch)
{
    const int radius = params.patch_size / 2;
    const float inv_sigma_spatial_sq = 1.0f / (2.0f * params.sigma_spatial * params.sigma_spatial);
    const float inv_sigma_color_sq = 1.0f / (2.0f * params.sigma_color * params.sigma_color);
    patch.center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);

    patch.n = 0;
    for (int i = -radius; i <= radius; i += params.radius_increment) {
        const float i_sq = static_cast<float>(i * i);
        for (int j = -radius; j <= radius; j += params.radius_increment) {
            const int2 ref_pt = make_int2(p.x + i, p.y + j);
            const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
            const float depth_n = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, ref_pt);
            const float3 Pw_n = Get3DPointonWorld_cu(ref_pt.x, ref_pt.y, depth_n, ref_camera);

            const float spatial_dist_sq = i_sq + static_cast<float>(j * j);
            const float color_dist = fabsf(ref_pix - patch.center_pix);
            const float w = expf(-spatial_dist_sq * inv_sigma_spatial_sq - color_dist * inv_sigma_color_sq);

            int k = patch.n++;
            if (k >= BilateralPatch::MAX_SAMPLES) { patch.n = BilateralPatch::MAX_SAMPLES; break; }
            patch.ref_pix[k] = ref_pix;
            patch.world_pt[k] = Pw_n;
            patch.bw[k] = w;
        }
        if (patch.n >= BilateralPatch::MAX_SAMPLES) break;
    }
}

// Per-source NCC using precomputed bilateral reference patch.
__device__ float ComputeNCC_Bilateral(
    const BilateralPatch& patch,
    const cudaTextureObject_t src_image, const Camera& src_camera,
    const PatchMatchParams& params)
{
    const float cost_max = 2.0f;
    const float kMinVar = 1e-5f;

    float sum_bw = 0.0f, sum_r = 0.0f, sum_rr = 0.0f;
    float sum_s = 0.0f, sum_ss = 0.0f, sum_rs = 0.0f;

    const float src_w = static_cast<float>(src_camera.width);
    const float src_inv_w = __fdividef(1.0f, src_w);

    for (int k = 0; k < patch.n; ++k) {
        float2 src_pt;
        float src_d;
        ProjectonCamera_cu(patch.world_pt[k], src_camera, src_pt, src_d);

        // Branchless spherical wrapping
        if (src_camera.model == SPHERE) {
            src_pt.x -= src_w * floorf(src_pt.x * src_inv_w);
        }

        // Bounds check
        if (src_pt.y < 0.0f || src_pt.y >= src_camera.height) continue;
        if (src_camera.model != SPHERE && (src_pt.x < 0.0f || src_pt.x >= src_camera.width)) continue;

        const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
        const float rv = patch.ref_pix[k];
        const float w = patch.bw[k];

        sum_bw += w;
        sum_r += w * rv;  sum_rr += w * rv * rv;
        sum_s += w * src_pix;  sum_ss += w * src_pix * src_pix;
        sum_rs += w * rv * src_pix;
    }

    if (sum_bw < 1e-6f) return cost_max;
    const float ib = 1.0f / sum_bw;
    const float mr = sum_r * ib, ms = sum_s * ib;
    const float vr = sum_rr * ib - mr * mr, vs = sum_ss * ib - ms * ms;
    if (vr < kMinVar || vs < kMinVar) return cost_max;
    return fmaxf(0.0f, fminf(cost_max, 1.0f - (sum_rs * ib - mr * ms) / sqrtf(vr * vs)));
}

// Legacy wrapper for single-source calls (e.g., if needed elsewhere)
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

    float depth_ref = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
    if (depth_ref <= 0.0f || depth_ref > 1000.0f) return cost_max;

    float3 Pw_center = Get3DPointonWorld_cu(p.x, p.y, depth_ref, ref_camera);
    float2 pt_center;
    float dummy_depth;
    ProjectonCamera_cu(Pw_center, src_camera, pt_center, dummy_depth);

    if (src_camera.model == SPHERE) {
        if (pt_center.y < radius || pt_center.y >= src_camera.height - radius)
            return cost_max;
    } else {
        if (pt_center.x < radius || pt_center.x >= src_camera.width - radius ||
            pt_center.y < radius || pt_center.y >= src_camera.height - radius)
            return cost_max;
    }

    BilateralPatch patch;
    PrecomputeBilateralPatch(ref_image, ref_camera, p, plane_hypothesis, params, patch);
    return ComputeNCC_Bilateral(patch, src_image, src_camera, params);
}

// ============================================================================
// TANGENT PLANE NCC (Pole-safe spherical matching)
// ============================================================================

// Precompute reference patch on tangent plane — source-independent, computed once.
__device__ void PrecomputeTangentPatch(
    const cudaTextureObject_t ref_image, const Camera& ref_cam,
    const int2 p, const float4& plane, const PatchMatchParams& params,
    TangentPatch& patch)
{
    float3 cd; PixelToDir(ref_cam, p, &cd);
    float3 right, up; ComputeTangentBasis(cd, right, up);
    const float delta = 2.f * CUDART_PI_F / static_cast<float>(ref_cam.width);
    const int radius = params.patch_size / 2;
    const float inv_ss = 1.f / (2.f * params.sigma_spatial * params.sigma_spatial);
    const float inv_cs = 1.f / (2.f * params.sigma_color * params.sigma_color);
    const float center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
    const float3 C = make_float3(
        -fmaf(ref_cam.R[0],ref_cam.t[0], fmaf(ref_cam.R[3],ref_cam.t[1], ref_cam.R[6]*ref_cam.t[2])),
        -fmaf(ref_cam.R[1],ref_cam.t[0], fmaf(ref_cam.R[4],ref_cam.t[1], ref_cam.R[7]*ref_cam.t[2])),
        -fmaf(ref_cam.R[2],ref_cam.t[0], fmaf(ref_cam.R[5],ref_cam.t[1], ref_cam.R[8]*ref_cam.t[2])));
    const float w_img = static_cast<float>(ref_cam.width);

    patch.n = 0;
    for (int i = -radius; i <= radius; i += params.radius_increment) {
        for (int j = -radius; j <= radius; j += params.radius_increment) {
            float3 sd = make_float3(
                fmaf(i*delta, right.x, fmaf(j*delta, up.x, cd.x)),
                fmaf(i*delta, right.y, fmaf(j*delta, up.y, cd.y)),
                fmaf(i*delta, right.z, fmaf(j*delta, up.z, cd.z)));
            float inv_len = rsqrtf(fmaf(sd.x,sd.x, fmaf(sd.y,sd.y, sd.z*sd.z)));
            sd.x *= inv_len; sd.y *= inv_len; sd.z *= inv_len;

            float2 rp = DirectionToPixelSpherical(ref_cam, sd);
            rp.x -= w_img * floorf(rp.x * __fdividef(1.0f, w_img));
            if (rp.y < 0.f || rp.y >= ref_cam.height) continue;
            float ref_val = tex2D<float>(ref_image, rp.x + 0.5f, rp.y + 0.5f);

            float depth = ComputeDepthFromDirection(plane, sd);
            if (depth <= 0.f || depth > 1000.f) continue;
            float3 pc = make_float3(sd.x*depth, sd.y*depth, sd.z*depth);
            float3 wp = make_float3(
                fmaf(ref_cam.R[0],pc.x, fmaf(ref_cam.R[3],pc.y, ref_cam.R[6]*pc.z)) + C.x,
                fmaf(ref_cam.R[1],pc.x, fmaf(ref_cam.R[4],pc.y, ref_cam.R[7]*pc.z)) + C.y,
                fmaf(ref_cam.R[2],pc.x, fmaf(ref_cam.R[5],pc.y, ref_cam.R[8]*pc.z)) + C.z);

            float sp_sq = static_cast<float>(i*i + j*j);
            float c_dist = fabsf(ref_val - center_pix);
            float w = expf(-sp_sq * inv_ss - c_dist * inv_cs);

            int k = patch.n++;
            if (k >= TangentPatch::MAX_SAMPLES) { patch.n = TangentPatch::MAX_SAMPLES; break; }
            patch.ref_pix[k] = ref_val;
            patch.world_pt[k] = wp;
            patch.bw[k] = w;
        }
        if (patch.n >= TangentPatch::MAX_SAMPLES) break;
    }
}

// Per-source NCC using precomputed reference patch.
__device__ float ComputeNCC_Tangent(
    const TangentPatch& patch,
    const cudaTextureObject_t src_image, const Camera& src_cam,
    const PatchMatchParams& params)
{
    const float cost_max = 2.f, kMinVar = 1e-5f;
    float sum_bw=0, sum_r=0, sum_rr=0, sum_s=0, sum_ss=0, sum_rs=0;
    const float src_w = static_cast<float>(src_cam.width);
    const float src_inv_w = __fdividef(1.0f, src_w);
    const float pole_rows = (src_cam.model == SPHERE)
        ? src_w * 0.5f * (5.f / 180.f) : 0.f;

    for (int k = 0; k < patch.n; ++k) {
        float2 sp; float sd;
        ProjectonCamera_cu(patch.world_pt[k], src_cam, sp, sd);
        if (src_cam.model == SPHERE) {
            sp.x -= src_w * floorf(sp.x * src_inv_w);
            if (sp.y < pole_rows || sp.y >= src_cam.height - pole_rows) continue;
        }
        if (sp.y < 0.f || sp.y >= src_cam.height) continue;
        if (src_cam.model != SPHERE && (sp.x < 0.f || sp.x >= src_cam.width)) continue;

        float sv = tex2D<float>(src_image, sp.x + 0.5f, sp.y + 0.5f);
        float rv = patch.ref_pix[k], w = patch.bw[k];
        sum_bw += w;
        sum_r += w*rv; sum_rr += w*rv*rv;
        sum_s += w*sv; sum_ss += w*sv*sv;
        sum_rs += w*rv*sv;
    }
    if (sum_bw < 1e-6f) return cost_max;
    float ib = 1.f / sum_bw;
    float mr = sum_r*ib, ms = sum_s*ib;
    float vr = sum_rr*ib - mr*mr, vs = sum_ss*ib - ms*ms;
    if (vr < kMinVar || vs < kMinVar) return cost_max;
    return fmaxf(0.f, fminf(cost_max, 1.f - (sum_rs*ib - mr*ms) / sqrtf(vr*vs)));
}

// ============================================================================

__device__ float ComputeMultiViewInitialCostandSelectedViews(const cudaTextureObject_t *images, const Camera *cameras, const int2 p, const float4 plane_hypothesis, unsigned int *selected_views, const PatchMatchParams params)
{
    float cost_max = 2.0f;
    float cost_vector[32] = {2.0f};
    float cost_vector_copy[32] = {2.0f};
    int cost_count = 0;
    int num_valid_views = 0;

    if (cameras[0].model == SPHERE) {
        TangentPatch patch;
        PrecomputeTangentPatch(images[0], cameras[0], p, plane_hypothesis, params, patch);
        for (int i = 1; i < params.num_images; ++i) {
            float c = ComputeNCC_Tangent(patch, images[i], cameras[i], params);
            cost_vector[i - 1] = c;
            cost_vector_copy[i - 1] = c;
            cost_count++;
            if (c < cost_max) {
                num_valid_views++;
            }
        }
    } else {
        BilateralPatch patch;
        PrecomputeBilateralPatch(images[0], cameras[0], p, plane_hypothesis, params, patch);
        for (int i = 1; i < params.num_images; ++i) {
            float c = ComputeNCC_Bilateral(patch, images[i], cameras[i], params);
            cost_vector[i - 1] = c;
            cost_vector_copy[i - 1] = c;
            cost_count++;
            if (c < cost_max) {
                num_valid_views++;
            }
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
    if (cameras[0].model == SPHERE) {
        TangentPatch patch;
        PrecomputeTangentPatch(images[0], cameras[0], p, plane_hypothesis, params, patch);
        for (int i = 1; i < params.num_images; ++i)
            cost_vector[i-1] = ComputeNCC_Tangent(patch, images[i], cameras[i], params);
    } else {
        BilateralPatch patch;
        PrecomputeBilateralPatch(images[0], cameras[0], p, plane_hypothesis, params, patch);
        for (int i = 1; i < params.num_images; ++i)
            cost_vector[i-1] = ComputeNCC_Bilateral(patch, images[i], cameras[i], params);
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

    float diff_col = p.x - backward_point.x;
    if (ref_camera.model == SPHERE) {
        const float w = static_cast<float>(ref_camera.width);
        // Branchless shortest-path wrapping: wrap into [-w/2, w/2)
        diff_col -= w * floorf((diff_col + w * 0.5f) * __fdividef(1.0f, w));
    }
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrt(diff_col * diff_col + diff_row * diff_row));
}

template<bool UseMask>
__global__ void RandomInitialization(cudaTextureObjects *texture_objects, Camera *cameras, float4 *plane_hypotheses,  float4 *scaled_plane_hypotheses, float *costs,  float *pre_costs,  RNGState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const uint8_t *ref_mask, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    int width = cameras[0].width;
    int height = cameras[0].height;

    if (p.x >= width || p.y >= height) {
        return;
    }

    const int center = p.y * width + p.x;
    rand_states[center].counter = 0;  // Initialize counter-based RNG
    const unsigned int pixel_key = (unsigned int)center;

    // Pole guard: exclude extreme pole pixels for spherical cameras
    if (cameras[0].model == SPHERE && IsNearPole(cameras[0], p.x, p.y, 5.0f)) {
        plane_hypotheses[center] = make_float4(0.f, 0.f, 1.f, 0.f);
        costs[center] = 2.0f;
        selected_views[center] = 0;
        return;
    }

    // Skip masked pixels (compiled out when UseMask=false)
    if (UseMask && ref_mask[center]) {
        plane_hypotheses[center] = make_float4(0.f, 0.f, 1.f, 0.f);
        costs[center] = 2.0f;
        selected_views[center] = 0;
        return;
    }

    if (params.planar_prior) {
        // Prior-assisted init: must be checked BEFORE the default random init
        if (plane_masks[center] > 0 && costs[center] >= 0.1f) {
            float perturbation = 0.02f;

            float4 plane_hypothesis = prior_planes[center];
            float depth_perturbed = plane_hypothesis.w;
            const float depth_min_perturbed = (1 - 3 * perturbation) * depth_perturbed;
            const float depth_max_perturbed = (1 + 3 * perturbation) * depth_perturbed;
            depth_perturbed = rng_uniform(&rand_states[center], pixel_key) * (depth_max_perturbed - depth_min_perturbed) + depth_min_perturbed;
            float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, plane_hypothesis, &rand_states[center], pixel_key, 3 * perturbation * M_PI);
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
    else if (!params.geom_consistency && !params.hierarchy) {
        // Default random init
        plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], pixel_key, params.depth_min, params.depth_max);
        costs[center] = ComputeMultiViewInitialCostandSelectedViews(texture_objects[0].images, cameras, p, plane_hypotheses[center], &selected_views[center], params);
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
                    ScaleVec3(&srcNorm, totalgauss);
                    n_total_val.x  = n_total_val.x + srcNorm.x;
                    n_total_val.y  = n_total_val.y + srcNorm.y;
                    n_total_val.z  = n_total_val.z + srcNorm.z;
                }
            }
            costs[center] = c_total_val / normalizing_factor;
            DivVec3(&n_total_val, normalizing_factor);
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

// Fast approximation for acos(c)^2: uses Taylor expansion 2(1-c) for aligned normals (c>0),
// falls back to full acosf for opposing normals (c<0). Max relative error ~4% at c=0.5.
__device__ __forceinline__ float ApproxAcosSq(float c) {
    c = fminf(fmaxf(c, -1.0f), 1.0f);
    if (c > 0.0f) return 2.0f * (1.0f - c);  // Taylor approx for small angles
    float a = acosf(c); return a * a;          // fallback for large angles
}

__device__ void PlaneHypothesisRefinement(const cudaTextureObject_t *images,
                                          const cudaTextureObject_t *depth_images,
                                          const Camera *cameras,
                                          float4 *plane_hypothesis,
                                          float *depth,
                                          float *cost,
                                          RNGState *rand_state,
                                          unsigned int pixel_key,
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
        depth_rand = SampleDepthInv(rand_state, pixel_key,
                                   fmaxf(depth_prior - 3 * depth_sigma, params.depth_min),
                                   fminf(depth_prior + 3 * depth_sigma, params.depth_max));
        plane_hypothesis_rand = GeneratePerturbedNormal(cameras[0], p, prior_planes[center], rand_state, pixel_key, angle_sigma);
    } else {
        // Standard random sampling
        depth_rand = SampleDepthInv(rand_state, pixel_key, params.depth_min, params.depth_max);
        plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, pixel_key, depth_rand);
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
        float cand = SampleDepthInv(rand_state, pixel_key, lo, hi);
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
        GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, pixel_key, perturbation * CUDART_PI_F);

    // 4) Evaluate candidates — with prior guidance, 3 most informative suffice
    const bool has_prior = params.planar_prior && plane_masks[center] > 0;
    const int num_planes = has_prior ? 3 : 5;
    float  depths_arr[5]  = { depth_rand, *depth, depth_rand, *depth, depth_perturbed };
    float4 normals_arr[5] = { *plane_hypothesis, plane_hypothesis_rand,
                                   plane_hypothesis_rand, plane_hypothesis_perturbed,
                                   *plane_hypothesis };

    for (int i = 0; i < num_planes; ++i) {
        float cost_vector[32] = { 2.0f };
        float4 temp_plane_hypothesis = normals_arr[i];
        temp_plane_hypothesis.w = GetDistance2Origin(cameras[0], p, depths_arr[i], temp_plane_hypothesis);

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
            temp_cost /= weight_norm;
        }

        // Validate depth
        const float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], temp_plane_hypothesis, p);
        if (depth_before < params.depth_min || depth_before > params.depth_max || depth_before >= 1e6f) {
            continue;  // Skip invalid depths
        }

        // Accept based on prior availability (ACMMP's strength preserved)
        if (has_prior) {
            // Prior-based acceptance
            float depth_prior = ComputeDepthfromPlaneHypothesis(cameras[0], prior_planes[center], p);
            float depth_diff = depth_before - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], temp_plane_hypothesis);
            float angle_diff_sq = ApproxAcosSq(angle_cos);

            float prior = gamma + expf(-(depth_diff * depth_diff / two_depth_sigma_squared +
                                         angle_diff_sq / two_angle_sigma_squared));
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

// Find best neighbor along a checkerboard direction with bounds checking
__device__ __forceinline__ int FindBestNeighborInDirection(
    const float* costs, 
    const int center, 
    const int width, 
    const int height, 
    const int2 p, 
    const int direction,
    const int base_offset)
{
    // Validate base offset is within bounds
    if (base_offset < 0 || base_offset >= width * height) {
        return center; // Return safe default
    }
    
    float cost_min = costs[base_offset];
    int cost_min_point = base_offset;
    
    // Direction: 0=up_near, 1=up_far, 2=down_near, 3=down_far, 4=left_near, 5=left_far, 6=right_near, 7=right_far
    if (direction == 1) { // up_far
        for (int i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int point_temp = base_offset - 2 * i * width;
                if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
    }
    else if (direction == 3) { // down_far
        for (int i = 1; i < 11; ++i) {
            if (p.y < height - 3 - 2 * i) {
                int point_temp = base_offset + 2 * i * width;
                if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
    }
    else if (direction == 5) { // left_far
        for (int i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int point_temp = base_offset - 2 * i;
                if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
    }
    else if (direction == 7) { // right_far
        for (int i = 1; i < 11; ++i) {
            if (p.x < width - 3 - 2 * i) {
                int point_temp = base_offset + 2 * i;
                if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                    cost_min = costs[point_temp];
                    cost_min_point = point_temp;
                }
            }
        }
    }
    else if (direction == 0 || direction == 2) { // up_near, down_near
        const int y_sign = (direction == 0) ? -1 : 1;
        for (int i = 0; i < 3; ++i) {
            bool valid_y = (direction == 0) ? (p.y > 1 + i) : (p.y < height - 2 - i);
            if (valid_y) {
                const int y_offset = y_sign * (1 + i) * width;
                const int x_offset = 1 + i;
                
                if (p.x > i) {
                    int point_temp = base_offset + y_offset - x_offset;
                    if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                        cost_min = costs[point_temp];
                        cost_min_point = point_temp;
                    }
                }
                if (p.x < width - 1 - i) {
                    int point_temp = base_offset + y_offset + x_offset;
                    if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                        cost_min = costs[point_temp];
                        cost_min_point = point_temp;
                    }
                }
            }
        }
    }
    else if (direction == 4 || direction == 6) { // left_near, right_near
        const int x_sign = (direction == 4) ? -1 : 1;
        for (int i = 0; i < 3; ++i) {
            bool valid_x = (direction == 4) ? (p.x > 1 + i) : (p.x < width - 2 - i);
            if (valid_x) {
                const int x_offset = x_sign * (1 + i);
                const int y_offset = (1 + i) * width;
                
                if (p.y > i) {
                    int point_temp = base_offset + x_offset - y_offset;
                    if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                        cost_min = costs[point_temp];
                        cost_min_point = point_temp;
                    }
                }
                if (p.y < height - 1 - i) {
                    int point_temp = base_offset + x_offset + y_offset;
                    if (point_temp >= 0 && point_temp < width * height && costs[point_temp] < cost_min) {
                        cost_min = costs[point_temp];
                        cost_min_point = point_temp;
                    }
                }
            }
        }
    }
    
    return cost_min_point;
}

// Accumulate view selection priors from near neighbors
__device__ __forceinline__ void ComputeViewSelectionPriors(
    float* view_selection_priors,
    const unsigned int* selected_views,
    const bool* valid_directions,
    const int center,
    const int width,
    const int height,
    const int num_images)
{
    // Initialize priors safely
    for (int i = 0; i < num_images - 1 && i < 32; ++i) {
        view_selection_priors[i] = 0.0f;
    }
    
    // Neighbor offsets for near directions only
    const int neighbor_offsets[4] = {-width, width, -1, 1}; // up, down, left, right
    const int neighbor_dirs[4] = {0, 2, 4, 6};
    
    for (int i = 0; i < 4; ++i) {
        if (valid_directions[neighbor_dirs[i]]) {
            const int neighbor_pos = center + neighbor_offsets[i];
            // Validate neighbor position
            if (neighbor_pos >= 0 && neighbor_pos < width * height) {
                for (int j = 0; j < num_images - 1 && j < 32; ++j) {
                    const float weight = isSet(selected_views[neighbor_pos], j) ? 0.9f : 0.1f;
                    view_selection_priors[j] += weight;
                }
            }
        }
    }
}

// Compute per-view sampling probabilities from neighbor costs
__device__ __forceinline__ void ComputeSamplingProbabilities(
    float* sampling_probs,
    const float cost_array[8][32],
    const float* view_selection_priors,
    const bool* valid_directions,
    const int num_images,
    const int iter)
{
    const float cost_threshold = 0.8f * expf((iter * iter) / (-90.0f));
    const float inv_neg_018 = 1.0f / (-0.18f);
    const float inv_neg_032 = 1.0f / (-0.32f);
    const float threshold_exp = expf(cost_threshold * cost_threshold * inv_neg_032);

    for (int i = 0; i < num_images - 1 && i < 32; i++) {
        float count = 0.0f;
        int count_false = 0;
        float tmpw = 0.0f;
        
        for (int j = 0; j < 8; j++) {
            if (valid_directions[j]) {
                const float cost_val = cost_array[j][i];
                if (cost_val < cost_threshold) {
                    tmpw += expf(cost_val * cost_val * inv_neg_018);
                    count += 1.0f;
                }
                if (cost_val > 1.2f) {
                    count_false++;
                }
            }
        }
        
        if (count > 2.0f && count_false < 3) {
            sampling_probs[i] = (tmpw / count) * view_selection_priors[i];
        } else if (count_false < 3) {
            sampling_probs[i] = threshold_exp * view_selection_priors[i];
        } else {
            sampling_probs[i] = 0.0f;
        }
    }
}

// Compute weighted multi-view costs for each neighbor direction
__device__ __forceinline__ void ComputeFinalCosts(
    float* final_costs,
    const float cost_array[8][32],
    const float* view_weights,
    const float weight_norm,
    const bool* valid_directions,
    const int* neighbor_positions,
    const cudaTextureObject_t* depths,
    const Camera* cameras,
    const float4* plane_hypotheses,
    const int2 p,
    const PatchMatchParams& params)
{
    const float inv_weight_norm = (weight_norm > 1e-6f) ? (1.0f / weight_norm) : 0.0f;
    
    for (int i = 0; i < 8; ++i) {
        if (valid_directions[i]) {
            float cost_sum = 0.0f;
            for (int j = 0; j < params.num_images - 1 && j < 32; ++j) {
                if (view_weights[j] > 0.0f) {
                    float base_cost = cost_array[i][j];
                    if (params.geom_consistency) {
                        const float geom_cost = ComputeGeomConsistencyCost(
                            depths[j + 1], cameras[0], cameras[j + 1], 
                            plane_hypotheses[neighbor_positions[i]], p);
                        base_cost += 0.2f * geom_cost;
                    }
                    cost_sum += view_weights[j] * base_cost;
                }
            }
            final_costs[i] = cost_sum * inv_weight_norm;
        } else {
            final_costs[i] = 2.0f; // High cost for invalid directions
        }
    }
}

template<bool UseMask>
__device__ void CheckerboardPropagation(
    const cudaTextureObject_t *images,
    const cudaTextureObject_t *depths,
    const Camera *cameras,
    float4 *plane_hypotheses,
    float *costs,
    float *pre_costs,
    RNGState *rand_states,
    unsigned int *selected_views,
    float4 *prior_planes,
    unsigned int *plane_masks,
    const uint8_t *ref_mask,
    const int2 p,
    const PatchMatchParams params,
    const int iter)
{
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    // Early exit for out-of-bounds
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0) {
        return;
    }

    // Pole guard: skip propagation at extreme poles for spherical cameras
    if (cameras[0].model == SPHERE && IsNearPole(cameras[0], p.x, p.y, 5.0f)) {
        return;
    }

    const int center = p.y * width + p.x;

    // Skip masked pixels (compiled out when UseMask=false)
    if (UseMask && ref_mask[center]) {
        return;
    }

    // Validate center index
    if (center < 0 || center >= width * height) {
        return;
    }

    // Early termination: skip pixels that have already converged
    if (iter > 0) {
        float current_cost = costs[center];
        float prev_cost = pre_costs[center];
        if (current_cost < 0.1f && fabsf(current_cost - prev_cost) < 0.002f) {
            return;  // Already converged
        }
    }

    // Calculate neighbor positions with bounds checking
    // For spherical cameras, horizontal neighbors wrap around the seam
    const bool is_sphere = (cameras[0].model == SPHERE);

    int left_near_x  = is_sphere ? ((p.x - 1 + width) % width) : (p.x - 1);
    int left_far_x   = is_sphere ? ((p.x - 3 + width) % width) : (p.x - 3);
    int right_near_x = is_sphere ? ((p.x + 1) % width) : (p.x + 1);
    int right_far_x  = is_sphere ? ((p.x + 3) % width) : (p.x + 3);

    int left_near = p.y * width + left_near_x;
    int left_far = p.y * width + left_far_x;
    int right_near = p.y * width + right_near_x;
    int right_far = p.y * width + right_far_x;
    int up_near = center - width;
    int up_far = center - 3 * width;
    int down_near = center + width;
    int down_far = center + 3 * width;

    // Evaluate 8 checkerboard neighbors (near/far x up/down/left/right).
    // Spherical cameras: horizontal neighbors always valid due to wrapping.
    // With --maxrregcount=96 + __launch_bounds__(512,2), the compiler will
    // naturally spill this large array to L1-cached local memory.
    float cost_array[8][32];
    // Initialize cost array
    for (int i = 0; i < 8; ++i) {
        for (int j = 0; j < 32; ++j) {
            cost_array[i][j] = 2.0f;
        }
    }
    
    bool flag[8] = {false};
    int num_valid_pixels = 0;
    const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // Neighbor cost caching: skip expensive NCC for converged neighbors (iter > 0).
    // A neighbor is "converged" if its cost hasn't changed between iterations.
    // In that case, fill cost_array with the neighbor's stored cost as proxy.
    #define EVAL_NEIGHBOR(dir_idx, neighbor_pos) do { \
        if (iter > 0 && (neighbor_pos) >= 0 && (neighbor_pos) < width * height && \
            fabsf(costs[neighbor_pos] - pre_costs[neighbor_pos]) < 0.001f) { \
            const float proxy = costs[neighbor_pos]; \
            for (int _v = 0; _v < 32; ++_v) cost_array[dir_idx][_v] = proxy; \
        } else { \
            ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[neighbor_pos], cost_array[dir_idx], params); \
        } \
    } while(0)

    if (p.y > 2) {
        flag[1] = true;
        num_valid_pixels++;
        up_far = FindBestNeighborInDirection(costs, center, width, height, p, 1, up_far);
        EVAL_NEIGHBOR(1, up_far);
    }

    if (p.y < height - 3) {
        flag[3] = true;
        num_valid_pixels++;
        down_far = FindBestNeighborInDirection(costs, center, width, height, p, 3, down_far);
        EVAL_NEIGHBOR(3, down_far);
    }

    if (p.x > 2 || is_sphere) {
        flag[5] = true;
        num_valid_pixels++;
        if (p.x > 2) {
            left_far = FindBestNeighborInDirection(costs, center, width, height, p, 5, left_far);
        }
        EVAL_NEIGHBOR(5, left_far);
    }

    if (p.x < width - 3 || is_sphere) {
        flag[7] = true;
        num_valid_pixels++;
        if (p.x < width - 3) {
            right_far = FindBestNeighborInDirection(costs, center, width, height, p, 7, right_far);
        }
        EVAL_NEIGHBOR(7, right_far);
    }

    if (p.y > 0) {
        flag[0] = true;
        num_valid_pixels++;
        up_near = FindBestNeighborInDirection(costs, center, width, height, p, 0, up_near);
        EVAL_NEIGHBOR(0, up_near);
    }

    if (p.y < height - 1) {
        flag[2] = true;
        num_valid_pixels++;
        down_near = FindBestNeighborInDirection(costs, center, width, height, p, 2, down_near);
        EVAL_NEIGHBOR(2, down_near);
    }

    if (p.x > 0 || is_sphere) {
        flag[4] = true;
        num_valid_pixels++;
        if (p.x > 0) {
            left_near = FindBestNeighborInDirection(costs, center, width, height, p, 4, left_near);
        }
        EVAL_NEIGHBOR(4, left_near);
    }

    if (p.x < width - 1 || is_sphere) {
        flag[6] = true;
        num_valid_pixels++;
        if (p.x < width - 1) {
            right_near = FindBestNeighborInDirection(costs, center, width, height, p, 6, right_near);
        }
        EVAL_NEIGHBOR(6, right_near);
    }

    #undef EVAL_NEIGHBOR

    // Update positions array with safe values
    const int final_positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    float view_weights[32] = {0.0f};
    float view_selection_priors[32];
    
    ComputeViewSelectionPriors(view_selection_priors, selected_views, flag, 
                               center, width, height, params.num_images);

    float sampling_probs[32];
    ComputeSamplingProbabilities(sampling_probs, cost_array, view_selection_priors, 
                                 flag, params.num_images, iter);

    TransformPDFToCDF(sampling_probs, params.num_images - 1);
    
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = rng_uniform(&rand_states[center], (unsigned int)center) - FLT_EPSILON;

        for (int image_id = 0; image_id < params.num_images - 1 && image_id < 32; ++image_id) {
            const float prob = sampling_probs[image_id];
            if (prob > rand_prob) {
                view_weights[image_id] += 1.0f;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    int num_selected_view = 0;
    float weight_norm = 0.0f;
    for (int i = 0; i < params.num_images - 1 && i < 32; ++i) {
        if (view_weights[i] > 0.0f) {
            setBit(temp_selected_views, i);
            weight_norm += view_weights[i];
            num_selected_view++;
        }
    }

    float final_costs[8];
    ComputeFinalCosts(final_costs, cost_array, view_weights, weight_norm,
                      flag, final_positions, depths, cameras, 
                      plane_hypotheses, p, params);

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    float cost_vector_now[32];
    for (int i = 0; i < 32; ++i) {
        cost_vector_now[i] = 2.0f;
    }
    
    ComputeMultiViewCostVector(images, cameras, p, plane_hypotheses[center], cost_vector_now, params);
    float cost_now = 0.0f;
    for (int i = 0; i < params.num_images - 1 && i < 32; ++i) {
        if (view_weights[i] > 0.0f) {
            float base_cost = cost_vector_now[i];
            if (params.geom_consistency) {
                base_cost += 0.2f * ComputeGeomConsistencyCost(depths[i+1], cameras[0], cameras[i+1], plane_hypotheses[center], p);
            }
            cost_now += view_weights[i] * base_cost;
        }
    }
    if (weight_norm > 1e-6f) {
        cost_now /= weight_norm;
    }
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
                    float depth_now_temp = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[final_positions[i]], p);
                    float depth_diff = depth_now_temp - depth_prior;
                    float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[final_positions[i]]);
                    float angle_diff_sq = ApproxAcosSq(angle_cos);
                    float prior = gamma + expf(-(depth_diff * depth_diff / two_depth_sigma_squared + angle_diff_sq / two_angle_sigma_squared));
                    restricted_final_costs[i] = expf(-final_costs[i] * final_costs[i] / beta) * prior;
                }
            }
            const int max_cost_idx = FindMaxCostIndex(restricted_final_costs, 8);

            float restricted_cost_now = 0.0f;
            float depth_now_temp = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
            float depth_diff = depth_now_temp - depth_prior;
            float angle_cos = Vec3DotVec3(prior_planes[center], plane_hypotheses[center]);
            float angle_diff_sq = ApproxAcosSq(angle_cos);
            float prior = gamma + expf(-(depth_diff * depth_diff / two_depth_sigma_squared + angle_diff_sq / two_angle_sigma_squared));
            restricted_cost_now = expf(-cost_now * cost_now / beta) * prior;

            if (flag[max_cost_idx]) {
                float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[final_positions[max_cost_idx]], p);

                if (depth_before >= params.depth_min && depth_before <= params.depth_max && restricted_final_costs[max_cost_idx] > restricted_cost_now) {
                    depth_now = depth_before;
                    plane_hypotheses[center] = plane_hypotheses[final_positions[max_cost_idx]];
                    costs[center] = final_costs[max_cost_idx];
                    restricted_cost = restricted_final_costs[max_cost_idx];
                    selected_views[center] = temp_selected_views;
                }
            }
        }
        else if (flag[min_cost_idx]) {
            float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[final_positions[min_cost_idx]], p);

            if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
                depth_now = depth_before;
                plane_hypotheses[center] = plane_hypotheses[final_positions[min_cost_idx]];
                costs[center] = final_costs[min_cost_idx];
            }
        }
    }

    float4 plane_hypotheses_now = plane_hypotheses[center];
    if (!params.planar_prior && flag[min_cost_idx]) {
        float depth_before = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[final_positions[min_cost_idx]], p);

        if (depth_before >= params.depth_min && depth_before <= params.depth_max && final_costs[min_cost_idx] < cost_now) {
            depth_now = depth_before;
            plane_hypotheses_now = plane_hypotheses[final_positions[min_cost_idx]];
            cost_now = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }

    PlaneHypothesisRefinement(images, depths, cameras, &plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], (unsigned int)center, view_weights, weight_norm, prior_planes, plane_masks, &restricted_cost, p, params);

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

template<bool UseMask>
__global__ void __launch_bounds__(512, 2)
BlackPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs,  float *pre_costs,  RNGState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const uint8_t *ref_mask, const PatchMatchParams params, const int iter)
{
    // Coalesced: all threads in a warp access the same row, stride-2 columns
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Black pixels: (col + row) % 2 == 0
    const int col = 2 * tx + (row & 1);
    int2 p = make_int2(col, row);

    CheckerboardPropagation<UseMask>(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs,  rand_states, selected_views, prior_planes, plane_masks, ref_mask, p, params, iter);
}

template<bool UseMask>
__global__ void __launch_bounds__(512, 2)
RedPixelUpdate(cudaTextureObjects *texture_objects, cudaTextureObjects *texture_depths, Camera *cameras, float4 *plane_hypotheses, float *costs,  float *pre_costs, RNGState *rand_states, unsigned int *selected_views, float4 *prior_planes, unsigned int *plane_masks, const uint8_t *ref_mask, const PatchMatchParams params, const int iter)
{
    // Coalesced: all threads in a warp access the same row, stride-2 columns
    const int tx = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    // Red pixels: (col + row) % 2 == 1
    const int col = 2 * tx + (1 - (row & 1));
    int2 p = make_int2(col, row);

    CheckerboardPropagation<UseMask>(texture_objects[0].images, texture_depths[0].images, cameras, plane_hypotheses, costs, pre_costs, rand_states, selected_views, prior_planes, plane_masks, ref_mask, p, params, iter);
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

    // Pole guard: output depth=0 at extreme poles for spherical cameras
    if (cameras[0].model == SPHERE && IsNearPole(cameras[0], p.x, p.y, 5.0f)) {
        plane_hypotheses[center].w = 0.f;
        return;
    }

    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__device__ __forceinline__ float FindMedianFast(float* arr, int n) {
    // For very small arrays, use insertion sort (fastest for n <= 8)
    if (n <= 8) {
        // Insertion sort - optimal for small arrays
        for (int i = 1; i < n; i++) {
            float key = arr[i];
            int j = i - 1;
            while (j >= 0 && arr[j] > key) {
                arr[j + 1] = arr[j];
                j--;
            }
            arr[j + 1] = key;
        }
    } else {
        // Use partial quickselect for larger arrays
        // Only sort enough to find median
        int left = 0, right = n - 1;
        int target = n / 2;
        
        while (left < right) {
            // Partition around a pivot
            float pivot = arr[right];
            int i = left - 1;
            
            for (int j = left; j < right; j++) {
                if (arr[j] <= pivot) {
                    i++;
                    float temp = arr[i];
                    arr[i] = arr[j];
                    arr[j] = temp;
                }
            }
            i++;
            float temp = arr[i];
            arr[i] = arr[right];
            arr[right] = temp;
            
            if (i == target) break;
            else if (i > target) right = i - 1;
            else left = i + 1;
        }
    }
    
    // Return median
    if (n % 2 == 0) {
        return (arr[n/2 - 1] + arr[n/2]) / 2.0f;
    } else {
        return arr[n/2];
    }
}

// Optimized neighbor pattern structure
struct NeighborPattern {
    int offset;
    int min_x, max_x, min_y, max_y; // Boundary conditions
};

// Precomputed neighbor patterns for better performance
__device__ __constant__ NeighborPattern NEIGHBOR_PATTERNS[20] = {
    // Vertical neighbors
    {-1 * 1, 0, INT_MAX, 1, INT_MAX},           // up
    {-3 * 1, 0, INT_MAX, 3, INT_MAX},           // upup  
    {-5 * 1, 0, INT_MAX, 5, INT_MAX},           // upup - 2*width
    {1 * 1, 0, INT_MAX, 0, INT_MAX-1},          // down
    {3 * 1, 0, INT_MAX, 0, INT_MAX-3},          // downdown
    {5 * 1, 0, INT_MAX, 0, INT_MAX-5},          // downdown + 2*width
    
    // Horizontal neighbors  
    {-1, 1, INT_MAX, 0, INT_MAX},               // left
    {-3, 3, INT_MAX, 0, INT_MAX},               // leftleft
    {-5, 5, INT_MAX, 0, INT_MAX},               // leftleft - 2
    {1, 0, INT_MAX-1, 0, INT_MAX},              // right
    {3, 0, INT_MAX-3, 0, INT_MAX},              // rightright
    {5, 0, INT_MAX-5, 0, INT_MAX},              // rightright + 2
    
    // Diagonal neighbors
    {-1 * 1 + 2, 0, INT_MAX-2, 1, INT_MAX},     // up + 2
    {1 * 1 + 2, 0, INT_MAX-2, 0, INT_MAX-1},    // down + 2  
    {-1 * 1 - 2, 2, INT_MAX, 1, INT_MAX},       // up - 2
    {1 * 1 - 2, 2, INT_MAX, 0, INT_MAX-1},      // down - 2
    {-1 - 2 * 1, 1, INT_MAX, 3, INT_MAX},       // left - 2*width
    {1 - 2 * 1, 0, INT_MAX-1, 3, INT_MAX},      // right - 2*width
    {-1 + 2 * 1, 1, INT_MAX, 0, INT_MAX-2},     // left + 2*width
    {1 + 2 * 1, 0, INT_MAX-1, 0, INT_MAX-2}     // right + 2*width
};

// Optimized CheckerboardFilter with compile-time mask specialization
template<bool UseMask>
__device__ void CheckerboardFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const uint8_t *ref_mask, const int2 p)
{
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    // Early exits with bounds checking
    if (p.x >= width || p.y >= height || p.x < 0 || p.y < 0) {
        return;
    }

    // Pole guard: skip filtering at extreme poles for spherical cameras
    if (cameras[0].model == SPHERE && IsNearPole(cameras[0], p.x, p.y, 5.0f)) {
        return;
    }

    const int center = p.y * width + p.x;

    // Validate center index
    if (center < 0 || center >= width * height) {
        return;
    }

    // Don't overwrite masked pixel with neighbor median (compiled out when UseMask=false)
    if (UseMask && ref_mask[center]) {
        return;
    }

    // Early exit for very low cost (unchanged from original)
    if (costs[center] < 0.001f) {
        return;
    }

    // Pre-allocate filter array with maximum possible size
    float filter[21];
    int index = 0;

    // Always include center pixel
    filter[index++] = plane_hypotheses[center].w;

    // Precompute width multipliers for efficiency
    const int width_1 = width;
    const int width_3 = 3 * width;
    const int width_5 = 5 * width;
    const bool is_sphere = (cameras[0].model == SPHERE);

    // Helper macro: compute index for pixel at (dx, dy) offset from p
    // For spherical cameras, wraps x horizontally
    #define WRAPPED_IDX(dx, dy) \
        ((p.y + (dy)) * width + (is_sphere ? (((p.x + (dx)) % width + width) % width) : (p.x + (dx))))

    // Skip masked neighbors (compiled out when UseMask=false)
    #define ADD_NEIGHBOR(idx_expr) do { \
        const int _ni = (idx_expr); \
        if (!(UseMask && ref_mask[_ni])) \
            filter[index++] = plane_hypotheses[_ni].w; \
    } while(0)

    // Optimized neighbor collection using precomputed patterns
    // Vertical neighbors (up/down directions)
    if (p.y > 0) {
        ADD_NEIGHBOR(center - width_1);
        if (p.y > 2) {
            const int upup = center - width_3;
            ADD_NEIGHBOR(upup);
            if (p.y > 4) {
                ADD_NEIGHBOR(upup - 2 * width_1);
            }
        }
    }

    if (p.y < height - 1) {
        ADD_NEIGHBOR(center + width_1);
        if (p.y < height - 3) {
            const int downdown = center + width_3;
            ADD_NEIGHBOR(downdown);
            if (p.y < height - 5) {
                ADD_NEIGHBOR(downdown + 2 * width_1);
            }
        }
    }

    // Horizontal neighbors (left/right directions)
    if (p.x > 0 || is_sphere) {
        ADD_NEIGHBOR(WRAPPED_IDX(-1, 0));
        if (p.x > 2 || is_sphere) {
            ADD_NEIGHBOR(WRAPPED_IDX(-3, 0));
            if (p.x > 4 || is_sphere) {
                ADD_NEIGHBOR(WRAPPED_IDX(-5, 0));
            }
        }
    }

    if (p.x < width - 1 || is_sphere) {
        ADD_NEIGHBOR(WRAPPED_IDX(1, 0));
        if (p.x < width - 3 || is_sphere) {
            ADD_NEIGHBOR(WRAPPED_IDX(3, 0));
            if (p.x < width - 5 || is_sphere) {
                ADD_NEIGHBOR(WRAPPED_IDX(5, 0));
            }
        }
    }

    // Diagonal neighbors
    if (p.y > 0) {
        if (p.x < width - 2 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(2, -1));
        if (p.x > 1 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(-2, -1));
    }
    if (p.y < height - 1) {
        if (p.x < width - 2 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(2, 1));
        if (p.x > 1 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(-2, 1));
    }
    if (p.y > 2) {
        if (p.x > 0 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(-1, -2));
        if (p.x < width - 1 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(1, -2));
    }
    if (p.y < height - 2) {
        if (p.x > 0 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(-1, 2));
        if (p.x < width - 1 || is_sphere) ADD_NEIGHBOR(WRAPPED_IDX(1, 2));
    }

    #undef WRAPPED_IDX
    #undef ADD_NEIGHBOR

    // Fast median computation and assignment
    const float median_value = FindMedianFast(filter, index);
    plane_hypotheses[center].w = median_value;
}
// Fused depth+normal+filter: GetDepthandNormal + median filter in one kernel
template<bool UseMask>
__global__ void PostProcessKernel(Camera *cameras, float4 *plane_hypotheses, float *costs, const uint8_t *ref_mask, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = cameras[0].width;
    const int height = cameras[0].height;

    if (p.x >= width || p.y >= height) return;

    const int center = p.y * width + p.x;

    // Pole guard
    if (cameras[0].model == SPHERE && IsNearPole(cameras[0], p.x, p.y, 5.0f)) {
        plane_hypotheses[center].w = 0.f;
        return;
    }

    // Zero depth for masked pixels (compiled out when UseMask=false)
    if (UseMask && ref_mask[center]) {
        plane_hypotheses[center] = make_float4(0.f, 0.f, 1.f, 0.f);
        return;
    }

    // Step 1: Compute depth from plane hypothesis and transform normal
    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

// All-pixel filter: processes both black and red pixels in one launch
template<bool UseMask>
__global__ void AllPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const uint8_t *ref_mask, const PatchMatchParams params)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    CheckerboardFilter<UseMask>(cameras, plane_hypotheses, costs, ref_mask, p);
}

template<bool UseMask>
__global__ void BlackPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const uint8_t *ref_mask, const PatchMatchParams params)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2;
    } else {
        p.y = p.y * 2 + 1;
    }

    CheckerboardFilter<UseMask>(cameras, plane_hypotheses, costs, ref_mask, p);
}

template<bool UseMask>
__global__ void RedPixelFilter(const Camera *cameras, float4 *plane_hypotheses, float *costs, const uint8_t *ref_mask, const PatchMatchParams params)
{
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) {
        p.y = p.y * 2 + 1;
    } else {
        p.y = p.y * 2;
    }

    CheckerboardFilter<UseMask>(cameras, plane_hypotheses, costs, ref_mask, p);
}

// ── Item 5: Batch Planar Prior GPU Kernels ────────────────────────────────

// Extract support points: one thread per 5×5 block, find min-cost pixel
__global__ void ExtractSupportPointsKernel(
    const float4* plane_hypotheses, const float* costs,
    int width, int height,
    int2* support_points_out, int* num_points_out, int max_points)
{
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    const int step = 5;
    const int x0 = bx * step;
    const int y0 = by * step;

    if (x0 >= width || y0 >= height) return;

    const int x1 = min(x0 + step, width);
    const int y1 = min(y0 + step, height);

    float min_cost = 2.0f;
    int best_x = x0, best_y = y0;

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            int c = y * width + x;
            float cost = costs[c];
            if (cost < min_cost) {
                min_cost = cost;
                best_x = x;
                best_y = y;
            }
        }
    }

    if (min_cost < 0.1f) {
        int idx = atomicAdd(num_points_out, 1);
        if (idx < max_points) {
            support_points_out[idx] = make_int2(best_x, best_y);
        }
    }
}

// Rasterize triangles: one thread per triangle, iterate bounding box
// num_triangles_ptr is a device pointer to avoid host sync; max_triangles bounds the launch grid.
__global__ void RasterizeTrianglesKernel(
    const int2* triangle_vertices,  // 3 int2 per triangle
    const int* num_triangles_ptr,   // device pointer to actual count
    int max_triangles,              // upper bound for grid launch
    unsigned int* plane_masks,
    int width, int height)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= *num_triangles_ptr || tid >= max_triangles) return;

    const int2 v0 = triangle_vertices[tid * 3 + 0];
    const int2 v1 = triangle_vertices[tid * 3 + 1];
    const int2 v2 = triangle_vertices[tid * 3 + 2];

    // Bounding box
    int minx = max(0, min(v0.x, min(v1.x, v2.x)));
    int maxx = min(width - 1, max(v0.x, max(v1.x, v2.x)));
    int miny = max(0, min(v0.y, min(v1.y, v2.y)));
    int maxy = min(height - 1, max(v0.y, max(v1.y, v2.y)));

    // Edge function coefficients
    int e01_dx = v1.y - v0.y, e01_dy = v0.x - v1.x;
    int e12_dx = v2.y - v1.y, e12_dy = v1.x - v2.x;
    int e20_dx = v0.y - v2.y, e20_dy = v2.x - v0.x;

    for (int y = miny; y <= maxy; y++) {
        for (int x = minx; x <= maxx; x++) {
            int w0 = e12_dx * (x - v1.x) + e12_dy * (y - v1.y);
            int w1 = e20_dx * (x - v2.x) + e20_dy * (y - v2.y);
            int w2 = e01_dx * (x - v0.x) + e01_dy * (y - v0.y);

            if ((w0 >= 0 && w1 >= 0 && w2 >= 0) || (w0 <= 0 && w1 <= 0 && w2 <= 0)) {
                plane_masks[y * width + x] = (unsigned int)(tid + 1);
            }
        }
    }
}

// Compute prior planes from triangulated depths
__global__ void ComputePriorPlanesKernel(
    const float4* plane_hypotheses,  // depths from Pass 1 (in .w field)
    const Camera* cameras,
    const int2* triangle_vertices,
    unsigned int* plane_masks,
    float4* prior_planes,
    float depth_min, float depth_max,
    int width, int height)
{
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height) return;

    const int center = p.y * width + p.x;
    unsigned int tri_id = plane_masks[center];

    if (tri_id == 0) {
        prior_planes[center] = make_float4(0.f, 0.f, 0.f, 0.f);
        return;
    }

    int tri_idx = tri_id - 1;
    const int2 v0 = triangle_vertices[tri_idx * 3 + 0];
    const int2 v1 = triangle_vertices[tri_idx * 3 + 1];
    const int2 v2 = triangle_vertices[tri_idx * 3 + 2];

    // Get depths at vertices
    float d0 = plane_hypotheses[v0.y * width + v0.x].w;
    float d1 = plane_hypotheses[v1.y * width + v1.x].w;
    float d2 = plane_hypotheses[v2.y * width + v2.x].w;

    // Validate depths
    if (d0 <= 0.f || d1 <= 0.f || d2 <= 0.f ||
        d0 != d0 || d1 != d1 || d2 != d2 ||
        d0 < depth_min || d0 > depth_max ||
        d1 < depth_min || d1 > depth_max ||
        d2 < depth_min || d2 > depth_max) {
        prior_planes[center] = make_float4(0.f, 0.f, 0.f, 0.f);
        plane_masks[center] = 0;
        return;
    }

    // Unproject vertices to 3D in camera space
    float X0[3], X1[3], X2[3];
    int2 ip0 = make_int2(v0.x, v0.y);
    int2 ip1 = make_int2(v1.x, v1.y);
    int2 ip2 = make_int2(v2.x, v2.y);
    Get3DPoint(cameras[0], ip0, d0, X0);
    Get3DPoint(cameras[0], ip1, d1, X1);
    Get3DPoint(cameras[0], ip2, d2, X2);

    // Cross product to get plane normal
    float e1x = X1[0] - X0[0], e1y = X1[1] - X0[1], e1z = X1[2] - X0[2];
    float e2x = X2[0] - X0[0], e2y = X2[1] - X0[1], e2z = X2[2] - X0[2];
    float nx = e1y * e2z - e1z * e2y;
    float ny = e1z * e2x - e1x * e2z;
    float nz = e1x * e2y - e1y * e2x;

    float len = sqrtf(nx * nx + ny * ny + nz * nz);
    if (len < 1e-8f) {
        prior_planes[center] = make_float4(0.f, 0.f, 0.f, 0.f);
        plane_masks[center] = 0;
        return;
    }

    float inv_len = 1.f / len;
    nx *= inv_len; ny *= inv_len; nz *= inv_len;
    float d = -(nx * X0[0] + ny * X0[1] + nz * X0[2]);

    // Store as plane params: (nx, ny, nz, d) where plane: nx*x + ny*y + nz*z + d = 0
    // But the convention in this code stores depth in .w for prior_planes
    // Actually, prior_planes stores the plane equation and depth is computed via ComputeDepthfromPlaneHypothesis
    float4 plane = make_float4(nx, ny, nz, d);

    // Validate: compute depth at current pixel from this plane
    float test_depth = ComputeDepthfromPlaneHypothesis(cameras[0], plane, p);
    if (test_depth < depth_min || test_depth > depth_max || test_depth != test_depth) {
        prior_planes[center] = make_float4(0.f, 0.f, 0.f, 0.f);
        plane_masks[center] = 0;
        return;
    }

    prior_planes[center] = plane;
}

// GPU-side grid triangulation: replaces CPU Delaunay with regular grid connectivity.
// Each valid pair of adjacent 5x5 blocks forms 2 triangles (a quad split diagonally).
// Eliminates the GPU→CPU→GPU round-trip for Subdiv2D.
__global__ void GridTriangulationKernel(
    const int2* support_points, const int num_support_points,
    const float* costs, int width, int height, int block_size,
    int2* triangle_vertices, int* num_triangles, int max_triangles)
{
    // Each thread handles one grid cell (bx, by) and creates up to 2 triangles
    // connecting (bx,by), (bx+1,by), (bx,by+1), (bx+1,by+1)
    const int grid_w = (width + block_size - 1) / block_size;
    const int grid_h = (height + block_size - 1) / block_size;

    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    if (bx >= grid_w - 1 || by >= grid_h - 1) return;

    // Map grid cell (bx,by) to support point index
    // Support points are stored linearly: index = by * grid_w + bx
    const int idx00 = by * grid_w + bx;
    const int idx10 = by * grid_w + (bx + 1);
    const int idx01 = (by + 1) * grid_w + bx;
    const int idx11 = (by + 1) * grid_w + (bx + 1);

    // Check all 4 corners are valid support points (within range)
    if (idx00 >= num_support_points || idx10 >= num_support_points ||
        idx01 >= num_support_points || idx11 >= num_support_points) return;

    const int2 p00 = support_points[idx00];
    const int2 p10 = support_points[idx10];
    const int2 p01 = support_points[idx01];
    const int2 p11 = support_points[idx11];

    // Validate all points are within image bounds
    if (p00.x < 0 || p00.x >= width || p00.y < 0 || p00.y >= height) return;
    if (p10.x < 0 || p10.x >= width || p10.y < 0 || p10.y >= height) return;
    if (p01.x < 0 || p01.x >= width || p01.y < 0 || p01.y >= height) return;
    if (p11.x < 0 || p11.x >= width || p11.y < 0 || p11.y >= height) return;

    // Validate costs at all corners (only connect blocks with good matches)
    const float cost_threshold = 0.1f;
    if (costs[p00.y * width + p00.x] >= cost_threshold) return;
    if (costs[p10.y * width + p10.x] >= cost_threshold) return;
    if (costs[p01.y * width + p01.x] >= cost_threshold) return;
    if (costs[p11.y * width + p11.x] >= cost_threshold) return;

    // Emit 2 triangles for this quad
    int tri_base = atomicAdd(num_triangles, 2);
    if (tri_base + 1 >= max_triangles) return;

    // Triangle 1: p00, p10, p01
    triangle_vertices[(tri_base + 0) * 3 + 0] = p00;
    triangle_vertices[(tri_base + 0) * 3 + 1] = p10;
    triangle_vertices[(tri_base + 0) * 3 + 2] = p01;

    // Triangle 2: p10, p11, p01
    triangle_vertices[(tri_base + 1) * 3 + 0] = p10;
    triangle_vertices[(tri_base + 1) * 3 + 1] = p11;
    triangle_vertices[(tri_base + 1) * 3 + 2] = p01;
}

// Regular grid support point extraction: one point per block, ordered by grid position.
// Unlike ExtractSupportPointsKernel which uses atomicAdd (unordered), this produces
// support points in a deterministic grid layout needed by GridTriangulationKernel.
__global__ void ExtractGridSupportPointsKernel(
    const float4* plane_hypotheses, const float* costs,
    int width, int height, int block_size,
    int2* support_points_out, int grid_w, int grid_h)
{
    const int bx = blockIdx.x * blockDim.x + threadIdx.x;
    const int by = blockIdx.y * blockDim.y + threadIdx.y;
    if (bx >= grid_w || by >= grid_h) return;

    const int x0 = bx * block_size;
    const int y0 = by * block_size;
    const int x1 = min(x0 + block_size, width);
    const int y1 = min(y0 + block_size, height);

    float min_cost = 2.0f;
    int best_x = (x0 + x1) / 2;  // Default to center of block
    int best_y = (y0 + y1) / 2;

    for (int y = y0; y < y1; y++) {
        for (int x = x0; x < x1; x++) {
            float cost = costs[y * width + x];
            if (cost < min_cost) {
                min_cost = cost;
                best_x = x;
                best_y = y;
            }
        }
    }

    const int idx = by * grid_w + bx;
    // Store best point (or block center if no good match found)
    support_points_out[idx] = make_int2(best_x, best_y);
}

// ── End Batch Planar Prior Kernels ────────────────────────────────────────

void ACMMP::RunPatchMatch(ProblemGPUResources* res, bool skip_host_download) {
    cudaStream_t s = stream_ ? stream_ : 0;

    const int width  = cameras[0].width;
    const int height = cameras[0].height;

    dim3 grid_init((width + 15) / 16, (height + 15) / 16, 1);
    dim3 blk_init(16, 16, 1);

    const int half_width = (width + 1) / 2;
    const int max_iterations = params.max_iterations;

    const int BLOCK_W = 32;
    const int BLOCK_H = 16;
    dim3 grid_cb((half_width + BLOCK_W - 1) / BLOCK_W, (height + BLOCK_H - 1) / BLOCK_H, 1);
    dim3 blk_cb(BLOCK_W, BLOCK_H, 1);

    // Template dispatch strategy:
    // - Propagation kernels (RandomInit, Black/RedPixelUpdate): use <true> only for PLANAR+mask
    //   GEOM phases use <false> because PLANAR already zeroed masked pixels, and <true> causes
    //   register pressure that regresses Scale 0 GEOM by ~69%.
    // - PostProcess/Filter kernels: always use <true> when masks exist, to ensure masked pixels
    //   stay zeroed after depth computation. These are lightweight so register pressure is not an issue.
    const bool use_mask_propagation = params.has_mask && !params.geom_consistency;

    if (use_mask_propagation) {
        RandomInitialization<true><<<grid_init, blk_init, 0, s>>>(
            res->texture_objects_cuda, res->cameras_cuda, res->plane_hypotheses_cuda,
            res->scaled_plane_hypotheses_cuda, res->costs_cuda, res->pre_costs_cuda,
            res->rand_states_cuda, res->selected_views_cuda, res->prior_planes_cuda,
            res->plane_masks_cuda, res->ref_mask_cuda, params);
        CUDA_CHECK(cudaPeekAtLastError());

        for (int i = 0; i < max_iterations; ++i) {
            BlackPixelUpdate<true><<<grid_cb, blk_cb, 0, s>>>(
                res->texture_objects_cuda, res->texture_depths_cuda, res->cameras_cuda,
                res->plane_hypotheses_cuda, res->costs_cuda, res->pre_costs_cuda,
                res->rand_states_cuda, res->selected_views_cuda, res->prior_planes_cuda,
                res->plane_masks_cuda, res->ref_mask_cuda, params, i);
            CUDA_CHECK(cudaPeekAtLastError());

            RedPixelUpdate<true><<<grid_cb, blk_cb, 0, s>>>(
                res->texture_objects_cuda, res->texture_depths_cuda, res->cameras_cuda,
                res->plane_hypotheses_cuda, res->costs_cuda, res->pre_costs_cuda,
                res->rand_states_cuda, res->selected_views_cuda, res->prior_planes_cuda,
                res->plane_masks_cuda, res->ref_mask_cuda, params, i);
            CUDA_CHECK(cudaPeekAtLastError());
        }
    } else {
        RandomInitialization<false><<<grid_init, blk_init, 0, s>>>(
            res->texture_objects_cuda, res->cameras_cuda, res->plane_hypotheses_cuda,
            res->scaled_plane_hypotheses_cuda, res->costs_cuda, res->pre_costs_cuda,
            res->rand_states_cuda, res->selected_views_cuda, res->prior_planes_cuda,
            res->plane_masks_cuda, nullptr, params);
        CUDA_CHECK(cudaPeekAtLastError());

        for (int i = 0; i < max_iterations; ++i) {
            BlackPixelUpdate<false><<<grid_cb, blk_cb, 0, s>>>(
                res->texture_objects_cuda, res->texture_depths_cuda, res->cameras_cuda,
                res->plane_hypotheses_cuda, res->costs_cuda, res->pre_costs_cuda,
                res->rand_states_cuda, res->selected_views_cuda, res->prior_planes_cuda,
                res->plane_masks_cuda, nullptr, params, i);
            CUDA_CHECK(cudaPeekAtLastError());

            RedPixelUpdate<false><<<grid_cb, blk_cb, 0, s>>>(
                res->texture_objects_cuda, res->texture_depths_cuda, res->cameras_cuda,
                res->plane_hypotheses_cuda, res->costs_cuda, res->pre_costs_cuda,
                res->rand_states_cuda, res->selected_views_cuda, res->prior_planes_cuda,
                res->plane_masks_cuda, nullptr, params, i);
            CUDA_CHECK(cudaPeekAtLastError());
        }
    }

    // PostProcess and Filter always use <true> when masks exist to zero masked pixels.
    // These kernels are lightweight — no register pressure concern.
    if (params.has_mask) {
        PostProcessKernel<true><<<grid_init, blk_init, 0, s>>>(res->cameras_cuda, res->plane_hypotheses_cuda, res->costs_cuda, res->ref_mask_cuda, params);
        CUDA_CHECK(cudaPeekAtLastError());

        AllPixelFilter<true><<<grid_init, blk_init, 0, s>>>(res->cameras_cuda, res->plane_hypotheses_cuda, res->costs_cuda, res->ref_mask_cuda, params);
        CUDA_CHECK(cudaPeekAtLastError());
    } else {
        PostProcessKernel<false><<<grid_init, blk_init, 0, s>>>(res->cameras_cuda, res->plane_hypotheses_cuda, res->costs_cuda, nullptr, params);
        CUDA_CHECK(cudaPeekAtLastError());

        AllPixelFilter<false><<<grid_init, blk_init, 0, s>>>(res->cameras_cuda, res->plane_hypotheses_cuda, res->costs_cuda, nullptr, params);
        CUDA_CHECK(cudaPeekAtLastError());
    }

    if (!skip_host_download) {
        // Asynchronously copy results from device to the resource's pinned host memory.
        CUDA_CHECK(cudaMemcpyAsync(res->planes_host_pinned, res->plane_hypotheses_cuda,
                        sizeof(float4) * width * height,
                        cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaMemcpyAsync(res->costs_host_pinned, res->costs_cuda,
                        sizeof(float) * width * height,
                        cudaMemcpyDeviceToHost, s));
    }

    // Wait for the stream to finish all operations.
    CUDA_CHECK(cudaStreamSynchronize(s));

    // Sequential mode: copy to class buffers for GetPlaneHypothesis/GetCost
    if (!batch_mode_ && !skip_host_download) {
        memcpy(plane_hypotheses_host, res->planes_host_pinned, sizeof(float4) * width * height);
        memcpy(costs_host, res->costs_host_pinned, sizeof(float) * width * height);
    }
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
    const cudaStream_t s = stream_ ? stream_ : 0;

    const int rows = jp_h.height;
    const int cols = jp_h.width;

    dim3 grid((cols + 15) / 16, (rows + 15) / 16, 1);
    dim3 blk (16, 16, 1);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record events on the SAME stream
    cudaEventRecord(start, s);

    // Launch on s (not default stream)
    JBU_cu<<<grid, blk, 0, s>>>(jp_d, jt_d, depth_d);
    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // Async copy on s, then fence s once
    CUDA_SAFE_CALL(cudaMemcpyAsync(
        depth_h, depth_d, sizeof(float) * rows * cols,
        cudaMemcpyDeviceToHost, s));

    CUDA_SAFE_CALL(cudaEventRecord(stop, s));
    CUDA_SAFE_CALL(cudaEventSynchronize(stop));

    float ms = 0.f;
    cudaEventElapsedTime(&ms, start, stop);
    // printf("Total time needed for computation: %f seconds\n", ms / 1000.f);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}
