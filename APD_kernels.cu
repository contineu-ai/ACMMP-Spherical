// APD_kernels.cu — APD (Adaptive Patch Deformation) kernels ported to
// ACMMP-Spherical. Geometry calls dispatch via _APD wrappers (spherical
// vs pinhole), RNG uses lightweight Philox-based RNGState, and kernels
// accept DataPassHelper for batched stream execution.
//
// Separate TU from ACMMP.cu — __device__ functions are marked static to
// avoid linker collisions with identically-named functions in ACMMP.cu.

#include "ACMMP.h"
#include "APD_device.cuh"
#include <math_constants.h>
#include <float.h>

// ============================================================================
// LOCAL CONSTANTS
// ============================================================================
#define APD_MAX_VIEWS 32           // Stack-allocated cost arrays (runtime num_images must be <= this)
#define APD_MAX_SEARCH_RADIUS 4096 // GenNeighbours directional search limit

// ============================================================================
// RNG — Philox4x32-10 (TU-local copies; same algorithm as ACMMP.cu)
// ============================================================================
__device__ __forceinline__ float apd_rng_uniform(RNGState* state, unsigned int pixel_key) {
    uint4 ctr = make_uint4(state->counter++, pixel_key, 0, 0);
    uint2 key = make_uint2(0xD2511F53u, 0xCD9E8D57u);
    uint4 result = curand_Philox4x32_10(ctr, key);
    return (result.x >> 8) * (1.0f / 16777215.0f) + (1.0f / 16777216.0f);
}

__device__ __forceinline__ unsigned int apd_rng_uint(RNGState* state, unsigned int pixel_key) {
    uint4 ctr = make_uint4(state->counter++, pixel_key, 0, 0);
    uint2 key = make_uint2(0xD2511F53u, 0xCD9E8D57u);
    uint4 result = curand_Philox4x32_10(ctr, key);
    return result.x;
}

// ============================================================================
// MASK HELPER
// ============================================================================
__device__ __forceinline__ bool apd_is_masked(const DataPassHelper *helper, int center) {
    return helper->ref_mask_cuda && helper->ref_mask_cuda[center] != 0;
}

// ============================================================================
// UTILITY DEVICE FUNCTIONS
// ============================================================================
static __device__ void sort_small(float *d, const int n) {
    int j;
    for (int i = 1; i < n; i++) {
        float tmp = d[i];
        for (j = i; j >= 1 && tmp < d[j - 1]; j--)
            d[j] = d[j - 1];
        d[j] = tmp;
    }
}

static __device__ void sort_small_weighted(short2 *points, float *w, int n) {
    int j;
    for (int i = 1; i < n; i++) {
        short2 tmp = points[i];
        float tmp_w = w[i];
        for (j = i; j >= 1 && tmp_w < w[j - 1]; j--) {
            points[j] = points[j - 1];
            w[j] = w[j - 1];
        }
        points[j] = tmp;
        w[j] = tmp_w;
    }
}

static __device__ int FindMinCostIndex(const float *costs, const int n) {
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

static __device__ void setBit(unsigned int *input, const unsigned int n) {
    (*input) |= (unsigned int)(1 << n);
}

static __device__ void unSetBit(unsigned int *input, const unsigned int n) {
    (*input) &= (unsigned int)(0xFFFFFFFE << n);
}

static __device__ int isSet(unsigned int input, const unsigned int n) {
    return (input >> n) & 1;
}

static __device__ void Mat33DotVec3(const float mat[9], const float4 vec, float4 *result) {
    result->x = mat[0] * vec.x + mat[1] * vec.y + mat[2] * vec.z;
    result->y = mat[3] * vec.x + mat[4] * vec.y + mat[5] * vec.z;
    result->z = mat[6] * vec.x + mat[7] * vec.y + mat[8] * vec.z;
}

static __device__ float Vec3DotVec3(const float4 vec1, const float4 vec2) {
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

static __device__ float Vec3DotVec3(const float3 vec1, const float3 vec2) {
    return vec1.x * vec2.x + vec1.y * vec2.y + vec1.z * vec2.z;
}

static __device__ float Vec2DotVec2(float2 a, float2 b) {
    return a.x * b.x + a.y * b.y;
}

static __device__ float Vec2CrossVec2(float2 a, float2 b) {
    return a.x * b.y - a.y * b.x;
}

static __device__ bool PointinTriangle(short2 A, short2 B, short2 C, int2 P) {
    float2 AB = make_float2(B.x - A.x, B.y - A.y);
    float2 BC = make_float2(C.x - B.x, C.y - B.y);
    float2 CA = make_float2(A.x - C.x, A.y - C.y);
    float AB_ = sqrtf(AB.x * AB.x + AB.y * AB.y);
    float BC_ = sqrtf(BC.x * BC.x + BC.y * BC.y);
    float CA_ = sqrtf(CA.x * CA.x + CA.y * CA.y);
    if (AB_ <= 2 || BC_ <= 2 || CA_ <= 2) return false;
    if (!(AB_ + BC_ > CA_ && BC_ + CA_ > AB_ && AB_ + CA_ > BC_)) return false;
    float2 PA = make_float2(A.x - P.x, A.y - P.y);
    float2 PB = make_float2(B.x - P.x, B.y - P.y);
    float2 PC = make_float2(C.x - P.x, C.y - P.y);
    float t1 = Vec2CrossVec2(PA, PB);
    float t2 = Vec2CrossVec2(PB, PC);
    float t3 = Vec2CrossVec2(PC, PA);
    return t1 * t2 >= 0 && t1 * t3 >= 0;
}

static __device__ void NormalizeVec3(float4 *vec) {
    const float normSquared = vec->x * vec->x + vec->y * vec->y + vec->z * vec->z;
    const float inverse_sqrt = rsqrtf(normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
    vec->z *= inverse_sqrt;
}

static __device__ void NormalizeVec2(float2 *vec) {
    const float normSquared = vec->x * vec->x + vec->y * vec->y;
    const float inverse_sqrt = rsqrtf(normSquared);
    vec->x *= inverse_sqrt;
    vec->y *= inverse_sqrt;
}

static __device__ void TransformPDFToCDF(float* probs, const int num_probs) {
    float prob_sum = 0.0f;
    for (int i = 0; i < num_probs; ++i) prob_sum += probs[i];
    const float inv_prob_sum = 1.0f / prob_sum;
    float cum_prob = 0.0f;
    for (int i = 0; i < num_probs; ++i) {
        cum_prob += probs[i] * inv_prob_sum;
        probs[i] = cum_prob;
    }
}

// ============================================================================
// GEOMETRY-AWARE DEVICE FUNCTIONS (use _APD dispatch wrappers)
// ============================================================================

static __device__ float4 GenerateRandomNormal(const Camera camera, const int2 p, RNGState *rand_state, unsigned int pixel_key, const float depth) {
    float4 normal;
    float q1 = 1.0f, q2 = 1.0f, s = 2.0f;
    while (s >= 1.0f) {
        q1 = 2.0f * apd_rng_uniform(rand_state, pixel_key) - 1.0f;
        q2 = 2.0f * apd_rng_uniform(rand_state, pixel_key) - 1.0f;
        s = q1 * q1 + q2 * q2;
    }
    const float sq = sqrtf(1.0f - s);
    normal.x = 2.0f * q1 * sq;
    normal.y = 2.0f * q2 * sq;
    normal.z = 1.0f - 2.0f * s;
    normal.w = 0;

    float4 view_direction = GetViewDirection_APD(camera, p, depth);
    float dot_product = normal.x * view_direction.x + normal.y * view_direction.y + normal.z * view_direction.z;
    if (dot_product > 0.0f) {
        normal.x = -normal.x;
        normal.y = -normal.y;
        normal.z = -normal.z;
    }
    NormalizeVec3(&normal);
    return normal;
}

static __device__ float4 GeneratePerturbedNormal(const Camera camera, const int2 p, const float4 normal, RNGState *rand_state, unsigned int pixel_key, const float perturbation) {
    float4 view_direction = GetViewDirection_APD(camera, p, 1.0f);

    const float a1 = (apd_rng_uniform(rand_state, pixel_key) - 0.5f) * perturbation;
    const float a2 = (apd_rng_uniform(rand_state, pixel_key) - 0.5f) * perturbation;
    const float a3 = (apd_rng_uniform(rand_state, pixel_key) - 0.5f) * perturbation;

    const float sin_a1 = sinf(a1), sin_a2 = sinf(a2), sin_a3 = sinf(a3);
    const float cos_a1 = cosf(a1), cos_a2 = cosf(a2), cos_a3 = cosf(a3);

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

static __device__ float4 GenerateRandomPlaneHypothesis(const Camera camera, const int2 p, RNGState *rand_state, unsigned int pixel_key, const float depth_min, const float depth_max) {
    float depth = apd_rng_uniform(rand_state, pixel_key) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis = GenerateRandomNormal(camera, p, rand_state, pixel_key, depth);
    plane_hypothesis.w = GetDistance2Origin_APD(camera, p, depth, plane_hypothesis);
    return plane_hypothesis;
}

static __device__ float4 GeneratePertubedPlaneHypothesis(const Camera camera, const int2 p, RNGState *rand_state, unsigned int pixel_key, const float perturbation, const float4 plane_hypothesis_now, const float depth_now, const float depth_min, const float depth_max) {
    float depth_perturbed = depth_now;
    float dist_perturbed = plane_hypothesis_now.w;
    const float dist_min_perturbed = (1 - perturbation) * dist_perturbed;
    const float dist_max_perturbed = (1 + perturbation) * dist_perturbed;
    float4 plane_hypothesis_temp = plane_hypothesis_now;
    int tries = 0;
    do {
        dist_perturbed = apd_rng_uniform(rand_state, pixel_key) * (dist_max_perturbed - dist_min_perturbed) + dist_min_perturbed;
        plane_hypothesis_temp.w = dist_perturbed;
        depth_perturbed = ComputeDepthfromPlaneHypothesis_APD(camera, plane_hypothesis_temp, p);
        if (++tries > 64) { depth_perturbed = depth_now; dist_perturbed = plane_hypothesis_now.w; break; }
    } while (depth_perturbed < depth_min || depth_perturbed > depth_max);

    float4 plane_hypothesis = GeneratePerturbedNormal(camera, p, plane_hypothesis_now, rand_state, pixel_key, perturbation * CUDART_PI_F);
    plane_hypothesis.w = dist_perturbed;
    return plane_hypothesis;
}

// Pinhole-only homography (uses precomputed camera.c[])
static __device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float *H) {
    float C_relative[3];
    C_relative[0] = ref_camera.c[0] - src_camera.c[0];
    C_relative[1] = ref_camera.c[1] - src_camera.c[1];
    C_relative[2] = ref_camera.c[2] - src_camera.c[2];

    float t_relative[3];
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

    float R_relative[9];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] * ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] * ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] * ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] * ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] * ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] * ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] * ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] * ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] * ref_camera.R[8];

    H[0] = R_relative[0] - t_relative[0] * plane_hypothesis.x / plane_hypothesis.w;
    H[1] = R_relative[1] - t_relative[0] * plane_hypothesis.y / plane_hypothesis.w;
    H[2] = R_relative[2] - t_relative[0] * plane_hypothesis.z / plane_hypothesis.w;
    H[3] = R_relative[3] - t_relative[1] * plane_hypothesis.x / plane_hypothesis.w;
    H[4] = R_relative[4] - t_relative[1] * plane_hypothesis.y / plane_hypothesis.w;
    H[5] = R_relative[5] - t_relative[1] * plane_hypothesis.z / plane_hypothesis.w;
    H[6] = R_relative[6] - t_relative[2] * plane_hypothesis.x / plane_hypothesis.w;
    H[7] = R_relative[7] - t_relative[2] * plane_hypothesis.y / plane_hypothesis.w;
    H[8] = R_relative[8] - t_relative[2] * plane_hypothesis.z / plane_hypothesis.w;

    float tmp[9];
    tmp[0] = H[0] / ref_camera.K[0];
    tmp[1] = H[1] / ref_camera.K[4];
    tmp[2] = -H[0] * ref_camera.K[2] / ref_camera.K[0] - H[1] * ref_camera.K[5] / ref_camera.K[4] + H[2];
    tmp[3] = H[3] / ref_camera.K[0];
    tmp[4] = H[4] / ref_camera.K[4];
    tmp[5] = -H[3] * ref_camera.K[2] / ref_camera.K[0] - H[4] * ref_camera.K[5] / ref_camera.K[4] + H[5];
    tmp[6] = H[6] / ref_camera.K[0];
    tmp[7] = H[7] / ref_camera.K[4];
    tmp[8] = -H[6] * ref_camera.K[2] / ref_camera.K[0] - H[7] * ref_camera.K[5] / ref_camera.K[4] + H[8];

    H[0] = src_camera.K[0] * tmp[0] + src_camera.K[2] * tmp[6];
    H[1] = src_camera.K[0] * tmp[1] + src_camera.K[2] * tmp[7];
    H[2] = src_camera.K[0] * tmp[2] + src_camera.K[2] * tmp[8];
    H[3] = src_camera.K[4] * tmp[3] + src_camera.K[5] * tmp[6];
    H[4] = src_camera.K[4] * tmp[4] + src_camera.K[5] * tmp[7];
    H[5] = src_camera.K[4] * tmp[5] + src_camera.K[5] * tmp[8];
    H[6] = src_camera.K[8] * tmp[6];
    H[7] = src_camera.K[8] * tmp[7];
    H[8] = src_camera.K[8] * tmp[8];
}

static __device__ float4 TransformNormal(const Camera camera, float4 plane_hypothesis) {
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[3] * plane_hypothesis.y + camera.R[6] * plane_hypothesis.z;
    transformed_normal.y = camera.R[1] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[7] * plane_hypothesis.z;
    transformed_normal.z = camera.R[2] * plane_hypothesis.x + camera.R[5] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

static __device__ float4 TransformNormal2RefCam(const Camera camera, float4 plane_hypothesis) {
    float4 transformed_normal;
    transformed_normal.x = camera.R[0] * plane_hypothesis.x + camera.R[1] * plane_hypothesis.y + camera.R[2] * plane_hypothesis.z;
    transformed_normal.y = camera.R[3] * plane_hypothesis.x + camera.R[4] * plane_hypothesis.y + camera.R[5] * plane_hypothesis.z;
    transformed_normal.z = camera.R[6] * plane_hypothesis.x + camera.R[7] * plane_hypothesis.y + camera.R[8] * plane_hypothesis.z;
    transformed_normal.w = plane_hypothesis.w;
    return transformed_normal;
}

static __device__ short2 GetNeighbourPoint(const int2 p, const int index, const DataPassHelper *helper) {
    const unsigned offset = helper->neighbours_map_cuda[p.x + p.y * helper->width] * NEIGHBOUR_NUM;
    short2 neighbour_pt = helper->neighbours_cuda[offset + index];
    return neighbour_pt;
}

// ============================================================================
// NCC COMPUTATION (adapted for spherical dispatch)
// ============================================================================

static __device__ float ComputeBilateralNCCOld(
    const int2 p, const int src_idx, const float4 plane_hypothesis,
    const DataPassHelper *helper
) {
    const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];
    const Camera ref_camera = helper->cameras_cuda[0];
    const cudaTextureObject_t src_image = helper->texture_objects_cuda[0].images[src_idx];
    const Camera src_camera = helper->cameras_cuda[src_idx];
    const int width = helper->width;

    const float cost_max = 2.0f;

    float H[9] = {0};
    if (ref_camera.model != SPHERE) {
        ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
    }
    float2 pt = ComputeCorrespondingPoint_APD(ref_camera, src_camera, plane_hypothesis, p, H);
    if (!CheckAndWrapSourcePoint(pt, src_camera)) {
        return cost_max;
    }

    const int radius = helper->params->strong_radius;
    const int increment = helper->params->strong_increment;

    float sum_ref = 0.0f, sum_ref_ref = 0.0f;
    float sum_src = 0.0f, sum_src_src = 0.0f;
    float sum_ref_src = 0.0f;
    float bilateral_weight_sum = 0.0f;

    for (int i = -radius; i <= radius; i += increment) {
        float sum_ref_row = 0.0f, sum_src_row = 0.0f;
        float sum_ref_ref_row = 0.0f, sum_src_src_row = 0.0f;
        float sum_ref_src_row = 0.0f, bilateral_weight_sum_row = 0.0f;

        for (int j = -radius; j <= radius; j += increment) {
            int2 ref_pt = make_int2(p.x + i, p.y + j);
            if (ref_camera.model == SPHERE) {
                ref_pt.x = ((ref_pt.x % width) + width) % width;
            }
            const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
            float2 src_pt = ComputeCorrespondingPoint_APD(ref_camera, src_camera, plane_hypothesis, ref_pt, H);
            CheckAndWrapSourcePoint(src_pt, src_camera);
            const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
            float weight = 1.0f;
            sum_ref_row += weight * ref_pix;
            sum_ref_ref_row += weight * ref_pix * ref_pix;
            sum_src_row += weight * src_pix;
            sum_src_src_row += weight * src_pix * src_pix;
            sum_ref_src_row += weight * ref_pix * src_pix;
            bilateral_weight_sum_row += weight;
        }
        sum_ref += sum_ref_row;
        sum_ref_ref += sum_ref_ref_row;
        sum_src += sum_src_row;
        sum_src_src += sum_src_src_row;
        sum_ref_src += sum_ref_src_row;
        bilateral_weight_sum += bilateral_weight_sum_row;
    }
    const float inv_bws = 1.0f / bilateral_weight_sum;
    sum_ref *= inv_bws; sum_ref_ref *= inv_bws;
    sum_src *= inv_bws; sum_src_src *= inv_bws;
    sum_ref_src *= inv_bws;

    const float var_ref = sum_ref_ref - sum_ref * sum_ref;
    const float var_src = sum_src_src - sum_src * sum_src;
    const float kMinVar = 1e-5f;
    if (var_ref < kMinVar || var_src < kMinVar) return cost_max;
    const float covar = sum_ref_src - sum_ref * sum_src;
    return max(0.0f, min(cost_max, 1.0f - covar / sqrtf(var_ref * var_src)));
}

static __device__ float ComputeBilateralNCCNew(
    const int2 p, const int src_idx, const float4 plane_hypothesis,
    const DataPassHelper *helper
) {
    const cudaTextureObject_t ref_image = helper->texture_objects_cuda[0].images[0];
    const Camera ref_camera = helper->cameras_cuda[0];
    const cudaTextureObject_t src_image = helper->texture_objects_cuda[0].images[src_idx];
    const Camera src_camera = helper->cameras_cuda[src_idx];
    const PatchMatchParams *params = helper->params;
    const uchar *weak_info = helper->weak_info_cuda;
    const int width = helper->width;
    const int height = helper->height;
    const int center = p.x + p.y * width;

    const float cost_max = 2.0f;

    float H[9] = {0};
    if (ref_camera.model != SPHERE) {
        ComputeHomography(ref_camera, src_camera, plane_hypothesis, H);
    }
    float2 pt = ComputeCorrespondingPoint_APD(ref_camera, src_camera, plane_hypothesis, p, H);
    if (!CheckAndWrapSourcePoint(pt, src_camera)) {
        return cost_max;
    }

    float cost = 0.0f;
    if (weak_info[center] == WEAK) {
        float center_cost = 0.0f;
        float strong_cost = 0.0f;
        int strong_count = 0;

        for (int k = 0; k < NEIGHBOUR_NUM; ++k) {
            const short2 neighbour_pt = GetNeighbourPoint(p, k, helper);
            if (neighbour_pt.x == -1 || neighbour_pt.y == -1) continue;

            float2 neighbour_src_pt = ComputeCorrespondingPoint_APD(ref_camera, src_camera, plane_hypothesis, neighbour_pt, H);
            if (!CheckAndWrapSourcePoint(neighbour_src_pt, src_camera)) {
                if (k != 0) {
                    unsigned int view_info = helper->selected_views_cuda[neighbour_pt.x + neighbour_pt.y * width];
                    if (isSet(view_info, src_idx - 1)) {
                        strong_cost += cost_max;
                        strong_count++;
                    }
                    continue;
                } else {
                    return cost_max;
                }
            }

            float sum_ref = 0.0f, sum_ref_ref = 0.0f;
            float sum_src = 0.0f, sum_src_src = 0.0f;
            float sum_ref_src = 0.0f, bilateral_weight_sum = 0.0f;
            int r = (k == 0 ? params->strong_radius : params->weak_radius);
            int inc = (k == 0 ? params->strong_increment : params->weak_increment);

            for (int i = -r; i <= r; i += inc) {
                float sr = 0, srs = 0, ss = 0, sss = 0, sref_src = 0, bws = 0;
                for (int j = -r; j <= r; j += inc) {
                    int2 ref_pt = make_int2(neighbour_pt.x + i, neighbour_pt.y + j);
                    if (ref_camera.model == SPHERE) {
                        ref_pt.x = ((ref_pt.x % width) + width) % width;
                    }
                    const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);
                    float2 src_pt = ComputeCorrespondingPoint_APD(ref_camera, src_camera, plane_hypothesis, ref_pt, H);
                    CheckAndWrapSourcePoint(src_pt, src_camera);
                    const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);
                    float weight = 1.0f;
                    sr += weight * ref_pix;
                    srs += weight * ref_pix * ref_pix;
                    ss += weight * src_pix;
                    sss += weight * src_pix * src_pix;
                    sref_src += weight * ref_pix * src_pix;
                    bws += weight;
                }
                sum_ref += sr; sum_ref_ref += srs;
                sum_src += ss; sum_src_src += sss;
                sum_ref_src += sref_src; bilateral_weight_sum += bws;
            }
            const float inv_bws = 1.0f / bilateral_weight_sum;
            sum_ref *= inv_bws; sum_ref_ref *= inv_bws;
            sum_src *= inv_bws; sum_src_src *= inv_bws;
            sum_ref_src *= inv_bws;

            const float var_ref = sum_ref_ref - sum_ref * sum_ref;
            const float var_src = sum_src_src - sum_src * sum_src;
            const float kMinVar = 1e-5f;
            float temp_cost;
            if (var_ref < kMinVar || var_src < kMinVar) {
                temp_cost = cost_max;
            } else {
                const float covar = sum_ref_src - sum_ref * sum_src;
                temp_cost = max(0.0f, min(cost_max, 1.0f - covar / sqrtf(var_ref * var_src)));
            }
            if (k == 0) {
                center_cost = temp_cost;
            } else {
                strong_cost += temp_cost;
                strong_count++;
            }
        }
        if (strong_count == 0) {
            cost = center_cost;
        } else {
            strong_cost /= strong_count;
            strong_cost = min(strong_cost, cost_max);
            cost = 0.25f * center_cost + 0.75f * strong_cost;
        }
    } else {
        // Should not reach here for non-WEAK pixels in New NCC
        cost = cost_max;
    }
    return cost;
}

static __device__ float ComputeMultiViewInitialCostandSelectedViews(const int2 p, DataPassHelper *helper) {
    PatchMatchParams *params = helper->params;
    unsigned int *selected_views = helper->selected_views_cuda;
    int center = p.x + p.y * helper->width;
    float4 plane_hypothesis = helper->plane_hypotheses_cuda[center];
    float cost_max = 2.0f;
    float cost_vector[APD_MAX_VIEWS] = {0};
    float cost_vector_copy[APD_MAX_VIEWS] = {0};
    int cost_count = 0;
    int num_valid_views = 0;
    for (int i = 1; i < params->num_images; ++i) {
        float c = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
        cost_vector[i - 1] = c;
        cost_vector_copy[i - 1] = c;
        cost_count++;
        if (c < cost_max) num_valid_views++;
    }
    sort_small(cost_vector, cost_count);
    selected_views[center] = 0;
    int top_k = min(num_valid_views, params->top_k);
    if (top_k > 0) {
        float cost = 0.0f;
        for (int i = 0; i < top_k; ++i) cost += cost_vector[i];
        float cost_threshold = cost_vector[top_k - 1];
        for (int i = 0; i < params->num_images - 1; ++i) {
            if (cost_vector_copy[i] <= cost_threshold) {
                setBit(&(selected_views[center]), i);
            }
        }
        return cost / top_k;
    }
    return cost_max;
}

static __device__ float ComputeMultiViewInitialCost(const int2 p, DataPassHelper *helper) {
    PatchMatchParams *params = helper->params;
    unsigned int *selected_views = helper->selected_views_cuda;
    int center = p.x + p.y * helper->width;
    float4 plane_hypothesis = helper->plane_hypotheses_cuda[center];
    const float cost_max = 2.0f;
    int cost_count = 0;
    float cost = 0.0f;
    for (int i = 1; i < params->num_images; ++i) {
        if (isSet(selected_views[center], i - 1)) {
            float c = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
            if (c < cost_max) {
                cost_count++;
                cost += c;
            } else {
                unSetBit(&(selected_views[center]), i - 1);
            }
        }
    }
    return (cost_count == 0) ? cost_max : cost / cost_count;
}

static __device__ void ComputeMultiViewCostVectorNew(const int2 p, float4 plane_hypothesis, float *cost_vector, DataPassHelper *helper) {
    for (int i = 1; i < helper->params->num_images; ++i) {
        cost_vector[i - 1] = ComputeBilateralNCCNew(p, i, plane_hypothesis, helper);
    }
}

static __device__ void ComputeMultiViewCostVectorOld(const int2 p, float4 plane_hypothesis, float *cost_vector, DataPassHelper *helper) {
    for (int i = 1; i < helper->params->num_images; ++i) {
        cost_vector[i - 1] = ComputeBilateralNCCOld(p, i, plane_hypothesis, helper);
    }
}

static __device__ float ComputeGeomConsistencyCost(const int2 p, const int src_idx, const float4 plane_hypothesis, DataPassHelper *helper) {
    const Camera ref_camera = helper->cameras_cuda[0];
    const Camera src_camera = helper->cameras_cuda[src_idx];
    const cudaTextureObject_t depth_image = helper->texture_depths_cuda[0].images[src_idx];
    const float max_cost = 3.0f;

    float depth = ComputeDepthfromPlaneHypothesis_APD(ref_camera, plane_hypothesis, p);
    float3 forward_point = Get3DPointonWorld_APD((float)p.x, (float)p.y, depth, ref_camera);

    float2 src_pt; float src_d;
    ProjectonCamera_APD(forward_point, src_camera, src_pt, src_d);
    if (!CheckAndWrapSourcePoint(src_pt, src_camera)) return max_cost;
    const float src_depth = tex2D<float>(depth_image, (int)src_pt.x + 0.5f, (int)src_pt.y + 0.5f);
    if (src_depth == 0.0f) return max_cost;

    float3 src_3D_pt = Get3DPointonWorld_APD(src_pt.x, src_pt.y, src_depth, src_camera);
    float2 backward_point; float ref_d;
    ProjectonCamera_APD(src_3D_pt, ref_camera, backward_point, ref_d);

    const float diff_col = p.x - backward_point.x;
    const float diff_row = p.y - backward_point.y;
    return min(max_cost, sqrtf(diff_col * diff_col + diff_row * diff_row));
}

// ============================================================================
// PROPAGATION / REFINEMENT DEVICE FUNCTIONS
// ============================================================================

static __device__ void PlaneHypothesisRefinementStrong(
    float4 *plane_hypothesis, float *depth, float *cost,
    RNGState *rand_state, unsigned int pixel_key,
    const uchar *view_weights, const float weight_norm,
    const int2 p, DataPassHelper *helper
) {
    float depth_perturbation = 0.02f;
    float normal_perturbation = 0.02f;
    const Camera *cameras = helper->cameras_cuda;
    const PatchMatchParams *params = helper->params;
    float depth_min = params->depth_min;
    float depth_max = params->depth_max;

    float depth_rand = apd_rng_uniform(rand_state, pixel_key) * (depth_max - depth_min) + depth_min;
    float4 plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, pixel_key, *depth);
    float depth_perturbed = *depth;
    const float depth_min_p = (1 - depth_perturbation) * depth_perturbed;
    const float depth_max_p = (1 + depth_perturbation) * depth_perturbed;
    { int tries = 0;
    do {
        depth_perturbed = apd_rng_uniform(rand_state, pixel_key) * (depth_max_p - depth_min_p) + depth_min_p;
        if (++tries > 64) { depth_perturbed = *depth; break; }
    } while (depth_perturbed < depth_min || depth_perturbed > depth_max);
    }
    float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, pixel_key, normal_perturbation * CUDART_PI_F);

    const int num_planes = 5;
    float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, depth_perturbed };
    float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis };

    for (int i = 0; i < num_planes; ++i) {
        float cost_vector[APD_MAX_VIEWS] = {0};
        float4 temp_ph = normals[i];
        temp_ph.w = GetDistance2Origin_APD(cameras[0], p, depths[i], temp_ph);
        ComputeMultiViewCostVectorOld(p, temp_ph, cost_vector, helper);

        float temp_cost = 0.0f;
        for (int j = 0; j < params->num_images - 1; ++j) {
            if (view_weights[j] > 0) temp_cost += view_weights[j] * cost_vector[j];
        }
        temp_cost /= weight_norm;

        float depth_before = ComputeDepthfromPlaneHypothesis_APD(cameras[0], temp_ph, p);
        if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
            *depth = depth_before;
            *plane_hypothesis = temp_ph;
            *cost = temp_cost;
        }
    }
}

static __device__ void PlaneHypothesisRefinementWeak(
    float4 *plane_hypothesis, float *depth, float *cost,
    RNGState *rand_state, unsigned int pixel_key,
    const uchar *view_weights, const float weight_norm,
    const int2 p, DataPassHelper *helper
) {
    float depth_perturbation = 0.02f;
    float normal_perturbation = 0.02f;
    const Camera *cameras = helper->cameras_cuda;
    const PatchMatchParams *params = helper->params;
    float depth_min = params->depth_min;
    float depth_max = params->depth_max;
    const int center = p.x + p.y * helper->width;

    {   // test the fit plane
        float4 fit_ph = helper->fit_plane_hypotheses_cuda[center];
        if (fit_ph.x == 0 && fit_ph.y == 0 && fit_ph.z == 0) return;
        float cost_vector[APD_MAX_VIEWS] = {0};
        ComputeMultiViewCostVectorNew(p, fit_ph, cost_vector, helper);
        float temp_cost = 0.0f;
        for (int j = 0; j < params->num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params->geom_consistency) {
                    temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, fit_ph, helper));
                } else {
                    temp_cost += view_weights[j] * cost_vector[j];
                }
            }
        }
        temp_cost /= weight_norm;
        float depth_before = ComputeDepthfromPlaneHypothesis_APD(cameras[0], fit_ph, p);
        if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
            *depth = depth_before;
            *plane_hypothesis = fit_ph;
            *cost = temp_cost;
        }
    }
    // random refine
    {
        float depth_rand = apd_rng_uniform(rand_state, pixel_key) * (depth_max - depth_min) + depth_min;
        float4 plane_hypothesis_rand = GenerateRandomNormal(cameras[0], p, rand_state, pixel_key, *depth);
        float depth_perturbed = *depth;
        const float depth_min_p = (1 - depth_perturbation) * depth_perturbed;
        const float depth_max_p = (1 + depth_perturbation) * depth_perturbed;
        { int tries = 0;
        do {
            depth_perturbed = apd_rng_uniform(rand_state, pixel_key) * (depth_max_p - depth_min_p) + depth_min_p;
            if (++tries > 64) { depth_perturbed = *depth; break; }
        } while (depth_perturbed < depth_min || depth_perturbed > depth_max);
        }
        float4 plane_hypothesis_perturbed = GeneratePerturbedNormal(cameras[0], p, *plane_hypothesis, rand_state, pixel_key, normal_perturbation * CUDART_PI_F);

        const int num_planes = 5;
        float depths[num_planes] = { depth_rand, *depth, depth_rand, *depth, depth_perturbed };
        float4 normals[num_planes] = { *plane_hypothesis, plane_hypothesis_rand, plane_hypothesis_rand, plane_hypothesis_perturbed, *plane_hypothesis };

        for (int i = 0; i < num_planes; ++i) {
            float cost_vector[APD_MAX_VIEWS] = {0};
            float4 temp_ph = normals[i];
            temp_ph.w = GetDistance2Origin_APD(cameras[0], p, depths[i], temp_ph);
            ComputeMultiViewCostVectorNew(p, temp_ph, cost_vector, helper);

            float temp_cost = 0.0f;
            for (int j = 0; j < params->num_images - 1; ++j) {
                if (view_weights[j] > 0) {
                    if (params->geom_consistency) {
                        temp_cost += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, temp_ph, helper));
                    } else {
                        temp_cost += view_weights[j] * cost_vector[j];
                    }
                }
            }
            temp_cost /= weight_norm;
            float depth_before = ComputeDepthfromPlaneHypothesis_APD(cameras[0], temp_ph, p);
            if (depth_before >= depth_min && depth_before <= depth_max && temp_cost < *cost) {
                *depth = depth_before;
                *plane_hypothesis = temp_ph;
                *cost = temp_cost;
            }
        }
    }
}

template<int NUM_IMAGES>
static __device__ void CheckerboardPropagationStrong(const int2 p, const int iter, DataPassHelper *helper) {
    const int width = helper->width;
    const int height = helper->height;
    float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    float *costs = helper->costs_cuda;
    RNGState *rand_states = helper->rand_states_cuda;
    unsigned int *selected_views = helper->selected_views_cuda;
    PatchMatchParams *params = helper->params;
    const Camera *cameras = helper->cameras_cuda;
    int num_images = params->num_images;

    if (p.x >= width || p.y >= height) return;
    const int center = p.y * width + p.x;
    const unsigned int pixel_key = (unsigned int)center;

    // Two-pass streaming accumulator: eliminates cost_array[8][NV] register spill.
    constexpr int NV = NUM_IMAGES - 1;
    bool flag[8] = { false };
    int num_valid_pixels = 0;
    float costMin; int costMinPoint;

    // Streaming accumulators (replaces cost_array[8][NV])
    float sp_count[NV];
    float sp_tmpw[NV];
    int sp_count_false[NV];
    for (int v = 0; v < NV; ++v) { sp_count[v] = 0.0f; sp_tmpw[v] = 0.0f; sp_count_false[v] = 0; }

    const float cost_threshold_sp = 0.8f * expf((float)(iter * iter) / (-90.0f));
    float cost_vector[NV]; // Reusable per-direction cost vector

    int left_near = center - 1;
    int left_far = center - 3;
    int right_near = center + 1;
    int right_far = center + 3;
    int up_near = center - width;
    int up_far = center - 3 * width;
    int down_near = center + width;
    int down_far = center + 3 * width;

    // Helper macro: accumulate cost_vector into streaming accumulators
    #define APD_ACCUMULATE_STATS() \
        for (int v = 0; v < num_images - 1; ++v) { \
            if (cost_vector[v] < cost_threshold_sp) { \
                sp_tmpw[v] += expf(cost_vector[v] * cost_vector[v] / (-0.18f)); \
                sp_count[v] += 1.0f; \
            } \
            if (cost_vector[v] > 1.2f) sp_count_false[v]++; \
        }

    // ---- Pass 1: Evaluate neighbors, accumulate per-view statistics ----

    // up_far
    if (p.y > 2) {
        flag[1] = true; num_valid_pixels++;
        costMin = costs[up_far]; costMinPoint = up_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y > 2 + 2 * i) {
                int pt = up_far - 2 * i * width;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        up_far = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[up_far], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // down_far
    if (p.y < height - 3) {
        flag[3] = true; num_valid_pixels++;
        costMin = costs[down_far]; costMinPoint = down_far;
        for (int i = 1; i < 11; ++i) {
            if (p.y < height - 3 - 2 * i) {
                int pt = down_far + 2 * i * width;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        down_far = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[down_far], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // left_far
    if (p.x > 2) {
        flag[5] = true; num_valid_pixels++;
        costMin = costs[left_far]; costMinPoint = left_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x > 2 + 2 * i) {
                int pt = left_far - 2 * i;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        left_far = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_far], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // right_far
    if (p.x < width - 3) {
        flag[7] = true; num_valid_pixels++;
        costMin = costs[right_far]; costMinPoint = right_far;
        for (int i = 1; i < 11; ++i) {
            if (p.x < width - 3 - 2 * i) {
                int pt = right_far + 2 * i;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        right_far = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_far], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // up_near
    if (p.y > 0) {
        flag[0] = true; num_valid_pixels++;
        costMin = costs[up_near]; costMinPoint = up_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y > 1 + i && p.x > i) {
                int pt = up_near - (1+i)*width - (1+i);
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
            if (p.y > 1 + i && p.x < width - 1 - i) {
                int pt = up_near - (1+i)*width + (1+i);
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        up_near = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[up_near], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // down_near
    if (p.y < height - 1) {
        flag[2] = true; num_valid_pixels++;
        costMin = costs[down_near]; costMinPoint = down_near;
        for (int i = 0; i < 3; ++i) {
            if (p.y < height - 2 - i && p.x > i) {
                int pt = down_near + (1+i)*width - (1+i);
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
            if (p.y < height - 2 - i && p.x < width - 1 - i) {
                int pt = down_near + (1+i)*width + (1+i);
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        down_near = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[down_near], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // left_near
    if (p.x > 0) {
        flag[4] = true; num_valid_pixels++;
        costMin = costs[left_near]; costMinPoint = left_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x > 1 + i && p.y > i) {
                int pt = left_near - (1+i) - (1+i)*width;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
            if (p.x > 1 + i && p.y < height - 1 - i) {
                int pt = left_near - (1+i) + (1+i)*width;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        left_near = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[left_near], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }
    // right_near
    if (p.x < width - 1) {
        flag[6] = true; num_valid_pixels++;
        costMin = costs[right_near]; costMinPoint = right_near;
        for (int i = 0; i < 3; ++i) {
            if (p.x < width - 2 - i && p.y > i) {
                int pt = right_near + (1+i) - (1+i)*width;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
            if (p.x < width - 2 - i && p.y < height - 1 - i) {
                int pt = right_near + (1+i) + (1+i)*width;
                if (costs[pt] < costMin) { costMin = costs[pt]; costMinPoint = pt; }
            }
        }
        right_near = costMinPoint;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[right_near], cost_vector, helper);
        APD_ACCUMULATE_STATS();
    }

    #undef APD_ACCUMULATE_STATS

    const int positions[8] = {up_near, up_far, down_near, down_far, left_near, left_far, right_near, right_far};

    // Multi-hypothesis Joint View Selection
    uchar *view_weights = &(helper->view_weight_cuda[center * helper->view_weight_stride]);
    for (int i = 0; i < helper->view_weight_stride; ++i) view_weights[i] = 0;
    float view_selection_priors[NV] = {0};

    int neighbor_positions[4] = { center - width, center + width, center - 1, center + 1 };
    for (int i = 0; i < 4; ++i) {
        if (flag[2 * i]) {
            for (int j = 0; j < num_images - 1; ++j) {
                if (isSet(selected_views[neighbor_positions[i]], j) == 1)
                    view_selection_priors[j] += 0.9f;
                else
                    view_selection_priors[j] += 0.1f;
            }
        }
    }

    // Compute sampling probabilities from streaming accumulators
    float sampling_probs[NV] = {0};
    const float threshold_exp = expf(cost_threshold_sp * cost_threshold_sp / (-0.32f));
    for (int i = 0; i < num_images - 1; i++) {
        if (sp_count[i] > 2.0f && sp_count_false[i] < 3)
            sampling_probs[i] = (sp_tmpw[i] / sp_count[i]) * view_selection_priors[i];
        else if (sp_count_false[i] < 3)
            sampling_probs[i] = threshold_exp * view_selection_priors[i];
        else
            sampling_probs[i] = 0.0f;
    }

    TransformPDFToCDF(sampling_probs, num_images - 1);
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = apd_rng_uniform(&rand_states[center], pixel_key) - FLT_EPSILON;
        for (int image_id = 0; image_id < num_images - 1; ++image_id) {
            if (sampling_probs[image_id] > rand_prob) {
                view_weights[image_id] += 1;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    float weight_norm = 0;
    for (int i = 0; i < num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            setBit(&temp_selected_views, i);
            weight_norm += view_weights[i];
        }
    }

    // ---- Pass 2: Recompute cost vectors, compute weighted final costs ----
    float final_costs[8] = {0};
    for (int d = 0; d < 8; ++d) {
        if (!flag[d]) continue;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[positions[d]], cost_vector, helper);
        for (int j = 0; j < num_images - 1; ++j) {
            if (view_weights[j] > 0) final_costs[d] += view_weights[j] * cost_vector[j];
        }
        final_costs[d] /= weight_norm;
    }

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    // Compute center pixel cost (reuse cost_vector)
    ComputeMultiViewCostVectorOld(p, plane_hypotheses[center], cost_vector, helper);
    float cost_now = 0.0f;
    for (int i = 0; i < num_images - 1; ++i) cost_now += view_weights[i] * cost_vector[i];
    cost_now /= weight_norm;
    costs[center] = cost_now;
    float depth_now = ComputeDepthfromPlaneHypothesis_APD(cameras[0], plane_hypotheses[center], p);
    float4 plane_hypotheses_now = plane_hypotheses[center];

    if (flag[min_cost_idx]) {
        float depth_before = ComputeDepthfromPlaneHypothesis_APD(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
        if (depth_before >= params->depth_min && depth_before <= params->depth_max && final_costs[min_cost_idx] < cost_now) {
            depth_now = depth_before;
            plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
            cost_now = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }
    PlaneHypothesisRefinementStrong(&plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], pixel_key, view_weights, weight_norm, p, helper);

    if (params->state == REFINE_INIT) {
        if (cost_now < costs[center] - 0.1f) {
            costs[center] = cost_now;
            plane_hypotheses[center] = plane_hypotheses_now;
        }
    } else {
        costs[center] = cost_now;
        plane_hypotheses[center] = plane_hypotheses_now;
    }
}

template<int NUM_IMAGES>
static __device__ void CheckerboardPropagationWeak(const int2 p, const int iter, DataPassHelper *helper) {
    const int width = helper->width;
    const int height = helper->height;
    float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    float *costs = helper->costs_cuda;
    RNGState *rand_states = helper->rand_states_cuda;
    unsigned int *selected_views = helper->selected_views_cuda;
    PatchMatchParams *params = helper->params;
    const Camera *cameras = helper->cameras_cuda;
    int num_images = params->num_images;

    if (p.x >= width || p.y >= height) return;
    const int center = p.y * width + p.x;
    const unsigned int pixel_key = (unsigned int)center;

    // Two-pass streaming accumulator: eliminates cost_array[8][NV] and new_plane_hypothesis[8].
    constexpr int NV = NUM_IMAGES - 1;
    bool flag[8] = { false };
    int num_valid_pixels = 0;
    int positions[8] = { 0 };

    // Streaming accumulators (replaces cost_array[8][NV])
    float sp_count[NV];
    float sp_tmpw[NV];
    int sp_count_false[NV];
    for (int v = 0; v < NV; ++v) { sp_count[v] = 0.0f; sp_tmpw[v] = 0.0f; sp_count_false[v] = 0; }

    const float cost_threshold_sp = 0.8f * expf((float)(iter * iter) / (-90.0f));
    float cost_vector[NV]; // Reusable per-direction cost vector

    // ---- Pass 1: Evaluate neighbors, accumulate per-view statistics ----
    for (int i = 0; i < 8; ++i) {
        const auto neighbour_pt = GetNeighbourPoint(p, i + 1, helper);
        if (neighbour_pt.x == -1 || neighbour_pt.y == -1 || helper->weak_info_cuda[neighbour_pt.x + neighbour_pt.y * width] != STRONG) {
            flag[i] = false;
            continue;
        }
        positions[i] = neighbour_pt.x + neighbour_pt.y * width;
        flag[i] = true;
        num_valid_pixels++;
        ComputeMultiViewCostVectorNew(p, plane_hypotheses[positions[i]], cost_vector, helper);
        for (int v = 0; v < num_images - 1; ++v) {
            if (cost_vector[v] < cost_threshold_sp) {
                sp_tmpw[v] += expf(cost_vector[v] * cost_vector[v] / (-0.18f));
                sp_count[v] += 1.0f;
            }
            if (cost_vector[v] > 1.2f) sp_count_false[v]++;
        }
    }

    // Multi-hypothesis Joint View Selection
    uchar *view_weights = &(helper->view_weight_cuda[center * helper->view_weight_stride]);
    for (int i = 0; i < helper->view_weight_stride; ++i) view_weights[i] = 0;
    float view_selection_priors[NV] = {0};
    for (int i = 0; i < 8; ++i) {
        const auto neighbour_pt = GetNeighbourPoint(p, i + 1, helper);
        if (neighbour_pt.x == -1 || neighbour_pt.y == -1) continue;
        for (int j = 0; j < num_images - 1; ++j) {
            if (isSet(selected_views[neighbour_pt.x + neighbour_pt.y * width], j) == 1)
                view_selection_priors[j] += 0.9f;
            else
                view_selection_priors[j] += 0.1f;
        }
    }

    // Compute sampling probabilities from streaming accumulators
    float sampling_probs[NV] = {0};
    const float threshold_exp = expf(cost_threshold_sp * cost_threshold_sp / (-0.32f));
    for (int i = 0; i < num_images - 1; i++) {
        if (sp_count[i] > 2.0f && sp_count_false[i] < 3)
            sampling_probs[i] = (sp_tmpw[i] / sp_count[i]) * view_selection_priors[i];
        else if (sp_count_false[i] < 3)
            sampling_probs[i] = threshold_exp * view_selection_priors[i];
        else
            sampling_probs[i] = 0.0f;
    }

    TransformPDFToCDF(sampling_probs, num_images - 1);
    for (int sample = 0; sample < 15; ++sample) {
        const float rand_prob = apd_rng_uniform(&rand_states[center], pixel_key) - FLT_EPSILON;
        for (int image_id = 0; image_id < num_images - 1; ++image_id) {
            if (sampling_probs[image_id] > rand_prob) {
                view_weights[image_id] += 1;
                break;
            }
        }
    }

    unsigned int temp_selected_views = 0;
    float weight_norm = 0;
    for (int i = 0; i < num_images - 1; ++i) {
        if (view_weights[i] > 0) {
            setBit(&temp_selected_views, i);
            weight_norm += view_weights[i];
        }
    }

    // ---- Pass 2: Recompute cost vectors, compute weighted final costs ----
    float final_costs[8] = {0};
    for (int d = 0; d < 8; ++d) {
        if (!flag[d]) continue;
        ComputeMultiViewCostVectorNew(p, plane_hypotheses[positions[d]], cost_vector, helper);
        for (int j = 0; j < num_images - 1; ++j) {
            if (view_weights[j] > 0) {
                if (params->geom_consistency)
                    final_costs[d] += view_weights[j] * (cost_vector[j] + params->geom_factor * ComputeGeomConsistencyCost(p, j + 1, plane_hypotheses[positions[d]], helper));
                else
                    final_costs[d] += view_weights[j] * cost_vector[j];
            }
        }
        final_costs[d] /= weight_norm;
    }

    const int min_cost_idx = FindMinCostIndex(final_costs, 8);

    // Compute center pixel cost (reuse cost_vector)
    ComputeMultiViewCostVectorNew(p, plane_hypotheses[center], cost_vector, helper);
    float cost_now = 0.0f;
    for (int i = 0; i < num_images - 1; ++i) {
        if (params->geom_consistency)
            cost_now += view_weights[i] * (cost_vector[i] + params->geom_factor * ComputeGeomConsistencyCost(p, i + 1, plane_hypotheses[center], helper));
        else
            cost_now += view_weights[i] * cost_vector[i];
    }
    cost_now /= weight_norm;
    costs[center] = cost_now;
    float depth_now = ComputeDepthfromPlaneHypothesis_APD(cameras[0], plane_hypotheses[center], p);
    float4 plane_hypotheses_now = plane_hypotheses[center];

    if (flag[min_cost_idx]) {
        float depth_before = ComputeDepthfromPlaneHypothesis_APD(cameras[0], plane_hypotheses[positions[min_cost_idx]], p);
        if (depth_before >= params->depth_min && depth_before <= params->depth_max && final_costs[min_cost_idx] < cost_now) {
            depth_now = depth_before;
            plane_hypotheses_now = plane_hypotheses[positions[min_cost_idx]];
            cost_now = final_costs[min_cost_idx];
            selected_views[center] = temp_selected_views;
        }
    }
    PlaneHypothesisRefinementWeak(&plane_hypotheses_now, &depth_now, &cost_now, &rand_states[center], pixel_key, view_weights, weight_norm, p, helper);

    if (params->state == REFINE_INIT) {
        if (cost_now < costs[center] - 0.1f) {
            costs[center] = cost_now;
            plane_hypotheses[center] = plane_hypotheses_now;
        }
    } else {
        costs[center] = cost_now;
        plane_hypotheses[center] = plane_hypotheses_now;
    }

    {// update cost with old method
        cost_now = 0.0f;
        ComputeMultiViewCostVectorOld(p, plane_hypotheses[center], cost_vector, helper);
        for (int i = 0; i < num_images - 1; ++i)
            cost_now += view_weights[i] * cost_vector[i];
        cost_now /= weight_norm;
        costs[center] = cost_now;
    }
}

static __device__ void CheckerboardFilterStrong(const int2 p, DataPassHelper *helper) {
    int width = helper->width;
    int height = helper->height;
    if (p.x >= width || p.y >= height) return;
    float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    float *costs = helper->costs_cuda;
    const int center = p.y * width + p.x;

    float filter[21];
    int index = 0;
    filter[index++] = plane_hypotheses[center].w;

    const int left = center - 1, leftleft = center - 3;
    const int up = center - width, upup = center - 3 * width;
    const int down = center + width, downdown = center + 3 * width;
    const int right = center + 1, rightright = center + 3;

    if (costs[center] < 0.001f) return;

    if (p.y>0 && helper->weak_info_cuda[up] == STRONG) filter[index++] = plane_hypotheses[up].w;
    if (p.y>2 && helper->weak_info_cuda[upup] == STRONG) filter[index++] = plane_hypotheses[upup].w;
    if (p.y>4 && helper->weak_info_cuda[upup - width*2] == STRONG) filter[index++] = plane_hypotheses[upup - width*2].w;
    if (p.y<height-1 && helper->weak_info_cuda[down] == STRONG) filter[index++] = plane_hypotheses[down].w;
    if (p.y<height-3 && helper->weak_info_cuda[downdown] == STRONG) filter[index++] = plane_hypotheses[downdown].w;
    if (p.y<height-5 && helper->weak_info_cuda[downdown + width*2] == STRONG) filter[index++] = plane_hypotheses[downdown + width*2].w;
    if (p.x>0 && helper->weak_info_cuda[left] == STRONG) filter[index++] = plane_hypotheses[left].w;
    if (p.x>2 && helper->weak_info_cuda[leftleft] == STRONG) filter[index++] = plane_hypotheses[leftleft].w;
    if (p.x>4 && helper->weak_info_cuda[leftleft - 2] == STRONG) filter[index++] = plane_hypotheses[leftleft - 2].w;
    if (p.x<width-1 && helper->weak_info_cuda[right] == STRONG) filter[index++] = plane_hypotheses[right].w;
    if (p.x<width-3 && helper->weak_info_cuda[rightright] == STRONG) filter[index++] = plane_hypotheses[rightright].w;
    if (p.x<width-5 && helper->weak_info_cuda[rightright + 2] == STRONG) filter[index++] = plane_hypotheses[rightright + 2].w;
    if (p.y>0 && p.x<width-2 && helper->weak_info_cuda[up+2] == STRONG) filter[index++] = plane_hypotheses[up+2].w;
    if (p.y<height-1 && p.x<width-2 && helper->weak_info_cuda[down+2] == STRONG) filter[index++] = plane_hypotheses[down+2].w;
    if (p.y>0 && p.x>1 && helper->weak_info_cuda[up-2] == STRONG) filter[index++] = plane_hypotheses[up-2].w;
    if (p.y<height-1 && p.x>1 && helper->weak_info_cuda[down-2] == STRONG) filter[index++] = plane_hypotheses[down-2].w;
    if (p.x>0 && p.y>2 && helper->weak_info_cuda[left - width*2] == STRONG) filter[index++] = plane_hypotheses[left - width*2].w;
    if (p.x<width-1 && p.y>2 && helper->weak_info_cuda[right - width*2] == STRONG) filter[index++] = plane_hypotheses[right - width*2].w;
    if (p.x>0 && p.y<height-2 && helper->weak_info_cuda[left + width*2] == STRONG) filter[index++] = plane_hypotheses[left + width*2].w;
    if (p.x<width-1 && p.y<height-2 && helper->weak_info_cuda[right + width*2] == STRONG) filter[index++] = plane_hypotheses[right + width*2].w;

    sort_small(filter, index);
    int median_index = index / 2;
    if (index % 2 == 0)
        plane_hypotheses[center].w = (filter[median_index - 1] + filter[median_index]) / 2;
    else
        plane_hypotheses[center].w = filter[median_index];
}

// ============================================================================
// GLOBAL KERNELS (APD_ prefixed, use DataPassHelper for stream batching)
// ============================================================================

__global__ void APD_InitRandomStates(DataPassHelper *helper) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= helper->width || p.y >= helper->height) return;
    const int center = p.y * helper->width + p.x;
    helper->rand_states_cuda[center].counter = 0;
}

__global__ void APD_RandomInitialization(DataPassHelper *helper) {
    const int width = helper->width;
    const int height = helper->height;
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (p.x >= width || p.y >= height) return;
    const int center = p.y * width + p.x;
    if (apd_is_masked(helper, center)) return;
    const unsigned int pixel_key = (unsigned int)center;
    Camera *cameras = helper->cameras_cuda;
    float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    float *costs = helper->costs_cuda;
    RNGState *rand_states = helper->rand_states_cuda;
    PatchMatchParams *params = helper->params;

    if (params->state == FIRST_INIT) {
        plane_hypotheses[center] = GenerateRandomPlaneHypothesis(cameras[0], p, &rand_states[center], pixel_key, params->depth_min, params->depth_max);
        costs[center] = ComputeMultiViewInitialCostandSelectedViews(p, helper);
    } else {
        float4 plane_hypothesis = plane_hypotheses[center];
        plane_hypothesis = TransformNormal2RefCam(cameras[0], plane_hypothesis);
        float depth = plane_hypothesis.w;
        plane_hypothesis.w = GetDistance2Origin_APD(cameras[0], p, depth, plane_hypothesis);
        plane_hypotheses[center] = plane_hypothesis;
        costs[center] = ComputeMultiViewInitialCost(p, helper);
    }
}

// Dispatch helper for APD Strong propagation
__device__ __forceinline__ void DispatchPropagationStrong(const int2 p, const int iter, DataPassHelper *helper) {
    const int ni = helper->params->num_images;
    if (ni <= 5)       CheckerboardPropagationStrong<5>(p, iter, helper);
    else if (ni <= 10) CheckerboardPropagationStrong<10>(p, iter, helper);
    else if (ni <= 16) CheckerboardPropagationStrong<16>(p, iter, helper);
    else               CheckerboardPropagationStrong<32>(p, iter, helper);
}

// Dispatch helper for APD Weak propagation
__device__ __forceinline__ void DispatchPropagationWeak(const int2 p, const int iter, DataPassHelper *helper) {
    const int ni = helper->params->num_images;
    if (ni <= 5)       CheckerboardPropagationWeak<5>(p, iter, helper);
    else if (ni <= 10) CheckerboardPropagationWeak<10>(p, iter, helper);
    else if (ni <= 16) CheckerboardPropagationWeak<16>(p, iter, helper);
    else               CheckerboardPropagationWeak<32>(p, iter, helper);
}

__launch_bounds__(512, 2)
__global__ void APD_BlackPixelUpdateStrong(const int iter, DataPassHelper *helper) {
    // Cache helper, cameras, and params in shared memory (all threads must participate before any early exit)
    __shared__ DataPassHelper s_helper;
    __shared__ Camera s_cameras[32];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) s_helper = *helper;
    __syncthreads();
    if (tid < s_helper.params->num_images) s_cameras[tid] = s_helper.cameras_cuda[tid];
    if (tid == 0) s_helper.cameras_cuda = s_cameras;
    __syncthreads();

    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) p.y = p.y * 2;
    else p.y = p.y * 2 + 1;
    if (p.x >= s_helper.width || p.y >= s_helper.height) return;
    const int center = p.x + p.y * s_helper.width;
    if (apd_is_masked(&s_helper, center)) return;
    if (s_helper.weak_info_cuda[center] == WEAK) return;
    DispatchPropagationStrong(p, iter, &s_helper);
}

__launch_bounds__(512, 2)
__global__ void APD_RedPixelUpdateStrong(const int iter, DataPassHelper *helper) {
    __shared__ DataPassHelper s_helper;
    __shared__ Camera s_cameras[32];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) s_helper = *helper;
    __syncthreads();
    if (tid < s_helper.params->num_images) s_cameras[tid] = s_helper.cameras_cuda[tid];
    if (tid == 0) s_helper.cameras_cuda = s_cameras;
    __syncthreads();

    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) p.y = p.y * 2 + 1;
    else p.y = p.y * 2;
    if (p.x >= s_helper.width || p.y >= s_helper.height) return;
    const int center = p.x + p.y * s_helper.width;
    if (apd_is_masked(&s_helper, center)) return;
    if (s_helper.weak_info_cuda[center] == WEAK) return;
    DispatchPropagationStrong(p, iter, &s_helper);
}

__launch_bounds__(512, 2)
__global__ void APD_BlackPixelUpdateWeak(const int iter, DataPassHelper *helper) {
    __shared__ DataPassHelper s_helper;
    __shared__ Camera s_cameras[32];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) s_helper = *helper;
    __syncthreads();
    if (tid < s_helper.params->num_images) s_cameras[tid] = s_helper.cameras_cuda[tid];
    if (tid == 0) s_helper.cameras_cuda = s_cameras;
    __syncthreads();

    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) p.y = p.y * 2;
    else p.y = p.y * 2 + 1;
    if (p.x >= s_helper.width || p.y >= s_helper.height) return;
    const int center = p.x + p.y * s_helper.width;
    if (apd_is_masked(&s_helper, center)) return;
    if (s_helper.weak_info_cuda[center] == WEAK)
        DispatchPropagationWeak(p, iter, &s_helper);
}

__launch_bounds__(512, 2)
__global__ void APD_RedPixelUpdateWeak(const int iter, DataPassHelper *helper) {
    __shared__ DataPassHelper s_helper;
    __shared__ Camera s_cameras[32];
    const int tid = threadIdx.y * blockDim.x + threadIdx.x;
    if (tid == 0) s_helper = *helper;
    __syncthreads();
    if (tid < s_helper.params->num_images) s_cameras[tid] = s_helper.cameras_cuda[tid];
    if (tid == 0) s_helper.cameras_cuda = s_cameras;
    __syncthreads();

    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) p.y = p.y * 2 + 1;
    else p.y = p.y * 2;
    if (p.x >= s_helper.width || p.y >= s_helper.height) return;
    const int center = p.x + p.y * s_helper.width;
    if (apd_is_masked(&s_helper, center)) return;
    if (s_helper.weak_info_cuda[center] == WEAK)
        DispatchPropagationWeak(p, iter, &s_helper);
}

__global__ void APD_GetDepthandNormal(DataPassHelper *helper) {
    const int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (p.x >= width || p.y >= height) return;
    const int center = p.y * width + p.x;
    Camera *cameras = helper->cameras_cuda;
    float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    plane_hypotheses[center].w = ComputeDepthfromPlaneHypothesis_APD(cameras[0], plane_hypotheses[center], p);
    plane_hypotheses[center] = TransformNormal(cameras[0], plane_hypotheses[center]);
}

__global__ void APD_BlackPixelFilterStrong(DataPassHelper *helper) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) p.y = p.y * 2;
    else p.y = p.y * 2 + 1;
    if (p.x >= helper->width || p.y >= helper->height) return;
    if (helper->weak_info_cuda[p.x + p.y * helper->width] != WEAK)
        CheckerboardFilterStrong(p, helper);
}

__global__ void APD_RedPixelFilterStrong(DataPassHelper *helper) {
    int2 p = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (threadIdx.x % 2 == 0) p.y = p.y * 2 + 1;
    else p.y = p.y * 2;
    if (p.x >= helper->width || p.y >= helper->height) return;
    if (helper->weak_info_cuda[p.x + p.y * helper->width] != WEAK)
        CheckerboardFilterStrong(p, helper);
}

__global__ void APD_FindNearestStrongPoint(DataPassHelper *helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) return;
    const uchar *weak_info = helper->weak_info_cuda;
    short2 *weak_nearest_strong = helper->weak_nearest_strong;
    const int center = point.x + point.y * width;
    const CameraModel model = helper->cameras_cuda[0].model;
    weak_nearest_strong[center].x = -1;
    weak_nearest_strong[center].y = -1;
    if (weak_info[center] != WEAK) return;

    const int radius = 100;
    float min_dist = 255.0f;
    for (int x = -radius; x <= radius; ++x) {
        for (int y = -radius; y <= radius; ++y) {
            int nx = point.x + x;
            int ny = point.y + y;
            if (model == SPHERE) {
                nx = ((nx % width) + width) % width;
            }
            if (nx < 0 || ny < 0 || nx >= width || ny >= height) continue;
            const int nc = nx + ny * width;
            if (weak_info[nc] == STRONG) {
                float dist = sqrtf((float)(x * x + y * y));
                if (dist < min_dist) {
                    min_dist = dist;
                    weak_nearest_strong[center].x = nx;
                    weak_nearest_strong[center].y = ny;
                }
            }
        }
    }
}

__global__ void APD_GenNeighbours(DataPassHelper *helper) {
    int width = helper->width;
    int height = helper->height;
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    if (point.x >= width || point.y >= height) return;
    const unsigned center = point.x + point.y * width;
    const uchar *weak_info = helper->weak_info_cuda;
    if (weak_info[center] != WEAK) return;

    const int min_margin = 6;
    const float depth_diff = helper->params->depth_max - helper->params->depth_min;
    const int *neighbours_map = helper->neighbours_map_cuda;
    const PatchMatchParams *params = helper->params;
    const short2 *weak_nearest_strong = helper->weak_nearest_strong;
    const Camera camera = helper->cameras_cuda[0];
    const unsigned offset = neighbours_map[center] * NEIGHBOUR_NUM;
    const float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    RNGState *rand_state = &(helper->rand_states_cuda[center]);
    const unsigned int pixel_key = (unsigned int)center;
    short2 *neighbours = &(helper->neighbours_cuda[offset]);
    uchar *weak_reliable = &(helper->weak_reliable_cuda[center]);

    for (int i = 0; i < NEIGHBOUR_NUM; ++i) {
        neighbours[i].x = -1;
        neighbours[i].y = -1;
    }
    neighbours[0] = make_short2(point.x, point.y);

    short2 strong_points[8 * 4];
    bool dir_valid[8 * 4];
    for (int i = 0; i < 32; ++i) {
        strong_points[i] = make_short2(-1, -1);
        dir_valid[i] = false;
    }
    int origin_direction_index = -1;
    int strong_point_size = 0;
    const int rotate_time = params->rotate_time;
    const float angle = 45.0f / rotate_time;
    const float cos_ang = cosf(angle * CUDART_PI_F / 180.0f);
    const float sin_ang = sinf(angle * CUDART_PI_F / 180.0f);
    const float thresh = cosf((angle / 2.0f) * CUDART_PI_F / 180.0f);
    const int shift_range = max((int)(tanf((angle / 2.0f) * CUDART_PI_F / 180.0f) * 20), 1);
    const float ransac_threshold = params->ransac_threshold;

    for (int origin_direction_x = -1; origin_direction_x <= 1; ++origin_direction_x) {
        for (int origin_direction_y = -1; origin_direction_y <= 1; ++origin_direction_y) {
            if (origin_direction_x == 0 && origin_direction_y == 0) continue;
            float2 origin_direction = make_float2((float)origin_direction_x, (float)origin_direction_y);
            NormalizeVec2(&origin_direction);
            origin_direction_index++;
            for (int rotate_iter = 0; rotate_iter < rotate_time; ++rotate_iter) {
                int dir_index = origin_direction_index * 4 + rotate_iter;
                for (int radius = 2; radius <= APD_MAX_SEARCH_RADIUS; radius = min(radius * 2, radius + 25)) {
                    float2 test_pt = make_float2(point.x + origin_direction.x * radius, point.y + origin_direction.y * radius);
                    // For spherical, x wraps; for pinhole, check bounds
                    if (camera.model != SPHERE) {
                        if (test_pt.x < 0 || test_pt.x >= width) break;
                    }
                    if (test_pt.y < 0 || test_pt.y >= height) break;

                    for (int radius_iter = 0; radius_iter < 4; ++radius_iter) {
                        int rand_x_shift = (apd_rng_uint(rand_state, pixel_key) % 2 == 0 ? 1 : -1) * (int)(apd_rng_uint(rand_state, pixel_key) % shift_range);
                        int rand_y_shift = (apd_rng_uint(rand_state, pixel_key) % 2 == 0 ? 1 : -1) * (int)(apd_rng_uint(rand_state, pixel_key) % shift_range);
                        float2 direction = make_float2(origin_direction.x * 20 + rand_x_shift, origin_direction.y * 20 + rand_y_shift);
                        NormalizeVec2(&direction);
                        int nx = (int)(point.x + direction.x * radius);
                        int ny = (int)(point.y + direction.y * radius);
                        if (camera.model == SPHERE) {
                            nx = ((nx % width) + width) % width;
                        }
                        if (nx < min_margin || ny < min_margin || nx >= width - min_margin || ny >= height - min_margin) continue;
                        short2 neighbour_pt = make_short2(nx, ny);
                        int neighbour_pt_center = neighbour_pt.x + neighbour_pt.y * width;
                        if (weak_info[neighbour_pt_center] != STRONG) {
                            neighbour_pt = weak_nearest_strong[neighbour_pt_center];
                            if (neighbour_pt.x == -1 || neighbour_pt.y == -1) continue;
                            neighbour_pt_center = neighbour_pt.x + neighbour_pt.y * width;
                        }
                        float2 test_direction = make_float2(neighbour_pt.x - point.x, neighbour_pt.y - point.y);
                        NormalizeVec2(&test_direction);
                        float cos_a = Vec2DotVec2(test_direction, origin_direction);
                        if (cos_a > thresh) {
                            strong_points[dir_index] = neighbour_pt;
                            dir_valid[dir_index] = true;
                            strong_point_size++;
                            break;
                        }
                    }
                    if (dir_valid[dir_index]) break;
                }
                // rotate
                float2 rotated;
                rotated.x = origin_direction.x * cos_ang - origin_direction.y * sin_ang;
                rotated.y = origin_direction.x * sin_ang + origin_direction.y * cos_ang;
                NormalizeVec2(&rotated);
                origin_direction = rotated;
            }
        }
    }

    if (strong_point_size <= 3) {
        *weak_reliable = 0;
        return;
    }

    float4 best_plane;
    int use_a_index = -1, use_b_index = -1, use_c_index = -1;
    bool has_valid_plane = false;
    short2 strong_points_valid[8 * 4];
    float3 strong_points_valid_3d[8 * 4];
    int valid_count = 0;
    float X[3];
    Get3DPoint_APD(camera, point, plane_hypotheses[center].w, X);
    float3 center_point_world = make_float3(X[0], X[1], X[2]);
    for (int i = 0; i < 32; ++i) {
        strong_points_valid[i] = make_short2(-1, -1);
        if (dir_valid[i]) {
            const auto &strong_point = strong_points[i];
            int strong_point_center = strong_point.x + strong_point.y * width;
            strong_points_valid[valid_count] = strong_points[i];
            Get3DPoint_APD(camera, strong_point, plane_hypotheses[strong_point_center].w, X);
            strong_points_valid_3d[valid_count] = make_float3(X[0], X[1], X[2]);
            valid_count++;
        }
    }

    {   // RANSAC
        int iteration = 50;
        float min_cost = FLT_MAX;
        int max_count = 3;
        while (iteration--) {
            int a_index = apd_rng_uint(rand_state, pixel_key) % valid_count;
            int b_index = apd_rng_uint(rand_state, pixel_key) % valid_count;
            int c_index = apd_rng_uint(rand_state, pixel_key) % valid_count;
            if (a_index == b_index || b_index == c_index || a_index == c_index) continue;
            if (!PointinTriangle(strong_points_valid[a_index], strong_points_valid[b_index], strong_points_valid[c_index], point)) continue;
            const float3 &A = strong_points_valid_3d[a_index];
            const float3 &B = strong_points_valid_3d[b_index];
            const float3 &C = strong_points_valid_3d[c_index];
            float3 A_C = make_float3(A.x-C.x, A.y-C.y, A.z-C.z);
            float3 B_C = make_float3(B.x-C.x, B.y-C.y, B.z-C.z);
            float4 cross_vec;
            cross_vec.x = A_C.y * B_C.z - B_C.y * A_C.z;
            cross_vec.y = -(A_C.x * B_C.z - B_C.x * A_C.z);
            cross_vec.z = A_C.x * B_C.y - B_C.x * A_C.y;
            if ((cross_vec.x == 0 && cross_vec.y == 0 && cross_vec.z == 0) || isnan(cross_vec.x) || isnan(cross_vec.y) || isnan(cross_vec.z)) continue;
            NormalizeVec3(&cross_vec);
            cross_vec.w = -(cross_vec.x * A.x + cross_vec.y * A.y + cross_vec.z * A.z);
            int temp_count = 0;
            float strong_dist = 0.0f;
            for (int si = 0; si < valid_count; ++si) {
                const float3 &tp = strong_points_valid_3d[si];
                float distance = fabsf(cross_vec.x * tp.x + cross_vec.y * tp.y + cross_vec.z * tp.z + cross_vec.w);
                if (distance / depth_diff < ransac_threshold) {
                    temp_count++;
                    strong_dist += distance;
                }
            }
            if (temp_count < 6) continue;
            if (temp_count > max_count) {
                max_count = temp_count;
                const float center_distance = fabsf(cross_vec.x * center_point_world.x + cross_vec.y * center_point_world.y + cross_vec.z * center_point_world.z + cross_vec.w);
                min_cost = center_distance;
                best_plane = cross_vec;
                has_valid_plane = true;
                use_a_index = a_index; use_b_index = b_index; use_c_index = c_index;
            } else if (temp_count == max_count) {
                const float center_distance = fabsf(cross_vec.x * center_point_world.x + cross_vec.y * center_point_world.y + cross_vec.z * center_point_world.z + cross_vec.w);
                if (center_distance < min_cost) {
                    max_count = temp_count;
                    min_cost = center_distance;
                    best_plane = cross_vec;
                    use_a_index = a_index; use_b_index = b_index; use_c_index = c_index;
                }
            }
        }
    }

    if (!has_valid_plane) {
        *weak_reliable = 0;
        return;
    }

    float weight[32];
    for (int i = 0; i < valid_count; ++i) {
        const float3 &tp = strong_points_valid_3d[i];
        float distance = fabsf(best_plane.x * tp.x + best_plane.y * tp.y + best_plane.z * tp.z + best_plane.w);
        if (distance / depth_diff >= ransac_threshold) {
            strong_points_valid[i] = make_short2(-1, -1);
            weight[i] = FLT_MAX;
            continue;
        }
        if (i == use_a_index || i == use_b_index || i == use_c_index) distance -= 1;
        weight[i] = distance;
    }
    sort_small_weighted(strong_points_valid, weight, valid_count);
    for (int i = 1; i < NEIGHBOUR_NUM; ++i) {
        neighbours[i] = strong_points_valid[i - 1];
    }
    *weak_reliable = 1;
}

__global__ void APD_NeigbourUpdate(DataPassHelper *helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) return;
    const int center = point.x + point.y * width;
    if (helper->weak_info_cuda[center] != WEAK) return;
    if (helper->weak_reliable_cuda[center] != 1)
        helper->weak_info_cuda[center] = UNKNOWN;
}

__global__ void APD_DepthToWeak(DataPassHelper *helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) return;
    const int min_margin = 6;
    const int center = point.x + point.y * width;
    if (apd_is_masked(helper, center)) { helper->weak_info_cuda[center] = UNKNOWN; return; }

    if (point.x < min_margin || point.y < min_margin || point.x >= width - min_margin || point.y >= height - min_margin) {
        helper->weak_info_cuda[center] = UNKNOWN;
        return;
    }

    const Camera *cameras = helper->cameras_cuda;
    const unsigned *selected_views = helper->selected_views_cuda;
    const int num_images = helper->params->num_images;
    const uchar *view_weight = &(helper->view_weight_cuda[helper->view_weight_stride * center]);
    float4 origin_plane_hypothesis = helper->plane_hypotheses_cuda[center];
    origin_plane_hypothesis = TransformNormal2RefCam(cameras[0], origin_plane_hypothesis);
    float origin_depth = origin_plane_hypothesis.w;
    if (origin_depth == 0) { helper->weak_info_cuda[center] = UNKNOWN; return; }

    float cost_now = 0.0f;
    float base_line = 0;
    int valid_neighbour = 0;
    float weight_normal = 0.0f;
    for (int src_index = 1; src_index < num_images; ++src_index) {
        int view_index = src_index - 1;
        if (isSet(selected_views[center], view_index)) {
            float4 temp_ph = origin_plane_hypothesis;
            temp_ph.w = GetDistance2Origin_APD(cameras[0], point, origin_depth, temp_ph);
            float temp_cost = ComputeBilateralNCCOld(point, src_index, temp_ph, helper);
            if (helper->params->geom_consistency)
                temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_ph, helper);
            cost_now += (temp_cost * view_weight[view_index]);
            weight_normal += view_weight[view_index];
            float c_dist[3];
            c_dist[0] = cameras[0].c[0] - cameras[src_index].c[0];
            c_dist[1] = cameras[0].c[1] - cameras[src_index].c[1];
            c_dist[2] = cameras[0].c[2] - cameras[src_index].c[2];
            base_line += sqrtf(c_dist[0]*c_dist[0] + c_dist[1]*c_dist[1] + c_dist[2]*c_dist[2]);
            valid_neighbour++;
        }
    }
    if (valid_neighbour == 0) { helper->weak_info_cuda[center] = UNKNOWN; return; }

    cost_now /= weight_normal;
    base_line /= valid_neighbour;

    float disp = DepthToDisparity_APD(cameras[0], base_line, origin_depth);
    const int radius = 30;
    const int p_costs_size = 2 * radius + 1;
    float p_costs[p_costs_size];

    for (int p_disp = -radius; p_disp <= radius; ++p_disp) {
        float p_depth = DisparityToDepth_APD(cameras[0], base_line, disp + (float)p_disp);
        if (p_depth < helper->params->depth_min || p_depth > helper->params->depth_max) {
            p_costs[p_disp + radius] = 2.0f;
            continue;
        }
        float4 temp_ph = origin_plane_hypothesis;
        temp_ph.w = GetDistance2Origin_APD(cameras[0], point, p_depth, temp_ph);
        float p_cost = 0.0f;
        for (int src_index = 1; src_index < num_images; ++src_index) {
            int view_index = src_index - 1;
            if (isSet(selected_views[center], view_index)) {
                float temp_cost = ComputeBilateralNCCOld(point, src_index, temp_ph, helper);
                if (helper->params->geom_consistency)
                    temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_ph, helper);
                p_cost += (temp_cost * view_weight[view_index]);
            }
        }
        p_cost /= weight_normal;
        p_costs[p_disp + radius] = min(2.0f, p_cost);
    }

    // find peaks
    bool is_peak[p_costs_size];
    for (int i = 0; i < p_costs_size; ++i) is_peak[i] = false;
    int peak_count = 0;
    int min_peak = 0;
    float min_cost = 2.0f;
    for (int i = 2; i < p_costs_size - 2; ++i) {
        if (p_costs[i-1] > p_costs[i] && p_costs[i+1] > p_costs[i]) {
            is_peak[i] = true;
            peak_count++;
            if (p_costs[i] < min_cost) { min_peak = i; min_cost = p_costs[i]; }
        }
    }

    if (abs(min_peak - radius) > helper->params->weak_peak_radius || p_costs[min_peak] > 0.5f) {
        helper->weak_info_cuda[center] = WEAK;
        return;
    }
    if (peak_count == 1) {
        helper->weak_info_cuda[center] = (p_costs[min_peak] <= 0.15f) ? STRONG : WEAK;
        return;
    }

    float var = 0.0f;
    for (int i = 2; i < p_costs_size - 2; ++i) {
        if (is_peak[i] && i != min_peak) {
            float dist = p_costs[i] - min_cost;
            var += dist * dist;
        }
    }
    var = sqrtf(var) / (peak_count - 1);
    helper->weak_info_cuda[center] = (var > 0.2f) ? STRONG : WEAK;
}

__global__ void APD_LocalRefine(DataPassHelper *helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) return;
    const int center = point.x + point.y * width;
    if (apd_is_masked(helper, center)) return;

    const Camera *cameras = helper->cameras_cuda;
    const unsigned *selected_views = helper->selected_views_cuda;
    const int num_images = helper->params->num_images;
    const uchar *view_weight = &(helper->view_weight_cuda[helper->view_weight_stride * center]);
    float4 origin_plane_hypothesis = helper->plane_hypotheses_cuda[center];
    origin_plane_hypothesis = TransformNormal2RefCam(cameras[0], origin_plane_hypothesis);
    float origin_depth = origin_plane_hypothesis.w;
    if (origin_depth == 0) return;

    float cost_now = 0.0f;
    float base_line = 0;
    int valid_neighbour = 0;
    float weight_normal = 0.0f;
    for (int src_index = 1; src_index < num_images; ++src_index) {
        int view_index = src_index - 1;
        if (isSet(selected_views[center], view_index)) {
            float4 temp_ph = origin_plane_hypothesis;
            temp_ph.w = GetDistance2Origin_APD(cameras[0], point, origin_depth, temp_ph);
            float temp_cost = ComputeBilateralNCCOld(point, src_index, temp_ph, helper);
            if (helper->params->geom_consistency)
                temp_cost += helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_ph, helper);
            cost_now += (temp_cost * view_weight[view_index]);
            weight_normal += view_weight[view_index];
            float c_dist[3];
            c_dist[0] = cameras[0].c[0] - cameras[src_index].c[0];
            c_dist[1] = cameras[0].c[1] - cameras[src_index].c[1];
            c_dist[2] = cameras[0].c[2] - cameras[src_index].c[2];
            base_line += sqrtf(c_dist[0]*c_dist[0] + c_dist[1]*c_dist[1] + c_dist[2]*c_dist[2]);
            valid_neighbour++;
        }
    }
    if (weight_normal == 0 || valid_neighbour == 0) return;
    cost_now /= weight_normal;
    base_line /= valid_neighbour;

    float disp = DepthToDisparity_APD(cameras[0], base_line, origin_depth);
    const int radius = 5;
    float min_cost = 2.0f;
    float best_depth = origin_depth;

    for (int p_disp = -radius; p_disp <= radius; ++p_disp) {
        float p_depth = DisparityToDepth_APD(cameras[0], base_line, disp + (float)p_disp);
        if (p_depth < helper->params->depth_min || p_depth > helper->params->depth_max) continue;
        float4 temp_ph = origin_plane_hypothesis;
        temp_ph.w = GetDistance2Origin_APD(cameras[0], point, p_depth, temp_ph);
        float temp_cost = 0.0f;
        for (int src_index = 1; src_index < num_images; ++src_index) {
            int view_index = src_index - 1;
            if (isSet(selected_views[center], view_index)) {
                temp_cost += (ComputeBilateralNCCOld(point, src_index, temp_ph, helper) * view_weight[view_index]);
                if (helper->params->geom_consistency)
                    temp_cost += (helper->params->geom_factor * ComputeGeomConsistencyCost(point, src_index, temp_ph, helper) * view_weight[view_index]);
            }
        }
        temp_cost /= weight_normal;
        if (temp_cost < min_cost) { min_cost = temp_cost; best_depth = p_depth; }
    }
    if (cost_now - min_cost > 0.1f)
        helper->plane_hypotheses_cuda[center].w = best_depth;
}

__global__ void APD_RANSACToGetFitPlane(DataPassHelper *helper) {
    const int2 point = make_int2(blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y);
    const int width = helper->width;
    const int height = helper->height;
    if (point.x >= width || point.y >= height) return;
    const uchar *weak_info = helper->weak_info_cuda;
    const int center = point.x + point.y * width;
    float4 *plane_hypotheses = helper->plane_hypotheses_cuda;
    float4 *fit_plane_hypothese = helper->fit_plane_hypotheses_cuda;
    if (weak_info[center] != WEAK) {
        fit_plane_hypothese[center] = plane_hypotheses[center];
        return;
    }
    RNGState *rand_state = &(helper->rand_states_cuda[center]);
    const unsigned int pixel_key = (unsigned int)center;
    const auto &camera = helper->cameras_cuda[0];

    short2 strong_points[NEIGHBOUR_NUM - 1];
    float3 strong_points_3d[NEIGHBOUR_NUM - 1];
    int strong_count = 0;
    float X[3];
    for (int i = 1; i < NEIGHBOUR_NUM; ++i) {
        short2 temp_point = GetNeighbourPoint(point, i, helper);
        if (temp_point.x == -1 || temp_point.y == -1) continue;
        strong_points[strong_count].x = temp_point.x;
        strong_points[strong_count].y = temp_point.y;
        const int temp_center = temp_point.x + temp_point.y * width;
        float depth = ComputeDepthfromPlaneHypothesis_APD(camera, plane_hypotheses[temp_center], make_int2(temp_point.x, temp_point.y));
        Get3DPoint_APD(camera, strong_points[strong_count], depth, X);
        strong_points_3d[strong_count] = make_float3(X[0], X[1], X[2]);
        strong_count++;
    }
    if (strong_count < 3) {
        fit_plane_hypothese[center] = plane_hypotheses[center];
        return;
    }

    int iteration = 50;
    float min_cost = FLT_MAX;
    float4 best_plane;
    bool has_best_plane = false;
    while (iteration--) {
        int a_index = apd_rng_uint(rand_state, pixel_key) % strong_count;
        int b_index = apd_rng_uint(rand_state, pixel_key) % strong_count;
        int c_index = apd_rng_uint(rand_state, pixel_key) % strong_count;
        if (a_index == b_index || b_index == c_index || a_index == c_index) continue;
        if (!PointinTriangle(strong_points[a_index], strong_points[b_index], strong_points[c_index], point)) continue;
        const float3 &A = strong_points_3d[a_index];
        const float3 &B = strong_points_3d[b_index];
        const float3 &C = strong_points_3d[c_index];
        float3 A_C = make_float3(A.x-C.x, A.y-C.y, A.z-C.z);
        float3 B_C = make_float3(B.x-C.x, B.y-C.y, B.z-C.z);
        float4 cross_vec;
        cross_vec.x = A_C.y * B_C.z - B_C.y * A_C.z;
        cross_vec.y = -(A_C.x * B_C.z - B_C.x * A_C.z);
        cross_vec.z = A_C.x * B_C.y - B_C.x * A_C.y;
        if ((cross_vec.x == 0 && cross_vec.y == 0 && cross_vec.z == 0) || isnan(cross_vec.x) || isnan(cross_vec.y) || isnan(cross_vec.z)) continue;
        NormalizeVec3(&cross_vec);
        cross_vec.w = -(cross_vec.x * A.x + cross_vec.y * A.y + cross_vec.z * A.z);
        float temp_cost = 0.0f;
        for (int si = 0; si < strong_count; ++si) {
            if (si == a_index || si == b_index || si == c_index) continue;
            const float3 &tp = strong_points_3d[si];
            temp_cost += fabsf(cross_vec.x * tp.x + cross_vec.y * tp.y + cross_vec.z * tp.z + cross_vec.w);
        }
        if (temp_cost < min_cost) {
            min_cost = temp_cost;
            best_plane = cross_vec;
            has_best_plane = true;
        }
        if (min_cost == 0) break;
    }

    if (has_best_plane) {
        float depth = ComputeDepthfromPlaneHypothesis_APD(camera, plane_hypotheses[center], point);
        float4 view_direction = GetViewDirection_APD(camera, point, depth);
        float dot_product = best_plane.x * view_direction.x + best_plane.y * view_direction.y + best_plane.z * view_direction.z;
        if (dot_product > 0) {
            best_plane.x = -best_plane.x;
            best_plane.y = -best_plane.y;
            best_plane.z = -best_plane.z;
            best_plane.w = -best_plane.w;
        }
        fit_plane_hypothese[center] = best_plane;
    } else {
        fit_plane_hypothese[center] = make_float4(0, 0, 0, 0);
    }
}

// ============================================================================
// HOST-CALLABLE LAUNCHER
// ============================================================================
void RunAPDPatchMatch_GPU(DataPassHelper *helper_cuda, int width, int height,
                          const PatchMatchParams &params, cudaStream_t stream) {
    const int BLOCK_W = 32;
    const int BLOCK_H = (BLOCK_W / 2);

    dim3 grid_full((width + 16 - 1) / 16, (height + 16 - 1) / 16, 1);
    dim3 block_full(16, 16, 1);

    dim3 grid_half((width + BLOCK_W - 1) / BLOCK_W, ((height / 2) + BLOCK_H - 1) / BLOCK_H, 1);
    dim3 block_half(BLOCK_W, BLOCK_H, 1);

    APD_InitRandomStates<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    APD_FindNearestStrongPoint<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    APD_GenNeighbours<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    APD_NeigbourUpdate<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    APD_RandomInitialization<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    for (int i = 0; i < params.max_iterations; ++i) {
        APD_BlackPixelUpdateStrong<<<grid_half, block_half, 0, stream>>>(i, helper_cuda);
        APD_RedPixelUpdateStrong<<<grid_half, block_half, 0, stream>>>(i, helper_cuda);

        APD_RANSACToGetFitPlane<<<grid_full, block_full, 0, stream>>>(helper_cuda);

        APD_BlackPixelUpdateWeak<<<grid_half, block_half, 0, stream>>>(i, helper_cuda);
        APD_RedPixelUpdateWeak<<<grid_half, block_half, 0, stream>>>(i, helper_cuda);
    }

    APD_GetDepthandNormal<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    APD_BlackPixelFilterStrong<<<grid_half, block_half, 0, stream>>>(helper_cuda);
    APD_RedPixelFilterStrong<<<grid_half, block_half, 0, stream>>>(helper_cuda);

    APD_DepthToWeak<<<grid_full, block_full, 0, stream>>>(helper_cuda);

    APD_LocalRefine<<<grid_full, block_full, 0, stream>>>(helper_cuda);
}
