#include "ACMMP.h"
#include <math_constants.h>  // for CUDART_PI_F

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

__device__ void ComputeHomography(const Camera ref_camera, const Camera src_camera, const float4 plane_hypothesis, float *H)
{
    float ref_C[3];
    float src_C[3];
    ref_C[0] = -(ref_camera.R[0] * ref_camera.t[0] + ref_camera.R[3] * ref_camera.t[1] + ref_camera.R[6] * ref_camera.t[2]);
    ref_C[1] = -(ref_camera.R[1] * ref_camera.t[0] + ref_camera.R[4] * ref_camera.t[1] + ref_camera.R[7] * ref_camera.t[2]);
    ref_C[2] = -(ref_camera.R[2] * ref_camera.t[0] + ref_camera.R[5] * ref_camera.t[1] + ref_camera.R[8] * ref_camera.t[2]);
    src_C[0] = -(src_camera.R[0] * src_camera.t[0] + src_camera.R[3] * src_camera.t[1] + src_camera.R[6] * src_camera.t[2]);
    src_C[1] = -(src_camera.R[1] * src_camera.t[0] + src_camera.R[4] * src_camera.t[1] + src_camera.R[7] * src_camera.t[2]);
    src_C[2] = -(src_camera.R[2] * src_camera.t[0] + src_camera.R[5] * src_camera.t[1] + src_camera.R[8] * src_camera.t[2]);

    float R_relative[9];
    float C_relative[3];
    float t_relative[3];
    R_relative[0] = src_camera.R[0] * ref_camera.R[0] + src_camera.R[1] * ref_camera.R[1] + src_camera.R[2] *ref_camera.R[2];
    R_relative[1] = src_camera.R[0] * ref_camera.R[3] + src_camera.R[1] * ref_camera.R[4] + src_camera.R[2] *ref_camera.R[5];
    R_relative[2] = src_camera.R[0] * ref_camera.R[6] + src_camera.R[1] * ref_camera.R[7] + src_camera.R[2] *ref_camera.R[8];
    R_relative[3] = src_camera.R[3] * ref_camera.R[0] + src_camera.R[4] * ref_camera.R[1] + src_camera.R[5] *ref_camera.R[2];
    R_relative[4] = src_camera.R[3] * ref_camera.R[3] + src_camera.R[4] * ref_camera.R[4] + src_camera.R[5] *ref_camera.R[5];
    R_relative[5] = src_camera.R[3] * ref_camera.R[6] + src_camera.R[4] * ref_camera.R[7] + src_camera.R[5] *ref_camera.R[8];
    R_relative[6] = src_camera.R[6] * ref_camera.R[0] + src_camera.R[7] * ref_camera.R[1] + src_camera.R[8] *ref_camera.R[2];
    R_relative[7] = src_camera.R[6] * ref_camera.R[3] + src_camera.R[7] * ref_camera.R[4] + src_camera.R[8] *ref_camera.R[5];
    R_relative[8] = src_camera.R[6] * ref_camera.R[6] + src_camera.R[7] * ref_camera.R[7] + src_camera.R[8] *ref_camera.R[8];
    C_relative[0] = (ref_C[0] - src_C[0]);
    C_relative[1] = (ref_C[1] - src_C[1]);
    C_relative[2] = (ref_C[2] - src_C[2]);
    t_relative[0] = src_camera.R[0] * C_relative[0] + src_camera.R[1] * C_relative[1] + src_camera.R[2] * C_relative[2];
    t_relative[1] = src_camera.R[3] * C_relative[0] + src_camera.R[4] * C_relative[1] + src_camera.R[5] * C_relative[2];
    t_relative[2] = src_camera.R[6] * C_relative[0] + src_camera.R[7] * C_relative[1] + src_camera.R[8] * C_relative[2];

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

__device__ float2 ComputeCorrespondingPoint(const float *H, const int2 p)
{
    float3 pt;
    pt.x = H[0] * p.x + H[1] * p.y + H[2];
    pt.y = H[3] * p.x + H[4] * p.y + H[5];
    pt.z = H[6] * p.x + H[7] * p.y + H[8];
    return make_float2(pt.x / pt.z, pt.y / pt.z);
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

    // 1) Depth of the ref center under the hypothesis, world point
    float depth_ref = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, p);
    float3 Pw_center = Get3DPointonWorld_cu(p.x, p.y, depth_ref, ref_camera);

    // 2) Project the center into the source; validate (SPHERE uses wrap/clamp)
    float2 pt_center; float dummy_depth;
    ProjectonCamera_cu(Pw_center, src_camera, pt_center, dummy_depth);

    if (src_camera.model == SPHERE) {
        pt_center.x = pt_center.x - floorf(pt_center.x / (float)src_camera.width) * (float)src_camera.width;   // wrap lon
        pt_center.y = fminf(fmaxf(pt_center.y, 0.0f), (float)src_camera.height - 1.0f);                         // clamp lat
    } else {
        if (pt_center.x < 0.0f || pt_center.x >= src_camera.width ||
            pt_center.y < 0.0f || pt_center.y >= src_camera.height) {
            return cost_max;
        }
    }

    // 3) Angular pixel scaling for SPHERE (use local small-angle metric)
    float scale_x = 1.0f, scale_y = 1.0f, sigma_spatial_eff = params.sigma_spatial;
    if (ref_camera.model == SPHERE) {
        const float lat_c = -((float)p.y - ref_camera.params[2]) / (float)ref_camera.height * CUDART_PI_F;
        scale_x = (2.0f * CUDART_PI_F / (float)ref_camera.width) * cosf(lat_c);  // dlon * cos(lat)
        scale_y = (CUDART_PI_F / (float)ref_camera.height);                       // dlat
        sigma_spatial_eff = params.sigma_spatial * (CUDART_PI_F / (float)ref_camera.height); // px → rad
    }

    // 4) Bilateral-weighted NCC accumulation
    const float ref_center_pix = tex2D<float>(ref_image, p.x + 0.5f, p.y + 0.5f);
    float sum_ref = 0.0f, sum_ref_ref = 0.0f;
    float sum_src = 0.0f, sum_src_src = 0.0f;
    float sum_ref_src = 0.0f, sum_bw = 0.0f;

    for (int i = -radius; i <= radius; i += params.radius_increment) {
        for (int j = -radius; j <= radius; j += params.radius_increment) {
            const int2 ref_pt = make_int2(p.x + i, p.y + j);

            // sample reference (assumes texture addressing handles edges as configured)
            const float ref_pix = tex2D<float>(ref_image, ref_pt.x + 0.5f, ref_pt.y + 0.5f);

            // compute 3D point for this ref offset under the same plane
            const float depth_n = ComputeDepthfromPlaneHypothesis(ref_camera, plane_hypothesis, ref_pt);
            float3 Pw_n = Get3DPointonWorld_cu(ref_pt.x, ref_pt.y, depth_n, ref_camera);

            // project into source
            float2 src_pt; float src_d;
            ProjectonCamera_cu(Pw_n, src_camera, src_pt, src_d);

            if (src_camera.model == SPHERE) {
                // wrap longitude, clamp latitude
                src_pt.x = src_pt.x - floorf(src_pt.x / (float)src_camera.width) * (float)src_camera.width;
                src_pt.y = fminf(fmaxf(src_pt.y, 0.0f), (float)src_camera.height - 1.0f);
            } else {
                if (src_pt.x < 0.0f || src_pt.x >= src_camera.width ||
                    src_pt.y < 0.0f || src_pt.y >= src_camera.height) {
                    continue;
                }
            }

            const float src_pix = tex2D<float>(src_image, src_pt.x + 0.5f, src_pt.y + 0.5f);

            // bilateral weight: angular distances for SPHERE, pixel distances otherwise
            const float dx = (ref_camera.model == SPHERE) ? (i * scale_x) : (float)i;
            const float dy = (ref_camera.model == SPHERE) ? (j * scale_y) : (float)j;

            const float w = ComputeBilateralWeight(dx, dy,
                                                   ref_pix,
                                                   ref_center_pix,
                                                   (ref_camera.model == SPHERE) ? sigma_spatial_eff : params.sigma_spatial,
                                                   params.sigma_color);

            sum_bw      += w;
            sum_ref     += w * ref_pix;
            sum_ref_ref += w * ref_pix * ref_pix;
            sum_src     += w * src_pix;
            sum_src_src += w * src_pix * src_pix;
            sum_ref_src += w * ref_pix * src_pix;
        }
    }

    if (sum_bw < 1e-6f) return cost_max;

    // normalize by total weight
    const float inv_bw = 1.0f / sum_bw;
    const float m_ref = sum_ref * inv_bw;
    const float m_src = sum_src * inv_bw;
    const float e_ref_ref = sum_ref_ref * inv_bw;
    const float e_src_src = sum_src_src * inv_bw;
    const float e_ref_src = sum_ref_src * inv_bw;

    const float var_ref = e_ref_ref - m_ref * m_ref;
    const float var_src = e_src_src - m_src * m_src;
    const float kMinVar = 1e-5f;
    if (var_ref < kMinVar || var_src < kMinVar) return cost_max;

    const float covar = e_ref_src - m_ref * m_src;
    float ncc_cost = 1.0f - covar / sqrtf(var_ref * var_src);
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
#include <queue>


// CUDA error checking macro
#ifndef CUDA_SAFE_CALL
#define CUDA_SAFE_CALL(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << " - " << cudaGetErrorString(err) << std::endl; \
        exit(EXIT_FAILURE); \
    } \
} while(0)
#endif

// Helper function to check GPU memory
bool checkGPUMemory(size_t required_bytes) {
    size_t free_mem, total_mem;
    cudaError_t mem_error = cudaMemGetInfo(&free_mem, &total_mem);
    if (mem_error != cudaSuccess) {
        std::cerr << "Warning: Could not query GPU memory" << std::endl;
        return true; // Assume it's okay
    }
    
    std::cout << "GPU Memory: " << free_mem / (1024*1024) << " MB free / " 
              << total_mem / (1024*1024) << " MB total" << std::endl;
    
    if (required_bytes > free_mem) {
        std::cerr << "Error: Insufficient GPU memory. Required: " << required_bytes / (1024*1024) 
                  << " MB, Available: " << free_mem / (1024*1024) << " MB" << std::endl;
        return false;
    }
    return true;
}

// Helper class to manage GPU texture resources
class TextureManager {
private:
    struct TextureData {
        cudaArray* depth_array = nullptr;
        cudaArray* normal_array = nullptr; 
        cudaArray* image_array = nullptr;
        cudaTextureObject_t depth_texture = 0;
        cudaTextureObject_t normal_texture = 0;
        cudaTextureObject_t image_texture = 0;
        bool loaded = false;
    };
    
    std::unordered_map<int, TextureData> textures;
    size_t max_textures_in_memory;
    std::queue<int> lru_queue;
    
public:
    TextureManager(size_t max_textures = 10) : max_textures_in_memory(max_textures) {}
    
    ~TextureManager() {
        for (auto& pair : textures) {
            releaseTexture(pair.first);
        }
    }
    
    bool loadTexture(int image_id, const cv::Mat_<float>& depth, 
                    const cv::Mat_<cv::Vec3f>& normal, const cv::Mat& image) {
        // Check if we need to free some memory
        while (lru_queue.size() >= max_textures_in_memory) {
            int old_id = lru_queue.front();
            lru_queue.pop();
            releaseTexture(old_id);
        }
        
        TextureData& data = textures[image_id];
        if (data.loaded) {
            return true; // Already loaded
        }
        
        int width = depth.cols;
        int height = depth.rows;
        
        try {
            // Create depth texture
            cudaChannelFormatDesc depth_desc = cudaCreateChannelDesc<float>();
            CUDA_SAFE_CALL(cudaMallocArray(&data.depth_array, &depth_desc, width, height));
            CUDA_SAFE_CALL(cudaMemcpy2DToArray(data.depth_array, 0, 0, 
                                               depth.ptr<float>(), depth.step[0], 
                                               width * sizeof(float), height, 
                                               cudaMemcpyHostToDevice));

            cudaResourceDesc depth_res_desc = {};
            depth_res_desc.resType = cudaResourceTypeArray;
            depth_res_desc.res.array.array = data.depth_array;

            cudaTextureDesc depth_tex_desc = {};
            depth_tex_desc.addressMode[0] = cudaAddressModeWrap;
            depth_tex_desc.addressMode[1] = cudaAddressModeClamp;
            depth_tex_desc.filterMode = cudaFilterModePoint;
            depth_tex_desc.readMode = cudaReadModeElementType;
            depth_tex_desc.normalizedCoords = false;

            CUDA_SAFE_CALL(cudaCreateTextureObject(&data.depth_texture, 
                                                   &depth_res_desc, &depth_tex_desc, NULL));

            // Create normal texture
            cv::Mat normal_rgba;
            cv::cvtColor(normal, normal_rgba, cv::COLOR_RGB2RGBA);
            
            cudaChannelFormatDesc normal_desc = cudaCreateChannelDesc<float4>();
            CUDA_SAFE_CALL(cudaMallocArray(&data.normal_array, &normal_desc, width, height));
            CUDA_SAFE_CALL(cudaMemcpy2DToArray(data.normal_array, 0, 0, 
                                               normal_rgba.ptr<float>(), normal_rgba.step[0], 
                                               width * sizeof(float4), height, 
                                               cudaMemcpyHostToDevice));

            cudaResourceDesc normal_res_desc = {};
            normal_res_desc.resType = cudaResourceTypeArray;
            normal_res_desc.res.array.array = data.normal_array;

            cudaTextureDesc normal_tex_desc = {};
            normal_tex_desc.addressMode[0] = cudaAddressModeWrap;
            normal_tex_desc.addressMode[1] = cudaAddressModeClamp;
            normal_tex_desc.filterMode = cudaFilterModePoint;
            normal_tex_desc.readMode = cudaReadModeElementType;
            normal_tex_desc.normalizedCoords = false;

            CUDA_SAFE_CALL(cudaCreateTextureObject(&data.normal_texture, 
                                                   &normal_res_desc, &normal_tex_desc, NULL));

            // Create image texture
            cv::Mat rgba_float;
            cv::Mat rgba;
            cv::cvtColor(image, rgba, cv::COLOR_BGR2RGBA);
            rgba.convertTo(rgba_float, CV_32FC4, 1.0/255.0);

            cudaChannelFormatDesc image_desc = cudaCreateChannelDesc<float4>();
            CUDA_SAFE_CALL(cudaMallocArray(&data.image_array, &image_desc, width, height));
            CUDA_SAFE_CALL(cudaMemcpy2DToArray(data.image_array, 0, 0, 
                                               rgba_float.ptr<float>(), rgba_float.step[0], 
                                               width * sizeof(float4), height, 
                                               cudaMemcpyHostToDevice));

            cudaResourceDesc image_res_desc = {};
            image_res_desc.resType = cudaResourceTypeArray;
            image_res_desc.res.array.array = data.image_array;

            cudaTextureDesc image_tex_desc = {};
            image_tex_desc.addressMode[0] = cudaAddressModeWrap;
            image_tex_desc.addressMode[1] = cudaAddressModeClamp;
            image_tex_desc.filterMode = cudaFilterModeLinear;
            image_tex_desc.readMode = cudaReadModeElementType;
            image_tex_desc.normalizedCoords = false;

            CUDA_SAFE_CALL(cudaCreateTextureObject(&data.image_texture, 
                                                   &image_res_desc, &image_tex_desc, NULL));

            data.loaded = true;
            lru_queue.push(image_id);
            return true;
            
        } catch (...) {
            releaseTexture(image_id);
            return false;
        }
    }
    
    void releaseTexture(int image_id) {
        auto it = textures.find(image_id);
        if (it == textures.end() || !it->second.loaded) return;
        
        TextureData& data = it->second;
        if (data.depth_texture) cudaDestroyTextureObject(data.depth_texture);
        if (data.normal_texture) cudaDestroyTextureObject(data.normal_texture);
        if (data.image_texture) cudaDestroyTextureObject(data.image_texture);
        if (data.depth_array) cudaFreeArray(data.depth_array);
        if (data.normal_array) cudaFreeArray(data.normal_array);
        if (data.image_array) cudaFreeArray(data.image_array);
        
        data = TextureData(); // Reset to defaults
    }
    
    cudaTextureObject_t getDepthTexture(int image_id) {
        auto it = textures.find(image_id);
        return (it != textures.end() && it->second.loaded) ? it->second.depth_texture : 0;
    }
    
    cudaTextureObject_t getNormalTexture(int image_id) {
        auto it = textures.find(image_id);
        return (it != textures.end() && it->second.loaded) ? it->second.normal_texture : 0;
    }
    
    cudaTextureObject_t getImageTexture(int image_id) {
        auto it = textures.find(image_id);
        return (it != textures.end() && it->second.loaded) ? it->second.image_texture : 0;
    }
};

// Streaming data loader - loads data on demand
class StreamingDataLoader {
private:
    std::string dense_folder;
    std::string img_folder;
    std::string cam_folder;
    bool geom_consistency;
    
    // Cache recently loaded data
    std::unordered_map<int, Camera> camera_cache;
    std::unordered_map<int, cv::Mat_<float>> depth_cache;
    std::unordered_map<int, cv::Mat_<cv::Vec3f>> normal_cache;
    std::unordered_map<int, cv::Mat> image_cache;
    
    size_t max_cache_size = 5; // Keep only 5 items in each cache
    
public:
    StreamingDataLoader(const std::string& folder, bool geom = false) 
        : dense_folder(folder), geom_consistency(geom) {
        img_folder = folder + "/images";
        cam_folder = folder + "/cams";
    }
    
    bool loadCamera(int image_id, Camera& cam) {
        auto it = camera_cache.find(image_id);
        if (it != camera_cache.end()) {
            cam = it->second;
            return true;
        }
        
        char buf[256];
        sprintf(buf, "%s/%08d_cam.txt", cam_folder.c_str(), image_id);
        std::string cam_file(buf);
        
        try {
            cam = ReadCamera(cam_file);
            
            // Validate camera after loading
            if (cam.width <= 0 || cam.height <= 0) {
                std::cerr << "Warning: Camera file " << cam_file << " has invalid dimensions: " 
                          << cam.width << "x" << cam.height << std::endl;
                return false;
            }
            
            // Manage cache size
            if (camera_cache.size() >= max_cache_size) {
                camera_cache.erase(camera_cache.begin());
            }
            camera_cache[image_id] = cam;
            return true;
        } catch (...) {
            std::cerr << "Warning: Failed to read camera file: " << cam_file << std::endl;
            return false;
        }
    }
    
    bool loadDepth(int image_id, cv::Mat_<float>& depth) {
        auto it = depth_cache.find(image_id);
        if (it != depth_cache.end()) {
            depth = it->second;
            return true;
        }
        
        char buf[256];
        std::string depth_suffix = geom_consistency ? "/depths_geom.dmb" : "/depths.dmb";
        sprintf(buf, "%s/ACMMP/2333_%08d%s", dense_folder.c_str(), image_id, depth_suffix.c_str());
        std::string depth_file(buf);
        
        if (readDepthDmb(depth_file, depth) != 0) {
            std::cerr << "Warning: Failed to read depth file: " << depth_file << std::endl;
            return false;
        }
        
        // Validate depth dimensions
        if (depth.cols <= 0 || depth.rows <= 0) {
            std::cerr << "Warning: Depth file " << depth_file << " has invalid dimensions: " 
                      << depth.cols << "x" << depth.rows << std::endl;
            return false;
        }
        
        // Manage cache size
        if (depth_cache.size() >= max_cache_size) {
            depth_cache.erase(depth_cache.begin());
        }
        depth_cache[image_id] = depth;
        return true;
    }
    
    bool loadNormal(int image_id, cv::Mat_<cv::Vec3f>& normal) {
        auto it = normal_cache.find(image_id);
        if (it != normal_cache.end()) {
            normal = it->second;
            return true;
        }
        
        char buf[256];
        sprintf(buf, "%s/ACMMP/2333_%08d/normals.dmb", dense_folder.c_str(), image_id);
        std::string normal_file(buf);
        
        if (readNormalDmb(normal_file, normal) != 0) {
            std::cerr << "Warning: Failed to read normal file: " << normal_file << std::endl;
            return false;
        }
        
        // Validate normal dimensions
        if (normal.cols <= 0 || normal.rows <= 0) {
            std::cerr << "Warning: Normal file " << normal_file << " has invalid dimensions: " 
                      << normal.cols << "x" << normal.rows << std::endl;
            return false;
        }
        
        // Manage cache size
        if (normal_cache.size() >= max_cache_size) {
            normal_cache.erase(normal_cache.begin());
        }
        normal_cache[image_id] = normal;
        return true;
    }
    
    bool loadImage(int image_id, cv::Mat& image) {
        auto it = image_cache.find(image_id);
        if (it != image_cache.end()) {
            image = it->second;
            return true;
        }
        
        char buf[256];
        sprintf(buf, "%s/%08d.jpg", img_folder.c_str(), image_id);
        std::string img_file(buf);
        
        image = cv::imread(img_file, cv::IMREAD_COLOR);
        if (image.empty()) {
            return false;
        }
        
        // Manage cache size
        if (image_cache.size() >= max_cache_size) {
            image_cache.erase(image_cache.begin());
        }
        image_cache[image_id] = image;
        return true;
    }
};

// Modified fusion kernel that works with sparse texture arrays
__global__ void EfficientFusionKernel(
    cudaTextureObject_t* depth_textures,
    cudaTextureObject_t* normal_textures,
    cudaTextureObject_t* image_textures,
    int* texture_image_ids,  // Maps texture indices to image IDs
    int num_loaded_textures,
    Camera* cameras,
    int* camera_image_ids,   // Maps camera indices to image IDs
    int num_cameras,
    int ref_image_id,
    int* src_image_ids,
    int num_src_images,
    PointList* output_points,
    int* valid_flags,
    int width,
    int height
) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int r = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (c >= width || r >= height) return;
    
    int idx = r * width + c;
    
    // Find reference camera
    int ref_cam_idx = -1;
    for (int i = 0; i < num_cameras; ++i) {
        if (camera_image_ids[i] == ref_image_id) {
            ref_cam_idx = i;
            break;
        }
    }
    if (ref_cam_idx == -1) {
        valid_flags[idx] = 0;
        return;
    }
    
    // Find reference texture
    int ref_tex_idx = -1;
    for (int i = 0; i < num_loaded_textures; ++i) {
        if (texture_image_ids[i] == ref_image_id) {
            ref_tex_idx = i;
            break;
        }
    }
    if (ref_tex_idx == -1) {
        valid_flags[idx] = 0;
        return;
    }
    
    const Camera& ref_cam = cameras[ref_cam_idx];
    
    // Sample reference depth
    float ref_depth = tex2D<float>(depth_textures[ref_tex_idx], c, r);
    
    if (ref_depth <= 0.0f) {
        valid_flags[idx] = 0;
        return;
    }

    // Get 3D point in world coordinates
    float3 PointX = Get3DPointonWorld_cu(static_cast<float>(c), static_cast<float>(r), ref_depth, ref_cam);
    
    // Sample reference normal and color
    float4 ref_normal_tex = tex2D<float4>(normal_textures[ref_tex_idx], c, r);
    float3 ref_normal = make_float3(ref_normal_tex.x, ref_normal_tex.y, ref_normal_tex.z);
    
    float4 ref_color = tex2D<float4>(image_textures[ref_tex_idx], c, r);
    
    // Initialize sums for averaging
    float3 point_sum = PointX;
    float3 normal_sum = ref_normal;
    float color_sum[3] = {
        ref_color.z * 255.0f,  // R (OpenCV uses BGR)
        ref_color.y * 255.0f,  // G
        ref_color.x * 255.0f   // B
    };
    int num_consistent = 1;
    
    // Check all source images
    for (int j = 0; j < num_src_images; ++j) {
        int src_image_id = src_image_ids[j];
        
        // Find source camera
        int src_cam_idx = -1;
        for (int i = 0; i < num_cameras; ++i) {
            if (camera_image_ids[i] == src_image_id) {
                src_cam_idx = i;
                break;
            }
        }
        if (src_cam_idx == -1) continue;
        
        // Find source texture
        int src_tex_idx = -1;
        for (int i = 0; i < num_loaded_textures; ++i) {
            if (texture_image_ids[i] == src_image_id) {
                src_tex_idx = i;
                break;
            }
        }
        if (src_tex_idx == -1) continue;
        
        const Camera& src_cam = cameras[src_cam_idx];
        
        // Project 3D point to source camera
        float2 proj_point;
        float proj_depth_in_src;
        ProjectonCamera_cu(PointX, src_cam, proj_point, proj_depth_in_src);
        
        int src_c = static_cast<int>(proj_point.x + 0.5f);
        int src_r = static_cast<int>(proj_point.y + 0.5f);
        
        // Check if projection is within image bounds
        if (src_c < 0 || src_c >= src_cam.width || src_r < 0 || src_r >= src_cam.height) 
            continue;
        
        // Sample source depth
        float src_depth = tex2D<float>(depth_textures[src_tex_idx], src_c, src_r);
        if (src_depth <= 0.0f) continue;
        
        // Get 3D point from source depth
        float3 PointX_src = Get3DPointonWorld_cu(static_cast<float>(src_c), static_cast<float>(src_r), src_depth, src_cam);
        
        // Reproject source 3D point back to reference camera
        float2 reproj_point_in_ref;
        float dummy_depth;
        ProjectonCamera_cu(PointX_src, ref_cam, reproj_point_in_ref, dummy_depth);
        
        // Calculate reprojection error
        float reproj_error = hypotf(c - reproj_point_in_ref.x, r - reproj_point_in_ref.y);
        
        // Calculate relative depth difference
        float relative_depth_diff = fabsf(proj_depth_in_src - src_depth) / src_depth;
        
        // Sample source normal
        float4 src_normal_tex = tex2D<float4>(normal_textures[src_tex_idx], src_c, src_r);
        float3 src_normal = make_float3(src_normal_tex.x, src_normal_tex.y, src_normal_tex.z);
        
        // Calculate angle between normals
        float dot_product = ref_normal.x * src_normal.x + ref_normal.y * src_normal.y + ref_normal.z * src_normal.z;
        dot_product = fmaxf(-1.0f, fminf(1.0f, dot_product));
        float angle = acosf(dot_product);
        
        // Check consistency
        if (reproj_error < 1.0 && relative_depth_diff < 0.01f && angle < 0.1f) {
            point_sum.x += PointX_src.x;
            point_sum.y += PointX_src.y;
            point_sum.z += PointX_src.z;
            
            normal_sum.x += src_normal.x;
            normal_sum.y += src_normal.y;
            normal_sum.z += src_normal.z;
            
            float4 src_color = tex2D<float4>(image_textures[src_tex_idx], src_c, src_r);
            color_sum[0] += src_color.z * 255.0f;
            color_sum[1] += src_color.y * 255.0f;
            color_sum[2] += src_color.x * 255.0f;
            
            num_consistent++;
        }
    }
    
    // Check if we have enough consistent views
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
        
        output_points[idx] = final_point;
        valid_flags[idx] = 1;
    } else {
        valid_flags[idx] = 0;
    }
}

// Memory-efficient main fusion function
void RunFusionCuda(const std::string &dense_folder,
                           const std::vector<Problem> &problems,
                           bool geom_consistency,
                           size_t max_textures_in_memory)
{
    const size_t N = problems.size();
    
    // Input validation
    if (N == 0) {
        std::cerr << "Error: No problems to process!" << std::endl;
        return;
    }
    
    if (max_textures_in_memory < 3) {
        std::cerr << "Warning: max_textures_in_memory too low, setting to 3" << std::endl;
        max_textures_in_memory = 3;
    }
    
    // Check CUDA device
    int device_count;
    cudaError_t cuda_status = cudaGetDeviceCount(&device_count);
    if (cuda_status != cudaSuccess || device_count == 0) {
        std::cerr << "Error: No CUDA devices available!" << std::endl;
        return;
    }
    
    // Print device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "[Efficient CUDA Fusion] Using device: " << prop.name << std::endl;
    std::cout << "[Efficient CUDA Fusion] Total GPU memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;
    std::cout << "[Efficient CUDA Fusion] Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    
    std::cout << "[Efficient CUDA Fusion] Starting with " << N << " images, max " 
              << max_textures_in_memory << " textures in memory..." << std::endl;

    StreamingDataLoader loader(dense_folder, geom_consistency);
    TextureManager texture_manager(max_textures_in_memory);
    
    // Collect all unique image IDs
    std::unordered_set<int> all_image_ids;
    for (const auto& problem : problems) {
        all_image_ids.insert(problem.ref_image_id);
        for (int src_id : problem.src_image_ids) {
            all_image_ids.insert(src_id);
        }
    }
    
    std::cout << "[Efficient CUDA Fusion] Total unique images: " << all_image_ids.size() << std::endl;
    
    // Pre-load all cameras (they're small)
    std::vector<Camera> all_cameras;
    std::vector<int> camera_image_ids;
    std::unordered_map<int, int> image_id_to_camera_idx;
    
    for (int image_id : all_image_ids) {
        Camera cam;
        if (loader.loadCamera(image_id, cam)) {
            image_id_to_camera_idx[image_id] = all_cameras.size();
            all_cameras.push_back(cam);
            camera_image_ids.push_back(image_id);
        }
    }
    
    std::cout << "[Efficient CUDA Fusion] Loaded " << all_cameras.size() << " cameras" << std::endl;
    
    // Copy cameras to GPU (small memory footprint)
    Camera* cameras_cuda = nullptr;
    int* camera_image_ids_cuda = nullptr;
    
    cudaError_t alloc_error;
    alloc_error = cudaMalloc(&cameras_cuda, all_cameras.size() * sizeof(Camera));
    if (alloc_error != cudaSuccess) {
        std::cerr << "Error allocating camera memory: " << cudaGetErrorString(alloc_error) << std::endl;
        return;
    }
    
    alloc_error = cudaMalloc(&camera_image_ids_cuda, camera_image_ids.size() * sizeof(int));
    if (alloc_error != cudaSuccess) {
        std::cerr << "Error allocating camera IDs memory: " << cudaGetErrorString(alloc_error) << std::endl;
        cudaFree(cameras_cuda);
        return;
    }
    
    cudaError_t copy_error;
    copy_error = cudaMemcpy(cameras_cuda, all_cameras.data(), 
                           all_cameras.size() * sizeof(Camera), cudaMemcpyHostToDevice);
    if (copy_error != cudaSuccess) {
        std::cerr << "Error copying cameras to GPU: " << cudaGetErrorString(copy_error) << std::endl;
        cudaFree(cameras_cuda);
        cudaFree(camera_image_ids_cuda);
        return;
    }
    
    copy_error = cudaMemcpy(camera_image_ids_cuda, camera_image_ids.data(), 
                           camera_image_ids.size() * sizeof(int), cudaMemcpyHostToDevice);
    if (copy_error != cudaSuccess) {
        std::cerr << "Error copying camera IDs to GPU: " << cudaGetErrorString(copy_error) << std::endl;
        cudaFree(cameras_cuda);
        cudaFree(camera_image_ids_cuda);
        return;
    }
    
    std::vector<PointList> all_points;
    size_t total_points = 0;
    
    // Process each reference image
    for (size_t prob_idx = 0; prob_idx < problems.size(); ++prob_idx) {
        const Problem& problem = problems[prob_idx];
        int ref_image_id = problem.ref_image_id;
        
        std::cout << "[Efficient CUDA Fusion] Processing " << (prob_idx + 1) << "/" << problems.size() 
                  << " (ID=" << ref_image_id << ")" << std::endl;
        
        // Determine which images we need for this problem
        std::unordered_set<int> needed_images;
        needed_images.insert(ref_image_id);
        for (int src_id : problem.src_image_ids) {
            needed_images.insert(src_id);
        }
        
        // Load required data for these images
        std::vector<int> loaded_image_ids;
        std::vector<cudaTextureObject_t> depth_textures_host;
        std::vector<cudaTextureObject_t> normal_textures_host;
        std::vector<cudaTextureObject_t> image_textures_host;
        
        Camera ref_camera;
        bool ref_camera_found = false;
        
        for (int image_id : needed_images) {
            // Load CPU data
            Camera cam;
            cv::Mat_<float> depth;
            cv::Mat_<cv::Vec3f> normal;
            cv::Mat image;
            
            if (!loader.loadCamera(image_id, cam) ||
                !loader.loadDepth(image_id, depth) ||
                !loader.loadNormal(image_id, normal) ||
                !loader.loadImage(image_id, image)) {
                std::cerr << "Warning: Failed to load data for image " << image_id << std::endl;
                continue;
            }
            
            // Validate loaded data dimensions
            if (depth.cols <= 0 || depth.rows <= 0) {
                std::cerr << "Warning: Invalid depth dimensions for image " << image_id 
                          << ": " << depth.cols << "x" << depth.rows << std::endl;
                continue;
            }
            
            if (normal.cols != depth.cols || normal.rows != depth.rows) {
                std::cerr << "Warning: Normal/depth dimension mismatch for image " << image_id 
                          << ": normal=" << normal.cols << "x" << normal.rows 
                          << " depth=" << depth.cols << "x" << depth.rows << std::endl;
                continue;
            }
            
            std::cout << "    Loaded image " << image_id << " with dimensions: " 
                      << depth.cols << "x" << depth.rows << std::endl;
            
            if (image_id == ref_image_id) {
                ref_camera = cam;
                ref_camera_found = true;
                
                // Extra validation for reference camera since it's critical
                if (ref_camera.width != depth.cols || ref_camera.height != depth.rows) {
                    std::cerr << "Warning: Reference camera/depth dimension mismatch for image " << image_id 
                              << ": camera=" << ref_camera.width << "x" << ref_camera.height 
                              << " depth=" << depth.cols << "x" << depth.rows << std::endl;
                    // Force correct dimensions
                    ref_camera.width = depth.cols;
                    ref_camera.height = depth.rows;
                }
                
                std::cout << "    Reference camera set with dimensions: " 
                          << ref_camera.width << "x" << ref_camera.height << std::endl;
            }
            
            // Rescale image and camera to match depth resolution
            cv::Mat_<cv::Vec3b> img_color;
            if (image.channels() == 3) {
                img_color = cv::Mat_<cv::Vec3b>(image);
            } else {
                cv::cvtColor(image, img_color, cv::COLOR_GRAY2BGR);
            }
            cv::Mat_<cv::Vec3b> scaled_color;
            RescaleImageAndCamera(img_color, scaled_color, depth, cam);
            image = cv::Mat(scaled_color);
            
            // Ensure camera dimensions are correctly set
            cam.width = depth.cols;
            cam.height = depth.rows;
            
            // Final validation
            if (cam.width <= 0 || cam.height <= 0) {
                std::cerr << "Warning: Invalid camera dimensions for image " << image_id 
                          << ": " << cam.width << "x" << cam.height << std::endl;
                continue;
            }
            
            // Load into GPU textures
            if (texture_manager.loadTexture(image_id, depth, normal, image)) {
                loaded_image_ids.push_back(image_id);
                depth_textures_host.push_back(texture_manager.getDepthTexture(image_id));
                normal_textures_host.push_back(texture_manager.getNormalTexture(image_id));
                image_textures_host.push_back(texture_manager.getImageTexture(image_id));
            }
        }
        
        if (!ref_camera_found || loaded_image_ids.empty()) {
            std::cerr << "Warning: Could not load reference image " << ref_image_id << std::endl;
            continue;
        }
        
        // Final validation of reference camera dimensions
        if (ref_camera.width <= 0 || ref_camera.height <= 0) {
            std::cerr << "Error: Invalid reference camera dimensions for image " << ref_image_id 
                      << ": " << ref_camera.width << "x" << ref_camera.height << std::endl;
            continue;
        }
        
        std::cout << "  Loaded " << loaded_image_ids.size() << " textures for this problem" << std::endl;
        std::cout << "  Reference camera dimensions: " << ref_camera.width << "x" << ref_camera.height << std::endl;
        
        // Process this reference image
        int width = ref_camera.width;
        int height = ref_camera.height;
        int total_pixels = width * height;
        
        // Copy texture arrays to GPU with error checking
        size_t texture_array_size = loaded_image_ids.size() * sizeof(cudaTextureObject_t);
        size_t image_ids_size = loaded_image_ids.size() * sizeof(int);
        size_t src_ids_size = problem.src_image_ids.size() * sizeof(int);
        size_t points_size = total_pixels * sizeof(PointList);
        size_t flags_size = total_pixels * sizeof(int);
        
        size_t total_required = 3 * texture_array_size + image_ids_size + src_ids_size + points_size + flags_size;
        
        if (!checkGPUMemory(total_required)) {
            std::cerr << "Warning: Skipping image " << ref_image_id << " due to insufficient GPU memory" << std::endl;
            continue;
        }
        
        // Declare all variables at the beginning to avoid goto issues
        cudaTextureObject_t* depth_textures_cuda = nullptr;
        cudaTextureObject_t* normal_textures_cuda = nullptr;
        cudaTextureObject_t* image_textures_cuda = nullptr;
        int* texture_image_ids_cuda = nullptr;
        int* src_image_ids_cuda = nullptr;
        PointList* output_points_cuda = nullptr;
        int* valid_flags_cuda = nullptr;
        cudaError_t alloc_error = cudaSuccess;
        cudaError_t copy_error = cudaSuccess;
        bool allocation_successful = true;
        
        // Allocate GPU memory
        alloc_error = cudaMalloc(&depth_textures_cuda, texture_array_size);
        if (alloc_error != cudaSuccess) allocation_successful = false;
        
        if (allocation_successful) {
            alloc_error = cudaMalloc(&normal_textures_cuda, texture_array_size);
            if (alloc_error != cudaSuccess) allocation_successful = false;
        }
        
        if (allocation_successful) {
            alloc_error = cudaMalloc(&image_textures_cuda, texture_array_size);
            if (alloc_error != cudaSuccess) allocation_successful = false;
        }
        
        if (allocation_successful) {
            alloc_error = cudaMalloc(&texture_image_ids_cuda, image_ids_size);
            if (alloc_error != cudaSuccess) allocation_successful = false;
        }
        
        if (allocation_successful) {
            alloc_error = cudaMalloc(&src_image_ids_cuda, src_ids_size);
            if (alloc_error != cudaSuccess) allocation_successful = false;
        }
        
        if (allocation_successful) {
            alloc_error = cudaMalloc(&output_points_cuda, points_size);
            if (alloc_error != cudaSuccess) allocation_successful = false;
        }
        
        if (allocation_successful) {
            alloc_error = cudaMalloc(&valid_flags_cuda, flags_size);
            if (alloc_error != cudaSuccess) allocation_successful = false;
        }
        
        if (!allocation_successful) {
            std::cerr << "Error allocating GPU memory for image " << ref_image_id << ": " << cudaGetErrorString(alloc_error) << std::endl;
            // Cleanup allocated memory
            if (depth_textures_cuda) cudaFree(depth_textures_cuda);
            if (normal_textures_cuda) cudaFree(normal_textures_cuda);
            if (image_textures_cuda) cudaFree(image_textures_cuda);
            if (texture_image_ids_cuda) cudaFree(texture_image_ids_cuda);
            if (src_image_ids_cuda) cudaFree(src_image_ids_cuda);
            if (output_points_cuda) cudaFree(output_points_cuda);
            if (valid_flags_cuda) cudaFree(valid_flags_cuda);
            continue;
        }
        
        // Copy data to GPU
        bool copy_successful = true;
        
        copy_error = cudaMemcpy(depth_textures_cuda, depth_textures_host.data(), texture_array_size, cudaMemcpyHostToDevice);
        if (copy_error != cudaSuccess) copy_successful = false;
        
        if (copy_successful) {
            copy_error = cudaMemcpy(normal_textures_cuda, normal_textures_host.data(), texture_array_size, cudaMemcpyHostToDevice);
            if (copy_error != cudaSuccess) copy_successful = false;
        }
        
        if (copy_successful) {
            copy_error = cudaMemcpy(image_textures_cuda, image_textures_host.data(), texture_array_size, cudaMemcpyHostToDevice);
            if (copy_error != cudaSuccess) copy_successful = false;
        }
        
        if (copy_successful) {
            copy_error = cudaMemcpy(texture_image_ids_cuda, loaded_image_ids.data(), image_ids_size, cudaMemcpyHostToDevice);
            if (copy_error != cudaSuccess) copy_successful = false;
        }
        
        if (copy_successful) {
            copy_error = cudaMemcpy(src_image_ids_cuda, problem.src_image_ids.data(), src_ids_size, cudaMemcpyHostToDevice);
            if (copy_error != cudaSuccess) copy_successful = false;
        }
        
        if (copy_successful) {
            copy_error = cudaMemset(valid_flags_cuda, 0, flags_size);
            if (copy_error != cudaSuccess) copy_successful = false;
        }
        
        if (!copy_successful) {
            std::cerr << "Error copying data to GPU for image " << ref_image_id << ": " << cudaGetErrorString(copy_error) << std::endl;
            // Cleanup
            cudaFree(depth_textures_cuda);
            cudaFree(normal_textures_cuda);
            cudaFree(image_textures_cuda);
            cudaFree(texture_image_ids_cuda);
            cudaFree(src_image_ids_cuda);
            cudaFree(output_points_cuda);
            cudaFree(valid_flags_cuda);
            continue;
        }
        
        // Validate dimensions
        if (width <= 0 || height <= 0) {
            std::cerr << "Error: Invalid image dimensions " << width << "x" << height << std::endl;
            cudaFree(depth_textures_cuda);
            cudaFree(normal_textures_cuda);
            cudaFree(image_textures_cuda);
            cudaFree(texture_image_ids_cuda);
            cudaFree(src_image_ids_cuda);
            cudaFree(output_points_cuda);
            cudaFree(valid_flags_cuda);
            continue;
        }
        
        // Safe block size - check device properties
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        
        int max_threads_per_block = prop.maxThreadsPerBlock;
        int max_block_dim_x = prop.maxThreadsDim[0];
        int max_block_dim_y = prop.maxThreadsDim[1];
        
        // Use smaller block size for compatibility
        dim3 block_size(16, 16);
        if (16 * 16 > max_threads_per_block || 16 > max_block_dim_x || 16 > max_block_dim_y) {
            // Fallback to smaller block size
            block_size = dim3(8, 8);
            if (8 * 8 > max_threads_per_block) {
                block_size = dim3(16, 1);  // Linear block
            }
        }
        
        // Calculate grid size with bounds checking
        dim3 grid_size((width + block_size.x - 1) / block_size.x, 
                      (height + block_size.y - 1) / block_size.y);
        
        // Validate grid dimensions
        if (grid_size.x == 0 || grid_size.y == 0 || 
            grid_size.x > prop.maxGridSize[0] || grid_size.y > prop.maxGridSize[1]) {
            std::cerr << "Error: Invalid grid size " << grid_size.x << "x" << grid_size.y << std::endl;
            std::cerr << "Max grid size: " << prop.maxGridSize[0] << "x" << prop.maxGridSize[1] << std::endl;
            cudaFree(depth_textures_cuda);
            cudaFree(normal_textures_cuda);
            cudaFree(image_textures_cuda);
            cudaFree(texture_image_ids_cuda);
            cudaFree(src_image_ids_cuda);
            cudaFree(output_points_cuda);
            cudaFree(valid_flags_cuda);
            continue;
        }
        
        std::cout << "  Launching kernel with block(" << block_size.x << "," << block_size.y 
                  << ") grid(" << grid_size.x << "," << grid_size.y << ")" << std::endl;
        
        // Launch kernel with error checking
        EfficientFusionKernel<<<grid_size, block_size>>>(
            depth_textures_cuda,
            normal_textures_cuda,
            image_textures_cuda,
            texture_image_ids_cuda,
            (int)loaded_image_ids.size(),
            cameras_cuda,
            camera_image_ids_cuda,
            (int)all_cameras.size(),
            ref_image_id,
            src_image_ids_cuda,
            (int)problem.src_image_ids.size(),
            output_points_cuda,
            valid_flags_cuda,
            width,
            height
        );
        
        cudaError_t launch_error = cudaGetLastError();
        if (launch_error != cudaSuccess) {
            std::cerr << "Error launching kernel: " << cudaGetErrorString(launch_error) << std::endl;
            cudaFree(depth_textures_cuda);
            cudaFree(normal_textures_cuda);
            cudaFree(image_textures_cuda);
            cudaFree(texture_image_ids_cuda);
            cudaFree(src_image_ids_cuda);
            cudaFree(output_points_cuda);
            cudaFree(valid_flags_cuda);
            continue;
        }
        
        cudaError_t sync_error = cudaDeviceSynchronize();
        if (sync_error != cudaSuccess) {
            std::cerr << "Error synchronizing device: " << cudaGetErrorString(sync_error) << std::endl;
            cudaFree(depth_textures_cuda);
            cudaFree(normal_textures_cuda);
            cudaFree(image_textures_cuda);
            cudaFree(texture_image_ids_cuda);
            cudaFree(src_image_ids_cuda);
            cudaFree(output_points_cuda);
            cudaFree(valid_flags_cuda);
            continue;
        }
        
        // Copy results back
        std::vector<PointList> points(total_pixels);
        std::vector<int> valid_flags(total_pixels);
        
        CUDA_SAFE_CALL(cudaMemcpy(points.data(), output_points_cuda, 
                                  total_pixels * sizeof(PointList), cudaMemcpyDeviceToHost));
        CUDA_SAFE_CALL(cudaMemcpy(valid_flags.data(), valid_flags_cuda, 
                                  total_pixels * sizeof(int), cudaMemcpyDeviceToHost));
        
        // Collect valid points
        size_t valid_count = 0;
        for (int j = 0; j < total_pixels; ++j) {
            if (valid_flags[j]) {
                all_points.push_back(points[j]);
                valid_count++;
            }
        }
        
        total_points += valid_count;
        std::cout << "  Generated " << valid_count << " points" << std::endl;
        
        // Cleanup GPU memory for this iteration
        cudaFree(depth_textures_cuda);
        cudaFree(normal_textures_cuda);
        cudaFree(image_textures_cuda);
        cudaFree(texture_image_ids_cuda);
        cudaFree(src_image_ids_cuda);
        cudaFree(output_points_cuda);
        cudaFree(valid_flags_cuda);
        
        // Force GPU memory cleanup
        cudaDeviceSynchronize();
    }
    
    // Write output
    std::string output_path = dense_folder + "/ACMMP/ACMM_model_cuda_efficient.ply";
    StoreColorPlyFileBinaryPointCloud(output_path, all_points);
    std::cout << "[Efficient CUDA Fusion] Complete! Wrote " << total_points 
              << " points to " << output_path << std::endl;
    
    // Cleanup
    cudaFree(cameras_cuda);
    cudaFree(camera_image_ids_cuda);
}

// Usage example:
/*
int main() {
    std::string dense_folder = "/path/to/your/scene";
    std::vector<Problem> problems = LoadProblems(dense_folder + "/problems.txt");
    bool geom_consistency = true;
    
    // Adjust based on your GPU memory (10 images ≈ 2-4GB GPU memory for 1920x1080 images)
    size_t max_textures = 8;  // Reduce if you get out-of-memory errors
    
    try {
        RunFusionCudaEfficient(dense_folder, problems, geom_consistency, max_textures);
    } catch (const std::exception& e) {
        std::cerr << "Fusion failed: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
*/