# APD-Spherical: Integrate APD-MVS Algorithm into ACMMP-Spherical

## Context

ACMMP-Spherical is ACMMP adapted for spherical/equirectangular cameras with LUT-accelerated projection, batch GPU processing, compressed output, and GPU fusion. The user wants to replace ACMMP's core MVS algorithm with APD-MVS (Adaptive Patch Deformation) while keeping all spherical infrastructure and optimizations.

**Branch**: `apd-spherical` in `/Dataset/mvs/adp/ACMMP-Spherical/`
**Output directory**: `APD/XXXXXXXX` (replacing `ACMMP/2333_XXXXXXXX`)

### Decisions Made
- **Pipeline**: APD's round-based (FIRST_INIT -> REFINE_INIT -> REFINE_ITER x3). Drop JBU upsampling + planar prior.
- **RNG**: Keep ACMMP's lightweight `RNGState` (4B) instead of APD's `curandState` (48B). Rewrite all `curand_uniform` calls.
- **Conventions**: No boost dep, use `std::string` paths, keep `MAX_IMAGES=256`.

---

## Step 1: Create Branch

```bash
cd /Dataset/mvs/adp/ACMMP-Spherical && git checkout -b apd-spherical
```

---

## Step 2: Modify `main.h` — Data Structure Merging

**File**: `/Dataset/mvs/adp/ACMMP-Spherical/main.h`

### 2a. Add APD enums (after CameraModel enum, ~line 43)
```cpp
enum RunState { FIRST_INIT, REFINE_INIT, REFINE_ITER };
enum PixelState { WEAK = 0, STRONG = 1, UNKNOWN = 2 };
```

### 2b. Add `c[3]` to Camera struct (after `t[3]`, ~line 54)
```cpp
float c[3]; // Camera center in world coords: c[j] = -(R[0+j]*t[0] + R[3+j]*t[1] + R[6+j]*t[2])
```
**Convention verified**: This is the same formula used inline by `Get3DPointonWorld_MultiRes` in `ACMMP_device.cuh:246-250`. APD's `ReadCamera` (APD.cpp:73-77) uses the identical formula. No convention mismatch.

### 2c. Extend Problem struct
```cpp
struct Problem {
    int ref_image_id;
    std::vector<int> src_image_ids;
    int max_image_size = 3200;
    int num_downscale = 0;
    int cur_image_size = 3200;
    int scale_size = 1;      // APD: downscale factor for current round
    int iteration = 0;       // APD: iteration tracking
};
```

### 2d. Add constants
```cpp
#define NEIGHBOUR_NUM 9
```

### 2e. Update function declarations
Add: `ComputeRoundNum`, `ProcessProblemAPD`.
Remove: `JointBilateralUpsampling`, `ComputeMultiScaleSettings`.

---

## Step 3: Modify `ACMMP.h` — APD Support Structures

**File**: `/Dataset/mvs/adp/ACMMP-Spherical/ACMMP.h`

### 3a. Add `DataPassHelper` struct (port from APD.h:40-65)
Change `curandState*` → `RNGState*`. Add `uint8_t* ref_mask_cuda`.

### 3b. Extend `PatchMatchParams` — add APD-specific fields
Add to existing struct (keep all ACMMP fields):
```cpp
int strong_radius = 5;  int strong_increment = 2;
int weak_radius = 5;    int weak_increment = 5;
bool use_APD = true;     int weak_peak_radius = 2;
int rotate_time = 4;    float ransac_threshold = 0.005f;
float geom_factor = 0.2f;  RunState state = FIRST_INIT;
```
Note: `disparity_min/max` (existing ACMMP fields) are NOT used anywhere in ACMMP.cu — verified by grep. No adaptation needed for spherical.

### 3c. DO NOT add APD GPU buffers to ACMMP class
**Buffer ownership decision**: All GPU buffers live in `ProblemGPUResources` (in `BatchACMMP.h`). The ACMMP class is a stateless processor that operates on resources. This prevents double-allocation and stale pointer bugs between batch/sequential modes. The ACMMP class only holds host-side data (images, cameras, weak_info_host, etc.).

Host-side additions to ACMMP class:
```cpp
cv::Mat weak_info_host;
cv::Mat neighbours_map_host;
cv::Mat selected_views_host;
int weak_count = 0;
```

New public methods:
```cpp
void RunAPDPatchMatch(ProblemGPUResources* res, bool skip_host_download = false);
void SetAPDParams(RunState state, bool use_apd, ...);
void APDInputInitialization(const std::string &dense_folder, const std::vector<Problem> &problems, const int idx);
void APDCudaSpaceInitialization(const std::string &dense_folder, const Problem &problem, ProblemGPUResources* res);
```

### 3d. Remove JBU class and structs
Remove `JBU`, `JBUParameters`, `JBUTexObj` from `ACMMP.h`. Also remove `JBU_cu` kernel from `ACMMP.cu` and `RunJBU` from `ACMMP.cpp`.

---

## Step 4: Create `APD_device.cuh` — Spherical Dispatch Layer

**New file**: `/Dataset/mvs/adp/ACMMP-Spherical/APD_device.cuh`

### Critical: Macro handling
`ACMMP_device.cuh` lines 512-518 define macros that remap `Get3DPoint` → `Get3DPoint_MultiRes` etc. These would silently break pinhole dispatch if APD code uses those names. Solution:

```cuda
#include "ACMMP_device.cuh"

// UNDEF the compatibility macros — APD code must use _Dispatch wrappers
#undef Get3DPoint
#undef GetViewDirection
#undef ComputeDepthfromPlaneHypothesis
#undef Get3DPointonWorld_cu
#undef Get3DPointonRefCam_cu
#undef ProjectonCamera_cu
#undef PixelToDir
```

Then define all `_Dispatch` wrappers that check `cam.model`:

**4a. `Get3DPoint_APD`** (replaces APD's `Get3DPoint`)
```cuda
__device__ __forceinline__ void Get3DPoint_APD(const Camera& cam, const int2 p, float depth, float *X) {
    if (cam.model == SPHERE) {
        Get3DPoint_MultiRes(cam, p, depth, X);
    } else {
        X[0] = depth * (p.x - cam.K[2]) / cam.K[0];
        X[1] = depth * (p.y - cam.K[5]) / cam.K[4];
        X[2] = depth;
    }
}
// + short2 overload
```

**4b-4e.** Same pattern for `GetViewDirection_APD`, `ComputeDepthfromPlaneHypothesis_APD`, `GetDistance2Origin_APD`, `Get3DPointonWorld_APD`.

**4f. `ComputeCorrespondingPoint_APD`** — per-pixel reprojection for spherical
```cuda
__device__ __forceinline__ float2 ComputeCorrespondingPoint_APD(
    const Camera& ref_cam, const Camera& src_cam,
    const float4& plane, const int2 p, const float *H_pinhole) {
    if (ref_cam.model == SPHERE) {
        float depth = ComputeDepthfromPlaneHypothesis_MultiRes(ref_cam, plane, p);
        float3 pw = Get3DPointonWorld_MultiRes((float)p.x, (float)p.y, depth, ref_cam);
        float2 sp; float sd;
        ProjectonCamera_MultiRes(pw, src_cam, sp, sd);
        return sp;
    } else {
        // Original APD homography path
        float3 pt;
        pt.x = H_pinhole[0]*p.x + H_pinhole[1]*p.y + H_pinhole[2];
        pt.y = H_pinhole[3]*p.x + H_pinhole[4]*p.y + H_pinhole[5];
        pt.z = H_pinhole[6]*p.x + H_pinhole[7]*p.y + H_pinhole[8];
        return make_float2(pt.x/pt.z, pt.y/pt.z);
    }
}
```

**4g. X-wrap helper** — handle multi-width overflow
```cuda
__device__ __forceinline__ bool CheckAndWrapSourcePoint(float2& src_pt, const Camera& cam) {
    if (cam.model == SPHERE) {
        float w = (float)cam.width;
        float h = (float)cam.height;
        // fmodf-style wrap for x (handles > ±w overflow)
        src_pt.x = fmodf(src_pt.x, w);
        if (src_pt.x < 0.f) src_pt.x += w;
        // Y poles: out of bounds = invalid
        if (src_pt.y < 0.f || src_pt.y >= h) return false;
        return true;
    } else {
        return (src_pt.x >= 0.f && src_pt.x < (float)cam.width &&
                src_pt.y >= 0.f && src_pt.y < (float)cam.height);
    }
}
```

**4h. Disparity conversion** (used by `DepthToWeak`, `LocalRefine`)
```cuda
__device__ __forceinline__ float DepthToDisparity_APD(const Camera& cam, float baseline, float depth) {
    if (cam.model == SPHERE) {
        float pixels_per_radian = (float)cam.width / (2.0f * M_PI);
        return pixels_per_radian * baseline / depth;
    } else {
        return cam.K[0] * baseline / depth;
    }
}
__device__ __forceinline__ float DisparityToDepth_APD(const Camera& cam, float baseline, float disparity) {
    if (cam.model == SPHERE) {
        float pixels_per_radian = (float)cam.width / (2.0f * M_PI);
        return pixels_per_radian * baseline / disparity;
    } else {
        return cam.K[0] * baseline / disparity;
    }
}
```
**Note on approximation**: `width/(2π) * baseline/depth` approximates angular disparity. It degrades at poles and with oblique baselines. Acceptable for depth classification (DepthToWeak) and local search (LocalRefine ±5 pixel range) but should be validated on test data. Near-pole pixels are already masked by ACMMP-Spherical's pole detection.

**4i. RNG wrapper** — reuse ACMMP's Philox-based `rng_uniform`
The existing `rng_uniform(RNGState*, uint pixel_key)` from ACMMP.cu line 11 is used directly. No new wrapper needed — just change all APD `curand_uniform(&rand_states[center])` to `rng_uniform(&rand_states[center], pixel_key)`.

---

## Step 5: Create `APD_kernels.cu` — APD Kernels in Separate TU

**New file**: `/Dataset/mvs/adp/ACMMP-Spherical/APD_kernels.cu`

### Why a separate file (not ACMMP.cu)
ACMMP.cu already defines many functions with identical names to APD.cu: `sort_small`, `NormalizeVec3`, `GetDistance2Origin`, `GenerateRandomNormal`, `RandomInitialization`, `BlackPixelUpdate`, etc. In CUDA's default (non-relocatable) compilation mode, `__device__` functions are internal to each `.cu` TU — **no linker collision**. `__global__` kernels have different parameter signatures so they're distinct overloads.

### File structure
```cuda
#include "ACMMP.h"
#include "APD_device.cuh"  // Includes ACMMP_device.cuh + undefs macros + defines _APD wrappers

// ====== UTILITY FUNCTIONS (local to this TU) ======
__device__ void sort_small(...) { /* same as APD.cu */ }
__device__ void NormalizeVec3(...) { /* same */ }
// ... all pure-math helpers copied verbatim

// ====== APD-SPECIFIC DEVICE FUNCTIONS ======
// Get3DPoint: calls Get3DPoint_APD from APD_device.cuh
// ComputeHomography: kept for pinhole, not called for spherical
// ComputeBilateralNCCOld/New: adapted with CheckAndWrapSourcePoint + ComputeCorrespondingPoint_APD

// ====== GLOBAL KERNELS ======
// Use DataPassHelper* parameter signature (distinct from ACMMP.cu's kernels)
__global__ void APD_InitRandomStates(DataPassHelper *helper) { ... }
__global__ void APD_RandomInitialization(DataPassHelper *helper) { ... }
__global__ void APD_BlackPixelUpdateStrong(int iter, DataPassHelper *helper) { ... }
__global__ void APD_RedPixelUpdateStrong(int iter, DataPassHelper *helper) { ... }
__global__ void APD_BlackPixelUpdateWeak(int iter, DataPassHelper *helper) { ... }
__global__ void APD_RedPixelUpdateWeak(int iter, DataPassHelper *helper) { ... }
__global__ void APD_GetDepthandNormal(DataPassHelper *helper) { ... }
__global__ void APD_BlackPixelFilterStrong(DataPassHelper *helper) { ... }
__global__ void APD_RedPixelFilterStrong(DataPassHelper *helper) { ... }
__global__ void APD_GenNeighbours(DataPassHelper *helper) { ... }
__global__ void APD_NeigbourUpdate(DataPassHelper *helper) { ... }
__global__ void APD_DepthToWeak(DataPassHelper *helper) { ... }
__global__ void APD_LocalRefine(DataPassHelper *helper) { ... }
__global__ void APD_FindNearestStrongPoint(DataPassHelper *helper) { ... }
__global__ void APD_RANSACToGetFitPlane(DataPassHelper *helper) { ... }
```

### Kernel adaptations summary
Every kernel that accesses geometry (projection, depth, correspondence) uses `_APD` dispatch wrappers. Every `curand_uniform` call → `rng_uniform`. Every source pixel lookup → `CheckAndWrapSourcePoint`. Every disparity formula → `DepthToDisparity_APD`.

### Host-callable RunAPDPatchMatch function
```cuda
void RunAPDPatchMatch_GPU(DataPassHelper *helper_cuda, int width, int height,
                          const PatchMatchParams &params, cudaStream_t stream);
```
Launches the APD kernel sequence (same order as APD.cu:2386-2494) using the `APD_*` prefixed kernels, with proper stream assignment for batch processing.

---

## Step 6: Modify `ACMMP.cpp` — Host-Side APD Logic

**File**: `/Dataset/mvs/adp/ACMMP-Spherical/ACMMP.cpp`

### 6a. Compute `c[3]` in `ReadCamera()`
After reading R and t:
```cpp
for (int j = 0; j < 3; ++j) {
    camera.c[j] = -(float)(double(camera.R[0+j])*double(camera.t[0]) +
                           double(camera.R[3+j])*double(camera.t[1]) +
                           double(camera.R[6+j])*double(camera.t[2]));
}
```

### 6b. Add `ReadBinMat` / `WriteBinMat`
Port from APD.cpp:1-48, convert `boost::filesystem::path` → `std::string`.
Binary format: 4-int header (version=1, rows, cols, cv::Mat type) + raw pixel data.
- `weak.bin`: CV_8UC1 (uchar per pixel, values 0=WEAK, 1=STRONG, 2=UNKNOWN)
- `selected_views.bin`: CV_32SC1 (unsigned int per pixel, bitset of selected view indices)
- `depths.dmb`: CV_32FC1
- `normals.dmb`: CV_32FC3

### 6c. `APDInputInitialization` method
Port from APD.cpp:399-583 adapted to ACMMP-Spherical conventions:
- Read images as `.png` (ACMMP) not `.jpg` (APD) — check which exists
- For SPHERE cameras: scale `params[1]` (cx) and `params[2]` (cy) alongside K matrix when `scale_size != 1`
- For SPHERE cameras: also scale `params[0]` (f) if it's used — **verify**: grep shows `params[0]` is only used in `colmap2mvsnet_acm.py` for writing cam files, NOT in CUDA code. SphericalLUT uses `params[1]` (cx) and `params[2]` (cy) exclusively. So `params[0]` doesn't need scaling for runtime. Scale it anyway for consistency.
- When `state != FIRST_INIT && use_APD`: read `weak.bin`, build `neighbours_map_host`, count `weak_count`
- When `!use_APD`: set all pixels STRONG, `weak_count = 0`
- When `state != FIRST_INIT`: read previous `depths.dmb`, `normals.dmb`, `selected_views.bin`

### 6d. `APDCudaSpaceInitialization` method
Sets up GPU buffers in ProblemGPUResources and populates DataPassHelper.

### 6e. Remove `RunJBU` function (no longer needed)

---

## Step 7: Modify `BatchACMMP.h` / `BatchACMMP.cu`

**Files**: `/Dataset/mvs/adp/ACMMP-Spherical/BatchACMMP.h`, `BatchACMMP.cu`

### 7a. Extend `ProblemGPUResources` with APD buffers
```cpp
// APD buffers — ALL owned here, not in ACMMP class
uchar *weak_info_cuda = nullptr;
uchar *weak_reliable_cuda = nullptr;
short2 *weak_nearest_strong_cuda = nullptr;
short2 *neighbours_cuda = nullptr;      // size: width*height*NEIGHBOUR_NUM (worst case)
int *neighbours_map_cuda = nullptr;
float4 *fit_plane_hypotheses_cuda = nullptr;
uchar *view_weight_cuda = nullptr;
DataPassHelper *helper_cuda = nullptr;
PatchMatchParams *params_dev_cuda = nullptr;
```

### 7b. Weak buffer sizing: allocate full w*h
For `neighbours_cuda`: allocate `width * height * NEIGHBOUR_NUM * sizeof(short2)` (worst case all pixels weak). This is ~36 bytes/pixel at 1000x1000 = 36MB — acceptable for GPU. NO guard/realloc needed.

### 7c. Update `allocate()` / `deallocate()` with APD buffers

### 7d. Replace `processProblemOnStream` — APD pipeline
New pipeline per problem per round:
```
APDInputInitialization → APDCudaSpaceInitialization → RunAPDPatchMatch_GPU → write outputs
```

### 7e. Output paths: `APD/XXXXXXXX/` instead of `ACMMP/2333_XXXXXXXX/`

---

## Step 8: Modify `main.cpp` — APD Round-Based Pipeline

**File**: `/Dataset/mvs/adp/ACMMP-Spherical/main.cpp`

Replace multi-scale loop (lines 682-741) with APD round-based pipeline:

```cpp
int round_num = ComputeRoundNum(dense_folder, problems);

int iteration = 0;
for (int i = 0; i < round_num; ++i) {
    int scale = (int)pow(2, round_num - 1 - i);

    // Phase 1: FIRST_INIT (round 0) or REFINE_INIT (round 1+)
    for (auto &p : problems) {
        p.scale_size = scale;
        p.iteration = iteration;
    }
    PatchMatchParams params;
    if (i == 0) {
        params.state = FIRST_INIT; params.use_APD = false;
    } else {
        params.state = REFINE_INIT; params.use_APD = true;
        params.ransac_threshold = 0.01f - i * 0.00125f;
        params.rotate_time = min((int)pow(2, i), 4);
    }
    params.geom_consistency = false;
    params.weak_peak_radius = 6;
    ProcessAllProblemsAPD(dense_folder, problems, params, use_batching);
    iteration++;

    // Phase 2: 3x REFINE_ITER
    for (int j = 0; j < 3; ++j) {
        params.state = REFINE_ITER;
        params.geom_consistency = true;
        params.weak_peak_radius = max(4 - 2*j, 2);
        ProcessAllProblemsAPD(dense_folder, problems, params, use_batching);
        iteration++;
    }
}

RunFusionCuda(dense_folder, problems, /*geom_consistency=*/true);
```

Keep: `InitializeLUTsForAllResolutions` (adapt resolution computation for round-based scale factors).
Keep: `GenerateSampleList`, `ProcessProblemsInParallel`/`WithMode` (adapted for APD params).
Remove: `ComputeMultiScaleSettings`, `JointBilateralUpsampling`.

---

## Step 9: Minor Modifications

### 9a. `FusionGPU.cu` / `FusionGPU.h`
- Update path pattern: `APD/XXXXXXXX/` instead of `ACMMP/2333_XXXXXXXX/`
- Read depth files from APD output (`depths.dmb` written by WriteBinMat)
- Optionally weight by weak info (WEAK pixels → lower fusion confidence)

### 9b. `CMakeLists.txt`
- Add `APD_kernels.cu` to `CUDA_SOURCES`
- Add `APD_device.cuh` to `HEADERS`
- Keep `CUDA::curand` (still needed for `curand_Philox4x32_10` used by `rng_uniform`)
- Remove JBU-related dead code if compiler warns

### 9c. `ACMMP_device.cuh`
**No changes**. The compatibility macros (lines 512-518) remain for ACMMP.cu backward compatibility. `APD_device.cuh` undefs them for APD code.

### 9d. `SphericalLUT_MultiRes.*`, `CompressedDMB.h`, `colmap2mvsnet_acm.py`, `masker.py`
**No changes**.

---

## Step 10: JBU Removal Consistency Check

Files to clean up JBU references:
- `ACMMP.h`: Remove `JBU` class, `JBUParameters`, `JBUTexObj` structs, `RunJBU` declaration
- `ACMMP.cu`: Remove `JBU_cu` kernel (~line 2033)
- `ACMMP.cpp`: Remove `RunJBU` function
- `main.cpp`: Remove `JointBilateralUpsampling` function and all calls
- `main.h`: Remove `JBU_NUM` define, `JointBilateralUpsampling` declaration

---

## File Change Summary

| File | Action | Lines est. |
|------|--------|-----------|
| `main.h` | **Modify** | +30 |
| `ACMMP.h` | **Modify** | +60, -40 (JBU removal) |
| `APD_device.cuh` | **Create** | ~300 |
| `APD_kernels.cu` | **Create** | ~2500 (ported from APD.cu) |
| `ACMMP.cu` | **Modify** | -30 (JBU kernel removal) |
| `ACMMP.cpp` | **Modify** | +400 (APD init, ReadBinMat), -50 (JBU) |
| `BatchACMMP.h` | **Modify** | +40 |
| `BatchACMMP.cu` | **Modify** | +200 (APD pipeline) |
| `main.cpp` | **Modify** | +80, -100 (replace multi-scale with rounds) |
| `FusionGPU.cu` | **Modify** | +10 (path update) |
| `CMakeLists.txt` | **Modify** | +3 |

---

## Verification

### Build
```bash
docker exec mvs bash -c "cd /Dataset/mvs/adp/ACMMP-Spherical && mkdir -p build && cd build && cmake .. && make -j\$(nproc)"
```

### Test on test100 dataset
```bash
docker exec mvs bash -c "/Dataset/mvs/adp/ACMMP-Spherical/build/ACMMP /Dataset/mvs/test100 --no-batch"
```
Then with batching:
```bash
docker exec mvs bash -c "/Dataset/mvs/adp/ACMMP-Spherical/build/ACMMP /Dataset/mvs/test100"
```

### Expected results
1. Creates `APD/XXXXXXXX/` per reference image
2. Each round writes `depths.dmb`, `normals.dmb`, `weak.bin`, `selected_views.bin`
3. Final fusion produces PLY point cloud
4. No crashes, reasonable depth maps for spherical images

### Incremental checkpoints
1. **After Steps 2-3**: Compile check — existing ACMMP pipeline still builds (data structures extended, JBU removed)
2. **After Steps 4-5**: Compile check — `APD_kernels.cu` + `APD_device.cuh` build successfully
3. **After Steps 6-8**: Full pipeline test with `--no-batch`
4. **After Step 7**: Batch mode test
5. **After Step 9**: Fusion test — verify PLY output

### Key spherical validation points
- X-wrapping at left/right boundary (pixel 0 ↔ width) — fmodf-based, handles >±w
- Pole region handling (latitude > ±80°) — ACMMP's IsNearPole already masks these
- GenNeighbours crossing image boundary — x-wrap applied to neighbor search
- Disparity calculation — angular formula validated against test data
- Near-pole DepthToWeak/LocalRefine — pole pixels already marked UNKNOWN by mask
