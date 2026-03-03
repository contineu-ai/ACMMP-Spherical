# ACMMP-Spherical Performance Benchmark

## Test Configuration — Dataset v1 (Runs 0–5)
- **GPU:** RTX 4090 (24GB VRAM, SM 89, 128 SMs)
- **Test dataset:** 100 consecutive images from middle of 1366-image dataset (orig IDs 633-732)
- **Image resolution:** 2400x1200 equirectangular
- **Multi-scale:** 3 scales (600x300 -> 1200x600 -> 2400x1200)
- **Pipeline per scale:** Planar prior + 2x geom consistency (batch mode), JBU upsampling between scales
- **Avg source views per image:** 16.1 (max 21)
- **Build container:** `mvs` Docker (CUDA 12.9, GCC 11.4, OpenCV 4.5.4)

## Test Configuration — Dataset v2 (Runs 7+)
- Same images, resolution, GPU, container
- **Avg source views per image:** 49.5 (max 99) — pair.txt regenerated 2026-03-03
- ~3x more NCC evaluations per pixel vs v1; timings NOT directly comparable to Runs 0–5

## Optimization Log

| Run | Description | Wall Time | Speedup | Dataset | Key Change |
|-----|-------------|-----------|---------|---------|------------|
| 0   | Baseline    | 3m 33.9s  | 1.00x   | v1 | 8 streams, 542MB/prob |
| 1   | Philox RNG  | 3m 29.8s  | 1.02x   | v1 | 4-byte RNG, 385MB/prob |
| 2   | +21 streams +zlib3 | 2m 43.8s | 1.31x | v1 | 21 streams, zlib level 3 |
| 4   | +early term +fused kernels +parallel JBU | 2m 24.6s | 1.49x | v1 | All Phase 2+3+5 |
| 5a  | +coalesced checkerboard (HDD) | 2m 24.5s | 1.49x | v1 | Stride-2 cols, no measurable GPU change |
| 5b  | +coalesced checkerboard (SSD) | 2m 20.3s | 1.53x | v1 | SSD I/O saves ~4s overhead |
| 7   | Baseline (committed code) | 3m 14.3s | 1.00x | v2 | New baseline for v2 dataset |
| 7a  | +all batch optimizations | 3m 14.8s | 1.00x | v2 | Timing-neutral (within noise) |
| 5x  | Sub-warp NCC (4 threads/pixel) | 3m 08.1s | — | v1 | **DISABLED**: regression |

**Note:** Run 6 was invalidated — the apparent "GEOM regression" was caused by dataset v2's 3x higher source view count, not code changes. Register analysis confirmed identical REG:128 STACK:3312 for baseline and optimized kernels.

## Runs

### Run 0: Baseline
- **Wall time:** 3m 33.9s (213.9s)
- **Memory:** 542MB/problem, 8 streams
- **Output:** 30.5M points

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 11s | 11s | 12s |
| 1 (1200x600) | 13s | 13s | 13s |
| 0 (2400x1200) | 27s | 24s | 24s |
| **Fusion** | 17s | | |
| **GPU subtotal** | **148s** | **Overhead** | **~66s** |

### Run 2: Philox RNG + 21 streams + zlib 3
- **Wall time:** 2m 43.8s (163.8s) — **1.31x**
- **Memory:** 385MB/problem, 21 streams
- **Output:** 31.4M points

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 8s | 8s | 7s |
| 0 (2400x1200) | 23s | 17s | 17s |
| **Fusion** | 17s | | |
| **GPU subtotal** | **98s** | **Overhead** | **~66s** |

### Run 4: All optimizations (Phase 2 + 3 + 5)
- **Wall time:** 2m 24.6s (144.6s) — **1.49x**
- **Memory:** 385MB/problem, 21 streams
- **Output:** 31.0M points

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 8s | 7s | 8s |
| 0 (2400x1200) | 23s | 16s | 16s |
| **Fusion** | 20s | | |
| **GPU subtotal** | **96s** | **Overhead** | **~29s** |

- Parallel JBU saved ~37s of overhead (66→29s)
- Early termination saves ~2s on GEOM passes at full resolution
- Fused post-processing eliminates 1 kernel launch per pass

### Run 5a: Coalesced checkerboard (HDD)
- **Wall time:** 2m 24.5s (144.5s) — **1.49x**
- **Memory:** 385MB/problem, 21 streams
- **Output:** 31.1M points

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 7s | 8s | 8s |
| 0 (2400x1200) | 23s | 16s | 16s |
| **Fusion** | 21s | | |
| **GPU subtotal** | **96s** | **Overhead** | **~28s** |

- Coalesced remap: warp threads now access same row with stride-2 columns (vs alternating rows)
- No measurable GPU improvement: 21 streams already saturate compute at full resolution
- Identical point count and quality

### Run 5b: Coalesced checkerboard (SSD — /home/kanao/mvs/test100/)
- **Wall time:** 2m 20.3s (140.3s) — **1.53x**
- **Memory:** 385MB/problem, 21 streams
- **Output:** 31.1M points

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 8s | 7s | 7s |
| 0 (2400x1200) | 22s | 15s | 15s |
| **Fusion** | 21s | | |
| **GPU subtotal** | **92s** | **Overhead** | **~27s** |

- SSD saves ~4s in I/O overhead (batch setup, image loading, CDMB writes)
- Scale 0 GPU time may be slightly better (~3s) — possibly within noise

### Run 7: Dataset v2 Baseline (committed code, HDD)
- **Wall time:** 3m 14.3s (194.3s)
- **Memory:** 385MB/problem, 21 streams
- **Output:** 31.0M points
- **Dataset:** v2 (49.5 avg source views)

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 10s | 8s | 8s |
| 0 (2400x1200) | 41s | 27s | 27s |
| **Fusion** | 21s | | |
| **GPU subtotal** | **160s** | **Overhead** | **~34s** |

### Run 7a: All batch optimizations (Dataset v2, HDD)
- **Wall time:** 3m 14.8s (194.8s) — **1.00x vs Run 7** (within noise)
- **Memory:** 385MB/problem, 21 streams
- **Output:** 31.0M points
- **Dataset:** v2 (49.5 avg source views)

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 10s | 8s | 9s |
| 0 (2400x1200) | 41s | 27s | 27s |
| **Fusion** | 21s | | |
| **GPU subtotal** | **161s** | **Overhead** | **~34s** |

**Changes in this run (all timing-neutral):**
- LRU image/camera/mask cache (50 entries, thread-safe shared_ptr)
- Texture object reuse (create once in allocate(), skip per-problem recreation)
- GPU early masking via template specialization: `<UseMask=true>` for PLANAR+mask, `<UseMask=false>` for GEOM (zero overhead)
- Batch planar prior (2-pass pipeline: Pass 1 → ExtractSupportPoints → Delaunay → Rasterize → ComputePriorPlanes → Pass 2)
- Eliminate intermediate host memcpy (pinned buffer direct read)
- JBU stream serialization (non-blocking stream, async copies)

**Register analysis:** `cuobjdump` confirms identical register/stack usage for all template variants:
- `BlackPixelUpdate<false>`: REG:128 STACK:3312 (same as committed baseline)
- `BlackPixelUpdate<true>`: REG:128 STACK:3312
- `RedPixelUpdate<false>/<true>`: REG:128 STACK:3312

### Run 5x: Sub-warp NCC — DISABLED (regression)
- **Wall time:** 3m 08.1s (188.1s) — **REGRESSION** (1.14x vs Run 0)
- **Output:** 31.0M points

| Scale | PLANAR | GEOM | GEOM+MULTI |
|-------|--------|------|------------|
| 2 (600x300) | 6s | 6s | 6s |
| 1 (1200x600) | 10s | 8s | 8s |
| 0 (2400x1200) | 41s | 27s | 27s |
| **Fusion** | 20s | | |
| **GPU subtotal** | **139s** | **Overhead** | **~29s** |

- 4 threads/pixel via `__shfl_sync(width=4)` distributed source views across sub-lanes
- Block (32,8)=256 threads, 8 pixels per warp (vs 512 threads, 256 pixels per warp)
- **Root cause of regression:** 4× fewer pixels in flight per SM → catastrophic texture latency hiding loss
- NCC is dominated by ~1300 texture reads per evaluation; fewer concurrent pixels = more stalls
- Scale 0 (compute-bound at 21 streams) hit hardest: 55s → 95s (+73%)
- Code preserved under `#define USE_SUBWARP_NCC` but disabled by default

---

## Time Budget Analysis (Run 5b, SSD, Dataset v1)
| Component | Time | % of Total |
|-----------|------|------------|
| Scale 0 GPU (3 passes) | 52s | 37% |
| Scale 1 GPU (3 passes) | 22s | 16% |
| Scale 2 GPU (3 passes) | 18s | 13% |
| Fusion | 21s | 15% |
| Overhead (JBU, batch setup, I/O) | ~27s | 19% |
| **Total** | **140s** | **100%** |

**Bottleneck:** Scale 0 (2400x1200) dominates at 37%. GPU is compute-saturated with 21 streams at full resolution — kernel-level optimizations (coalesced access, sub-warp parallelism) provide no measurable benefit. Further gains require algorithmic changes (fewer iterations, reduced patch size, or hierarchical coarse-to-fine convergence).

---

## Changes Applied
1. **Philox RNG** (`ACMMP.h`, `ACMMP.cu`, `BatchACMMP.h/.cu`): `RNGState` (4 bytes) replaces `curandState` (48 bytes). Saves 127MB/problem.
2. **Dynamic concurrency** (`BatchACMMP.cu`): `multiProcessorCount/6` = 21 streams on RTX 4090. Cap 24.
3. **zlib level 3** (`CompressedDMB.h`): Compression 6→3.
4. **Early termination** (`ACMMP.cu`): Skip converged pixels (cost<0.1, delta<0.002) in iterations >0.
5. **Fused post-processing** (`ACMMP.cu`): `PostProcessKernel` + `AllPixelFilter` replaces 3 kernels.
6. **Parallel JBU** (`main.cpp`): OpenMP `#pragma omp parallel for schedule(dynamic)` on JBU loop.
7. **Coalesced checkerboard** (`ACMMP.cu`): Warp threads access same row with stride-2 cols. Grid changed from half-height to half-width. No measurable GPU improvement.
8. **LRU image cache** (`BatchACMMP.h/.cu`): Thread-safe cache with shared_ptr, max 50 entries. Avoids redundant disk reads for overlapping source views.
9. **Texture object reuse** (`BatchACMMP.cu`, `ACMMP.cpp`): Create texture objects once in `allocate()`, skip per-problem recreation.
10. **Eliminate intermediate memcpy** (`ACMMP.cu`, `BatchACMMP.cu`): Batch mode reads directly from pinned buffers, skips class-level host copies.
11. **GPU early masking** (`ACMMP.cu`): Template-specialized `<UseMask=true/false>` on all 9 kernels/device functions. GEOM phases use `<false>` (zero overhead). PLANAR+mask phases use `<true>`. PostProcess/Filter use `<true>` when masks exist. No register pressure impact (confirmed by cuobjdump).
12. **Batch planar prior** (`ACMMP.cu`, `BatchACMMP.cu`): 2-pass pipeline with GPU kernels (ExtractSupportPoints, RasterizeTriangles, ComputePriorPlanes) + CPU Delaunay.
13. **JBU stream serialization** (`ACMMP.cpp`): Non-blocking stream, async copies, stream-ordered execution.

### Attempted but disabled
14. **Sub-warp NCC** (`ACMMP.cu`, `#define USE_SUBWARP_NCC`): 4 threads/pixel via `__shfl_sync(width=4)`. Block (32,8), REG:128, STACK:3328. **Caused 30% regression** — fewer pixels in flight per SM killed texture latency hiding. Code preserved but disabled.
