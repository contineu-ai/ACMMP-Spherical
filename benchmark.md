# ACMMP-Spherical Performance Benchmark

## Test Configuration
- **GPU:** RTX 4090 (24GB VRAM, SM 89, 128 SMs)
- **Test dataset:** 100 consecutive images from middle of 1366-image dataset (orig IDs 633-732)
- **Image resolution:** 2400x1200 equirectangular
- **Multi-scale:** 3 scales (600x300 -> 1200x600 -> 2400x1200)
- **Pipeline per scale:** Planar prior + 2x geom consistency (batch mode), JBU upsampling between scales
- **Avg source views per image:** 16.1 (max 21)
- **Build container:** `mvs` Docker (CUDA 12.9, GCC 11.4, OpenCV 4.5.4)

## Optimization Log

| Run | Description | Wall Time | Speedup | Key Change |
|-----|-------------|-----------|---------|------------|
| 0   | Baseline    | 3m 33.9s  | 1.00x   | 8 streams, 542MB/prob |
| 1   | Philox RNG  | 3m 29.8s  | 1.02x   | 4-byte RNG, 385MB/prob |
| 2   | +21 streams +zlib3 | 2m 43.8s | 1.31x | 21 streams, zlib level 3 |
| 4   | +early term +fused kernels +parallel JBU | 2m 24.6s | 1.49x | All Phase 2+3+5 |

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

---

## Time Budget Analysis (Run 4)
| Component | Time | % of Total |
|-----------|------|------------|
| Scale 0 GPU (3 passes) | 55s | 38% |
| Scale 1 GPU (3 passes) | 23s | 16% |
| Scale 2 GPU (3 passes) | 18s | 12% |
| Fusion | 20s | 14% |
| Overhead (JBU, batch setup, I/O) | ~29s | 20% |
| **Total** | **145s** | **100%** |

**Bottleneck:** Scale 0 (2400x1200) dominates. 21 concurrent streams are compute-bound at full resolution.

---

## Changes Applied
1. **Philox RNG** (`ACMMP.h`, `ACMMP.cu`, `BatchACMMP.h/.cu`): `RNGState` (4 bytes) replaces `curandState` (48 bytes). Saves 127MB/problem.
2. **Dynamic concurrency** (`BatchACMMP.cu`): `multiProcessorCount/6` = 21 streams on RTX 4090. Cap 24.
3. **zlib level 3** (`CompressedDMB.h`): Compression 6→3.
4. **Early termination** (`ACMMP.cu`): Skip converged pixels (cost<0.1, delta<0.002) in iterations >0.
5. **Fused post-processing** (`ACMMP.cu`): `PostProcessKernel` + `AllPixelFilter` replaces 3 kernels.
6. **Parallel JBU** (`main.cpp`): OpenMP `#pragma omp parallel for schedule(dynamic)` on JBU loop.
