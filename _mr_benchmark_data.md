# Compile Guard Benchmark Data

Collected on L40 GPU, Linux 5.15. All times are median cold-compile
(kernel cache cleared between samples).

## Part 1: Isolated Warp Kernels (5 samples each)

Each kernel is in its own Warp module so compile guards apply independently.

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only (trivial) | 1.639s | 0.338s | **4.9x** | 0.409s | 0.146s | **2.8x** |
| Vector math | 1.675s | 0.387s | **4.3x** | 0.442s | 0.193s | **2.3x** |
| Mat + quat + transform | 1.763s | 0.560s | **3.2x** | 0.484s | 0.290s | **1.7x** |
| Volume sampling | 1.742s | 0.614s | **2.8x** | 0.773s | 0.623s | **1.2x** |
| Mesh queries | 1.848s | 0.813s | **2.3x** | 1.026s | 0.910s | **1.1x** |

## Part 2: Warp FEM Examples — Compile Time (3 samples each)

Total compile time summed from `took N ms (compiled)` lines. Each sample
spawns a fresh subprocess with `WARP_CACHE_PATH` pointed at a freshly
wiped temp directory.

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fem.navier_stokes | 68.0s | 21.1s | **3.22x** | 23.6s | 15.2s | **1.55x** |
| fem.stokes | 56.7s | 16.5s | **3.44x** | 19.1s | 11.6s | **1.65x** |
| fem.deformed_geometry | 51.1s | 15.0s | **3.41x** | 20.1s | 12.9s | **1.56x** |
| fem.diffusion_3d | 43.2s | 12.3s | **3.51x** | 15.1s | 9.0s | **1.68x** |
| fem.convection_diffusion | 31.9s | 8.5s | **3.75x** | 11.0s | 6.4s | **1.72x** |

FEM examples see **3.2x–3.8x CPU** and **1.55x–1.72x CUDA** compile speedup.
On CPU, `fem.navier_stokes` drops from 68s to 21s. On CUDA, the same example
drops from 24s to 15s.

## Summary

- **Isolated kernels, CPU**: 2.3x–4.9x speedup depending on feature usage
- **Isolated kernels, CUDA**: 1.1x–2.8x speedup
- **Warp FEM examples, CPU**: 3.2x–3.8x compile speedup (20–47s saved per example)
- **Warp FEM examples, CUDA**: 1.55x–1.72x compile speedup (5–8s saved per example)
- No regressions on any workload

## Comparison with previous measurements (2026-03-21)

All numbers are branch median compile times. Changes are within noise
unless noted.

### Isolated Kernels

| Kernel | CPU prev → now | CUDA prev → now |
| --- | --- | --- |
| Scalar only | 0.329s → 0.338s (+2.7%) | 0.146s → 0.146s (0%) |
| Vector math | 0.380s → 0.387s (+1.8%) | 0.192s → 0.193s (+0.5%) |
| Mat + quat + transform | 0.536s → 0.560s (+4.5%) | 0.288s → 0.290s (+0.7%) |
| Volume sampling | 0.611s → 0.614s (+0.5%) | 0.614s → 0.623s (+1.5%) |
| Mesh queries | 0.824s → 0.813s (−1.3%) | 0.912s → 0.910s (−0.2%) |

All within sampling noise (< 5%). No regressions.

### FEM Examples

| Example | CPU prev → now | CUDA prev → now |
| --- | --- | --- |
| fem.navier_stokes | 22.6s → 21.1s (−6.6%) | 15.7s → 15.2s (−3.2%) |
| fem.stokes | 17.8s → 16.5s (−7.3%) | 12.0s → 11.6s (−3.3%) |
| fem.deformed_geometry | 16.1s → 15.0s (−6.8%) | 13.4s → 12.9s (−3.7%) |
| fem.diffusion_3d | 13.1s → 12.3s (−6.1%) | 9.2s → 9.0s (−2.2%) |
| fem.convection_diffusion | 9.4s → 8.5s (−9.6%) | 6.8s → 6.4s (−5.9%) |

FEM compile times improved ~6–10% on CPU and ~2–6% on CUDA compared to
the previous measurement. This is attributable to the WP_NO_BACKWARD
compile guard and mat.h split added after the previous measurement.

## Notes

- "main" branch: commit 737e4e58 (latest main as of 2026-03-22)
- "branch": commit 3c805116 (ershi/robust-compile-guards)
- Benchmarks run sequentially (no parallel processes) with isolated caches
- CUDA driver cache disabled via CUDA_CACHE_DISABLE=1
- Kernel cache fully wiped between each sample via shutil.rmtree
