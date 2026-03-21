# Compile Guard Benchmark Data

Collected on L40 GPU, Linux 5.15. All times are median cold-compile
(kernel cache cleared between samples).

## Part 1: Isolated Warp Kernels (5 samples each)

Each kernel is in its own Warp module so compile guards apply independently.

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only (trivial) | 1.617s | 0.329s | **4.9x** | 0.410s | 0.146s | **2.8x** |
| Vector math | 1.639s | 0.380s | **4.3x** | 0.447s | 0.192s | **2.3x** |
| Noise + random | 1.663s | 0.432s | **3.8x** | 0.421s | 0.190s | **2.2x** |
| Mat + quat + transform | 1.706s | 0.536s | **3.2x** | 0.492s | 0.288s | **1.7x** |
| Volume sampling | 1.721s | 0.611s | **2.8x** | 0.775s | 0.614s | **1.3x** |
| Mesh queries | 1.815s | 0.824s | **2.2x** | 1.025s | 0.912s | **1.1x** |

## Part 2: Warp Examples — CUDA Compile Time (3 samples each)

Compile time only (not wall-clock). Each example run via
`python -m warp.examples.<name> --headless --num-frames 1`.

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| fem.navier_stokes | 67.1s | 22.6s | **2.97x** | 23.7s | 15.7s | **1.51x** |
| fem.stokes | 55.6s | 17.8s | **3.12x** | 19.2s | 12.0s | **1.60x** |
| fem.deformed_geometry | 50.6s | 16.1s | **3.14x** | 19.9s | 13.4s | **1.49x** |
| fem.diffusion_3d | 42.7s | 13.1s | **3.26x** | 15.2s | 9.2s | **1.65x** |
| fem.convection_diffusion | 31.6s | 9.4s | **3.36x** | 11.0s | 6.8s | **1.62x** |
| fem.apic_fluid | — | — | — | 21.5s | 14.5s | **1.48x** |
| fem.mixed_elasticity | — | — | — | 30.0s | 20.7s | **1.45x** |
| optim.diffray | 4.2s | 2.0s | **2.10x** | 2.1s | 1.8s | **1.17x** |

FEM examples see **3.0x–3.4x CPU** and **1.45x–1.65x CUDA** compile speedup.
On CPU, `fem.navier_stokes` drops from 67s to 23s. On CUDA, the same example
drops from 24s to 16s. Two FEM examples (apic_fluid, mixed_elasticity) require
CUDA so CPU data is unavailable.

## Part 3: Newton Physics Engine — Full Module Compilation (3 samples each)

Newton is a real-world Warp consumer with 30+ compiled modules spanning
simulation, geometry, solvers, and math. This measures total cold-compile time
for all Newton modules.

| | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **Newton total (all modules)** | **58.9s** | **22.1s** | **2.7x** | **163.6s** | **86.9s** | **1.9x** |

### Top Newton modules by CPU compile time

| Module | CPU main | CPU branch | Speedup |
| --- | ---: | ---: | ---: |
| `sim.articulation` | 5.52s | 4.36s | 1.3x |
| `geometry.kernels` | 3.82s | 2.85s | 1.3x |
| `geometry.sdf_texture` | 3.13s | 2.25s | 1.4x |
| `geometry.sdf_utils` | 2.57s | 1.63s | 1.6x |
| `geometry.sdf_hydroelastic` | 2.25s | 1.08s | 2.1x |
| `geometry.inertia` | 2.02s | 0.80s | 2.5x |
| `geometry.contact_reduction_global` | 1.94s | 0.78s | 2.5x |
| `geometry.broad_phase_sap` | 1.94s | 0.64s | 3.0x |
| `geometry.broad_phase_nxn` | 1.91s | 0.62s | 3.1x |
| `utils.mesh` | 1.85s | 0.60s | 3.1x |
| `sim.graph_coloring` | 1.68s | 0.39s | 4.3x |
| `geometry.hashtable` | 1.63s | 0.35s | 4.7x |
| `math.spatial` | 1.61s | 0.29s | 5.5x |

Modules using many features (articulation, kernels, sdf_texture) see modest
1.3-1.6x gains since fewer headers can be excluded. Modules using fewer features
(hashtable, spatial, graph_coloring) see 4-5x gains.

### Newton examples — end-to-end CUDA (3 samples each)

Wall-clock time from `python -m newton.examples <name> --viewer null --num-frames 1`.
Includes Python import, module compilation, and 1 simulation frame.

| Example | main wall | branch wall | Speedup | main compile | branch compile |
| --- | ---: | ---: | ---: | ---: | ---: |
| robot_ur10 | 67.9s | 55.0s | **1.23x** | 50.9s | 42.8s |
| mpm_granular | 53.9s | 44.8s | **1.20x** | 36.1s | 27.7s |
| robot_allegro_hand | 101.2s | 86.3s | **1.17x** | 87.8s | 77.1s |
| robot_cartpole | 55.7s | 49.1s | **1.13x** | 50.6s | 44.1s |
| robot_panda_hydro | 163.8s | 150.6s | **1.09x** | 136.2s | 123.3s |
| softbody_hanging | 42.0s | 38.8s | **1.08x** | 37.4s | 34.3s |
| robot_g1 | 109.8s | 102.7s | **1.07x** | 93.2s | 85.0s |
| diffsim_ball | 14.5s | 13.6s | **1.07x** | 10.6s | 9.7s |
| robot_h1 | 96.6s | 91.2s | **1.06x** | 87.3s | 77.4s |
| robot_anymal_d | 96.3s | 91.0s | **1.06x** | 87.2s | 78.0s |
| cloth_hanging | 57.3s | 54.6s | **1.05x** | 46.4s | 43.2s |
| basic_shapes | 63.7s | 60.8s | **1.05x** | 52.8s | 48.9s |
| basic_joints | 47.1s | 45.0s | **1.05x** | 43.5s | 41.0s |

Newton examples show 1.05x–1.23x wall-time improvements on CUDA. Robot examples
(G1, H1, Anymal, etc.) save 5–15 seconds per cold start. Heavier examples like
`robot_panda_hydro` (164s → 151s) save 13 seconds despite using nearly all
Warp features via the MuJoCo solver.

## Summary

- **Isolated kernels, CPU**: 2.2x–4.9x speedup depending on feature usage
- **Isolated kernels, CUDA**: 1.1x–2.8x speedup
- **Warp FEM examples, CPU**: 3.0x–3.4x compile speedup (20–45s saved per example)
- **Warp FEM examples, CUDA**: 1.45x–1.65x compile speedup (4–10s saved per example)
- **Newton total, CPU**: 58.9s → 22.1s (**2.7x**)
- **Newton total, CUDA**: 163.6s → 86.9s (**1.9x**)
- No regressions on any workload

## Notes

- "main" branch has no compile guard system — all headers always included
- Newton modules compiled via `wp.load_module(module="newton", recursive=True)`
- Warp kernel benchmarks use `_bench_comparison.py`, Newton uses `_bench_newton.py`
- Both scripts clear `wp.clear_kernel_cache()` between samples for cold-compile measurement
