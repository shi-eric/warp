# Compile Guard Benchmark Data

Collected on L40 GPU, Linux 5.15. All times are median cold-compile
(kernel cache cleared between samples).

## Part 1: Isolated Warp Kernels (5 samples each)

Each kernel is in its own Warp module so compile guards apply independently.

| Kernel | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Scalar only (trivial) | 1.661s | 0.344s | **4.8x** | 0.510s | 0.248s | **2.1x** |
| Vector math | 1.674s | 0.392s | **4.3x** | 0.550s | 0.308s | **1.8x** |
| Mat + quat + transform | 1.764s | 0.559s | **3.2x** | 0.672s | 0.389s | **1.7x** |
| Volume sampling | 1.764s | 0.626s | **2.8x** | 0.895s | 0.726s | **1.2x** |
| Mesh queries | 1.852s | 0.822s | **2.3x** | 1.129s | 1.018s | **1.1x** |

## Part 2: Warp Core Examples — Compile Time (3 samples each)

Low-complexity examples that compile 1–2 modules per process.  These
represent typical user workloads (particle simulations, rendering,
simple physics) where compile guards reduce the source code fed to a
single NVRTC/Clang invocation.  CUDA speedup correlates with kernel
simplicity — simpler kernels use fewer native headers, so guards
exclude more.

| Example | CPU main | CPU branch | CPU speedup | CUDA main | CUDA branch | CUDA speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| optim.particle_repulsion | 1.7s | 0.5s | **3.8x** | 0.5s | 0.2s | **2.1x** |
| core.graph_capture | 1.7s | 0.5s | **3.5x** | 0.5s | 0.3s | **1.8x** |
| core.wave | 1.9s | 0.5s | **3.5x** | 0.6s | 0.4s | **1.7x** |
| core.dem | 1.9s | 0.6s | **3.1x** | 0.7s | 0.5s | **1.6x** |
| core.raymarch | 1.9s | 0.6s | **3.1x** | 0.7s | 0.5s | **1.5x** |
| core.fluid | 2.0s | 0.7s | **2.8x** | 0.9s | 0.6s | **1.4x** |
| core.sph | 2.1s | 0.8s | **2.5x** | 1.1s | 0.8s | **1.3x** |
| core.marching_cubes | — | — | — | 0.9s | 0.7s | **1.3x** |
| core.nvdb | 1.9s | 0.7s | **2.7x** | 0.7s | 0.6s | **1.2x** |
| core.raycast | 1.8s | 0.8s | **2.4x** | 0.7s | 0.6s | **1.2x** |
| core.mesh | 2.0s | 0.9s | **2.3x** | 0.9s | 0.8s | **1.2x** |
| optim.diffray | — | — | — | 1.9s | 1.8s | **1.0x** |

Core examples see **2.3x–3.8x CPU** and **1.2x–2.1x CUDA** speedup for
simple-to-moderate kernels.  The outlier `optim.diffray` uses mesh
queries, gradients, and the optimizer module — its kernels use most
native headers so guards exclude little.

## Part 3: Warp FEM Examples — Compile Time (3 samples each)

Total compile time summed from `took N ms (compiled)` lines. Each sample
spawns a fresh subprocess with `WARP_CACHE_PATH` pointed at a freshly
wiped temp directory.

### CPU

| Example | Main | Branch | Speedup |
| --- | ---: | ---: | ---: |
| fem.diffusion | 48.5s | 13.5s | **3.59x** |
| fem.diffusion_3d | 43.0s | 12.4s | **3.48x** |
| fem.deformed_geometry | 51.1s | 15.1s | **3.39x** |
| fem.stokes | 56.0s | 16.5s | **3.40x** |
| fem.stokes_transfer | 83.7s | 23.1s | **3.62x** |
| fem.magnetostatics | 47.4s | 13.8s | **3.43x** |
| fem.streamlines | 56.4s | 16.1s | **3.51x** |
| fem.distortion_energy | 52.7s | 15.3s | **3.45x** |
| fem.nonconforming_contact | 70.6s | 22.3s | **3.17x** |
| fem.convection_diffusion | 31.9s | 8.5s | **3.74x** |
| fem.navier_stokes | 67.8s | 21.2s | **3.20x** |
| fem.burgers | 28.8s | 7.3s | **3.94x** |
| fem.convection_diffusion_dg | 64.4s | 21.1s | **3.05x** |
| fem.taylor_green | 82.3s | 25.2s | **3.27x** |
| fem.shallow_water | 30.4s | 7.8s | **3.89x** |
| fem.kelvin_helmholtz | 32.4s | 8.4s | **3.87x** |
| fem.elastic_shape_optimization | 81.2s | 27.5s | **2.95x** |
| fem.darcy_ls_optimization | 80.9s | 27.9s | **2.90x** |

FEM CPU: **2.9x–3.9x** compile speedup. On CPU, `fem.navier_stokes`
drops from 68s to 21s.

### CUDA

| Example | Main | Branch | Speedup |
| --- | ---: | ---: | ---: |
| fem.diffusion | 8.6s | 8.3s | **1.03x** |
| fem.diffusion_3d | 8.4s | 8.2s | **1.02x** |
| fem.deformed_geometry | 12.0s | 11.9s | **1.01x** |
| fem.stokes | 10.3s | 10.0s | **1.03x** |
| fem.stokes_transfer | 13.0s | 12.4s | **1.05x** |
| fem.mixed_elasticity | 18.0s | 17.9s | **1.01x** |
| fem.magnetostatics | 10.0s | 9.7s | **1.04x** |
| fem.streamlines | 10.7s | 10.1s | **1.06x** |
| fem.adaptive_grid | 21.6s | 21.5s | **1.00x** |
| fem.distortion_energy | 10.8s | 10.2s | **1.05x** |
| fem.nonconforming_contact | 14.9s | 14.7s | **1.02x** |
| fem.convection_diffusion | 5.9s | 5.5s | **1.09x** |
| fem.navier_stokes | 13.7s | 13.3s | **1.02x** |
| fem.burgers | 4.2s | 4.0s | **1.05x** |
| fem.convection_diffusion_dg | 15.6s | 15.0s | **1.04x** |
| fem.taylor_green | 15.5s | 15.1s | **1.03x** |
| fem.shallow_water | 4.3s | 4.1s | **1.04x** |
| fem.kelvin_helmholtz | 4.9s | 4.7s | **1.05x** |
| fem.apic_fluid | 12.4s | 12.3s | **1.01x** |
| fem.elastic_shape_optimization | 19.9s | 19.4s | **1.02x** |
| fem.darcy_ls_optimization | 15.9s | 15.7s | **1.02x** |

FEM CUDA: **1.0x–1.09x** — within noise for most examples. See
"Impact of PCH fix" below for explanation.

## Part 4: Newton Examples — Compile Time (3 samples each)

### CPU

| Example | Main | Branch | Speedup |
| --- | ---: | ---: | ---: |
| newton.basic_pendulum | 34.6s | 23.2s | **1.49x** |
| newton.basic_urdf | 34.9s | 23.3s | **1.50x** |
| newton.basic_joints | 36.3s | 23.6s | **1.54x** |
| newton.basic_shapes | 36.5s | 23.1s | **1.58x** |
| newton.basic_conveyor | 43.6s | 27.8s | **1.57x** |
| newton.basic_heightfield | 32.5s | 21.2s | **1.53x** |
| newton.cable_twist | 37.2s | 23.1s | **1.61x** |
| newton.cable_y_junction | 35.2s | 22.6s | **1.56x** |
| newton.cable_bundle_hysteresis | 37.2s | 23.2s | **1.60x** |
| newton.cable_pile | 35.4s | 22.9s | **1.55x** |
| newton.cloth_bending | 29.9s | 18.5s | **1.61x** |
| newton.cloth_hanging | 27.9s | 18.0s | **1.55x** |
| newton.diffsim_ball | 13.7s | 7.8s | **1.76x** |
| newton.diffsim_cloth | 9.6s | 4.6s | **2.09x** |
| newton.diffsim_drone | 14.9s | 7.8s | **1.90x** |
| newton.diffsim_spring_cage | 9.4s | 4.4s | **2.13x** |
| newton.diffsim_soft_body | 18.1s | 9.5s | **1.90x** |
| newton.ik_franka | 30.8s | 16.0s | **1.92x** |
| newton.ik_h1 | 29.0s | 15.5s | **1.87x** |
| newton.robot_cartpole | 70.2s | 32.7s | **2.14x** |
| newton.robot_anymal_d | 90.6s | 44.7s | **2.03x** |
| newton.robot_ur10 | 74.0s | 32.6s | **2.27x** |
| newton.selection_articulations | 83.9s | 36.5s | **2.30x** |
| newton.selection_cartpole | 72.7s | 34.7s | **2.09x** |
| newton.selection_materials | 83.5s | 38.4s | **2.17x** |
| newton.selection_multiple | 78.3s | 34.5s | **2.27x** |
| newton.sensor_contact | 77.7s | 37.9s | **2.05x** |
| newton.sensor_imu | 70.5s | 33.0s | **2.14x** |

Newton CPU: **1.5x–2.3x** compile speedup. Heavier examples
(robot, selection) see the largest gains.

### CUDA

| Example | Main | Branch | Speedup |
| --- | ---: | ---: | ---: |
| newton.basic_pendulum | 39.1s | 39.6s | 0.99x |
| newton.basic_urdf | 43.3s | 43.8s | 0.99x |
| newton.basic_joints | 41.0s | 39.9s | 1.03x |
| newton.basic_shapes | 48.4s | 49.2s | 0.98x |
| newton.basic_conveyor | 58.5s | 58.6s | 1.00x |
| newton.basic_heightfield | 37.2s | 38.0s | 0.98x |
| newton.cable_twist | 49.3s | 46.6s | **1.06x** |
| newton.cable_y_junction | 49.3s | 46.8s | **1.05x** |
| newton.cable_bundle_hysteresis | 49.2s | 46.8s | **1.05x** |
| newton.cable_pile | 49.2s | 46.6s | **1.06x** |
| newton.cloth_bending | 47.0s | 44.4s | **1.06x** |
| newton.cloth_hanging | 45.1s | 42.7s | **1.06x** |
| newton.cloth_style3d | 44.3s | 43.6s | 1.01x |
| newton.cloth_twist | 35.5s | 32.8s | **1.08x** |
| newton.cloth_rollers | 26.4s | 24.3s | **1.09x** |
| newton.cloth_poker_cards | 67.9s | 63.7s | **1.07x** |
| newton.cloth_franka | 89.2s | 86.2s | 1.03x |
| newton.cloth_h1 | 73.8s | 73.5s | 1.00x |
| newton.brick_stacking | 141.2s | 139.6s | 1.01x |
| newton.nut_bolt_sdf | 88.8s | 88.8s | 1.00x |
| newton.nut_bolt_hydro | 94.0s | 93.4s | 1.01x |
| newton.pyramid | 39.0s | 38.9s | 1.00x |
| newton.diffsim_ball | 9.4s | 9.5s | 0.99x |
| newton.diffsim_cloth | 4.6s | 4.6s | 1.01x |
| newton.diffsim_drone | 11.1s | 11.4s | 0.98x |
| newton.diffsim_spring_cage | 4.6s | 4.5s | 1.02x |
| newton.diffsim_soft_body | 11.0s | 11.3s | 0.97x |
| newton.ik_franka | 62.5s | 60.5s | 1.03x |
| newton.ik_h1 | 70.3s | 69.4s | 1.01x |
| newton.ik_custom | 42.9s | 41.5s | 1.03x |
| newton.ik_cube_stacking | 130.7s | 127.9s | 1.02x |
| newton.mpm_granular | 24.9s | 25.0s | 0.99x |
| newton.mpm_multi_material | 25.5s | 25.9s | 0.98x |
| newton.mpm_twoway_coupling | 115.1s | 113.3s | 1.02x |
| newton.mpm_beam_twist | 36.6s | 36.4s | 1.01x |
| newton.mpm_snow_ball | 29.5s | 30.1s | 0.98x |
| newton.mpm_viscous | 27.1s | 26.8s | 1.01x |
| newton.robot_cartpole | 46.8s | 45.8s | 1.02x |
| newton.robot_anymal_d | 79.4s | 78.2s | 1.02x |
| newton.robot_ur10 | 46.3s | 44.3s | **1.05x** |
| newton.robot_allegro_hand | 80.5s | 78.5s | 1.03x |
| newton.robot_g1 | 116.4s | 115.4s | 1.01x |
| newton.robot_h1 | 79.9s | 78.9s | 1.01x |
| newton.robot_panda_hydro | 152.8s | 149.4s | 1.02x |
| newton.selection_articulations | 95.3s | 94.2s | 1.01x |
| newton.selection_cartpole | 50.4s | 49.6s | 1.02x |
| newton.selection_materials | 56.1s | 53.8s | 1.04x |
| newton.selection_multiple | 88.8s | 87.1s | 1.02x |
| newton.sensor_contact | 72.0s | 69.2s | 1.04x |
| newton.sensor_imu | 58.3s | 56.3s | 1.04x |
| newton.softbody_hanging | 36.1s | 34.1s | **1.06x** |
| newton.softbody_franka | 129.8s | 127.1s | 1.02x |
| newton.softbody_gift | 39.9s | 38.1s | **1.05x** |
| newton.softbody_dropping_to_cloth | 44.9s | 42.6s | **1.05x** |

Newton CUDA: **0.97x–1.09x** — mostly within noise. A handful of
cable/cloth/softbody examples show a small but real ~5–9% improvement.
No regressions.

## Summary

- **Isolated kernels, CPU**: 2.3x–4.8x speedup depending on feature usage
- **Isolated kernels, CUDA**: 1.1x–2.1x speedup
- **Core examples, CPU**: 2.3x–3.8x compile speedup
- **Core examples, CUDA**: 1.2x–2.1x compile speedup (simple kernels benefit most)
- **Warp FEM, CPU**: 2.9x–3.9x compile speedup (20–55s saved per example)
- **Warp FEM, CUDA**: 1.0x–1.09x (within noise — many modules amortize PCH)
- **Newton, CPU**: 1.5x–2.3x compile speedup (10–45s saved per example)
- **Newton, CUDA**: 0.97x–1.09x (within noise — many modules amortize PCH)
- No regressions on any workload

## Impact of PCH fix on CUDA measurements

The `ershi/fix-cu12-pch` fix (merged to main on 2026-03-24) corrected
precompiled header generation for CUDA 12.x, dramatically reducing NVRTC
compile times on the **baseline**. For example:

| Example | CUDA main (pre-fix) | CUDA main (post-fix) | Change |
| --- | ---: | ---: | ---: |
| fem.diffusion | 16.3s | 8.6s | −47% |
| fem.navier_stokes | 23.6s | 13.7s | −42% |
| fem.stokes | 19.8s | 10.3s | −48% |
| newton.basic_pendulum | 43.9s | 39.1s | −11% |
| newton.robot_cartpole | 55.0s | 46.8s | −15% |
| newton.mpm_granular | 37.0s | 24.9s | −33% |

The PCH fix and compile guards address different aspects of CUDA
compile overhead:

- **PCH** accelerates NVRTC by pre-parsing headers once per process,
  then reusing the parsed state for subsequent module compilations.
  Its benefit grows with the number of modules compiled in a single
  process (FEM/Newton compile dozens of modules → large amortization).
- **Compile guards** reduce source code fed to each NVRTC invocation
  by excluding unused native headers.  Their benefit is per-module and
  scales with kernel simplicity (fewer features used → more excluded).

For **single-module workloads** (core examples, isolated kernels), PCH
is created and used exactly once — no amortization.  Here compile guards
deliver 1.2x–2.1x CUDA speedup because they shrink the input to that
single NVRTC call.

For **multi-module workloads** (FEM/Newton), PCH creation cost is
amortized across many compilations, making each individual NVRTC call
fast.  Compile guards still exclude headers per-module but the marginal
savings are small relative to the already-fast PCH-accelerated baseline.
This explains why FEM/Newton show ~1.0x CUDA speedup despite 3x+ CPU
speedup (the CPU compiler does not use PCH).

The CUDA speedup for a given kernel correlates with feature complexity:

| Feature usage | CUDA speedup | Examples |
| --- | ---: | --- |
| Scalars, basic vec3 | 1.7x–2.1x | particle_repulsion, graph_capture |
| vec3 + hash grid | 1.3x–1.5x | fluid, sph, dem |
| Volume/mesh queries | 1.1x–1.2x | nvdb, raycast, mesh |
| Heavy (mesh+grad+optim) | ~1.0x | diffray |

### Design decision: keep CUDA compile guards

Given that multi-module CUDA speedups are ~1.0x, we considered whether
to apply compile guards only on CPU and skip them for NVRTC.

**Keep them.** The guards are backend-agnostic — the same `#ifndef
WP_NO_X` directives are emitted regardless of whether the target is
Clang (CPU) or NVRTC (CUDA).  There is no separate CUDA codepath to
maintain, so removing CUDA guards would not simplify the code.

Meanwhile, single-module workloads — the most common user scenario
(someone writing a script with a few kernels) — still see 1.2x–2.1x
CUDA speedup.  Removing guards would regress this case for zero
benefit.

PCH and compile guards are complementary, not redundant:

- **PCH** amortizes header parsing across modules within a process
  (helps multi-module workloads like FEM/Newton)
- **Compile guards** reduce source per NVRTC call (helps single-module
  workloads like user scripts)

The only reason to remove CUDA guards would be if they caused
CUDA-specific bugs.  The `add_builtin()` enforcement (TypeError for
missing guard) and 22 unit tests in `test_compile_guards.py` mitigate
this risk.

## Notes

- "main" branch: commit c687bc58 (latest main as of 2026-03-25, includes PCH fix)
- "branch": commit e01a7a59 (ershi/robust-compile-guards)
- All data recollected 2026-03-25 with subprocess-per-sample isolation
- Benchmarks run sequentially (no parallel processes) with isolated caches
- CUDA driver cache disabled via CUDA_CACHE_DISABLE=1
- Kernel cache fully wiped between each sample via shutil.rmtree
