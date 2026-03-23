# Compile-Time Benchmark Comparison

- Baseline: `_benchmark_baseline_all.json`
- Branch: `_benchmark_branch_all.json`


### FEM — cpu

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| fem.diffusion | 48.6s | 13.6s | **3.58x** | -72.1% | improved |
| fem.diffusion_3d | 43.1s | 12.3s | **3.50x** | -71.4% | improved |
| fem.deformed_geometry | 51.3s | 15.1s | **3.40x** | -70.6% | improved |
| fem.stokes | 56.4s | 16.5s | **3.41x** | -70.7% | improved |
| fem.stokes_transfer | 83.7s | 23.3s | **3.59x** | -72.2% | improved |
| fem.mixed_elasticity | N/A | N/A | — | — | both failed |
| fem.magnetostatics | 47.8s | 13.8s | **3.46x** | -71.1% | improved |
| fem.streamlines | 56.3s | 16.2s | **3.47x** | -71.2% | improved |
| fem.adaptive_grid | N/A | N/A | — | — | both failed |
| fem.distortion_energy | 52.8s | 15.4s | **3.43x** | -70.9% | improved |
| fem.nonconforming_contact | 70.8s | 22.4s | **3.16x** | -68.3% | improved |
| fem.convection_diffusion | 32.0s | 8.5s | **3.75x** | -73.3% | improved |
| fem.navier_stokes | 68.3s | 21.3s | **3.20x** | -68.8% | improved |
| fem.burgers | 28.8s | 7.3s | **3.94x** | -74.6% | improved |
| fem.convection_diffusion_dg | 64.3s | 21.0s | **3.06x** | -67.3% | improved |
| fem.taylor_green | 82.5s | 25.4s | **3.25x** | -69.2% | improved |
| fem.shallow_water | 30.7s | 7.9s | **3.91x** | -74.4% | improved |
| fem.kelvin_helmholtz | 32.5s | 8.4s | **3.86x** | -74.1% | improved |
| fem.elastic_shape_optimization | 80.6s | 28.0s | **2.87x** | -65.2% | improved |
| fem.darcy_ls_optimization | 81.1s | 28.9s | **2.81x** | -64.4% | improved |

### FEM — cuda:0

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| fem.diffusion | 16.3s | 9.9s | **1.65x** | -39.3% | improved |
| fem.diffusion_3d | 15.1s | 9.2s | **1.65x** | -39.4% | improved |
| fem.deformed_geometry | 20.2s | 14.2s | **1.42x** | -29.8% | improved |
| fem.stokes | 19.8s | 12.2s | **1.62x** | -38.3% | improved |
| fem.stokes_transfer | 26.4s | 15.5s | **1.70x** | -41.2% | improved |
| fem.mixed_elasticity | 30.3s | 20.9s | **1.45x** | -30.9% | improved |
| fem.magnetostatics | 17.7s | 11.3s | **1.57x** | -36.4% | improved |
| fem.streamlines | 19.2s | 12.0s | **1.61x** | -37.7% | improved |
| fem.adaptive_grid | 32.0s | 24.5s | **1.31x** | -23.5% | improved |
| fem.distortion_energy | 18.9s | 12.1s | **1.56x** | -36.0% | improved |
| fem.nonconforming_contact | 26.0s | 17.4s | **1.50x** | -33.3% | improved |
| fem.convection_diffusion | 11.0s | 6.6s | **1.66x** | -39.9% | improved |
| fem.navier_stokes | 23.6s | 15.7s | **1.51x** | -33.6% | improved |
| fem.burgers | 8.6s | 4.7s | **1.84x** | -45.7% | improved |
| fem.convection_diffusion_dg | 24.9s | 17.6s | **1.41x** | -29.3% | improved |
| fem.taylor_green | 27.4s | 18.0s | **1.52x** | -34.3% | improved |
| fem.shallow_water | 9.0s | 4.7s | **1.90x** | -47.4% | improved |
| fem.kelvin_helmholtz | 9.7s | 5.4s | **1.80x** | -44.5% | improved |
| fem.apic_fluid | 21.3s | 14.4s | **1.48x** | -32.3% | improved |
| fem.elastic_shape_optimization | 32.1s | 22.9s | **1.40x** | -28.6% | improved |
| fem.darcy_ls_optimization | 26.5s | 18.6s | **1.43x** | -30.0% | improved |

### NEWTON — cpu

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| newton.basic_pendulum | 35.7s | 23.7s | **1.51x** | -33.7% | improved |
| newton.basic_urdf | 36.0s | 23.9s | **1.51x** | -33.6% | improved |
| newton.basic_joints | 36.6s | 24.0s | **1.53x** | -34.5% | improved |
| newton.basic_shapes | 36.9s | 23.4s | **1.57x** | -36.4% | improved |
| newton.basic_conveyor | 43.7s | 28.2s | **1.55x** | -35.4% | improved |
| newton.basic_heightfield | 32.5s | 21.9s | **1.48x** | -32.5% | improved |
| newton.cable_twist | 37.5s | 23.6s | **1.59x** | -37.0% | improved |
| newton.cable_y_junction | 35.9s | 22.8s | **1.57x** | -36.4% | improved |
| newton.cable_bundle_hysteresis | 38.8s | 24.2s | **1.60x** | -37.6% | improved |
| newton.cable_pile | 37.0s | 23.0s | **1.61x** | -38.0% | improved |
| newton.cloth_bending | 30.5s | 19.0s | **1.61x** | -37.9% | improved |
| newton.cloth_hanging | 28.4s | 18.3s | **1.55x** | -35.5% | improved |
| newton.diffsim_ball | 13.9s | 8.1s | **1.73x** | -42.1% | improved |
| newton.diffsim_cloth | 9.8s | 4.5s | **2.18x** | -54.2% | improved |
| newton.diffsim_drone | 15.1s | 8.0s | **1.89x** | -47.0% | improved |
| newton.diffsim_spring_cage | 9.5s | 4.5s | **2.12x** | -52.9% | improved |
| newton.diffsim_soft_body | 18.3s | 9.8s | **1.87x** | -46.6% | improved |
| newton.ik_franka | 30.9s | 16.3s | **1.89x** | -47.2% | improved |
| newton.ik_h1 | 29.3s | 16.5s | **1.78x** | -43.7% | improved |
| newton.ik_cube_stacking | N/A | N/A | — | — | both failed |
| newton.robot_cartpole | 70.6s | 33.6s | **2.10x** | -52.3% | improved |
| newton.robot_anymal_d | 91.3s | 45.5s | **2.01x** | -50.2% | improved |
| newton.robot_ur10 | 75.0s | 33.2s | **2.26x** | -55.7% | improved |
| newton.selection_articulations | 85.5s | 37.0s | **2.31x** | -56.7% | improved |
| newton.selection_cartpole | 73.1s | 35.6s | **2.05x** | -51.3% | improved |
| newton.selection_materials | 84.9s | 38.9s | **2.18x** | -54.2% | improved |
| newton.selection_multiple | 78.9s | 35.1s | **2.25x** | -55.5% | improved |
| newton.sensor_contact | 80.4s | 38.8s | **2.07x** | -51.7% | improved |
| newton.sensor_imu | 72.5s | 33.4s | **2.17x** | -54.0% | improved |

### NEWTON — cuda:0

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| newton.basic_pendulum | 43.9s | 40.3s | **1.09x** | -8.2% | improved |
| newton.basic_urdf | 47.3s | 45.4s | **1.04x** | -4.0% | noise |
| newton.basic_joints | 44.1s | 41.8s | **1.06x** | -5.2% | noise |
| newton.basic_shapes | 53.1s | 51.3s | **1.03x** | -3.4% | ok |
| newton.basic_conveyor | 64.2s | 62.0s | **1.04x** | -3.5% | noise |
| newton.basic_heightfield | 40.4s | 39.6s | **1.02x** | -2.0% | ok |
| newton.cable_twist | 52.6s | 49.3s | **1.07x** | -6.3% | noise |
| newton.cable_y_junction | 51.4s | 48.3s | **1.06x** | -5.9% | improved |
| newton.cable_bundle_hysteresis | 51.8s | 48.8s | **1.06x** | -5.9% | improved |
| newton.cable_pile | 51.1s | 48.8s | **1.05x** | -4.5% | ok |
| newton.cloth_bending | 49.6s | 46.7s | **1.06x** | -5.8% | improved |
| newton.cloth_hanging | 47.8s | 44.5s | **1.07x** | -6.8% | improved |
| newton.cloth_style3d | 50.2s | 46.7s | **1.07x** | -6.9% | improved |
| newton.cloth_twist | 36.9s | 34.6s | **1.07x** | -6.2% | improved |
| newton.cloth_rollers | 28.0s | 25.0s | **1.12x** | -10.6% | improved |
| newton.cloth_poker_cards | 72.2s | 66.8s | **1.08x** | -7.5% | improved |
| newton.cloth_franka | 96.5s | 88.8s | **1.09x** | -7.9% | improved |
| newton.cloth_h1 | 79.2s | 77.1s | **1.03x** | -2.6% | noise |
| newton.brick_stacking | 157.1s | 148.2s | **1.06x** | -5.7% | improved |
| newton.nut_bolt_sdf | 101.7s | 93.1s | **1.09x** | -8.4% | improved |
| newton.nut_bolt_hydro | 108.6s | 98.3s | **1.11x** | -9.5% | improved |
| newton.pyramid | 43.1s | 41.5s | **1.04x** | -3.8% | noise |
| newton.diffsim_ball | 10.8s | 9.9s | **1.09x** | -8.1% | improved |
| newton.diffsim_cloth | 5.8s | 4.7s | **1.23x** | -18.4% | improved |
| newton.diffsim_drone | 12.7s | 11.7s | **1.08x** | -7.6% | improved |
| newton.diffsim_spring_cage | 5.5s | 4.7s | **1.16x** | -14.1% | improved |
| newton.diffsim_soft_body | 13.4s | 11.7s | **1.14x** | -12.2% | noise |
| newton.ik_franka | 66.9s | 63.3s | **1.06x** | -5.4% | improved |
| newton.ik_h1 | 73.3s | 72.9s | **1.00x** | -0.4% | noise |
| newton.ik_custom | 48.3s | 43.6s | **1.11x** | -9.8% | improved |
| newton.ik_cube_stacking | 148.4s | 134.6s | **1.10x** | -9.3% | improved |
| newton.mpm_granular | 37.0s | 27.6s | **1.34x** | -25.3% | improved |
| newton.mpm_multi_material | 37.9s | 28.4s | **1.34x** | -25.1% | improved |
| newton.mpm_twoway_coupling | 136.4s | 120.6s | **1.13x** | -11.6% | improved |
| newton.mpm_beam_twist | 62.6s | 48.3s | **1.29x** | -22.8% | improved |
| newton.mpm_snow_ball | 47.0s | 33.3s | **1.41x** | -29.1% | improved |
| newton.mpm_viscous | 39.5s | 29.1s | **1.36x** | -26.4% | improved |
| newton.robot_cartpole | 55.0s | 47.6s | **1.16x** | -13.5% | improved |
| newton.robot_anymal_d | 91.1s | 80.6s | **1.13x** | -11.5% | improved |
| newton.robot_ur10 | 55.0s | 45.9s | **1.20x** | -16.6% | improved |
| newton.robot_allegro_hand | 93.4s | 83.1s | **1.12x** | -11.0% | improved |
| newton.robot_g1 | 128.7s | 119.8s | **1.07x** | -6.9% | improved |
| newton.robot_h1 | 91.7s | 81.9s | **1.12x** | -10.7% | improved |
| newton.robot_panda_hydro | 170.2s | 155.3s | **1.10x** | -8.8% | improved |
| newton.selection_articulations | 108.7s | 99.4s | **1.09x** | -8.6% | improved |
| newton.selection_cartpole | 59.4s | 53.0s | **1.12x** | -10.8% | improved |
| newton.selection_materials | 65.7s | 57.1s | **1.15x** | -13.2% | improved |
| newton.selection_multiple | 99.4s | 90.8s | **1.09x** | -8.6% | improved |
| newton.sensor_contact | 80.1s | 73.1s | **1.10x** | -8.7% | noise |
| newton.sensor_imu | 65.7s | 66.4s | **0.99x** | +1.0% | noise |
| newton.softbody_hanging | 38.0s | 39.0s | **0.97x** | +2.6% | ok |
| newton.softbody_franka | 140.9s | 143.2s | **0.98x** | +1.6% | noise |
| newton.softbody_gift | 41.9s | 42.6s | **0.99x** | +1.5% | ok |
| newton.softbody_dropping_to_cloth | 47.3s | 48.1s | **0.98x** | +1.7% | noise |

## Summary

No statistically significant regressions detected.
