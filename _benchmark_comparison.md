# Compile-Time Benchmark Comparison

- Baseline: `_benchmark_baseline_all.json`
- Branch: `_benchmark_branch_all.json`


### FEM — cpu

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| fem.diffusion | 48.5s | 13.5s | **3.59x** | -72.2% | improved |
| fem.diffusion_3d | 43.0s | 12.4s | **3.48x** | -71.3% | improved |
| fem.deformed_geometry | 51.1s | 15.1s | **3.39x** | -70.5% | improved |
| fem.stokes | 56.0s | 16.5s | **3.40x** | -70.6% | improved |
| fem.stokes_transfer | 83.7s | 23.1s | **3.62x** | -72.4% | improved |
| fem.mixed_elasticity | N/A | N/A | — | — | both failed |
| fem.magnetostatics | 47.4s | 13.8s | **3.43x** | -70.8% | improved |
| fem.streamlines | 56.4s | 16.1s | **3.51x** | -71.5% | improved |
| fem.adaptive_grid | N/A | N/A | — | — | both failed |
| fem.distortion_energy | 52.7s | 15.3s | **3.45x** | -71.0% | improved |
| fem.nonconforming_contact | 70.6s | 22.3s | **3.17x** | -68.4% | improved |
| fem.convection_diffusion | 31.9s | 8.5s | **3.74x** | -73.3% | improved |
| fem.navier_stokes | 67.8s | 21.2s | **3.20x** | -68.8% | improved |
| fem.burgers | 28.8s | 7.3s | **3.94x** | -74.6% | improved |
| fem.convection_diffusion_dg | 64.4s | 21.1s | **3.05x** | -67.2% | improved |
| fem.taylor_green | 82.3s | 25.2s | **3.27x** | -69.4% | improved |
| fem.shallow_water | 30.4s | 7.8s | **3.89x** | -74.3% | improved |
| fem.kelvin_helmholtz | 32.4s | 8.4s | **3.87x** | -74.2% | improved |
| fem.elastic_shape_optimization | 81.2s | 27.5s | **2.95x** | -66.1% | improved |
| fem.darcy_ls_optimization | 80.9s | 27.9s | **2.90x** | -65.5% | improved |

### FEM — cuda:0

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| fem.diffusion | 8.6s | 8.3s | **1.03x** | -2.7% | ok |
| fem.diffusion_3d | 8.4s | 8.2s | **1.02x** | -1.7% | noise |
| fem.deformed_geometry | 12.0s | 11.9s | **1.01x** | -1.2% | noise |
| fem.stokes | 10.3s | 10.0s | **1.03x** | -3.3% | ok |
| fem.stokes_transfer | 13.0s | 12.4s | **1.05x** | -4.7% | noise |
| fem.mixed_elasticity | 18.0s | 17.9s | **1.01x** | -0.9% | ok |
| fem.magnetostatics | 10.0s | 9.7s | **1.04x** | -3.6% | ok |
| fem.streamlines | 10.7s | 10.1s | **1.06x** | -5.6% | improved |
| fem.adaptive_grid | 21.6s | 21.5s | **1.00x** | -0.2% | noise |
| fem.distortion_energy | 10.8s | 10.2s | **1.05x** | -4.8% | ok |
| fem.nonconforming_contact | 14.9s | 14.7s | **1.02x** | -1.8% | noise |
| fem.convection_diffusion | 5.9s | 5.5s | **1.09x** | -8.1% | improved |
| fem.navier_stokes | 13.7s | 13.3s | **1.02x** | -2.3% | ok |
| fem.burgers | 4.2s | 4.0s | **1.05x** | -4.8% | ok |
| fem.convection_diffusion_dg | 15.6s | 15.0s | **1.04x** | -3.6% | ok |
| fem.taylor_green | 15.5s | 15.1s | **1.03x** | -2.8% | noise |
| fem.shallow_water | 4.3s | 4.1s | **1.04x** | -4.0% | ok |
| fem.kelvin_helmholtz | 4.9s | 4.7s | **1.05x** | -4.3% | ok |
| fem.apic_fluid | 12.4s | 12.3s | **1.01x** | -1.0% | noise |
| fem.elastic_shape_optimization | 19.9s | 19.4s | **1.02x** | -2.3% | noise |
| fem.darcy_ls_optimization | 15.9s | 15.7s | **1.02x** | -1.9% | noise |

### NEWTON — cpu

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| newton.basic_pendulum | 34.6s | 23.2s | **1.49x** | -32.8% | improved |
| newton.basic_urdf | 34.9s | 23.3s | **1.50x** | -33.2% | improved |
| newton.basic_joints | 36.3s | 23.6s | **1.54x** | -35.1% | improved |
| newton.basic_shapes | 36.5s | 23.1s | **1.58x** | -36.9% | improved |
| newton.basic_conveyor | 43.6s | 27.8s | **1.57x** | -36.2% | improved |
| newton.basic_heightfield | 32.5s | 21.2s | **1.53x** | -34.8% | improved |
| newton.cable_twist | 37.2s | 23.1s | **1.61x** | -37.7% | improved |
| newton.cable_y_junction | 35.2s | 22.6s | **1.56x** | -35.9% | improved |
| newton.cable_bundle_hysteresis | 37.2s | 23.2s | **1.60x** | -37.7% | improved |
| newton.cable_pile | 35.4s | 22.9s | **1.55x** | -35.3% | improved |
| newton.cloth_bending | 29.9s | 18.5s | **1.61x** | -38.1% | improved |
| newton.cloth_hanging | 27.9s | 18.0s | **1.55x** | -35.7% | improved |
| newton.diffsim_ball | 13.7s | 7.8s | **1.76x** | -43.2% | improved |
| newton.diffsim_cloth | 9.6s | 4.6s | **2.09x** | -52.3% | improved |
| newton.diffsim_drone | 14.9s | 7.8s | **1.90x** | -47.3% | improved |
| newton.diffsim_spring_cage | 9.4s | 4.4s | **2.13x** | -53.0% | improved |
| newton.diffsim_soft_body | 18.1s | 9.5s | **1.90x** | -47.4% | improved |
| newton.ik_franka | 30.8s | 16.0s | **1.92x** | -47.9% | improved |
| newton.ik_h1 | 29.0s | 15.5s | **1.87x** | -46.5% | improved |
| newton.ik_cube_stacking | N/A | N/A | — | — | both failed |
| newton.robot_cartpole | 70.2s | 32.7s | **2.14x** | -53.3% | improved |
| newton.robot_anymal_d | 90.6s | 44.7s | **2.03x** | -50.7% | improved |
| newton.robot_ur10 | 74.0s | 32.6s | **2.27x** | -55.9% | improved |
| newton.selection_articulations | 83.9s | 36.5s | **2.30x** | -56.5% | improved |
| newton.selection_cartpole | 72.7s | 34.7s | **2.09x** | -52.2% | improved |
| newton.selection_materials | 83.5s | 38.4s | **2.17x** | -54.0% | improved |
| newton.selection_multiple | 78.3s | 34.5s | **2.27x** | -55.9% | improved |
| newton.sensor_contact | 77.7s | 37.9s | **2.05x** | -51.2% | improved |
| newton.sensor_imu | 70.5s | 33.0s | **2.14x** | -53.2% | improved |

### NEWTON — cuda:0

| Example | Main | Branch | Speedup | Delta | Status |
| --- | ---: | ---: | ---: | ---: | --- |
| newton.basic_pendulum | 39.1s | 39.6s | **0.99x** | +1.1% | noise |
| newton.basic_urdf | 43.3s | 43.8s | **0.99x** | +1.3% | noise |
| newton.basic_joints | 41.0s | 39.9s | **1.03x** | -2.7% | ok |
| newton.basic_shapes | 48.4s | 49.2s | **0.98x** | +1.7% | noise |
| newton.basic_conveyor | 58.5s | 58.6s | **1.00x** | +0.3% | noise |
| newton.basic_heightfield | 37.2s | 38.0s | **0.98x** | +2.2% | noise |
| newton.cable_twist | 49.3s | 46.6s | **1.06x** | -5.6% | improved |
| newton.cable_y_junction | 49.3s | 46.8s | **1.05x** | -5.2% | improved |
| newton.cable_bundle_hysteresis | 49.2s | 46.8s | **1.05x** | -4.8% | ok |
| newton.cable_pile | 49.2s | 46.6s | **1.06x** | -5.4% | improved |
| newton.cloth_bending | 47.0s | 44.4s | **1.06x** | -5.5% | improved |
| newton.cloth_hanging | 45.1s | 42.7s | **1.06x** | -5.2% | improved |
| newton.cloth_style3d | 44.3s | 43.6s | **1.01x** | -1.4% | noise |
| newton.cloth_twist | 35.5s | 32.8s | **1.08x** | -7.6% | improved |
| newton.cloth_rollers | 26.4s | 24.3s | **1.09x** | -8.2% | improved |
| newton.cloth_poker_cards | 67.9s | 63.7s | **1.07x** | -6.2% | improved |
| newton.cloth_franka | 89.2s | 86.2s | **1.03x** | -3.3% | ok |
| newton.cloth_h1 | 73.8s | 73.5s | **1.00x** | -0.4% | noise |
| newton.brick_stacking | 141.2s | 139.6s | **1.01x** | -1.2% | noise |
| newton.nut_bolt_sdf | 88.8s | 88.8s | **1.00x** | +0.0% | noise |
| newton.nut_bolt_hydro | 94.0s | 93.4s | **1.01x** | -0.6% | noise |
| newton.pyramid | 39.0s | 38.9s | **1.00x** | -0.1% | noise |
| newton.diffsim_ball | 9.4s | 9.5s | **0.99x** | +0.5% | noise |
| newton.diffsim_cloth | 4.6s | 4.6s | **1.01x** | -0.6% | noise |
| newton.diffsim_drone | 11.1s | 11.4s | **0.98x** | +2.5% | noise |
| newton.diffsim_spring_cage | 4.6s | 4.5s | **1.02x** | -2.2% | ok |
| newton.diffsim_soft_body | 11.0s | 11.3s | **0.97x** | +3.1% | ok |
| newton.ik_franka | 62.5s | 60.5s | **1.03x** | -3.1% | ok |
| newton.ik_h1 | 70.3s | 69.4s | **1.01x** | -1.4% | noise |
| newton.ik_custom | 42.9s | 41.5s | **1.03x** | -3.1% | noise |
| newton.ik_cube_stacking | 130.7s | 127.9s | **1.02x** | -2.1% | ok |
| newton.mpm_granular | 24.9s | 25.0s | **0.99x** | +0.6% | noise |
| newton.mpm_multi_material | 25.5s | 25.9s | **0.98x** | +1.6% | ok |
| newton.mpm_twoway_coupling | 115.1s | 113.3s | **1.02x** | -1.6% | ok |
| newton.mpm_beam_twist | 36.6s | 36.4s | **1.01x** | -0.5% | noise |
| newton.mpm_snow_ball | 29.5s | 30.1s | **0.98x** | +2.1% | noise |
| newton.mpm_viscous | 27.1s | 26.8s | **1.01x** | -1.1% | noise |
| newton.robot_cartpole | 46.8s | 45.8s | **1.02x** | -2.2% | ok |
| newton.robot_anymal_d | 79.4s | 78.2s | **1.02x** | -1.5% | noise |
| newton.robot_ur10 | 46.3s | 44.3s | **1.05x** | -4.3% | ok |
| newton.robot_allegro_hand | 80.5s | 78.5s | **1.03x** | -2.5% | ok |
| newton.robot_g1 | 116.4s | 115.4s | **1.01x** | -0.9% | noise |
| newton.robot_h1 | 79.9s | 78.9s | **1.01x** | -1.3% | noise |
| newton.robot_panda_hydro | 152.8s | 149.4s | **1.02x** | -2.2% | noise |
| newton.selection_articulations | 95.3s | 94.2s | **1.01x** | -1.2% | noise |
| newton.selection_cartpole | 50.4s | 49.6s | **1.02x** | -1.7% | noise |
| newton.selection_materials | 56.1s | 53.8s | **1.04x** | -4.2% | ok |
| newton.selection_multiple | 88.8s | 87.1s | **1.02x** | -1.9% | noise |
| newton.sensor_contact | 72.0s | 69.2s | **1.04x** | -3.9% | ok |
| newton.sensor_imu | 58.3s | 56.3s | **1.04x** | -3.4% | ok |
| newton.softbody_hanging | 36.1s | 34.1s | **1.06x** | -5.5% | improved |
| newton.softbody_franka | 129.8s | 127.1s | **1.02x** | -2.0% | ok |
| newton.softbody_gift | 39.9s | 38.1s | **1.05x** | -4.6% | noise |
| newton.softbody_dropping_to_cloth | 44.9s | 42.6s | **1.05x** | -5.1% | improved |

## Summary

No statistically significant regressions detected.
