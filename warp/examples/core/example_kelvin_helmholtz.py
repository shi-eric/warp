# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###########################################################################
# Example Kelvin-Helmholtz Instability (MHD)
#
# Simulates the Kelvin-Helmholtz instability in a magnetized plasma
# using ideal MHD on a 3D grid. Two layers of plasma slide past each
# other, and the velocity shear triggers vortex roll-up. A background
# magnetic field along the flow direction partially stabilizes the
# instability, creating complex current sheet structures.
#
# Uses a Lax-Friedrichs scheme for isothermal MHD in conservative form.
# The conserved variables are density (ρ), momentum (ρv), and magnetic
# field (B).
#
# Demonstrates:
#   - Ideal MHD equations (density, momentum, magnetic field)
#   - Conservative Lax-Friedrichs scheme
#   - Kelvin-Helmholtz vortex roll-up
#   - Magnetic field amplification at vortex boundaries
#   - Vorticity isosurface visualization
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def mhd_lax_friedrichs(
    # Conserved variables (input)
    rho: wp.array3d[wp.float32],
    mx: wp.array3d[wp.float32],  # ρ*vx (momentum)
    my: wp.array3d[wp.float32],
    mz: wp.array3d[wp.float32],
    bx: wp.array3d[wp.float32],
    by: wp.array3d[wp.float32],
    bz: wp.array3d[wp.float32],
    # Output
    rho_out: wp.array3d[wp.float32],
    mx_out: wp.array3d[wp.float32],
    my_out: wp.array3d[wp.float32],
    mz_out: wp.array3d[wp.float32],
    bx_out: wp.array3d[wp.float32],
    by_out: wp.array3d[wp.float32],
    bz_out: wp.array3d[wp.float32],
    dt_over_dx: float,
    cs2: float,
):
    """Central-difference + small explicit diffusion for isothermal MHD."""
    i, j, k = wp.tid()

    nx = rho.shape[0]
    ny = rho.shape[1]
    nz = rho.shape[2]

    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny

    # Current cell primitives
    r = rho[i, j, k]
    ri = 1.0 / wp.max(r, 0.001)
    vx_c = mx[i, j, k] * ri
    vy_c = my[i, j, k] * ri

    Bx_c = bx[i, j, k]
    By_c = by[i, j, k]

    # Primitives at neighbors
    r_ip = rho[ip, j, k]
    ri_ip = 1.0 / wp.max(r_ip, 0.001)
    vx_ip = mx[ip, j, k] * ri_ip
    vy_ip = my[ip, j, k] * ri_ip

    r_im = rho[im, j, k]
    ri_im = 1.0 / wp.max(r_im, 0.001)
    vx_im = mx[im, j, k] * ri_im
    vy_im = my[im, j, k] * ri_im

    r_jp = rho[i, jp, k]
    ri_jp = 1.0 / wp.max(r_jp, 0.001)
    vx_jp = mx[i, jp, k] * ri_jp
    vy_jp = my[i, jp, k] * ri_jp

    r_jm = rho[i, jm, k]
    ri_jm = 1.0 / wp.max(r_jm, 0.001)
    vx_jm = mx[i, jm, k] * ri_jm
    vy_jm = my[i, jm, k] * ri_jm

    Bx_ip = bx[ip, j, k]
    By_ip = by[ip, j, k]
    Bx_im = bx[im, j, k]
    By_im = by[im, j, k]
    Bx_jp = bx[i, jp, k]
    By_jp = by[i, jp, k]
    Bx_jm = bx[i, jm, k]
    By_jm = by[i, jm, k]

    inv2 = 0.5

    # ∂ρ/∂t = -∂(ρvx)/∂x - ∂(ρvy)/∂y
    drho = (
        (r_ip * vx_ip - r_im * vx_im) * inv2
        + (r_jp * vy_jp - r_jm * vy_jm) * inv2
    ) * dt_over_dx

    # Total pressure (gas + magnetic)
    b2_ip = Bx_ip * Bx_ip + By_ip * By_ip
    b2_im = Bx_im * Bx_im + By_im * By_im
    b2_jp = Bx_jp * Bx_jp + By_jp * By_jp
    b2_jm = Bx_jm * Bx_jm + By_jm * By_jm

    pt_ip = cs2 * r_ip + 0.5 * b2_ip
    pt_im = cs2 * r_im + 0.5 * b2_im
    pt_jp = cs2 * r_jp + 0.5 * b2_jp
    pt_jm = cs2 * r_jm + 0.5 * b2_jm

    # ∂(ρvx)/∂t = -∂(ρvx²+pt-Bx²)/∂x - ∂(ρvx*vy-Bx*By)/∂y
    dmx = (
        (r_ip * vx_ip * vx_ip + pt_ip - Bx_ip * Bx_ip
         - r_im * vx_im * vx_im - pt_im + Bx_im * Bx_im) * inv2
        + (r_jp * vx_jp * vy_jp - Bx_jp * By_jp
           - r_jm * vx_jm * vy_jm + Bx_jm * By_jm) * inv2
    ) * dt_over_dx

    # ∂(ρvy)/∂t = -∂(ρvy*vx-By*Bx)/∂x - ∂(ρvy²+pt-By²)/∂y
    dmy = (
        (r_ip * vy_ip * vx_ip - By_ip * Bx_ip
         - r_im * vy_im * vx_im + By_im * Bx_im) * inv2
        + (r_jp * vy_jp * vy_jp + pt_jp - By_jp * By_jp
           - r_jm * vy_jm * vy_jm - pt_jm + By_jm * By_jm) * inv2
    ) * dt_over_dx

    # Induction: ∂By/∂t = -∂(vx*By-vy*Bx)/∂x  (only x-derivative for By)
    dby = (
        (vx_ip * By_ip - vy_ip * Bx_ip
         - vx_im * By_im + vy_im * Bx_im) * inv2
    ) * dt_over_dx

    # Explicit diffusion for stability (small Laplacian term)
    nu = 0.001  # Numerical viscosity coefficient
    lap_rho = (rho[ip, j, k] + rho[im, j, k] + rho[i, jp, k] + rho[i, jm, k] - 4.0 * r) * nu
    lap_mx = (mx[ip, j, k] + mx[im, j, k] + mx[i, jp, k] + mx[i, jm, k] - 4.0 * mx[i, j, k]) * nu
    lap_my = (my[ip, j, k] + my[im, j, k] + my[i, jp, k] + my[i, jm, k] - 4.0 * my[i, j, k]) * nu
    lap_by = (by[ip, j, k] + by[im, j, k] + by[i, jp, k] + by[i, jm, k] - 4.0 * by[i, j, k]) * nu

    rho_out[i, j, k] = wp.max(r - drho + lap_rho, 0.001)
    mx_out[i, j, k] = mx[i, j, k] - dmx + lap_mx
    my_out[i, j, k] = my[i, j, k] - dmy + lap_my
    mz_out[i, j, k] = mz[i, j, k]
    bx_out[i, j, k] = bx[i, j, k]  # Constant
    by_out[i, j, k] = by[i, j, k] - dby + lap_by
    bz_out[i, j, k] = bz[i, j, k]


@wp.kernel
def compute_vorticity_magnitude(
    mx: wp.array3d[wp.float32],
    my: wp.array3d[wp.float32],
    rho: wp.array3d[wp.float32],
    vorticity: wp.array3d[wp.float32],
    inv2dx: float,
):
    """Compute |ω_z| = |∂vy/∂x - ∂vx/∂y| (dominant component for 2D KH)."""
    i, j, k = wp.tid()

    nx = mx.shape[0]
    ny = mx.shape[1]

    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny

    # Compute velocities from momentum
    vy_ip = my[ip, j, k] / wp.max(rho[ip, j, k], 0.001)
    vy_im = my[im, j, k] / wp.max(rho[im, j, k], 0.001)
    vx_jp = mx[i, jp, k] / wp.max(rho[i, jp, k], 0.001)
    vx_jm = mx[i, jm, k] / wp.max(rho[i, jm, k], 0.001)

    wz = (vy_ip - vy_im - vx_jp + vx_jm) * inv2dx
    vorticity[i, j, k] = wp.abs(wz)


class Example:
    def __init__(self, stage_path="example_kelvin_helmholtz.usd", grid_size=64):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 8

        n = grid_size
        self.dx = 1.0 / n
        # CFL: dt < dx / (|v| + cs + vA)
        # v~0.5, cs=1, vA~0.1 → max speed ~1.6
        # Use small dt to reduce Lax-Friedrichs dissipation
        self.dt = 0.04 * self.dx
        self.cs2 = 1.0  # Isothermal sound speed squared

        # Initial conditions
        rho_init = np.ones((n, n, n), dtype=np.float32)
        mx_init = np.zeros((n, n, n), dtype=np.float32)  # ρ*vx
        my_init = np.zeros((n, n, n), dtype=np.float32)
        mz_init = np.zeros((n, n, n), dtype=np.float32)
        bx_init = np.zeros((n, n, n), dtype=np.float32)
        by_init = np.zeros((n, n, n), dtype=np.float32)
        bz_init = np.zeros((n, n, n), dtype=np.float32)

        for j in range(n):
            y = (j + 0.5) / n
            # Smooth shear profile using tanh
            shear_width = 0.03
            vx_val = 0.5 * (np.tanh((y - 0.25) / shear_width) - np.tanh((y - 0.75) / shear_width) - 1.0)

            # Density contrast: heavier in the middle layer (ρ=2 vs ρ=1)
            rho_val = 1.0 + 0.5 * (np.tanh((y - 0.25) / shear_width) - np.tanh((y - 0.75) / shear_width) + 1.0)

            for i in range(n):
                x = (i + 0.5) / n
                # Perturbation to seed instability
                vy_pert = 0.1 * np.sin(4 * np.pi * x) * (
                    np.exp(-((y - 0.25)**2) / (2 * shear_width**2))
                    + np.exp(-((y - 0.75)**2) / (2 * shear_width**2))
                )

                mx_init[i, j, :] = rho_val * vx_val
                my_init[i, j, :] = rho_val * vy_pert
                rho_init[i, j, :] = rho_val

        # Uniform Bx along flow
        bx_init[:] = 0.1

        self.rho = wp.array(rho_init, dtype=wp.float32)
        self.mx = wp.array(mx_init, dtype=wp.float32)
        self.my = wp.array(my_init, dtype=wp.float32)
        self.mz = wp.array(mz_init, dtype=wp.float32)
        self.bx = wp.array(bx_init, dtype=wp.float32)
        self.by = wp.array(by_init, dtype=wp.float32)
        self.bz = wp.array(bz_init, dtype=wp.float32)

        self.rho_out = wp.zeros_like(self.rho)
        self.mx_out = wp.zeros_like(self.mx)
        self.my_out = wp.zeros_like(self.my)
        self.mz_out = wp.zeros_like(self.mz)
        self.bx_out = wp.zeros_like(self.bx)
        self.by_out = wp.zeros_like(self.by)
        self.bz_out = wp.zeros_like(self.bz)

        self.vorticity = wp.zeros((n, n, n), dtype=wp.float32)
        self.mc = wp.MarchingCubes(nx=n, ny=n, nz=n)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(pos=(1.2, 0.8, 1.2), target=(0.5, 0.5, 0.5), fov=50)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size
            dt_dx = self.dt / self.dx

            for _ in range(self.substeps):
                wp.launch(
                    kernel=mhd_lax_friedrichs,
                    dim=(n, n, n),
                    inputs=[
                        self.rho, self.mx, self.my, self.mz,
                        self.bx, self.by, self.bz,
                        self.rho_out, self.mx_out, self.my_out, self.mz_out,
                        self.bx_out, self.by_out, self.bz_out,
                        dt_dx, self.cs2,
                    ],
                )

                self.rho, self.rho_out = self.rho_out, self.rho
                self.mx, self.mx_out = self.mx_out, self.mx
                self.my, self.my_out = self.my_out, self.my
                self.mz, self.mz_out = self.mz_out, self.mz
                self.bx, self.bx_out = self.bx_out, self.bx
                self.by, self.by_out = self.by_out, self.by
                self.bz, self.bz_out = self.bz_out, self.bz

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            n = self.grid_size

            wp.launch(
                kernel=compute_vorticity_magnitude,
                dim=(n, n, n),
                inputs=[self.mx, self.my, self.rho, self.vorticity, 0.5 / self.dx],
            )

            self.mc.surface(self.vorticity, threshold=2.0)

            self.renderer.begin_frame(self.sim_time)
            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    name="vorticity",
                    colors=(0.8, 0.3, 0.2),
                )
            self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to the output USD file.",
    )
    parser.add_argument("--num-frames", type=int, default=300, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=64, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                r = example.rho.numpy()
                v = example.vorticity.numpy()
                print(f"Frame {i}: rho=[{r.min():.3f},{r.max():.3f}], vort_max={v.max():.3f}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_kelvin_helmholtz.png")
                print("Saved example_kelvin_helmholtz.png")
