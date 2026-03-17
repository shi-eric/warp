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
# Example Cahn-Hilliard (Spinodal Decomposition)
#
# Simulates phase separation in a binary alloy using the Cahn-Hilliard
# equation. A uniformly mixed system is quenched below the miscibility
# gap, triggering spinodal decomposition — spontaneous separation into
# two phases that form complex interconnected labyrinthine domains.
#
# The Cahn-Hilliard equation:
#   ∂c/∂t = M ∇²μ
#   μ = f'(c) - κ ∇²c
#   f(c) = c²(1-c)²  (double-well free energy)
#
# This is a 4th-order PDE solved by splitting into two 2nd-order
# equations (the chemical potential μ and the concentration c).
#
# Demonstrates:
#   - 4th-order PDE via operator splitting
#   - Spinodal decomposition and domain coarsening
#   - Conservation of total concentration
#   - Coarsening dynamics (domain size ~ t^{1/3})
#   - Marching cubes visualization of phase boundaries
#
###########################################################################

import math

import numpy as np

import warp as wp
import warp.render


@wp.kernel
def compute_chemical_potential(
    c: wp.array3d[wp.float32],
    mu: wp.array3d[wp.float32],
    kappa: float,
):
    """Compute chemical potential μ = f'(c) - κ∇²c.

    f(c) = c²(1-c)² → f'(c) = 2c(1-c)(1-2c)
    """
    i, j, k = wp.tid()

    nx = c.shape[0]
    ny = c.shape[1]
    nz = c.shape[2]

    # Periodic neighbors
    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    kp = (k + 1) % nz
    km = (k - 1 + nz) % nz

    ci = c[i, j, k]

    # Double-well derivative: f'(c) = 2c(1-c)(1-2c)
    df = 2.0 * ci * (1.0 - ci) * (1.0 - 2.0 * ci)

    # Laplacian of c
    lap_c = (
        c[ip, j, k] + c[im, j, k]
        + c[i, jp, k] + c[i, jm, k]
        + c[i, j, kp] + c[i, j, km]
        - 6.0 * ci
    )

    mu[i, j, k] = df - kappa * lap_c


@wp.kernel
def update_concentration(
    c: wp.array3d[wp.float32],
    mu: wp.array3d[wp.float32],
    c_out: wp.array3d[wp.float32],
    mobility: float,
    dt: float,
):
    """Update concentration: ∂c/∂t = M ∇²μ."""
    i, j, k = wp.tid()

    nx = mu.shape[0]
    ny = mu.shape[1]
    nz = mu.shape[2]

    ip = (i + 1) % nx
    im = (i - 1 + nx) % nx
    jp = (j + 1) % ny
    jm = (j - 1 + ny) % ny
    kp = (k + 1) % nz
    km = (k - 1 + nz) % nz

    # Laplacian of μ
    lap_mu = (
        mu[ip, j, k] + mu[im, j, k]
        + mu[i, jp, k] + mu[i, jm, k]
        + mu[i, j, kp] + mu[i, j, km]
        - 6.0 * mu[i, j, k]
    )

    c_out[i, j, k] = c[i, j, k] + mobility * dt * lap_mu


class Example:
    def __init__(self, stage_path="example_cahn_hilliard.usd", grid_size=128):
        self.grid_size = grid_size
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0
        self.substeps = 20

        n = grid_size

        # Cahn-Hilliard parameters
        self.mobility = 1.0
        self.kappa = 0.5  # Interface energy (controls interface width)
        self.dt = 0.005

        # Initialize: uniform c=0.5 + small random noise (spinodal region)
        rng = np.random.default_rng(42)
        c_init = 0.5 + 0.05 * rng.standard_normal((n, n, n)).astype(np.float32)
        c_init = np.clip(c_init, 0.01, 0.99)

        self.c = wp.array(c_init, dtype=wp.float32)
        self.c_out = wp.zeros((n, n, n), dtype=wp.float32)
        self.mu = wp.zeros((n, n, n), dtype=wp.float32)

        # Marching cubes for visualization
        self.mc = wp.MarchingCubes(nx=n, ny=n, nz=n)

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            self.renderer.setup_camera(
                pos=(n * 1.5, n * 0.9, n * 1.5),
                target=(n / 2, n / 2, n / 2),
                fov=50,
            )

    def step(self):
        with wp.ScopedTimer("step", active=False):
            n = self.grid_size

            for _ in range(self.substeps):
                # Step 1: compute chemical potential
                wp.launch(
                    kernel=compute_chemical_potential,
                    dim=(n, n, n),
                    inputs=[self.c, self.mu, self.kappa],
                )

                # Step 2: update concentration
                wp.launch(
                    kernel=update_concentration,
                    dim=(n, n, n),
                    inputs=[self.c, self.mu, self.c_out, self.mobility, self.dt],
                )

                self.c, self.c_out = self.c_out, self.c

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            # Isosurface at c=0.5 (phase boundary)
            self.mc.surface(self.c, threshold=0.5)

            self.renderer.begin_frame(self.sim_time)
            if self.mc.verts is not None and len(self.mc.verts) > 0:
                self.renderer.render_mesh(
                    name="phase_boundary",
                    points=self.mc.verts,
                    indices=self.mc.indices,
                    colors=(0.3, 0.6, 0.9),
                )
            self.renderer.end_frame()

    # ── Validation methods ───────────────────────────────────────────

    def compute_total_concentration(self):
        """Check conservation: total concentration should be constant.

        The Cahn-Hilliard equation conserves ∫c dV exactly (it's a
        continuity equation for c). Any drift indicates numerical error.
        """
        c = self.c.numpy()
        return c.mean(), c.sum()

    def compute_free_energy(self):
        """Compute total Ginzburg-Landau free energy.

        F = ∫ [f(c) + κ/2 |∇c|²] dV

        This should decrease monotonically (2nd law of thermodynamics).
        The rate of decrease slows as domains coarsen.
        """
        c = self.c.numpy()

        # Bulk free energy: f(c) = c²(1-c)²
        f_bulk = c**2 * (1.0 - c)**2

        # Gradient energy: κ/2 |∇c|²
        dcdx = np.diff(c, axis=0, append=c[:1, :, :])
        dcdy = np.diff(c, axis=1, append=c[:, :1, :])
        dcdz = np.diff(c, axis=2, append=c[:, :, :1])
        grad_sq = dcdx**2 + dcdy**2 + dcdz**2
        f_grad = 0.5 * self.kappa * grad_sq

        return f_bulk.sum(), f_grad.sum(), f_bulk.sum() + f_grad.sum()

    def compute_domain_size(self):
        """Estimate average domain size from structure factor.

        The characteristic length scale L(t) of the domains should
        follow the Lifshitz-Slyozov coarsening law: L(t) ~ t^{1/3}.

        Computed from the first moment of the spherically-averaged
        structure factor S(k).
        """
        c = self.c.numpy()
        c_fluct = c - c.mean()

        # 3D FFT
        ck = np.fft.fftn(c_fluct)
        sk = np.abs(ck)**2

        n = self.grid_size
        # Spherical average
        kx = np.fft.fftfreq(n) * 2 * np.pi
        ky = np.fft.fftfreq(n) * 2 * np.pi
        kz = np.fft.fftfreq(n) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        K = np.sqrt(KX**2 + KY**2 + KZ**2)

        # First moment: <k> = Σ k*S(k) / Σ S(k)
        k_avg = np.sum(K * sk) / (np.sum(sk) + 1e-10)

        # Domain size ~ 2π / <k>
        if k_avg > 0:
            L = 2.0 * np.pi / k_avg
        else:
            L = 0.0

        return L


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument(
        "--stage-path",
        type=lambda x: None if x == "None" else str(x),
        default=None,
        help="Path to output USD file. If None, uses NativeRenderer.",
    )
    parser.add_argument("--num-frames", type=int, default=200, help="Total number of frames.")
    parser.add_argument("--grid-size", type=int, default=128, help="Grid resolution per axis.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_size=args.grid_size)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 50 == 0:
                c_mean, c_total = example.compute_total_concentration()
                f_bulk, f_grad, f_total = example.compute_free_energy()
                L = example.compute_domain_size()
                print(f"Frame {i}: c_mean={c_mean:.6f}, F={f_total:.1f}, L={L:.2f}")

        if example.renderer:
            if hasattr(example.renderer, 'save'):
                example.renderer.save()
            if hasattr(example.renderer, 'save_image'):
                example.renderer.save_image("example_cahn_hilliard.png")
                print("Saved example_cahn_hilliard.png")
