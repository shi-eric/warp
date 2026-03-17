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
# Example Hodgkin-Huxley Neural Network
#
# Simulates a 3D lattice of neurons with Hodgkin-Huxley dynamics,
# coupled by nearest-neighbor synaptic connections. A stimulus at one
# point triggers a propagating action potential wave that spreads
# through the network.
#
# The Hodgkin-Huxley model uses real biophysical parameters:
#   - Resting potential: ~-65 mV
#   - Action potential peak: ~+40 mV
#   - Sodium, potassium, and leak conductances
#   - Voltage-dependent gating variables (m, h, n)
#
# Visualization uses marching cubes to extract the isosurface of
# excited neurons (voltage above threshold), producing a growing
# wavefront.
#
# Demonstrates:
#   - Hodgkin-Huxley neural dynamics on GPU
#   - 3D stencil-based synaptic coupling
#   - Marching cubes isosurface extraction
#   - Biophysically realistic spiking network
#
# Validation:
#   - Action potential shape (resting ~-65 mV, peak ~+40 mV)
#   - Propagation speed (wave front velocity)
#   - Refractory period (~2-4 ms for HH)
#
###########################################################################

import numpy as np

import warp as wp
import warp.render


# ---- Hodgkin-Huxley rate functions ----
# All voltages in mV, time in ms, currents in µA/cm²
# Standard HH parameters from the 1952 paper

@wp.func
def alpha_n(V: float):
    """Rate constant α_n(V) for potassium activation gate."""
    dv = V + 55.0
    if wp.abs(dv) < 0.001:
        return 0.1
    return 0.01 * dv / (1.0 - wp.exp(-dv / 10.0))


@wp.func
def beta_n(V: float):
    """Rate constant β_n(V) for potassium activation gate."""
    return 0.125 * wp.exp(-(V + 65.0) / 80.0)


@wp.func
def alpha_m(V: float):
    """Rate constant α_m(V) for sodium activation gate."""
    dv = V + 40.0
    if wp.abs(dv) < 0.001:
        return 1.0
    return 0.1 * dv / (1.0 - wp.exp(-dv / 10.0))


@wp.func
def beta_m(V: float):
    """Rate constant β_m(V) for sodium activation gate."""
    return 4.0 * wp.exp(-(V + 65.0) / 18.0)


@wp.func
def alpha_h(V: float):
    """Rate constant α_h(V) for sodium inactivation gate."""
    return 0.07 * wp.exp(-(V + 65.0) / 20.0)


@wp.func
def beta_h(V: float):
    """Rate constant β_h(V) for sodium inactivation gate."""
    return 1.0 / (1.0 + wp.exp(-(V + 35.0) / 10.0))


@wp.kernel
def hh_step(
    V: wp.array3d[wp.float32],
    m: wp.array3d[wp.float32],
    h: wp.array3d[wp.float32],
    n: wp.array3d[wp.float32],
    V_new: wp.array3d[wp.float32],
    m_new: wp.array3d[wp.float32],
    h_new: wp.array3d[wp.float32],
    n_new: wp.array3d[wp.float32],
    g_Na: float,      # 120 mS/cm²
    g_K: float,       # 36 mS/cm²
    g_L: float,       # 0.3 mS/cm²
    E_Na: float,      # 50 mV
    E_K: float,       # -77 mV
    E_L: float,       # -54.387 mV
    C_m: float,       # 1 µF/cm²
    g_syn: float,     # Synaptic coupling conductance
    E_syn: float,     # Synaptic reversal potential (0 mV for excitatory)
    I_ext: float,     # External stimulus current
    stim_i: int,
    stim_j: int,
    stim_k: int,
    stim_radius: int,
    dt: float,
):
    """One Hodgkin-Huxley time step with nearest-neighbor synaptic coupling."""
    i, j, k = wp.tid()

    nx = V.shape[0]
    ny = V.shape[1]
    nz = V.shape[2]

    v = V[i, j, k]
    m_val = m[i, j, k]
    h_val = h[i, j, k]
    n_val = n[i, j, k]

    # Ionic currents
    I_Na = g_Na * m_val * m_val * m_val * h_val * (v - E_Na)
    I_K = g_K * n_val * n_val * n_val * n_val * (v - E_K)
    I_L = g_L * (v - E_L)

    # Synaptic current from 6 nearest neighbors
    # Synapse activates when presynaptic V > -20 mV (simplified)
    I_syn = float(0.0)
    count = 0

    if i > 0:
        v_pre = V[i - 1, j, k]
        s = 1.0 / (1.0 + wp.exp(-(v_pre + 20.0) / 2.0))
        I_syn = I_syn + g_syn * s * (v - E_syn)
        count = count + 1
    if i < nx - 1:
        v_pre = V[i + 1, j, k]
        s = 1.0 / (1.0 + wp.exp(-(v_pre + 20.0) / 2.0))
        I_syn = I_syn + g_syn * s * (v - E_syn)
        count = count + 1
    if j > 0:
        v_pre = V[i, j - 1, k]
        s = 1.0 / (1.0 + wp.exp(-(v_pre + 20.0) / 2.0))
        I_syn = I_syn + g_syn * s * (v - E_syn)
        count = count + 1
    if j < ny - 1:
        v_pre = V[i, j + 1, k]
        s = 1.0 / (1.0 + wp.exp(-(v_pre + 20.0) / 2.0))
        I_syn = I_syn + g_syn * s * (v - E_syn)
        count = count + 1
    if k > 0:
        v_pre = V[i, j, k - 1]
        s = 1.0 / (1.0 + wp.exp(-(v_pre + 20.0) / 2.0))
        I_syn = I_syn + g_syn * s * (v - E_syn)
        count = count + 1
    if k < nz - 1:
        v_pre = V[i, j, k + 1]
        s = 1.0 / (1.0 + wp.exp(-(v_pre + 20.0) / 2.0))
        I_syn = I_syn + g_syn * s * (v - E_syn)
        count = count + 1

    # External stimulus
    I_stim = float(0.0)
    di = i - stim_i
    dj = j - stim_j
    dk = k - stim_k
    if di * di + dj * dj + dk * dk <= stim_radius * stim_radius:
        I_stim = I_ext

    # Membrane equation: C_m * dV/dt = -I_Na - I_K - I_L - I_syn + I_stim
    dV = (-I_Na - I_K - I_L - I_syn + I_stim) / C_m

    # Gating variable updates
    an = alpha_n(v)
    bn = beta_n(v)
    am = alpha_m(v)
    bm = beta_m(v)
    ah = alpha_h(v)
    bh = beta_h(v)

    dm = am * (1.0 - m_val) - bm * m_val
    dh = ah * (1.0 - h_val) - bh * h_val
    dn = an * (1.0 - n_val) - bn * n_val

    # Forward Euler
    V_new[i, j, k] = v + dV * dt
    m_new[i, j, k] = wp.clamp(m_val + dm * dt, 0.0, 1.0)
    h_new[i, j, k] = wp.clamp(h_val + dh * dt, 0.0, 1.0)
    n_new[i, j, k] = wp.clamp(n_val + dn * dt, 0.0, 1.0)


@wp.kernel
def voltage_to_field(
    V: wp.array3d[wp.float32],
    field: wp.array3d[wp.float32],
    threshold: float,
):
    """Convert voltage field to a scalar field for marching cubes.

    Values < 0 are "inside" the isosurface (excited neurons).
    """
    i, j, k = wp.tid()
    # Invert: excited = negative (inside surface)
    field[i, j, k] = threshold - V[i, j, k]


class Example:
    def __init__(self, stage_path="example_neural_spikes.usd", grid_dim=32):
        self.grid_dim = grid_dim
        self.sim_time = 0.0
        self.frame_dt = 1.0 / 30.0  # Rendering at 30 fps
        # HH dynamics need small dt (~0.01 ms)
        self.hh_dt = 0.01  # ms
        # Each render frame advances by ~0.5 ms of neural time
        self.steps_per_frame = 50
        self.neural_time = 0.0  # ms

        # Hodgkin-Huxley parameters (standard 1952 squid axon values)
        self.g_Na = 120.0   # mS/cm²
        self.g_K = 36.0     # mS/cm²
        self.g_L = 0.3      # mS/cm²
        self.E_Na = 50.0    # mV
        self.E_K = -77.0    # mV
        self.E_L = -54.387  # mV
        self.C_m = 1.0      # µF/cm²

        # Synaptic coupling
        self.g_syn = 0.8    # Coupling strength (mS/cm²)
        self.E_syn = 0.0    # Excitatory reversal potential (mV)

        # Stimulus: inject current at center for first N steps
        self.stim_center = grid_dim // 2
        self.stim_radius = max(2, grid_dim // 16)
        self.stim_current = 20.0   # µA/cm² (strong enough to trigger AP)
        self.stim_duration = 2.0   # ms

        N = grid_dim

        # Initialize at resting potential
        V_rest = -65.0
        # Steady-state gating variables at rest
        m_inf = float(alpha_m.func(V_rest) / (alpha_m.func(V_rest) + beta_m.func(V_rest))) if False else 0.05
        h_inf = 0.6
        n_inf = 0.32

        # Actually compute steady-state properly
        am = 0.1 * (V_rest + 40.0) / (1.0 - np.exp(-(V_rest + 40.0) / 10.0))
        bm = 4.0 * np.exp(-(V_rest + 65.0) / 18.0)
        m_inf = am / (am + bm)

        ah = 0.07 * np.exp(-(V_rest + 65.0) / 20.0)
        bh = 1.0 / (1.0 + np.exp(-(V_rest + 35.0) / 10.0))
        h_inf = ah / (ah + bh)

        an = 0.01 * (V_rest + 55.0) / (1.0 - np.exp(-(V_rest + 55.0) / 10.0))
        bn = 0.125 * np.exp(-(V_rest + 65.0) / 80.0)
        n_inf = an / (an + bn)

        self.V = wp.array(np.full((N, N, N), V_rest, dtype=np.float32), dtype=wp.float32)
        self.m = wp.array(np.full((N, N, N), m_inf, dtype=np.float32), dtype=wp.float32)
        self.h = wp.array(np.full((N, N, N), h_inf, dtype=np.float32), dtype=wp.float32)
        self.n = wp.array(np.full((N, N, N), n_inf, dtype=np.float32), dtype=wp.float32)

        self.V_new = wp.zeros((N, N, N), dtype=wp.float32)
        self.m_new = wp.zeros((N, N, N), dtype=wp.float32)
        self.h_new = wp.zeros((N, N, N), dtype=wp.float32)
        self.n_new = wp.zeros((N, N, N), dtype=wp.float32)

        # Marching cubes for isosurface visualization
        self.mc_field = wp.zeros((N, N, N), dtype=wp.float32)
        self.mc = wp.MarchingCubes(N, N, N, max_verts=500000, max_tris=500000)
        self.voltage_threshold = -20.0  # mV threshold for "excited"

        # Validation tracking
        self.center_voltage_trace = []  # V at stimulus center over time
        self.time_trace = []
        self.wavefront_positions = []  # Track wavefront for speed measurement

        if stage_path and stage_path.endswith((".usd", ".usda", ".usdc")):
            self.renderer = wp.render.UsdRenderer(stage_path)
        else:
            self.renderer = wp.render.NativeRenderer(512, 512)
            half = self.grid_dim / 2.0
            dist = self.grid_dim * 1.2
            self.renderer.setup_camera(
                pos=(half + dist * 0.7, half + dist * 0.4, half + dist * 0.7),
                target=(half, half, half),
                fov=45,
            )
            self.renderer.set_environment("dark")
            self.renderer.bg_top = wp.vec3(0.02, 0.02, 0.06)
            self.renderer.bg_bottom = wp.vec3(0.01, 0.01, 0.03)

    def step(self):
        with wp.ScopedTimer("step", active=False):
            N = self.grid_dim
            sc = self.stim_center
            sr = self.stim_radius

            for _ in range(self.steps_per_frame):
                # Determine if stimulus is active
                I_ext = self.stim_current if self.neural_time < self.stim_duration else 0.0

                wp.launch(
                    kernel=hh_step,
                    dim=(N, N, N),
                    inputs=[
                        self.V, self.m, self.h, self.n,
                        self.V_new, self.m_new, self.h_new, self.n_new,
                        self.g_Na, self.g_K, self.g_L,
                        self.E_Na, self.E_K, self.E_L,
                        self.C_m, self.g_syn, self.E_syn,
                        I_ext, sc, sc, sc, sr,
                        self.hh_dt,
                    ],
                )

                # Swap buffers
                self.V, self.V_new = self.V_new, self.V
                self.m, self.m_new = self.m_new, self.m
                self.h, self.h_new = self.h_new, self.h
                self.n, self.n_new = self.n_new, self.n

                self.neural_time += self.hh_dt

            # Record center voltage for validation
            V_np = self.V.numpy()
            self.center_voltage_trace.append(float(V_np[sc, sc, sc]))
            self.time_trace.append(self.neural_time)

            # Track wavefront (furthest excited neuron from center)
            excited = V_np > self.voltage_threshold
            if np.any(excited):
                coords = np.argwhere(excited)
                dists = np.sqrt(np.sum((coords - sc) ** 2, axis=1))
                self.wavefront_positions.append(float(np.max(dists)))
            else:
                self.wavefront_positions.append(0.0)

            # Convert voltage field for marching cubes
            wp.launch(
                kernel=voltage_to_field,
                dim=(N, N, N),
                inputs=[self.V, self.mc_field, self.voltage_threshold],
            )
            self.mc.surface(self.mc_field, 0.0)

            self.sim_time += self.frame_dt

    def render(self):
        if self.renderer is None:
            return

        with wp.ScopedTimer("render", active=False):
            self.renderer.begin_frame(self.sim_time)

            # Render isosurface of excited neurons
            if self.mc.verts is not None and len(self.mc.verts) > 0:
                try:
                    verts = self.mc.verts.numpy()
                    indices = self.mc.indices.numpy()
                    if len(verts) > 0 and len(indices) > 0:
                        self.renderer.render_mesh(
                            name="wavefront",
                            points=verts,
                            indices=indices,
                            colors=(0.2, 0.5, 1.0),
                        )
                except Exception:
                    pass

            self.renderer.end_frame()

    # ---- Validation methods ----

    def get_action_potential_trace(self):
        """Return (time_ms, voltage_mV) at the stimulus center.

        Expected: resting ~-65 mV → rapid rise to ~+40 mV → repolarization
        to ~-75 mV → return to rest. Duration ~1-2 ms.
        """
        return np.array(self.time_trace), np.array(self.center_voltage_trace)

    def get_propagation_speed(self):
        """Estimate wave propagation speed in grid-cells per ms.

        Uses the wavefront distance over time. For coupled HH neurons
        typical speed is O(1) grid-cell/ms depending on coupling.
        """
        if len(self.wavefront_positions) < 10:
            return 0.0

        wp_arr = np.array(self.wavefront_positions)
        t_arr = np.array(self.time_trace)

        # Find the advancing front (where distance is growing)
        growing = np.where(np.diff(wp_arr) > 0.1)[0]
        if len(growing) < 2:
            return 0.0

        i0 = growing[0]
        i1 = growing[-1]
        dr = wp_arr[i1] - wp_arr[i0]
        dt = t_arr[i1] - t_arr[i0]
        if dt < 0.001:
            return 0.0
        return dr / dt  # grid cells per ms

    def get_refractory_period(self):
        """Estimate refractory period from the voltage trace.

        Refractory period = time from peak to when voltage returns
        above a re-excitation threshold (~-50 mV).
        """
        V_trace = np.array(self.center_voltage_trace)
        t_trace = np.array(self.time_trace)

        if len(V_trace) < 5:
            return 0.0

        # Find peak
        peak_idx = np.argmax(V_trace)
        if peak_idx >= len(V_trace) - 1:
            return 0.0

        # Find when voltage returns above -50 mV after the undershoot
        after_peak = V_trace[peak_idx:]
        re_threshold = -50.0

        # Find minimum after peak (undershoot)
        min_idx = np.argmin(after_peak)

        # Find recovery after minimum
        after_min = after_peak[min_idx:]
        recovery = np.where(after_min > re_threshold)[0]
        if len(recovery) == 0:
            return 0.0

        total_idx = peak_idx + min_idx + recovery[0]
        if total_idx >= len(t_trace):
            return 0.0

        return t_trace[total_idx] - t_trace[peak_idx]

    def get_voltage_stats(self):
        """Return min/max/mean voltage across the grid."""
        V_np = self.V.numpy()
        return {
            "min": float(V_np.min()),
            "max": float(V_np.max()),
            "mean": float(V_np.mean()),
            "excited_fraction": float(np.mean(V_np > self.voltage_threshold)),
        }


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
    parser.add_argument("--grid-dim", type=int, default=64, help="Grid dimension (N³ neurons).")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(stage_path=args.stage_path, grid_dim=args.grid_dim)

        for i in range(args.num_frames):
            example.step()
            example.render()

            if i % 20 == 0:
                stats = example.get_voltage_stats()
                print(f"Frame {i} (t={example.neural_time:.1f} ms): "
                      f"V=[{stats['min']:.1f}, {stats['max']:.1f}] mV, "
                      f"excited={stats['excited_fraction']:.3f}")

        speed = example.get_propagation_speed()
        print(f"\nPropagation speed: {speed:.2f} cells/ms")

        if example.renderer:
            if hasattr(example.renderer, "save"):
                example.renderer.save()
            if hasattr(example.renderer, "save_image"):
                example.renderer.save_image("example_neural_spikes.png")
