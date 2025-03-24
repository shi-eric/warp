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

"""Warp version of cuboltz_bench from stlbm (https://gitlab.com/unigehpfs/stlbm)

Double-population scheme using D3Q19
"""

import argparse
import timeit

import numpy as np

import warp as wp

wp.set_module_options({"enable_backward": False})

wp.init()

problem_dtype = wp.float64

vec19 = wp.types.vector(length=19, dtype=problem_dtype)
vec3 = wp.types.vector(length=3, dtype=problem_dtype)


FLUID = wp.constant(0)
WALL = wp.constant(1)
WALL_M = wp.constant(2)
BOUNCE = wp.constant(3)
CORNER = wp.constant(4)
MOVING_WALL = wp.constant(5)

bounds_priority_np = np.array(
    [
        0,  # fluid,
        1,  # wall,
        2,  # wall_m,
        3,  # bounce,
        4,  # corner,
        5,  # moving_wall,
    ],
    dtype=int,
)


@wp.struct
class Domain:
    nx: int = 0
    ny: int = 0
    nz: int = 0


@wp.struct
class LBMVars:
    u: wp.array3d(dtype=vec3)
    u_star: wp.array3d(dtype=vec3)  # Purely for output diagnostics
    rho_star: wp.array3d(dtype=problem_dtype)  # Purely for output diagnostics
    f0: wp.array4d(dtype=problem_dtype)
    f1: wp.array4d(dtype=problem_dtype)
    boundary_flags: wp.array3d(dtype=wp.int32)
    boundary_values: wp.array3d(dtype=wp.int32)
    boundary_dirs: wp.array3d(dtype=wp.int32)


@wp.func
def set_flag(
    i: int,
    j: int,
    k: int,
    maybe_flag: wp.int32,
    direction: int,
    bounds_priority: wp.array(dtype=wp.int8),
    boundary_flags: wp.array3d(dtype=wp.int32),
    boundary_values: wp.array3d(dtype=wp.int32),
    boundary_dirs: wp.array3d(dtype=wp.int32),
):
    if maybe_flag >= CORNER:
        boundary_dirs[i, j, k] = direction

    if bounds_priority[boundary_flags[i, j, k]] < bounds_priority[maybe_flag]:
        boundary_flags[i, j, k] = maybe_flag

        if maybe_flag == BOUNCE:
            boundary_values[i, j, k] = BOUNCE


@wp.kernel
def mark_flags(
    domain: Domain,
    outer_wall_bc_config: wp.array(dtype=wp.int32),
    bounds_priority: wp.array(dtype=wp.int8),
    boundary_flags: wp.array3d(dtype=wp.int32),
    boundary_values: wp.array3d(dtype=wp.int32),
    boundary_dirs: wp.array3d(dtype=wp.int32),
):
    i, j, k = wp.tid()

    if i == 0:
        set_flag(i, j, k, outer_wall_bc_config[0], 0, bounds_priority, boundary_flags, boundary_values, boundary_dirs)
    elif i == domain.nx - 1:
        set_flag(i, j, k, outer_wall_bc_config[1], 1, bounds_priority, boundary_flags, boundary_values, boundary_dirs)

    if j == 0:
        set_flag(i, j, k, outer_wall_bc_config[2], 2, bounds_priority, boundary_flags, boundary_values, boundary_dirs)
    elif j == domain.ny - 1:
        set_flag(i, j, k, outer_wall_bc_config[3], 3, bounds_priority, boundary_flags, boundary_values, boundary_dirs)

    if k == 0:
        set_flag(i, j, k, outer_wall_bc_config[4], 4, bounds_priority, boundary_flags, boundary_values, boundary_dirs)
    elif k == domain.nz - 1:
        set_flag(i, j, k, outer_wall_bc_config[5], 5, bounds_priority, boundary_flags, boundary_values, boundary_dirs)


@wp.func
def ibar(i: int, nbdir: int):
    # This function simply computes the matching index of node i for reflecting boundary conditions
    # The streaming kernel hard-codes the matching indices instead of calling this function
    return (i + nbdir // 2 - 1) % (nbdir - 1) + 1


@wp.kernel
def find_wall(
    lattice_velocities: wp.array(dtype=wp.vec3b),
    boundary_flags: wp.array3d(dtype=wp.int32),
    boundary_values: wp.array3d(dtype=wp.int32),
    boundary_dirs: wp.array3d(dtype=wp.int32),
):
    # Defines flags for domain boundary conditions. Finds the lattice cells exactly at the limit of a
    # bounce back obstacle and flags them as "wall", only used at initialization.
    i, j, k = wp.tid()

    if boundary_flags[i, j, k] < BOUNCE:  # FLUID, WALL, or WALL_M
        # Loop over all boundary directions
        for neighbor_index in range(1, 19):
            neighbor_i = i + wp.int32(lattice_velocities[neighbor_index][0])
            neighbor_j = j + wp.int32(lattice_velocities[neighbor_index][1])
            neighbor_k = k + wp.int32(lattice_velocities[neighbor_index][2])

            if neighbor_i < 0 or neighbor_i >= boundary_flags.shape[0]:
                continue
            if neighbor_j < 0 or neighbor_j >= boundary_flags.shape[1]:
                continue
            if neighbor_k < 0 or neighbor_k >= boundary_flags.shape[2]:
                continue

            if boundary_flags[neighbor_i, neighbor_j, neighbor_k] == MOVING_WALL:
                boundary_flags[i, j, k] = WALL_M
                boundary_dirs[i, j, k] = boundary_dirs[neighbor_i, neighbor_j, neighbor_k]
                boundary_values[i, j, k] = wp.bit_or(
                    boundary_values[i, j, k], wp.lshift(1, ibar(neighbor_index, lattice_velocities.shape[0]))
                )

            if boundary_flags[neighbor_i, neighbor_j, neighbor_k] == BOUNCE:
                if boundary_flags[i, j, k] != WALL_M:
                    boundary_flags[i, j, k] = WALL

                boundary_values[i, j, k] = wp.bit_or(
                    boundary_values[i, j, k], wp.lshift(1, ibar(neighbor_index, lattice_velocities.shape[0]))
                )


@wp.func
def second_order_equilibrium(rho: problem_dtype, t: wp.int32, cu: problem_dtype, usqr: problem_dtype):
    return rho * problem_dtype(t) * (problem_dtype(1) + cu + cu * cu / problem_dtype(2.0) - usqr)


@wp.func
def get_equilibrium(rho: problem_dtype, u_init: vec3):
    # Computes the second-order BGK equilibrium.
    usqr = problem_dtype(3.0 / 2.0) * (u_init[0] * u_init[0] + u_init[1] * u_init[1] + u_init[2] * u_init[2])
    rho = rho / problem_dtype(36.0)

    f_eq = vec19()
    f_eq[0] = second_order_equilibrium(rho, 12, problem_dtype(0), usqr)
    f_eq[1] = second_order_equilibrium(rho, 2, problem_dtype(-3.0) * u_init[0], usqr)
    f_eq[10] = f_eq[1] - rho * problem_dtype(-12.0) * u_init[0]
    f_eq[2] = second_order_equilibrium(rho, 2, problem_dtype(-3.0) * u_init[1], usqr)
    f_eq[11] = f_eq[2] - rho * problem_dtype(-12.0) * u_init[1]
    f_eq[3] = second_order_equilibrium(rho, 2, problem_dtype(-3.0) * u_init[2], usqr)
    f_eq[12] = f_eq[3] - rho * problem_dtype(-12.0) * u_init[2]

    cu = problem_dtype(3.0) * (-u_init[0] - u_init[1])
    f_eq[4] = second_order_equilibrium(rho, 1, cu, usqr)
    f_eq[13] = f_eq[4] - rho * problem_dtype(2) * cu

    cu = problem_dtype(3.0) * (-u_init[0] + u_init[1])
    f_eq[5] = second_order_equilibrium(rho, 1, cu, usqr)
    f_eq[14] = f_eq[5] - rho * problem_dtype(2.0) * cu

    cu = problem_dtype(3.0) * (-u_init[0] - u_init[2])
    f_eq[6] = second_order_equilibrium(rho, 1, cu, usqr)
    f_eq[15] = f_eq[6] - rho * problem_dtype(2.0) * cu

    cu = problem_dtype(3.0) * (-u_init[0] + u_init[2])
    f_eq[7] = second_order_equilibrium(rho, 1, cu, usqr)
    f_eq[16] = f_eq[7] - rho * problem_dtype(2.0) * cu

    cu = problem_dtype(3.0) * (-u_init[1] - u_init[2])
    f_eq[8] = second_order_equilibrium(rho, 1, cu, usqr)
    f_eq[17] = f_eq[8] - rho * problem_dtype(2.0) * cu

    cu = problem_dtype(3.0) * (-u_init[1] + u_init[2])
    f_eq[9] = second_order_equilibrium(rho, 1, cu, usqr)
    f_eq[18] = f_eq[9] - rho * problem_dtype(2.0) * cu

    return f_eq


@wp.kernel
def init_velocity(rho: problem_dtype, u_init: vec3, domain_data: LBMVars):
    i, j, k = wp.tid()

    domain_data.u_star[i, j, k] = u_init

    eq_vec = get_equilibrium(rho, u_init)

    for velocity_dir in range(19):
        domain_data.f0[velocity_dir, i, j, k] = eq_vec[velocity_dir]
        domain_data.f1[velocity_dir, i, j, k] = eq_vec[velocity_dir]


@wp.func
def streaming(f0: wp.array(dtype=problem_dtype, ndim=4), i: int, j: int, k: int):
    fin = vec19()

    fin[0] = f0[0, i, j, k]
    fin[1] = f0[1, i + 1, j, k]
    fin[2] = f0[2, i, j + 1, k]
    fin[3] = f0[3, i, j, k + 1]
    fin[4] = f0[4, i + 1, j + 1, k]
    fin[5] = f0[5, i + 1, j - 1, k]
    fin[6] = f0[6, i + 1, j, k + 1]
    fin[7] = f0[7, i + 1, j, k - 1]
    fin[8] = f0[8, i, j + 1, k + 1]
    fin[9] = f0[9, i, j + 1, k - 1]
    fin[10] = f0[10, i - 1, j, k]
    fin[11] = f0[11, i, j - 1, k]
    fin[12] = f0[12, i, j, k - 1]
    fin[13] = f0[13, i - 1, j - 1, k]
    fin[14] = f0[14, i - 1, j + 1, k]
    fin[15] = f0[15, i - 1, j, k - 1]
    fin[16] = f0[16, i - 1, j, k + 1]
    fin[17] = f0[17, i, j - 1, k - 1]
    fin[18] = f0[18, i, j - 1, k + 1]

    return fin


@wp.func
def streaming_bounce(
    f0: wp.array(dtype=problem_dtype, ndim=4), i: int, j: int, k: int, fin: vec19, where_bitflag: wp.int32
):
    fout = fin

    fout[0] = f0[0, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 1)) > 0:
        fout[1] = f0[10, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 2)) > 0:
        fout[2] = f0[11, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 3)) > 0:
        fout[3] = f0[12, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 4)) > 0:
        fout[4] = f0[13, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 5)) > 0:
        fout[5] = f0[14, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 6)) > 0:
        fout[6] = f0[15, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 7)) > 0:
        fout[7] = f0[16, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 8)) > 0:
        fout[8] = f0[17, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 9)) > 0:
        fout[9] = f0[18, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 10)) > 0:
        fout[10] = f0[1, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 11)) > 0:
        fout[11] = f0[2, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 12)) > 0:
        fout[12] = f0[3, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 13)) > 0:
        fout[13] = f0[4, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 14)) > 0:
        fout[14] = f0[5, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 15)) > 0:
        fout[15] = f0[6, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 16)) > 0:
        fout[16] = f0[7, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 17)) > 0:
        fout[17] = f0[8, i, j, k]

    if wp.bit_and(where_bitflag, wp.lshift(1, 18)) > 0:
        fout[18] = f0[9, i, j, k]

    return fout


@wp.func
def streaming_wall2(fin: vec19, boundary_dir: wp.int32, lid_vel: problem_dtype):
    u0 = lid_vel
    u1 = problem_dtype(0)
    u2 = problem_dtype(0)

    fout = fin

    if boundary_dir == 1:
        fout[1] = fin[1] - problem_dtype(1.0 / 3.0) * u0

    if boundary_dir == 3:
        fout[2] = fin[2] - problem_dtype(1.0 / 3.0) * u1

    if boundary_dir == 5:
        fout[3] = fin[3] - problem_dtype(1.0 / 3.0) * u2

    if boundary_dir == 1 or boundary_dir == 3:
        fout[4] = fin[4] + problem_dtype(1.0 / 6.0) * (-u0 - u1)

    if boundary_dir == 1 or boundary_dir == 2:
        fout[5] = fin[5] + problem_dtype(1.0 / 6.0) * (-u0 + u1)

    if boundary_dir == 1 or boundary_dir == 5:
        fout[6] = fin[6] + problem_dtype(1.0 / 6.0) * (-u0 - u2)

    if boundary_dir == 1 or boundary_dir == 4:
        fout[7] = fin[7] + problem_dtype(1.0 / 6.0) * (-u0 + u2)

    if boundary_dir == 3 or boundary_dir == 5:
        fout[8] = fin[8] + problem_dtype(1.0 / 6.0) * (-u1 - u2)

    if boundary_dir == 3 or boundary_dir == 4:
        fout[9] = fin[9] + problem_dtype(1.0 / 6.0) * (-u1 + u2)

    if boundary_dir == 0:
        fout[10] = fin[10] + problem_dtype(1.0 / 6.0) * u0

    if boundary_dir == 2:
        fout[11] = fin[11] + problem_dtype(1.0 / 6.0) * u1

    if boundary_dir == 4:
        fout[12] = fin[12] + problem_dtype(1.0 / 6.0) * u2

    if boundary_dir == 0 or boundary_dir == 2:
        fout[13] = fin[13] + problem_dtype(1.0 / 6.0) * (u0 + u1)

    if boundary_dir == 0 or boundary_dir == 3:
        fout[14] = fin[14] + problem_dtype(1.0 / 6.0) * (u0 - u1)

    if boundary_dir == 0 or boundary_dir == 4:
        fout[15] = fin[15] + problem_dtype(1.0 / 6.0) * (u0 + u2)

    if boundary_dir == 0 or boundary_dir == 5:
        fout[16] = fin[16] + problem_dtype(1.0 / 6.0) * (u0 - u2)

    if boundary_dir == 2 or boundary_dir == 4:
        fout[17] = fin[17] + problem_dtype(1.0 / 6.0) * (u1 + u2)

    if boundary_dir == 2 or boundary_dir == 5:
        fout[18] = fin[18] + problem_dtype(1.0 / 6.0) * (u1 - u2)

    return fout


@wp.func
def calc_macroscopic(f: vec19):
    X_M1 = f[1] + f[4] + f[5] + f[6] + f[7]
    X_P1 = f[10] + f[13] + f[14] + f[15] + f[16]
    X_0 = f[0] + f[2] + f[3] + f[8] + f[9] + f[11] + f[12] + f[17] + f[18]
    Y_M1 = f[2] + f[4] + f[8] + f[9] + f[14]
    Y_P1 = f[5] + f[11] + f[13] + f[17] + f[18]
    Z_M1 = f[3] + f[6] + f[8] + f[16] + f[18]
    Z_P1 = f[7] + f[9] + f[12] + f[15] + f[17]

    rho = X_M1 + X_P1 + X_0
    one_over_rho = problem_dtype(1.0) / rho

    return rho, vec3((X_P1 - X_M1) * one_over_rho, (Y_P1 - Y_M1) * one_over_rho, (Z_P1 - Z_M1) * one_over_rho)


@wp.kernel
def collide_and_stream(omega_in: float, lid_vel_in: float, domain_data: LBMVars, do_output: bool):
    i, j, k = wp.tid()

    omega = problem_dtype(omega_in)
    lid_vel = problem_dtype(lid_vel_in)

    boundary = domain_data.boundary_flags[i, j, k]

    if boundary == BOUNCE or boundary == MOVING_WALL:
        return

    finl = streaming(domain_data.f0, i, j, k)

    if boundary == WALL or boundary == WALL_M:
        where_bitflag = domain_data.boundary_values[i, j, k]
        finl = streaming_bounce(domain_data.f0, i, j, k, finl, where_bitflag)

    if boundary == WALL_M:
        vel_dir = domain_data.boundary_dirs[i, j, k]
        finl = streaming_wall2(finl, vel_dir, lid_vel)

    rho, uvec = calc_macroscopic(finl)

    if boundary < BOUNCE:
        feq = get_equilibrium(rho, uvec)

        # BGK collision model
        finl = (problem_dtype(1) - omega) * finl + omega * feq

        if do_output:
            domain_data.u_star[i, j, k] = uvec

    for velocity_dir in range(19):
        domain_data.f1[velocity_dir, i, j, k] = finl[velocity_dir]


@wp.kernel
def export_macroscopic_from_f0(domain_data: LBMVars):
    i, j, k = wp.tid()

    boundary = domain_data.boundary_flags[i, j, k]

    f_array = domain_data.f0

    fvec = vec19(
        f_array[0, i, j, k],
        f_array[1, i, j, k],
        f_array[2, i, j, k],
        f_array[3, i, j, k],
        f_array[4, i, j, k],
        f_array[5, i, j, k],
        f_array[6, i, j, k],
        f_array[7, i, j, k],
        f_array[8, i, j, k],
        f_array[9, i, j, k],
        f_array[10, i, j, k],
        f_array[11, i, j, k],
        f_array[12, i, j, k],
        f_array[13, i, j, k],
        f_array[14, i, j, k],
        f_array[15, i, j, k],
        f_array[16, i, j, k],
        f_array[17, i, j, k],
        f_array[18, i, j, k],
    )

    rho, uvec = calc_macroscopic(fvec)

    domain_data.rho_star[i, j, k] = rho

    if boundary < BOUNCE:
        domain_data.u_star[i, j, k] = uvec
    else:
        domain_data.u_star[i, j, k] = vec3()


@wp.kernel
def calc_energy(u_star: wp.array3d(dtype=vec3), total_energy: wp.array(dtype=problem_dtype)):
    i, j, k = wp.tid()

    u_star_val = u_star[i, j, k]

    energy_element = wp.dot(u_star_val, u_star_val)

    t = wp.tile(energy_element)
    s = wp.tile_sum(t)

    wp.tile_atomic_add(total_energy, s)


class Example:
    def __init__(self, resolution):
        self.N = resolution

        # Global resolution
        self.domain = Domain()
        self.domain.nx = self.N
        self.domain.ny = self.N
        self.domain.nz = self.N

        wp.set_mempool_enabled(wp.get_cuda_device(), True)
        wp.set_mempool_release_threshold(wp.get_cuda_device(), 1)

        self.shape = (resolution, resolution, resolution)  # Shape of 3-D fields that have a halo
        self.fshape = (19, resolution, resolution, resolution)

        self.lbm_vars = LBMVars()

        self.lbm_vars.u = wp.zeros(self.shape, dtype=vec3)

        # rho_star and u_star are purely for output
        self.lbm_vars.rho_star = wp.empty(self.shape, dtype=problem_dtype)
        self.lbm_vars.u_star = wp.zeros(self.shape, dtype=vec3)

        self.lbm_vars.f0 = wp.zeros(self.fshape, dtype=problem_dtype)
        self.lbm_vars.f1 = wp.zeros(self.fshape, dtype=problem_dtype)

        self.lbm_vars.boundary_flags = wp.full(self.shape, FLUID)
        self.lbm_vars.boundary_values = wp.zeros(self.shape, dtype=wp.int32)
        self.lbm_vars.boundary_dirs = wp.zeros(self.shape, dtype=wp.int32)

        self.outer_wall_bc_config = wp.array([BOUNCE, BOUNCE, BOUNCE, BOUNCE, MOVING_WALL, BOUNCE], dtype=wp.int32)

        self.bounds_priority = wp.from_numpy(bounds_priority_np, dtype=wp.int8)

        self.total_energy_d = wp.zeros(1, dtype=problem_dtype)
        self.total_energy_h = wp.zeros(1, dtype=problem_dtype, device="cpu")

        wp.launch(
            mark_flags,
            self.shape,
            inputs=[
                self.domain,
                self.outer_wall_bc_config,
                self.bounds_priority,
                self.lbm_vars.boundary_flags,
                self.lbm_vars.boundary_values,
                self.lbm_vars.boundary_dirs,
            ],
        )

        lattice_velocities_np = np.array(
            [
                [0, 0, 0],  # 0
                [-1, 0, 0],  # 1
                [0, -1, 0],  # 2
                [0, 0, -1],  # 3
                [-1, -1, 0],  # 4
                [-1, 1, 0],  # 5
                [-1, 0, -1],  # 6
                [-1, 0, 1],  # 7
                [0, -1, -1],  # 8
                [0, -1, 1],  # 9
                [1, 0, 0],  # 10
                [0, 1, 0],  # 11
                [0, 0, 1],  # 12
                [1, 1, 0],  # 13
                [1, -1, 0],  # 14
                [1, 0, 1],  # 15
                [1, 0, -1],  # 16
                [0, 1, 1],  # 17
                [0, 1, -1],  # 18
            ],
            dtype=np.int32,
        )

        self.up_x_indices = np.where(lattice_velocities_np[:, 0] == 1)[0].tolist()
        self.down_x_indices = np.where(lattice_velocities_np[:, 0] == -1)[0].tolist()

        LATTICE_VELOCITIES = wp.array(lattice_velocities_np, dtype=wp.vec3b)

        wp.launch(
            find_wall,
            self.shape,
            inputs=[
                LATTICE_VELOCITIES,
                self.lbm_vars.boundary_flags,
                self.lbm_vars.boundary_values,
                self.lbm_vars.boundary_dirs,
            ],
        )

        self.ulb = 0.02  # Lid velocity
        self.dx = 1.0 / (self.N - 2.0)
        self.dt = self.dx * self.ulb

        Re = 100.0
        nu = self.ulb * (self.N - 2.0) / Re
        self.omega = 1.0 / (3.0 * nu + 0.5)

    def initial_conditions(self):
        # Initialization of the populations
        wp.launch(init_velocity, self.shape, inputs=[problem_dtype(1), vec3(0, 0, 0), self.lbm_vars])

    def run_benchmark(self, max_iter=2000):
        bench_ini_iter = 1000
        output_frame = 2000

        self.initial_conditions()

        num_bench_iter = 0

        start_time = 0

        print(f"Starting {bench_ini_iter} warmup iterations")

        for iter in range(max_iter):
            do_output = iter < bench_ini_iter and iter > 0 and (iter % output_frame == 0 or iter == 149)

            if iter == bench_ini_iter:
                print(f"Starting {max_iter - bench_ini_iter} benchmark iterations")
                wp.synchronize_device()
                start_time = timeit.default_timer()

            if iter >= bench_ini_iter:
                num_bench_iter += 1

            # LBM collision-streaming cycle, in parallel over every cell. Stream precedes collision.
            wp.launch(
                collide_and_stream,
                self.shape,
                inputs=[self.omega, self.ulb, self.lbm_vars, do_output],
            )

            # Swap populations pointer, f0 are population to be read and f1 the population to be written
            # this is the double population scheme.
            (self.lbm_vars.f0, self.lbm_vars.f1) = (self.lbm_vars.f1, self.lbm_vars.f0)

            # Output average kinetic energy for validation.
            if do_output:
                wp.launch(calc_energy, self.shape, inputs=[self.lbm_vars.u_star, self.total_energy_d])

                wp.copy(self.total_energy_h, self.total_energy_d)

                actual_energy = 0.5 * self.total_energy_h.numpy()[0] * self.dx * self.dx / (self.dt * self.dt)

                print(f"Energy {actual_energy} iteration {iter}")

                self.total_energy_d.zero_()

        wp.synchronize_device()
        elapsed_secs = timeit.default_timer() - start_time
        mlups = self.domain.nx * self.domain.ny * self.domain.nz * num_bench_iter / elapsed_secs / 1e6

        print(f"MLups {mlups}")

        return mlups


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-N", required=True, help="Grid resolution", type=int)
    parser.add_argument("--max-iter", default=2000, help="Maximum number of solve iterations per run", type=int)
    parser.add_argument("--repeat", default=1, help="Number of times to repeat the benchmark", type=int)
    parser.add_argument("--mode", choices=("benchmark", "visualize", "time_history"), default="benchmark")

    args = parser.parse_args()

    example = Example(args.N)

    mlups = [None] * args.repeat
    for run_index in range(len(mlups)):
        mlups[run_index] = example.run_benchmark(args.max_iter)

    print("RESULT: MLUPS")
    for run_index in range(len(mlups)):
        print(f"{mlups[run_index]:.4f}")
