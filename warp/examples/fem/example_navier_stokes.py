# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

###########################################################################
# Example Navier Stokes
#
# This example solves a 2D Navier-Stokes flow problem:
#
# Du/dt -nu D(u) + grad p = 0
# Div u = 0
#
# with (hard) velocity-Dirichlet boundary conditions
# and using semi-Lagrangian advection
###########################################################################

import warp as wp
import warp.fem as fem
from warp.fem.utils import array_axpy
from warp.sparse import bsr_copy, bsr_mm, bsr_mv

try:
    from .bsr_utils import SaddleSystem, bsr_solve_saddle
    from .mesh_utils import gen_trimesh
    from .plot_utils import Plot
except ImportError:
    from bsr_utils import SaddleSystem, bsr_solve_saddle
    from mesh_utils import gen_trimesh
    from plot_utils import Plot

wp.init()


@fem.integrand
def u_boundary_value(s: fem.Sample, domain: fem.Domain, v: fem.Field, top_vel: float):
    # Horizontal velocity on top of domain, zero elsewhere
    if domain(s)[1] == 1.0:
        return wp.dot(wp.vec2f(top_vel, 0.0), v(s))

    return wp.dot(wp.vec2f(0.0, 0.0), v(s))


@fem.integrand
def mass_form(
    s: fem.Sample,
    u: fem.Field,
    v: fem.Field,
):
    return wp.dot(u(s), v(s))


@fem.integrand
def inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float):
    return mass_form(s, u, v) / dt


@fem.integrand
def viscosity_form(s: fem.Sample, u: fem.Field, v: fem.Field, nu: float):
    return 2.0 * nu * wp.ddot(fem.D(u, s), fem.D(v, s))


@fem.integrand
def viscosity_and_inertia_form(s: fem.Sample, u: fem.Field, v: fem.Field, dt: float, nu: float):
    return inertia_form(s, u, v, dt) + viscosity_form(s, u, v, nu)


@fem.integrand
def transported_inertia_form(s: fem.Sample, domain: fem.Domain, u: fem.Field, v: fem.Field, dt: float):
    pos = domain(s)
    vel = u(s)

    conv_pos = pos - 0.5 * vel * dt
    conv_s = fem.lookup(domain, conv_pos, s)
    conv_vel = u(conv_s)

    conv_pos = conv_pos - 0.5 * conv_vel * dt
    conv_vel = u(fem.lookup(domain, conv_pos, conv_s))

    return wp.dot(conv_vel, v(s)) / dt


@fem.integrand
def div_form(
    s: fem.Sample,
    u: fem.Field,
    q: fem.Field,
):
    return -q(s) * fem.div(u, s)


class Example:
    def __init__(self, quiet=False, degree=2, resolution=25, Re=1000.0, top_velocity=1.0, tri_mesh=False):
        self._quiet = quiet

        res = resolution
        self.sim_dt = 1.0 / resolution
        self.current_frame = 0

        viscosity = top_velocity / Re

        if tri_mesh:
            positions, tri_vidx = gen_trimesh(res=wp.vec2i(res))
            geo = fem.Trimesh2D(tri_vertex_indices=tri_vidx, positions=positions)
        else:
            geo = fem.Grid2D(res=wp.vec2i(res))

        domain = fem.Cells(geometry=geo)
        boundary = fem.BoundarySides(geo)

        # Functions spaces: Q(d)-Q(d-1)
        u_degree = degree
        u_space = fem.make_polynomial_space(geo, degree=u_degree, dtype=wp.vec2)
        p_space = fem.make_polynomial_space(geo, degree=u_degree - 1)

        # Viscosity and inertia
        u_test = fem.make_test(space=u_space, domain=domain)
        u_trial = fem.make_trial(space=u_space, domain=domain)

        u_matrix = fem.integrate(
            viscosity_and_inertia_form,
            fields={"u": u_trial, "v": u_test},
            values={"nu": viscosity, "dt": self.sim_dt},
        )

        # Pressure-velocity coupling
        p_test = fem.make_test(space=p_space, domain=domain)
        div_matrix = fem.integrate(div_form, fields={"u": u_trial, "q": p_test})

        # Enforcing the Dirichlet boundary condition the hard way;
        # build projector for velocity left- and right-hand-sides
        u_bd_test = fem.make_test(space=u_space, domain=boundary)
        u_bd_trial = fem.make_trial(space=u_space, domain=boundary)
        u_bd_projector = fem.integrate(mass_form, fields={"u": u_bd_trial, "v": u_bd_test}, nodal=True)
        u_bd_value = fem.integrate(
            u_boundary_value,
            fields={"v": u_bd_test},
            values={"top_vel": top_velocity},
            nodal=True,
            output_dtype=wp.vec2d,
        )

        fem.normalize_dirichlet_projector(u_bd_projector, u_bd_value)

        u_bd_rhs = wp.zeros_like(u_bd_value)
        fem.project_linear_system(u_matrix, u_bd_rhs, u_bd_projector, u_bd_value, normalize_projector=False)

        # div_bd_rhs = div_matrix * u_bd_rhs
        div_bd_rhs = wp.zeros(shape=(div_matrix.nrow,), dtype=div_matrix.scalar_type)
        bsr_mv(div_matrix, u_bd_value, y=div_bd_rhs, alpha=-1.0)

        # div_matrix = div_matrix - div_matrix * bd_projector
        bsr_mm(x=bsr_copy(div_matrix), y=u_bd_projector, z=div_matrix, alpha=-1.0, beta=1.0)

        # Assemble saddle system
        self._saddle_system = SaddleSystem(u_matrix, div_matrix)

        # Save data for computing time steps rhs
        self._u_bd_projector = u_bd_projector
        self._u_bd_rhs = u_bd_rhs
        self._u_test = u_test
        self._div_bd_rhs = div_bd_rhs

        # Velocitiy and pressure fields
        self._u_field = u_space.make_field()
        self._p_field = p_space.make_field()

        self.renderer = Plot()
        self.renderer.add_surface_vector("velocity", self._u_field)

    def step(self):
        self.current_frame += 1

        u_rhs = fem.integrate(
            transported_inertia_form,
            fields={"u": self._u_field, "v": self._u_test},
            values={"dt": self.sim_dt},
            output_dtype=wp.vec2d,
        )

        # Apply boundary conditions
        # u_rhs = (I - P) * u_rhs + u_bd_rhs
        bsr_mv(self._u_bd_projector, x=u_rhs, y=u_rhs, alpha=-1.0, beta=1.0)
        array_axpy(x=self._u_bd_rhs, y=u_rhs, alpha=1.0, beta=1.0)

        p_rhs = self._div_bd_rhs

        x_u = wp.empty_like(u_rhs)
        x_p = wp.empty_like(p_rhs)
        wp.utils.array_cast(out_array=x_u, in_array=self._u_field.dof_values)
        wp.utils.array_cast(out_array=x_p, in_array=self._p_field.dof_values)

        bsr_solve_saddle(
            saddle_system=self._saddle_system,
            tol=1.0e-6,
            x_u=x_u,
            x_p=x_p,
            b_u=u_rhs,
            b_p=p_rhs,
            quiet=self._quiet,
        )

        wp.utils.array_cast(in_array=x_u, out_array=self._u_field.dof_values)
        wp.utils.array_cast(in_array=x_p, out_array=self._p_field.dof_values)

    def render(self):
        self.renderer.begin_frame(time=self.current_frame * self.sim_dt)
        self.renderer.add_surface_vector("velocity", self._u_field)
        self.renderer.end_frame()


if __name__ == "__main__":
    import argparse

    wp.set_module_options({"enable_backward": False})

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--device", type=str, default=None, help="Override the default Warp device.")
    parser.add_argument("--resolution", type=int, default=25, help="Grid resolution.")
    parser.add_argument("--degree", type=int, default=2, help="Polynomial degree of shape functions.")
    parser.add_argument("--num_frames", type=int, default=1000, help="Total number of frames.")
    parser.add_argument(
        "--top_velocity",
        type=float,
        default=1.0,
        help="Horizontal velocity boundary condition at the top of the domain.",
    )
    parser.add_argument("--Re", type=float, default=1000.0, help="Reynolds number.")
    parser.add_argument("--tri_mesh", action="store_true", help="Use a triangular mesh.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run in headless mode, suppressing the opening of any graphical windows.",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppresses the printing out of iteration residuals.")

    args = parser.parse_known_args()[0]

    with wp.ScopedDevice(args.device):
        example = Example(
            quiet=args.quiet,
            degree=args.degree,
            resolution=args.resolution,
            Re=args.Re,
            top_velocity=args.top_velocity,
            tri_mesh=args.tri_mesh,
        )

        for k in range(args.num_frames):
            print(f"Frame {k}:")
            example.step()
            example.render()

        example.renderer.add_surface_vector("velocity_final", example._u_field)

        if not args.headless:
            example.renderer.plot(streamlines={"velocity_final"})
