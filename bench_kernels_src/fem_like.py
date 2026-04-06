"""FEM-like kernel — exercises vec2, mat22, and mat operations with enable_backward=False.

Mimics the compile-time profile of warp/examples/fem/example_navier_stokes.py:
vec2 fields, strain tensors (mat22 via D operator), dot/ddot products, and
spatial lookups. Uses enable_backward=False so adjoint header functions
become pure parsing overhead — making this benchmark sensitive to whether
adj_* code is conditionally excluded from the headers.
"""

import warp as wp

wp.set_module_options({"enable_backward": False})


@wp.func
def strain_rate(grad_u: wp.mat22) -> wp.mat22:
    """Symmetric part of velocity gradient (like fem.D on a vec2 field)."""
    return 0.5 * (grad_u + wp.transpose(grad_u))


@wp.func
def velocity_at(pos: wp.vec2, phase: float) -> wp.vec2:
    """Synthetic velocity field (avoids needing actual FEM infrastructure)."""
    return wp.vec2(wp.sin(pos[0] + phase), wp.cos(pos[1] + phase))


@wp.kernel
def kernel(
    positions: wp.array(dtype=wp.vec2),
    velocities: wp.array(dtype=wp.vec2),
    grad_u: wp.array(dtype=wp.mat22),
    viscosity: float,
    dt: float,
    out_force: wp.array(dtype=wp.vec2),
):
    i = wp.tid()
    pos = positions[i]
    vel = velocities[i]

    # Strain tensor from velocity gradient (like fem.D)
    eps = strain_rate(grad_u[i])

    # Viscous stress: 2 * nu * D(u) : D(v) (like viscosity_form integrand)
    stress = 2.0 * viscosity * eps

    # Force from stress divergence (simplified)
    fx = stress[0, 0] + stress[0, 1]
    fy = stress[1, 0] + stress[1, 1]

    # Semi-Lagrangian advection (like transported_inertia_form)
    conv_pos = pos - 0.5 * vel * dt
    conv_vel = velocity_at(conv_pos, dt)
    conv_pos = conv_pos - 0.5 * conv_vel * dt
    final_vel = velocity_at(conv_pos, dt)

    out_force[i] = final_vel / dt - wp.vec2(fx, fy)
