"""Vector math kernel — exercises vec.h."""
import warp as wp

@wp.kernel
def kernel(
    positions: wp.array(dtype=wp.vec3),
    velocities: wp.array(dtype=wp.vec3),
    dt: float,
):
    i = wp.tid()
    positions[i] = positions[i] + velocities[i] * dt
