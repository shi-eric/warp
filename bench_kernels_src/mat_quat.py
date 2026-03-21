
import warp as wp

@wp.kernel
def kernel(
    positions: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quatf),
    transforms: wp.array(dtype=wp.mat44),
    output: wp.array(dtype=wp.vec3),
):
    """Matrix + quaternion — uses mat.h, quat.h, vec.h."""
    tid = wp.tid()
    p = positions[tid]
    q = rotations[tid]
    m = transforms[tid]
    rotated = wp.quat_rotate(q, p)
    output[tid] = wp.transform_point(m, rotated)
