
import warp as wp

@wp.kernel
def kernel(
    points: wp.array(dtype=wp.vec3),
    normals: wp.array(dtype=wp.vec3),
    result: wp.array(dtype=float),
):
    """Vector math — uses vec.h."""
    tid = wp.tid()
    p = points[tid]
    n = normals[tid]
    result[tid] = wp.dot(p, n) / (wp.length(p) + 1e-6)
