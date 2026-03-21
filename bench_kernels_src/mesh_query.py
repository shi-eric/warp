
import warp as wp

@wp.kernel
def kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    distances: wp.array(dtype=float),
):
    """Mesh queries — uses mesh.h, bvh.h, intersect.h, tile.h, vec.h, mat.h."""
    tid = wp.tid()
    p = points[tid]
    sign = float(0.0)
    f = int(0)
    u = float(0.0)
    v = float(0.0)
    has_point = wp.mesh_query_point(mesh_id, p, 1000.0, sign, f, u, v)
    if has_point:
        closest = wp.mesh_eval_position(mesh_id, f, u, v)
        distances[tid] = wp.length(p - closest) * sign
