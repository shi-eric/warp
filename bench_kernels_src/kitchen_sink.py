
import warp as wp

@wp.kernel
def kernel(
    mesh_id: wp.uint64,
    volume: wp.uint64,
    seed: int,
    points: wp.array(dtype=wp.vec3),
    rotations: wp.array(dtype=wp.quatf),
    matrices: wp.array(dtype=wp.mat44),
    output: wp.array(dtype=float),
):
    """All features — mesh, volume, noise, rand, vec, mat, quat."""
    tid = wp.tid()
    p = points[tid]
    q = rotations[tid]
    m = matrices[tid]
    p2 = wp.transform_point(m, wp.quat_rotate(q, p))
    sign = float(0.0)
    f = int(0)
    u = float(0.0)
    v = float(0.0)
    wp.mesh_query_point(mesh_id, p2, 1000.0, sign, f, u, v)
    vol_val = wp.volume_sample_f(volume, p2, wp.Volume.LINEAR)
    state = wp.rand_init(seed, tid)
    n = wp.noise(wp.uint32(seed), p2[0] + wp.randf(state))
    output[tid] = vol_val + n + sign
