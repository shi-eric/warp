
import warp as wp

@wp.kernel
def kernel(
    seed: int,
    positions: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    """Noise + random — uses noise.h, rand.h, vec.h."""
    tid = wp.tid()
    state = wp.rand_init(seed, tid)
    p = positions[tid] + wp.vec3(wp.randf(state), wp.randf(state), wp.randf(state))
    output[tid] = wp.noise(wp.uint32(seed), p[0]) + wp.noise(wp.uint32(seed), p[1])
