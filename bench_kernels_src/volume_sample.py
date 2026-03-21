
import warp as wp

@wp.kernel
def kernel(
    volume: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    output: wp.array(dtype=float),
):
    """Volume sampling — uses volume.h, vec.h, mat.h."""
    tid = wp.tid()
    p = points[tid]
    output[tid] = wp.volume_sample_f(volume, p, wp.Volume.LINEAR)
