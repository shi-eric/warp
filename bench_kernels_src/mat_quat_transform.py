"""Mat + quat + transform kernel — exercises mat.h, quat.h, spatial.h."""
import warp as wp

@wp.kernel
def kernel(
    transforms: wp.array(dtype=wp.transform),
    points: wp.array(dtype=wp.vec3),
    out: wp.array(dtype=wp.vec3),
):
    i = wp.tid()
    t = transforms[i]
    p = points[i]
    out[i] = wp.transform_point(t, p)
