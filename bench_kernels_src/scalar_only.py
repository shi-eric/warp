"""Scalar-only kernel — exercises no vec/mat/quat headers."""
import warp as wp

@wp.kernel
def kernel(
    x: wp.array2d(dtype=wp.float64),
    y: wp.array2d(dtype=wp.float64),
):
    i, j = wp.tid()
    x[i, j] += y[i, j]
