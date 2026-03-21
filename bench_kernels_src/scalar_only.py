
import warp as wp

@wp.kernel
def kernel(x: wp.array2d(dtype=float), y: wp.array2d(dtype=float)):
    """Scalar math only — best case for guards."""
    i, j = wp.tid()
    x[i, j] += y[i, j]
