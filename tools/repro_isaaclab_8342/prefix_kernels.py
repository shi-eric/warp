"""Small Warp kernels for reducing the graph-capture workload."""

import warp as wp


@wp.kernel
def prefix(values: wp.array2d[float]):
    world, component = wp.tid()
    values[world, component] = values[world, component] + 1.0
