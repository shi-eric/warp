Fix compile errors when a `@wp.func` passes a register tile to a tile function that needs shared memory,
such as `wp.tile_cholesky()` or `wp.tile_matmul()`,
and wrong gradients on CUDA when a `@wp.func` reads elements (`t[i, j]`) of a register tile passed to it.
The caller's tile now moves to shared memory, as it does when the tile function is called directly in the kernel.
