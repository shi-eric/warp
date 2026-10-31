Add experimental support for calling NVSHMEM from Warp kernels on Linux. New `wp.nvshmem_*` built-ins query
processing elements (PEs), read and write single elements or whole arrays on other PEs, and signal and synchronize
between PEs, for arrays of any dtype. They operate on NVSHMEM symmetric memory allocated with nvshmem4py and wrapped
as a `wp.array`. Warp links the NVSHMEM device library of the installed NVSHMEM at runtime, so an NVSHMEM 3.9 or
newer wheel is required; `WARP_NVSHMEM_DEVICE_LIBRARY` selects a device library explicitly. The element and bulk write
built-ins are differentiable: the backward pass pulls the destination's gradient from the remote processing element,
Warp synchronizes the adjoint launches of such kernels with an NVSHMEM barrier (`wp.config.nvshmem_adjoint_barrier`),
and gradient arrays that other processing elements read must be allocated in symmetric memory.
