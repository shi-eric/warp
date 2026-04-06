
#define WP_TILE_BLOCK_DIM 256
#define WP_NO_CRT
#define WP_NO_BVH
#define WP_NO_FABRIC
#define WP_NO_FLOAT16_OPS
#define WP_NO_FLOAT64_OPS
#define WP_NO_HASHGRID
#define WP_NO_INTERSECT
#define WP_NO_MAT
#define WP_NO_MATNN
#define WP_NO_MESH
#define WP_NO_NOISE
#define WP_NO_QUAT
#define WP_NO_RAND
#define WP_NO_SVD
#define WP_NO_TEXTURE
#define WP_NO_TILE
#define WP_NO_VEC
#define WP_NO_VOLUME
#include "builtin.h"

// avoid namespacing of float type for casting to float type, this is to avoid wp::float(x), which is not valid in C++
#define float(x) cast_float(x)
#define adj_float(x, adj_x, adj_ret) adj_cast_float(x, adj_x, adj_ret)

#define int(x) cast_int(x)
#define adj_int(x, adj_x, adj_ret) adj_cast_int(x, adj_x, adj_ret)

#define builtin_tid1d() wp::tid(task_index, dim)
#define builtin_tid2d(x, y) wp::tid(x, y, task_index, dim)
#define builtin_tid3d(x, y, z) wp::tid(x, y, z, task_index, dim)
#define builtin_tid4d(x, y, z, w) wp::tid(x, y, z, w, task_index, dim)

#define builtin_block_dim() wp::block_dim()

struct wp_args_array2d_augassign_kernel_ed3ca2e3 {
    wp::array_t<wp::float32> x;
    wp::array_t<wp::float32> y;
};


void array2d_augassign_kernel_ed3ca2e3_cpu_kernel_forward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_array2d_augassign_kernel_ed3ca2e3 *_wp_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_x = _wp_args->x;
    wp::array_t<wp::float32> var_y = _wp_args->y;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // forward
    // def array2d_augassign_kernel(                                                          <L 42>
    // i, j = wp.tid()                                                                        <L 46>
    builtin_tid2d(var_0, var_1);
    // x[i, j] += y[i, j]                                                                     <L 47>
    var_2 = wp::address(var_y, var_0, var_1);
    var_4 = wp::load(var_2);
    var_3 = wp::atomic_add(var_x, var_0, var_1, var_4);
}



void array2d_augassign_kernel_ed3ca2e3_cpu_kernel_backward(
    wp::launch_bounds_t dim,
    size_t task_index,
    wp_args_array2d_augassign_kernel_ed3ca2e3 *_wp_args,
    wp_args_array2d_augassign_kernel_ed3ca2e3 *_wp_adj_args)
{
    //---------
    // argument vars
    wp::array_t<wp::float32> var_x = _wp_args->x;
    wp::array_t<wp::float32> var_y = _wp_args->y;
    wp::array_t<wp::float32> adj_x = _wp_adj_args->x;
    wp::array_t<wp::float32> adj_y = _wp_adj_args->y;
    //---------
    // primal vars
    wp::int32 var_0;
    wp::int32 var_1;
    wp::float32* var_2;
    wp::float32 var_3;
    wp::float32 var_4;
    //---------
    // dual vars
    wp::int32 adj_0 = {};
    wp::int32 adj_1 = {};
    wp::float32 adj_2 = {};
    wp::float32 adj_3 = {};
    wp::float32 adj_4 = {};
    //---------
    // forward
    // def array2d_augassign_kernel(                                                          <L 42>
    // i, j = wp.tid()                                                                        <L 46>
    builtin_tid2d(var_0, var_1);
    // x[i, j] += y[i, j]                                                                     <L 47>
    var_2 = wp::address(var_y, var_0, var_1);
    var_4 = wp::load(var_2);
    // var_3 = wp::atomic_add(var_x, var_0, var_1, var_4);
    //---------
    // reverse
    wp::adj_atomic_add(var_x, var_0, var_1, var_4, adj_x, adj_0, adj_1, adj_2, adj_3);
    wp::adj_address(var_y, var_0, var_1, adj_y, adj_0, adj_1, adj_2);
    // adj: x[i, j] += y[i, j]                                                                <L 47>
    // adj: i, j = wp.tid()                                                                   <L 46>
    // adj: def array2d_augassign_kernel(                                                     <L 42>
    return;
}



extern "C" {

// Python CPU entry points
WP_API void array2d_augassign_kernel_ed3ca2e3_cpu_forward(
    wp::launch_bounds_t dim,
    wp_args_array2d_augassign_kernel_ed3ca2e3 *_wp_args)
{
#ifndef WP_NO_TILE
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        array2d_augassign_kernel_ed3ca2e3_cpu_kernel_forward(dim, task_index, _wp_args);
    }
}

} // extern C



extern "C" {

WP_API void array2d_augassign_kernel_ed3ca2e3_cpu_backward(
    wp::launch_bounds_t dim,
    wp_args_array2d_augassign_kernel_ed3ca2e3 *_wp_args,
    wp_args_array2d_augassign_kernel_ed3ca2e3 *_wp_adj_args)
{
#ifndef WP_NO_TILE
    wp::tile_shared_storage_t tile_mem;
#if defined(WP_ENABLE_TILES_IN_STACK_MEMORY)
    wp::shared_tile_storage = &tile_mem;
#endif
#endif

    for (size_t task_index = 0; task_index < dim.size; ++task_index)
    {
        array2d_augassign_kernel_ed3ca2e3_cpu_kernel_backward(dim, task_index, _wp_args, _wp_adj_args);
    }
}

} // extern C

