Compilation and startup performance
===================================

.. currentmodule:: warp

Warp compiles kernels just in time the first time a module is loaded for a
device. Warm runs are usually much faster because Warp reuses cached generated
source and compiled binaries, but applications with many kernels, many generic
instances, or MathDx-backed tile operations can still spend noticeable time in
startup or cold compilation.

Warm-cache benefits assume :attr:`warp.config.cache_kernels` remains enabled
and the cache directory persists across application runs. This is usually the
default, but containerized, CI, or serverless deployments may need to set
``WARP_CACHE_PATH`` or :attr:`warp.config.kernel_cache_dir` to a persistent
location before :func:`wp.init <warp.init>`.

This page is a practical decision guide for reducing startup latency. It links
to the deeper documentation for each mechanism rather than replacing it.

Start with measurement
----------------------

Before changing code or configuration, first check the module-load output. By
default, Warp prints a line showing the module name, target device, elapsed load
time, and whether the module was compiled or loaded from cache. These lines
usually identify which module is expensive and whether startup time is coming
from cold compilation or warm-cache loading.

Set :attr:`warp.config.log_level` to ``wp.LOG_DEBUG`` to also print when each
module load begins, along with extra details such as ``block_dim``. This helps
identify which module is currently compiling during a long startup pause.

When measuring true cold-start compilation, account for all relevant caches:

- Warp's kernel cache, cleared with :func:`warp.clear_kernel_cache`.
- Warp's LTO cache for tile-based MathDx operations, cleared with
  :func:`warp.clear_lto_cache`.
- The NVIDIA CUDA driver compute cache, which is separate from Warp's caches.

See :ref:`benchmarking-cold-start-compilation` for the full benchmarking
workflow and deeper compile-time tracing when module-load timing is not enough.

Decision table
--------------

.. list-table::
   :header-rows: 1
   :widths: 24 17 23 18 28

   * - Technique
     - Use
     - Scope
     - Expected benefit
     - Tradeoff
   * - Define kernels before first module load
     - Production
     - General
     - Avoids repeated recompiles as a module's kernel set grows
     - Requires organizing imports and moving avoidable runtime kernel creation
       out of hot paths
   * - Explicitly overload generic kernels
     - Production
     - General
     - Avoids repeated generic instantiation and module reloads
     - More boilerplate; avoid overloads for type combinations that are never
       launched
   * - Keep runtime specializations intentional
     - Production
     - General; especially tiled code
     - Avoids extra module and LTO compile work from unused block sizes, tile
       shapes, dtypes, or layouts
     - Requires deciding which runtime-specialized cases the application
       actually launches
   * - Keep module options stable
     - Production
     - General
     - Improves cache reuse by avoiding unnecessary module hash changes
     - Requires choosing compile options before loading latency-sensitive
       modules
   * - Preload modules with :func:`wp.load_module <warp.load_module>`
     - Production
     - General
     - Moves compilation out of latency-sensitive code paths
     - Does not reduce total compile work unless combined with better module
       organization or parallel loading
   * - Use parallel module loading
     - Production, after measuring
     - General
     - Can reduce wall-clock load time when several modules compile
       independently
     - Limited by serial work and module imbalance; too many small modules can
       add overhead
   * - Compile modules ahead of time
     - Production
     - General; CUDA architecture choices matter
     - Moves compilation into a build or deployment step
     - Requires managing generated artifacts, architectures, and PTX versus
       CUBIN portability
   * - Disable backward compilation when gradients are not needed
     - Production if autodiff is not used
     - General
     - Reduces generated code for kernels that would otherwise include adjoint
       paths
     - Gradients through those kernels are unavailable or invalid
   * - Limit excessive static unrolling
     - Production, after measuring
     - General
     - Reduces generated source size for kernels with large static loops
     - May reduce runtime performance if loops stop unrolling profitably
   * - Lower :attr:`warp.config.optimization_level`
     - Development iteration
     - Mostly CUDA; measure CPU impact
     - Can reduce compile time while iterating
     - Runtime performance may drop; restore production settings before
       benchmarking or shipping
   * - Disable MathDx-backed tile paths temporarily
     - Development iteration
     - CUDA tile operations
     - Avoids expensive MathDx LTO compilation for affected tile operations
     - Runtime performance may drop and some fallback behavior is more limited
   * - Keep debug and profiling compile flags off unless needed
     - Development hygiene
     - General; flag-dependent
     - Avoids extra generated code, metadata, or tracing overhead
     - Less diagnostic information is available until the flags are re-enabled

Define kernels before first module load
---------------------------------------

Warp compiles kernels by module. The module hash includes the live kernels,
generic kernel instances, structs, functions, and module options. If a module
has already loaded and then a new kernel or generic overload is added to that
module, the hash changes and Warp must compile or load another module variant
the next time a kernel from that module launches.

The issue is not dynamic definition by itself. Kernels defined in a loop or
helper before the first :func:`wp.launch <warp.launch>` or
:func:`wp.load_module <warp.load_module>` that loads the module are part of the
first module hash. The costly pattern is introducing avoidable new kernels or
overloads after the module has already loaded, especially in startup or frame
paths that define a kernel immediately before launching it.

Prefer this structure:

.. code-block:: python

    import warp as wp


    @wp.kernel
    def first_kernel(values: wp.array[float]):
        values[wp.tid()] *= 2.0


    @wp.kernel
    def second_kernel(values: wp.array[float]):
        values[wp.tid()] += 1.0


    def run(values):
        wp.launch(first_kernel, dim=values.shape, inputs=[values])
        wp.launch(second_kernel, dim=values.shape, inputs=[values])

Once the module's kernel set is stable, later launches can reuse the loaded
module or warm cache. If the application genuinely needs runtime-specialized
kernels, try to create the expected specializations before latency-sensitive
work begins.

For generic kernels, declare the type combinations you expect to launch before
the first launch:

.. code-block:: python

    from typing import Any

    import warp as wp


    @wp.kernel
    def scale(x: wp.array[Any], s: Any):
        i = wp.tid()
        x[i] = x[i] * s


    scale_f32 = wp.overload(scale, {"x": wp.array[wp.float32], "s": wp.float32})
    scale_f64 = wp.overload(scale, {"x": wp.array[wp.float64], "s": wp.float64})

Only declare overloads that the application actually uses. Excessive overloads
increase generated code and compile work.

Keep runtime specializations intentional
----------------------------------------

During ordinary runtime compilation, Warp chooses the CUDA target architecture
from the local device. Users usually do not need to manage GPU architecture
variants unless they are compiling modules ahead of time.

The runtime choices that can multiply compile work are the specializations an
application actually launches. Avoid unnecessary ``block_dim`` variants in
latency-sensitive code. Block size is a kernel and module compilation option,
so changing it can create extra compiled module and cache variants for
otherwise identical kernels.

For tile-heavy code, choose a small, stable set of supported tile shapes,
dtypes, and memory layouts. One-off combinations can create extra module work
and additional MathDx/LTO compilation for tile operations.

When preloading modules, preload only the variants the application will
actually launch. Preloading every possible ``block_dim`` or tile combination
can move compile work earlier without reducing the amount of work.

Choose stable compilation options
---------------------------------

Module options are part of the compiled module identity. Set options before
loading modules, and avoid toggling them repeatedly for the same module during
one run.

For kernels with large static loops, lowering ``max_unroll`` can reduce
generated source size:

.. code-block:: python

    wp.set_module_options({"max_unroll": 8})

Measure the runtime effect before keeping this in production.

Disable backward code generation when gradients are not needed
--------------------------------------------------------------

Warp generates backward code for kernels that can participate in automatic
differentiation. If an application does not need gradients for those kernels,
disable backward code generation to reduce generated code and compile work.

Disable it globally before module creation:

.. code-block:: python

    import warp as wp

    wp.config.enable_backward = False
    wp.init()

For modules that do not need gradients, set the module option before the module
loads:

.. code-block:: python

    import warp as wp

    wp.set_module_options({"enable_backward": False})

For individual kernels, use the decorator option:

.. code-block:: python

    @wp.kernel(enable_backward=False)
    def integrate(values: wp.array[float]):
        values[wp.tid()] += 1.0

Only disable backward generation for kernels whose adjoints will not be used by
:class:`Tape <warp.Tape>` or differentiable framework integrations.

Preload modules before latency-sensitive work
---------------------------------------------

Use :func:`wp.load_module <warp.load_module>` to compile and load modules at a
controlled point, such as application startup, before CUDA graph capture, or
before entering a frame loop:

.. code-block:: python

    import warp as wp
    import my_app.kernels

    wp.init()
    wp.load_module(my_app.kernels, device="cuda:0")

For packages with registered Warp submodules, ``recursive=True`` can load a
module hierarchy:

.. code-block:: python

    wp.load_module(my_app.kernels, device="cuda:0", recursive=True)

If there are several independent modules with similar compile costs, parallel
loading can reduce wall-clock startup time:

.. code-block:: python

    wp.load_module(my_app.kernels, device="cuda:0", recursive=True, max_workers=4)

The same default can be configured globally:

.. code-block:: python

    wp.config.load_module_max_workers = 4

Parallel loading is useful only when there is enough independent module work.
It does not help a program dominated by one large module, and it does not help
modules that are defined dynamically after preloading.

Use :func:`wp.force_load <warp.force_load>` when you need lower-level control
over a known set of Warp modules. Most applications should prefer
:func:`wp.load_module <warp.load_module>` because it accepts Python module
objects and supports recursive loading.

Consider module boundaries
--------------------------

Large modules are convenient, but a change to one kernel can invalidate the
whole module. Smaller modules can isolate expensive kernels, reduce cache
invalidation blast radius, and give parallel loading more independent work.

Splitting modules has costs:

- Shared functions and structs may be duplicated across generated modules.
- Too many tiny modules can add launch-time and load-time overhead.
- Runtime-created unique modules cannot always be preloaded by package name.

Use ``module="unique"`` for kernels that need a separate module identity, and
add ``module_options=...`` when that unique module needs different compilation
options:

.. code-block:: python

    @wp.kernel(module="unique", module_options={"enable_backward": False})
    def preprocessing_kernel(values: wp.array[float]):
        values[wp.tid()] = values[wp.tid()] * 0.5

This is most useful when the isolated kernel has different compilation options
or changes independently from the rest of the application.

Ahead-of-time and build-step compilation
----------------------------------------

For production deployments, compile stable modules during a build step and load
them at runtime:

.. code-block:: python

    # build_kernels.py
    import warp as wp
    import my_app.kernels

    wp.init()
    wp.compile_aot_module(
        my_app.kernels,
        arch=[80, 86, 90],
        module_dir="build/warp_modules",
        use_ptx=False,
    )

.. code-block:: python

    # run_app.py
    import warp as wp
    import my_app.kernels

    wp.init()
    wp.load_aot_module(my_app.kernels, module_dir="build/warp_modules", use_ptx=False)

CUBIN output avoids driver JIT compilation for matching GPU architectures, but
it is less portable. PTX is more portable across compatible GPUs, but the CUDA
driver may still JIT-compile it on first use.

Development iteration tradeoffs
-------------------------------

For rapid iteration, it can be reasonable to compile less optimized kernels and
restore production settings before performance benchmarking:

.. code-block:: python

    import warp as wp

    wp.config.optimization_level = 0

Lower optimization levels can reduce CUDA compile time, but they may reduce
kernel runtime performance. Measure before relying on this for CPU compile
time.

If a development workload uses MathDx-backed tile operations and the cold LTO
compile cost dominates iteration, temporarily use the Warp/native fallback
paths:

.. code-block:: python

    import warp as wp

    wp.config.enable_mathdx_gemm = False
    wp.config.enable_mathdx_solver = False
    wp.config.enable_mathdx_fft = False

These flags affect operations such as :func:`wp.tile_matmul <warp._src.lang.tile_matmul>`,
tile solvers, :func:`wp.tile_fft <warp._src.lang.tile_fft>`, and
:func:`wp.tile_ifft <warp._src.lang.tile_ifft>` when running on GPU with MathDx
available. The fallback paths can compile faster, but may run slower. FFT
fallback support is also more limited, such as requiring power-of-two sizes.

Keep these diagnostic or code-expanding options disabled unless needed:

- ``mode="debug"``
- :attr:`warp.config.lineinfo`
- :attr:`warp.config.compile_time_trace`
- :attr:`warp.config.enable_vector_component_overwrites`

Tile and MathDx notes
---------------------

MathDx-backed tile operations can generate LTO objects whose cold compile cost
is independent from ordinary module compilation. Warp stores these objects in a
dedicated LTO cache. Clear it with :func:`warp.clear_lto_cache` only when
benchmarking cold compilation or intentionally invalidating LTO artifacts.

The MathDx fallback flags are global settings and module options:

.. code-block:: python

    wp.config.enable_mathdx_gemm = False
    wp.config.enable_mathdx_solver = False
    wp.config.enable_mathdx_fft = False

.. code-block:: python

    wp.set_module_options(
        {
            "enable_mathdx_gemm": False,
            "enable_mathdx_solver": False,
            "enable_mathdx_fft": False,
        }
    )

Use module options when only one module should use fallback tile paths.

Where to go next
----------------

- :ref:`Compilation Model` explains Warp's Python-to-C++/CUDA compilation model
  and module-load output.
- :doc:`runtime` explains kernel-cache behavior and cache clearing.
- :doc:`configuration` lists global, module, and kernel options.
- :doc:`generics` explains implicit and explicit generic instantiation.
- :doc:`tiles` covers tile operations, MathDx requirements, and LTO errors.
- :ref:`code_generation` describes generated source, module hashing, and
  ahead-of-time workflows.
- :doc:`../deep_dive/profiling` describes compile-time tracing and cold-start
  measurement with :ref:`benchmarking-cold-start-compilation`.
