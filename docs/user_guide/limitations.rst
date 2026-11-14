Limitations
===========

.. currentmodule:: warp

This section summarizes various limitations and currently unsupported features in Warp.
Problems, questions, and feature requests can be opened on `GitHub Issues <https://github.com/NVIDIA/warp/issues>`_.

Unsupported Features
--------------------

Warp kernels and user functions do not support these Python features at runtime:

* Lambda functions
* List comprehensions
* Exceptions
* Recursion
* Runtime evaluation of expressions, e.g.: eval()
* Dynamic structures such as lists, sets, dictionaries, etc.

Python host code can use these features. Expressions evaluated by :func:`wp.static() <warp.static>`
can also use list comprehensions, lambda calls, and dictionary lookups when their inputs are known
at compile time and their result is a supported kernel value.

Kernels and User Functions
--------------------------

* Strings cannot be passed into kernels.
* :func:`wp.atomic_add() <warp._src.lang.atomic_add>` and :func:`wp.atomic_sub() <warp._src.lang.atomic_sub>`
  do not support :class:`wp.float16 <float16>` or :class:`wp.bfloat16 <bfloat16>`
  on GPUs with compute capability below 7.0.
  Kernels compiled for those GPUs skip these updates without raising an error.
* :func:`wp.atomic_add() <warp._src.lang.atomic_add>` and :func:`wp.atomic_sub() <warp._src.lang.atomic_sub>`
  do not support :class:`wp.float64 <float64>` on GPUs with compute capability below 6.0.
  Kernels compiled for those GPUs skip these updates without raising an error.
* Using ``wp.atomic_add()`` or related functions on the same memory address from
  overlapping CPU and GPU kernels is currently unsupported.
* :func:`wp.tid() <warp._src.lang.tid>` cannot be called from user functions.
* Modifying the value of a :func:`wp.constant() <warp.constant>` during runtime will not trigger
  recompilation of the affected kernels if the modules have already been loaded
  (e.g. through a :func:`wp.launch() <warp.launch>` or a :func:`wp.load_module() <warp.load_module>`).
* A :func:`wp.constant() <warp.constant>` used without an explicit type constructor is treated as
  :class:`wp.float32 <float32>` (for Python floats) or :class:`wp.int32 <int32>` (for Python integers).
  See :ref:`Constants <runtime-constants>` for preserving 64-bit precision.
* Python ``IntFlag`` values behave like raw integers in Warp kernels: bitwise inversion (``~``)
  produces the integer bitwise complement, not a masked combination of flags as in standard Python ``IntFlag`` behavior.
* :ref:`Function parameters <callable-parameters>` in user functions only support direct inline calls with
  user-defined :func:`@wp.func <warp.func>` functions and simple built-in Warp functions such as ``wp.sin``,
  ``wp.cos``, ``wp.sqrt``, ``wp.add``, and ``wp.min``.
  Arbitrary Python callables are not supported. Some built-in Warp functions, such as ``wp.printf``, cannot be used
  as ``wp.Function`` arguments because they need special handling during kernel compilation.
  Rebinding a function-valued local to a different function or to a non-function value is not supported.
  User functions with ``wp.Function`` parameters also cannot define custom gradient or replay functions.

:func:`wp.tid() <warp._src.lang.tid>` returns signed 32-bit thread coordinates. In a launch with a nonzero thread
count, the extent corresponding to each coordinate may be at most :math:`2^{31}`, so the largest coordinate is
:math:`2^{31}-1`. Warp raises a ``ValueError`` if any such extent is larger. This limit applies to scalar and
tuple-valued ``wp.tid()``.

The total number of threads in a multidimensional grid may exceed :math:`2^{31}` when each returned coordinate stays
within these limits. Use 64-bit arithmetic when flattening such coordinates into a linear index. See
:ref:`large-array launch indexing <large_launch_indexing>` for an example.

Warp uses one-dimensional CUDA thread blocks with dimensions ``(block_dim, 1, 1)``, even when
the logical Warp grid is multidimensional. Multidimensional CUDA thread blocks are not supported.

By default, kernels use
`grid-stride loops <https://developer.nvidia.com/blog/cuda-pro-tip-write-flexible-kernels-grid-stride-loops/>`_.
A CUDA thread processes additional elements of the logical Warp grid when the launch needs more threads
than the CUDA grid supplies. Set ``max_blocks`` to cap the number of CUDA blocks and control this behavior.

Kernels compiled with ``grid_stride=False`` use no grid-stride loop. On CUDA, launching such a kernel with
``max_blocks > 0`` raises a ``RuntimeError``. Use ``grid_stride=True`` when capping the block count.

Differentiability
-----------------
Please see the :ref:`Limitations and Workarounds <limitations_and_workarounds>` section in the Differentiability page for auto-differentiation limitations.

Arrays
------

* Arrays can have a maximum of four dimensions.
* Each dimension of a Warp array cannot be greater than the maximum value representable by a 32-bit signed integer,
  :math:`2^{31}-1`. As a result, one-dimensional Warp arrays cannot represent larger logical data sets; split them
  across multiple dimensions instead. See :ref:`large-array launch indexing <large_launch_indexing>` for the launch
  and indexing pattern.
* There is currently no dedicated complex scalar type. :func:`tile_fft <warp.tile_fft>` and
  :func:`tile_ifft <warp.tile_ifft>` use ``wp.vec2f`` or ``wp.vec2d`` for complex values: the first
  component is the real part and the second is the imaginary part. Arithmetic on these types follows
  vector rules.
* ``wp.config.launch_array_access_mode = wp.config.LaunchArrayAccessMode.CHECKED``
  only fully verifies cross-device :class:`wp.array <warp.array>` arguments when
  Warp can classify the pointer and prove the relevant access requirements.
  Custom arrays or external wrappers whose pointer kind or specific access state
  cannot be verified warn and proceed in checked mode; use
  ``wp.config.LaunchArrayAccessMode.STRICT`` to reject cross-device launches before
  checking access. Directly passed ``__array_interface__`` or
  ``__cuda_array_interface__`` objects are not fully access-verified. See
  :ref:`launch_array_access_checks` for details.

Structs
-------

* Structs cannot have generic members, i.e. of type ``typing.Any``.
* Structs do not support inheritance. Consider using composition instead.

Volumes
-------

* The sparse topology of a non-rebuildable :class:`Volume` cannot be changed after allocation. Volumes created with
  ``rebuildable=True`` or explicit capacity arguments can change topology within their reserved capacity.

Multiple Processes
------------------

* A CUDA context created in the parent process cannot be used in a *forked* child process.
  Use the spawn start method instead, or avoid creating CUDA contexts in the parent process.
* Clearing a shared kernel or LTO cache while another process is compiling or loading kernels is unsupported
  and can cause compilation failures, module-loading failures, or process termination.
  If shared caches need clearing, clear them before starting workers. Use a separate cache directory
  for each process that may clear its cache independently.
  See the :ref:`Configuration` section for how the cache directory may be changed.

Scalar Math Functions
---------------------

This section details some limitations and differences from CPython semantics for scalar math functions.

Modulus Operator
""""""""""""""""

Deviation from Python behavior can occur when the modulus operator (``%``) is used with a negative dividend or divisor
(also see :func:`wp.mod() <warp._src.lang.mod>`).
The behavior of the modulus operator in a Warp kernel follows that of C++11: The sign of the result follows the sign of
*dividend*. In Python, the sign of the result follows the sign of the *divisor*:

.. code-block:: python

    @wp.kernel
    def modulus_test():
        # Kernel-scope behavior:
        a = -3 % 2 # a is -1 
        b = 3 % -2 # b is 1

    # Python-scope behavior:
    a = -3 % 2 # a is 1
    b = 3 % -2 # b is -1

For integer operands, the divisor must be nonzero. A zero divisor causes undefined behavior in Warp kernels;
Python raises a ``ZeroDivisionError`` exception.

Power Operator
""""""""""""""

The power operator (``**``) in Warp kernels only works on floating-point numbers (also see :func:`wp.pow() <pow>`).
In Python, the power operator can also be used on integers.

Inverse Sine and Cosine
"""""""""""""""""""""""

:func:`wp.asin() <warp._src.lang.asin>` and :func:`wp.acos() <warp._src.lang.acos>` automatically clamp the input to fall in the range [-1, 1].
In Python, using :external+python:py:func:`math.asin` or :external+python:py:func:`math.acos`
with an input outside [-1, 1] raises a ``ValueError`` exception.

Rounding
""""""""

:func:`wp.round() <warp._src.lang.round>` rounds halfway cases away from zero, but Python's
:external+python:py:func:`round` rounds halfway cases to the nearest even
choice (Banker's rounding). Use :func:`wp.rint() <warp._src.lang.rint>` when Banker's rounding is
desired. Unlike Python, the return type in Warp of both of these rounding
functions is the same type as the input:

.. code-block:: python

    @wp.kernel
    def halfway_rounding_test():
        # Kernel-scope behavior:
        a = wp.round(0.5) # a is 1.0
        b = wp.rint(0.5)  # b is 0.0
        c = wp.round(1.5) # c is 2.0
        d = wp.rint(1.5)  # d is 2.0

    # Python-scope behavior:
    a = round(0.5) # a is 0
    c = round(1.5) # c is 2

Bit Count
"""""""""

:func:`wp.bit_count() <warp._src.lang.bit_count>` counts set bits in an integer's fixed-width binary
representation, including the sign bit for signed inputs. In contrast, Python's
:external+python:py:meth:`int.bit_count` counts set bits in the absolute value.

For example, ``wp.bit_count(wp.int8(-1))`` returns ``8``, whereas Python's ``(-1).bit_count()`` returns ``1``.
The difference applies both in Warp kernels and when calling ``wp.bit_count()`` from Python.

.. _variable-scope:

Conditional Initialization
--------------------------

Warp does not check for unassigned local variables at runtime. Assign a value on every execution path
before reading a local variable. In this function, ``out`` is uninitialized when ``cond`` is ``False``:

.. code-block:: python

    @wp.func
    def foo(cond: bool, x: int):
        if cond:
            out = x + 123

        print(out)

Reading ``out`` when ``cond`` is ``False`` can cause undefined behavior. Python would raise an
``UnboundLocalError``. Give ``out`` an initial value before the ``if`` statement, or assign it in both branches:

.. code-block:: python

    @wp.func
    def foo(cond: bool, x: int):
        out = 0
        if cond:
            out = x + 123

        print(out)

Literal assignments can be compiled as constants and mask missing initialization.

.. _limitations-arrays-in-structs:

Arrays in Structs
-----------------

When you assign an array to a struct field, Warp copies the array's native descriptor into the struct.
Changing flags on the Python array later may leave that copy unchanged:

.. code-block:: python

    @wp.struct
    class MyStruct:
        arr: wp.array[float]

    a = wp.zeros(10, dtype=float)

    s = MyStruct()
    s.arr = a

    # Enable gradients on the Python array without updating the struct descriptor.
    a.requires_grad = True

``s.arr`` is the same Python array as ``a``, so ``s.arr.requires_grad`` is ``True``. The struct's copied
descriptor still has no gradient pointer, so backward kernel launches may not accumulate gradients.
Reassign the field to copy the updated descriptor into the struct:

.. code-block:: python

    s.arr = a

Tile reductions and atomics on struct-valued tiles do not accumulate the contents of array fields.
Which array descriptor survives is unspecified. See :ref:`tile-struct-element-types` for details and an example.
