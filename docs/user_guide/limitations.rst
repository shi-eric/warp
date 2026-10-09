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

You can use these features in Python host code and in expressions that :func:`wp.static() <warp.static>`
evaluates during code generation. For example, a static expression can use a list comprehension, lambda call,
or dictionary lookup if its inputs are known at compile time and its result is a supported kernel value.

Kernels and User Functions
--------------------------

* Strings cannot be passed into kernels.
* :func:`wp.atomic_add() <warp._src.lang.atomic_add>` currently does not support :class:`wp.float16 <float16>` or
  :class:`wp.bfloat16 <bfloat16>` on GPUs with compute capability below 7.0.
  Use :class:`wp.float32 <float32>` or a supported device for these atomic operations.
* Using ``wp.atomic_add()`` or related functions on the same memory address from
  overlapping CPU and GPU kernels is currently unsupported.
* :func:`wp.tid() <warp._src.lang.tid>` cannot be called from user functions.
* Modifying the value of a :class:`wp.constant() <warp.constant>` during runtime will not trigger
  recompilation of the affected kernels if the modules have already been loaded
  (e.g. through a :func:`wp.launch() <warp.launch>` or a :func:`wp.load_module() <warp.load_module>`).
* Untyped Python float and integer constants are compiled as 32-bit values, which can lose precision.
  See :ref:`Constants <runtime-constants>` for using explicit types.
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

Warp launches CUDA thread blocks with dimensions ``(block_dim, 1, 1)``.
Multidimensional CUDA thread blocks are not supported.

Capping the CUDA block count with ``max_blocks`` is not supported for kernels compiled with
``grid_stride=False``.

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
* Warp currently has no dedicated complex scalar type. :func:`tile_fft <warp.tile_fft>` and :func:`tile_ifft <warp.tile_ifft>`
  use ``wp.vec2f`` or ``wp.vec2d`` for complex values: the first component is the real part and the second is
  the imaginary part. Arithmetic on these types follows vector rules.
* Checked launch validation does not fully verify access to arrays whose pointer kind or access state
  Warp cannot determine. Unverified pointers warn and proceed. See :ref:`launch_array_access_checks`
  for details, including limitations for directly passed array-interface objects.

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
* Clearing a shared kernel or LTO cache while another process is compiling or loading kernels is currently
  unsupported and can cause compilation failures, module-loading failures, or process termination.
  Clear shared caches before starting workers, or use a separate cache directory for each process.
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

Conditional Initialization
--------------------------

Warp does not check for unassigned local variables at runtime. Assign a value on every execution path before
reading a local variable. In this function, ``out`` is uninitialized when ``cond`` is ``False``:

.. code-block:: python

    @wp.func
    def show_conditional_initialization(cond: bool, x: int):
        if cond:
            out = x + 123

        print(out)

When ``cond`` is ``False``, Warp skips ``x + 123`` and leaves ``out`` uninitialized. Reading it can cause
undefined behavior rather than Python's ``UnboundLocalError``. Give ``out`` an initial value before the ``if``
statement, or assign it in both branches:

.. code-block:: python

    @wp.func
    def show_conditional_initialization(cond: bool, x: int):
        out = 0
        if cond:
            out = x + 123

        print(out)

Literal assignments can mask missing initialization. Always initialize the variable on every execution path.

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

``s.arr`` is the same Python array as ``a``, so ``s.arr.requires_grad`` is ``True``. The struct's copied descriptor
still has no gradient pointer, so backward kernel launches may not accumulate gradients. Reassign the field
to copy the updated descriptor into the struct:

.. code-block:: python

    s.arr = a

Tile reductions and atomics on struct-valued tiles do not accumulate the contents of array fields.
The surviving array descriptor is unspecified. To combine array payloads deterministically, accumulate
their contents explicitly in a kernel. See :ref:`tile-struct-element-types` for an example.
