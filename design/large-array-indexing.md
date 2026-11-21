# 64-bit Indexing for Large 1-D Arrays

**Status**: Implemented on branch `ershi/large-array-indexing`, which is stacked on `ershi/linear-tid` (which adds `wp.linear_tid()`, see "Launch index"), itself stacked on `ershi/fix-adjoint-array-descriptor`, and depends on the latter's reverse-pass fix (see "Autodiff")

**Issue**: [GH-1673](https://github.com/NVIDIA/warp/issues/1673)

## Motivation

HPC users want to work with 1-D Warp arrays that hold more than 2³¹ − 1 elements: allocate them, fill and copy them, exchange them with NumPy and DLPack consumers, launch a kernel over every element, and read, write, and differentiate through them inside kernels.

[GH-1673](https://github.com/NVIDIA/warp/issues/1673) reports a 1-D `wp.array` allocation of 2,291,155,456 elements failing in `check_array_shape` on a GPU with free memory, from a FEM solver whose CSR matrices reach 1.6 to 2.3 billion nonzeros. It asks for either 64-bit shapes in `array_t` or a documented large-array API, 1-D contiguous first, with DLPack export of the whole buffer; this design takes the second route.

Before this work, Warp could not do this:

- The native array descriptor `array_t` (`warp/native/array.h`, mirrored by `array_t` in `warp/_src/types.py`) stores `shape` and `strides` as `int32[4]`. `check_array_shape()` rejected any dimension of 2³¹ or more.
- Kernel indexing functions (`index()`, `adj_index()`, `adj_address()`, atomics) take `int` indices.
- `wp.tid()` returns `int`. A kernel that uses it cannot be launched with an extent above 2³¹ (`_TID_MAX_EXTENT` in `warp/_src/codegen.py`).

Two things were already 64-bit and made the feature cheap to add: the per-access byte-offset arithmetic (`byte_offset()` multiplies by a `size_t` stride), and the linear launch index (`launch_bounds_t::size`, the CUDA `_idx`, and the CPU `task_index` are all `size_t`).

The overriding constraint is that users who do not need large arrays must pay nothing. `array_t` is shared by every kernel, by structs that contain arrays, by NumPy structured dtypes, and by the precompiled native library. Widening it would grow every descriptor and change an ABI that existing code depends on. The feature is therefore opt-in, and the opt-in is visible in the type system so that a mismatch is an error and never a silent truncation.

## Requirements

| ID  | Requirement | Priority | Status |
| --- | ----------- | -------- | ------ |
| R1  | A 1-D `wp.array` can have up to 2⁶³ − 1 elements. It is the same Python class as before; no flag or separate class is needed to create it. | Must | Done. `wp.zeros`, `wp.empty`, `wp.full`, `wp.array(ndarray)`, `wp.clone`, `wp.zeros_like`, `wp.empty_like` verified with real arrays of 2³¹ + 16 elements. |
| R2  | A kernel parameter annotated `wp.largearray[dtype]` accepts a 1-D `wp.array` of any length, small or large. | Must | Done. The dtype must be concrete; see "Annotation". |
| R2a | A `wp.func` parameter annotated `wp.largearray[dtype]` accepts a `largearray` variable. | Must | Done. |
| R3  | Inside a kernel, a `largearray` can be read and written with any integer index type. The index is widened to `int64`. Negative indices wrap as they do for `wp.array`. | Must | Done, with the literal-index caveat under "Code generation". |
| R4  | The atomic built-ins work on `largearray` with an `int64` index. | Must | Done: `add`, `sub`, `min`, `max`, `cas`, `exch`, `and`, `or`, `xor`. |
| R5  | `len(x)` on a `largearray` returns `int64`. | Must | Done. |
| R6  | `x.shape[0]` on a `largearray` returns `int64`. | Should | Done. Any other index into `x.shape` is a code-generation error. |
| R7  | `wp.linear_tid()` returns the linear launch index as `int64`. | Must | Done, on `ershi/linear-tid`. |
| R8  | A kernel that does not call `wp.tid()` can be launched over more than 2³¹ threads on CPU and CUDA, including when it calls `wp.linear_tid()`. | Must | Done; verified with a real launch of 2³¹ + 16 threads on both. |
| R9  | Reverse-mode autodiff works through `largearray` reads, writes, and atomics, including `wp.Tape`. | Must | Done. |
| R10 | Passing a large array anywhere that needs the 32-bit descriptor raises a clear Python error naming the array length and suggesting `wp.largearray`. | Must | Done. See "Large-array guard on the 32-bit paths". |
| R11 | Existing users pay nothing on the hot paths: kernels that do not use `largearray` or `wp.linear_tid()` generate byte-identical kernel and function bodies to what they generated before; `sizeof(array_t)` and its field layout do not change; array creation, cached descriptor packing, `view()` and `reshape()` cost the same as before. In launch packing, an argument that is exactly a `wp.array` does no added work. Three kinds of launch argument do pay a small cost in `pack_arg()`: a raw `array_t` descriptor passed as an argument (about 10 ns), an object passed through an array interface such as a NumPy array (one type test, within noise), and an array subclass (one type test, plus a 25 ns inline size check). The other small costs are outside launch packing. Every cost is listed with its measured size and frequency under "Costs that remain for existing users". | Must | Done. The one deliberate change to generated code for existing kernels, the reverse-pass fix under "Autodiff", lives on the separate branch this one depends on. See "No-cost check". |
| R12 | Large arrays work with `fill_()`, `zero_()`, `wp.copy()` (contiguous, same or different device), `.numpy()`, `__array_interface__`, `__cuda_array_interface__`, and DLPack export and import. | Must | Done; verified with real large arrays. |
| R13 | `wp.to_torch()`, `wp.from_torch()`, and the JAX equivalents work with large arrays. | Should | Done; verified with real large arrays. JAX's own indexing of such an array needs `jax_enable_x64`. |
| R14 | Slicing a large array from Python scope into a view that itself fits the 32-bit descriptor. | Could | Not done. Slicing a large array always raises. |

**Non-goals**

- Multi-dimensional arrays with a dimension above 2³¹ − 1. `largearray` is 1-D only in this version.
- Large arrays inside `wp.struct`.
- Slicing, `view()`, `reshape()`, and in-kernel slices of large arrays.
- `wp.utils` (scan, sort, sum, inner), sparse, FEM, hash grid, BVH, and every other native routine that takes an `int` count.
- `wp.indexedarray` and `wp.fabricarray` over large data.
- Changing how an `int64` index behaves on a regular `wp.array`.
- A `wp.tid()` that returns 64-bit coordinates.
- A 64-bit `range()`. Kernel `range()` keeps its `int32` arguments, so `for i in range(len(x))` does not compile for a `largearray` ("Couldn't find function overload for 'range' that matched inputs with types: [int64]"). Looping over more than 2³¹ elements in one thread is not a use case. A short loop with large `int64` bounds (for example CSR offsets) uses an `int32` trip count and adds the `int64` start, or a `while` loop. `i` here is an `int64` (for example from `wp.linear_tid()`), so the `+ 1` must be written `+ wp.int64(1)`; see the literal rules under "Code generation":

  ```python
  start = row_ptr[i]
  for k in range(int(row_ptr[i + wp.int64(1)] - start)):
      j = start + wp.int64(k)
  ```

**Anti-goals**

- Any change to the generated code, descriptor size, or run time of existing kernels, or to the cost of kernel launches.
- Silent truncation of a length or index. Every unsupported path must raise.
- A global or module-wide switch that changes the type of `len()`, `shape`, or `wp.tid()` for code that did not ask for it.

## Design

### Approach

The distinction between 32-bit and 64-bit indexing lives in the kernel parameter annotation and nowhere else.

```python
@wp.kernel
def axpy(x: wp.largearray[float], y: wp.largearray[float], a: float):
    i = wp.linear_tid()  # int64
    y[i] = a * x[i] + y[i]


n = 3_000_000_000
x = wp.zeros(n, dtype=float)  # an ordinary wp.array
y = wp.zeros(n, dtype=float)
wp.launch(axpy, dim=n, inputs=[x, y, 2.0])
```

At Python scope there is one array class. A `wp.array` already keeps its shape as Python integers, so a large one differs from a small one only in the value of `shape[0]`. This matches what Torch and NumPy users expect: a large allocation just works.

At kernel scope, `wp.largearray[dtype]` selects a separate native descriptor, `largearray_t<T>`, with a 64-bit length and its own overloads of the indexing, adjoint, and atomic functions. This follows the precedent of `indexedarray_t` and the Fabric array types, which are also distinct native types with their own overloads. Because existing kernels never mention `largearray_t`, their generated code and the `array_t` ABI are untouched by construction.

When a kernel is launched, argument packing chooses the descriptor from the parameter annotation:

- A `largearray` parameter packs any 1-D `wp.array` into a `largearray_t`.
- An `array` parameter packs into `array_t` as before, and raises if the array is too long for it.

### Alternatives Considered

**Separate Python class (`wp.array64(...)` or `wp.zeros(..., large=True)`).** The first version of this design. Rejected because it forces users to decide at allocation time, doubles the Python-side surface that has to be taught about the new class, and needs duplicate or generic kernels to serve both sizes. Keeping one Python class and letting a `largearray` parameter accept any length removes all three problems.

**The name `wp.array64`.** The annotation was first called `wp.array64[dtype]` and was renamed to `wp.largearray[dtype]` before release. Next to `wp.float64` and `wp.int64`, `array64` reads like "an array of 64-bit elements", and Warp's existing numeric suffixes on `array` (`wp.array1d` to `wp.array4d`) count dimensions. `wp.largearray` follows Warp's pattern for other array kinds (`indexedarray`, `fixedarray`, `fabricarray`) and matches the "large arrays" wording of the user guide and the error messages. The native types were renamed with it (`largearray_t`, `largearray_desc_t`), because the code generator derives the native type name from the Python family class name.

**Module or kernel compile option** (for example `module_options={"index64": True}`) that widens the descriptor for every array in the module. Rejected because:

- The types of `len(x)`, `x.shape[0]`, and possibly `wp.tid()` would change because of a flag set far from the code. With Warp's strict integer typing this produces confusing errors. JAX's `jax_enable_x64` has the same shape and is a well-known pain point.
- Every struct that contains an array would need two layouts.
- The precompiled native library only knows the 32-bit layout.
- Every array in the opted-in module pays for the wider descriptor, including the small ones.

**Flag bit on the existing `array_t`**, storing a 64-bit length in the unused `shape[1..3]` slots of a 1-D array. Rejected because the type system cannot tell the two cases apart. Existing 32-bit paths would need a run-time flag check, which penalises everyone, or would silently misread the length.

**Widen `array_t` for everyone**, as Torch does. Rejected by the constraint that existing users must not pay.

**Unsigned 64-bit indices.** Rejected. Signed `int64` keeps negative-index wrapping, matches NumPy's `intp`, and 2⁶³ elements is far beyond any real allocation.

### Key Implementation Details

#### Python-side array

- `check_array_shape()` allows a 1-D shape up to 2⁶³ − 1 (`LARGEARRAY_MAX_SIZE`). Arrays with two or more dimensions keep the per-dimension 2³¹ − 1 limit (`ARRAY_MAX_EXTENT`). The function keeps the old loop and constant comparison; only a shape that is already out of 32-bit range takes the new branch, which returns `True` for a valid large 1-D shape.
- An array is "large" when it is 1-D and `shape[0] > 2³¹ − 1`. Arrays that fit the 32-bit descriptor are unchanged.
- `requires_grad=True` allocates a gradient array of the same length, which is itself large.
- `fill_()`, `zero_()`, contiguous `wp.copy()`, `.numpy()`, the array interfaces, DLPack, and the Torch and JAX conversions needed no change: every intermediate size was already `size_t` or a Python integer. This was verified with real arrays (see "Testing").
- `a[0]` on any 1-D array, large or not, raises the existing `RuntimeError: Item indexing is not supported on wp.array objects`; `__getitem__` tests the key type before the size.
- **Copies on a tape.** `wp.copy()` records `adj_copy` on an active `wp.Tape` when source and destination both have gradients (`wp.clone()` goes through it too). That adjoint slices the gradients and accumulates them with a `wp.array` kernel, neither of which supports large arrays, so `backward()` used to fail late with the slicing error and its `wp.largearray` advice. `_reject_large_tape_copy()` now raises a `ValueError` when such a copy is recorded: "wp.copy() of an array with more than 2**31 - 1 elements (got N) cannot be recorded on a wp.Tape, ...", suggesting a copy outside the tape or a kernel with `wp.largearray` parameters. It runs inside the existing `if runtime.tape:` branch, after the copy itself, so copies that are not recorded pay nothing and behave as before; the cost is that the data has already been copied when the error is raised. It is tested by calling the helper directly, because a real copy of a fake-large array would read past its buffer.
- Every error for a large array on a 32-bit path comes from `large_array_error()` and reads, for example: `The 32-bit array descriptor does not support arrays with more than 2**31 - 1 elements (got 2147483664). Pass the array to a kernel parameter annotated wp.largearray[...] instead.` A launch wraps it in `RuntimeError: Error launching kernel '<name>', argument '<arg>': ...`.

#### Large-array guard on the 32-bit paths

`array.__ctype__()` builds the 32-bit `array_t` and is the choke point for R10. The guard is split by how often each path runs:

- **`__ctype__()`: a size check in the uncached branch.** Inside `if self.ctype is None:`, the branch that runs only when the descriptor is first built (or rebuilt after a `grad`, `requires_grad` or `retain_grad` change resets `ctype`), `__ctype__()` checks `self.ndim == 1 and self.shape[0] > ARRAY_MAX_EXTENT` and raises `large_array_error(self, "The 32-bit array descriptor")`. The check is in the method body and reads the array's current shape, so it holds however the method is called: bound, through the class (`wp.array.__ctype__(big)`), or through `super().__ctype__()` in a subclass that does not override it. It also catches a `shape` reassigned to a large value **before** the descriptor is first built. A large array never caches a descriptor, so it raises on every call. The cached path (`if self.ctype is None` false, return `self.ctype`), which every later kernel launch uses, is unchanged. Measured cost of the check: about 40 ns on the first descriptor build (826 to 869 ns and 804 to 844 ns in two interleaved CPU measurements); the cached call, `wp.zeros` and a CPU launch with one array argument were unchanged within noise.
- **Limit:** `shape` is a plain attribute. If it is reassigned to a large value **after** the small descriptor was cached, `__ctype__()` keeps returning the stale cached descriptor, with the old length. This is the same staleness any array has when `shape` is reassigned after its descriptor is built; it does not truncate a length that was large when the descriptor was made.
- **`reshape()` and `view()`: marked at creation.** `_init_new` and `_init_from_ptr`, the only places a `wp.array` gets a non-zero shape, call `_mark_large(length)` when `check_array_shape()` reports a large shape. `_mark_large()` sets the instance attribute `_is_large = True` (the class default is `False`) and installs callable stand-ins (`_LargeArrayRejection`) as the instance attributes `reshape` and `view`. Instance attributes shadow the class methods, so every bound call on a large array raises, and arrays that fit run the class methods, which contain no check. Calling these through the class on a large array (`wp.array.reshape(big, ...)`) does not raise, but the result has the correct shape and any later use of it on a 32-bit path is guarded. The stand-ins hold only the length and a label, so they create no reference cycle and do not delay freeing large buffers. `__ctype__` is **not** replaced per instance; it relies on its own check above.
- **Array subclasses that override `__ctype__()`.** Such a subclass may build its own `array_t` from the current shape without calling `super()`, so the check above never runs. `_validate_array_subclass_descriptor()`, which launch packing and struct array fields already call for non-exact `wp.array` instances, rejects a large 1-D array (by current shape, with the test written inline) before inspecting the descriptor. The non-contiguous path of `wp.copy` applies the same current-shape check (`_reject_large_subclass()`) to subclass instances. Exact `wp.array` instances return before this check, so the common launch path is unchanged. Every other entry point listed below that passes a subclass's descriptor or length to native code (`fill_()` on a non-contiguous array, `wp.Mesh`, `wp.HashGrid.build()`, `wp.Bvh`, `wp.indexedarray`) checks the creation-time mark `_is_large`, which a subclass instance created large carries as well.
- **Slicing.** `__getitem__` is looked up on the type and cannot be shadowed per instance, so it checks `self._is_large` (one attribute load) before slicing. The check comes after the integer-key test, so integer indexing keeps its existing error and does not pay for it.
- Code that reaches `__ctype__()` raises: kernel launches into a `wp.array` parameter (the launch wraps the `ValueError` in a `RuntimeError` naming the kernel and argument), struct field assignment, the `wp.Mesh` points and velocities setters, and other utilities that pass arrays to native code as descriptors.
- Paths that build an `array_t` from foreign metadata call `check_descriptor_shape()`: `array_ctype_from_interface()`, `wp.from_torch(return_ctype=True)`, `wp.from_paddle(return_ctype=True)`, and both `array_t` constructions in the JAX FFI.
- Paths that pass a length to native code as a C `int`, or pass a descriptor that a subclass may have built, check explicitly before any native call or memory access. Each check is written inline (no helper call on the non-raising path) and covers only the arrays whose length can reach the count; an array that an earlier check forces to the same length is not tested again:
  - `wp.indexedarray`: the data array and each index array (`_is_large`).
  - `wp.utils`: `array_scan` (input and output), `radix_sort_pairs` (keys and values), `segmented_sort_pairs` (keys, values, segment starts; the ends must match the starts in size), `runlength_encode` (values and both outputs), `array_sum` (values; `out` must have the reduced shape), `array_inner` (`a`; `b` must have the same shape and `out` the reduced shape), `array_cast` (both arrays); `graph_coloring_assign` and `graph_coloring_balance` (node count, by current shape).
  - `wp.Bvh` (`lowers`, after the test that `uppers` has the same length; `groups` is tested for the same length after this check, still before any native call), `wp.Mesh` (points, indices, velocities; the points and velocities setters need no check, since they require the length the mesh was built with), `wp.HashGrid.build()` (points; groups must match in size) and `wp.HashGrid.reserve()` (the requested count).
  - `wp.Volume.allocate_by_tiles()`, `allocate_by_voxels()` and `rebuild()`: the points and the point mask, by current shape (`_reject_large_volume_points()`), before the arrays are copied to another device. These pass `shape[0]` beside `.ptr` as a C `int`, which ctypes truncates without an error, so a 2³² + k point array would have built a volume from k points.
  - Textures: width, height and depth must not exceed 2³¹ − 1 (they reach the native texture constructors as C `int`s); a large 1-D array as 1-D texture data raises "Texture dimensions must not exceed 2**31 - 1".
  - `array.fill_()` on a non-contiguous array (`_is_large`; a contiguous fill uses a `size_t` count and works).
  - `wp.sparse.bsr_set_from_triplets()` (the triplet count, after the test that columns and values match the rows) and `wp.sparse.bsr_set_transpose()`, whose native code indexes a scratch map of `2 * nnz` C `int`s: it raises for more than 2³⁰ − 1 blocks before allocating anything (before this work, `wp.empty(2 * nnz)` raised for the same matrices; it now succeeds for a 1-D length, so the check is explicit).
- **Audit.** Every native entry point registered in `warp/_src` with a 32-bit integer parameter (152 of 332 registered functions) was listed with its call sites, and each call that passes an array length, `.size`, `len()` or a count derived from them was checked. Counts that are not array lengths (device ordinals, flags, block sizes, dtype codes) were excluded. Natives without registered argument types take no arguments. The FEM, `optim`, `render` and geometry (marching cubes, surface nets, Delaunay) modules call no natives with counts directly: they launch kernels with `wp.array` parameters or call the `wp.utils` functions above. Deterministic mode already bounds its record counts (at most 2³¹ − 1 scatter slots and 2³¹ − 1 threads for counters) and receives its destination arrays through launch packing.

**History of this decision.** The first version checked `_is_large` on every `__ctype__()`, `reshape()`, `view()` and slice, which cost 25–55 ns on every array's first `__ctype__()` and 20–50 ns on every reshape, view and slice. A second version ("A") moved all of it to creation time and also replaced `__ctype__` per instance, for zero added cost. It was rejected because two call forms then returned a descriptor with a silently wrapped length (`shape[0] = -2147483632` for 2³¹ + 16 elements) instead of raising: an unbound `wp.array.__ctype__(big)` call (including `super().__ctype__()`), and reassigning `arr.shape` to a large value after creation. The project owner chose the mechanism above ("C"): it raises for both of those call forms, subject to the cached-then-reassigned limit above, and its cost is paid once per descriptor build, outside the hot launch path.

Other known limitations of the creation-time marking: re-running `__init__` on a large array leaves the `reshape`/`view` stand-ins in place; `help()` and `inspect.signature()` on a large array's `reshape`/`view` show the stand-in; and a future 32-bit-only method has to be registered in `_mark_large()` by hand.

#### Costs that remain for existing users

Measured with interleaved runs (variant order rotated each round, medians), CPython 3.12, on the machine described under "Testing". Each figure is per call of the operation named, compared with the base commit `8a60f7374`, and was measured in isolation: the base and pre-fix versions of each function were compiled from `git show` into the live module, so all versions run against the same helpers. Scripts and raw output: `.superpowers/bench/bench_pack_arg.py`, `bench_launch.py`, `bench_guards.py` (not checked in).

| What | Where | Cost | How often |
| --- | --- | --- | --- |
| Size check in the uncached branch | `array.__ctype__()` | about 40 ns | Once per array when its 32-bit descriptor is built, and again after a `grad`, `requires_grad` or `retain_grad` change resets it. Never on the cached path that launches use. |
| `_is_large` load | `array.__getitem__()` | one attribute load, unchanged (66.2 vs 66.4 ns for the check plus the key-type test) | Every Python-scope slice with a non-integer key; integer keys no longer pay it. |
| Launch with `wp.array` arguments | `pack_arg()` | none. Per argument: forward 1.6 ns faster to 8.2 ns faster than base, adjoint 7.9 to 39 ns faster (it reads `arg_type._adjoint_family` instead of the global `warp.array`); a second copy of the base function differed from the first by up to 6 ns. Whole CPU launch with four arguments: +7.5 ns forward and +22 ns adjoint on about 7.3 and 10.8 µs, where two copies of base differed by 42 and 72 ns; CUDA: +7.0 and −7.2 ns (base copies: 4.9 and 5.6 ns). Before this fix: +55 to +64 ns forward and +86 to +128 ns adjoint. | Every launch. |
| Dispatch test for a raw `array_t` value | `pack_arg()` | +10 to +11 ns per argument (on a 250 ns path) | Each `array_t` descriptor passed directly to an array parameter. |
| Dispatch test for a value not of exactly the expected family | `pack_arg()` | one `type(...) is` test; measured +0.7 to +6 ns (within noise) on paths of 0.95 to 2.7 µs | Each array-subclass instance, object with an array interface (NumPy, Torch), or wrong-family argument. |
| Inline subclass size check | `_validate_array_subclass_descriptor()` | 25 ns (was 35 ns through a helper call) | Each array-subclass instance passed to a `wp.array` parameter or assigned to a struct field. |
| Subclass tests on both arrays | `wp.copy()` between non-contiguous arrays | 40 ns for two plain arrays (was 71 ns in a loop) | Per non-contiguous copy. |
| Inline `_is_large` loads | `wp.utils` | `array_inner`, `array_sum`: one load, 12 ns; `array_scan`, `radix_sort_pairs`, `array_cast`: two, 25 ns; `segmented_sort_pairs`, `runlength_encode`: three, 37 ns (was 105 ns for three arrays through the helper). Whole small CPU calls (1.7 to 37 µs): +30 to +57 ns compared with base, against +106 to +144 ns before this fix. | Per call. |
| One `_is_large` load | `wp.Bvh()`, `wp.indexedarray()` (data, and each index array), `wp.HashGrid.build()`, `array.fill_()` on a non-contiguous array | 11 ns each (`wp.Bvh` was 23 ns) | Per call. |
| Two or three `_is_large` loads | `wp.Mesh()` | 25 ns without velocities | Per mesh. |
| `_reject_large_volume_points()` | `wp.Volume.allocate_by_tiles()`, `allocate_by_voxels()`, `rebuild()` | 24 ns per points array, and again for a point mask | Per call. |
| Three integer compares | `wp.Texture1D/2D/3D()` | 22 ns | Per texture. |
| One integer compare | `graph_coloring_assign()`, `graph_coloring_balance()`, `bsr_set_from_triplets()` (was 104 ns), `HashGrid.reserve()` | 6 to 8 ns | Per call. |
| `_check_transpose_nnz()` call | `bsr_set_transpose()` | 23 ns measured here (the reviewer measured about 33 ns) | Per call with a non-empty source. |
| `_reject_large_tape_copy()` | `wp.copy()` | 38 ns | Only for a copy recorded on a tape (both arrays have gradients and a tape is recording). |
| Type test per `and`/`or` operand; `linear_tid` scan | code generation | build time only; generated code is unchanged | Per `and`/`or` operand during code generation; the scan only for `external_constant_params` kernels. |

The guard expressions were timed alone in a loop of 200,000 evaluations against an empty loop, with the arrays held in module globals, so each figure includes a global load that the real code (a local variable) does not pay. Whole calls were timed in loops of 2,000 (`wp.utils`), 20,000 (`pack_arg`) and 5,000 (`wp.launch`), 41 to 61 rounds each.

**Opting in.** Moving the `wp.largearray` dispatch off the `wp.array` path makes it later for `wp.largearray` arguments: `pack_arg()` now costs 690 to 780 ns per `wp.largearray` argument against about 600 ns before (+95 to +104 ns adjoint, +173 to +177 ns forward), because those arguments now also pass the `array_t` test and `concrete_array_type()` before reaching `_pack_largearray_arg()`.

#### Annotation

- `wp.largearray[dtype]` is a kernel-annotation type built on the `_ArrayAnnotationBase` machinery (`_LargeArrayAnnotation`). It is 1-D only; there is no `ndim` parameter, and `wp.largearray[a, b]` raises `TypeError`.
- It is not a class users instantiate. `wp.largearray(...)` raises `TypeError: wp.largearray is a kernel parameter annotation and cannot be instantiated. Create the array with wp.array, wp.zeros, or wp.empty.` The factory spelling of an annotation, `wp.largearray(dtype=float)`, adds `To annotate a kernel parameter, write wp.largearray[float] instead of wp.largearray(dtype=float).` Only the internal family descriptor `largearray(dtype=Any)`, used to register built-ins, can be constructed.
- `largearray` is a separate array family (in `array_types`, type code prefix `a64`), so kernels overloaded on `wp.array[float]` and `wp.largearray[float]` hash differently.
- **Launch packing.** `pack_arg` dispatches a `wp.largearray` parameter to `_pack_largearray_arg()` only on branches that a plain `wp.array` argument never takes, so a launch with `wp.array` arguments runs the same operations as on the base commit. Every parameter type carries `_adjoint_family`, the family its gradient must belong to (`wp.array` for every family, `largearray` for `_LargeArrayAnnotation`), which adjoint packing reads in place of the global `warp.array`. No argument is an instance of `largearray` (only the bare `largearray(dtype=Any)` descriptor is, and it fails the dtype check), so no argument for a `wp.largearray` parameter passes the exact-family test, forward or adjoint, and the dispatch sits in the branch for arguments that are not of exactly the expected family (array subclasses, objects with an array interface, wrong families). A raw `array_t` descriptor is tested in its own branch, where the dispatch rejects it for a `wp.largearray` parameter. Deciding once per kernel argument when the kernel is built would remove even these tests, but needs a change to the launch loop and to `Launch`.
- **Cost for opted-in users.** Because the dispatch now comes later, a `wp.largearray` argument costs about 100 to 175 ns more per launch than before this change (see "Opting in" under "Costs that remain for existing users"). Removing that would need a packer chosen per argument when the kernel is built; this was left as a follow-up decision for the owner.
- Passing a non-1-D array, an `indexedarray`, a Fabric array, a raw `array_t` descriptor, a NumPy array, or an array of the wrong dtype to a `largearray` parameter is a `RuntimeError` at launch. A `largearray_t` descriptor and `None` are accepted. An array subclass that overrides `__ctype__()` is rejected, because the 64-bit descriptor is built from the array's attributes and would ignore the custom descriptor.
- The dtype must be concrete. A generic `wp.largearray[Any]` parameter raises a type mismatch at launch, because argument-type inference maps a `wp.array` value to the `wp.array` family. Kernels that must serve several dtypes declare explicit overloads with `wp.overload(k, {"x": wp.largearray[float]})`. Supporting inference is a small change in `infer_argument_types()` if it is wanted later.
- `wp.largearray` is the name chosen in this design; it replaced the first name, `wp.array64` (see "Alternatives Considered"). It can still change before release; nothing else depends on it.

#### Native descriptor

`largearray_t<T>` holds only what a 1-D array needs:

| Field | Type | Notes |
| ----- | ---- | ----- |
| `data` | `T*` | |
| `grad` | `T*` | Null when the array has no gradient. |
| `shape` | `shape64_t` (`int64 dims[1]`) | The length. Named like `array_t::shape` so `x.shape[0]` reuses the existing attribute path. |
| `stride` | `int32` | Byte stride, so strided small 1-D arrays still work. Packing raises if a stride does not fit. |
| `flags` | `uint16` | Same meaning as in `array_t`. |

`largearray_desc_t` also has `explicit operator bool()` (non-null `data`), for `if x:` and `not x` on an optional `wp.largearray` parameter that may be launched with `None`. It is explicit, unlike `array_t`'s implicit `operator T*()`, so a `largearray_t` never converts to a pointer or an integer; codegen emits `bool(x)` where a value must convert by assignment (the operands of `and`/`or`). `sizeof(largearray_t) == 32` (checked with `ctypes.sizeof` and with `sizeof` in a compiled CPU and CUDA kernel), smaller than `array_t` (56). Because `largearray` is not allowed in structs or exposed through NumPy structured dtypes, its layout is internal to generated kernels and can change later (for example to add dimensions) without an ABI break. The Python mirror is `warp._src.types.largearray_t`; `array.__ctype64__()` builds and caches it, and the cache is reset everywhere the 32-bit `ctype` cache is reset.

**`largearray_t` is an alias, not a plain template.** The native type is `template <typename T, int NDim> struct largearray_desc_t` with `static_assert(NDim == 1)`, and `template <typename T> using largearray_t = largearray_desc_t<T, 1>;`. `array.h` has generic `template <template <typename> class A, typename T>` overloads of `address`, `array_store`, every atomic, `len`, and the gradient-free adjoint fallbacks, all taking `int` indices. A one-parameter `template <typename T> struct largearray_t` matched them: an `int32` or `int16` index resolved to the generic `int` overload (forward results correct, but the adjoint picked the gradient-free fallback and dropped gradients silently), and a `uint32` or `uint64` index was ambiguous. A two-parameter template cannot be deduced as `A<T>`, so none of the generic overloads is viable, and each `largearray` call has exactly one candidate whose `int64` parameter takes any integer by value. This was checked with g++, nvcc 12.9 and 13.5, and then through Warp's Clang and NVRTC. The generated type name is still `wp::largearray_t<T>`. A codegen cast was rejected because it would not cover explicit `wp.atomic_*` calls; an overload per integer type was rejected because it would add about eight times as many templates to a header every module parses.

The overload set, in one section at the end of `array.h`, mirrors the 1-D `array_t` functions with `int64 i`: `byte_offset`, `index`, `index_grad`, `address`, `array_store`, the atomic family, `len`, `extract(shape64_t, int)`, `adj_address`, `adj_array_store`, the atomic adjoints, the composite-slot helpers and their adjoints (`array_atomic_*_slot`, `adj_array_store_slot`), and an `element_is_wide_aligned` that returns `false` (wide loads are not specialised for `largearray`). Negative indices wrap and debug builds assert `-n <= i < n`, in `int64`. `FP_VERIFY` uses the index-free variants, because the indexed ones print with `%d`. There is no `adj_len`, since `len` is not differentiable.

#### Built-in registration

The Python built-ins for `largearray` (`address`, `array_store`, the nine atomics, `len`, `extract` on `shape64_t`, and `unot` for `not x`) are `hidden=True` and `export=False`, so they add 14 overloads in total and do not appear in the stubs or the built-in reference. Each is the last entry in its overload list, so every existing call resolves before reaching a `largearray` overload. There is no `view` overload and no multi-index overload.

#### Code generation

- The index of any integer type is passed by value to the native `int64` parameter; codegen adds no cast.
- **Literals in `int64` arithmetic.** Because Warp types a Python integer literal as `int32` and has strict integer typing, arithmetic between an `int64` value (`wp.linear_tid()`, `len(x)`, `x.shape[0]`, an element of an `int64` array) and a literal is a code-generation error (`Input types must be the same, got ['int64', 'int32']`). `i + 1` and `len(x) - 1` must be written `i + wp.int64(1)` and `len(x) - wp.int64(1)`. This is existing Warp behaviour, but `largearray` code meets it constantly; the user guide calls it out.
- **Literal indices.** Warp types a Python integer literal as `int32`, so `x[2147483652]` would emit `const wp::int32 var = 2147483652;` and wrap to a negative index. `check_largearray_indices()` (called from `Adjoint.eval_indices` for `largearray` targets and from the atomic dispatch) raises `WarpCodegenError: Index 2147483652 into a wp.largearray does not fit in a 32-bit integer literal. Wrap it in wp.int64(), e.g. x[wp.int64(2147483652)].` when an index `Var` is an `int32` constant outside the `int32` range. It catches a bare literal, a negated literal, a `wp.constant`, and a variable assigned once from a literal (`i = 2147483652; x[i]`), in subscripts, stores, augmented assignments, composite slots and explicit atomics. It does not catch a constant expression (`x[2147483647 + 5]`), a literal copied through a second variable (`j = i`), arithmetic on it (`x[i + 0]`), or a literal assigned in a branch; these wrap silently, as they do for `wp.array`. The general fix, typing out-of-range literals as `int64`, would change generated code for existing kernels and is out of scope. `wp.int64(2147483652)` emits `2147483652ll` and is always correct.
- **Errors.** These run only after overload resolution has failed, so they cost nothing for code that resolves (`Adjoint._largearray_resolution_error`):
  - slicing or partial indexing: `wp.largearray does not support slicing or views.`
  - any other built-in without a `largearray` overload (for example `wp.lower_bound`, `x[0, 1] = v`): `Built-in '<name>' does not support wp.largearray with these arguments.`
  - a user function across the two families: `Couldn't find function overload for '<f>' (wp.largearray and wp.array do not convert to each other). Expected inputs: (...)`.
  - `x.shape[i]` with `i` other than the constant `0`: `wp.largearray has one dimension; its shape can only be indexed with the constant 0.`
  - a struct field annotated `wp.largearray[...]`: `TypeError: wp.largearray is not supported in structs.` (raised when the struct is declared).
- **Null checks.** `if x:`, `if not x:`, and `x and y` / `x or y` in a condition work on a `wp.largearray` parameter, as for `wp.array`. `emit_BoolOp` writes a `largearray` operand as `bool(var)`, because the explicit conversion does not apply to the plain assignment it emits (`var_out = var_x;`); the type test runs at code generation only, and other operands are emitted as before. `wp.where(x, a, b)` and `a if x else b` (which lowers to `where`) with a `largearray` condition fail with `Built-in 'where' does not support wp.largearray with these arguments.`
- `largearray` is a closed type inside a kernel: a `wp.array` variable cannot be passed to a `largearray` function parameter, and a `largearray` does not bind to `wp.array` or generic array parameters. The conversion from `wp.array` happens only when a launch packs its arguments. A helper that must serve both types is written once per type.
- Composite element updates (`y[i][1] += v`, `y[i].z = v` on a `largearray[vec3]`) take the same array-slot path as `wp.array`, so their gradients are kept.
- Feature interactions:

  | Feature | Behaviour with `largearray` |
  | --- | --- |
  | Reverse autodiff, `wp.Tape`, `enable_backward=False` | Works; matches `wp.array`. |
  | Forward-mode jets (and reverse over them) | Works; matches `wp.array`. |
  | CPU blocks (`enable_cpu_blocks`) | Works. |
  | CUDA graph capture (without APIC) | Works. |
  | `wp.config.llvm_cuda`, `verify_autograd_array_access`, recorded launches (`record_cmd`) | Works. |
  | CPU linear-contiguous specialisation | `largearray` arguments are not specialised (strides stay run-time); correct, not optimised. |
  | Deterministic mode | Rejected at code generation (`WarpDeterminismError`) for atomics into a `largearray`, a consumed atomic counter in a `largearray`, and gradient accumulation into a `largearray`, which every backward-enabled kernel that reads a `largearray` needs. Reads and plain stores with `enable_backward=False` work. Supporting it would need a 64-bit flat index and 64-bit destination counts in the deterministic sort-reduce. |
  | APIC recording (all CPU graph capture, CUDA `apic=True`) | Rejected: `NotImplementedError: APIC capture does not yet support wp.largearray launch params`. |
  | Wide aligned loads/stores | Not used for `largearray`. |

#### Launch index

`wp.linear_tid()` is not part of this branch: it is the commit "Add wp.linear_tid() 64-bit launch index" ([GH-2389](https://github.com/NVIDIA/warp/issues/2389)) on the separate branch `ershi/linear-tid`, on which this branch is stacked, so that it can be reviewed and merged first. It was developed during this work and is described here because the design depends on it. Of the items below, only the `warp.cu` block-count change is a commit on this branch.

- `wp.linear_tid()` takes no arguments and returns `int64`: the linear index of the current thread over the whole launch, in `[0, total launch size)`, row-major with the last dimension varying fastest. Its type does not depend on the launch size.
- Like `wp.tid()`, it may only be called from a kernel (calling it from a `wp.func` is a code-generation error), and it is not differentiable. A kernel built with `entry_point_abi="external_constant_params"`, whose entry point defines neither `dim` nor `_idx`, rejects it at code generation as it rejects `wp.tid()` ("cannot use wp.linear_tid()"); `_uses_tid_call()` takes the built-in name to look for.
- Calling it does not mark the kernel as using `wp.tid()`, so the 2³¹ extent limit does not apply. A kernel that calls both keeps the limit.
- It is the macro `builtin_linear_tid()`, defined in the CPU module header as `static_cast<wp::int64>(task_index)` and in the CUDA header as `static_cast<wp::int64>(_idx)`, so no arithmetic is added to any launch path.
- The new macro is the only change to the module header text. The internal CPU template `wp::linear_tid` (a specialisation of `wp.tid()`) keeps its name; it does not clash with the built-in, which reaches generated code only as the macro.
- `launch_bounds_t::shape` stays `uint32`; it is read only by `wp.tid()`, which is rejected for launches that do not fit. The total size travels in the `size_t` `size` field.
- `wp_cuda_launch_kernel` (`warp/native/warp.cu`) now computes the grid-stride block count in `size_t` and clamps it to the cap before narrowing. Before, the `size_t` count was narrowed to `int` first, which for a large launch could wrap to a negative value (then mapped to the cap) or to a small positive one (launching fewer blocks than intended). Results were still correct in both cases, because the grid-stride loop visits every work item whatever the block count; only the number of blocks, and so the parallelism, could be wrong. The new code always launches `min(natural block count, cap)` blocks.
- The CPU linear-contiguous loop with `int task_index` is still dispatched only when the launch size fits in `int32`; larger launches use the `size_t` loop.

#### Autodiff

- The adjoint of a `largearray` parameter is a `largearray_t` whose `data` is the gradient buffer, built by packing the gradient array (or the explicit adjoint argument) the same way as the forward argument.
- `wp.Tape` needs no new concepts: `backward()` launches the adjoint kernel with the same dimension, and `zero()` goes through `zero_()`.
- **Reverse-pass fix that also affects `wp.array`.** This fix is not part of this branch: it is the commit "Keep adjoint array descriptors intact" ([GH-2388](https://github.com/NVIDIA/warp/issues/2388)) on the separate branch `ershi/fix-adjoint-array-descriptor`, based on `main` at `a318f75e5`, on which this branch is stacked. It was found during this work (where it was at first a commit on this branch). Before the fix, a kernel that read a descriptor field of any array (`x.shape` or `x.ptr`) emitted `adj_x.shape = adj_N;` or `adj_x.data = (T*)adj_N;` in the reverse pass, with `adj_N` zero-initialised. That zeroed the adjoint descriptor's shape or data pointer. When the field was read after an element access, the later `adj_address` call then wrapped a negative index by zero and wrote outside the gradient buffer (heap corruption and `double free` on CPU, wrong gradients on CUDA), or sent the gradient to `x.grad` instead of an explicit adjoint array. `emit_Attribute` no longer emits a reverse statement for a field of an array descriptor, since those fields have no gradient. It is the one deliberate change to generated code for kernels that do not use `largearray`: kernels that read `.shape` or `.ptr` of an array lose that reverse-pass assignment. That branch covers it with `test_grad.py::test_gradient_array_descriptor_attributes` and has its own `fixed` changelog entry.

#### Documentation and changelog

- `docs/user_guide/runtime.rst` has a "Large Arrays" section: the example above, what is supported at Python and kernel scope (including the null check), the error a user sees, and the unsupported list (including every guarded API, the tape-copy limit, the `range()` workaround and the literal-index rule). `docs/user_guide/limitations.rst` and `docs/user_guide/basics.rst` point to it, and the `wp.tid()` limit paragraph points to `wp.linear_tid()`.
- `wp.largearray` appears in the API reference. The built-in reference entry for `wp.linear_tid()` is on `ershi/linear-tid`.
- `changelog/1673.added.md` covers large arrays and `wp.largearray`. `wp.linear_tid()` has its own issue, [GH-2389](https://github.com/NVIDIA/warp/issues/2389), and its own fragment, `changelog/2389.added.md`, on `ershi/linear-tid`.

## Testing

**Unit tests** (`warp/tests/test_largearray.py`, class `TestLargeArray`, in `default_suite`; CPU and CUDA through `add_function_test()`, except the pure-Python checks of the annotation, the descriptor layout and the shape boundaries, and the CPU-only tests of CPU blocks, the CPU linear-contiguous path and graph coloring, which are plain `TestCase` methods on a fixed device). They never allocate more than 2³¹ elements, add no environment switches, and pass four times in a row on one kernel cache. Small arrays through `largearray` parameters cover:

- Read, write, augmented assignment with `int16`, `int32`, `int64`, `uint32`, `uint64` and negative indices; each atomic compared with the `wp.array` version; `len()` and `shape[0]` returning `int64`; scalar, vector, matrix, `int8`, `float64` and struct dtypes; strided arrays; `wp.func` parameters; the closed-type and unsupported-built-in errors; null arrays; and the null check (`if x:`, `not x`, `and`, `or`) with `None` and real arrays, compared with the `wp.array` version.
- `wp.linear_tid()` as the index into `largearray` parameters throughout. Its own tests are on `ershi/linear-tid`, in `test_launch.py`: 1-D, 2-D and folded launches against `wp.tid()`, indexing a `wp.array` with its `int64` value, a CUDA launch of 2³¹ + 8 threads that writes only the last four indices, the unchanged `wp.tid()` limit, a smoke test that a kernel calling it builds and runs as an adjoint launch, and (in `test_codegen.py`) its rejection under `entry_point_abi="external_constant_params"`.
- Gradients through reads, writes, atomics, `wp.func`, composite slots, `shape`/`len`, `wp.Tape.zero()`, and a replaced `x.grad`, each compared with the `wp.array` overload and an analytic value; jets; CPU blocks; CUDA graph capture; the deterministic and APIC rejections; `llvm_cuda`; `verify_autograd_array_access`; recorded launches; the CPU linear-contiguous path with mixed arguments.
- Launch packing: small, large, `None`, verbatim `largearray_t`, adjoint arguments, mixed signatures, the type errors, array subclasses, and type identity (hashes of `wp.array` and `wp.largearray` overloads differ).
- Literal indices: eight rejected forms (including a `wp.constant` and `i = 2147483652; x[i]`) and the accepted `wp.int64(...)`, small literal, negative literal and `wp.uint32(...)` forms. `uint32` indexing has a small-array smoke test only; sign extension of a `uint32` index above 2³¹ would need an array that large.

Error paths for large arrays use a "fake-large" array: a 16-byte buffer wrapped by `wp.array(ptr=..., shape=(2**31 + 16,))`. This is safe because those paths raise before dereferencing it. It covers `__ctype__`, slicing, `reshape`, `view`, the guard surviving a `grad`/`retain_grad` reset, a class-level `wp.array.__ctype__(big)` call and a `super().__ctype__()` call from a subclass, a `shape` reassigned to a large value before the descriptor is built, a small array's cached descriptor being rebuilt after a `grad` change, struct fields, `wp.indexedarray`, `array_ctype_from_interface`, `check_descriptor_shape`, every guarded `wp.utils` function, graph coloring, `wp.Bvh`, `wp.Mesh`, `wp.HashGrid.build()` and `reserve()`, `wp.Volume.allocate_by_tiles()`, `allocate_by_voxels()`, `rebuild()` and the point mask, 1-D texture data, `fill_()` and `wp.copy()` on non-contiguous arrays, `bsr_set_from_triplets`, `bsr_set_transpose` (through its host-side `nnz` bound), integer indexing (`a[0]`), and launches into `wp.array` and `wp.largearray` parameters. Where an array subclass that builds its own descriptor could reach native code (`wp.Mesh`, `wp.HashGrid.build()`, `fill_()`, `wp.copy()`, launches, struct fields), the same test runs with such a subclass. The tape-copy check is tested by calling `_reject_large_tape_copy()` directly, because it runs after the copy and a real copy of a fake-large array would read past its buffer. A check that fires before memory is touched is what makes these tests safe, so the versions without the fixes were not run where they would have dereferenced the buffer (Volume, Mesh and HashGrid with a subclass, non-contiguous `fill_()` of a subclass, texture, transpose). The `wp.array` regression test for the reverse-pass fix, in `test_grad.py`, is on the branch that holds the fix; `test_large.py` was updated here for the new 1-D limit.

**Real large arrays** were verified once, during development, with a script that is not checked in (N = 2³¹ + 16 elements; boundary indices 0, 2³¹ − 1, 2³¹, N − 1). Results on an RTX PRO 6000 Blackwell (MIG 2g.48gb), CUDA Toolkit 13.5, driver 13.2, against the branch at the point where kernel indexing, autodiff and the deterministic-mode rejection were complete (before the literal-index check, the size check on the descriptor build, the final-review fixes and the rename to `wp.largearray`):

| Check | CPU | CUDA |
| --- | --- | --- |
| `wp.zeros`/`fill_`/`zero_`, `wp.full`, `zeros_like`, `clone`, `empty`, `empty_like`; write past 2³¹ read back through `int64` and `uint32` indices | PASS | PASS |
| Launch over all N threads with `wp.linear_tid()` | PASS (1.87 s) | PASS (0.01 s) |
| `wp.copy` full length and with offsets past 2³¹ | PASS | PASS |
| `wp.copy` CPU ↔ CUDA (pageable and pinned) | PASS | |
| `.numpy()`, `__array_interface__`, `wp.array(ndarray, copy=False)`, `wp.from_numpy` | PASS | n/a (zero-copy NumPy is CPU only; the CUDA `.numpy()` round trip passed inside the CPU check) |
| DLPack export/import, `__cuda_array_interface__` | PASS | PASS |
| `wp.to_torch`/`wp.from_torch` (including a large `requires_grad` tensor) | PASS | PASS |
| `wp.to_jax`/`wp.from_jax` (JAX indexing needs `jax_enable_x64`) | PASS | PASS |
| Gradient across the 2³¹ boundary (`float16`, tape, `zero()`) | PASS | PASS |
| Byte offsets past 4 GB (`float64`, 2²⁹ + 16 elements) | PASS | PASS |
| Index past 2³² (`int8`, 2³² + 16 elements) | PASS | PASS |
| Real large array rejected by slicing, `reshape`, `view`, struct fields, `wp.utils.array_sum`, and a `wp.array` kernel parameter | PASS | PASS |

Torch (2.15.0+cu132) and JAX (0.12.1) ran in separate temporary environments because their CUDA wheels cannot be installed together. Not checked: mempool-disabled CUDA allocation, multi-GPU peer copies, and a `wp.copy` with a count above 2³¹ combined with non-zero offsets. This run found the literal-index truncation described under "Code generation".

**No-cost check.** Not a unit test. These results, like the benchmark results below, were measured before the branch was rebased onto `ershi/fix-adjoint-array-descriptor`, when it still contained the reverse-pass fix as one of its own commits and before its history was squashed; the base was `8a60f7374`. The generated `.cpp` and `.cu` of the modules built by a broad sample of existing tests and examples were compared between the base commit `8a60f7374` and this branch, each built in its own worktree with its own kernel cache, ignoring the module header. `sizeof(array_t)` was compared on both. Result:

- About 40 test classes (arrays, indexed and Fabric arrays, fixed arrays, structs, atomics, launches, generics, functions, autodiff, tape, jets, deterministic mode, tiles including MathDx matmul, vectors, matrices, quaternions, meshes, BVH, hash grid, volumes, sparse, and the core, optim and FEM examples) built 1,506 generated files on the base commit and 1,518 on the branch; 1,467 (490 `.cpp`, 977 `.cu`, 1,413 modules) had the same module hash on both sides and were compared after dropping the module header and comment-only lines (source echoes and `file:line` markers, which shift when a source file changes). Modules whose grouping depends on test scheduling (`wp.map`, jets, `test_grad`) were also compared function by function: 40,011 top-level functions and structs exist on both sides.
- Header: the only difference is the new `builtin_linear_tid()` macro (CPU and CUDA), which now comes from `ershi/linear-tid`. The measured commits also renamed the internal CPU template that the `builtin_tid1d()` macro calls; that rename has since been dropped.
- Bodies: 103 modules differ, and every difference is the removal of `adj_<array>.shape = adj_N;` or `adj_<array>.data = (T*)adj_N;` from a reverse pass, which is the deliberate fix under "Autodiff" (now on the separate branch, so against `main` these differences belong to that branch). Two other kinds of difference were traced to the test harness, not the branch: user-function overload indices (`jet_dot_34` vs `jet_dot_16`), which depend on which functions a worker process built first, and a swapped pair of variable numbers in 20 FEM `dispatch_bilinear_kernel_fn` kernels, which reproduces on the base commit alone between two runs with different `PYTHONHASHSEED` values.
- `ctypes.sizeof(warp._src.types.array_t)` is 56 on both, with identical field offsets (`data` 0, `grad` 8, `shape` 16, `strides` 32, `ndim` 48, `flags` 50).
- The fixes after the final review (guards, null check, launch-packing dispatch, tape-copy check) were checked the same way against the branch as it stood before them, by code generation only: the 120 Warp modules defined by 37 existing test modules (arrays, array reductions, atomics, booleans, code generation, conditionals, copies, gradients, custom gradients, indexed, Fabric and fixed arrays, launches, structs, tapes, utilities, vectors, matrices, quaternions, functions, generics, tiles, array subclasses, coloring, textures, math, meshes, mesh queries, marching cubes, hash grid, BVH, volumes, APIC meshes, sparse) gave 232 `.cpp` and `.cu` files with 5,058 kernel entry points. Apart from source line numbers in comments, which shift with the edited Python files, they are identical.

**Benchmarks.** The existing ASV suite (`asv/benchmarks/`) is compared between the base commit and the branch with `asv continuous`; any reproducible regression blocks the merge. A new benchmark, `asv/benchmarks/api/largearray.py`, records the cost of opting in: an `axpy` over 2²⁴ `float32` elements written with `wp.array` + `wp.tid()` and with `wp.largearray` + `wp.linear_tid()`, on CPU and CUDA, and the host overhead of a `dim=1` launch of each.

Results, base `8a60f7374` against the branch after the final-review fixes (measured before the rebase and squash, see "No-cost check"), run with `asv continuous --launch-method spawn --append-samples --interleave-rounds --no-only-changed` on one machine (CPU plus one CUDA device):

- **Core group** (`api`, `codegen`, `compile_conditional`, `control_flow`, `array_fill`, `atomics`, `memory_access`, `cpu_vectorization`, `cpu_helper_inlining`, `fixedarray`, `floordiv`, `deterministic`, `native_library_footprint`), measured right after the final-review fixes, 140 result rows. One benchmark was flagged: `api.array.ArrayEmpty.time_empty`, 83.0 ± 4 μs against 173 ± 300 μs. It did not reproduce: rerun twice on its own after three further small commits (described below) it was 83.1 μs against 83.1 μs both times, and ASV reported no significant change. The same benchmark was flagged once in an earlier run with the same ±300 μs spread and was level on rerun then too. Launch benchmarks: `KernelLaunch.time_standard_launch` 7.46 μs against 7.61 μs (1.02), `time_launch_object` 2.96 μs against 2.95 μs (1.00); `StructSetup.time_construct_simple` 899 ns against 889 ns (0.99); `ArrayZeros.time_zeros` 86.5 μs against 86.6 μs (1.00).
- **Remaining groups** (`examples`, `fem`, `sparse`, `optim`, `tile`, `spatial_query`, `bvh_build`, `volume_build`, `texture`), measured after those three commits, 444 result rows: none got worse and none got better at ASV's 1.1 factor. One benchmark, `sparse.transpose.BsrSetTransposeCompact.time_100_transposes('overallocated')`, failed on both the base commit and the branch with a CUDA out-of-memory error during graph creation, so it gives no comparison.
- The three small commits made after the core group was measured restore three lines in an error-message helper, correct comments and docs, and add one test; they do not touch a measured path, so the core group was not rerun.

Cost of opting in, from the new benchmark, measured with the core group (the same `axpy` kernel over 2²⁴ `float32` elements):

| | `wp.array` + `wp.tid()` | `wp.largearray` + `wp.linear_tid()` |
| --- | --- | --- |
| CUDA | 69.2 ms | 69.2 ms |
| CPU | 2.50 ms | 8.11 ms |
| Launch overhead, CPU | 6.42 μs | 6.89 μs |
| Launch overhead, CUDA | 9.11 μs | 9.70 μs |

On CUDA the 64-bit path costs nothing for this kernel. On CPU it is about 3.2 times slower. The cause has not been established; the `wp.array` variant is eligible for the linear-contiguous CPU loop specialisation and the `wp.largearray` variant is not, which is the most likely explanation. CPU parity was not a requirement of this design.

An earlier gate (after the kernel-access work) flagged `api.struct.StructSetup.time_construct_simple` and `codegen.mat22_augassign.CompileModule.time_cuda_codegen`; neither reproduced in interleaved measurements outside ASV. Those measurements did find the per-call guard cost that led to option A (see "History of this decision"), and the gate then passed on option A. Option C, the mechanism now in the code, has its own ASV result under "Results" above.

**Not tested in CI.** Large allocations are not exercised by the unit tests. How to test them robustly in CI is deferred.

## Open Items

- **Literal indices** that reach a `largearray` index through arithmetic, copies, branches or constant expressions still wrap silently.
- **Generic `wp.largearray[Any]` parameters** do not accept arrays at launch.
- **Pre-existing, found during this work, not changed:** `wp.indexedarray` composite-slot updates (`y[i][1] += v`) silently drop their gradients; `wp.config.verify_autograd_array_access` has no effect on modules loaded before it is set.
- **Possible follow-ups:** R14 (slicing a large array into a view that fits), multi-dimensional `largearray`, `largearray` in structs, deterministic-mode and APIC support, `wp.autograd.jacobian` tests with `largearray` (probed by hand and correct, but untested).
