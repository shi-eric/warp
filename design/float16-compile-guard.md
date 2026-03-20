# Float16 Compile Guard (`WP_NO_FLOAT16`)

Mirror the `WP_NO_BFLOAT16` pattern to skip float16 overloads in `builtin.h` for modules that don't use `wp.float16`. This would further reduce LLVM CPU codegen time for the common case (float32/float64-only kernels).

## Codegen change (`warp/_src/context.py`)

Extend the existing detection in `ModuleBuilder.codegen` (around line 2262) to also check for `"float16"`:

```python
type_defines = ""
if "bfloat16" not in source:
    type_defines += "#define WP_NO_BFLOAT16\n"
if "float16" not in source:
    type_defines += "#define WP_NO_FLOAT16\n"
```

## `builtin.h` guard locations

Float16 code follows the same pattern as bfloat16. Wrap each with `#ifndef WP_NO_FLOAT16` / `#endif`:

1. **Struct + typedef** (~lines 99–169): `struct half { ... }; typedef half float16;`
2. **Conversion functions** (in all 3 platform branches — `__CUDA_ARCH__`, `__clang__`, native):
   - `float_to_half()`, `half_to_float()`
3. **Arithmetic operators** (~lines 401–420): `operator+`, `-`, `*`, `/` for `half`, plus mixed-type `half * float`, `half * double`
4. **`adj_cast_float(float16 ...)`** — single line near `adj_cast_float(bfloat16 ...)`
5. **`adj_float16` template** — single line near `adj_bfloat16`
6. **`isfinite(half)`, `isnan(half)`, `isinf(half)`** — one line each
7. **`print(float16)`** — one line
8. **`DECLARE_FLOAT_OPS(float16)`** — macro invocation
9. **`DECLARE_ADJ_APPROX_DIV(float16)`** — macro invocation
10. **`mod(float16 ...)`** — function definition
11. **`adj_erf`, `adj_erfc`, `adj_erfinv`, `adj_erfcinv` for float16** — if they exist (check; bfloat16 has them but float16 may use the generic `DECLARE_ADJOINTS` macro instead)
12. **Math function overloads** (~lines 1192–1333): `abs(half)`, `acos(half)`, ..., `frac(half)`, `sqrt(half)`, `log(half)`, etc.
13. **`DECLARE_ADJOINTS(float16)`** — macro invocation (this is the biggest single expansion)
14. **`DECLARE_INTERP_FUNCS(float16)`** — macro invocation
15. **`adj_print(half ...)`** — single line
16. **`approx_rcp(float16)`, `approx_div(float16, float16)`** — in both CUDA and CPU branches

## `cuda_crt.h` guard locations

1. **`struct __nv_fp16` + conversion helpers + `atomicAdd(__nv_fp16 ...)`** — the float16 CRT block (analogous to the bfloat16 block already guarded)

## `tile.h` guard locations

1. **`adj_tile_astype`** — the `is_same<..., wp::float16>` checks (already done for bfloat16, same pattern)

## Verification

1. Rebuild native lib: `uv run build_lib.py --quick`
2. Run the codegen benchmark to confirm further speedup for non-float16 kernels
3. Run `uv run warp/tests/test_half.py` (or equivalent float16 tests) to confirm float16 kernels still compile correctly
4. Run the full test suite: `uv run --extra dev -m warp.tests -s autodetect`
