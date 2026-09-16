# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Immutable CUDA compilation records used by targeted export.

This module deliberately has no Warp imports.  It can therefore describe a
captured CUDA compilation without loading the runtime or a native library.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import weakref
from collections.abc import Mapping
from dataclasses import asdict, dataclass, field, fields
from typing import Any, ClassVar, Literal

CUDA_COMPILE_RECORD_VERSION = 2
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_int(value: object, name: str) -> int:
    if type(value) is not int:
        raise TypeError(f"{name} must be an int, got {type(value).__name__}")
    return value


def _require_non_negative_int(value: object, name: str) -> int:
    value = _require_int(value, name)
    if value < 0:
        raise ValueError(f"{name} must be non-negative, got {value}")
    return value


def _require_sha256(value: object, name: str) -> str:
    if type(value) is not str or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256 hexadecimal digest")
    return value


def _sha256_file(path: str) -> str:
    with open(path, "rb") as file:
        return hashlib.sha256(file.read()).hexdigest()


def recipe_output_macro(recipe_key: str, output_name: str) -> str:
    """Return the stable source macro for a MathDx recipe output."""
    if re.fullmatch(r"[0-9a-f]{16}", recipe_key) is None:
        raise ValueError(f"Invalid MathDx recipe key {recipe_key!r}")
    if re.fullmatch(r"[a-z][a-z0-9_]*", output_name) is None:
        raise ValueError(f"Invalid MathDx recipe output {output_name!r}")
    return f"WP_MATHDX_{recipe_key.upper()}_{output_name.upper()}"


class MathDxRecipe:
    """Common derived identity of a concrete, normalized MathDx input."""

    kind: ClassVar[str]
    output_names: ClassVar[frozenset[str]] = frozenset()

    def __post_init__(self):
        for item in fields(self):
            _require_int(getattr(self, item.name), f"{self.kind} {item.name}")
        # Exercise derived fields at construction, rejecting unsupported enums.
        _ = self.symbol, self.declaration

    @property
    def key(self) -> str:
        payload = json.dumps(_recipe_to_json(self), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    def cache_key(self, effective_arch: int) -> str:
        """Return the architecture-specific cache key for this recipe."""
        _require_non_negative_int(effective_arch, "effective_arch")
        return f"{self.kind}:{self.key}:sm{effective_arch}"


def _mathdx_type(precision: int, element_type: int = 0) -> str:
    types = {
        (2, 0): "wp::bfloat16",
        (3, 0): "wp::float16",
        (5, 0): "wp::float32",
        (6, 0): "wp::float64",
        (3, 1): "wp::vec2h",
        (5, 1): "wp::vec2f",
        (6, 1): "wp::vec2d",
    }
    try:
        return types[precision, element_type]
    except KeyError as error:
        raise ValueError(f"Unsupported MathDx precision/element type {(precision, element_type)!r}") from error


@dataclass(frozen=True)
class DotRecipe(MathDxRecipe):
    """Normalized cuBLASDx compilation inputs, including optional strides."""

    kind: ClassVar[str] = "dot"
    M: int
    N: int
    K: int
    a_prec: int
    b_prec: int
    c_prec: int
    element_type: int
    a_arrangement: int
    b_arrangement: int
    c_arrangement: int
    num_threads: int
    lda: int = 0
    ldb: int = 0
    ldc: int = 0

    @property
    def symbol(self) -> str:
        name = (
            f"dot_{self.M}_{self.N}_{self.K}_{self.num_threads}_{self.a_arrangement}_{self.b_arrangement}_"
            f"{self.c_arrangement}_{self.a_prec}_{self.b_prec}_{self.c_prec}_{self.element_type}"
        )
        if any((self.lda, self.ldb, self.ldc)):
            name += f"_{self.lda}_{self.ldb}_{self.ldc}"
        return name

    @property
    def declaration(self) -> str:
        a = _mathdx_type(self.a_prec, self.element_type)
        b = _mathdx_type(self.b_prec, self.element_type)
        c = _mathdx_type(self.c_prec, self.element_type)
        return f"void {self.symbol}({c}*, {a}*, {b}*, {c}*, {c}*);"


@dataclass(frozen=True)
class FftRecipe(MathDxRecipe):
    """Normalized cuFFTDx compilation inputs."""

    kind: ClassVar[str] = "fft"
    output_names: ClassVar[frozenset[str]] = frozenset(("shared_memory_bytes",))
    size: int
    ept: int
    direction_enum: int
    precision: int

    @property
    def direction(self) -> str:
        if self.direction_enum not in (0, 1):
            raise ValueError(f"Unsupported FFT direction {self.direction_enum!r}")
        return "forward" if self.direction_enum == 0 else "inverse"

    @property
    def symbol(self) -> str:
        return f"fft_{self.size}_{self.ept}_{self.direction}_{self.precision}"

    @property
    def declaration(self) -> str:
        dtype = _mathdx_type(self.precision, 1)
        return f"void {self.symbol}({dtype}*, char*);"


@dataclass(frozen=True)
class SolverRecipe(MathDxRecipe):
    """Normalized cuSolverDx compilation inputs."""

    kind: ClassVar[str] = "solver"
    M: int
    N: int
    K: int
    solver_enum: int
    side_enum: int
    diag_enum: int
    a_arrangement: int
    b_arrangement: int
    fill_mode: int
    precision_enum: int
    num_threads: int

    @property
    def solver(self) -> str:
        names = {0: "getrf", 1: "getrf_no_pivot", 2: "potrf", 3: "potrs", 4: "trsm"}
        try:
            return names[self.solver_enum]
        except KeyError as error:
            raise ValueError(f"Unsupported solver {self.solver_enum!r}") from error

    @property
    def symbol(self) -> str:
        return (
            f"{self.solver}_{self.M}_{self.N}_{self.K}_{self.num_threads}_{self.a_arrangement}_"
            f"{self.b_arrangement}_{self.precision_enum}_{self.side_enum if self.side_enum >= 0 else 'x'}_"
            f"{self.diag_enum if self.diag_enum >= 0 else 'x'}_{self.fill_mode}"
        )

    @property
    def declaration(self) -> str:
        dtype = _mathdx_type(self.precision_enum)
        if self.solver_enum == 0:
            parameters = f"({dtype}*, int*, int*)"
        elif self.solver_enum in (1, 2):
            parameters = f"({dtype}*, int*)"
        else:
            parameters = f"({dtype}*, {dtype}*)"
        return f"void {self.symbol}{parameters};"


_RECIPE_TYPES = {recipe.kind: recipe for recipe in (DotRecipe, FftRecipe, SolverRecipe)}


def kernel_smem_macro(symbol: str) -> str:
    """Return a stable macro for one kernel entry point's resolved shared memory."""
    if type(symbol) is not str or not symbol:
        raise ValueError("Kernel symbol must be a non-empty string")
    return "WP_KERNEL_SMEM_" + hashlib.sha256(symbol.encode()).hexdigest().upper()


@dataclass(frozen=True)
class CompiledKernel:
    """Kernel entry points and shared memory resolved for a CUDA target."""

    forward_name: str
    backward_name: str
    forward_smem_bytes: int
    backward_smem_bytes: int
    cluster_dim: int = 1


@dataclass(frozen=True)
class CudaKernel:
    """Kernel entry points with target-dependent shared-memory requirements."""

    forward_name: str
    backward_name: str
    forward_smem: IntExpression
    backward_smem: IntExpression
    cluster_dim: int = 1

    def resolve(self, outputs: Mapping[str, Mapping[str, int]]) -> CompiledKernel:
        return CompiledKernel(
            self.forward_name,
            self.backward_name,
            self.forward_smem.resolve(outputs),
            self.backward_smem.resolve(outputs),
            self.cluster_dim,
        )


@dataclass(frozen=True)
class IntExpression:
    """A canonical integer expression whose recipe outputs resolve per target."""

    op: Literal["constant", "recipe_output", "add", "maximum"]
    value: int | None = None
    recipe_key: str | None = None
    output_name: str | None = None
    operands: tuple[IntExpression, ...] = ()

    @classmethod
    def constant(cls, value: int) -> IntExpression:
        """Return an integer literal expression."""
        return cls("constant", value=_require_int(value, "value"))

    @classmethod
    def recipe_output(cls, recipe_key: str, output_name: str) -> IntExpression:
        """Return an expression which obtains one value from a MathDx recipe."""
        recipe_output_macro(recipe_key, output_name)
        return cls("recipe_output", recipe_key=recipe_key, output_name=output_name)

    @classmethod
    def add(cls, *operands: IntExpression) -> IntExpression:
        """Return a flattened sum with all literals folded into one operand."""
        flattened: list[IntExpression] = []
        constant = 0
        for operand in operands:
            if not isinstance(operand, cls):
                raise TypeError(f"add operands must be IntExpression instances, got {type(operand).__name__}")
            children = operand.operands if operand.op == "add" else (operand,)
            for child in children:
                if child.op == "constant":
                    constant += _require_int(child.value, "constant value")
                else:
                    flattened.append(child)
        if constant:
            flattened.append(cls.constant(constant))
        if not flattened:
            return cls.constant(0)
        flattened.sort(key=_expression_sort_key)
        if len(flattened) == 1:
            return flattened[0]
        return cls("add", operands=tuple(flattened))

    @classmethod
    def maximum(cls, *operands: IntExpression) -> IntExpression:
        """Return a flattened, de-duplicated maximum with literals folded."""
        flattened: list[IntExpression] = []
        for operand in operands:
            if not isinstance(operand, cls):
                raise TypeError(f"maximum operands must be IntExpression instances, got {type(operand).__name__}")
            flattened.extend(operand.operands if operand.op == "maximum" else (operand,))
        unique = {operand: operand for operand in flattened}
        operands = sorted(unique.values(), key=_expression_sort_key)
        constants = [operand.value for operand in operands if operand.op == "constant"]
        if constants:
            constant = max(_require_int(value, "constant value") for value in constants)
            operands = [operand for operand in operands if operand.op != "constant"]
            operands.append(cls.constant(constant))
            operands.sort(key=_expression_sort_key)
        if not operands:
            raise ValueError("maximum requires at least one operand")
        if len(operands) == 1:
            return operands[0]
        return cls("maximum", operands=tuple(operands))

    def resolve(self, outputs: Mapping[str, Mapping[str, int]]) -> int:
        """Resolve this expression using target-specific MathDx recipe outputs."""
        if self.op == "constant":
            return _require_non_negative_int(self.value, "constant value")
        if self.op == "recipe_output":
            if self.recipe_key not in outputs or self.output_name not in outputs[self.recipe_key]:
                raise ValueError(f"Missing MathDx recipe output {self.recipe_key!r}:{self.output_name!r}")
            value = outputs[self.recipe_key][self.output_name]
            if type(value) is not int:
                raise ValueError(f"recipe output must be an integer, got {type(value).__name__}")
            if value < 0:
                raise ValueError(f"recipe output must be non-negative, got {value}")
            return value
        if self.op == "add":
            return _require_non_negative_int(sum(operand.resolve(outputs) for operand in self.operands), "sum")
        if self.op == "maximum":
            if not self.operands:
                raise ValueError("maximum requires at least one operand")
            return _require_non_negative_int(max(operand.resolve(outputs) for operand in self.operands), "maximum")
        raise ValueError(f"Unsupported integer expression operation {self.op!r}")

    @property
    def source_expression(self) -> str:
        """Return the stable C++ expression used in generated source."""
        if self.op == "constant":
            return str(self.value)
        if self.op == "recipe_output":
            return recipe_output_macro(self.recipe_key, self.output_name)
        raise ValueError("Source expressions must be constants or direct recipe outputs")


def _expression_sort_key(expression: IntExpression) -> str:
    return json.dumps(_expression_to_json(expression), sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class CudaNativeOptions:
    """The runtime-independent subset of options passed to ``build_cuda()``."""

    mode: str
    optimization_level: int
    verify_fp: bool
    fast_math: bool
    fuse_fp: bool
    lineinfo: bool
    compile_time_trace: bool
    llvm_cuda: bool
    use_precompiled_headers: bool
    extra_cuda_include_dirs: tuple[str, ...]
    cuda_arch_suffix: str


@dataclass(frozen=True)
class _CudaSource:
    """Weak-referenceable owner of immutable source bytes shared by live records."""

    data: bytes


_cuda_sources: weakref.WeakValueDictionary[str, _CudaSource] = weakref.WeakValueDictionary()
_cuda_sources_lock = threading.Lock()


def _share_cuda_source(source: bytes, digest: str) -> _CudaSource:
    with _cuda_sources_lock:
        snapshot = _cuda_sources.get(digest)
        if snapshot is None:
            snapshot = _CudaSource(source)
            _cuda_sources[digest] = snapshot
        elif snapshot.data != source:
            raise ValueError("CUDA source digest collision")
        return snapshot


@dataclass(frozen=True)
class CudaCompileRecord:
    """Canonical description of a CUDA compilation with non-serialized source bytes."""

    version: int
    module_hash: str
    block_dim: int
    source_basename: str
    source_sha256: str
    native_options: CudaNativeOptions
    recipes: tuple[MathDxRecipe, ...]
    kernels: tuple[CudaKernel, ...]
    dependencies: tuple[tuple[str, str], ...]
    _source: _CudaSource = field(repr=False, metadata={"serialize": False})

    @property
    def source(self) -> bytes:
        """Return source bytes shared across records for as long as they are live."""
        return self._source.data

    @classmethod
    def create(
        cls,
        *,
        module_hash: bytes | str,
        block_dim: int,
        source: bytes,
        source_basename: str,
        native_options: CudaNativeOptions,
        recipes: tuple[MathDxRecipe, ...],
        kernels: tuple[CudaKernel, ...],
        dependencies: tuple[str | os.PathLike[str] | tuple[str, str], ...],
    ) -> CudaCompileRecord:
        """Capture source and dependency digests into a canonical record."""
        if isinstance(module_hash, bytes):
            if len(module_hash) != 32:
                raise ValueError("module_hash must contain 32 bytes")
            module_hash = module_hash.hex()
        _require_sha256(module_hash, "module_hash")
        _require_non_negative_int(block_dim, "block_dim")
        if block_dim == 0:
            raise ValueError("block_dim must be greater than zero")
        if not isinstance(native_options, CudaNativeOptions):
            raise TypeError("native_options must be a CudaNativeOptions instance")
        _validate_native_options(native_options)

        _validate_source_basename(source_basename)
        if type(source) is not bytes:
            raise TypeError("source must be immutable bytes")
        source_sha256 = hashlib.sha256(source).hexdigest()

        canonical_recipes = _canonical_recipes(recipes)
        canonical_kernels = _canonical_kernels(kernels, canonical_recipes)
        canonical_dependencies = _canonical_dependencies(dependencies)
        return cls(
            CUDA_COMPILE_RECORD_VERSION,
            module_hash,
            block_dim,
            source_basename,
            source_sha256,
            native_options,
            canonical_recipes,
            canonical_kernels,
            canonical_dependencies,
            _share_cuda_source(source, source_sha256),
        )

    def to_json(self) -> dict[str, Any]:
        """Return the versioned record payload without its in-memory source."""
        return {
            "version": self.version,
            "module_hash": self.module_hash,
            "block_dim": self.block_dim,
            "source_basename": self.source_basename,
            "source_sha256": self.source_sha256,
            "native_options": _native_options_to_json(self.native_options),
            "recipes": [_recipe_to_json(recipe) for recipe in self.recipes],
            "kernels": [_kernel_to_json(kernel) for kernel in self.kernels],
            "dependencies": [list(dependency) for dependency in self.dependencies],
        }

    def _canonical_json_bytes(self) -> bytes:
        """Return the versioned record payload in its unique JSON representation."""
        payload = self.to_json()
        return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()

    def fingerprint(self) -> str:
        """Return the SHA-256 fingerprint of the canonical JSON payload."""
        return hashlib.sha256(self._canonical_json_bytes()).hexdigest()

    @classmethod
    def from_json(cls, payload: object, source: bytes) -> CudaCompileRecord:
        """Restore a record from a cache manifest and its immutable source bytes."""
        try:
            return _record_from_json(payload, source)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"Invalid CUDA compile record: {error}") from error

    def validate(self, module_name: str) -> None:
        """Ensure the frozen source and external dependencies remain valid."""
        if type(module_name) is not str:
            raise TypeError("module_name must be a string")
        if not isinstance(self.source, bytes) or hashlib.sha256(self.source).hexdigest() != self.source_sha256:
            raise RuntimeError(f"CUDA compile record for module {module_name!r}: frozen source is invalid")

        for dependency_path, dependency_sha256 in self.dependencies:
            try:
                current_sha256 = _sha256_file(dependency_path)
            except OSError as error:
                raise RuntimeError(
                    f"CUDA compile record for module {module_name!r}: dependency {dependency_path!r} is missing"
                ) from error
            if current_sha256 != dependency_sha256:
                raise RuntimeError(
                    f"CUDA compile record for module {module_name!r}: dependency {dependency_path!r} changed after capture"
                )


def _canonical_recipes(recipes: object) -> tuple[MathDxRecipe, ...]:
    if not isinstance(recipes, tuple):
        raise TypeError("recipes must be a tuple")
    canonical: list[MathDxRecipe] = []
    keys: set[str] = set()
    symbols: set[str] = set()
    for recipe in recipes:
        if type(recipe) not in _RECIPE_TYPES.values():
            raise TypeError("recipes must contain MathDxRecipe instances")
        if recipe.key in keys:
            raise ValueError(f"Duplicate MathDx recipe key {recipe.key!r}")
        if recipe.symbol in symbols:
            raise ValueError(f"Duplicate MathDx recipe symbol {recipe.symbol!r}")
        keys.add(recipe.key)
        symbols.add(recipe.symbol)
        canonical.append(recipe)
    return tuple(sorted(canonical, key=lambda recipe: (recipe.key, recipe.symbol)))


def _validate_expression_recipes(expression: IntExpression, recipes: Mapping[str, MathDxRecipe]) -> None:
    if expression.op == "constant":
        _require_int(expression.value, "constant value")
        return
    if expression.op == "recipe_output":
        recipe_output_macro(expression.recipe_key, expression.output_name)
        recipe = recipes.get(expression.recipe_key)
        if recipe is None:
            raise ValueError(f"Metadata refers to absent MathDx recipe {expression.recipe_key!r}")
        if expression.output_name not in recipe.output_names:
            raise ValueError(f"MathDx recipe {recipe.key!r} does not declare output {expression.output_name!r}")
        return
    if expression.op in ("add", "maximum"):
        if not expression.operands:
            raise ValueError(f"{expression.op} requires at least one operand")
        for operand in expression.operands:
            if not isinstance(operand, IntExpression):
                raise TypeError(f"{expression.op} operands must be IntExpression instances")
            _validate_expression_recipes(operand, recipes)
        return
    raise ValueError(f"Unsupported integer expression operation {expression.op!r}")


def _canonical_kernels(kernels: object, recipes: tuple[MathDxRecipe, ...]) -> tuple[CudaKernel, ...]:
    if not isinstance(kernels, tuple):
        raise TypeError("kernels must be a tuple")
    recipes_by_key = {recipe.key: recipe for recipe in recipes}
    seen = set()
    for kernel in kernels:
        if not isinstance(kernel, CudaKernel):
            raise TypeError("kernels must contain CudaKernel instances")
        if type(kernel.forward_name) is not str or not kernel.forward_name or type(kernel.backward_name) is not str:
            raise ValueError("Kernel names must be strings and the forward name must be non-empty")
        for name in (kernel.forward_name, kernel.backward_name):
            if name:
                if name in seen:
                    raise ValueError(f"Duplicate kernel symbol {name!r}")
                seen.add(name)
        if _require_non_negative_int(kernel.cluster_dim, "cluster_dim") == 0:
            raise ValueError("cluster_dim must be greater than zero")
        for expression in (kernel.forward_smem, kernel.backward_smem):
            if not isinstance(expression, IntExpression):
                raise TypeError("Kernel shared memory must be an IntExpression")
            _validate_expression_recipes(expression, recipes_by_key)
    return tuple(sorted(kernels, key=lambda kernel: kernel.forward_name))


def _kernel_to_json(kernel: CudaKernel) -> dict[str, Any]:
    return {
        "forward_name": kernel.forward_name,
        "backward_name": kernel.backward_name,
        "forward_smem": _expression_to_json(kernel.forward_smem),
        "backward_smem": _expression_to_json(kernel.backward_smem),
        "cluster_dim": kernel.cluster_dim,
    }


def _kernel_from_json(payload: object) -> CudaKernel:
    if not isinstance(payload, dict):
        raise ValueError("kernels must be objects")
    _require_exact_keys(payload, {item.name for item in fields(CudaKernel)}, "kernel")
    return CudaKernel(
        **(
            payload
            | {
                "forward_smem": _expression_from_json(payload["forward_smem"]),
                "backward_smem": _expression_from_json(payload["backward_smem"]),
            }
        )
    )


def _canonical_dependencies(dependencies: object) -> tuple[tuple[str, str], ...]:
    if not isinstance(dependencies, tuple):
        raise TypeError("dependencies must be a tuple")
    result: dict[str, str] = {}
    for dependency in dependencies:
        if isinstance(dependency, tuple):
            if len(dependency) != 2 or type(dependency[0]) is not str:
                raise TypeError("dependency entries must be paths or (path, sha256) pairs")
            path, digest = dependency
            if not os.path.isabs(path):
                raise ValueError(f"dependency {path!r} must be absolute")
            path = os.path.realpath(path)
            _require_sha256(digest, "dependency sha256")
        else:
            path = os.fspath(dependency)
            if not os.path.isabs(path):
                raise ValueError(f"dependency {path!r} must be absolute")
            path = os.path.realpath(path)
            if not os.path.isfile(path):
                raise ValueError(f"dependency {path!r} must name an existing file")
            digest = _sha256_file(path)
        if path in result and result[path] != digest:
            raise ValueError(f"Conflicting dependency digests for {path!r}")
        result[path] = digest
    return tuple(sorted(result.items()))


def _expression_to_json(expression: IntExpression) -> dict[str, Any]:
    if expression.op == "constant":
        return {"op": "constant", "value": _require_int(expression.value, "constant value")}
    if expression.op == "recipe_output":
        recipe_output_macro(expression.recipe_key, expression.output_name)
        return {"op": "recipe_output", "recipe_key": expression.recipe_key, "output_name": expression.output_name}
    if expression.op in ("add", "maximum"):
        return {"op": expression.op, "operands": [_expression_to_json(operand) for operand in expression.operands]}
    raise ValueError(f"Unsupported integer expression operation {expression.op!r}")


def _expression_from_json(payload: object) -> IntExpression:
    if not isinstance(payload, dict) or type(payload.get("op")) is not str:
        raise ValueError("integer expressions must be objects with an operation")
    op = payload["op"]
    if op == "constant":
        _require_exact_keys(payload, {"op", "value"}, "constant expression")
        return IntExpression.constant(_require_int(payload["value"], "constant value"))
    if op == "recipe_output":
        _require_exact_keys(payload, {"op", "recipe_key", "output_name"}, "recipe output expression")
        if type(payload["recipe_key"]) is not str or type(payload["output_name"]) is not str:
            raise ValueError("recipe output expression values must be strings")
        return IntExpression.recipe_output(payload["recipe_key"], payload["output_name"])
    if op in ("add", "maximum"):
        _require_exact_keys(payload, {"op", "operands"}, f"{op} expression")
        operands = payload["operands"]
        if not isinstance(operands, list):
            raise ValueError(f"{op} expression operands must be a list")
        factory = IntExpression.add if op == "add" else IntExpression.maximum
        return factory(*(_expression_from_json(operand) for operand in operands))
    raise ValueError(f"Unsupported integer expression operation {op!r}")


def _recipe_to_json(recipe: MathDxRecipe) -> dict[str, Any]:
    return {"kind": recipe.kind, **asdict(recipe)}


def _recipe_from_json(payload: object) -> MathDxRecipe:
    if not isinstance(payload, dict) or type(payload.get("kind")) is not str:
        raise ValueError("recipes must be objects with a kind")
    recipe_type = _RECIPE_TYPES.get(payload["kind"])
    if recipe_type is None:
        raise ValueError(f"Unsupported MathDx recipe kind {payload['kind']!r}")
    names = {item.name for item in fields(recipe_type)}
    _require_exact_keys(payload, names | {"kind"}, "recipe")
    return recipe_type(**{name: payload[name] for name in names})


def _native_options_to_json(options: CudaNativeOptions) -> dict[str, Any]:
    _validate_native_options(options)
    return asdict(options) | {"extra_cuda_include_dirs": list(options.extra_cuda_include_dirs)}


def _native_options_from_json(payload: object) -> CudaNativeOptions:
    if not isinstance(payload, dict):
        raise ValueError("native_options must be an object")
    _require_exact_keys(payload, {field.name for field in fields(CudaNativeOptions)}, "native_options")
    include_dirs = payload["extra_cuda_include_dirs"]
    if not isinstance(include_dirs, list) or any(type(path) is not str for path in include_dirs):
        raise ValueError("extra_cuda_include_dirs must be a list of strings")
    options = CudaNativeOptions(**(payload | {"extra_cuda_include_dirs": tuple(include_dirs)}))
    _validate_native_options(options)
    return options


def _validate_native_options(options: CudaNativeOptions) -> None:
    if type(options.mode) is not str or not options.mode:
        raise ValueError("native_options mode must be a non-empty string")
    _require_non_negative_int(options.optimization_level, "native_options optimization_level")
    for name in (
        "verify_fp",
        "fast_math",
        "fuse_fp",
        "lineinfo",
        "compile_time_trace",
        "llvm_cuda",
        "use_precompiled_headers",
    ):
        if type(getattr(options, name)) is not bool:
            raise ValueError(f"native_options {name} must be a bool")
    if not isinstance(options.extra_cuda_include_dirs, tuple) or any(
        type(path) is not str for path in options.extra_cuda_include_dirs
    ):
        raise ValueError("native_options extra_cuda_include_dirs must be a tuple of strings")
    if type(options.cuda_arch_suffix) is not str:
        raise ValueError("native_options cuda_arch_suffix must be a string")


def _require_exact_keys(payload: dict[str, Any], expected: set[str], name: str) -> None:
    actual = set(payload)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(f"{name} fields are invalid (missing={missing}, extra={extra})")


def _validate_source_basename(source_basename: object) -> None:
    if (
        type(source_basename) is not str
        or not source_basename
        or source_basename in (".", "..")
        or os.path.basename(source_basename) != source_basename
        or "/" in source_basename
        or "\\" in source_basename
    ):
        raise ValueError("source_basename must not contain path separators or traversal")


def _record_from_json(payload: object, source: bytes) -> CudaCompileRecord:
    if not isinstance(payload, dict):
        raise ValueError("record payload must be an object")
    serialized_fields = {item.name for item in fields(CudaCompileRecord) if item.metadata.get("serialize", True)}
    _require_exact_keys(payload, serialized_fields, "record")
    if type(payload["version"]) is not int or payload["version"] != CUDA_COMPILE_RECORD_VERSION:
        raise ValueError(f"unsupported CUDA compile record version {payload['version']!r}")
    _require_sha256(payload["module_hash"], "module_hash")
    _require_non_negative_int(payload["block_dim"], "block_dim")
    if payload["block_dim"] == 0:
        raise ValueError("block_dim must be greater than zero")
    source_basename = payload["source_basename"]
    _validate_source_basename(source_basename)
    _require_sha256(payload["source_sha256"], "source_sha256")
    if not isinstance(source, bytes) or hashlib.sha256(source).hexdigest() != payload["source_sha256"]:
        raise ValueError("source contents do not match source_sha256")
    native_options = _native_options_from_json(payload["native_options"])

    recipes_payload = payload["recipes"]
    if not isinstance(recipes_payload, list):
        raise ValueError("recipes must be a list")
    recipes = _canonical_recipes(tuple(_recipe_from_json(recipe) for recipe in recipes_payload))

    if not isinstance(payload["kernels"], list):
        raise ValueError("kernels must be a list")
    kernels = _canonical_kernels(tuple(_kernel_from_json(kernel) for kernel in payload["kernels"]), recipes)
    dependencies = _canonical_dependencies(_string_pair_tuple(payload["dependencies"], "dependencies"))
    record = CudaCompileRecord(
        version=payload["version"],
        module_hash=payload["module_hash"],
        block_dim=payload["block_dim"],
        source_basename=source_basename,
        source_sha256=payload["source_sha256"],
        native_options=native_options,
        recipes=recipes,
        kernels=kernels,
        dependencies=dependencies,
        _source=_share_cuda_source(source, payload["source_sha256"]),
    )
    return record


def _string_pair_tuple(payload: object, name: str) -> tuple[tuple[str, str], ...]:
    if not isinstance(payload, list):
        raise ValueError(f"{name} must be a list")
    pairs: list[tuple[str, str]] = []
    for pair in payload:
        if not isinstance(pair, list) or len(pair) != 2 or type(pair[0]) is not str or type(pair[1]) is not str:
            raise ValueError(f"{name} entries must be string pairs")
        pairs.append((pair[0], pair[1]))
    return tuple(pairs)
