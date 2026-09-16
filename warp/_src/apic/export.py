# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prepare metadata and publish the files belonging to one APIC save."""

import ctypes
import os
import shutil
import tempfile

import warp
from warp._src import context
from warp._src.apic.types import (
    APIC_BINARY_CPU_OBJECT,
    APIC_BINARY_CUBIN,
    APIC_BINARY_PTX,
    APICExportBinding,
    APICExportDescriptor,
    APICExportKernel,
    APICExportModule,
)
from warp._src.cuda_build import compile_cuda


def _publish(staged_wrp, staged_modules, wrp_path, modules_dir):
    """Publish a bundle, retaining the old companions until the graph is replaced."""
    backup = None
    published_modules = False
    try:
        if os.path.lexists(modules_dir):
            if os.path.islink(modules_dir) or not os.path.isdir(modules_dir):
                raise RuntimeError(f"APIC companion path is not a directory: {modules_dir}")
            backup = tempfile.mkdtemp(prefix=".warp-apic-old-", dir=os.path.dirname(modules_dir))
            os.rmdir(backup)
            os.replace(modules_dir, backup)
        os.replace(staged_modules, modules_dir)
        published_modules = True
        try:
            os.replace(staged_wrp, wrp_path)
        except OSError as exc:
            raise RuntimeError(f"Failed to save APIC graph to {wrp_path}. {exc}") from exc
    except Exception:
        if published_modules:
            shutil.rmtree(modules_dir)
        if backup is not None and os.path.exists(backup):
            os.replace(backup, modules_dir)
        raise
    else:
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)


def save(graph, path, inputs, outputs, target_arch, use_ptx):
    """Serialize a capture using metadata owned exclusively by this call."""
    if not graph.apic:
        raise RuntimeError(
            "Graph was not captured with apic=True. Pass apic=True to capture_begin() or ScopedCapture()."
        )
    capture = graph._apic_capture
    if capture is None or graph.apic_state is None:
        raise RuntimeError("Graph has no APIC recording state.")
    if not isinstance(use_ptx, bool):
        raise TypeError("use_ptx must be a Boolean.")
    if use_ptx and target_arch is None:
        raise ValueError("use_ptx=True requires an explicit target_arch.")
    if target_arch is not None:
        if isinstance(target_arch, bool) or not isinstance(target_arch, int):
            raise TypeError("target_arch must be a positive integer.")
        if target_arch <= 0:
            raise ValueError("target_arch must be a positive integer.")
        if not graph.device.is_cuda:
            raise ValueError("target_arch requires a CUDA APIC graph.")
        context.init()
        if target_arch not in context.runtime.nvrtc_supported_archs:
            raise ValueError(
                f"target_arch {target_arch} is not supported by the bundled CUDA compiler. "
                f"Supported architectures: {sorted(context.runtime.nvrtc_supported_archs)}."
            )

    runtime = context.runtime
    state = graph.apic_state
    save_arch = target_arch if target_arch is not None else (graph.device.arch if graph.device.is_cuda else 0)
    kind_values = {"cubin": APIC_BINARY_CUBIN, "ptx": APIC_BINARY_PTX, "object": APIC_BINARY_CPU_OBJECT}
    modules = []
    binaries = []
    targeted_kernels = {}
    for module_hash, info in capture.collected_modules.items():
        executable = info["module_exec"]
        if target_arch is None:
            binary_path = executable.binary_path
            kind = executable.binary_kind
            arch = executable.compile_arch or 0
            suffix = executable.arch_suffix
        else:
            kind = "ptx" if use_ptx else "cubin"
            record = info["compile_record"]
            try:
                if record is None:
                    raise RuntimeError(
                        "CUDA compile record is unavailable; load the module normally and recapture the graph."
                    )
                if record.native_options.llvm_cuda:
                    raise RuntimeError("Targeted export is unsupported for LLVM CUDA.")
                suffix = context._validate_cuda_arch_suffix(
                    target_arch, target_arch, runtime.toolkit_version, suffix=record.native_options.cuda_arch_suffix
                )
                artifact, _ = compile_cuda(
                    record,
                    info["module_name"],
                    warp.config.kernel_cache_dir,
                    target_arch,
                    suffix,
                    kind,
                    pch_dir=runtime.get_nvrtc_pch_dir(),
                    use_cache=warp.config.cache_kernels,
                )
                binary_path, kind, arch, suffix = (
                    artifact.binary_path,
                    artifact.binary_kind,
                    artifact.target_arch,
                    artifact.arch_suffix,
                )
                targeted_kernels[module_hash] = {kernel.forward_name: kernel for kernel in artifact.kernels}
            except Exception as exc:
                raise RuntimeError(
                    f"Cannot export captured module {info['module_name']!r} hash {module_hash[:8]} "
                    f"as {kind} for target_arch={target_arch}: {exc}"
                ) from exc
        if not binary_path or not os.path.isfile(binary_path):
            raise RuntimeError(
                f"APIC: Could not find compiled binary for module {info['module_name']} at {binary_path}. "
                "Ensure modules are compiled before calling capture_save()."
            )
        filename = f"wp_{module_hash}.o" if kind == "object" else f"wp_{module_hash}.sm{arch}{suffix}.{kind}"
        binaries.append((binary_path, filename))
        modules.append(
            APICExportModule(
                module_hash.encode(),
                info["module_name"].encode(),
                filename.encode(),
                kind_values[kind],
                arch,
                suffix.encode(),
            )
        )

    kernels = []
    for info in capture.collected_kernels.values():
        captured = info["descriptor"]
        kernel = captured
        if target_arch is not None:
            kernel = targeted_kernels[info["module_hash"]].get(captured.forward_name)
            if kernel is None or kernel.backward_name != captured.backward_name:
                raise RuntimeError(
                    f"Cannot export captured kernel {info['kernel_key']!r}: target entry points do not match the capture."
                )
        for symbol, size in (
            (kernel.forward_name, kernel.forward_smem_bytes),
            (kernel.backward_name, kernel.backward_smem_bytes),
        ):
            if isinstance(size, bool) or not isinstance(size, int) or not 0 <= size <= 0x7FFFFFFF:
                raise RuntimeError(
                    f"Cannot export captured symbol {symbol!r}: shared memory size must be a non-negative 32-bit integer."
                )
        kernels.append(
            APICExportKernel(
                info["kernel_key"].encode(),
                info["module_hash"].encode(),
                captured.forward_name.encode(),
                captured.backward_name.encode(),
                kernel.forward_smem_bytes,
                kernel.backward_smem_bytes,
                info["block_dim"],
            )
        )

    # Outputs take precedence when the same name occurs in both dictionaries.
    binding_arrays = dict(inputs or {})
    binding_arrays.update(outputs or {})
    bindings = [
        APICExportBinding(name.encode(), capture.get_region_id(array)) for name, array in binding_arrays.items()
    ]
    # ctypes structures retain their encoded strings; these arrays and descriptor
    # remain alive throughout the synchronous native save below.
    module_array = (APICExportModule * len(modules))(*modules)
    kernel_array = (APICExportKernel * len(kernels))(*kernels)
    binding_array = (APICExportBinding * len(bindings))(*bindings)
    descriptor = APICExportDescriptor(
        module_array, len(modules), kernel_array, len(kernels), binding_array, len(bindings)
    )

    for region_id, base_ptr, capacity, _base in capture._regions.values():
        if graph.device.is_cuda:
            if region_id in capture._transient_regions:
                # Graph-scoped storage is regenerated by replay; its live backing
                # no longer exists after capture ends.
                runtime.core.wp_apic_register_memory_region(state, region_id, capacity, 1, None)
                continue
            host_buffer = (ctypes.c_uint8 * capacity)()
            ok = runtime.core.wp_memcpy_d2h(
                graph.device.context,
                ctypes.addressof(host_buffer),
                ctypes.c_void_p(base_ptr),
                capacity,
                graph.device.stream.cuda_stream,
            )
            warp.synchronize_device(graph.device)
            if not ok:
                raise RuntimeError(
                    f"APIC: region {region_id} could not be snapshotted for capture_save "
                    "(device-to-host copy failed for a non-transient region); the saved "
                    "graph would be missing this region's initial data."
                )
            data = ctypes.addressof(host_buffer)
        else:
            data = ctypes.c_void_p(base_ptr)
        runtime.core.wp_apic_register_memory_region(state, region_id, capacity, 1, data)
    for mesh_id in capture.collected_mesh_ids:
        if not runtime.core.wp_apic_register_mesh(state, ctypes.c_uint64(mesh_id)):
            raise RuntimeError(f"APIC: failed to register mesh for capture_save. {runtime.get_error_string()}")

    path = os.fspath(path)
    wrp_path = os.path.abspath(path if path.endswith(".wrp") else path + ".wrp")
    modules_dir = wrp_path[:-4] + "_modules"
    parent = os.path.dirname(wrp_path)
    os.makedirs(parent, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".warp-apic-save-", dir=parent) as staging:
        staged_modules = os.path.join(staging, "modules")
        os.mkdir(staged_modules)
        for source, filename in binaries:
            shutil.copy2(source, os.path.join(staged_modules, filename))
        staged_wrp = os.path.join(staging, "graph.wrp")
        cuda_context = graph.device.context if graph.device.is_cuda else None
        if not runtime.core.wp_apic_state_save(
            state, staged_wrp.encode(), save_arch, cuda_context, ctypes.byref(descriptor)
        ):
            raise RuntimeError(f"Failed to save APIC graph to {wrp_path}. {runtime.get_error_string()}")
        _publish(staged_wrp, staged_modules, wrp_path, modules_dir)
