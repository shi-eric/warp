"""Run a stock Newton benchmark with graph-capture progress markers.

Invoke with the reported Isaac Lab checkout's interpreter and benchmark CLI
arguments. The wrapper logs entry and return without changing capture settings.
"""

import argparse
import ctypes
import faulthandler
import importlib.metadata
import json
import runpy
import sys
import time
from pathlib import Path

import gymnasium as gym

faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--allocation-only", action="store_true")
parser.add_argument("--mujoco-only", action="store_true")
parser.add_argument("--kinematics-only", action="store_true")
parser.add_argument("--kernel-only", action="store_true")
parser.add_argument("--raw-allocation", action="store_true")
parser.add_argument("--nonblocking", action="store_true")
parser.add_argument("--new-blocking", action="store_true")
parser.add_argument(
    "--warmup-capture",
    choices=("solver", "early"),
    help="Capture one allocation to reserve the graph memory range: when Newton schedules capture, or before gym.make",
)
parser.add_argument("--dump-scene", type=Path)
options, benchmark_args = parser.parse_known_args()
sys.argv = [sys.argv[0], *benchmark_args]
retained_allocations = []
warmed_devices = set()
print("Diagnostic options:", vars(options), flush=True)
print(
    "Package versions:",
    {
        name: importlib.metadata.version(name)
        for name in ("warp-lang", "newton", "torch", "mujoco-warp", "ovrtx", "ovstage")
    },
    flush=True,
)
original_make = gym.make


def benchmark_device():
    devices = [benchmark_args[i + 1] for i, arg in enumerate(benchmark_args[:-1]) if arg == "--device"]
    return devices[-1] if devices else "cuda:0"


def reserve_graph_memory(device):
    """Make the process's first graph allocation, so the driver reserves its graph memory range now."""
    import warp as wp  # noqa: PLC0415

    if device in warmed_devices:
        return
    with wp.ScopedCapture(device=device, force_module_load=False):
        scratch = wp.empty(1, dtype=wp.int32, device=device)
        del scratch
    warmed_devices.add(device)
    print(f"GRAPH_WARMUP_CAPTURE device={device} perf_counter={time.perf_counter():.3f}", flush=True)


def logged_make(*args, **kwargs):
    print(f"GYM_MAKE_ENTER perf_counter={time.perf_counter():.3f}", flush=True)
    # Import after AppLauncher initializes the backend plugins.
    from isaaclab_newton.physics.newton_manager import NewtonManager, NewtonQueries  # noqa: PLC0415

    if options.dump_scene:
        import numpy as np  # noqa: PLC0415
        import ovstage  # noqa: PLC0415
        from isaaclab_ov.renderers.ovrtx_renderer import OVRTXRenderer  # noqa: PLC0415

        options.dump_scene.mkdir(parents=True, exist_ok=True)
        clone_jobs = {}
        original_clone = ovstage.Stage.clone

        def record_clone(stage, source, targets, ordinal):
            clone_jobs.setdefault(id(stage), []).append({"source": source, "targets": list(targets)})
            return original_clone(stage, source, targets, ordinal)

        ovstage.Stage.clone = record_clone
        original_init = OVRTXRenderer.__init__

        def dumping_init(renderer, cfg):
            cfg.temp_usd_dir = str(options.dump_scene)
            return original_init(renderer, cfg)

        OVRTXRenderer.__init__ = dumping_init
        original_camera_init = OVRTXRenderer._initialize_camera_render_data_from_spec_ovstage

        def record_camera_init(renderer, spec, render_data):
            result = original_camera_init(renderer, spec, render_data)
            np.save(options.dump_scene / "env_positions.npy", renderer._clone_plan.positions)
            recipe = {
                "clones": clone_jobs.get(id(renderer.backend.stage), []),
                "env_template": renderer._clone_plan.env_template,
                "num_envs": spec.num_instances,
                "camera_source": spec.camera_prim_paths[0],
                "render_product": render_data.render_product_path,
                "object_paths": list(renderer._sdp.backend.transform_paths),
                "geometry_paths": list(renderer._geometry_paths),
            }
            (options.dump_scene / "recipe.json").write_text(json.dumps(recipe), encoding="utf-8")
            print(f"SCENE_RECIPE_SAVED {options.dump_scene}", flush=True)
            return result

        OVRTXRenderer._initialize_camera_render_data_from_spec_ovstage = record_camera_init

    if options.warmup_capture == "early":
        reserve_graph_memory(benchmark_device())
    elif options.warmup_capture == "solver":
        from isaaclab_newton.physics import newton_manager  # noqa: PLC0415

        original_invalidate = NewtonManager._invalidate_graph.__func__

        def invalidate_with_warmup(cls):
            # Where Newton captured before #8058.
            original_invalidate(cls)
            if NewtonManager._graph_capture_pending:
                reserve_graph_memory(newton_manager.PhysicsManager._device)

        NewtonManager._invalidate_graph = classmethod(invalidate_with_warmup)

    original_capture = NewtonQueries.capture_graph

    def logged_capture(device, target, *, relaxed=False):
        import warp as wp  # noqa: PLC0415

        if options.allocation_only:

            def allocate_only():
                temporary = wp.empty(8192, dtype=wp.int32, device=device)
                retained_allocations.append(temporary)
                temporary.fill_(42)

            target = allocate_only
        elif options.mujoco_only:
            target = NewtonManager._solver._mujoco_warp_step
        elif options.kinematics_only:
            import mujoco_warp as mjw  # noqa: PLC0415

            def kinematics_only():
                solver = NewtonManager._solver
                mjw.kinematics(solver.mjw_model, solver.mjw_data)
                temporary = wp.empty(8192, dtype=wp.int32, device=device)
                retained_allocations.append(temporary)
                temporary.fill_(42)

            target = kinematics_only
        elif options.kernel_only or options.raw_allocation:
            import prefix_kernels  # noqa: PLC0415

            wp.load_module(module=prefix_kernels, device=device)
            if options.raw_allocation:
                driver = ctypes.WinDLL("nvcuda.dll")
                driver.cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
                driver.cuMemAllocAsync.restype = ctypes.c_int
                driver.cuMemFreeAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
                driver.cuMemFreeAsync.restype = ctypes.c_int

            def kernel_only():
                values = NewtonManager._solver.mjw_data.qpos
                wp.launch(prefix_kernels.prefix, dim=values.shape, inputs=[values], device=device)
                if options.raw_allocation:
                    address = ctypes.c_uint64()
                    with wp.ScopedDevice(device):
                        print("RAW_CUDA_ALLOC_ENTER", flush=True)
                        status = driver.cuMemAllocAsync(
                            ctypes.byref(address), 8192 * 4, wp.get_stream(device).cuda_stream
                        )
                        print(f"RAW_CUDA_ALLOC_RETURN status={status}", flush=True)
                        if status:
                            raise RuntimeError(f"CUDA allocation failed with status {status}")
                        status = driver.cuMemFreeAsync(address.value, wp.get_stream(device).cuda_stream)
                        if status:
                            raise RuntimeError(f"CUDA free failed with status {status}")
                    return
                temporary = wp.empty(8192, dtype=wp.int32, device=device)
                retained_allocations.append(temporary)
                temporary.fill_(42)

            target = kernel_only
        started = time.monotonic()
        print(
            f"GRAPH_CAPTURE_ENTER device={device} relaxed={relaxed} perf_counter={time.perf_counter():.3f}",
            flush=True,
        )
        if options.nonblocking or options.new_blocking:
            import torch  # noqa: PLC0415

            stream = (
                wp.Stream(device=device)
                if options.new_blocking
                else wp.stream_from_torch(torch.cuda.Stream(device=device))
            )
            with wp.ScopedStream(stream):
                graph = original_capture(device, target, relaxed=relaxed)
        else:
            graph = original_capture(device, target, relaxed=relaxed)
        print(f"GRAPH_CAPTURE_RETURN seconds={time.monotonic() - started:.3f}", flush=True)
        return graph

    NewtonQueries.capture_graph = staticmethod(logged_capture)
    return original_make(*args, **kwargs)


gym.make = logged_make
runpy.run_module("isaaclab.benchmark.entrypoints.runtime", run_name="__main__")
faulthandler.cancel_dump_traceback_later()
