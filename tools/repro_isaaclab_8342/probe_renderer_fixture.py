"""Replay exported renderer inputs without importing Isaac Lab or physics."""

import argparse
import ctypes
import faulthandler
import json
import math
import os
import re
from pathlib import Path

import numpy as np
import ovrtx
import ovstage
import prefix_kernels

import warp as wp

parser = argparse.ArgumentParser()
parser.add_argument("--fixture", type=Path, required=True)
parser.add_argument("--envs", type=int)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--hierarchy", choices=("CPU_INCREMENTAL", "GPU_INCREMENTAL"), default="GPU_INCREMENTAL")
parser.add_argument("--gpu-transforms", action="store_true")
parser.add_argument("--torch", action="store_true")
parser.add_argument("--raw-allocation", action="store_true")
parser.add_argument("--attach-only", action="store_true")
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--release-products", action="store_true")
parser.add_argument("--sync-before-capture", action="store_true")
parser.add_argument("--log-streams", action="store_true")
args = parser.parse_args()
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
os.environ["OVRTX_SKIP_USD_CHECK"] = "1"
recipe = json.loads((args.fixture / "recipe.json").read_text(encoding="utf-8"))
num_envs = args.envs or recipe["num_envs"]
if not 1 <= num_envs <= recipe["num_envs"]:
    raise ValueError("Environment count must be within the exported fixture")


def selected(path):
    match = re.search(r"/env_(\d+)(?:/|$)", path)
    return match is None or int(match.group(1)) < num_envs


wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache")
wp.init()
wp.set_device("cuda:0")
if args.torch:
    import torch

    torch_buffer = torch.zeros(8192, dtype=torch.int32, device="cuda:0")
    print("Torch:", torch.__version__, flush=True)
values = wp.zeros((8192, 32), dtype=wp.float32)
wp.load_module(module=prefix_kernels, device="cuda:0")
driver = ctypes.WinDLL("nvcuda.dll")
driver.cuMemAllocAsync.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t, ctypes.c_void_p]
driver.cuMemAllocAsync.restype = ctypes.c_int
driver.cuMemFreeAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p]
driver.cuMemFreeAsync.restype = ctypes.c_int
core = ctypes.WinDLL(str(Path(wp.__file__).parent / "bin" / "warp.dll"))
core.wp_cuda_context_get_stream.argtypes = [ctypes.c_void_p]
core.wp_cuda_context_get_stream.restype = ctypes.c_void_p
stage = ovstage.Stage(
    "warp-8342-exported-scene",
    config=ovstage.StageConfig(
        runtime_default_hierarchy_computation_model=getattr(ovstage.HierarchyComputationModel, args.hierarchy)
    ),
)
paths = ovstage.PathDictionary(stage)
renderer = ovrtx.Renderer(
    ovrtx.RendererConfig(
        read_gpu_transforms=True,
        active_cuda_gpus="0",
        log_level="warning",
        keep_system_alive=True,
        suppress_deprecation_warnings=True,
        texture_streaming_mode=ovrtx.TextureStreamingMode.SYNCHRONOUS,
    )
)
scene = (args.fixture / "ovrtx_renderer_stage.usda").read_text(encoding="utf-8")
columns = math.ceil(math.sqrt(num_envs))
rows = math.ceil(num_envs / columns)
image_shape = (rows * 64, columns * 64)
scene = re.sub(
    r"uniform int2 resolution = \(\d+, \d+\)", f"uniform int2 resolution = ({columns * 64}, {rows * 64})", scene
)
print("Populating exported USD", num_envs, "environments", flush=True)
ovstage.population.open_usd_from_string(stage, scene, ordinal=1, domains=ovstage.PopulationDomain.RENDERING)
for job in recipe["clones"]:
    targets = [path for path in job["targets"] if selected(path)]
    if targets:
        stage.clone(job["source"], targets, ordinal=1)


def write(
    prim_paths, attribute, tensors, *, ordinal=1, semantic=ovstage.AttributeSemantic.NONE, is_array=False, stream=None
):
    path_list = paths.create_path_list_from_strings(prim_paths)
    with stage.query_from_path_list(path_list) as query:
        stage.write_attribute(
            query, attribute, tensors=tensors, ordinal=ordinal, semantic=semantic, is_array=is_array, cuda_stream=stream
        ).wait()
    paths.destroy_path_list(path_list)


matrix_dtype = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)
env_paths = [recipe["env_template"].format(i) for i in range(num_envs)]
positions = np.load(args.fixture / "env_positions.npy")[:num_envs]
env_xforms = np.tile(np.eye(4, dtype=np.float64), (num_envs, 1, 1))
env_xforms[:, 3, :3] = positions
write(
    env_paths,
    "omni:xform",
    ovstage.make_dltensor(env_xforms.reshape(-1), dtype=matrix_dtype, shape=[num_envs]),
    semantic=ovstage.AttributeSemantic.MATRIX,
)
camera_paths = [recipe["camera_source"].replace("/env_0/", f"/env_{i}/") for i in range(num_envs)]
tokens = np.array([paths.intern_token(f"env_{i}") for i in range(num_envs)], dtype=np.uint64)
write(env_paths, "primvars:omni:scenePartition", tokens, semantic=ovstage.AttributeSemantic.TOKEN_ID)
write(camera_paths, "omni:scenePartition", tokens, semantic=ovstage.AttributeSemantic.TOKEN_ID)
write(
    [recipe["render_product"]],
    "camera",
    np.array([paths.intern_path(path) for path in camera_paths], dtype=np.uint64),
    semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
    is_array=True,
)
object_paths = [path for path in recipe["object_paths"] if selected(path)]
write(object_paths, "omni:resetXformStack", np.ones(len(object_paths), dtype=np.bool_))
write(camera_paths, "omni:resetXformStack", np.ones(num_envs, dtype=np.bool_))
stage.advance_write_floor(1).wait()
print("Attaching", len(object_paths), "body paths", flush=True)
renderer.attach_ovstage(stage)
print("Attached", flush=True)
for iteration in range(args.rounds):
    ordinal = iteration + 2
    retained = []
    if args.gpu_transforms:
        for prim_paths, is_camera in ((object_paths, False), (camera_paths, True)):
            host = np.tile(np.eye(4, dtype=np.float64), (len(prim_paths), 1, 1))
            if is_camera:
                host[:, 3, 2] = 5.0
            matrices = wp.array(host, dtype=wp.mat44d)
            retained.append(matrices)
            write(
                prim_paths,
                "omni:xform",
                ovstage.make_dltensor(matrices, dtype=matrix_dtype),
                ordinal=ordinal,
                semantic=ovstage.AttributeSemantic.MATRIX,
                stream=wp.get_stream().cuda_stream or 1,
            )
    stage.advance_write_floor(ordinal).wait()
    if not args.attach_only:
        print(f"Round {iteration + 1}: rendering", flush=True)
        products = renderer.step({recipe["render_product"]}, delta_time=1.0 / 60, ordinal=ordinal)
        if not args.no_map:
            for product in products.values():
                for frame in product.frames:
                    for render_var in frame.render_vars.values():
                        stream = wp.get_stream().cuda_stream or 1
                        mapping = render_var.map(device=ovrtx.Device.CUDA, sync_stream=stream)
                        view = wp.from_dlpack(mapping)
                        copied = wp.empty_like(view)
                        wp.copy(copied, view)
                        mapping.unmap(stream=stream)
                        del view, mapping
                        retained.append(copied)
        if args.release_products:
            if not args.no_map:
                del product, frame, render_var
            del products
    print(f"Round {iteration + 1}: capture enter", flush=True)
    if args.sync_before_capture:
        wp.synchronize_device()
    if args.log_streams:
        print(
            "CUDA streams:",
            {"Python": wp.get_stream().cuda_stream, "native": core.wp_cuda_context_get_stream(wp.get_device().context)},
            flush=True,
        )
    with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
        wp.launch(prefix_kernels.prefix, dim=values.shape, inputs=[values])
        if args.log_streams:
            print(
                "Before allocation streams:",
                {
                    "Python": wp.get_stream().cuda_stream,
                    "native": core.wp_cuda_context_get_stream(wp.get_device().context),
                },
                flush=True,
            )
        if args.raw_allocation:
            address = ctypes.c_uint64()
            with wp.ScopedDevice("cuda:0"):
                status = driver.cuMemAllocAsync(ctypes.byref(address), 8192 * 4, wp.get_stream().cuda_stream)
                if status:
                    raise RuntimeError(f"CUDA allocation failed with status {status}")
                status = driver.cuMemFreeAsync(address.value, wp.get_stream().cuda_stream)
                if status:
                    raise RuntimeError(f"CUDA free failed with status {status}")
        else:
            temporary = wp.empty(8192, dtype=wp.int32)
            temporary.fill_(42)
    print(f"Round {iteration + 1}: capture return", flush=True)
    wp.capture_launch(capture.graph)
    if not args.raw_allocation:
        np.testing.assert_array_equal(temporary.numpy(), np.full(8192, 42, dtype=np.int32))
    wp.synchronize_device()
    del retained, capture
np.testing.assert_array_equal(values.numpy(), np.full((8192, 32), args.rounds, dtype=np.float32))
renderer.detach_ovstage()
renderer.destroy()
paths.destroy()
stage.destroy()
faulthandler.cancel_dump_traceback_later()
print("Passed: independent exported renderer fixture", flush=True)
