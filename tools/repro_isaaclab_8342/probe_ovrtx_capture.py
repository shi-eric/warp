"""Reproduce the captured allocation hang with only Warp, NumPy, and OVRTX.

The default 8,192-camera, 32-sphere inline scene has reproduced a hang in
cuMemAllocAsync with no OVStage or Torch installed or loaded. Other settings
are reduction candidates, rather than confirmed reproductions.
"""

import argparse
import faulthandler
import importlib.metadata
import importlib.util
import math
from pathlib import Path

import numpy as np
import ovrtx
import prefix_kernels

import warp as wp

parser = argparse.ArgumentParser()
parser.add_argument("--cameras", type=int, default=8192)
parser.add_argument("--objects", type=int, default=32)
parser.add_argument("--rounds", type=int, default=3)
parser.add_argument("--cpu-transforms", action="store_true")
parser.add_argument(
    "--no-transforms", action="store_true", help="Render the authored static scene without publishing transforms"
)
parser.add_argument("--no-map", action="store_true")
parser.add_argument("--retain-products", action="store_true")
parser.add_argument("--warmup-render", action="store_true")
parser.add_argument("--no-prefix", action="store_true", help="Capture only allocation and fill")
parser.add_argument(
    "--inline-environments",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Author all partitions directly in USD; legacy cloning is a diagnostic control",
)
args = parser.parse_args()
if min(args.cameras, args.objects, args.rounds) < 1:
    parser.error("Camera, object, and round counts must be positive")
for package in ("torch", "ovstage"):
    if importlib.util.find_spec(package) is not None:
        raise RuntimeError(f"Run this probe in an environment without {package} installed")
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache")
wp.init()
wp.set_device("cuda:0")
print("Versions:", {name: importlib.metadata.version(name) for name in ("warp-lang", "ovrtx", "numpy")}, flush=True)
print("Torch and OVStage unavailable", flush=True)
values = wp.zeros((8192, 32), dtype=wp.float32)
wp.load_module(module=prefix_kernels)
columns = math.ceil(math.sqrt(args.cameras))
rows = math.ceil(args.cameras / columns)
image_shape = (rows * 64, columns * 64)
spheres = "\n".join(f'def Sphere "Sphere_{i}"\n{{\n    double radius = 1\n}}' for i in range(args.objects))
scene = f"""#usda 1.0
def Xform "World"
{{
    def Xform "envs"
    {{
        def Xform "env_0"
        {{
            {spheres}
            def Camera "Camera"
            {{
                double3 xformOp:translate = (0, 0, 5)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }}
        }}
    }}
}}
def Scope "Render"
{{
    def RenderProduct "Product" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    )
    {{
        rel camera = </World/envs/env_0/Camera>
        uint[] deviceIds = [0]
        token omni:rtx:rendermode = "Minimal"
        int omni:rtx:minimal:mode = 3
        token omni:rtx:background:source:type = "color"
        color3f omni:rtx:background:source:color = (0.1, 0.1, 0.1)
        float omni:rtx:rt:ambientLight:intensity = 1.0
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</Render/Color>]
        uniform int2 resolution = ({image_shape[1]}, {image_shape[0]})
    }}
    def RenderVar "Color"
    {{
        uniform string sourceName = "LdrColor"
    }}
}}
"""
if args.inline_environments:
    environments = []
    for i in range(args.cameras):
        environments.append(f"""
        def Xform "env_{i}"
        {{
            token primvars:omni:scenePartition = "env_{i}"
            {spheres}
            def Camera "Camera"
            {{
                token omni:scenePartition = "env_{i}"
                double3 xformOp:translate = (0, 0, 5)
                uniform token[] xformOpOrder = ["xformOp:translate"]
            }}
        }}""")
    render_scope = 'def Scope "Render"' + scene.split('def Scope "Render"', 1)[1]
    camera_relationship = ", ".join(f"</World/envs/env_{i}/Camera>" for i in range(args.cameras))
    render_scope = render_scope.replace(
        "rel camera = </World/envs/env_0/Camera>", f"rel camera = [{camera_relationship}]"
    )
    scene = (
        '#usda 1.0\ndef Xform "World"\n{\ndef Xform "envs"\n{\n' + "\n".join(environments) + "\n}\n}\n" + render_scope
    )
    del environments, render_scope
renderer = ovrtx.Renderer(ovrtx.RendererConfig(read_gpu_transforms=True, active_cuda_gpus="0", log_level="warning"))
print("Opening inline USD through OVRTX", flush=True)
renderer.open_usd_from_string(scene)
print("USD opened", flush=True)
environment_paths = [f"/World/envs/env_{i}" for i in range(args.cameras)]
camera_paths = [f"{path}/Camera" for path in environment_paths]
object_paths = [f"{path}/Sphere_{obj}" for path in environment_paths for obj in range(args.objects)]
if args.cameras > 1 and not args.inline_environments:
    renderer.clone_usd(environment_paths[0], environment_paths[1:])
    tokens = [f"env_{i}" for i in range(args.cameras)]
    renderer.write_attribute(environment_paths, "primvars:omni:scenePartition", tokens)
    renderer.write_attribute(camera_paths, "omni:scenePartition", tokens)
if not args.inline_environments:
    renderer.write_array_attribute(["/Render/Product"], "camera", [camera_paths], semantic=ovrtx.Semantic.PATH_STRING)
print("Scene populated", flush=True)
if args.warmup_render:
    print("Initializing renderer with a CPU-transform frame", flush=True)
    warmup_products = renderer.step({"/Render/Product"}, delta_time=1.0 / 60)
    for product in warmup_products.values():
        for frame in product.frames:
            for render_var in frame.render_vars.values():
                mapped = render_var.map(device=ovrtx.Device.CPU)
                pixels = np.from_dlpack(mapped)
                np.testing.assert_array_equal(pixels.shape[:2], image_shape)
                del pixels
                mapped.unmap()
                del mapped
    del warmup_products, product, frame, render_var
    print("Renderer initialized", flush=True)


def publish_xforms(prim_paths, *, is_camera):
    host = np.tile(np.eye(4, dtype=np.float64), (len(prim_paths), 1, 1))
    if is_camera:
        host[:, 3, 2] = 5.0
    matrices = host if args.cpu_transforms else wp.array(host, dtype=wp.mat44d)
    renderer.write_attribute(
        prim_paths,
        "omni:xform",
        matrices,
        semantic=ovrtx.Semantic.XFORM_MAT4x4,
        data_access=ovrtx.DataAccess.SYNC if args.cpu_transforms else ovrtx.DataAccess.ASYNC,
        cuda_stream=None if args.cpu_transforms else wp.get_stream().cuda_stream or 1,
    )
    return matrices


for iteration in range(args.rounds):
    xforms = (
        []
        if args.no_transforms
        else [publish_xforms(object_paths, is_camera=False), publish_xforms(camera_paths, is_camera=True)]
    )
    print(f"Round {iteration + 1}: renderer step", flush=True)
    products = renderer.step({"/Render/Product"}, delta_time=1.0 / 60)
    images = []
    if not args.no_map:
        stream_handle = wp.get_stream().cuda_stream or 1
        for product in products.values():
            for frame in product.frames:
                for render_var in frame.render_vars.values():
                    mapped = render_var.map(device=ovrtx.Device.CUDA, sync_stream=stream_handle)
                    view = wp.from_dlpack(mapped)
                    copied = wp.empty_like(view)
                    wp.copy(copied, view)
                    mapped.unmap(stream=stream_handle)
                    del view, mapped
                    images.append(copied)
        if not args.retain_products:
            del frame, product, render_var, products
    print(f"Round {iteration + 1}: before captured allocation", flush=True)
    with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
        if not args.no_prefix:
            wp.launch(prefix_kernels.prefix, dim=values.shape, inputs=[values])
        temporary = wp.empty(8192, dtype=wp.int32)
        temporary.fill_(42)
    print(f"Round {iteration + 1}: after captured allocation", flush=True)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(temporary.numpy(), np.full(8192, 42, dtype=np.int32))
    for copied in images:
        np.testing.assert_array_equal(copied.numpy().shape[:2], image_shape)
    if images:
        del copied
    del images, temporary, capture, xforms
    if args.no_map or args.retain_products:
        del products
np.testing.assert_array_equal(
    values.numpy(), np.full((8192, 32), 0 if args.no_prefix else args.rounds, dtype=np.float32)
)
renderer.destroy()
faulthandler.cancel_dump_traceback_later()
print(f"Passed: {args.rounds} standalone OVRTX render/capture/replay rounds", flush=True)
