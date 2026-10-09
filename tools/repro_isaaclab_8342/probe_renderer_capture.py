"""Reproduce Isaac Lab #8342 with procedural spheres and public OVRTX packages.

The 8,192-camera, 32-sphere GPU-transform configuration has reproduced a
captured allocation hang with only Warp, NumPy, OVRTX, and OVStage installed.
Torch is an optional diagnostic control. Other configurations are reductions.
"""

import argparse
import faulthandler
import importlib.metadata
import math
from pathlib import Path

import numpy as np
import ovrtx
import ovstage

import warp as wp

parser = argparse.ArgumentParser()
parser.add_argument("--rounds", type=int, default=100)
parser.add_argument("--mujoco", action="store_true", help="Also capture MuJoCo-Warp steps for 8,192 contact worlds")
parser.add_argument("--torch", action="store_true", help="Initialize Torch's CUDA runtime before rendering")
parser.add_argument("--gpu-map", action="store_true", help="Consume and release CUDA image mappings before capture")
parser.add_argument(
    "--release-products", action="store_true", help="Drop all native frame/product references before capture"
)
parser.add_argument("--cameras", type=int, default=1, help="Clone the simple scene into this many camera partitions")
parser.add_argument("--hierarchy", choices=("CPU_INCREMENTAL", "GPU_INCREMENTAL", "GPU_GLOBAL"))
parser.add_argument("--shading-mode", type=int, choices=(0, 1, 2, 3), default=2)
parser.add_argument("--objects", type=int, default=1, help="Procedural spheres per environment")
parser.add_argument("--gpu-transforms", action="store_true")
parser.add_argument(
    "--kernel-first", action="store_true", help="Launch a small Warp kernel before the captured allocation"
)
args = parser.parse_args()
if args.release_products and not args.gpu_map:
    parser.error("--release-products requires --gpu-map")
faulthandler.enable()
faulthandler.dump_traceback_later(120, repeat=True)
wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache")
wp.init()
wp.set_device("cuda:0")
if args.torch:
    import torch

    torch_buffer = torch.zeros(8192, dtype=torch.int32, device="cuda:0")
    print("Torch:", torch.__version__, "CUDA:", torch.version.cuda, flush=True)
print("Versions:", {name: importlib.metadata.version(name) for name in ("warp-lang", "ovrtx", "ovstage")}, flush=True)
if args.cameras < 1:
    raise ValueError("Camera count must be positive")
if args.objects < 1:
    raise ValueError("Object count must be positive")
columns = math.ceil(math.sqrt(args.cameras))
rows = math.ceil(args.cameras / columns)
image_shape = (rows * 64, columns * 64)
if args.kernel_first:
    import prefix_kernels

    values = wp.zeros((8192, 32), dtype=wp.float32)
    wp.load_module(module=prefix_kernels)

stage_config = (
    ovstage.StageConfig(
        runtime_default_hierarchy_computation_model=getattr(ovstage.HierarchyComputationModel, args.hierarchy)
    )
    if args.hierarchy
    else None
)
print("Hierarchy:", args.hierarchy or "SDK default (CPU_INCREMENTAL)", flush=True)
stage = ovstage.Stage("warp-8342-render-capture", config=stage_config)
paths = ovstage.PathDictionary(stage)
renderer = ovrtx.Renderer(ovrtx.RendererConfig(read_gpu_transforms=True, active_cuda_gpus="0", log_level="warning"))
ovstage.population.open_usd_from_string(
    stage,
    """#usda 1.0
def Xform "World"
{
    def Xform "envs"
    {
    def Xform "env_0"
    {
    def Sphere "Sphere"
    {
        double radius = 1
    }
    def Camera "Camera"
    {
        double3 xformOp:translate = (0, 0, 5)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
    }
    }
}
def Scope "Render"
{
    def RenderProduct "Product" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    )
    {
        rel camera = </World/envs/env_0/Camera>
        uint[] deviceIds = [0]
        token omni:rtx:rendermode = "Minimal"
        int omni:rtx:minimal:mode = 2
        token omni:rtx:background:source:type = "color"
        color3f omni:rtx:background:source:color = (0.1, 0.1, 0.1)
        float omni:rtx:rt:ambientLight:intensity = 1.0
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</Render/Color>]
        uniform int2 resolution = (64, 64)
    }
    def RenderVar "Color"
    {
        uniform string sourceName = "LdrColor"
    }
}
""".replace("resolution = (64, 64)", f"resolution = ({image_shape[1]}, {image_shape[0]})")
    .replace("minimal:mode = 2", f"minimal:mode = {args.shading_mode}")
    .replace(
        '    def Sphere "Sphere"\n    {\n        double radius = 1\n    }',
        "\n".join(
            f'    def Sphere "Sphere_{i}"\n    {{\n        double radius = 1\n    }}' for i in range(args.objects)
        ),
    ),
    ordinal=1,
)
camera_paths = [f"/World/envs/env_{i}/Camera" for i in range(args.cameras)]
if args.cameras > 1:
    stage.clone("/World/envs/env_0", [f"/World/envs/env_{i}" for i in range(1, args.cameras)], ordinal=1)
    tokens = np.array([paths.intern_token(f"env_{i}") for i in range(args.cameras)], dtype=np.uint64)
    for prim_paths, attribute in (
        ([f"/World/envs/env_{i}" for i in range(args.cameras)], "primvars:omni:scenePartition"),
        (camera_paths, "omni:scenePartition"),
    ):
        path_list = paths.create_path_list_from_strings(prim_paths)
        with stage.query_from_path_list(path_list) as query:
            stage.write_attribute(
                query, attribute, ordinal=1, tensors=tokens, is_array=False, semantic=ovstage.AttributeSemantic.TOKEN_ID
            ).wait()
        paths.destroy_path_list(path_list)
    path_list = paths.create_path_list_from_strings(["/Render/Product"])
    with stage.query_from_path_list(path_list) as query:
        stage.write_attribute(
            query,
            "camera",
            ordinal=1,
            tensors=np.array([paths.intern_path(path) for path in camera_paths], dtype=np.uint64),
            is_array=True,
            semantic=ovstage.AttributeSemantic.RELATIONSHIP_PATH_ID,
        ).wait()
    paths.destroy_path_list(path_list)
stage.advance_write_floor(1, ovstage.Scope.ALL).wait()
print("Attaching populated stage", flush=True)
renderer.attach_ovstage(stage)
print("Stage attached", flush=True)
object_paths = [f"/World/envs/env_{env}/Sphere_{obj}" for env in range(args.cameras) for obj in range(args.objects)]
matrix_dtype = ovstage.DLDataType(code=ovstage.DLDataTypeCode.kDLFloat, bits=64, lanes=16)


def publish_xforms(prim_paths, *, is_camera, ordinal):
    host = np.tile(np.eye(4, dtype=np.float64), (len(prim_paths), 1, 1))
    if is_camera:
        host[:, 3, 2] = 5.0
    matrices = wp.array(host, dtype=wp.mat44d)
    path_list = paths.create_path_list_from_strings(prim_paths)
    with stage.query_from_path_list(path_list) as query:
        stage.write_attribute(
            query,
            "omni:xform",
            ordinal=ordinal,
            tensors=ovstage.make_dltensor(matrices, dtype=matrix_dtype),
            is_array=False,
            semantic=ovstage.AttributeSemantic.MATRIX,
            cuda_stream=wp.get_stream().cuda_stream or 1,
        ).wait()
    paths.destroy_path_list(path_list)
    return matrices


if args.mujoco:
    import mujoco
    import mujoco_warp as mjw

    model_host = mujoco.MjModel.from_xml_string("""
        <mujoco><option timestep="0.002" solver="Newton" iterations="10"/>
          <worldbody><geom type="plane" size="10 10 0.1"/>
            <body pos="0 0 0.1"><freejoint/>
              <geom type="sphere" size="0.1" mass="1"/>
            </body>
          </worldbody>
        </mujoco>""")
    model = mjw.put_model(model_host)
    data = mjw.make_data(model_host, nworld=8192, nconmax=8, njmax=32)
    initial_time = data.time.numpy().copy()

for iteration in range(args.rounds):
    ordinal = iteration + 2 if args.gpu_transforms else 1
    xforms = []
    if args.gpu_transforms:
        xforms.append(publish_xforms(object_paths, is_camera=False, ordinal=ordinal))
        xforms.append(publish_xforms(camera_paths, is_camera=True, ordinal=ordinal))
        stage.advance_write_floor(ordinal).wait()
    print(f"Round {iteration + 1}: renderer step", flush=True)
    products = renderer.step({"/Render/Product"}, delta_time=1.0 / 60, ordinal=ordinal)
    images = []
    if args.gpu_map:
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
    if args.release_products:
        del frame, product, products, render_var
    print(f"Round {iteration + 1}: before captured allocation", flush=True)
    with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
        if args.kernel_first:
            wp.launch(prefix_kernels.prefix, dim=values.shape, inputs=[values])
        temporary = wp.empty(8192, dtype=wp.int32)
        temporary.fill_(42)
        if args.mujoco:
            mjw.step(model, data)
    print(f"Round {iteration + 1}: after captured allocation", flush=True)
    wp.capture_launch(capture.graph)
    np.testing.assert_array_equal(temporary.numpy(), np.full(8192, 42, dtype=np.int32))
    if args.gpu_map:
        for copied in images:
            np.testing.assert_array_equal(copied.numpy().shape[:2], image_shape)
        del images, copied
    else:
        for product in products.values():
            for frame in product.frames:
                for render_var in frame.render_vars.values():
                    mapped = render_var.map(device=ovrtx.Device.CPU)
                    pixels = np.from_dlpack(mapped)
                    np.testing.assert_array_equal(pixels.shape[:2], image_shape)
                    del pixels
                    mapped.unmap()
                    del mapped
    del temporary, capture
    del xforms
    if not args.release_products:
        del frame, product, products, render_var

if args.mujoco:
    np.testing.assert_allclose(data.time.numpy(), initial_time + 0.002 * args.rounds, atol=1e-6)
    if not np.isfinite(data.qpos.numpy()).all():
        raise RuntimeError("MuJoCo positions became nonfinite")
if args.kernel_first:
    np.testing.assert_array_equal(values.numpy(), np.full((8192, 32), args.rounds, dtype=np.float32))

renderer.detach_ovstage()
paths.destroy()
stage.destroy()
renderer.destroy()
faulthandler.cancel_dump_traceback_later()
print(f"Passed: {args.rounds} render/capture/replay rounds", flush=True)
