# /// script
# dependencies = ["numpy==2.5.1", "ovrtx==0.5.0.377615", "warp-lang==1.17.0"]
# ///
"""Reduce Isaac Lab #8342 to a generated OVRTX scene and one captured allocation.

Confirmed on Windows with an A40 / driver 595.97: the first captured operation,
``wp.empty(8192)``, stalls in ``cuMemAllocAsync``. It retains GPU transform
publication and 8,192 camera partitions with 32 spheres each. No image mapping
or captured kernels are needed. The failure is intermittent.
"""

import faulthandler
from pathlib import Path

import numpy as np
import ovrtx

import warp as wp

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache")
wp.init()
wp.set_device("cuda:0")

spheres = "\n".join(f'def Sphere "Sphere_{i}"\n{{\n    double radius = 1\n}}' for i in range(32))
environments = "\n".join(
    f"""def Xform "env_{i}" {{
        token primvars:omni:scenePartition = "env_{i}"
        {spheres}
        def Camera "Camera" {{
            token omni:scenePartition = "env_{i}"
            double3 xformOp:translate = (0, 0, 5)
            uniform token[] xformOpOrder = ["xformOp:translate"]
        }}
    }}"""
    for i in range(8192)
)
cameras = [f"/World/envs/env_{i}/Camera" for i in range(8192)]
camera_relationship = ", ".join(f"<{path}>" for path in cameras)
scene = f"""#usda 1.0
def Xform "World"
{{
    def Xform "envs"
    {{
        {environments}
    }}
}}
def Scope "Render" {{
    def RenderProduct "Product" (prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]) {{
        rel camera = [{camera_relationship}]
        uint[] deviceIds = [0]
        token omni:rtx:rendermode = "Minimal"
        int omni:rtx:minimal:mode = 3
        token omni:rtx:background:source:type = "color"
        color3f omni:rtx:background:source:color = (0.1, 0.1, 0.1)
        float omni:rtx:rt:ambientLight:intensity = 1.0
        token[] omni:rtx:waitForEvents = ["AllLoadingFinished", "OnlyOnFirstRequest"]
        rel orderedVars = [</Render/Color>]
        uniform int2 resolution = (5824, 5824)
    }}
    def RenderVar "Color"
    {{
        uniform string sourceName = "LdrColor"
    }}
}}
"""
renderer = ovrtx.Renderer(ovrtx.RendererConfig(read_gpu_transforms=True, active_cuda_gpus="0", log_level="warning"))
print("Opening generated scene", flush=True)
renderer.open_usd_from_string(scene)
objects = [f"/World/envs/env_{i}/Sphere_{j}" for i in range(8192) for j in range(32)]
transforms = []
for paths, z in ((objects, 0.0), (cameras, 5.0)):
    host = np.tile(np.eye(4, dtype=np.float64), (len(paths), 1, 1))
    host[:, 3, 2] = z
    matrices = wp.array(host, dtype=wp.mat44d)
    transforms.append(matrices)
    renderer.write_attribute(
        paths,
        "omni:xform",
        matrices,
        semantic=ovrtx.Semantic.XFORM_MAT4x4,
        data_access=ovrtx.DataAccess.ASYNC,
        cuda_stream=wp.get_stream().cuda_stream,
    )
print("Rendering", flush=True)
products = renderer.step({"/Render/Product"}, delta_time=1.0 / 60)
print("Before captured allocation", flush=True)
with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
    allocation = wp.empty(8192, dtype=wp.int32)
print("After captured allocation", flush=True)
wp.capture_launch(capture.graph)
allocation.fill_(42)
np.testing.assert_array_equal(allocation.numpy(), np.full(8192, 42, dtype=np.int32))
del products
renderer.destroy()
faulthandler.cancel_dump_traceback_later()
print("Passed", flush=True)
