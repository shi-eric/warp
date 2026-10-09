"""Check whether the public OVRTX and OVStage renderer initializes here."""

import faulthandler
import importlib.metadata
import os
from pathlib import Path

import ovrtx
import ovstage

faulthandler.enable()
os.environ["OMNICLIENT_HUB_MODE"] = "disabled"
faulthandler.dump_traceback_later(60, repeat=True)
print("OVRTX", importlib.metadata.version("ovrtx"), "OVStage", importlib.metadata.version("ovstage"), flush=True)
config = ovrtx.RendererConfig(
    read_gpu_transforms=True,
    active_cuda_gpus="0",
    log_level="warning",
    log_file_path=str(Path(__file__).parent / "renderer-native.log"),
)
print("Creating renderer", flush=True)
print("Creating stage", flush=True)
stage = ovstage.Stage("warp-8342")
print("Stage created", flush=True)
renderer = ovrtx.Renderer(config)
print("Renderer created", flush=True)
ovstage.population.open_usd_from_string(
    stage,
    """#usda 1.0
(
    defaultPrim = "World"
    upAxis = "Z"
)
def Xform "World"
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
def Scope "Render"
{
    def RenderProduct "Product" (
        prepend apiSchemas = ["OmniRtxSettingsCommonAdvancedAPI_1"]
    )
    {
        rel camera = </World/Camera>
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
""",
    ordinal=1,
)
print("Stage populated", flush=True)
stage.advance_write_floor(ordinal=1).wait()
renderer.attach_ovstage(stage)
print("Stage attached", flush=True)
output = renderer.step({"/Render/Product"}, 1.0 / 60.0, ordinal=1)
print("Rendered", repr(output), flush=True)
renderer.destroy()
stage.destroy()
faulthandler.cancel_dump_traceback_later()
print("Passed", flush=True)
