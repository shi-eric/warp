# /// script
# dependencies = ["cuda-bindings==12.9.5"]
# ///
"""Reproduce the Isaac Lab #8342 allocation stall with only CUDA Python on Windows.

No OVRTX, Warp, NumPy, or Vulkan is needed. A captured 32 KiB ``cuMemAllocAsync``
stalls in the CUDA driver when both of the following hold:

1. UVM-Lite is active in the process. One managed allocation initializes it,
   as OVRTX does during renderer startup.
2. GPU virtual address space that CUDA did not reserve occupies the start of the
   next VA window that CUDA reserves, here the 96 GiB graph-allocation window.
   OVRTX's Vulkan allocations play this role. This script reserves the range
   directly with ``D3DKMTReserveGpuVirtualAddress``.

Each failed window placement reserves the range with UVM-Lite, fails to reserve
it in the GPU address space, releases it from UVM-Lite (about 2 seconds of
kernel CPU time on an A40 with driver 595.97), and retries 2 MiB higher. The
stall therefore lasts about one second per MiB of foreign VA. Several GiB, as
with the renderer, is effectively a hang. Without managed memory, the same
collision costs about a millisecond.

Controls: ``--no-managed`` skips the managed allocation, and ``--warmup`` captures
one allocation before the foreign VA exists so the graph window is already
reserved. ``--vulkan`` replaces the D3DKMT reservation with ordinary Vulkan
device-memory allocations and requires ``uv run --with vulkan==1.3.275.1``.
"""

import argparse
import ctypes
import sys
import threading
import time
from ctypes import wintypes

from cuda.bindings import driver as cu

parser = argparse.ArgumentParser()
parser.add_argument(
    "--obstacle-mib", type=int, default=16, help="Foreign GPU VA to reserve; about 1 s of stall per MiB"
)
parser.add_argument("--no-managed", action="store_true", help="Control: do not initialize UVM-Lite")
parser.add_argument(
    "--warmup", action="store_true", help="Control: capture one allocation before the foreign VA exists"
)
parser.add_argument("--vulkan", action="store_true", help="Allocate Vulkan device memory instead of reserving VA")
args = parser.parse_args()
if sys.platform != "win32":
    parser.error("This reproduction requires Windows (WDDM)")
if args.obstacle_mib < 1:
    parser.error("--obstacle-mib must be positive")


def check(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:] or None


class LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class D3DKMT_OPENADAPTERFROMLUID(ctypes.Structure):
    _fields_ = [("AdapterLuid", LUID), ("hAdapter", ctypes.c_uint32)]


class D3DDDI_RESERVEGPUVIRTUALADDRESS(ctypes.Structure):
    _fields_ = [
        ("hAdapter", ctypes.c_uint32),
        ("BaseAddress", ctypes.c_uint64),
        ("MinimumAddress", ctypes.c_uint64),
        ("MaximumAddress", ctypes.c_uint64),
        ("Size", ctypes.c_uint64),
        ("ReservationType", ctypes.c_uint32),
        ("DriverProtection", ctypes.c_uint64),
        ("VirtualAddress", ctypes.c_uint64),
        ("PagingFenceValue", ctypes.c_uint64),
    ]


class D3DKMT_FREEGPUVIRTUALADDRESS(ctypes.Structure):
    _fields_ = [("hAdapter", ctypes.c_uint32), ("BaseAddress", ctypes.c_uint64), ("Size", ctypes.c_uint64)]


# Observed in the driver's UVM-Lite IOCTLs: CUDA places each 96 GiB window first-fit from this base.
CUDA_WINDOW_SIZE = 96 << 30
CUDA_WINDOW_MINIMUM = 0x20_0000_0000


def reserve_foreign_va(device, size):
    """Reserve GPU VA outside CUDA at the start of the next 96 GiB window CUDA will choose."""
    gdi = ctypes.WinDLL("gdi32.dll")
    gdi.D3DKMTOpenAdapterFromLuid.argtypes = [ctypes.POINTER(D3DKMT_OPENADAPTERFROMLUID)]
    gdi.D3DKMTReserveGpuVirtualAddress.argtypes = [ctypes.POINTER(D3DDDI_RESERVEGPUVIRTUALADDRESS)]
    gdi.D3DKMTFreeGpuVirtualAddress.argtypes = [ctypes.POINTER(D3DKMT_FREEGPUVIRTUALADDRESS)]
    luid, _ = check(cu.cuDeviceGetLuid(device))
    luid = luid if isinstance(luid, (bytes, bytearray)) else bytes(ord(c) for c in luid)
    adapter = D3DKMT_OPENADAPTERFROMLUID()
    ctypes.memmove(ctypes.byref(adapter.AdapterLuid), bytes(luid).ljust(8, b"\0"), 8)
    if status := gdi.D3DKMTOpenAdapterFromLuid(ctypes.byref(adapter)):
        raise RuntimeError(f"D3DKMTOpenAdapterFromLuid failed: {status & 0xFFFFFFFF:#x}")

    def reserve(**fields):
        reservation = D3DDDI_RESERVEGPUVIRTUALADDRESS(hAdapter=adapter.hAdapter, **fields)
        if status := gdi.D3DKMTReserveGpuVirtualAddress(ctypes.byref(reservation)):
            raise RuntimeError(f"D3DKMTReserveGpuVirtualAddress failed: {status & 0xFFFFFFFF:#x}")
        return reservation.VirtualAddress

    # Find the first free window as CUDA will, release it, then occupy its start.
    base = reserve(MinimumAddress=CUDA_WINDOW_MINIMUM, Size=CUDA_WINDOW_SIZE)
    probe = D3DKMT_FREEGPUVIRTUALADDRESS(hAdapter=adapter.hAdapter, BaseAddress=base, Size=CUDA_WINDOW_SIZE)
    if status := gdi.D3DKMTFreeGpuVirtualAddress(ctypes.byref(probe)):
        raise RuntimeError(f"D3DKMTFreeGpuVirtualAddress failed: {status & 0xFFFFFFFF:#x}")
    return reserve(BaseAddress=base, Size=size)


def allocate_vulkan(size):
    """Allocate ordinary Vulkan device memory in 2 MiB blocks, as a renderer would."""
    import vulkan as vk  # noqa: PLC0415 (Optional dependency.)

    instance = vk.vkCreateInstance(
        vk.VkInstanceCreateInfo(pApplicationInfo=vk.VkApplicationInfo(apiVersion=vk.VK_MAKE_VERSION(1, 2, 0))), None
    )
    physical = next(
        p for p in vk.vkEnumeratePhysicalDevices(instance) if vk.vkGetPhysicalDeviceProperties(p).vendorID == 0x10DE
    )
    queue = vk.VkDeviceQueueCreateInfo(queueFamilyIndex=0, pQueuePriorities=[1.0])
    device = vk.vkCreateDevice(physical, vk.VkDeviceCreateInfo(pQueueCreateInfos=[queue]), None)
    properties = vk.vkGetPhysicalDeviceMemoryProperties(physical)
    memory_type = next(
        i
        for i in range(properties.memoryTypeCount)
        if properties.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
    )
    block = 2 << 20
    info = vk.VkMemoryAllocateInfo(allocationSize=block, memoryTypeIndex=memory_type)
    return instance, device, [vk.vkAllocateMemory(device, info, None) for _ in range(max(1, size // block))]


def captured_allocation(stream):
    check(cu.cuStreamBeginCapture(stream, cu.CUstreamCaptureMode.CU_STREAM_CAPTURE_MODE_THREAD_LOCAL))
    start = time.perf_counter()
    pointer = check(cu.cuMemAllocAsync(32768, stream))
    elapsed = time.perf_counter() - start
    check(cu.cuMemFreeAsync(pointer, stream))
    check(cu.cuGraphDestroy(check(cu.cuStreamEndCapture(stream))))
    return int(pointer), elapsed


def report_progress(done):
    started = time.perf_counter()
    while not done.wait(10):
        print(f"  ...captured cuMemAllocAsync still running after {time.perf_counter() - started:.0f} s", flush=True)


check(cu.cuInit(0))
device = check(cu.cuDeviceGet(0))
check(cu.cuCtxSetCurrent(check(cu.cuDevicePrimaryCtxRetain(device))))
print("Device:", check(cu.cuDeviceGetName(64, device)).split(b"\0")[0].decode(), flush=True)
stream = check(cu.cuStreamCreate(0))
# Like Warp, use the default stream-ordered pool before any capture.
pointer = int(check(cu.cuMemAllocAsync(1 << 20, stream)))
check(cu.cuStreamSynchronize(stream))
print(f"Pool allocation at {pointer:#x}", flush=True)
if args.warmup:
    pointer, elapsed = captured_allocation(stream)
    print(f"Warm-up captured allocation at {pointer:#x} took {elapsed:.3f} s", flush=True)
if not args.no_managed:
    pointer = int(check(cu.cuMemAllocManaged(0x4820, cu.CUmemAttach_flags.CU_MEM_ATTACH_GLOBAL.value)))
    print(f"Managed allocation at {pointer:#x} (UVM-Lite initialized)", flush=True)
size = args.obstacle_mib << 20
if args.vulkan:
    vulkan_objects = allocate_vulkan(size)
    print(f"Allocated {len(vulkan_objects[2])} x 2 MiB of Vulkan device memory", flush=True)
else:
    base = reserve_foreign_va(device, size)
    print(f"Reserved foreign GPU VA [{base:#x}, {base + size:#x}) with D3DKMT", flush=True)

print("Capturing a 32 KiB cuMemAllocAsync", flush=True)
done = threading.Event()
threading.Thread(target=report_progress, args=(done,), daemon=True).start()
pointer, elapsed = captured_allocation(stream)
done.set()
print(f"Captured allocation at {pointer:#x} took {elapsed:.3f} s", flush=True)
if elapsed > 1.0:
    print("REPRODUCED: the driver retried the graph window placement past the foreign VA", flush=True)
elif args.no_managed or args.warmup or args.vulkan:
    print("No stall", flush=True)
else:
    # The driver occasionally places a window away from the first free range.
    print(f"INCONCLUSIVE: the graph window avoided the foreign VA at {base:#x}; rerun", flush=True)
