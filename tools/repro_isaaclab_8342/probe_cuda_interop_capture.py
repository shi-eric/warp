"""Probe a graphics/CUDA handoff without OVRTX, OVStage, or Torch.

This is a reduction candidate, not a confirmed reproduction of Isaac Lab #8342.
The Vulkan backends share an exportable buffer/image and binary semaphores with CUDA.
The CUDA backend provides an event/copy control without graphics resources.
"""

import argparse
import ctypes
import faulthandler
import importlib.metadata
import importlib.util
from pathlib import Path

import numpy as np
import prefix_kernels
from cuda.bindings import driver as cu

import warp as wp


def checked(result):
    if result[0] != cu.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA call failed: {result[0]}")
    return result[1] if len(result) == 2 else result[1:]


class CudaBuffer:
    def __init__(self, size, stream_flags):
        self.size = size
        self.stream = checked(cu.cuStreamCreate(stream_flags))
        self.event = checked(cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING))
        self.pointer = checked(cu.cuMemAlloc(size))

    def publish(self, warp_stream):
        checked(cu.cuMemsetD32Async(self.pointer, 42, self.size // 4, self.stream))
        checked(cu.cuEventRecord(self.event, self.stream))
        if warp_stream is not None:
            checked(cu.cuStreamWaitEvent(warp_stream, self.event, 0))

    def release(self, warp_stream):
        pass

    def copy_to_staging(self, pointer):
        checked(cu.cuMemcpyDtoDAsync(pointer, self.pointer, self.size, self.stream))

    def destroy(self):
        checked(cu.cuMemFree(self.pointer))
        checked(cu.cuEventDestroy(self.event))
        checked(cu.cuStreamDestroy(self.stream))


class VulkanBuffer:
    def __init__(self, size, stream_flags, *, image=False, image_side=None, producer_repeats=1, non_dedicated=False):
        import vulkan as vk  # noqa: PLC0415 (The CUDA control must not load Vulkan.)

        self.vk = vk
        self.size = size
        self.image = image
        self.width = image_side or 4096
        self.height = image_side or size // (self.width * 4)
        self.stream = checked(cu.cuStreamCreate(stream_flags))
        self.event = checked(cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING))
        self.instance = vk.vkCreateInstance(
            vk.VkInstanceCreateInfo(pApplicationInfo=vk.VkApplicationInfo(apiVersion=vk.VK_MAKE_VERSION(1, 2, 0))),
            None,
        )
        physical_devices = [
            physical
            for physical in vk.vkEnumeratePhysicalDevices(self.instance)
            if vk.vkGetPhysicalDeviceProperties(physical).vendorID == 0x10DE
        ]
        if len(physical_devices) != 1:
            raise RuntimeError("This diagnostic requires exactly one NVIDIA Vulkan device")
        physical = physical_devices[0]
        print("Vulkan device:", vk.vkGetPhysicalDeviceProperties(physical).deviceName, flush=True)
        self.family = next(
            i
            for i, properties in enumerate(vk.vkGetPhysicalDeviceQueueFamilyProperties(physical))
            if properties.queueFlags & vk.VK_QUEUE_TRANSFER_BIT
        )
        self.device = vk.vkCreateDevice(
            physical,
            vk.VkDeviceCreateInfo(
                pQueueCreateInfos=[vk.VkDeviceQueueCreateInfo(queueFamilyIndex=self.family, pQueuePriorities=[1.0])],
                ppEnabledExtensionNames=["VK_KHR_external_memory_win32", "VK_KHR_external_semaphore_win32"],
            ),
            None,
        )
        self.queue = vk.vkGetDeviceQueue(self.device, self.family, 0)
        if image:
            self.resource = vk.vkCreateImage(
                self.device,
                vk.VkImageCreateInfo(
                    pNext=vk.VkExternalMemoryImageCreateInfo(
                        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                    ),
                    imageType=vk.VK_IMAGE_TYPE_2D,
                    format=vk.VK_FORMAT_R8G8B8A8_UNORM,
                    extent=vk.VkExtent3D(width=self.width, height=self.height, depth=1),
                    mipLevels=1,
                    arrayLayers=1,
                    samples=vk.VK_SAMPLE_COUNT_1_BIT,
                    tiling=vk.VK_IMAGE_TILING_OPTIMAL,
                    usage=vk.VK_IMAGE_USAGE_TRANSFER_SRC_BIT
                    | vk.VK_IMAGE_USAGE_TRANSFER_DST_BIT
                    | vk.VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT,
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                    initialLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                ),
                None,
            )
            requirements = vk.vkGetImageMemoryRequirements(self.device, self.resource)
            dedicated = vk.VkMemoryDedicatedAllocateInfo(image=self.resource)
        else:
            self.resource = vk.vkCreateBuffer(
                self.device,
                vk.VkBufferCreateInfo(
                    pNext=vk.VkExternalMemoryBufferCreateInfo(
                        handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
                    ),
                    size=size,
                    usage=vk.VK_BUFFER_USAGE_TRANSFER_SRC_BIT | vk.VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    sharingMode=vk.VK_SHARING_MODE_EXCLUSIVE,
                ),
                None,
            )
            requirements = vk.vkGetBufferMemoryRequirements(self.device, self.resource)
            dedicated = None if non_dedicated else vk.VkMemoryDedicatedAllocateInfo(buffer=self.resource)
        memory_properties = vk.vkGetPhysicalDeviceMemoryProperties(physical)
        memory_type = next(
            i
            for i in range(memory_properties.memoryTypeCount)
            if requirements.memoryTypeBits & (1 << i)
            and memory_properties.memoryTypes[i].propertyFlags & vk.VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT
        )
        export = vk.VkExportMemoryAllocateInfo(
            handleTypes=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT,
            pNext=vk.VkExportMemoryWin32HandleInfoKHR(
                dwAccess=0x10000000,
                pNext=dedicated,
            ),
        )
        self.memory = vk.vkAllocateMemory(
            self.device,
            vk.VkMemoryAllocateInfo(pNext=export, allocationSize=requirements.size, memoryTypeIndex=memory_type),
            None,
        )
        if image:
            vk.vkBindImageMemory(self.device, self.resource, self.memory, 0)
        else:
            vk.vkBindBufferMemory(self.device, self.resource, self.memory, 0)
        get_memory_handle = vk.vkGetDeviceProcAddr(self.device, "vkGetMemoryWin32HandleKHR")
        handle = get_memory_handle(
            self.device,
            vk.VkMemoryGetWin32HandleInfoKHR(
                memory=self.memory, handleType=vk.VK_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32_BIT
            ),
        )
        descriptor = cu.CUDA_EXTERNAL_MEMORY_HANDLE_DESC()
        descriptor.type = cu.CUexternalMemoryHandleType.CU_EXTERNAL_MEMORY_HANDLE_TYPE_OPAQUE_WIN32
        descriptor.handle.win32.handle = int(vk.ffi.cast("uintptr_t", handle))
        descriptor.size = requirements.size
        descriptor.flags = 1 if dedicated is not None else 0
        self.external_memory = checked(cu.cuImportExternalMemory(descriptor))
        self.close_handle(descriptor.handle.win32.handle)
        if image:
            mapping = cu.CUDA_EXTERNAL_MEMORY_MIPMAPPED_ARRAY_DESC()
            mapping.arrayDesc.Width = self.width
            mapping.arrayDesc.Height = self.height
            mapping.arrayDesc.Format = cu.CUarray_format.CU_AD_FORMAT_UNSIGNED_INT8
            mapping.arrayDesc.NumChannels = 4
            mapping.arrayDesc.Flags = 10  # CUDA_ARRAY3D_SURFACE_LDST | CUDA_ARRAY3D_COLOR_ATTACHMENT.
            mapping.numLevels = 1
            self.mipmapped = checked(cu.cuExternalMemoryGetMappedMipmappedArray(self.external_memory, mapping))
            self.array = checked(cu.cuMipmappedArrayGetLevel(self.mipmapped, 0))
        else:
            mapping = cu.CUDA_EXTERNAL_MEMORY_BUFFER_DESC()
            mapping.size = size
            self.pointer = checked(cu.cuExternalMemoryGetMappedBuffer(self.external_memory, mapping))
        self.ready, self.cuda_ready = self.create_semaphore()
        self.done, self.cuda_done = self.create_semaphore()
        self.pool = vk.vkCreateCommandPool(self.device, vk.VkCommandPoolCreateInfo(queueFamilyIndex=self.family), None)
        self.fill, self.acquire = vk.vkAllocateCommandBuffers(
            self.device,
            vk.VkCommandBufferAllocateInfo(
                commandPool=self.pool, level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY, commandBufferCount=2
            ),
        )
        if image:
            initial = vk.vkAllocateCommandBuffers(
                self.device,
                vk.VkCommandBufferAllocateInfo(
                    commandPool=self.pool,
                    level=vk.VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                    commandBufferCount=1,
                ),
            )[0]
            vk.vkBeginCommandBuffer(initial, vk.VkCommandBufferBeginInfo())
            transition = vk.VkImageMemoryBarrier(
                srcAccessMask=0,
                dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                oldLayout=vk.VK_IMAGE_LAYOUT_UNDEFINED,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                srcQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                dstQueueFamilyIndex=vk.VK_QUEUE_FAMILY_IGNORED,
                image=self.resource,
                subresourceRange=self.image_range(),
            )
            vk.vkCmdPipelineBarrier(
                initial,
                vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                0,
                0,
                None,
                0,
                None,
                1,
                [transition],
            )
            vk.vkEndCommandBuffer(initial)
            vk.vkQueueSubmit(self.queue, 1, [vk.VkSubmitInfo(pCommandBuffers=[initial])], vk.VK_NULL_HANDLE)
            vk.vkQueueWaitIdle(self.queue)
        vk.vkBeginCommandBuffer(self.fill, vk.VkCommandBufferBeginInfo())
        for repeat in range(producer_repeats):
            if repeat:
                vk.vkCmdPipelineBarrier(
                    self.fill,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
                    0,
                    1,
                    [
                        vk.VkMemoryBarrier(
                            srcAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                            dstAccessMask=vk.VK_ACCESS_TRANSFER_WRITE_BIT,
                        )
                    ],
                    0,
                    None,
                    0,
                    None,
                )
            # Alternate writes, finishing with the value checked below.
            value = 42 if (producer_repeats - repeat) % 2 else 0
            if image:
                vk.vkCmdClearColorImage(
                    self.fill,
                    self.resource,
                    vk.VK_IMAGE_LAYOUT_GENERAL,
                    vk.VkClearColorValue(float32=[value / 255.0] * 4),
                    1,
                    [self.image_range()],
                )
            else:
                vk.vkCmdFillBuffer(self.fill, self.resource, 0, size, value)
        self.barrier(self.fill, release=True)
        vk.vkEndCommandBuffer(self.fill)
        vk.vkBeginCommandBuffer(self.acquire, vk.VkCommandBufferBeginInfo())
        self.barrier(self.acquire, release=False)
        vk.vkEndCommandBuffer(self.acquire)

    @staticmethod
    def close_handle(handle):
        close = ctypes.WinDLL("kernel32").CloseHandle
        close.argtypes = [ctypes.c_void_p]
        close.restype = ctypes.c_int
        if not close(handle):
            raise RuntimeError("CloseHandle failed for a Vulkan export")

    def create_semaphore(self):
        vk = self.vk
        semaphore = vk.vkCreateSemaphore(
            self.device,
            vk.VkSemaphoreCreateInfo(
                pNext=vk.VkExportSemaphoreCreateInfo(
                    handleTypes=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT,
                    pNext=vk.VkExportSemaphoreWin32HandleInfoKHR(dwAccess=0x10000000),
                )
            ),
            None,
        )
        get_handle = vk.vkGetDeviceProcAddr(self.device, "vkGetSemaphoreWin32HandleKHR")
        handle = get_handle(
            self.device,
            vk.VkSemaphoreGetWin32HandleInfoKHR(
                semaphore=semaphore, handleType=vk.VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32_BIT
            ),
        )
        descriptor = cu.CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC()
        descriptor.type = cu.CUexternalSemaphoreHandleType.CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_WIN32
        descriptor.handle.win32.handle = int(vk.ffi.cast("uintptr_t", handle))
        imported = checked(cu.cuImportExternalSemaphore(descriptor))
        self.close_handle(descriptor.handle.win32.handle)
        return semaphore, imported

    def barrier(self, command, *, release):
        vk = self.vk
        arguments = {
            "srcAccessMask": vk.VK_ACCESS_TRANSFER_WRITE_BIT if release else 0,
            "dstAccessMask": 0 if release else vk.VK_ACCESS_TRANSFER_WRITE_BIT,
            "srcQueueFamilyIndex": self.family if release else vk.VK_QUEUE_FAMILY_EXTERNAL,
            "dstQueueFamilyIndex": vk.VK_QUEUE_FAMILY_EXTERNAL if release else self.family,
        }
        if self.image:
            barrier = vk.VkImageMemoryBarrier(
                **arguments,
                image=self.resource,
                subresourceRange=self.image_range(),
                oldLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
                newLayout=vk.VK_IMAGE_LAYOUT_GENERAL,
            )
        else:
            barrier = vk.VkBufferMemoryBarrier(**arguments, buffer=self.resource, offset=0, size=self.size)
        vk.vkCmdPipelineBarrier(
            command,
            vk.VK_PIPELINE_STAGE_TRANSFER_BIT if release else vk.VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
            vk.VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT if release else vk.VK_PIPELINE_STAGE_TRANSFER_BIT,
            0,
            0,
            None,
            0 if self.image else 1,
            None if self.image else [barrier],
            1 if self.image else 0,
            [barrier] if self.image else None,
        )

    def image_range(self):
        return self.vk.VkImageSubresourceRange(
            aspectMask=self.vk.VK_IMAGE_ASPECT_COLOR_BIT,
            baseMipLevel=0,
            levelCount=1,
            baseArrayLayer=0,
            layerCount=1,
        )

    def copy_to_staging(self, pointer):
        if self.image:
            copy = cu.CUDA_MEMCPY3D()
            copy.srcMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_ARRAY
            copy.srcArray = self.array
            copy.dstMemoryType = cu.CUmemorytype.CU_MEMORYTYPE_DEVICE
            copy.dstDevice = pointer
            copy.dstPitch = self.width * 4
            copy.dstHeight = self.height
            copy.WidthInBytes = self.width * 4
            copy.Height = self.height
            copy.Depth = 1
            checked(cu.cuMemcpy3DAsync(copy, self.stream))
        else:
            checked(cu.cuMemcpyDtoDAsync(pointer, self.pointer, self.size, self.stream))

    def publish(self, warp_stream):
        vk = self.vk
        vk.vkQueueSubmit(
            self.queue,
            1,
            [vk.VkSubmitInfo(pCommandBuffers=[self.fill], pSignalSemaphores=[self.ready])],
            vk.VK_NULL_HANDLE,
        )
        checked(
            cu.cuWaitExternalSemaphoresAsync(
                [self.cuda_ready], [cu.CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS()], 1, self.stream
            )
        )
        checked(cu.cuEventRecord(self.event, self.stream))
        if warp_stream is not None:
            checked(cu.cuStreamWaitEvent(warp_stream, self.event, 0))

    def release(self, warp_stream):
        vk = self.vk
        checked(
            cu.cuSignalExternalSemaphoresAsync(
                [self.cuda_done], [cu.CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS()], 1, warp_stream
            )
        )
        vk.vkQueueSubmit(
            self.queue,
            1,
            [
                vk.VkSubmitInfo(
                    pWaitSemaphores=[self.done],
                    pWaitDstStageMask=[vk.VK_PIPELINE_STAGE_TRANSFER_BIT],
                    pCommandBuffers=[self.acquire],
                )
            ],
            vk.VK_NULL_HANDLE,
        )

    def destroy(self):
        vk = self.vk
        vk.vkQueueWaitIdle(self.queue)
        if self.image:
            checked(cu.cuMipmappedArrayDestroy(self.mipmapped))
        else:
            checked(cu.cuMemFree(self.pointer))
        checked(cu.cuDestroyExternalMemory(self.external_memory))
        for semaphore, imported in ((self.ready, self.cuda_ready), (self.done, self.cuda_done)):
            checked(cu.cuDestroyExternalSemaphore(imported))
            vk.vkDestroySemaphore(self.device, semaphore, None)
        checked(cu.cuEventDestroy(self.event))
        checked(cu.cuStreamDestroy(self.stream))
        vk.vkDestroyCommandPool(self.device, self.pool, None)
        if self.image:
            vk.vkDestroyImage(self.device, self.resource, None)
        else:
            vk.vkDestroyBuffer(self.device, self.resource, None)
        vk.vkFreeMemory(self.device, self.memory, None)
        vk.vkDestroyDevice(self.device, None)
        vk.vkDestroyInstance(self.instance, None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("cuda", "vulkan", "vulkan-image"), default="vulkan")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--image-mib", type=int, default=128)
    parser.add_argument("--image-side", type=int, help="Override image dimensions and byte count with a square image")
    parser.add_argument("--buffers", type=int, default=1)
    parser.add_argument("--producer-flags", type=int, choices=(0, 1), default=1)
    parser.add_argument(
        "--producer-repeats", type=int, default=1, help="Queue repeated Vulkan writes before the CUDA wait"
    )
    parser.add_argument(
        "--report-pending", action="store_true", help="Query the producer event immediately before capture"
    )
    parser.add_argument("--staging", action="store_true", help="Allocate/copy/free a temporary on the producer stream")
    parser.add_argument(
        "--background", action="store_true", help="Queue all producers without joining them to the Warp stream"
    )
    parser.add_argument(
        "--non-dedicated", action="store_true", help="Export Vulkan buffer memory without a dedicated allocation"
    )
    args = parser.parse_args()
    if min(args.rounds, args.image_mib, args.buffers, args.producer_repeats) < 1:
        parser.error("Round, image, and buffer counts must be positive")
    if args.backend == "vulkan-image" and not (args.staging or args.background):
        parser.error("--backend vulkan-image requires --staging or --background")
    if args.background and args.staging:
        parser.error("--background and --staging are separate controls")
    if args.non_dedicated and args.backend != "vulkan":
        parser.error("--non-dedicated requires the vulkan buffer backend")
    if args.image_side is not None and (args.image_side < 1 or args.backend != "vulkan-image"):
        parser.error("--image-side requires a positive side length and the vulkan-image backend")
    for package in ("ovrtx", "ovstage", "torch"):
        if importlib.util.find_spec(package) is not None:
            raise RuntimeError(f"Run this probe in an environment without {package} installed")
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    wp.config.kernel_cache_dir = str(Path(__file__).parent / "kernel_cache")
    wp.init()
    wp.set_device("cuda:0")
    checked(cu.cuCtxSetCurrent(wp.get_device().context))
    versions = {name: importlib.metadata.version(name) for name in ("warp-lang", "cuda-bindings", "numpy")}
    print("Versions:", versions, "Args:", vars(args), flush=True)
    print("OVRTX, OVStage, and Torch unavailable", flush=True)
    values = wp.zeros((8192, 32), dtype=wp.float32)
    wp.load_module(module=prefix_kernels)
    if args.backend == "cuda":
        buffers = [CudaBuffer(args.image_mib * 1024 * 1024, args.producer_flags) for _ in range(args.buffers)]
    else:
        size = args.image_side**2 * 4 if args.image_side else args.image_mib * 1024 * 1024
        buffers = [
            VulkanBuffer(
                size,
                args.producer_flags,
                image=args.backend == "vulkan-image",
                image_side=args.image_side,
                producer_repeats=args.producer_repeats,
                non_dedicated=args.non_dedicated,
            )
            for _ in range(args.buffers)
        ]
    warp_stream = cu.CUstream(wp.get_stream().cuda_stream)
    for iteration in range(args.rounds):
        source = buffers[iteration % len(buffers)]
        if args.background:
            for source in buffers:
                source.publish(None)
        else:
            source.publish(warp_stream)
        pointer = None if args.backend == "vulkan-image" else source.pointer
        if args.staging:
            pointer = checked(cu.cuMemAllocAsync(source.size, source.stream))
            source.copy_to_staging(pointer)
            ready = checked(cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING))
            checked(cu.cuEventRecord(ready, source.stream))
            checked(cu.cuStreamWaitEvent(warp_stream, ready, 0))
        copied = None
        if not args.background:
            view = wp.array(ptr=int(pointer), shape=(source.size // 4,), dtype=wp.uint32, device="cuda:0")
            copied = wp.empty_like(view)
            wp.copy(copied, view)
            source.release(warp_stream)
            del view
        if args.staging:
            released = checked(cu.cuEventCreate(cu.CUevent_flags.CU_EVENT_DISABLE_TIMING))
            checked(cu.cuEventRecord(released, warp_stream))
            checked(cu.cuStreamWaitEvent(source.stream, released, 0))
            checked(cu.cuMemFreeAsync(pointer, source.stream))
            checked(cu.cuEventDestroy(ready))
            checked(cu.cuEventDestroy(released))
        if args.report_pending:
            pending = []
            for producer in buffers if args.background else [source]:
                status = cu.cuEventQuery(producer.event)[0]
                if status not in (cu.CUresult.CUDA_SUCCESS, cu.CUresult.CUDA_ERROR_NOT_READY):
                    raise RuntimeError(f"Producer event query failed: {status}")
                pending.append(status == cu.CUresult.CUDA_ERROR_NOT_READY)
            print(f"Round {iteration + 1}: producer pending={pending}", flush=True)
        print(f"Round {iteration + 1}: before captured allocation", flush=True)
        with wp.ScopedCapture(capture_mode=wp.CaptureMode.THREAD_LOCAL) as capture:
            wp.launch(prefix_kernels.prefix, dim=values.shape, inputs=[values])
            temporary = wp.empty(8192, dtype=wp.int32)
            temporary.fill_(42)
        print(f"Round {iteration + 1}: after captured allocation", flush=True)
        wp.capture_launch(capture.graph)
        np.testing.assert_array_equal(temporary.numpy(), np.full(8192, 42, dtype=np.int32))
        if copied is not None:
            np.testing.assert_array_equal(
                copied.numpy(), np.uint32(0x2A2A2A2A if args.backend == "vulkan-image" else 42)
            )
        del copied, temporary, capture
        for producer in buffers if args.background else [source]:
            if args.background:
                producer.release(producer.stream)
                checked(cu.cuStreamSynchronize(producer.stream))
            if args.backend != "cuda":
                producer.vk.vkQueueWaitIdle(producer.queue)  # Command buffers must complete before resubmission.
    np.testing.assert_array_equal(values.numpy(), np.full((8192, 32), args.rounds, dtype=np.float32))
    wp.synchronize_device()
    for source in buffers:
        source.destroy()
    faulthandler.cancel_dump_traceback_later()
    print(f"Passed: {args.rounds} {args.backend}/CUDA/Warp capture rounds", flush=True)


if __name__ == "__main__":
    main()
