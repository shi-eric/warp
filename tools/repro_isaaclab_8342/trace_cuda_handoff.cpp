// Record CUDA API arguments without invoking CUDA from a CUPTI callback.
#include <windows.h>
#include <cstdio>
#include <cstring>
#include <cupti.h>

static HANDLE output = INVALID_HANDLE_VALUE;
static SRWLOCK lock = SRWLOCK_INIT;
static CUpti_SubscriberHandle subscriber;

static void CUPTIAPI callback(void*, CUpti_CallbackDomain domain, CUpti_CallbackId id, const void* data)
{
    const auto* info = static_cast<const CUpti_CallbackData*>(data);
    const char* name = info->functionName;
    if (domain == CUPTI_CB_DOMAIN_DRIVER_API) {
        if (!strstr(name, "AllocAsync") && !strstr(name, "FreeAsync") && !strstr(name, "Ctx") &&
            !strstr(name, "LaunchKernel") && !strstr(name, "StreamBeginCapture") && !strstr(name, "StreamEndCapture"))
            return;
    } else if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        if (!strstr(name, "External") && !strstr(name, "MallocAsync") && !strstr(name, "FreeAsync") &&
            !strstr(name, "Event") && !strstr(name, "Memcpy") && !strstr(name, "Stream"))
            return;
    } else {
        return;
    }
    const bool exit = info->callbackSite == CUPTI_API_EXIT;
    const int status = exit ? *static_cast<const int*>(info->functionReturnValue) : -1;
    char detail[1400] = {};
    if (domain == CUPTI_CB_DOMAIN_RUNTIME_API) {
        switch (id) {
        case CUPTI_RUNTIME_TRACE_CBID_cudaImportExternalMemory_v10000: {
            const auto* p = static_cast<const cudaImportExternalMemory_v10000_params*>(info->functionParams);
            const auto* d = p->memHandleDesc;
            snprintf(detail, sizeof(detail), "type=%d handle=%p size=%llu flags=%u result=%p", int(d->type),
                     d->handle.win32.handle, d->size, d->flags, exit && status == 0 ? *p->extMem_out : nullptr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaImportExternalSemaphore_v10000: {
            const auto* p = static_cast<const cudaImportExternalSemaphore_v10000_params*>(info->functionParams);
            const auto* d = p->semHandleDesc;
            snprintf(detail, sizeof(detail), "type=%d handle=%p flags=%u result=%p", int(d->type),
                     d->handle.win32.handle, d->flags, exit && status == 0 ? *p->extSem_out : nullptr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaExternalMemoryGetMappedBuffer_v10000: {
            const auto* p = static_cast<const cudaExternalMemoryGetMappedBuffer_v10000_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "memory=%p offset=%llu size=%llu flags=%u result=%p", p->extMem,
                     p->bufferDesc->offset, p->bufferDesc->size, p->bufferDesc->flags,
                     exit && status == 0 ? *p->devPtr : nullptr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaExternalMemoryGetMappedMipmappedArray_v10000: {
            const auto* p = static_cast<const cudaExternalMemoryGetMappedMipmappedArray_v10000_params*>(info->functionParams);
            const auto* d = p->mipmapDesc;
            snprintf(detail, sizeof(detail),
                     "memory=%p offset=%llu extent=%zux%zux%zu channels=%d,%d,%d,%d kind=%d flags=%u levels=%u result=%p",
                     p->extMem, d->offset, d->extent.width, d->extent.height, d->extent.depth,
                     d->formatDesc.x, d->formatDesc.y, d->formatDesc.z, d->formatDesc.w, int(d->formatDesc.f),
                     d->flags, d->numLevels, exit && status == 0 ? *p->mipmap : nullptr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaWaitExternalSemaphoresAsync_v11020: {
            const auto* p = static_cast<const cudaWaitExternalSemaphoresAsync_v11020_params*>(info->functionParams);
            const auto count = p->numExtSems;
            snprintf(detail, sizeof(detail), "stream=%p count=%u semaphore=%p value=%llu flags=%u", p->stream,
                     count, count ? p->extSemArray[0] : nullptr,
                     count ? p->paramsArray[0].params.fence.value : 0,
                     count ? p->paramsArray[0].flags : 0);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMallocAsync_v11020: {
            const auto* p = static_cast<const cudaMallocAsync_v11020_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "stream=%p bytes=%zu result=%p", p->hStream, p->size,
                     exit && status == 0 ? *p->devPtr : nullptr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaFreeAsync_v11020: {
            const auto* p = static_cast<const cudaFreeAsync_v11020_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "stream=%p pointer=%p", p->hStream, p->devPtr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpy3DAsync_v3020: {
            const auto* p = static_cast<const cudaMemcpy3DAsync_v3020_params*>(info->functionParams);
            const auto* d = p->p;
            snprintf(detail, sizeof(detail), "stream=%p sourceArray=%p destination=%p pitch=%zu extent=%zux%zux%zu kind=%d",
                     p->stream, d->srcArray, d->dstPtr.ptr, d->dstPtr.pitch,
                     d->extent.width, d->extent.height, d->extent.depth, int(d->kind));
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaMemcpyAsync_v3020: {
            const auto* p = static_cast<const cudaMemcpyAsync_v3020_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "stream=%p source=%p destination=%p bytes=%zu kind=%d",
                     p->stream, p->src, p->dst, p->count, int(p->kind));
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventRecord_v3020: {
            const auto* p = static_cast<const cudaEventRecord_v3020_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "stream=%p event=%p", p->stream, p->event);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamWaitEvent_v3020: {
            const auto* p = static_cast<const cudaStreamWaitEvent_v3020_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "stream=%p event=%p flags=%u", p->stream, p->event, p->flags);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaEventDestroy_v3020: {
            const auto* p = static_cast<const cudaEventDestroy_v3020_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "event=%p", p->event);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamCreateWithFlags_v5000: {
            const auto* p = static_cast<const cudaStreamCreateWithFlags_v5000_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "flags=%u result=%p", p->flags,
                     exit && status == 0 ? *p->pStream : nullptr);
            break;
        }
        case CUPTI_RUNTIME_TRACE_CBID_cudaStreamBeginCapture_v10000: {
            const auto* p = static_cast<const cudaStreamBeginCapture_v10000_params*>(info->functionParams);
            snprintf(detail, sizeof(detail), "stream=%p mode=%d", p->stream, int(p->mode));
            break;
        }
        default: break;
        }
    } else if (id == CUPTI_DRIVER_TRACE_CBID_cuMemAllocAsync) {
        const auto* p = static_cast<const cuMemAllocAsync_params*>(info->functionParams);
        snprintf(detail, sizeof(detail), "stream=%p bytes=%zu result=0x%llx", p->hStream, p->bytesize,
                 exit && status == 0 ? *p->dptr : 0);
    }
    LARGE_INTEGER timestamp;
    QueryPerformanceCounter(&timestamp);
    char line[1800];
    const int size = snprintf(line, sizeof(line), "qpc=%lld tid=%lu domain=%u site=%s name=%s context=%p uid=%u correlation=%u status=%d %s\n",
                              timestamp.QuadPart, GetCurrentThreadId(), unsigned(domain), exit ? "EXIT" : "ENTER",
                              name, info->context, info->contextUid, info->correlationId, status, detail);
    AcquireSRWLockExclusive(&lock);
    DWORD written;
    WriteFile(output, line, DWORD(size), &written, nullptr);
    ReleaseSRWLockExclusive(&lock);
}

extern "C" __declspec(dllexport) int start_trace(const char* path)
{
    output = CreateFileA(path, GENERIC_WRITE, FILE_SHARE_READ, nullptr, CREATE_ALWAYS, FILE_ATTRIBUTE_NORMAL, nullptr);
    if (output == INVALID_HANDLE_VALUE)
        return -1;
    auto result = cuptiSubscribe(&subscriber, callback, nullptr);
    if (result == CUPTI_SUCCESS)
        result = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_RUNTIME_API);
    if (result == CUPTI_SUCCESS)
        result = cuptiEnableDomain(1, subscriber, CUPTI_CB_DOMAIN_DRIVER_API);
    return int(result);
}

extern "C" __declspec(dllexport) int stop_trace()
{
    const auto result = cuptiUnsubscribe(subscriber);
    CloseHandle(output);
    output = INVALID_HANDLE_VALUE;
    return int(result);
}
