// Windows CUDA driver stall: once UVM-Lite is active, placing a new CUDA VA range over foreign GPU VA
// retries in 2 MiB steps, and each failed attempt costs ~2 s of kernel CPU (Isaac Lab #8342).
//
// Build (x64 developer prompt):
//   cl /nologo /EHsc /O2 /I"%CUDA_PATH%\include" minimal_uvm_va_stall.cpp
//      /link /LIBPATH:"%CUDA_PATH%\lib\x64" cuda.lib
// Usage: minimal_uvm_va_stall.exe [--no-managed] [--obstacle-mib N]
// Expected: ~1 s per MiB of obstacle (16 s by default); ~0.01 s with --no-managed.
// Exit code: 0 no stall, 1 stall, 3 error.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// D3DKMT structs from d3dkmthk.h, declared here so only the CUDA toolkit is needed.
struct D3DKMT_OPENADAPTERFROMLUID {
    LUID AdapterLuid;
    UINT hAdapter;
};
struct D3DDDI_RESERVEGPUVIRTUALADDRESS {
    UINT hAdapter;
    UINT64 BaseAddress, MinimumAddress, MaximumAddress, Size;
    UINT ReservationType;
    UINT64 DriverProtection, VirtualAddress, PagingFenceValue;
};
struct D3DKMT_FREEGPUVIRTUALADDRESS {
    UINT hAdapter;
    UINT64 BaseAddress, Size;
};

#define CHECK(call)                                                                                                    \
    do {                                                                                                               \
        if (CUresult result = (call)) {                                                                                \
            std::fprintf(stderr, "%s failed: %d\n", #call, result);                                                    \
            return 3;                                                                                                  \
        }                                                                                                              \
    } while (0)

int main(int argc, char** argv)
{
    bool managed = true;
    unsigned long long obstacle_size = 16ull << 20;
    for (int i = 1; i < argc; ++i) {
        if (!std::strcmp(argv[i], "--no-managed"))
            managed = false;
        else if (!std::strcmp(argv[i], "--obstacle-mib") && i + 1 < argc && std::atoll(argv[i + 1]) > 0)
            obstacle_size = static_cast<unsigned long long>(std::atoll(argv[++i])) << 20;
        else {
            std::fprintf(stderr, "Usage: %s [--no-managed] [--obstacle-mib N]\n", argv[0]);
            return 3;
        }
    }

    CHECK(cuInit(0));
    CUdevice device;
    CHECK(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK(cuDevicePrimaryCtxRetain(&context, device));
    CHECK(cuCtxSetCurrent(context));

    // Ingredient 1: any managed allocation initializes UVM-Lite for the whole process.
    CUdeviceptr pointer = 0;
    if (managed)
        CHECK(cuMemAllocManaged(&pointer, 0x4820, CU_MEM_ATTACH_GLOBAL));

    // Ingredient 2: GPU VA that CUDA does not own (Vulkan memory in Isaac Lab) at the address where CUDA
    // will place its next 96 GiB range. CUDA places ranges first-fit from 0x2000000000, so find that spot
    // with a probe reservation, free it, and occupy its start.
    HMODULE gdi = LoadLibraryW(L"gdi32.dll");
    auto open_adapter = reinterpret_cast<LONG(APIENTRY*)(D3DKMT_OPENADAPTERFROMLUID*)>(
        GetProcAddress(gdi, "D3DKMTOpenAdapterFromLuid"));
    auto reserve = reinterpret_cast<LONG(APIENTRY*)(D3DDDI_RESERVEGPUVIRTUALADDRESS*)>(
        GetProcAddress(gdi, "D3DKMTReserveGpuVirtualAddress"));
    auto release = reinterpret_cast<LONG(APIENTRY*)(D3DKMT_FREEGPUVIRTUALADDRESS*)>(
        GetProcAddress(gdi, "D3DKMTFreeGpuVirtualAddress"));
    D3DKMT_OPENADAPTERFROMLUID adapter = {};
    unsigned int node_mask = 0;
    CHECK(cuDeviceGetLuid(reinterpret_cast<char*>(&adapter.AdapterLuid), &node_mask, device));
    if (!open_adapter || !reserve || !release || open_adapter(&adapter))
        return 3;
    D3DDDI_RESERVEGPUVIRTUALADDRESS probe = {};
    probe.hAdapter = adapter.hAdapter;
    probe.MinimumAddress = 0x20'0000'0000ull;
    probe.Size = 96ull << 30;
    if (reserve(&probe))
        return 3;
    D3DKMT_FREEGPUVIRTUALADDRESS free_probe = {adapter.hAdapter, probe.VirtualAddress, probe.Size};
    if (release(&free_probe))
        return 3;
    D3DDDI_RESERVEGPUVIRTUALADDRESS obstacle = {};
    obstacle.hAdapter = adapter.hAdapter;
    obstacle.BaseAddress = probe.VirtualAddress;
    obstacle.Size = obstacle_size;
    if (reserve(&obstacle))
        return 3;
    std::printf("Foreign GPU VA at [%#llx, %#llx)\n", obstacle.VirtualAddress, obstacle.VirtualAddress + obstacle_size);

    // Trigger: the process's first stream-ordered allocation must reserve the default pool's 96 GiB range.
    // With UVM-Lite active, each placement attempt over the foreign VA fails, is released from UVM-Lite
    // (~2 s), and is retried 2 MiB higher. Without UVM-Lite, the driver skips past it in milliseconds.
    // (In Isaac Lab, the first captured allocation reserving the graph-memory range hits the same path.)
    auto start = std::chrono::steady_clock::now();
    CHECK(cuMemAllocAsync(&pointer, 32768, nullptr));
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    std::printf("First cuMemAllocAsync at %#llx took %.3f s%s\n", pointer, seconds,
                seconds > 1.0 ? ": STALLED" : "");
    return seconds > 1.0 ? 1 : 0;
}
