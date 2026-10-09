// Optional tracer for minimal_uvm_va_stall.cpp: log the CUDA driver's UVM-Lite IOCTLs to stderr.
//
// Link it into the repro; a static initializer patches nvcuda64.dll's import of DeviceIoControl
// before main runs, so the repro source is unchanged:
//   cl /nologo /EHsc /O2 /I"%CUDA_PATH%\include" minimal_uvm_va_stall.cpp uvm_ioctl_trace.cpp
//      /link /LIBPATH:"%CUDA_PATH%\lib\x64" cuda.lib
//
// Codes 0x22e004/0x22e008/0x22e00c take {base, length, status}; their roles (reserve, release,
// commit) are inferred from the layout and behavior.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <cuda.h>

#include <chrono>
#include <cstdio>
#include <cstring>

namespace {

using DeviceIoControlFn = BOOL(WINAPI*)(HANDLE, DWORD, LPVOID, DWORD, LPVOID, DWORD, LPDWORD, LPOVERLAPPED);
DeviceIoControlFn original = nullptr;
const auto started = std::chrono::steady_clock::now();

const char* name_of(DWORD code)
{
    switch (code) {
    case 0x22e004: return "reserve";
    case 0x22e008: return "release";
    case 0x22e00c: return "commit";
    default: return nullptr;
    }
}

BOOL WINAPI traced(HANDLE handle, DWORD code, LPVOID in, DWORD in_size, LPVOID out, DWORD out_size, LPDWORD returned,
                   LPOVERLAPPED overlapped)
{
    const char* name = name_of(code);
    if (!name || in_size < 16)
        return original(handle, code, in, in_size, out, out_size, returned, overlapped);
    UINT64 range[2];
    std::memcpy(range, in, sizeof(range));
    auto start = std::chrono::steady_clock::now();
    BOOL ok = original(handle, code, in, in_size, out, out_size, returned, overlapped);
    auto end = std::chrono::steady_clock::now();
    std::fprintf(stderr, "[%8.3fs] %-7s base=%#llx length=%#llx ok=%d %10.3f ms\n",
                 std::chrono::duration<double>(start - started).count(), name, range[0], range[1], ok,
                 std::chrono::duration<double, std::milli>(end - start).count());
    return ok;
}

// Return nvcuda64.dll's import address table slot for kernel32.dll!DeviceIoControl.
void** find_slot(HMODULE module)
{
    auto base = reinterpret_cast<BYTE*>(module);
    auto nt = reinterpret_cast<IMAGE_NT_HEADERS*>(base + reinterpret_cast<IMAGE_DOS_HEADER*>(base)->e_lfanew);
    auto& directory = nt->OptionalHeader.DataDirectory[IMAGE_DIRECTORY_ENTRY_IMPORT];
    for (auto d = reinterpret_cast<IMAGE_IMPORT_DESCRIPTOR*>(base + directory.VirtualAddress); d->Name; ++d) {
        if (_stricmp(reinterpret_cast<char*>(base + d->Name), "kernel32.dll"))
            continue;
        auto names = reinterpret_cast<IMAGE_THUNK_DATA*>(base + (d->OriginalFirstThunk ? d->OriginalFirstThunk : d->FirstThunk));
        auto slots = reinterpret_cast<IMAGE_THUNK_DATA*>(base + d->FirstThunk);
        for (; names->u1.AddressOfData; ++names, ++slots) {
            if (IMAGE_SNAP_BY_ORDINAL(names->u1.Ordinal))
                continue;
            auto by_name = reinterpret_cast<IMAGE_IMPORT_BY_NAME*>(base + names->u1.AddressOfData);
            if (!std::strcmp(by_name->Name, "DeviceIoControl"))
                return reinterpret_cast<void**>(&slots->u1.Function);
        }
    }
    return nullptr;
}

// nvcuda64.dll is loaded by cuInit through the nvcuda.dll loader, so initialize first, then patch.
const bool installed = [] {
    HMODULE module = nullptr;
    if (cuInit(0) != CUDA_SUCCESS || !(module = GetModuleHandleW(L"nvcuda64.dll"))) {
        std::fprintf(stderr, "uvm_ioctl_trace: nvcuda64.dll not loaded; not tracing\n");
        return false;
    }
    void** slot = find_slot(module);
    if (!slot) {
        std::fprintf(stderr, "uvm_ioctl_trace: DeviceIoControl import not found; not tracing\n");
        return false;
    }
    DWORD protection;
    VirtualProtect(slot, sizeof(void*), PAGE_READWRITE, &protection);
    original = reinterpret_cast<DeviceIoControlFn>(*slot);
    *slot = reinterpret_cast<void*>(&traced);
    VirtualProtect(slot, sizeof(void*), protection, &protection);
    std::fprintf(stderr, "uvm_ioctl_trace: tracing UVM-Lite IOCTLs from nvcuda64.dll\n");
    return true;
}();

}  // namespace
