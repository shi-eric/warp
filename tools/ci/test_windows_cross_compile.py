# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
import pathlib
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import build_lib
import build_llvm
from warp._src import build_dll


class TestMsvcToolchainSelection(unittest.TestCase):
    def test_native_arm64_uses_native_arm64_tools(self):
        self.assertEqual(build_dll._msvc_toolchain_layout("aarch64"), ("HostARM64", "arm64"))

    def test_native_arm64_uses_arm64_environment(self):
        script, arguments = build_dll._msvc_environment_script("C:/VisualStudio", "aarch64")

        self.assertEqual(script, "C:/VisualStudio/Common7/Tools/VsDevCmd.bat")
        self.assertEqual(arguments, ["-arch=arm64", "-host_arch=arm64"])

    def test_x86_64_host_can_target_arm64_tools(self):
        self.assertEqual(build_dll._msvc_toolchain_layout("x86_64", "aarch64"), ("HostX64", "arm64"))

    def test_x86_64_host_uses_arm64_cross_environment(self):
        script, arguments = build_dll._msvc_environment_script("C:/VisualStudio", "x86_64", "aarch64")

        self.assertEqual(script, "C:/VisualStudio/VC/Auxiliary/Build/vcvarsall.bat")
        self.assertEqual(arguments, ["amd64_arm64"])

    def test_manual_toolchain_uses_host_tools_and_target_libraries(self):
        with patch.dict(os.environ, {"INCLUDE": "", "LIB": "", "PATH": ""}):
            compiler = build_dll.set_msvc_env("/MSVC", "/WindowsSDK", host_arch="x86_64", target_arch="aarch64")

            self.assertEqual(compiler, "/MSVC/bin/HostX64/arm64/cl.exe")
            self.assertIn("/MSVC/lib/arm64", os.environ["LIB"])
            self.assertIn("/WindowsSDK/lib/ucrt/arm64", os.environ["LIB"])
            self.assertIn("/MSVC/bin/HostX64/arm64", os.environ["PATH"])
            self.assertIn("/MSVC/bin/HostX64/x64", os.environ["PATH"])

    def test_auto_discovery_configures_x86_64_to_arm64_environment(self):
        def run_command(command, print_success_output=True):
            if "vswhere.exe" in command:
                return b"C:/VisualStudio"
            if 'vcvarsall.bat" amd64_arm64' in command:
                return (
                    f"banner\n{build_dll._VCVARS_ENV_DUMP_MARKER}\n"
                    "VSCMD_ARG_HOST_ARCH=x64\n"
                    "VSCMD_ARG_TGT_ARCH=arm64\n"
                    "VCToolsVersion=14.40\n"
                ).encode()
            self.fail(f"Unexpected command: {command}")

        with (
            patch.object(build_dll.os, "name", "nt"),
            patch.object(build_dll.os.path, "expandvars", return_value="C:/vswhere.exe"),
            patch.object(build_dll.os.path, "isfile", return_value=True),
            patch.object(build_dll, "run_cmd", side_effect=run_command),
            patch.object(build_dll.shutil, "which", return_value="C:/VisualStudio/VC/bin/HostX64/arm64/cl.exe"),
            patch.dict(os.environ, {}, clear=True),
        ):
            compiler = build_dll.find_host_compiler(host_arch="x86_64", target_arch="aarch64")

            self.assertEqual(compiler, "C:/VisualStudio/VC/bin/HostX64/arm64/cl.exe")
            self.assertEqual(os.environ["VSCMD_ARG_HOST_ARCH"], "x64")
            self.assertEqual(os.environ["VSCMD_ARG_TGT_ARCH"], "arm64")


class TestMathDxTargetSelection(unittest.TestCase):
    def test_arm64_validation_uses_arm64_libraries(self):
        with tempfile.TemporaryDirectory() as libmathdx_path:
            os.makedirs(os.path.join(libmathdx_path, "include"))
            os.makedirs(os.path.join(libmathdx_path, "lib", "arm64"))

            with patch.object(build_lib.platform, "system", return_value="Windows"):
                self.assertTrue(build_lib.validate_libmathdx_path(libmathdx_path, target_arch="aarch64"))

    def test_packman_pull_uses_target_architecture(self):
        def run_packman(command, **kwargs):
            platform_index = command.index("--platform") + 1
            self.assertEqual(command[platform_index], "windows-aarch64")
            return ""

        with (
            patch.object(build_lib.platform, "system", return_value="Windows"),
            patch.object(build_lib.subprocess, "check_output", side_effect=run_packman),
            patch.dict(os.environ, {}, clear=True),
        ):
            path = build_lib.find_libmathdx(13, "/warp", target_arch="aarch64")

        self.assertEqual(path, "/warp/_build/target-deps/libmathdx")


class TestBuildLibTargetSelection(unittest.TestCase):
    def test_windows_cross_build_allows_cuda_13_4(self):
        output = io.StringIO()
        with (
            patch.object(build_lib.platform, "system", return_value="Windows"),
            patch.object(build_lib, "machine_architecture", return_value="x86_64"),
            patch.object(build_lib, "generate_exports_header_file"),
            patch.object(build_lib, "generate_version_header"),
            patch.object(build_lib.build_dll, "find_host_compiler", return_value="C:/MSVC/cl.exe"),
            patch.object(build_lib.build_dll, "build_dll"),
            patch.dict(os.environ, {"GITLAB_CI": "1"}),
            redirect_stdout(output),
        ):
            result = build_lib.main(
                [
                    "--target-arch",
                    "aarch64",
                    "--cuda-path",
                    "C:/CUDA/v13.4",
                    "--no-use-libmathdx",
                    "--no-standalone",
                    "--no-verbose",
                ]
            )

        self.assertEqual(result, 0, output.getvalue())

    def test_windows_cross_build_rejects_old_cuda_before_fetching_mathdx(self):
        fetches = []
        output = io.StringIO()

        def find_libmathdx(*args, **kwargs):
            fetches.append((args, kwargs))

        with (
            patch.object(build_lib.platform, "system", return_value="Windows"),
            patch.object(build_lib, "machine_architecture", return_value="x86_64"),
            patch.object(build_lib, "find_libmathdx", side_effect=find_libmathdx),
            patch.object(build_lib, "generate_exports_header_file"),
            patch.object(build_lib, "generate_version_header"),
            patch.object(build_lib.build_dll, "get_cuda_toolkit_version", return_value=(13, 3)),
            patch.object(build_lib.build_dll, "find_host_compiler", return_value="C:/MSVC/cl.exe"),
            redirect_stdout(output),
        ):
            result = build_lib.main(
                [
                    "--target-arch",
                    "aarch64",
                    "--cuda-path",
                    "C:/CUDA/v13.3",
                    "--no-standalone",
                    "--no-verbose",
                ]
            )

        self.assertEqual(result, 1)
        self.assertEqual(fetches, [])
        self.assertIn("CUDA Toolkit 13.4", output.getvalue())

    def test_windows_cross_build_rejects_source_llvm(self):
        output = io.StringIO()
        with (
            patch.object(build_lib.platform, "system", return_value="Windows"),
            patch.object(build_lib, "machine_architecture", return_value="x86_64"),
            patch.object(build_lib, "generate_exports_header_file"),
            patch.object(build_lib, "generate_version_header"),
            patch.object(build_lib.build_dll, "set_msvc_env", return_value="C:/MSVC/bin/HostX64/arm64/cl.exe"),
            patch.object(build_lib.build_dll, "build_dll"),
            patch.object(build_lib.build_llvm, "check_build_dependencies"),
            patch.object(build_lib.build_llvm, "build_llvm_clang_from_source"),
            patch.object(build_lib.build_llvm, "build_warp_clang"),
            patch.dict(os.environ, {"GITLAB_CI": "1"}),
            redirect_stdout(output),
        ):
            result = build_lib.main(
                [
                    "--target-arch",
                    "aarch64",
                    "--build-llvm",
                    "--no-cuda",
                    "--msvc-path",
                    "C:/MSVC",
                    "--sdk-path",
                    "C:/WindowsSDK",
                    "--no-verbose",
                ]
            )

        self.assertEqual(result, 1)
        self.assertIn("cannot cross-compile LLVM from source", output.getvalue())

    def test_windows_cross_build_uses_arm64_output_directory(self):
        observed = {}

        def build_runtime(args, dll_path, cpp_paths, cu_paths):
            observed["host_arch"] = args.host_arch
            observed["target_arch"] = args.target_arch
            observed["dll_path"] = dll_path

        with (
            patch.object(build_lib.platform, "system", return_value="Windows"),
            patch.object(build_lib, "machine_architecture", return_value="x86_64"),
            patch.object(build_lib, "generate_exports_header_file"),
            patch.object(build_lib, "generate_version_header"),
            patch.object(build_lib.build_dll, "set_msvc_env", return_value="C:/MSVC/bin/HostX64/arm64/cl.exe"),
            patch.object(build_lib.build_dll, "build_dll", side_effect=build_runtime),
            patch.dict(os.environ, {"GITLAB_CI": "1"}),
        ):
            result = build_lib.main(
                [
                    "--target-arch",
                    "aarch64",
                    "--no-cuda",
                    "--no-use-libmathdx",
                    "--no-standalone",
                    "--msvc-path",
                    "C:/MSVC",
                    "--sdk-path",
                    "C:/WindowsSDK",
                    "--no-verbose",
                ]
            )

        self.assertEqual(result, 0)
        self.assertEqual(observed["host_arch"], "x86_64")
        self.assertEqual(observed["target_arch"], "aarch64")
        base_path = os.path.dirname(os.path.realpath(build_lib.__file__))
        self.assertEqual(observed["dll_path"], os.path.join(base_path, "warp", "bin", "arm64", "warp.dll"))


class TestLlvmTargetSelection(unittest.TestCase):
    def test_source_build_uses_requested_target_architecture(self):
        observed = {}

        def build_for_arch(args, arch, llvm_source):
            observed["arch"] = arch

        args = SimpleNamespace(llvm_source_path="C:/llvm-project", target_arch="aarch64")
        with (
            patch.object(build_llvm.sys, "platform", "win32"),
            patch.object(build_llvm, "build_llvm_clang_from_source_for_arch", side_effect=build_for_arch),
        ):
            build_llvm.build_llvm_clang_from_source(args)

        self.assertEqual(observed["arch"], "aarch64")

    def test_warp_clang_uses_requested_target_architecture(self):
        observed = {}

        def build_for_arch(args, lib_name, arch):
            observed["arch"] = arch

        args = SimpleNamespace(target_arch="aarch64")
        with (
            patch.object(build_llvm.sys, "platform", "win32"),
            patch.object(build_llvm, "build_warp_clang_for_arch", side_effect=build_for_arch),
        ):
            build_llvm.build_warp_clang(args, "warp-clang.dll")

        self.assertEqual(observed["arch"], "aarch64")

    def test_warp_clang_uses_cross_build_output_directory(self):
        observed = {}

        def build_for_arch(args, dll_path, **kwargs):
            observed["dll_path"] = dll_path

        args = SimpleNamespace(
            bin_subdir="arm64",
            build_llvm=False,
            llvm_path=None,
            mode="release",
        )
        with (
            patch.object(build_llvm, "build_path", "C:/warp"),
            patch.object(build_llvm.os, "name", "nt"),
            patch.object(build_llvm, "fetch_prebuilt_libraries"),
            patch.object(build_llvm, "prebuilt_library_path", return_value="C:/llvm"),
            patch.object(build_llvm.os.path, "isdir", return_value=True),
            patch.object(build_llvm.os, "listdir", return_value=["LLVM.lib"]),
            patch.object(build_llvm, "build_dll_for_arch", side_effect=build_for_arch),
        ):
            build_llvm.build_warp_clang_for_arch(args, "warp-clang.dll", "aarch64")

        self.assertEqual(observed["dll_path"], "C:/warp/bin/arm64/warp-clang.dll")


class TestRuntimeTargetSelection(unittest.TestCase):
    def _run_windows_arm64_cuda_build(self, cuda_version):
        commands = []
        warp_home_path = pathlib.PosixPath(build_dll.__file__).parent.parent
        path_result = SimpleNamespace(parent=SimpleNamespace(parent=warp_home_path))
        args = SimpleNamespace(
            clang_build_toolchain=False,
            compile_time_trace=False,
            cuda_path="C:/CUDA",
            fast_math=False,
            host_arch="x86_64",
            host_compiler="C:/MSVC/bin/HostX64/arm64/cl.exe",
            jobs=1,
            libmathdx_path="C:/MathDx",
            llvm_path=None,
            mode="release",
            quick=True,
            sanitize=None,
            use_dynamic_cuda=False,
            verbose=False,
            verify_fp=False,
        )
        with (
            patch.object(build_dll.os, "name", "nt"),
            patch.object(build_dll.sys, "platform", "win32"),
            patch.object(build_dll.pathlib, "Path", return_value=path_result),
            patch.object(build_dll, "get_cuda_toolkit_version", return_value=cuda_version),
            patch.object(build_dll, "find_nvcc_executable", return_value="nvcc"),
            patch.object(build_dll, "get_llvm_include_paths", return_value=[]),
            patch.object(build_dll, "run_cmd", side_effect=commands.append),
        ):
            build_dll.build_dll_for_arch(
                args,
                "C:/warp/bin/arm64/warp.dll",
                cpp_paths=[],
                cu_paths=["C:/warp/native/warp.cu"],
                arch="aarch64",
            )

        return commands

    def _run_windows_arm64_cpu_link(self):
        commands = []
        warp_home_path = pathlib.PosixPath(build_dll.__file__).parent.parent
        path_result = SimpleNamespace(parent=SimpleNamespace(parent=warp_home_path))
        args = SimpleNamespace(
            cuda_path=None,
            fast_math=False,
            host_arch="x86_64",
            host_compiler="C:/MSVC/bin/HostX64/arm64/cl.exe",
            jobs=1,
            libmathdx_path=None,
            llvm_path=None,
            mode="release",
            quick=True,
            sanitize=None,
            verbose=False,
            verify_fp=False,
        )
        with (
            patch.object(build_dll.os, "name", "nt"),
            patch.object(build_dll.sys, "platform", "win32"),
            patch.object(build_dll.pathlib, "Path", return_value=path_result),
            patch.object(build_dll, "get_llvm_include_paths", return_value=[]),
            patch.object(build_dll, "run_cmd", side_effect=commands.append),
        ):
            build_dll.build_dll_for_arch(
                args,
                "C:/warp/bin/arm64/warp.dll",
                cpp_paths=[],
                cu_paths=None,
                arch="aarch64",
            )

        return commands

    def test_runtime_build_uses_requested_target_architecture(self):
        observed = {}

        def build_for_arch(args, dll_path, cpp_paths, cu_paths, arch, libs):
            observed["arch"] = arch

        args = SimpleNamespace(target_arch="aarch64")
        with (
            patch.object(build_dll.sys, "platform", "win32"),
            patch.object(build_dll, "build_dll_for_arch", side_effect=build_for_arch),
        ):
            build_dll.build_dll(args, "warp.dll", [], [], libs=[])

        self.assertEqual(observed["arch"], "aarch64")

    def test_windows_arm64_cpu_link_uses_target_machine(self):
        commands = self._run_windows_arm64_cpu_link()
        link_command = next(command for command in commands if "/out:" in command)
        self.assertIn("/MACHINE:ARM64", link_command)

    def test_windows_arm64_cuda_commands_use_target_libraries(self):
        commands = self._run_windows_arm64_cuda_build((13, 4))

        cuda_command = next(command for command in commands if " -c " in command)
        link_command = next(command for command in commands if "/out:" in command)
        self.assertIn("-target-dir arm64", cuda_command)
        self.assertIn('--compiler-bindir="C:/MSVC/bin/HostX64/arm64"', cuda_command)
        self.assertIn('/LIBPATH:"C:/CUDA/lib/arm64"', link_command)
        self.assertIn('/LIBPATH:"C:/MathDx/lib/arm64"', link_command)

    def test_windows_arm64_rejects_cuda_before_13_4(self):
        with self.assertRaisesRegex(RuntimeError, "CUDA Toolkit 13.4"):
            self._run_windows_arm64_cuda_build((13, 3))


if __name__ == "__main__":
    unittest.main()
