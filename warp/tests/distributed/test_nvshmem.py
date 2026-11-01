# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import glob
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from typing import ClassVar
from unittest import mock

import numpy as np

import warp as wp
import warp._src.context as warp_context
import warp._src.nvshmem as nvshmem_support
from warp.tests.unittest_utils import *

# Auxiliary scripts for multi-PE tests (same directory)
_DISTRIBUTED_DIR = os.path.dirname(__file__)


@wp.func
def nvshmem_style_helper(x: float):
    return x * 2.0


def _is_module_installed(name):
    return importlib.util.find_spec(name) is not None


def _is_nvshmem_device_library_available():
    """Whether compiling an NVSHMEM module would find a device library (override, host library, or wheel)."""
    toolkit_version = warp_context.runtime.toolkit_version
    cuda_major = toolkit_version[0] if toolkit_version else 13
    try:
        nvshmem_support.find_device_library(None, cuda_major)
    except RuntimeError:
        return False
    return True


class TestNVSHMEMCodegen(unittest.TestCase):
    """Tests for compiling and loading NVSHMEM modules in a single process."""

    def test_user_function_with_nvshmem_prefix(self):
        """A user function whose name starts with nvshmem_ should not require NVSHMEM."""

        @wp.kernel(module="unique")
        def helper_kernel(a: wp.array[float]):
            a[0] = nvshmem_style_helper(a[0])

        a = wp.ones(1, dtype=float, device="cpu")
        wp.launch(helper_kernel, dim=1, inputs=[a], device="cpu")
        np.testing.assert_allclose(a.numpy(), [2.0])

    def test_builtins_declare_nvshmem_requirement(self):
        """Every wp.nvshmem_* builtin carries requires_nvshmem, and no other builtin does."""
        for key, func in warp_context.builtin_functions.items():
            for f in [func, *getattr(func, "overloads", [])]:
                self.assertEqual(f.requires_nvshmem, key.startswith("nvshmem_"), key)

    def test_nvshmem_p_rejects_mismatched_value_type(self):
        """nvshmem_p() should require the value type to match the array dtype."""

        @wp.kernel(module="unique")
        def mismatched_kernel(dest: wp.array[wp.float32]):
            wp.nvshmem_p(dest, 0, wp.float64(1.0), 1)

        with self.assertRaisesRegex(RuntimeError, r"must match the array dtype"):
            wp.load_module(mismatched_kernel.module, device="cpu")

    def test_nvshmem_builtins_reject_cpu(self):
        """Modules that use NVSHMEM builtins should fail to compile for the CPU."""

        @wp.kernel(module="unique")
        def cpu_kernel(out_pe: wp.array[wp.int32]):
            out_pe[0] = wp.nvshmem_my_pe()

        with self.assertRaisesRegex(RuntimeError, r"only supported on CUDA devices"):
            wp.load_module(cpu_kernel.module, device="cpu")

    _HOST_RECORD: ClassVar[dict] = {
        "path": "/opt/nvshmem/lib/libnvshmem_device.ltoir.fatbin",
        "kind": "fatbin",
        "source": "host",
        "version": None,
        "sha256": "0" * 64,
    }
    _OVERRIDE_RECORD: ClassVar[dict] = {**_HOST_RECORD, "source": "override"}
    _LOADED_3_8_0 = (3 << 16) | (8 << 8)

    @classmethod
    def _fake_runtime(cls, init_result, **overrides):
        core = {
            "wp_is_nvshmem_enabled": 1,
            "wp_nvshmem_get_loaded_version": cls._LOADED_3_8_0,
            "wp_nvshmem_cumodule_init": init_result,
            "wp_nvshmem_cumodule_finalize": 0,
        } | overrides
        fake_core = SimpleNamespace(**{key: mock.Mock(return_value=value) for key, value in core.items()})
        return fake_core, mock.patch.object(warp_context, "runtime", SimpleNamespace(core=fake_core))

    def test_nvshmem_runtime_errors(self):
        """Loading an NVSHMEM module should explain why the NVSHMEM runtime is unusable."""
        cases = (
            ("not enabled", {"wp_is_nvshmem_enabled": 0}, self._HOST_RECORD, None, r"Linux CUDA builds"),
            ("not loaded", {"wp_nvshmem_get_loaded_version": 0}, self._HOST_RECORD, None, r"no NVSHMEM host library"),
            (
                "too old",
                {},
                self._HOST_RECORD,
                None,
                r"requires NVSHMEM 3\.9 or newer, but the loaded host library is NVSHMEM 3\.8\.0",
            ),
            ("not initialized", {}, self._OVERRIDE_RECORD, -1, r"NVSHMEM is not initialized"),
            ("init failure", {}, self._OVERRIDE_RECORD, 7, r"nvshmemx_cumodule_init failed \(7\)"),
        )
        for name, overrides, record, init_result, message in cases:
            fake_core, patched = self._fake_runtime(init_result, **overrides)
            with self.subTest(name), patched, self.assertRaisesRegex(RuntimeError, message):
                warp_context._check_nvshmem_host_runtime(record)
                warp_context._init_nvshmem_module(123)
            # Only a failure reported by NVSHMEM itself may have registered the module.
            if init_result is not None and init_result > 0:
                fake_core.wp_nvshmem_cumodule_finalize.assert_called_once_with(123)
            else:
                fake_core.wp_nvshmem_cumodule_finalize.assert_not_called()

    def test_override_skips_host_version_floor(self):
        """An explicit device library override is trusted with a host library older than 3.9."""
        _, patched = self._fake_runtime(0)
        with patched:
            warp_context._check_nvshmem_host_runtime(self._OVERRIDE_RECORD)
            warp_context._init_nvshmem_module(123)

    def test_nvshmem_kernel_compiles(self):
        """A kernel using NVSHMEM builtins should compile without nvshmem4py or an NVSHMEM runtime."""
        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")
        if not _is_nvshmem_device_library_available():
            self.skipTest("NVSHMEM device library not embedded in this build.")

        @wp.kernel(module="unique")
        def nvshmem_kernel(out_pe: wp.array[wp.int32], out_npes: wp.array[wp.int32]):
            out_pe[0] = wp.nvshmem_my_pe()
            out_npes[0] = wp.nvshmem_n_pes()

        with tempfile.TemporaryDirectory(prefix="wp_nvshmem_aot_") as module_dir:
            artifacts = wp.compile_aot_module(nvshmem_kernel.module, device="cuda:0", module_dir=module_dir)
            self.assertTrue(artifacts)
            self.assertTrue(all(os.path.isfile(path) for path in artifacts))

    def test_loading_without_nvshmem_runtime_raises(self):
        """Loading an NVSHMEM module without an NVSHMEM runtime should raise, and retrying should raise again."""
        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")
        if not _is_nvshmem_device_library_available():
            self.skipTest("NVSHMEM device library not embedded in this build.")
        if warp_context.runtime.core.wp_nvshmem_get_loaded_version() != 0:
            self.skipTest("An NVSHMEM host library is already loaded in this process.")

        @wp.kernel(module="unique")
        def nvshmem_kernel(out_pe: wp.array[wp.int32]):
            out_pe[0] = wp.nvshmem_my_pe()

        for _attempt in range(2):
            with self.assertRaisesRegex(RuntimeError, r"Initialize NVSHMEM before loading the module"):
                wp.load_module(nvshmem_kernel.module, device="cuda:0")


class TestNVSHMEMDeviceLibrary(unittest.TestCase):
    """Tests for locating the NVSHMEM device library that Warp links into kernels."""

    def setUp(self):
        self.support = nvshmem_support
        self.tmp = tempfile.TemporaryDirectory(prefix="wp_nvshmem_lib_")
        self.addCleanup(self.tmp.cleanup)
        self.root = self.tmp.name
        self.env = mock.patch.dict(os.environ)
        self.env.start()
        self.addCleanup(self.env.stop)
        os.environ.pop(nvshmem_support.DEVICE_LIBRARY_ENV, None)

    def _write(self, relative_path, data=b"device library bytes"):
        path = os.path.join(self.root, relative_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        return path

    def _fake_distribution(self, name, version, with_library=True):
        """Lay out an installed wheel: a dist-info directory and the device library at the wheel's lib path."""
        site = os.path.join(self.root, "site")
        dist_info = os.path.join(site, f"{name.replace('-', '_')}-{version}.dist-info")
        os.makedirs(dist_info)
        with open(os.path.join(dist_info, "METADATA"), "w") as f:
            f.write(f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n")
        if with_library:
            self._write(os.path.join("site", "nvidia", "nvshmem", "lib", self.support.DEVICE_LIBRARY_FILENAME))
        return site

    def test_override_selects_file_and_kind(self):
        """The override environment variable wins and the extension picks the nvJitLink input kind."""
        for filename, kind in (("x.ltoir.fatbin", "fatbin"), ("y.ltoir", "ltoir")):
            with self.subTest(filename=filename):
                path = self._write(filename)
                os.environ[self.support.DEVICE_LIBRARY_ENV] = path
                lib = self.support.find_device_library(None, 13, search_path=[])
                self.assertEqual((lib.path, lib.kind, lib.source), (path, kind, "override"))

    def test_override_missing_file_raises(self):
        """An override that names a missing file fails with the path in the message."""
        path = os.path.join(self.root, "missing.fatbin")
        os.environ[self.support.DEVICE_LIBRARY_ENV] = path
        with self.assertRaisesRegex(RuntimeError, re.escape(path)):
            self.support.find_device_library(None, 13, search_path=[])

    def test_host_directory_lookup(self):
        """The device library next to the loaded host library is used when no override is set."""
        path = self._write(self.support.DEVICE_LIBRARY_FILENAME)
        lib = self.support.find_device_library(self.root, 13, search_path=[])
        self.assertEqual((lib.path, lib.kind, lib.source), (path, "fatbin", "host"))

    def test_distribution_lookup(self):
        """Without a loaded host library, the installed nvidia-nvshmem wheel for the CUDA major version is used."""
        site = self._fake_distribution("nvidia-nvshmem-cu13", "3.9.0")
        lib = self.support.find_device_library(None, 13, search_path=[site])
        expected = os.path.join(site, "nvidia", "nvshmem", "lib", self.support.DEVICE_LIBRARY_FILENAME)
        self.assertEqual((lib.path, lib.kind, lib.source, lib.version), (expected, "fatbin", "distribution", "3.9.0"))

    def test_no_device_library_raises(self):
        """The error names every location tried and the NVSHMEM version requirement."""
        site = self._fake_distribution("nvidia-nvshmem-cu13", "3.8.0", with_library=False)
        host_dir = os.path.join(self.root, "host")
        os.makedirs(host_dir)
        with self.assertRaisesRegex(RuntimeError, r"NVSHMEM 3\.9 or newer") as raised:
            self.support.find_device_library(host_dir, 13, search_path=[site])
        self.assertIn(os.path.join(host_dir, self.support.DEVICE_LIBRARY_FILENAME), str(raised.exception))
        self.assertIn("nvidia-nvshmem-cu13", str(raised.exception))

    def test_record_and_hash(self):
        """Reading the library returns its bytes and digest, and the record carries the identity fields."""
        data = b"\x7fNCE" + bytes(range(64))
        path = self._write("z.ltoir", data)
        os.environ[self.support.DEVICE_LIBRARY_ENV] = path
        lib = self.support.find_device_library(None, 13, search_path=[])
        contents, digest = self.support.read_device_library(lib)
        self.assertEqual(contents, data)
        self.assertEqual(digest, hashlib.sha256(data).hexdigest())
        record = self.support.device_library_record(lib, digest)
        self.assertEqual(set(record), {"path", "kind", "source", "version", "sha256"})
        self.assertEqual((record["path"], record["kind"], record["sha256"]), (path, "ltoir", digest))


def _skip_unless_distributed(test_case):
    """Skip ``test_case`` when ``mpirun``, NVSHMEM support, or nvshmem4py and mpi4py are unavailable.

    Returns:
        The path of ``mpirun``.
    """
    mpirun = shutil.which("mpirun")
    if mpirun is None:
        test_case.skipTest("mpirun not found in PATH")

    if not wp.is_nvshmem_enabled():
        test_case.skipTest("NVSHMEM not enabled in this build.")
    if not _is_nvshmem_device_library_available():
        test_case.skipTest("NVSHMEM device library not embedded in this build.")
    if not _is_module_installed("nvshmem") or not _is_module_installed("mpi4py"):
        test_case.skipTest("nvshmem4py and mpi4py are required.")
    return mpirun


def _run_mpirun(test_case, command, env, timeout):
    """Run ``command`` and return its standard output, failing ``test_case`` on a nonzero exit status."""
    result = subprocess.run(command, capture_output=True, text=True, timeout=timeout, check=False, env=env)
    if result.returncode != 0:
        test_case.fail(
            f"mpirun failed with return code {result.returncode}\nstdout: {result.stdout}\nstderr: {result.stderr}"
        )
    return result.stdout


def run_distributed_script(test_case, script_name, n_pes=2, env=None, timeout=120):
    """Run a script from ``warp/tests/distributed/`` through ``mpirun`` and return its standard output.

    Skips ``test_case`` when ``mpirun``, NVSHMEM support, or nvshmem4py and mpi4py are unavailable, and
    fails it when ``mpirun`` exits with a nonzero status.

    Args:
        test_case: The :class:`unittest.TestCase` to skip or fail.
        script_name: File name of the script in ``warp/tests/distributed/``.
        n_pes: Number of processing elements to launch.
        env: Environment for ``mpirun``. Inherits the current environment when ``None``.
        timeout: Seconds to wait for ``mpirun`` to finish.
    """
    mpirun = _skip_unless_distributed(test_case)
    script = os.path.join(_DISTRIBUTED_DIR, script_name)
    return _run_mpirun(test_case, [mpirun, "-np", str(n_pes), sys.executable, script], env, timeout)


def run_distributed_module(test_case, module_name, args=(), n_pes=2, env=None, timeout=120):
    """Run a Python module through ``mpirun`` with ``python -m`` and return its standard output.

    Skips ``test_case`` when ``mpirun``, NVSHMEM support, or nvshmem4py and mpi4py are unavailable, and
    fails it when ``mpirun`` exits with a nonzero status.

    Args:
        test_case: The :class:`unittest.TestCase` to skip or fail.
        module_name: Dotted name of the module, such as ``warp.examples.distributed.example_nvshmem_basic``.
        args: Command-line arguments for the module.
        n_pes: Number of processing elements to launch.
        env: Environment for ``mpirun``. Inherits the current environment when ``None``.
        timeout: Seconds to wait for ``mpirun`` to finish.
    """
    mpirun = _skip_unless_distributed(test_case)
    command = [mpirun, "-np", str(n_pes), sys.executable, "-m", module_name, *args]
    return _run_mpirun(test_case, command, env, timeout)


class TestNVSHMEMMultiPE(unittest.TestCase):
    """Multi-PE tests launched in subprocesses."""

    def _run_distributed(self, script_name, n_pes=2, env=None):
        """Run a script from warp/tests/distributed/ via mpirun."""
        return run_distributed_script(self, script_name, n_pes=n_pes, env=env)

    def _run_uid_distributed(self, script_name, n_pes=2):
        """Run a local P2P test through NVSHMEM's Hydra launcher and UID bootstrap."""
        if not wp.is_nvshmem_enabled():
            self.skipTest("NVSHMEM not enabled in this build.")
        if not _is_nvshmem_device_library_available():
            self.skipTest("NVSHMEM device library not embedded in this build.")
        if wp.get_cuda_device_count() < n_pes:
            self.skipTest(f"Test requires {n_pes} CUDA devices.")
        if not _is_module_installed("nvshmem"):
            self.skipTest("nvshmem4py is not installed.")

        nvshmem_home = os.environ.get("NVSHMEM_HOME")
        launcher = None
        if nvshmem_home:
            candidate = os.path.join(nvshmem_home, "bin", "nvshmrun.hydra")
            if os.path.isfile(candidate):
                launcher = candidate
        if launcher is None:
            launcher = shutil.which("nvshmrun.hydra") or shutil.which("nvshmrun")
        if launcher is None:
            self.skipTest("NVSHMEM Hydra launcher not found.")

        script = os.path.join(_DISTRIBUTED_DIR, script_name)
        with tempfile.TemporaryDirectory(prefix="wp_nvshmem_uid_") as temp_dir:
            env = os.environ.copy()
            env.update(
                {
                    "NVSHMEM_BOOTSTRAP": "UID",
                    "NVSHMEM_REMOTE_TRANSPORT": "none",
                    "NVSHMEM_DISABLE_LOCAL_ONLY_PROXY": "1",
                    "NVSHMEM_DISABLE_NCCL": "1",
                    "NVSHMEM_DISABLE_NVLS": "1",
                    "NVSHMEM_SYMMETRIC_SIZE": "67108864",
                    "NVSHMEM_TEST_CACHE_ROOT": os.path.join(temp_dir, "cache"),
                    "NVSHMEM_TEST_UID_FILE": os.path.join(temp_dir, "uid"),
                }
            )
            if nvshmem_home:
                nvshmem_lib_path = os.path.join(nvshmem_home, "lib")
                env["LD_LIBRARY_PATH"] = os.pathsep.join(
                    path for path in (nvshmem_lib_path, env.get("LD_LIBRARY_PATH")) if path
                )

            result = subprocess.run(
                [launcher, "-launcher", "fork", "-prepend-rank", "-n", str(n_pes), sys.executable, script],
                capture_output=True,
                text=True,
                timeout=120,
                check=False,
                env=env,
            )
        if result.returncode != 0:
            self.fail(
                f"NVSHMEM UID launch failed with return code {result.returncode}\n"
                f"stdout: {result.stdout}\nstderr: {result.stderr}"
            )
        return result.stdout

    def test_cached_module_in_new_process(self):
        """A module loaded from the kernel cache in a new process should get initialized NVSHMEM state."""
        with tempfile.TemporaryDirectory(prefix="wp_nvshmem_cache_") as cache_dir:
            env = os.environ | {"WARP_CACHE_PATH": cache_dir}
            first_run = self._run_distributed("aux_nvshmem_cache.py", n_pes=1, env=env)
            second_run = self._run_distributed("aux_nvshmem_cache.py", n_pes=1, env=env)

        self.assertIn("(compiled)", first_run)
        self.assertIn("(cached)", second_run)
        for output in (first_run, second_run):
            self.assertIn("PE 0 query: [0, 1]", output)

    def test_device_library_change_invalidates_cache(self):
        """A cached module linked against one device library is recompiled when the library changes."""
        override = os.environ.get(nvshmem_support.DEVICE_LIBRARY_ENV)
        if not override:
            self.skipTest(f"{nvshmem_support.DEVICE_LIBRARY_ENV} is not set.")
        # NVSHMEM ships the multi-architecture fatbin and per-architecture raw LTOIR files side by side.
        # Pick the newest raw LTOIR that the test device can still link.
        arch = wp.get_device("cuda:0").arch
        candidates = sorted(
            (int(m.group(1)), path)
            for path in glob.glob(os.path.join(os.path.dirname(override), "libnvshmem_device_sm_*.ltoir"))
            if (m := re.search(r"sm_(\d+)\.ltoir$", path)) and int(m.group(1)) <= arch
        )
        if not candidates:
            self.skipTest("No per-architecture raw LTOIR next to the override to switch to.")
        alternate = candidates[-1][1]

        with tempfile.TemporaryDirectory(prefix="wp_nvshmem_cache_") as cache_dir:
            copied = shutil.copy(override, os.path.join(cache_dir, "copy.fatbin"))
            runs = ((override, "(compiled)"), (override, "(cached)"), (copied, "(cached)"), (alternate, "(compiled)"))
            for library, expected in runs:
                env = os.environ | {"WARP_CACHE_PATH": cache_dir, nvshmem_support.DEVICE_LIBRARY_ENV: library}
                output = self._run_distributed("aux_nvshmem_cache.py", n_pes=1, env=env)
                with self.subTest(library=os.path.basename(library), expected=expected):
                    self.assertIn(expected, output)
                    self.assertIn("PE 0 query: [0, 1]", output)
                    # The cached binary must now be the one linked against this library's bytes.
                    (meta_path,) = glob.glob(os.path.join(cache_dir, "*", "wp___main___*", "*.meta"))
                    with open(meta_path) as meta_file:
                        record = json.load(meta_file)["nvshmem"]
                    with open(library, "rb") as library_file:
                        self.assertEqual(record["sha256"], hashlib.sha256(library_file.read()).hexdigest())

    def test_host_too_old_without_override(self):
        """A host library older than 3.9 with no override fails at compile, naming the requirement."""
        env = {key: value for key, value in os.environ.items() if key != nvshmem_support.DEVICE_LIBRARY_ENV}
        output = self._run_distributed("aux_nvshmem_too_old.py", n_pes=1, env=env)
        if "SKIPPED" in output:
            self.skipTest(output.strip().splitlines()[-1])
        self.assertIn("PASSED", output)

    def test_reload_after_unload(self):
        """A module loaded after another NVSHMEM module is unloaded should get initialized NVSHMEM state."""
        output = self._run_distributed("aux_nvshmem_reload.py", n_pes=1)
        self.assertIn("PE 0: reload after unload: PASSED", output)

    def test_pe_query(self):
        """Verify my_pe and n_pes return correct values."""
        output = self._run_distributed("aux_nvshmem_pe_query.py")
        self.assertIn("PASSED", output)

    def test_rma(self):
        """Verify NVSHMEM element, bulk, and signal operations between two PEs."""
        output = self._run_distributed("aux_nvshmem_rma.py")
        for check in (
            "PE 1: p into strided view",
            "PE 1: p with 2D index",
            "PE 1: p of vec3",
            "PE 1: p of float16",
            "PE 0: g of int64 and vec3",
            "PE 1: contiguous put",
            "PE 1: strided put",
            "PE 1: 2D non-contiguous put",
            "PE 0: strided get",
            "PE 1: put_signal, signal_op, and signal_wait_until",
        ):
            self.assertIn(f"{check}: PASSED", output)

    def test_uid_local_p2p(self):
        """Verify UID bootstrap and device-side P2P with remote transports disabled."""
        output = self._run_uid_distributed("aux_nvshmem_uid_p2p.py")
        self.assertIn("PE 0/2 on cuda:0: query=(0, 2)", output)
        self.assertIn("PE 1/2 on cuda:1: query=(1, 2), symmetric_buffer=42.0, remote_transport=none: PASSED", output)


if __name__ == "__main__":
    unittest.main(verbosity=2)
