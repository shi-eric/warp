# ruff: noqa: PLC0415

import contextlib
import io
import json
import os
import pathlib
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from .support import temporary_fixture_cases


class TestRunnerDiagnostics(unittest.TestCase):
    def test_help_names_warp_tests_entry_point(self):
        from warp._src.test_runner.runner import _create_argument_parser

        usage = _create_argument_parser().format_usage()

        self.assertTrue(usage.startswith("usage: warp.tests "), usage)

    def test_environment_configuration_prefers_cli_and_ignores_blank_values(self):
        from warp._src.test_runner.artifacts import resolve_diagnostics_root

        with mock.patch.dict(
            os.environ,
            {"WARP_TEST_DIAGNOSTICS_DIR": "from-environment"},
            clear=False,
        ):
            self.assertEqual(resolve_diagnostics_root("from-cli"), pathlib.Path("from-cli"))
            self.assertEqual(
                resolve_diagnostics_root(None),
                pathlib.Path("from-environment"),
            )
            self.assertIsNone(resolve_diagnostics_root("   "))

    @staticmethod
    def _timing(**overrides):
        from warp._src.test_runner.common import SuiteTiming

        values = {
            "suite_index": 0,
            "suite_name": "fixture.TestTiming",
            "unit_type": "class",
            "test_count": 1,
            "worker_index": 0,
            "pid": 4818,
            "started_offset_seconds": 1.0,
            "finished_offset_seconds": 2.0,
            "elapsed_seconds": 1.0,
            "completion_order": 1,
            "status": "complete",
            "outcomes": {"OK": 1},
        }
        values.update(overrides)
        return SuiteTiming(**values)

    def test_quiet_run_suppresses_ordinary_worker_lifecycle_output(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.runner import _make_worker_event_callback

        events = (
            WorkerEvent(
                sequence=1,
                event=EventKind.TEST_STARTED,
                worker_index=3,
                pid=4818,
                monotonic_ns=93_000_000_000,
                wall_time_ns=99_000_000_000,
                suite_index=9,
                suite_name="fixture.TestSlow",
                test_id="fixture.TestSlow.test_waits",
            ),
            WorkerEvent(
                sequence=2,
                event=EventKind.SUITE_FINISHED,
                worker_index=3,
                pid=4818,
                monotonic_ns=93_427_000_000,
                wall_time_ns=100_000_000_000,
                suite_index=9,
                suite_name="fixture.TestSlow",
                elapsed_seconds=93.427,
                test_count=184,
            ),
        )
        output = io.StringIO()

        with mock.patch("sys.stderr", output):
            callback = _make_worker_event_callback(SimpleNamespace(verbose=0))
            for event in events:
                callback(event)

        self.assertEqual(output.getvalue(), "")

    def test_quiet_run_suppresses_final_timing_summary(self):
        from warp._src.test_runner.runner import _RunDiagnostics

        args = SimpleNamespace(
            coverage=False,
            coverage_branch=False,
            level="class",
            pattern=None,
            suite="autodetect",
            testNamePatterns=None,
            verbose=0,
            warp_debug=False,
        )
        tracker = mock.Mock()
        tracker.suite_timings.return_value = (self._timing(),)
        diagnostics = _RunDiagnostics()
        diagnostics.configure(args, 1, 1, 1, None)
        diagnostics.tracker = tracker
        output = io.StringIO()

        with mock.patch("sys.stderr", output):
            diagnostics.finalize()

        self.assertEqual(output.getvalue(), "")
        tracker.suite_timings.assert_called_once_with((), unit_type="class")

    def test_verbose_test_lifecycle_prefix_includes_worker_and_pid(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.runner import _make_worker_event_callback

        event = WorkerEvent(
            sequence=2,
            event=EventKind.TEST_STARTED,
            worker_index=3,
            pid=4818,
            monotonic_ns=10,
            wall_time_ns=20,
            suite_index=9,
            suite_name="fixture.TestSlow",
            test_id="fixture.TestSlow.test_waits",
        )
        output = io.StringIO()

        with mock.patch("sys.stderr", output):
            _make_worker_event_callback(SimpleNamespace(verbose=2))(event)

        self.assertEqual(output.getvalue(), "[worker 3 pid=4818] fixture.TestSlow.test_waits ...\n")

    def test_cleanup_exception_after_clean_body_retains_evidence(self):
        from warp._src.test_runner.runner import main

        class CleanupFailure(RuntimeError):
            pass

        with temporary_fixture_cases() as (module, _, root):
            diagnostics_root = root / "diagnostics"
            coverage_root = root / "coverage"
            coverage_root.mkdir()
            suite = unittest.defaultTestLoader.loadTestsFromTestCase(module.PassingCase)

            class RaisingTemporaryDirectory:
                def __enter__(self):
                    return str(coverage_root)

                def __exit__(self, exception_type, exception, traceback):
                    raise CleanupFailure("temporary directory cleanup failed")

            argv = [
                "-s",
                "autodetect",
                "-j",
                "1",
                "--maxjobs",
                "1",
                "-q",
                "--diagnostics-dir",
                str(diagnostics_root),
            ]
            with mock.patch("warp.tests.unittest_suites.auto_discover_suite", return_value=suite):
                with mock.patch(
                    "warp._src.test_runner.runner.tempfile.TemporaryDirectory",
                    RaisingTemporaryDirectory,
                ):
                    with self.assertRaisesRegex(CleanupFailure, "temporary directory cleanup failed"):
                        main(argv)

            run_dir = next(diagnostics_root.glob("run-*"))
            json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
            json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertTrue(list(run_dir.glob("worker-*.events.jsonl")))
            self.assertFalse(any(thread.name == "worker-event-monitor" for thread in threading.enumerate()))

    def test_missing_coverage_preflight_writes_compact_diagnostics(self):
        from warp._src.test_runner.runner import main

        with tempfile.TemporaryDirectory() as directory:
            diagnostics_root = pathlib.Path(directory, "diagnostics")
            stderr = io.StringIO()
            argv = ["--coverage", "--diagnostics-dir", str(diagnostics_root)]

            with mock.patch("warp._src.test_runner.runner.COVERAGE_AVAILABLE", False):
                with contextlib.redirect_stderr(stderr):
                    with self.assertRaises(SystemExit) as raised:
                        main(argv)

            self.assertEqual(raised.exception.code, 2)
            self.assertIn("coverage was not found", stderr.getvalue())
            run_dir = next(diagnostics_root.glob("run-*"))
            run_metadata = json.loads(run_dir.joinpath("run.json").read_text(encoding="utf-8"))
            timings = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertTrue(run_metadata["coverage"])
            self.assertEqual(timings["suites"], [])
            self.assertFalse(any(thread.name == "worker-event-monitor" for thread in threading.enumerate()))
