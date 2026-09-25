# ruff: noqa: PLC0415

import contextlib
import io
import pathlib
import signal
import tempfile
import unittest
from unittest import mock


class TestPostmortemDiagnostics(unittest.TestCase):
    def test_nonempty_fault_log_is_fatal_evidence(self):
        from warp._src.test_runner import postmortem as diagnostics

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            fault_path = run_dir / "worker-2-pid-8123.fault.log"
            cases = (
                b"Fatal Python error: Segmentation fault\n" + b"x" * (33 * 1024),
                b"Windows fatal exception: access violation\r\n",
            )
            for contents in cases:
                with self.subTest(header=contents[:32]):
                    fault_path.write_bytes(contents)
                    with mock.patch.object(
                        diagnostics,
                        "_read_artifact_tail",
                        side_effect=AssertionError("fault evidence must not scan the console tail"),
                    ):
                        evidence = diagnostics._artifact_evidence(run_dir, 2, 8123)

                    self.assertEqual(evidence["fault"]["state"], "non_empty")
                    self.assertIs(evidence["fault"]["fatal_traceback_evidence"], True)

            fault_path.write_bytes(cases[0])
            bounded_tail = diagnostics._read_artifact_tail(fault_path)
            self.assertEqual(len(bounded_tail.encode("utf-8")), 32 * 1024)
            self.assertNotIn("Fatal Python error", bounded_tail)

            fault_path.write_bytes(b"")
            evidence = diagnostics._artifact_evidence(run_dir, 2, 8123)
            self.assertEqual(evidence["fault"]["state"], "empty")
            self.assertIs(evidence["fault"]["fatal_traceback_evidence"], False)

            # Worker sinks open before WORKER_STARTED is emitted, so the PID
            # must still recover artifacts when no worker index was recorded.
            evidence = diagnostics._artifact_evidence(run_dir, None, 8123)
            self.assertEqual(evidence["fault"]["path"], str(fault_path))
            self.assertEqual(evidence["fault"]["state"], "empty")

            fault_path.unlink()
            evidence = diagnostics._artifact_evidence(run_dir, 2, 8123)
            self.assertEqual(evidence["fault"]["state"], "missing")
            self.assertIsNotNone(evidence["fault"]["path"])
            self.assertIsNone(evidence["fault"]["fatal_traceback_evidence"])

            evidence = diagnostics._artifact_evidence(None, 2, 8123)
            self.assertIsNone(evidence["fault"]["path"])
            self.assertIsNone(evidence["fault"]["fatal_traceback_evidence"])

    def test_pool_report_prioritizes_abnormal_workers_and_compacts_parent_exits(self):
        from warp._src.test_runner.common import EventKind, ProcessExit, ProcessExitProvenance, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker
        from warp._src.test_runner.postmortem import (
            build_crash_snapshot,
            format_pool_failure,
            print_pool_failure_evidence,
        )

        tracker = WorkerStateTracker(history_limit=3)

        def event(worker_index, pid, kind, sequence, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=worker_index,
                pid=pid,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=worker_index,
                suite_name=f"fixture.Worker{worker_index}",
                **fields,
            )

        tracker.handle_event(event(0, 7001, EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(0, 7001, EventKind.SUITE_STARTED, 2, gil_enabled=False))
        tracker.handle_event(event(1, 7002, EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(1, 7002, EventKind.WORKER_INITIALIZED, 2, gil_enabled=False))
        tracker.handle_event(event(1, 7002, EventKind.SUITE_STARTED, 3, gil_enabled=False))
        tracker.handle_event(
            event(
                1,
                7002,
                EventKind.TEST_OUTCOME,
                4,
                test_id="fixture.Worker1.test_failure",
                outcome="ERROR",
                elapsed_seconds=0.75,
            )
        )
        tracker.handle_event(
            event(
                1,
                7002,
                EventKind.TEST_STOPPED,
                5,
                test_id="fixture.Worker1.test_failure",
            )
        )
        tracker.handle_event(
            event(
                1,
                7002,
                EventKind.GIL_STATE_CHANGED,
                6,
                previous_gil_enabled=False,
                gil_enabled=True,
                observed_at=EventKind.SUITE_FINISHED.value,
            )
        )
        tracker.handle_event(event(1, 7002, EventKind.SUITE_FINISHED, 7, gil_enabled=True))
        tracker.handle_event(event(2, 7003, EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(2, 7003, EventKind.SUITE_STARTED, 2, gil_enabled=False))
        tracker.handle_event(
            event(
                2,
                7003,
                EventKind.TEST_STARTED,
                3,
                test_id="fixture.Worker2.test_candidate",
            )
        )

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            for worker_index, pid, marker in (
                (0, 7001, "parent-tail-marker"),
                (1, 7002, "abnormal-tail-marker"),
                (2, 7003, "unresolved-tail-marker"),
            ):
                run_dir.joinpath(f"worker-{worker_index}-pid-{pid}.output.log").write_text(
                    marker,
                    encoding="utf-8",
                )

            snapshot = build_crash_snapshot(
                RuntimeError("pool failed"),
                (),
                tracker.snapshots(now_ns=8_000_000_000),
                (
                    ProcessExit(7001, -signal.SIGTERM, "SIGTERM", ProcessExitProvenance.PARENT_TERMINATED.value),
                    ProcessExit(7002, 86, None, ProcessExitProvenance.INDEPENDENTLY_ABNORMAL.value),
                    ProcessExit(7003, -signal.SIGTERM, "SIGTERM", ProcessExitProvenance.UNRESOLVED.value),
                ),
                run_dir,
            )

            summary = format_pool_failure(snapshot)
            self.assertIn("parent-terminated workers (1)", summary)
            self.assertIn("worker 0 (PID 7001, SIGTERM, final GIL state=disabled)", summary)
            self.assertLess(
                summary.index("exit provenance=independently_abnormal"),
                summary.index("parent-terminated workers (1)"),
            )
            self.assertIn(
                "last non-OK test=fixture.Worker1.test_failure outcome=ERROR duration=0.750s",
                summary,
            )
            self.assertIn("final GIL state=enabled", summary)
            self.assertIn(
                "first observed GIL change=disabled->enabled at suite_finished",
                summary,
            )
            self.assertIn(
                "note: SIGTERM/SIGKILL exits recorded before parent cleanup are usually pool "
                "cleanup racing shutdown; check journal position and fault logs to confirm.",
                summary,
            )

            console = io.StringIO()
            with contextlib.redirect_stderr(console):
                print_pool_failure_evidence(snapshot)
            output = console.getvalue()
            self.assertIn("abnormal-tail-marker", output)
            self.assertIn("unresolved-tail-marker", output)
            self.assertNotIn("parent-tail-marker", output)
            self.assertLess(output.index("abnormal-tail-marker"), output.index("unresolved-tail-marker"))
