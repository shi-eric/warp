# ruff: noqa: PLC0415

import concurrent.futures
import contextlib
import io
import json
import multiprocessing
import queue as queue_module
import sys
import tempfile
import threading
import time
import unittest
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace
from unittest import mock

from .support import (
    LocalManager,
    NonCancellingFuture,
    PublishingManagerThread,
    StubExecutor,
    temporary_fixture_cases,
)


class TestParallelCrashIntegration(unittest.TestCase):
    def test_runner_main_uses_warp_owned_module(self):
        from warp._src.test_runner.runner import main

        self.assertEqual(main.__module__, "warp._src.test_runner.runner")

    @staticmethod
    def _args(**overrides):
        values = {
            "buffer": False,
            "coverage": False,
            "coverage_branch": False,
            "failfast": False,
            "junit_report_xml": "enabled",
            "level": "class",
            "no_shared_cache": False,
            "verbose": 0,
            "warp_debug": False,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    @staticmethod
    def _suites(module, *case_names):
        loader = unittest.TestLoader()
        return [loader.loadTestsFromTestCase(getattr(module, case_name)) for case_name in case_names]

    @staticmethod
    def _confirmed_result(class_name):
        return (
            1,
            [],
            [],
            0,
            0,
            0,
            [(class_name, "test_confirmed", 0.0, "OK", None, None)],
        )

    def _run_stub_executor(self, executor, suites):
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.pool import run_parallel_suites

        event_queue = queue_module.Queue()
        tracker = WorkerStateTracker()
        monitor = WorkerEventMonitor(event_queue, tracker)
        monitor.start()
        try:
            with mock.patch(
                "warp._src.test_runner.pool.concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ):
                return run_parallel_suites(
                    suites,
                    1,
                    LocalManager(),
                    self._args(),
                    tempfile.gettempdir(),
                    event_queue,
                    tracker,
                    monitor,
                    None,
                    time.monotonic_ns(),
                )
        finally:
            monitor.stop_and_drain()

    def test_monitor_drains_events_before_stopping(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.pool import run_parallel_suites

        event_queue = queue_module.Queue()
        tracker = WorkerStateTracker(history_limit=3)
        handled_sequences = []
        monitor = WorkerEventMonitor(
            event_queue,
            tracker,
            on_event=lambda event: handled_sequences.append(event.sequence),
        )
        monitor.start()
        for index in range(64):
            event_queue.put(
                WorkerEvent(
                    sequence=index,
                    event=EventKind.WORKER_STARTED,
                    worker_index=index,
                    pid=1234 + index,
                    monotonic_ns=100 + index,
                    wall_time_ns=1000 + index,
                )
            )

        completed = concurrent.futures.Future()
        completed.set_result(self._confirmed_result("PassingCase"))
        executor = StubExecutor([completed])

        with temporary_fixture_cases() as (module, _, _):
            with mock.patch(
                "warp._src.test_runner.pool.concurrent.futures.ProcessPoolExecutor",
                return_value=executor,
            ):
                result = run_parallel_suites(
                    self._suites(module, "PassingCase"),
                    1,
                    LocalManager(),
                    self._args(),
                    tempfile.gettempdir(),
                    event_queue,
                    tracker,
                    monitor,
                    None,
                    time.monotonic_ns(),
                )

        # `run_parallel_suites` drains the monitor itself on the success path (no
        # separate barrier round-trip needed), so every event queued before the
        # call is reflected in tracker state by the time it returns.
        self.assertIsNone(result.pool_failure)
        self.assertFalse(result.diagnostics_degraded)
        self.assertEqual(handled_sequences, list(range(64)))
        self.assertEqual(
            [snapshot.pid for snapshot in tracker.snapshots(now_ns=1000)],
            list(range(1234, 1298)),
        )
        self.assertIsNone(monitor.error)
        self.assertTrue(monitor.stopped)

    def test_salvages_accepted_future_when_later_submit_fails(self):
        from concurrent.futures.process import BrokenProcessPool

        with temporary_fixture_cases() as (module, _, _):
            completed = concurrent.futures.Future()
            completed.set_result(self._confirmed_result("PassingCase"))
            executor = StubExecutor(
                [
                    completed,
                    BrokenProcessPool("later submission failed"),
                ]
            )

            result = self._run_stub_executor(
                executor,
                self._suites(module, "PassingCase", "FirstPidCase"),
            )

            self.assertIsNotNone(result.pool_failure)
            self.assertEqual(set(result.results_by_index), {0})
            self.assertEqual(result.results_by_index[0][6][0][0], "PassingCase")
            self.assertEqual(result.pool_failure.exception_type, "BrokenProcessPool")
            self.assertEqual(result.pool_failure.reason, "later submission failed")
            self.assertEqual(result.pool_failure.snapshot["failure"]["type"], "BrokenProcessPool")
            self.assertEqual(result.pool_failure.snapshot["failure"]["reason"], "later submission failed")

    def test_salvages_result_published_while_failed_executor_quiesces(self):
        from concurrent.futures.process import BrokenProcessPool

        with temporary_fixture_cases() as (module, _, _):
            broken = concurrent.futures.Future()
            broken.set_exception(BrokenProcessPool("worker exited"))
            published_during_shutdown = NonCancellingFuture()
            published_result = self._confirmed_result("FirstPidCase")
            manager_thread = PublishingManagerThread(published_during_shutdown, published_result)
            executor = StubExecutor([broken, published_during_shutdown], manager_thread=manager_thread)

            result = self._run_stub_executor(
                executor,
                self._suites(module, "PassingCase", "FirstPidCase"),
            )

            self.assertTrue(manager_thread.joined)
            self.assertIsNotNone(result.pool_failure)
            self.assertEqual(set(result.results_by_index), {1})
            self.assertEqual(result.results_by_index[1][6][0][0], "FirstPidCase")

    def test_classification_reports_started_without_result(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker
        from warp._src.test_runner.postmortem import classify_suites

        with temporary_fixture_cases() as (module, _, _):
            suites = self._suites(
                module,
                "BlockingCase",
                "HardExitCase",
                "PassingCase",
                "OrdinaryFailureCase",
                "ShouldNotStartCase",
            )
            tracker = WorkerStateTracker()
            for worker_index, pid, suite_index, suite_name in (
                (0, 7001, 0, "BlockingCase"),
                (1, 7002, 1, "HardExitCase"),
                (2, 7003, 2, "PassingCase"),
                (3, 7004, 3, "OrdinaryFailureCase"),
                (4, 7005, 4, "ShouldNotStartCase"),
            ):
                tracker.handle_event(
                    WorkerEvent(
                        sequence=1,
                        event=EventKind.SUITE_STARTED,
                        worker_index=worker_index,
                        pid=pid,
                        monotonic_ns=1,
                        wall_time_ns=1,
                        suite_index=suite_index,
                        suite_name=suite_name,
                        test_count=1,
                    )
                )
            snapshots = tracker.snapshots(now_ns=2)

            classifications = classify_suites(
                suites,
                {},
                dict.fromkeys(range(len(suites)), "unresolved"),
                snapshots,
            )

        snapshots_by_suite_index = {snapshot.suite_index: snapshot for snapshot in snapshots}
        for classification in classifications:
            self.assertEqual(classification.status, "started")
            snapshot = snapshots_by_suite_index[classification.suite_index]
            self.assertEqual(classification.worker_index, snapshot.worker_index)
            self.assertEqual(classification.pid, snapshot.pid)

    def _run_parallel(self, module, root, case_names, process_count, **arg_overrides):
        from warp._src.test_runner.common import EVENT_HISTORY_LIMIT
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.pool import run_parallel_suites

        spawn_context = multiprocessing.get_context("spawn")
        event_queue = spawn_context.SimpleQueue()
        tracker = WorkerStateTracker(history_limit=EVENT_HISTORY_LIMIT)
        monitor = WorkerEventMonitor(event_queue, tracker)
        run_dir = root / "diagnostics" / "run"
        temp_dir = root / "coverage"
        run_dir.mkdir(parents=True)
        temp_dir.mkdir()
        suites = self._suites(module, *case_names)
        args = self._args(**arg_overrides)
        monitor.start()
        try:
            with spawn_context.Manager() as manager:
                result = run_parallel_suites(
                    suites,
                    process_count,
                    manager,
                    args,
                    str(temp_dir),
                    event_queue,
                    tracker,
                    monitor,
                    run_dir,
                    time.monotonic_ns(),
                )
        finally:
            monitor.stop_and_drain()
            event_queue.close()
        return result, run_dir

    @staticmethod
    def _classification(result, suite_index):
        return next(item for item in result.suite_classifications if item.suite_index == suite_index)

    @staticmethod
    def _worker_for_test(snapshot, test_name):
        for worker in snapshot["workers"]:
            if test_name in json.dumps(worker, sort_keys=True):
                return worker
        raise AssertionError(f"No worker retained lifecycle state for {test_name}")

    @staticmethod
    def _worker_artifact(run_dir, worker, source):
        pattern = f"worker-{worker['worker_index']}-pid-{worker['pid']}.{source}.log"
        matches = list(run_dir.glob(pattern))
        if len(matches) != 1:
            raise AssertionError(f"Expected one artifact matching {pattern}, found {matches}")
        return matches[0]

    @staticmethod
    def _autodiscovered_suite(module, *case_names):
        loader = unittest.TestLoader()
        return unittest.TestSuite(loader.loadTestsFromTestCase(getattr(module, case_name)) for case_name in case_names)

    def _run_main(self, module, root, case_names, *extra_args):
        from warp._src.test_runner.runner import main

        diagnostics_root = root / "main-diagnostics"
        suite = self._autodiscovered_suite(module, *case_names)
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
            *extra_args,
        ]
        with mock.patch("warp.tests.unittest_suites.auto_discover_suite", return_value=suite):
            result = main(argv)
        return result, diagnostics_root

    def test_preserves_completed_out_of_order_result(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("BlockingCase", "PassingCase", "HardExitCase"),
                process_count=2,
            )

            self.assertIsNotNone(result.pool_failure)
            self.assertEqual(set(result.results_by_index), {1})
            self.assertTrue(marker_dir.joinpath("blocking-started").exists())
            self._worker_for_test(result.pool_failure.snapshot, "BlockingCase.test_wait_for_pool_shutdown")
            self._worker_for_test(result.pool_failure.snapshot, "HardExitCase.test_exit")
            records = [record for value in result.results_by_index.values() for record in value[6]]
            self.assertFalse(any(record[3] == "ERROR" for record in records))

    def test_hard_exit_lists_all_worker_states(self):
        from warp._src.test_runner.common import ProcessExitProvenance

        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("BlockingCase", "PassingCase", "HardExitCase"),
                process_count=2,
            )

            snapshot = result.pool_failure.snapshot
            self.assertEqual(len(snapshot["workers"]), 2)
            self.assertEqual({worker["worker_index"] for worker in snapshot["workers"]}, {0, 1})
            self.assertIn("last known state", result.pool_failure.formatted_summary)
            self.assertIn("candidate", result.pool_failure.formatted_summary)
            self.assertNotIn("caused by", result.pool_failure.formatted_summary)

            blocking = self._worker_for_test(snapshot, "BlockingCase.test_wait_for_pool_shutdown")
            crashed = self._worker_for_test(snapshot, "HardExitCase.test_exit")
            self.assertEqual(blocking["provenance"], ProcessExitProvenance.PARENT_TERMINATED.value)
            self.assertEqual(self._classification(result, 0).status, "started")
            self.assertEqual(crashed["provenance"], ProcessExitProvenance.INDEPENDENTLY_ABNORMAL.value)
            self.assertEqual(crashed["exit_code"], 86)
            self.assertEqual(self._classification(result, 2).status, "started")

    def test_cleanup_abort_reports_test_cleanup(self):
        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(module, root, ("CleanupAbortCase",), process_count=1)

            worker = self._worker_for_test(result.pool_failure.snapshot, "CleanupAbortCase.test_abort_during_gc")
            self.assertEqual(worker["phase"], "test_cleanup")

    def test_class_finalization_abort_names_suite_not_completed_test(self):
        from warp._src.test_runner.postmortem import make_pool_failure_test_record

        with temporary_fixture_cases() as (module, _, root):
            console = io.StringIO()
            with contextlib.redirect_stderr(console):
                result, run_dir = self._run_parallel(
                    module,
                    root,
                    ("FinalizationAbortCase",),
                    process_count=1,
                    verbose=0,
                )

            worker = result.pool_failure.snapshot["workers"][0]
            completed_test = "fixture_cases.FinalizationAbortCase.test_completes_before_class_finalization"
            self.assertEqual(worker["phase"], "test_stopped")
            self.assertIsNone(worker["current_test_id"])
            self.assertIsNone(worker["current_outcome"])
            self.assertIsNone(worker["current_elapsed_seconds"])
            self.assertEqual(worker["recent_tests"][-1]["test_id"], completed_test)
            self.assertEqual(worker["recent_tests"][-1]["outcome"], "OK")
            self.assertIsNotNone(worker["recent_tests"][-1]["elapsed_seconds"])
            self.assertTrue(worker["artifacts"]["fault"]["fatal_traceback_evidence"])
            on_disk = json.loads(run_dir.joinpath("crash-snapshot.json").read_text(encoding="utf-8"))
            self.assertTrue(on_disk["workers"][0]["artifacts"]["fault"]["fatal_traceback_evidence"])
            report = run_dir.joinpath("pool-failure.txt").read_text(encoding="utf-8")
            self.assertIn("candidate=suite_finalization:FinalizationAbortCase", report)
            self.assertIn("fatal traceback evidence=yes", report)
            junit_record = make_pool_failure_test_record(on_disk, run_dir / "crash-snapshot.json")
            self.assertIn("fatal traceback evidence=yes", junit_record[5])
            self.assertNotIn("Current thread", junit_record[5])
            self.assertIn("Parallel worker pool failed", console.getvalue())

    def test_abort_retains_fault_and_output_logs(self):
        with temporary_fixture_cases() as (module, _, root):
            result, run_dir = self._run_parallel(module, root, ("AbortCase",), process_count=1)

            worker = self._worker_for_test(result.pool_failure.snapshot, "AbortCase.test_abort")
            output_path = self._worker_artifact(run_dir, worker, "output")
            fault_path = self._worker_artifact(run_dir, worker, "fault")
            self.assertIn(b"abort-marker", output_path.read_bytes())
            self.assertIn(b"Fatal Python error", fault_path.read_bytes())

    def test_partial_junit_has_one_pool_failure_error(self):
        from warp._src.test_runner.postmortem import make_pool_failure_test_record
        from warp._src.test_runner.runner import main
        from warp.tests.unittest_utils import write_junit_results

        with temporary_fixture_cases() as (module, _, root):
            result, run_dir = self._run_parallel(
                module,
                root,
                ("BlockingCase", "PassingCase", "HardExitCase"),
                process_count=2,
            )
            snapshot = result.pool_failure.snapshot
            self.assertEqual(snapshot["suite_counts"], {"discovered": 3, "confirmed": 1})
            self.assertNotIn("confirmed_indexes", snapshot)
            self.assertNotIn("recent_tests", snapshot)
            for worker in snapshot["workers"]:
                self.assertIn("worker_index", worker)
                self.assertIn("pid", worker)
                self.assertIn("exit_code", worker)
                self.assertIn("signal_name", worker)
                self.assertIn("provenance", worker)
                self.assertIn("phase", worker)
                self.assertNotIn("candidate", worker)
                self.assertNotIn("diagnostics", worker)
                self.assertIn("current_elapsed_seconds", worker)
                self.assertIn("current_outcome", worker)
                self.assertIn("transition_age_seconds", worker)
                self.assertLessEqual(len(worker["recent_tests"]), 3)
                self.assertEqual(set(worker["artifacts"]), {"journal", "output", "fault"})
                for artifact in worker["artifacts"].values():
                    self.assertIn(artifact["state"], {"missing", "empty", "non_empty", "unreadable"})
                    self.assertIn("path", artifact)
                    self.assertIn("size_bytes", artifact)
            summary = result.pool_failure.formatted_summary
            self.assertIn("confirmed 1/3 discovered suites", summary)
            self.assertIn("transition age=", summary)
            self.assertIn("current/partial elapsed=", summary)
            self.assertIn("recent completed tests", summary)
            self.assertIn("artifacts:", summary)
            on_disk = json.loads(run_dir.joinpath("crash-snapshot.json").read_text(encoding="utf-8"))
            self.assertEqual(on_disk["suite_counts"], {"discovered": 3, "confirmed": 1})
            report = run_dir.joinpath("pool-failure.txt").read_text(encoding="utf-8")
            self.assertIn("confirmed 1/3 discovered suites", report)
            self.assertIn("recent completed tests", report)
            test_records = [
                record for index in sorted(result.results_by_index) for record in result.results_by_index[index][6]
            ]
            pool_record = make_pool_failure_test_record(
                result.pool_failure.snapshot,
                run_dir / "crash-snapshot.json",
            )
            junit_path = root / "rspec.xml"
            write_junit_results(
                str(junit_path),
                test_records,
                0.0,
                extra_records=(pool_record,),
            )

            xml_root = ElementTree.parse(junit_path).getroot()
            pool_errors = xml_root.findall("./testcase[@classname='warp.tests.parallel']")
            self.assertEqual(len(pool_errors), 1)
            self.assertEqual(pool_errors[0].attrib["name"], "WorkerPoolCrash")
            self.assertIsNotNone(pool_errors[0].find("error"))
            junit_error = pool_errors[0].find("error").text
            self.assertIn("confirmed 1/3 discovered suites", junit_error)
            self.assertIn("current/partial elapsed=", junit_error)
            self.assertIn("recent completed tests", junit_error)
            self.assertNotIn("hard-exit-marker", junit_error)
            self.assertEqual(int(xml_root.attrib["errors"]), 1)
            self.assertEqual(int(xml_root.attrib["tests"]), len(xml_root.findall("testcase")))
            self.assertEqual(list(junit_path.parent.glob(".rspec.xml.*.tmp")), [])

            main_junit_path = root / "main-rspec.xml"
            main_diagnostics = root / "main-junit-diagnostics"
            suite = self._autodiscovered_suite(module, "BlockingCase", "PassingCase", "HardExitCase")
            argv = [
                "-s",
                "autodetect",
                "-j",
                "2",
                "--maxjobs",
                "2",
                "-q",
                "--diagnostics-dir",
                str(main_diagnostics),
                "--junit-report-xml",
                str(main_junit_path),
            ]
            with mock.patch("warp.tests.unittest_suites.auto_discover_suite", return_value=suite):
                with self.assertRaises(SystemExit) as raised:
                    main(argv)
            self.assertEqual(raised.exception.code, 1)

            main_root = ElementTree.parse(main_junit_path).getroot()
            main_pool_errors = main_root.findall("./testcase[@classname='warp.tests.parallel']")
            self.assertEqual(len(main_pool_errors), 1)
            self.assertEqual(main_pool_errors[0].attrib["name"], "WorkerPoolCrash")
            self.assertIsNotNone(main_pool_errors[0].find("error"))
            self.assertEqual(int(main_root.attrib["errors"]), 1)
            self.assertEqual(int(main_root.attrib["tests"]), len(main_root.findall("testcase")))
            self.assertEqual(int(main_root.attrib["tests"]), 2)

    def test_clean_results_remain_in_discovery_order(self):
        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("FirstPidCase", "PassingCase", "SecondPidCase"),
                process_count=2,
            )

            self.assertIsNone(result.pool_failure)
            self.assertEqual(list(result.results_by_index), [0, 1, 2])
            self.assertEqual(
                [result.results_by_index[index][6][0][0] for index in result.results_by_index],
                ["FirstPidCase", "PassingCase", "SecondPidCase"],
            )

            main_result, diagnostics_root = self._run_main(
                module,
                root,
                ("FirstPidCase", "PassingCase", "SecondPidCase"),
            )
            self.assertIsNone(main_result)
            run_dir = next(diagnostics_root.glob("run-*"))
            timing_payload = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertEqual(
                [record["suite_name"] for record in timing_payload["suites"]],
                ["FirstPidCase", "PassingCase", "SecondPidCase"],
            )
            self.assertTrue(all(record["status"] == "complete" for record in timing_payload["suites"]))
            self.assertEqual(list(run_dir.glob("worker-*")), [])

    def test_ordinary_failure_does_not_become_pool_failure(self):
        with temporary_fixture_cases() as (module, _, root):
            result, _ = self._run_parallel(module, root, ("OrdinaryFailureCase",), process_count=1)

            self.assertIsNone(result.pool_failure)
            self.assertEqual(len(result.results_by_index[0][2]), 1)
            self.assertEqual(self._classification(result, 0).status, "confirmed")

            with self.assertRaises(SystemExit) as raised:
                self._run_main(module, root, ("OrdinaryFailureCase",))
            self.assertEqual(raised.exception.code, 1)
            run_dir = next((root / "main-diagnostics").glob("run-*"))
            timing_payload = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertEqual(len(timing_payload["suites"]), 1)
            self.assertEqual(timing_payload["suites"][0]["status"], "complete")
            self.assertEqual(timing_payload["suites"][0]["outcomes"], {"FAIL": 1})
            self.assertTrue(list(run_dir.glob("worker-*.events.jsonl")))
            self.assertTrue(list(run_dir.glob("worker-*.output.log")))
            self.assertTrue(list(run_dir.glob("worker-*.fault.log")))

    def test_diagnostics_setup_failure_preserves_primary_test_failure(self):
        with temporary_fixture_cases() as (module, _, root):
            console = io.StringIO()
            with (
                mock.patch(
                    "warp._src.test_runner.runner.create_diagnostics_run_dir",
                    side_effect=OSError("diagnostics setup unavailable"),
                ),
                contextlib.redirect_stderr(console),
                self.assertRaises(SystemExit) as raised,
            ):
                self._run_main(
                    module,
                    root,
                    ("OrdinaryFailureCase",),
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("ordinary assertion marker", console.getvalue())
            self.assertIn(
                "Failed to create diagnostics run directory: diagnostics setup unavailable",
                console.getvalue(),
            )

    def test_diagnostics_finalization_failure_does_not_replace_test_failure(self):
        with temporary_fixture_cases() as (module, _, root):
            console = io.StringIO()
            with (
                mock.patch(
                    "warp._src.test_runner.runner.finalize_diagnostics",
                    side_effect=OSError("diagnostics finalization unavailable"),
                ),
                contextlib.redirect_stderr(console),
                self.assertRaises(SystemExit) as raised,
            ):
                self._run_main(
                    module,
                    root,
                    ("OrdinaryFailureCase",),
                )

            self.assertEqual(raised.exception.code, 1)
            self.assertIn("ordinary assertion marker", console.getvalue())
            self.assertIn(
                "Failed to clean up durable diagnostics: diagnostics finalization unavailable",
                console.getvalue(),
            )

    def test_process_controls_escape_after_success_and_failure(self):
        primary_cases = (("PassingCase", "OK"), ("OrdinaryFailureCase", "FAILED"))
        for control in (KeyboardInterrupt("stop now"), SystemExit(73)):
            for case_name, expected_status in primary_cases:
                with self.subTest(control=type(control).__name__, primary=case_name):
                    with temporary_fixture_cases() as (module, _, root):
                        console = io.StringIO()
                        with (
                            mock.patch(
                                "warp._src.test_runner.runner.finalize_diagnostics",
                                side_effect=control,
                            ),
                            contextlib.redirect_stderr(console),
                            self.assertRaises(type(control)) as raised,
                        ):
                            self._run_main(
                                module,
                                root,
                                (case_name,),
                            )

                        self.assertIs(raised.exception, control)
                        self.assertIn(expected_status, console.getvalue())
                        run_dir = next((root / "main-diagnostics").glob("run-*"))
                        self.assertTrue(run_dir.joinpath("run.json").is_file())
                        self.assertTrue(list(run_dir.glob("worker-*.events.jsonl")))
                        self.assertFalse(any(thread.name == "worker-event-monitor" for thread in threading.enumerate()))

    def test_failfast_prevents_unstarted_suite_execution(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            result, _ = self._run_parallel(
                module,
                root,
                ("OrdinaryFailureCase", "ShouldNotStartCase"),
                process_count=1,
                failfast=True,
            )

            self.assertIsNone(result.pool_failure)
            self.assertEqual(len(result.results_by_index[0][2]), 1)
            self.assertFalse(marker_dir.joinpath("unexpected-start").exists())
            self.assertEqual(self._classification(result, 1).status, "skipped")

    def test_output_pressure_keeps_worker_attribution(self):
        with temporary_fixture_cases() as (module, _, root):
            result, run_dir = self._run_parallel(
                module,
                root,
                ("OutputPressureCase", "AbortCase"),
                process_count=2,
            )

            pressure_worker = self._worker_for_test(
                result.pool_failure.snapshot, "OutputPressureCase.test_output_pressure"
            )
            abort_worker = self._worker_for_test(result.pool_failure.snapshot, "AbortCase.test_abort")
            pressure_output_path = self._worker_artifact(run_dir, pressure_worker, "output")
            abort_output_path = self._worker_artifact(run_dir, abort_worker, "output")
            pressure_output = pressure_output_path.read_bytes()
            abort_output = abort_output_path.read_bytes()
            self.assertIn(b"output-pressure-marker", pressure_output)
            self.assertIn(b"abort-marker", abort_output)
            for output_path in run_dir.glob("worker-*.output.log"):
                output = output_path.read_bytes()
                if output_path != pressure_output_path:
                    self.assertNotIn(b"output-pressure-marker", output)
                if output_path != abort_output_path:
                    self.assertNotIn(b"abort-marker", output)
            if pressure_worker["pid"] == abort_worker["pid"]:
                self.assertEqual(pressure_worker["worker_index"], abort_worker["worker_index"])
                lifecycle_state = json.dumps(pressure_worker, sort_keys=True)
                self.assertIn("OutputPressureCase.test_output_pressure", lifecycle_state)
                self.assertIn("AbortCase.test_abort", lifecycle_state)

    def test_crash_remains_failure(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            with self.assertRaises(SystemExit) as raised:
                self._run_main(module, root, ("AbortCase",))

            self.assertEqual(raised.exception.code, 1)
            snapshot_path = next((root / "main-diagnostics").glob("run-*/crash-snapshot.json"))
            snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
            self.assertEqual(snapshot["failure"]["type"], "BrokenProcessPool")
            attempts_path = marker_dir / "abort-attempts"
            self.assertTrue(attempts_path.is_file())
            self.assertEqual(attempts_path.read_text(encoding="utf-8"), "1")

    @unittest.skipIf(sys.version_info < (3, 11), "Process isolation requires Python 3.11 or newer")
    def test_isolation_uses_fresh_worker_pids(self):
        with temporary_fixture_cases() as (module, marker_dir, root):
            result, _ = self._run_main(
                module,
                root,
                ("FirstPidCase", "SecondPidCase"),
                "--isolate-test-processes",
            )

            self.assertIsNone(result)
            first_pid = marker_dir.joinpath("first-pid").read_text(encoding="utf-8")
            second_pid = marker_dir.joinpath("second-pid").read_text(encoding="utf-8")
            self.assertNotEqual(first_pid, second_pid)
