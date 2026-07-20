# ruff: noqa: PLC0415

import json
import multiprocessing
import pathlib
import queue as queue_module
import tempfile
import threading
import unittest
from types import SimpleNamespace
from unittest import mock

from .support import (
    RaisingQueue,
    RecordingQueue,
    StubSinkFile,
    assert_thread_stops,
    publish_large_events,
)


class TestDiagnosticEvents(unittest.TestCase):
    def test_reporter_observes_gil_at_suite_boundaries(self):
        from warp._src.test_runner import common as gil
        from warp._src.test_runner import events as diagnostics
        from warp._src.test_runner.common import EventKind

        queue = RecordingQueue()
        reporter = diagnostics.WorkerEventReporter(
            event_queue=queue,
            worker_index=3,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        self.assertTrue(hasattr(diagnostics, "get_gil_enabled"))
        self.assertTrue(hasattr(reporter, "emit_gil_observed"))
        with mock.patch.object(
            gil.sys,
            "_is_gil_enabled",
            side_effect=(False, False, True),
            create=True,
        ):
            reporter.emit_gil_observed(EventKind.WORKER_INITIALIZED)
            reporter.emit_gil_observed(
                EventKind.SUITE_STARTED,
                suite_index=27,
                suite_name="fixture.TestGilTransition",
                test_count=1,
            )
            reporter.emit(
                EventKind.TEST_STOPPED,
                test_id="fixture.TestGilTransition.test_import",
            )
            reporter.emit_gil_observed(EventKind.SUITE_FINISHED)

        self.assertEqual(
            [event.event for event in queue.items],
            [
                EventKind.WORKER_INITIALIZED,
                EventKind.SUITE_STARTED,
                EventKind.TEST_STOPPED,
                EventKind.GIL_STATE_CHANGED,
                EventKind.SUITE_FINISHED,
            ],
        )
        initialized, started, _, changed, finished = queue.items
        self.assertFalse(initialized.gil_enabled)
        self.assertFalse(started.gil_enabled)
        self.assertFalse(changed.previous_gil_enabled)
        self.assertTrue(changed.gil_enabled)
        self.assertEqual(changed.observed_at, EventKind.SUITE_FINISHED.value)
        self.assertEqual(changed.suite_index, 27)
        self.assertEqual(changed.suite_name, "fixture.TestGilTransition")
        self.assertEqual(changed.test_id, "fixture.TestGilTransition.test_import")
        self.assertTrue(finished.gil_enabled)
        self.assertEqual(finished.suite_index, 27)

    def test_reporter_sends_event_and_journal_line(self):
        from warp._src.test_runner.common import EventKind
        from warp._src.test_runner.events import WorkerEventReporter

        queue = RecordingQueue()
        with tempfile.TemporaryDirectory() as directory:
            reporter = WorkerEventReporter(
                event_queue=queue,
                worker_index=3,
                run_dir=pathlib.Path(directory),
                run_start_monotonic_ns=100,
            )
            event = reporter.emit(
                EventKind.TEST_STARTED,
                suite_index=27,
                suite_name="fixture.TestCrash",
                test_id="fixture.TestCrash.test_abort",
            )
            reporter.close()

            self.assertEqual(event.worker_index, 3)
            self.assertEqual(event.sequence, 1)
            self.assertEqual(queue.items, [event])

            journal = next(pathlib.Path(directory).glob("*.events.jsonl"))
            payload = json.loads(journal.read_text(encoding="utf-8"))
            self.assertEqual(payload, event.to_dict())

    def test_tracker_distinguishes_outcome_cleanup_and_completion(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker(history_limit=3)

        def event(kind, sequence, test_id=None, outcome=None, elapsed_seconds=None):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestCleanup",
                test_id=test_id,
                outcome=outcome,
                elapsed_seconds=elapsed_seconds,
            )

        tracker.handle_event(event(EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(EventKind.WORKER_INITIALIZED, 2))
        tracker.handle_event(event(EventKind.SUITE_STARTED, 3))
        tracker.handle_event(event(EventKind.TEST_STARTED, 4, "fixture.TestCleanup.test_ok"))
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                5,
                "fixture.TestCleanup.test_ok",
                outcome="OK",
                elapsed_seconds=0.75,
            )
        )
        tracker.handle_event(event(EventKind.TEST_CLEANUP_STARTED, 6, "fixture.TestCleanup.test_ok"))

        snapshot = tracker.snapshots(now_ns=6_500_000_000)[0]
        self.assertEqual(snapshot.phase, "test_cleanup")
        self.assertEqual(snapshot.current_outcome, "OK")
        self.assertEqual(snapshot.current_test_id, "fixture.TestCleanup.test_ok")
        self.assertEqual(snapshot.current_elapsed_seconds, 2.5)
        self.assertEqual(snapshot.transition_age_seconds, 0.5)

        tracker.handle_event(event(EventKind.TEST_STOPPED, 7, "fixture.TestCleanup.test_ok"))
        snapshot = tracker.snapshots(now_ns=7_500_000_000)[0]
        self.assertEqual(snapshot.phase, "test_stopped")
        self.assertEqual(snapshot.recent_tests[-1].test_id, "fixture.TestCleanup.test_ok")
        self.assertEqual(snapshot.recent_tests[-1].outcome, "OK")
        self.assertEqual(snapshot.recent_tests[-1].elapsed_seconds, 0.75)
        self.assertIsNone(snapshot.current_test_id)
        self.assertIsNone(snapshot.current_outcome)
        self.assertIsNone(snapshot.current_elapsed_seconds)
        self.assertEqual(snapshot.transition_age_seconds, 0.5)

        tracker.handle_event(event(EventKind.SUITE_FINALIZING, 8))
        snapshot = tracker.snapshots(now_ns=8_500_000_000)[0]
        self.assertEqual(snapshot.phase, "suite_finalizing")
        self.assertIsNone(snapshot.current_test_id)

    def test_tracker_records_gil_change_without_changing_phase(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker(history_limit=3)

        def event(kind, sequence, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestGilTransition",
                **fields,
            )

        tracker.handle_event(event(EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(EventKind.WORKER_INITIALIZED, 2, gil_enabled=False))
        tracker.handle_event(event(EventKind.SUITE_STARTED, 3, gil_enabled=False))
        tracker.handle_event(
            event(
                EventKind.TEST_STARTED,
                4,
                test_id="fixture.TestGilTransition.test_import",
            )
        )
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                5,
                test_id="fixture.TestGilTransition.test_import",
                outcome="OK",
                elapsed_seconds=0.5,
            )
        )
        tracker.handle_event(
            event(
                EventKind.TEST_STOPPED,
                6,
                test_id="fixture.TestGilTransition.test_import",
            )
        )

        before_change = tracker.snapshots(now_ns=6_500_000_000)[0]
        self.assertTrue(hasattr(before_change, "initial_gil_enabled"))
        tracker.handle_event(
            event(
                EventKind.GIL_STATE_CHANGED,
                7,
                previous_gil_enabled=False,
                gil_enabled=True,
                observed_at=EventKind.SUITE_FINISHED.value,
            )
        )
        after_change = tracker.snapshots(now_ns=7_500_000_000)[0]

        self.assertEqual(after_change.phase, "test_stopped")
        self.assertEqual(after_change.last_transition_ns, 6_000_000_000)
        self.assertFalse(after_change.initial_gil_enabled)
        self.assertTrue(after_change.gil_enabled)
        self.assertEqual(after_change.first_gil_state_change.previous_gil_enabled, False)
        self.assertEqual(after_change.first_gil_state_change.gil_enabled, True)
        self.assertEqual(after_change.first_gil_state_change.observed_at, "suite_finished")
        self.assertEqual(after_change.first_gil_state_change.suite_index, 4)
        self.assertEqual(
            after_change.first_gil_state_change.test_id,
            "fixture.TestGilTransition.test_import",
        )

    def test_tracker_preserves_last_non_ok_outcome_beyond_recent_history(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker(history_limit=3)

        def event(kind, sequence, test_id=None, outcome=None):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestOutcomes",
                test_id=test_id,
                outcome=outcome,
                elapsed_seconds=0.25 if outcome is not None else None,
            )

        tracker.handle_event(event(EventKind.WORKER_STARTED, 1))
        tracker.handle_event(event(EventKind.SUITE_STARTED, 2))
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                3,
                test_id="fixture.TestOutcomes.test_failure",
                outcome="FAIL",
            )
        )
        tracker.handle_event(event(EventKind.TEST_STOPPED, 4, test_id="fixture.TestOutcomes.test_failure"))
        sequence = 5
        for index in range(4):
            test_id = f"fixture.TestOutcomes.test_ok_{index}"
            tracker.handle_event(event(EventKind.TEST_OUTCOME, sequence, test_id=test_id, outcome="OK"))
            tracker.handle_event(event(EventKind.TEST_STOPPED, sequence + 1, test_id=test_id))
            sequence += 2

        snapshot = tracker.snapshots(now_ns=14_000_000_000)[0]
        self.assertTrue(hasattr(snapshot, "last_non_ok_test"))
        self.assertEqual(
            [test.test_id for test in snapshot.recent_tests],
            [
                "fixture.TestOutcomes.test_ok_1",
                "fixture.TestOutcomes.test_ok_2",
                "fixture.TestOutcomes.test_ok_3",
            ],
        )
        self.assertEqual(snapshot.last_non_ok_test.test_id, "fixture.TestOutcomes.test_failure")
        self.assertEqual(snapshot.last_non_ok_test.outcome, "FAIL")
        self.assertEqual(snapshot.last_non_ok_test.elapsed_seconds, 0.25)

    def test_journal_replay_recovers_lagging_tracker_without_regression(self):
        from warp._src.test_runner import events as diagnostics
        from warp._src.test_runner.common import EventKind, WorkerEvent

        self.assertTrue(hasattr(diagnostics, "replay_worker_event_journals"))
        tracker = diagnostics.WorkerStateTracker(history_limit=3)

        def event(kind, sequence, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=2,
                pid=8123,
                monotonic_ns=sequence * 1_000_000_000,
                wall_time_ns=sequence * 1000,
                suite_index=4,
                suite_name="fixture.TestJournalRecovery",
                **fields,
            )

        journal_events = (
            event(EventKind.WORKER_STARTED, 1),
            event(EventKind.WORKER_INITIALIZED, 2, gil_enabled=False),
            event(EventKind.SUITE_STARTED, 3, gil_enabled=False),
            event(
                EventKind.TEST_OUTCOME,
                4,
                test_id="fixture.TestJournalRecovery.test_failure",
                outcome="ERROR",
                elapsed_seconds=0.5,
            ),
            event(
                EventKind.TEST_STOPPED,
                5,
                test_id="fixture.TestJournalRecovery.test_failure",
            ),
            event(
                EventKind.GIL_STATE_CHANGED,
                6,
                previous_gil_enabled=False,
                gil_enabled=True,
                observed_at=EventKind.SUITE_FINISHED.value,
                test_id="fixture.TestJournalRecovery.test_failure",
            ),
            event(EventKind.SUITE_FINISHED, 7, gil_enabled=True),
        )
        tracker.handle_event(journal_events[0])
        tracker.handle_event(journal_events[1])

        with tempfile.TemporaryDirectory() as directory:
            journal = pathlib.Path(directory, "worker-2-8123.events.jsonl")
            journal.write_text(
                "".join(json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in journal_events),
                encoding="utf-8",
            )
            errors = diagnostics.replay_worker_event_journals(directory, tracker)

        self.assertEqual(errors, ())
        recovered = tracker.snapshots(now_ns=8_000_000_000)[0]
        self.assertEqual(recovered.phase, "idle")
        self.assertEqual(
            recovered.last_non_ok_test.test_id,
            "fixture.TestJournalRecovery.test_failure",
        )
        self.assertTrue(recovered.gil_enabled)

        tracker.handle_event(journal_events[2])
        after_late_duplicate = tracker.snapshots(now_ns=9_000_000_000)[0]
        self.assertEqual(after_late_duplicate.phase, "idle")
        self.assertTrue(after_late_duplicate.gil_enabled)

    def test_tracker_accepts_gil_annotation_as_first_recovered_event(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker()
        event = WorkerEvent(
            sequence=6,
            event=EventKind.GIL_STATE_CHANGED,
            worker_index=2,
            pid=8123,
            monotonic_ns=6_000_000_000,
            wall_time_ns=6000,
            suite_index=4,
            suite_name="fixture.TestJournalRecovery",
            previous_gil_enabled=False,
            gil_enabled=True,
            observed_at=EventKind.SUITE_FINISHED.value,
        )

        try:
            tracker.handle_event(event)
        except KeyError as error:
            self.fail(f"GIL recovery annotation raised {error!r}")
        snapshot = tracker.snapshots(now_ns=7_000_000_000)[0]
        self.assertEqual(snapshot.phase, "unknown")
        self.assertTrue(snapshot.gil_enabled)

    def test_initializing_snapshot_names_worker_initializer(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker
        from warp._src.test_runner.postmortem import build_crash_snapshot, format_pool_failure

        tracker = WorkerStateTracker()
        tracker.handle_event(
            WorkerEvent(
                sequence=1,
                event=EventKind.WORKER_STARTED,
                worker_index=2,
                pid=8123,
                monotonic_ns=1,
                wall_time_ns=1,
            )
        )

        crash_snapshot = build_crash_snapshot(
            RuntimeError("initialization failed"),
            (),
            tracker.snapshots(now_ns=2),
            (),
            None,
        )

        self.assertEqual(crash_snapshot["workers"][0]["phase"], "initializing")
        self.assertIn("candidate=initializer:worker initialization", format_pool_failure(crash_snapshot))

    def test_reporter_emit_propagates_process_controls(self):
        from warp._src.test_runner.common import EventKind
        from warp._src.test_runner.events import WorkerEventReporter

        for exception in (KeyboardInterrupt("queue interrupted"), SystemExit(74)):
            with self.subTest(exception=type(exception).__name__):
                reporter = WorkerEventReporter(
                    event_queue=RaisingQueue(exception),
                    worker_index=0,
                    run_dir=None,
                    run_start_monotonic_ns=0,
                )
                with self.assertRaises(type(exception)) as raised:
                    reporter.emit(EventKind.WORKER_STARTED)

                self.assertIs(raised.exception, exception)
                reporter.close()

    def test_shutdown_seals_handlers_before_closing_the_queue(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerEventMonitor, WorkerStateTracker
        from warp._src.test_runner.runner import _RunDiagnostics

        class ClosableQueue:
            def __init__(self):
                self.queue = queue_module.Queue()
                self.published = []
                self.closed = threading.Event()

            def get(self):
                return self.queue.get()

            def put(self, item):
                self.published.append(item)
                self.queue.put(item)

            def close(self):
                self.closed.set()
                self.queue.put(None)

        def event(sequence):
            return WorkerEvent(
                sequence=sequence,
                event=EventKind.WORKER_STARTED,
                worker_index=sequence,
                pid=7000 + sequence,
                monotonic_ns=sequence,
                wall_time_ns=sequence,
            )

        event_queue = ClosableQueue()
        tracker = WorkerStateTracker()
        callback_started = threading.Event()
        release_callback = threading.Event()
        handled_sequences = []
        callback_error = KeyboardInterrupt("callback interrupted")

        def blocking_callback(item):
            handled_sequences.append(item.sequence)
            callback_started.set()
            if not release_callback.wait(5.0):
                raise TimeoutError("callback release timed out")
            raise callback_error

        monitor = WorkerEventMonitor(event_queue, tracker, on_event=blocking_callback)
        seal_started = threading.Event()
        original_seal = monitor.seal

        def observed_seal():
            seal_started.set()
            original_seal()

        monitor.seal = observed_seal
        diagnostics = _RunDiagnostics()
        diagnostics.attach_monitor(event_queue, tracker, monitor)
        diagnostics.degrade_event_monitor()
        monitor.start()
        event_queue.put(event(1))
        event_queue.put(event(2))
        try:
            self.assertTrue(callback_started.wait(5.0))

            stop_errors = []

            def stop_diagnostics():
                try:
                    diagnostics.stop_monitor()
                except BaseException as error:
                    stop_errors.append(error)

            stop_thread = threading.Thread(target=stop_diagnostics, daemon=True)
            stop_thread.start()
            self.assertTrue(seal_started.wait(5.0))
            self.assertTrue(stop_thread.is_alive())
            release_callback.set()
            assert_thread_stops(self, stop_thread)
            assert_thread_stops(self, monitor._thread)

            self.assertEqual(stop_errors, [callback_error])
            self.assertEqual(handled_sequences, [1])
            self.assertTrue(event_queue.closed.is_set())
            self.assertNotIn(None, event_queue.published)
        finally:
            release_callback.set()
            if not event_queue.closed.is_set():
                event_queue.close()
            monitor._thread.join(5.0)

    def test_spawn_simple_queue_pressure_preserves_every_event(self):
        from warp._src.test_runner.events import WorkerEventMonitor

        class PressureTracker:
            def __init__(self):
                self.events = []
                self.error = None

            def handle_event(self, event):
                self.events.append(event)

            def handle_monitor_error(self, error):
                if self.error is None:
                    self.error = error
                return self.error

            def raise_monitor_error(self):
                if self.error is not None:
                    raise self.error

        spawn_context = multiprocessing.get_context("spawn")
        event_queue = spawn_context.SimpleQueue()
        tracker = PressureTracker()
        monitor = WorkerEventMonitor(event_queue, tracker)
        producer = spawn_context.Process(
            target=publish_large_events,
            args=(event_queue, 256, 262_144),
        )
        monitor.start()
        producer.start()
        try:
            assert_thread_stops(self, producer)
            self.assertEqual(producer.exitcode, 0)

            stop_errors = []

            def stop_monitor():
                try:
                    monitor.stop_and_drain()
                except BaseException as error:
                    stop_errors.append(error)

            stop_thread = threading.Thread(target=stop_monitor, daemon=True)
            stop_thread.start()
            assert_thread_stops(self, stop_thread)

            self.assertEqual(stop_errors, [])
            self.assertEqual([event["index"] for event in tracker.events], list(range(256)))
            self.assertTrue(all(len(event["payload"]) == 262_144 for event in tracker.events))
        finally:
            if producer.is_alive():
                producer.terminate()
                producer.join(5.0)
            event_queue.close()


class TestWorkerSinkIsolation(unittest.TestCase):
    def _configure(self, *, journal_error=None, open_side_effect=None, dup2_side_effect=None, enable_error=None):
        from warp._src.test_runner import events as diagnostics

        queue = RecordingQueue()
        output_file = StubSinkFile(80)
        fault_file = StubSinkFile(81)
        warnings = []

        if open_side_effect is None:

            def open_side_effect(path, *args, **kwargs):
                return output_file if str(path).endswith(".output.log") else fault_file

        if dup2_side_effect is None:

            def dup2_side_effect(source, target):
                pass

        open_journal = mock.Mock(return_value=82)
        if journal_error is not None:
            open_journal.side_effect = journal_error
        enable = mock.Mock()
        if enable_error is not None:
            enable.side_effect = enable_error

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(diagnostics.os, "open", open_journal),
                mock.patch("builtins.open", side_effect=open_side_effect),
                mock.patch.object(diagnostics.os, "dup", side_effect=(90, 91)),
                mock.patch.object(diagnostics.os, "dup2", side_effect=dup2_side_effect) as dup2,
                mock.patch.object(diagnostics.os, "close"),
                mock.patch.object(diagnostics.os, "write", side_effect=lambda descriptor, payload: len(payload)),
                mock.patch.object(diagnostics.faulthandler, "disable"),
                mock.patch.object(diagnostics.faulthandler, "enable", enable),
                mock.patch.object(diagnostics.atexit, "register"),
                mock.patch.object(diagnostics, "warn", side_effect=warnings.append),
            ):
                reporter = diagnostics.configure_worker_diagnostics(
                    event_queue=queue,
                    worker_index=4,
                    run_dir=pathlib.Path(directory),
                    run_start_monotonic_ns=0,
                )
                reporter.emit(diagnostics.EventKind.TEST_STARTED, test_id="fixture.Case.test_lives")
                diagnostics.close_worker_diagnostics()

        return SimpleNamespace(
            queue=queue,
            warnings=warnings,
            output_file=output_file,
            fault_file=fault_file,
            dup2=dup2,
            enable=enable,
        )

    def test_journal_open_failure_keeps_queue_and_other_sinks(self):
        result = self._configure(journal_error=OSError("journal unavailable"))

        self.assertEqual(
            [event.event.value for event in result.queue.items],
            ["worker_started", "test_started", "worker_shutdown"],
        )
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("event journal", result.warnings[0])
        self.assertTrue(result.output_file.closed)
        self.assertTrue(result.fault_file.closed)
        result.enable.assert_called_once()

    def test_output_open_failure_keeps_queue_and_fault_sink(self):
        fault_file = StubSinkFile(81)

        def open_sink(path, *args, **kwargs):
            if str(path).endswith(".output.log"):
                raise OSError("output unavailable")
            return fault_file

        result = self._configure(open_side_effect=open_sink)

        self.assertEqual(len(result.queue.items), 3)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("worker output", result.warnings[0])
        self.assertTrue(fault_file.closed)
        result.enable.assert_called_once()

    def test_fault_open_failure_keeps_queue_and_output_sink(self):
        output_file = StubSinkFile(80)

        def open_sink(path, *args, **kwargs):
            if str(path).endswith(".fault.log"):
                raise OSError("fault unavailable")
            return output_file

        result = self._configure(open_side_effect=open_sink)

        self.assertEqual(len(result.queue.items), 3)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("faulthandler sink", result.warnings[0])
        self.assertTrue(output_file.closed)
        result.enable.assert_not_called()

    def test_faulthandler_enable_failure_disables_only_fault_sink(self):
        result = self._configure(enable_error=RuntimeError("faulthandler unavailable"))

        self.assertEqual(len(result.queue.items), 3)
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("faulthandler", result.warnings[0])
        self.assertTrue(result.fault_file.closed)
        self.assertTrue(result.output_file.closed)
