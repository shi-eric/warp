# ruff: noqa: PLC0415

import json
import os
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock


class TestArtifacts(unittest.TestCase):
    def test_atomic_json_replaces_complete_document(self):
        from warp._src.test_runner.artifacts import atomic_write_json

        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory, "run.json")
            path.write_text('{"old":true}', encoding="utf-8")
            atomic_write_json(path, {"version": 1, "status": "complete"})
            self.assertEqual(json.loads(path.read_text(encoding="utf-8"))["status"], "complete")
            self.assertEqual(list(path.parent.glob(".run.json.*.tmp")), [])

    def test_run_directory_creation_never_reuses_an_existing_run(self):
        from warp._src.test_runner.artifacts import create_diagnostics_run_dir

        with tempfile.TemporaryDirectory() as directory:
            root = pathlib.Path(directory)
            first = create_diagnostics_run_dir(root, wall_time_ns=0, parent_pid=1234)
            second = create_diagnostics_run_dir(root, wall_time_ns=0, parent_pid=1234)

            self.assertNotEqual(first, second)
            self.assertTrue(first.is_dir())
            self.assertTrue(second.is_dir())


class TestModuleLoadCollection(unittest.TestCase):
    def test_collection_aggregates_churn_repeats_and_slowest(self):
        from warp._src.test_runner.module_loads import collect_module_load_summary

        with tempfile.TemporaryDirectory() as temp_root:
            run_dir = pathlib.Path(temp_root)
            log = run_dir / "worker-0-pid-100.output.log"
            log.write_text(
                "Module wp.sim abc1234 load on device 'cuda:0' took 100.0 ms (compiled)\n"
                "Module wp.sim abc1234 load on device 'cuda:0' took 50.0 ms (compiled)\n"
                "Module wp.sim def5678 load on device 'cuda:0' took 25.0 ms (error)\n"
                "Module wp.fem 1111111 load on device 'cpu' took 10.0 ms (cached)\n"
                "Module wp.top load on device 'cpu' took 500.0 ms (compiled)\n",
                encoding="utf-8",
            )
            summary = collect_module_load_summary(run_dir, no_shared_cache=False)

        self.assertEqual(summary.status_counts, {"compiled": 3, "cached": 1, "error": 1})
        self.assertEqual(len(summary.hash_churn), 1)
        churn = summary.hash_churn[0]
        self.assertEqual((churn.module, churn.device), ("wp.sim", "cuda:0"))
        self.assertEqual(churn.hashes, ("abc1234", "def5678"))
        self.assertEqual((churn.attempts, churn.compiled_count, churn.error_count), (3, 2, 1))
        self.assertEqual(churn.aggregate_ms, 175.0)
        self.assertEqual(len(summary.repeated_compilations), 1)
        repeat = summary.repeated_compilations[0]
        self.assertEqual((repeat.module, repeat.module_hash, repeat.compilations), ("wp.sim", "abc1234", 2))
        self.assertEqual(repeat.aggregate_ms, 150.0)
        self.assertEqual([record.module for record in summary.slowest_compiled], ["wp.top", "wp.sim", "wp.sim"])

    def test_summary_serializes_read_errors(self):
        from warp._src.test_runner.module_loads import ModuleLoadSummary

        error = "worker-0.output.log: permission denied"
        summary = ModuleLoadSummary(
            files_inspected=1,
            parsed_records=0,
            complete=False,
            no_shared_cache=False,
            status_counts={},
            hash_churn=(),
            repeated_compilations=(),
            slowest_compiled=(),
            read_errors=(error,),
        )

        self.assertEqual(summary.to_dict().get("read_errors"), [error])


class TestSuiteTimings(unittest.TestCase):
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

    def test_tracker_measures_worker_occupancy_from_suite_events(self):
        from warp._src.test_runner.common import EventKind, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker()

        def event(kind, sequence, monotonic_ns, **fields):
            return WorkerEvent(
                sequence=sequence,
                event=kind,
                worker_index=4,
                pid=4818,
                monotonic_ns=monotonic_ns,
                wall_time_ns=1_000_000_000 + monotonic_ns,
                suite_index=9,
                suite_name="fixture.TestSlow",
                test_count=184,
                **fields,
            )

        tracker.handle_event(event(EventKind.SUITE_STARTED, 1, 2_000_000_000))
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                2,
                3_000_000_000,
                test_id="fixture.TestSlow.test_ok",
                outcome="OK",
            )
        )
        tracker.handle_event(
            event(
                EventKind.TEST_OUTCOME,
                3,
                4_000_000_000,
                test_id="fixture.TestSlow.test_skip",
                outcome="SKIP",
            )
        )
        tracker.handle_event(event(EventKind.SUITE_FINISHED, 4, 7_500_000_000))

        timing = tracker.suite_timings()[0]
        self.assertEqual(timing.suite_index, 9)
        self.assertEqual(timing.suite_name, "fixture.TestSlow")
        self.assertEqual(timing.unit_type, "class")
        self.assertEqual(timing.test_count, 184)
        self.assertEqual(timing.worker_index, 4)
        self.assertEqual(timing.pid, 4818)
        self.assertEqual(timing.started_offset_seconds, 2.0)
        self.assertEqual(timing.finished_offset_seconds, 7.5)
        self.assertEqual(timing.elapsed_seconds, 5.5)
        self.assertEqual(timing.completion_order, 1)
        self.assertEqual(timing.status, "complete")
        self.assertEqual(timing.outcomes, {"OK": 1, "SKIP": 1})

    def test_tracker_keeps_active_and_classification_only_records_separate(self):
        from warp._src.test_runner.common import EventKind, SuiteClassification, WorkerEvent
        from warp._src.test_runner.events import WorkerStateTracker

        tracker = WorkerStateTracker()
        tracker.handle_event(
            WorkerEvent(
                sequence=1,
                event=EventKind.SUITE_STARTED,
                worker_index=2,
                pid=8123,
                monotonic_ns=4_000_000_000,
                wall_time_ns=5_000_000_000,
                suite_index=1,
                suite_name="fixture.TestActive",
                test_count=3,
            )
        )
        classifications = (
            SuiteClassification(1, "fixture.TestActive", 3, "started", 2, 8123),
            SuiteClassification(2, "fixture.TestCancelled", 4, "skipped"),
            SuiteClassification(3, "fixture.TestPending", 5, "never_started"),
        )

        records = tracker.suite_timings(classifications, unit_type="module")

        self.assertEqual([record.suite_index for record in records], [1, 2, 3])
        self.assertEqual(records[0].status, "started")
        self.assertEqual(records[0].started_offset_seconds, 4.0)
        self.assertIsNone(records[0].finished_offset_seconds)
        self.assertIsNone(records[0].elapsed_seconds)
        self.assertEqual(records[0].worker_index, 2)
        self.assertEqual(records[0].unit_type, "module")
        self.assertEqual(records[1].status, "skipped")
        self.assertIsNone(records[1].worker_index)
        self.assertEqual(records[2].status, "never_started")

    def test_slowest_summary_ranks_only_twenty_completed_records(self):
        from warp._src.test_runner.artifacts import format_slowest_suites

        records = [
            self._timing(
                suite_index=index,
                suite_name=f"fixture.TestTiming{index:02d}",
                elapsed_seconds=float(index + 1),
                finished_offset_seconds=float(index + 2),
                completion_order=index + 1,
            )
            for index in range(25)
        ]
        records.append(
            self._timing(
                suite_index=25,
                suite_name="fixture.TestIncomplete",
                finished_offset_seconds=None,
                elapsed_seconds=None,
                completion_order=None,
                status="started",
                outcomes={},
            )
        )

        summary = format_slowest_suites(records, limit=20)
        completed_block, incomplete_block = summary.split("Incomplete suites:")
        completed_rows = [line for line in completed_block.splitlines() if line.startswith("  ")]

        self.assertEqual(len(completed_rows), 20)
        self.assertEqual(
            [line.split()[1] for line in completed_rows],
            [f"fixture.TestTiming{index:02d}" for index in range(24, 4, -1)],
        )
        self.assertNotIn("fixture.TestIncomplete", completed_block)
        self.assertIn("fixture.TestIncomplete", incomplete_block)

    def test_writes_compact_timing_json_atomically(self):
        from warp._src.test_runner.artifacts import write_suite_timings

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            write_suite_timings(run_dir, {"worker_count": 2}, [self._timing()])

            payload = json.loads(run_dir.joinpath("suite-timings.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["metadata"], {"worker_count": 2})
            self.assertEqual(payload["suites"][0]["suite_name"], "fixture.TestTiming")
            self.assertEqual(list(run_dir.glob(".suite-timings.json.*.tmp")), [])

    def test_finalization_removes_only_current_run_worker_evidence_on_success(self):
        from warp._src.test_runner.artifacts import finalize_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            retained = (run_dir / "run.json", run_dir / "suite-timings.json", run_dir / "notes.txt")
            generated = (
                run_dir / "worker-0-4818.events.jsonl",
                run_dir / "worker-0-pid-4818.output.log",
                run_dir / "worker-0-pid-4818.fault.log",
            )
            for path in (*retained, *generated):
                path.write_text(path.name, encoding="utf-8")

            finalize_diagnostics(run_dir, retain_worker_evidence=False)

            self.assertTrue(all(path.exists() for path in retained))
            self.assertTrue(all(not path.exists() for path in generated))

    def test_finalization_keeps_all_evidence_when_requested(self):
        from warp._src.test_runner.artifacts import finalize_diagnostics

        with tempfile.TemporaryDirectory() as directory:
            run_dir = pathlib.Path(directory)
            paths = (
                run_dir / "run.json",
                run_dir / "suite-timings.json",
                run_dir / "worker-0-4818.events.jsonl",
                run_dir / "worker-0-pid-4818.output.log",
                run_dir / "worker-0-pid-4818.fault.log",
                run_dir / "crash-snapshot.json",
                run_dir / "pool-failure.txt",
            )
            for path in paths:
                path.write_text(path.name, encoding="utf-8")

            finalize_diagnostics(run_dir, retain_worker_evidence=True)

            self.assertTrue(all(path.exists() for path in paths))

    def test_metadata_allowlists_environment_values(self):
        from warp._src.test_runner.artifacts import build_run_metadata

        args = SimpleNamespace(
            coverage=True,
            coverage_branch=True,
            level="class",
            pattern="test_timing*.py",
            suite="autodetect",
            testNamePatterns=["*timing*"],
            warp_debug=True,
        )
        environment = {
            "CUDA_VISIBLE_DEVICES": "0,1",
            "WARP_CACHE_PATH": "/opt/warp-cache",
            "CI_JOB_ID": "123456",
            "GITHUB_ACTIONS": "true",
            "SECRET_TOKEN": "do-not-serialize",
        }
        with mock.patch.dict(os.environ, environment, clear=True):
            metadata = build_run_metadata(
                args,
                process_count=3,
                run_start_monotonic_ns=10,
                run_start_wall_time_ns=20,
                parent_gil_enabled_initial=None,
            )

        encoded = json.dumps(metadata, sort_keys=True)
        self.assertEqual(metadata["worker_count"], 3)
        self.assertEqual(metadata["level"], "class")
        self.assertEqual(metadata["ci_providers"], ["GitHub Actions"])
        self.assertEqual(metadata["cuda_visible_devices"], "0,1")
        self.assertEqual(metadata["warp_cache_path"], "/opt/warp-cache")
        self.assertNotIn("CI_JOB_ID", encoded)
        self.assertNotIn("123456", encoded)
        self.assertNotIn("SECRET_TOKEN", encoded)
        self.assertNotIn("do-not-serialize", encoded)
        self.assertNotIn("environment", metadata)
        self.assertNotIn("serial_fallback", metadata)

    def test_run_metadata_distinguishes_parent_gil_endpoints(self):
        import inspect

        from warp._src.test_runner.artifacts import build_run_metadata

        self.assertIn("parent_gil_enabled_initial", inspect.signature(build_run_metadata).parameters)
        args = SimpleNamespace(
            coverage=False,
            coverage_branch=False,
            level="class",
            pattern="test*.py",
            suite="default",
            testNamePatterns=[],
            warp_debug=False,
        )
        metadata = build_run_metadata(
            args,
            process_count=2,
            run_start_monotonic_ns=10,
            run_start_wall_time_ns=20,
            parent_gil_enabled_initial=False,
            finished={
                "run_finished_monotonic_ns": 30,
                "run_finished_wall_time_ns": 40,
                "parent_gil_enabled_final": True,
            },
        )

        self.assertFalse(metadata["parent_gil_enabled_initial"])
        self.assertTrue(metadata["parent_gil_enabled_final"])
        self.assertNotIn("gil_enabled", metadata)
