# ruff: noqa: PLC0415

import io
import pathlib
import tempfile
import threading
import unittest
import xml.etree.ElementTree as ElementTree
from types import SimpleNamespace
from unittest import mock

from .fixture_sources import JUNIT_FIXTURE_SOURCES, JUNIT_ORDINARY_SOURCE
from .support import LocalManager, RaisingQueue, RecordingQueue, assert_thread_stops, temporary_junit_module


class TestResultLifecycle(unittest.TestCase):
    class PassingTest(unittest.TestCase):
        def test_passes(self):
            pass

    class TwoPassingTests(unittest.TestCase):
        def test_first_passes(self):
            pass

        def test_second_passes(self):
            pass

    class FailingSubTest(unittest.TestCase):
        def test_subtest_failure(self):
            with self.subTest(value=1):
                self.fail("subtest failure")

    class DocumentedFailure(unittest.TestCase):
        def test_fails(self):
            """Worker docstring marker that should not appear in diagnostics."""
            self.fail("expected failure")

    def setUp(self):
        from warp._src.test_runner.events import (
            WorkerEventReporter,
            install_worker_reporter,
        )

        self.queue = RecordingQueue()
        self.reporter = WorkerEventReporter(
            event_queue=self.queue,
            worker_index=0,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        install_worker_reporter(self.reporter)
        self.addCleanup(install_worker_reporter, None)
        self.addCleanup(self.reporter.close)

    def _assert_result_lifecycle(self, result_class):
        cleanup_calls = []
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.PassingTest)
        with mock.patch("gc.collect", side_effect=lambda: cleanup_calls.append("gc")):
            runner = unittest.TextTestRunner(
                resultclass=result_class,
                stream=io.StringIO(),
                verbosity=0,
            )
            result = runner.run(suite)

        self.assertTrue(result.wasSuccessful())
        expected = [
            "test_started",
            "test_outcome",
            "test_cleanup_started",
            "test_stopped",
        ]
        self.assertEqual([event.event.value for event in self.queue.items], expected)
        self.assertEqual(cleanup_calls, ["gc"])
        self.assertEqual(self.queue.items[1].outcome, "OK")
        self.assertGreaterEqual(self.queue.items[1].elapsed_seconds, 0.0)

    def test_parallel_junit_result_emits_test_lifecycle(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult

        self._assert_result_lifecycle(ParallelJunitTestResult)

    def test_parallel_junit_result_preserves_subtest_failures_without_worker_rendering(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult

        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.FailingSubTest)
        runner = unittest.TextTestRunner(
            resultclass=ParallelJunitTestResult,
            stream=io.StringIO(),
            verbosity=0,
        )
        result = runner.run(suite)

        self.assertEqual(len(result.failures), 1)
        self.assertEqual(len(result.errors), 0)
        self.assertEqual(result.test_record[0][3], "FAIL")

    def test_manager_emits_suite_and_test_lifecycles(self):
        from warp._src.test_runner.worker import ParallelTestManager

        manager = mock.Mock()
        manager.Event.return_value = threading.Event()
        args = SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            junit_report_xml=None,
            verbose=0,
        )
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.TwoPassingTests)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ParallelTestManager(manager, args, temp_dir).run_tests(4, suite)

        self.assertEqual(result[0], 2)
        self.assertEqual(
            [event.event.value for event in self.queue.items],
            [
                "suite_started",
                "test_started",
                "test_outcome",
                "test_cleanup_started",
                "test_stopped",
                "test_started",
                "test_outcome",
                "test_cleanup_started",
                "test_stopped",
                "suite_finalizing",
                "suite_finished",
            ],
        )

    def test_manager_suppresses_test_docstrings_in_failure_diagnostics(self):
        from warp._src.test_runner.worker import ParallelTestManager

        args = SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            verbose=0,
        )
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.DocumentedFailure)

        with tempfile.TemporaryDirectory() as temp_dir:
            result = ParallelTestManager(LocalManager(), args, temp_dir).run_tests(0, suite)

        self.assertEqual(len(result[2]), 1)
        self.assertIn("test_fails", result[2][0])
        self.assertNotIn("Worker docstring marker", result[2][0])

    def test_reporter_emit_failure_does_not_deadlock_worker_completion(self):
        from warp._src.test_runner import events
        from warp._src.test_runner.events import WorkerEventReporter, install_worker_reporter
        from warp._src.test_runner.worker import ParallelTestManager

        reporter = WorkerEventReporter(
            event_queue=RaisingQueue(OSError("event queue unavailable")),
            worker_index=0,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        install_worker_reporter(reporter)

        def assert_reporter_uninstalled():
            self.assertIsNone(events.emit_worker_event(events.EventKind.WORKER_STARTED))

        self.addCleanup(assert_reporter_uninstalled)
        self.addCleanup(install_worker_reporter, None)
        self.addCleanup(reporter.close)
        args = SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            junit_report_xml="enabled",
            level="class",
            verbose=0,
        )
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(self.PassingTest)
        results = []
        errors = []
        warnings = []

        def run_worker():
            try:
                with tempfile.TemporaryDirectory() as temp_dir:
                    results.append(ParallelTestManager(LocalManager(), args, temp_dir).run_tests(0, suite))
            except BaseException as error:
                errors.append(error)

        with mock.patch.object(events, "warn", side_effect=warnings.append):
            worker_thread = threading.Thread(target=run_worker, daemon=True)
            worker_thread.start()
            assert_thread_stops(self, worker_thread)

        self.assertEqual(errors, [])
        self.assertEqual(results[0][:6], (1, [], [], 0, 0, 0))
        self.assertEqual(len(results[0][6]), 1)
        self.assertEqual(results[0][6][0][3], "OK")
        self.assertEqual(warnings, ["Failed to send worker event: event queue unavailable"])


class TestParallelJunitFixtures(unittest.TestCase):
    def setUp(self):
        from warp._src.test_runner.events import WorkerEventReporter, install_worker_reporter

        self.queue = RecordingQueue()
        self.reporter = WorkerEventReporter(
            event_queue=self.queue,
            worker_index=0,
            run_dir=None,
            run_start_monotonic_ns=0,
        )
        install_worker_reporter(self.reporter)
        self.addCleanup(install_worker_reporter, None)
        self.addCleanup(self.reporter.close)

    @staticmethod
    def _args():
        return SimpleNamespace(
            buffer=False,
            coverage=False,
            coverage_branch=False,
            failfast=False,
            junit_report_xml="enabled",
            level="module",
            verbose=0,
        )

    def _run_module(self, scenario, source):
        from warp._src.test_runner.worker import ParallelTestManager
        from warp.tests.unittest_utils import write_junit_results

        with temporary_junit_module(scenario, source) as (module, root):
            suite = unittest.defaultTestLoader.loadTestsFromModule(module)
            first_event = len(self.queue.items)
            result = ParallelTestManager(LocalManager(), self._args(), root).run_tests(
                0,
                suite,
                suite_name=module.__name__,
            )
            events = self.queue.items[first_event:]

            junit_path = root / "results.xml"
            write_junit_results(
                str(junit_path),
                result[6],
                sum(record[2] for record in result[6]),
            )
            xml_root = ElementTree.parse(junit_path).getroot()

        return module.__name__, result, events, xml_root

    def _assert_fixture_result(self, scenario, hook, target_kind, outcome, *, ordinary_test=False):
        module_name, result, events, xml_root = self._run_module(scenario, JUNIT_FIXTURE_SOURCES[scenario])
        target = module_name if target_kind == "module" else f"{module_name}.FixtureCase"
        fixture_id = f"{hook} ({target})"

        self.assertEqual(xml_root.tag, "testsuite")
        fixture_records = [record for record in result[6] if record[1] == hook]
        self.assertEqual(len(fixture_records), 1)
        self.assertEqual(fixture_records[0][:4], (target, hook, 0.0, outcome))

        fixture_cases = [case for case in xml_root.findall("testcase") if case.get("name") == hook]
        self.assertEqual(len(fixture_cases), 1)
        fixture_case = fixture_cases[0]
        self.assertEqual(fixture_case.get("classname"), target)
        self.assertEqual(float(fixture_case.get("time")), 0.0)
        if outcome == "ERROR":
            self.assertIsNotNone(fixture_case.find("error"))
            self.assertEqual(len(result[1]), 1)
            self.assertEqual(result[3], 0)
        else:
            self.assertIsNotNone(fixture_case.find("skipped"))
            self.assertEqual(len(result[1]), 0)
            self.assertEqual(result[3], 1)
        self.assertEqual(len(result[2]), 0)

        fixture_events = [event for event in events if event.test_id == fixture_id]
        self.assertEqual(
            [event.event.value for event in fixture_events],
            ["test_started", "test_outcome", "test_cleanup_started", "test_stopped"],
        )
        self.assertEqual(fixture_events[1].outcome, outcome)

        if ordinary_test:
            self.assertEqual(result[0], 1)
            ordinary_records = [record for record in result[6] if record[1] == "test_runs"]
            self.assertEqual(len(ordinary_records), 1)
            ordinary_record = ordinary_records[0]
            self.assertEqual(ordinary_record[0], "FixtureCase")
            self.assertEqual(ordinary_record[3], "OK")
            self.assertGreater(ordinary_record[2], 0.0)

            ordinary_cases = [case for case in xml_root.findall("testcase") if case.get("name") == "test_runs"]
            self.assertEqual(len(ordinary_cases), 1)
            self.assertEqual(ordinary_cases[0].get("classname"), "FixtureCase")
            self.assertEqual(float(ordinary_cases[0].get("time")), ordinary_record[2])

            ordinary_id = f"{module_name}.FixtureCase.test_runs"
            lifecycle_events = [event.event.value for event in events if event.test_id in (ordinary_id, fixture_id)]
            self.assertEqual(
                lifecycle_events,
                [
                    "test_started",
                    "test_outcome",
                    "test_cleanup_started",
                    "test_stopped",
                    "test_started",
                    "test_outcome",
                    "test_cleanup_started",
                    "test_stopped",
                ],
            )
        else:
            self.assertEqual(result[0], 0)
            self.assertEqual(len(result[6]), 1)

    def test_records_set_up_class_error(self):
        self._assert_fixture_result("setUpClass", "setUpClass", "class", "ERROR")

    def test_records_tear_down_class_error_after_normal_test(self):
        self._assert_fixture_result(
            "tearDownClass",
            "tearDownClass",
            "class",
            "ERROR",
            ordinary_test=True,
        )

    def test_records_set_up_module_error(self):
        self._assert_fixture_result("setUpModule", "setUpModule", "module", "ERROR")

    def test_records_tear_down_module_error_after_normal_test(self):
        self._assert_fixture_result(
            "tearDownModule",
            "tearDownModule",
            "module",
            "ERROR",
            ordinary_test=True,
        )

    def test_records_fixture_skip(self):
        self._assert_fixture_result("fixtureSkip", "setUpClass", "class", "SKIP")

    def test_preserves_ordinary_and_subtest_records(self):
        module_name, result, events, xml_root = self._run_module("ordinary", JUNIT_ORDINARY_SOURCE)

        self.assertEqual(result[0], 2)
        self.assertEqual(len(result[1]), 0)
        self.assertEqual(len(result[2]), 1)
        self.assertEqual(result[3], 0)
        self.assertEqual(
            [(record[0], record[1], record[3]) for record in result[6]],
            [
                ("OrdinaryCase", "test_passes", "OK"),
                ("SubtestCase", "test_subtest_failure", "FAIL"),
            ],
        )
        self.assertTrue(all(record[2] > 0.0 for record in result[6]))

        xml_cases = xml_root.findall("testcase")
        self.assertEqual(
            [(case.get("classname"), case.get("name")) for case in xml_cases],
            [
                ("OrdinaryCase", "test_passes"),
                ("SubtestCase", "test_subtest_failure"),
            ],
        )
        self.assertEqual(
            [float(case.get("time")) for case in xml_cases],
            [record[2] for record in result[6]],
        )
        self.assertIsNone(xml_cases[0].find("failure"))
        self.assertIsNotNone(xml_cases[1].find("failure"))

        expected_lifecycles = []
        for class_name, test_name, _ in (
            ("OrdinaryCase", "test_passes", "OK"),
            ("SubtestCase", "test_subtest_failure", "FAIL"),
        ):
            test_id = f"{module_name}.{class_name}.{test_name}"
            test_events = [event for event in events if event.test_id == test_id]
            expected_lifecycles.append(
                (
                    [event.event.value for event in test_events],
                    test_events[1].outcome,
                )
            )
        self.assertEqual(
            expected_lifecycles,
            [
                (["test_started", "test_outcome", "test_cleanup_started", "test_stopped"], "OK"),
                (["test_started", "test_outcome", "test_cleanup_started", "test_stopped"], "FAIL"),
            ],
        )

    def test_records_unknown_test_like_objects_without_changing_test_count(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult, write_junit_results

        class IdentifiedTestLike:
            def id(self):
                return "unknown.fixture.hook"

        class StringOnlyTestLike:
            def __str__(self):
                return "undotted-holder"

        runner = unittest.TextTestRunner(
            resultclass=ParallelJunitTestResult,
            stream=io.StringIO(),
            verbosity=0,
        )
        result = runner._makeResult()
        result.addSkip(IdentifiedTestLike(), "identified skip")
        result.addSkip(StringOnlyTestLike(), "string skip")

        self.assertEqual(result.testsRun, 0)
        self.assertEqual(
            [record[:4] for record in result.test_record],
            [
                ("unknown.fixture", "hook", 0.0, "SKIP"),
                ("StringOnlyTestLike", "undotted-holder", 0.0, "SKIP"),
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            junit_path = pathlib.Path(directory, "unknown.xml")
            write_junit_results(
                str(junit_path),
                result.test_record,
                0.0,
            )
            xml_cases = ElementTree.parse(junit_path).getroot().findall("testcase")

        self.assertEqual(
            [(case.get("classname"), case.get("name"), float(case.get("time"))) for case in xml_cases],
            [
                ("unknown.fixture", "hook", 0.0),
                ("StringOnlyTestLike", "undotted-holder", 0.0),
            ],
        )
        for test_id in ("unknown.fixture.hook", "undotted-holder"):
            test_events = [event for event in self.queue.items if event.test_id == test_id]
            self.assertEqual(
                [event.event.value for event in test_events],
                ["test_started", "test_outcome", "test_cleanup_started", "test_stopped"],
            )
            self.assertEqual(test_events[1].outcome, "SKIP")

    def test_unknown_test_like_object_falls_back_when_id_raises(self):
        from warp.tests.unittest_utils import ParallelJunitTestResult, write_junit_results

        class RaisingIdTestLike:
            def __init__(self):
                self.id_calls = 0

            def id(self):
                self.id_calls += 1
                raise RuntimeError("identifier unavailable")

            def __str__(self):
                return "fallback.fixture.skip"

        runner = unittest.TextTestRunner(
            resultclass=ParallelJunitTestResult,
            stream=io.StringIO(),
            verbosity=0,
        )
        result = runner._makeResult()
        test = RaisingIdTestLike()
        result.addSkip(test, "raising ID skip")

        self.assertEqual(result.testsRun, 0)
        self.assertEqual(result.skipped, [(test, "raising ID skip")])
        self.assertEqual(test.id_calls, 1)
        self.assertEqual(
            [record[:4] for record in result.test_record],
            [("fallback.fixture", "skip", 0.0, "SKIP")],
        )

        with tempfile.TemporaryDirectory() as directory:
            junit_path = pathlib.Path(directory, "raising-id.xml")
            write_junit_results(
                str(junit_path),
                result.test_record,
                0.0,
            )
            xml_cases = ElementTree.parse(junit_path).getroot().findall("testcase")

        self.assertEqual(len(xml_cases), 1)
        self.assertEqual(xml_cases[0].get("classname"), "fallback.fixture")
        self.assertEqual(xml_cases[0].get("name"), "skip")
        self.assertEqual(float(xml_cases[0].get("time")), 0.0)
        self.assertEqual(xml_cases[0].find("skipped").get("message"), "raising ID skip")

        test_events = [event for event in self.queue.items if event.test_id == "fallback.fixture.skip"]
        self.assertEqual(
            [event.event.value for event in test_events],
            ["test_started", "test_outcome", "test_cleanup_started", "test_stopped"],
        )
        self.assertEqual(test_events[1].outcome, "SKIP")
