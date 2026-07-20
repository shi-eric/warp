FIXTURE_CASES_SOURCE = """import gc
import os
import pathlib
import time
import unittest

MARKER_DIR = pathlib.Path(os.environ["WARP_RUNNER_FIXTURE_DIR"])


class BlockingCase(unittest.TestCase):
    def test_wait_for_pool_shutdown(self):
        MARKER_DIR.joinpath("blocking-started").write_text("ready", encoding="utf-8")
        deadline = time.monotonic() + 30.0
        while not MARKER_DIR.joinpath("release-blocker").exists():
            if time.monotonic() >= deadline:
                self.fail("Parent did not terminate the blocked worker")
            time.sleep(0.01)


class PassingCase(unittest.TestCase):
    def test_pass(self):
        MARKER_DIR.joinpath("passing-complete").write_text("ready", encoding="utf-8")


class HardExitCase(unittest.TestCase):
    def test_exit(self):
        deadline = time.monotonic() + 30.0
        marker = MARKER_DIR.joinpath("passing-complete")
        while not marker.exists():
            if time.monotonic() >= deadline:
                self.fail("Passing suite did not complete")
            time.sleep(0.01)
        os.write(2, b"hard-exit-marker\\n")
        os._exit(86)


class AbortCase(unittest.TestCase):
    def test_abort(self):
        attempts_path = MARKER_DIR / "abort-attempts"
        attempts = int(attempts_path.read_text(encoding="utf-8")) if attempts_path.exists() else 0
        attempts_path.write_text(str(attempts + 1), encoding="utf-8")
        os.write(2, b"abort-marker\\n")
        os.abort()


class CleanupAbortCase(unittest.TestCase):
    def test_abort_during_gc(self):
        gc.collect = os.abort


class FinalizationAbortCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.abort()

    def test_completes_before_class_finalization(self):
        pass


class OrdinaryFailureCase(unittest.TestCase):
    def test_failure(self):
        self.fail("ordinary assertion marker")


class ShouldNotStartCase(unittest.TestCase):
    def test_not_started_after_failfast(self):
        MARKER_DIR.joinpath("unexpected-start").write_text("started", encoding="utf-8")


class OutputPressureCase(unittest.TestCase):
    def test_output_pressure(self):
        chunk = b"output-pressure-marker " + b"x" * 4060 + b"\\n"
        for _ in range(256):
            os.write(1, chunk)
            os.write(2, chunk)


class FirstPidCase(unittest.TestCase):
    def test_record_pid(self):
        MARKER_DIR.joinpath("first-pid").write_text(str(os.getpid()), encoding="utf-8")


class SecondPidCase(unittest.TestCase):
    def test_record_pid(self):
        MARKER_DIR.joinpath("second-pid").write_text(str(os.getpid()), encoding="utf-8")
"""


JUNIT_FIXTURE_SOURCES = {
    "setUpClass": """import unittest


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise RuntimeError("class setup failed")

    def test_runs(self):
        pass
""",
    "tearDownClass": """import time
import unittest


class FixtureCase(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        raise RuntimeError("class teardown failed")

    def test_runs(self):
        time.sleep(0.02)
""",
    "setUpModule": """import unittest


def setUpModule():
    raise RuntimeError("module setup failed")


class FixtureCase(unittest.TestCase):
    def test_runs(self):
        pass
""",
    "tearDownModule": """import time
import unittest


def tearDownModule():
    raise RuntimeError("module teardown failed")


class FixtureCase(unittest.TestCase):
    def test_runs(self):
        time.sleep(0.02)
""",
    "fixtureSkip": """import unittest


class FixtureCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("class fixture skipped")

    def test_runs(self):
        pass
""",
}


JUNIT_ORDINARY_SOURCE = """import time
import unittest


class OrdinaryCase(unittest.TestCase):
    def test_passes(self):
        time.sleep(0.02)


class SubtestCase(unittest.TestCase):
    def test_subtest_failure(self):
        time.sleep(0.02)
        with self.subTest(value=1):
            self.fail("subtest failure")
"""
