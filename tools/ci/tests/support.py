import concurrent.futures
import contextlib
import importlib
import os
import pathlib
import sys
import tempfile
import threading
from types import SimpleNamespace
from unittest import mock

from .fixture_sources import FIXTURE_CASES_SOURCE


@contextlib.contextmanager
def temporary_fixture_cases():
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        import_directory = root / "importable"
        marker_directory = root / "markers"
        import_directory.mkdir()
        marker_directory.mkdir()
        import_directory.joinpath("fixture_cases.py").write_text(FIXTURE_CASES_SOURCE, encoding="utf-8")

        import_path = str(import_directory)
        with mock.patch.dict(os.environ, {"WARP_RUNNER_FIXTURE_DIR": str(marker_directory)}, clear=False):
            sys.path.insert(0, import_path)
            importlib.invalidate_caches()
            module = importlib.import_module("fixture_cases")
            try:
                yield module, marker_directory, root
            finally:
                sys.modules.pop("fixture_cases", None)
                try:
                    sys.path.remove(import_path)
                except ValueError:
                    pass
                importlib.invalidate_caches()


@contextlib.contextmanager
def temporary_junit_module(scenario, source):
    with tempfile.TemporaryDirectory() as directory:
        root = pathlib.Path(directory)
        module_name = f"junit_fixture_{scenario.lower()}"
        root.joinpath(f"{module_name}.py").write_text(source, encoding="utf-8")

        import_path = str(root)
        sys.path.insert(0, import_path)
        importlib.invalidate_caches()
        module = importlib.import_module(module_name)
        try:
            yield module, root
        finally:
            sys.modules.pop(module_name, None)
            try:
                sys.path.remove(import_path)
            except ValueError:
                pass
            importlib.invalidate_caches()


class RecordingQueue:
    def __init__(self):
        self.items = []

    def put(self, item):
        self.items.append(item)


class RaisingQueue:
    def __init__(self, exception):
        self.exception = exception

    def put(self, item):
        raise self.exception


def assert_thread_stops(test_case, thread, timeout=5.0):
    thread.join(timeout)
    test_case.assertFalse(thread.is_alive())


def publish_large_events(queue, count, payload_size):
    payload = "x" * payload_size
    for index in range(count):
        queue.put({"index": index, "payload": payload})


class LocalManager:
    @staticmethod
    def Event():
        return threading.Event()

    @staticmethod
    def Lock():
        return threading.Lock()

    @staticmethod
    def Value(value_type, initial_value):
        return SimpleNamespace(value=initial_value)


class StubExecutor:
    def __init__(self, submit_results, manager_thread=None):
        self._submit_results = iter(submit_results)
        self._executor_manager_thread = manager_thread
        self._processes = {}

    def submit(self, function, index, suite):
        result = next(self._submit_results)
        if isinstance(result, BaseException):
            raise result
        return result

    def shutdown(self, wait=True, cancel_futures=False):
        self._executor_manager_thread = None


class NonCancellingFuture(concurrent.futures.Future):
    def cancel(self):
        return False


class PublishingManagerThread:
    def __init__(self, future, result):
        self.future = future
        self.result = result
        self.joined = False

    def join(self):
        self.joined = True
        self.future.set_result(self.result)


class StubSinkFile:
    def __init__(self, descriptor):
        self.descriptor = descriptor
        self.closed = False

    def fileno(self):
        return self.descriptor

    def close(self):
        self.closed = True
