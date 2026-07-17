"""Browser-free tests for the single-worker job serializer."""

import sys
import threading
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parents[2] / "examples" / "designlab"
sys.path.insert(0, str(EXAMPLES_DIR))

from designlab_jobs import JobBusyError, JobWorker  # noqa: E402


def test_submit_runs_job_on_worker_thread() -> None:
    worker = JobWorker()
    worker.start()
    done = threading.Event()
    seen_thread: list[threading.Thread] = []

    def job() -> None:
        seen_thread.append(threading.current_thread())
        done.set()

    worker.submit(job)
    assert done.wait(timeout=5.0)
    assert seen_thread[0] is not threading.current_thread()


def test_submit_rejects_while_busy() -> None:
    worker = JobWorker()
    worker.start()
    release = threading.Event()
    started = threading.Event()

    def blocking_job() -> None:
        started.set()
        release.wait(timeout=5.0)

    worker.submit(blocking_job)
    assert started.wait(timeout=5.0)
    assert worker.busy is True

    with pytest.raises(JobBusyError):
        worker.submit(lambda: None)

    release.set()


def test_busy_clears_after_job_completes() -> None:
    worker = JobWorker()
    worker.start()
    done = threading.Event()
    worker.submit(done.set)
    assert done.wait(timeout=5.0)

    # Poll briefly: `busy` clears in a `finally` right after the job
    # function returns, which can race this assertion by a few microseconds.
    for _ in range(1000):
        if not worker.busy:
            break
    assert worker.busy is False


def test_busy_clears_and_on_error_fires_after_job_raises() -> None:
    seen_errors: list[BaseException] = []
    done = threading.Event()

    def on_error(error: BaseException) -> None:
        seen_errors.append(error)
        done.set()

    worker = JobWorker(on_error=on_error)
    worker.start()

    def failing_job() -> None:
        raise RuntimeError("boom")

    worker.submit(failing_job)
    assert done.wait(timeout=5.0)
    assert isinstance(seen_errors[0], RuntimeError)
    for _ in range(1000):
        if not worker.busy:
            break
    assert worker.busy is False

    # The worker thread survived the failure and can run another job.
    ran_again = threading.Event()
    worker.submit(ran_again.set)
    assert ran_again.wait(timeout=5.0)


def test_next_seq_is_monotonically_increasing() -> None:
    worker = JobWorker()
    assert [worker.next_seq() for _ in range(3)] == [1, 2, 3]
