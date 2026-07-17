"""Single-worker-thread job serialization for the designlab Generate button.

Exactly one generate job runs at a time, on one background thread; a
submission while a job is running is rejected (``JobBusyError``) rather
than queued, matching the roadmap's Phase 2 decision to keep this simple
and add cancel/parallelism only if measurement later shows it is needed.
Driveable without a GUI: tests submit a job function directly.
"""

from __future__ import annotations

import queue
import sys
import threading
from collections.abc import Callable


class JobBusyError(Exception):
    """A job was submitted while another job was still running."""


class JobWorker:
    """Runs at most one submitted job at a time on a dedicated thread."""

    def __init__(
        self, *, on_error: Callable[[BaseException], None] | None = None
    ) -> None:
        self._lock = threading.Lock()
        self._busy = False
        self._seq = 0
        self._queue: queue.Queue[Callable[[], None]] = queue.Queue()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._started = False
        self._on_error = on_error or self._default_on_error

    @staticmethod
    def _default_on_error(error: BaseException) -> None:
        print(f"JOB ERROR {error}", file=sys.stderr)

    def start(self) -> None:
        """Start the background thread. Safe to call at most once."""
        if self._started:
            return
        self._started = True
        self._thread.start()

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._busy

    def next_seq(self) -> int:
        """Return a fresh, monotonically increasing job sequence number."""
        with self._lock:
            self._seq += 1
            return self._seq

    def submit(self, job: Callable[[], None]) -> None:
        """Run ``job`` on the worker thread; raise if one is already running.

        ``job`` is responsible for reporting its own outcome (e.g. via a
        closure over a status callback); this class only serializes
        execution and tracks busy/idle.
        """
        with self._lock:
            if self._busy:
                raise JobBusyError("a generate job is already running")
            self._busy = True
        self._queue.put(job)

    def _run(self) -> None:
        while True:
            job = self._queue.get()
            try:
                job()
            except Exception as error:  # last-resort safety net; keep the thread alive
                self._on_error(error)
            finally:
                with self._lock:
                    self._busy = False


__all__ = ["JobBusyError", "JobWorker"]
