"""Thread-safe background jobs for responsive Streamlit workflows."""

from __future__ import annotations

import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass


class JobCancelled(Exception):
    """Raised at a safe checkpoint after a user requests cancellation."""


@dataclass(frozen=True)
class JobSnapshot:
    run_id: str
    state: str
    message: str
    progress: float
    history: tuple[str, ...]
    error: str

    @property
    def active(self) -> bool:
        return self.state in {"running", "cancelling"}


class BackgroundJob:
    """Run one cancellable task and expose immutable progress snapshots."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._run_id = ""
        self._state = "idle"
        self._message = ""
        self._progress = 0.0
        self._history: list[str] = []
        self._error = ""

    def start(self, worker: Callable[["BackgroundJob"], None]) -> str:
        with self._lock:
            if self._state in {"running", "cancelling"}:
                raise RuntimeError("A report-generation job is already active.")
            self._cancel_event = threading.Event()
            self._run_id = uuid.uuid4().hex
            self._state = "running"
            self._message = "מתחיל בבדיקת המקורות"
            self._progress = 0.0
            self._history = [self._message]
            self._error = ""
            run_id = self._run_id
            self._thread = threading.Thread(
                target=self._execute,
                args=(run_id, worker),
                name=f"report-generation-{run_id[:8]}",
                daemon=True,
            )
            self._thread.start()
            return run_id

    def _execute(self, run_id: str, worker: Callable[["BackgroundJob"], None]) -> None:
        try:
            worker(self)
            self.checkpoint()
        except JobCancelled:
            with self._lock:
                if self._run_id == run_id:
                    self._state = "cancelled"
                    self._message = "העיבוד הופסק לבקשת המשתמש"
                    self._history.append(self._message)
        except Exception as exc:
            with self._lock:
                if self._run_id == run_id:
                    self._state = "failed"
                    self._message = "הפקת הדוח נכשלה"
                    self._error = str(exc)
                    self._history.append(self._message)
        else:
            with self._lock:
                if self._run_id == run_id:
                    self._state = "complete"
                    self._message = "הדוח נשמר ומוכן לפתיחה"
                    self._progress = 1.0
                    self._history.append(self._message)

    def update(self, message: str, progress: float) -> None:
        with self._lock:
            if self._state not in {"running", "cancelling"}:
                return
            if self._state == "running":
                self._message = message
            self._progress = min(1.0, max(self._progress, progress))
            if not self._history or self._history[-1] != message:
                self._history.append(message)
                self._history = self._history[-12:]

    def cancel(self) -> None:
        with self._lock:
            if self._state != "running":
                return
            self._cancel_event.set()
            self._state = "cancelling"
            self._message = "התקבלה בקשת עצירה; ממתין לסיום הפעולה הפעילה"
            self._history.append(self._message)

    def cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    def checkpoint(self) -> None:
        if self.cancel_requested():
            raise JobCancelled()

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(
                run_id=self._run_id,
                state=self._state,
                message=self._message,
                progress=self._progress,
                history=tuple(self._history),
                error=self._error,
            )


class BackgroundJobRegistry:
    """Keep one report-generation job per Training Workspace."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._jobs: dict[str, BackgroundJob] = {}

    def for_training(self, training_id: str) -> BackgroundJob:
        with self._lock:
            return self._jobs.setdefault(training_id, BackgroundJob())
