"""Human-readable progress for the serial, resumable CPU workflow."""

from __future__ import annotations

import time


class WorkflowProgress:
    """Count completed stages; never count a failed or unfinished stage."""

    def __init__(self, total: int):
        if total < 1:
            raise ValueError("workflow must have at least one stage")
        self.total = total
        self.completed = 0
        self._active: str | None = None
        self._started = 0.0

    def start(self, name: str) -> None:
        if self._active is not None or self.completed >= self.total:
            raise RuntimeError("progress stages must run sequentially")
        self._active = name
        self._started = time.monotonic()
        print(f"[{self.completed + 1}/{self.total}] START {name}", flush=True)

    def finish(self, *, reused: bool = False) -> None:
        if self._active is None:
            raise RuntimeError("no progress stage is active")
        name = self._active
        elapsed = time.monotonic() - self._started
        self.completed += 1
        self._active = None
        status = "REUSED" if reused else "DONE"
        print(f"[{self.completed}/{self.total}] {status} {name} "
              f"({elapsed:.1f}s; {100 * self.completed / self.total:.0f}%)",
              flush=True)

    def reuse(self, name: str) -> None:
        self.start(name)
        self.finish(reused=True)

    def detail(self, message: str) -> None:
        """Report activity within a stage without falsely advancing the count."""
        index = self.completed + (self._active is not None)
        print(f"[{index}/{self.total}] {message}", flush=True)

    def check_complete(self) -> None:
        if self._active is not None or self.completed != self.total:
            raise RuntimeError("workflow progress count does not match completed stages")
