"""Bounded concurrency settings for external I/O work."""

from __future__ import annotations

import os


def worker_count(environment_name: str, default: int, task_count: int) -> int:
    if task_count <= 0:
        return 1
    try:
        configured = int(os.getenv(environment_name, str(default)))
    except ValueError:
        configured = default
    return max(1, min(configured, task_count))
