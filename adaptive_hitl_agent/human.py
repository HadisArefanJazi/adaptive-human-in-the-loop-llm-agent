from __future__ import annotations

from typing import Protocol

from .types import Task


class HumanReviewer(Protocol):
    """Backend called when the router requests a human answer."""

    def answer(self, task: Task) -> str:
        ...


class OracleHumanReviewer:
    """Idealized benchmark reviewer; reads the reference answer, not the question."""

    def answer(self, task: Task) -> str:
        return task.reference_answer
