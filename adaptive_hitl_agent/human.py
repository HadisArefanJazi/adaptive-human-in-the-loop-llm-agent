from __future__ import annotations

from typing import Protocol as protocol

from .types import task_record


class human_reviewer(protocol):
    """backend called when the router requests a human answer."""

    def answer(self, task: task_record) -> str:
        ...


class oracle_human_reviewer:
    """idealized benchmark reviewer; reads the reference answer, not the question."""

    def answer(self, task: task_record) -> str:
        return task.reference_answer
