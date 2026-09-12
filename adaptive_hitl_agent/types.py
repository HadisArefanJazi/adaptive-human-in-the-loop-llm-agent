from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Any


class Action(IntEnum):
    """Actions available to the routing policy."""

    ANSWER_DIRECTLY = 0
    RETRIEVE = 1
    USE_TOOL = 2
    ASK_HUMAN = 3


@dataclass(frozen=True)
class Task:
    task_id: str
    question: str
    acceptable_answers: tuple[str, ...]
    split: str
    category: str

    def __post_init__(self) -> None:
        for name in ("task_id", "question", "category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Task {name} must be a non-empty string")
        if (
            not isinstance(self.acceptable_answers, tuple)
            or not self.acceptable_answers
            or any(
                not isinstance(answer, str) or not answer.strip()
                for answer in self.acceptable_answers
            )
        ):
            raise ValueError("A task needs a tuple of non-empty reference strings")
        if self.split not in ("train", "test"):
            raise ValueError("Task split must be 'train' or 'test'")

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        answers = payload.get("acceptable_answers")
        if "acceptable_answers" not in payload:
            answers = [payload["answer"]]
        if not isinstance(answers, (list, tuple)):
            raise ValueError("acceptable_answers must be a list of answers")

        return cls(
            task_id=payload["id"],
            question=payload["question"],
            acceptable_answers=tuple(answers),
            split=payload["split"],
            category=payload["category"],
        )

    @property
    def reference_answer(self) -> str:
        return self.acceptable_answers[0]


@dataclass
class ResourceUsage:
    tokens: int = 0
    retrieval_calls: int = 0
    tool_calls: int = 0
    human_calls: int = 0
    latency_units: float = 0.0

    def copy(self) -> "ResourceUsage":
        return ResourceUsage(**self.as_dict())

    def delta(self, earlier: "ResourceUsage") -> "ResourceUsage":
        return ResourceUsage(
            tokens=self.tokens - earlier.tokens,
            retrieval_calls=self.retrieval_calls - earlier.retrieval_calls,
            tool_calls=self.tool_calls - earlier.tool_calls,
            human_calls=self.human_calls - earlier.human_calls,
            latency_units=self.latency_units - earlier.latency_units,
        )

    def as_dict(self) -> dict[str, int | float]:
        return {
            "tokens": self.tokens,
            "retrieval_calls": self.retrieval_calls,
            "tool_calls": self.tool_calls,
            "human_calls": self.human_calls,
            "latency_units": self.latency_units,
        }


@dataclass(frozen=True)
class ModelAnswer:
    text: str
    confidence: float


@dataclass(frozen=True)
class Observation:
    features: tuple[float, ...]
    feature_names: tuple[str, ...]
    direct_confidence: float
    math_signal: float
    ambiguity_signal: float
    has_retrieval: bool
    has_tool_output: bool
    step_index: int

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.features, strict=True))


@dataclass(frozen=True)
class StepResult:
    observation: Observation | None
    reward: float
    done: bool
    answer: str | None
    correct: bool | None
    action: Action
    info: dict[str, Any] = field(default_factory=dict)


def normalize_answer(text: str) -> str:
    """Normalize short answers for deterministic benchmark scoring."""

    lowered = text.casefold().replace("°", "")
    words = re.findall(r"[\w#.-]+", lowered)
    normalized = " ".join(words)
    # Keep leading signs and decimal points: -5 and .5 are not 5.
    return normalized.strip().rstrip(".")


def is_correct(answer: str, acceptable_answers: tuple[str, ...]) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and any(
        normalized == normalize_answer(candidate) for candidate in acceptable_answers
    )
