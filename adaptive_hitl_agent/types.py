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

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "Task":
        answers = payload.get("acceptable_answers")
        if answers is None:
            answers = [payload["answer"]]

        acceptable_answers = []
        for answer in answers:
            acceptable_answers.append(str(answer))

        return cls(
            task_id=str(payload["id"]),
            question=str(payload["question"]),
            acceptable_answers=tuple(acceptable_answers),
            split=str(payload["split"]),
            category=str(payload["category"]),
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
        return ResourceUsage(
            tokens=self.tokens,
            retrieval_calls=self.retrieval_calls,
            tool_calls=self.tool_calls,
            human_calls=self.human_calls,
            latency_units=self.latency_units,
        )

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
    retrieval_signal: float
    math_signal: float
    ambiguity_signal: float
    has_retrieval: bool
    has_tool_output: bool
    step_index: int

    def as_dict(self) -> dict[str, float]:
        result = {}
        for name, value in zip(self.feature_names, self.features):
            result[name] = value
        return result


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
    return normalized.strip(" .-")


def is_correct(answer: str, acceptable_answers: tuple[str, ...]) -> bool:
    normalized = normalize_answer(answer)
    for candidate in acceptable_answers:
        if normalized == normalize_answer(candidate):
            return True
    return False
