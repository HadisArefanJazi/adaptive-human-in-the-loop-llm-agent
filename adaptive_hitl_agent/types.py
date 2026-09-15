from __future__ import annotations

from enum import IntEnum as int_enum
from typing import Any as any_type

from .records import immutable_record, missing, mutable_record


class action_type(int_enum):
    """actions available to the routing policy."""

    answer_directly = 0
    retrieve = 1
    use_tool = 2
    ask_human = 3


class task_record(immutable_record):
    task_id: str
    question: str
    acceptable_answers: tuple[str, ...]
    split: str
    category: str

    __match_args__ = (
        "task_id",
        "question",
        "acceptable_answers",
        "split",
        "category",
    )

    def __init__(
        self,
        task_id: str,
        question: str,
        acceptable_answers: tuple[str, ...],
        split: str,
        category: str,
    ) -> None:
        object.__setattr__(self, "task_id", task_id)
        object.__setattr__(self, "question", question)
        object.__setattr__(self, "acceptable_answers", acceptable_answers)
        object.__setattr__(self, "split", split)
        object.__setattr__(self, "category", category)
        self._validate()

    def _validate(self) -> None:
        for name in ("task_id", "question", "category"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"task_record {name} must be a non-empty string")
        if (
            not isinstance(self.acceptable_answers, tuple)
            or not self.acceptable_answers
            or any(
                not isinstance(answer, str) or not answer.strip()
                for answer in self.acceptable_answers
            )
        ):
            raise ValueError("a task needs a tuple of non-empty reference strings")
        if self.split not in ("train", "test"):
            raise ValueError("task_record split must be 'train' or 'test'")

    @classmethod
    def from_dict(cls, payload: dict[str, any_type]) -> "task_record":
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


class resource_usage(mutable_record):
    tokens: int = 0
    retrieval_calls: int = 0
    tool_calls: int = 0
    human_calls: int = 0
    latency_units: float = 0.0

    __match_args__ = (
        "tokens",
        "retrieval_calls",
        "tool_calls",
        "human_calls",
        "latency_units",
    )

    def __init__(
        self,
        tokens: int = 0,
        retrieval_calls: int = 0,
        tool_calls: int = 0,
        human_calls: int = 0,
        latency_units: float = 0.0,
    ) -> None:
        self.tokens = tokens
        self.retrieval_calls = retrieval_calls
        self.tool_calls = tool_calls
        self.human_calls = human_calls
        self.latency_units = latency_units

    def copy(self) -> "resource_usage":
        return resource_usage(**self.as_dict())

    def delta(self, earlier: "resource_usage") -> "resource_usage":
        return resource_usage(
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


class model_answer(immutable_record):
    text: str
    confidence: float

    __match_args__ = (
        "text",
        "confidence",
    )

    def __init__(
        self,
        text: str,
        confidence: float,
    ) -> None:
        object.__setattr__(self, "text", text)
        object.__setattr__(self, "confidence", confidence)


class observation_record(immutable_record):
    features: tuple[float, ...]
    feature_names: tuple[str, ...]
    direct_confidence: float
    math_signal: float
    ambiguity_signal: float
    has_retrieval: bool
    has_tool_output: bool
    step_index: int

    __match_args__ = (
        "features",
        "feature_names",
        "direct_confidence",
        "math_signal",
        "ambiguity_signal",
        "has_retrieval",
        "has_tool_output",
        "step_index",
    )

    def __init__(
        self,
        features: tuple[float, ...],
        feature_names: tuple[str, ...],
        direct_confidence: float,
        math_signal: float,
        ambiguity_signal: float,
        has_retrieval: bool,
        has_tool_output: bool,
        step_index: int,
    ) -> None:
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "direct_confidence", direct_confidence)
        object.__setattr__(self, "math_signal", math_signal)
        object.__setattr__(self, "ambiguity_signal", ambiguity_signal)
        object.__setattr__(self, "has_retrieval", has_retrieval)
        object.__setattr__(self, "has_tool_output", has_tool_output)
        object.__setattr__(self, "step_index", step_index)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.features, strict=True))


class step_result(immutable_record):
    observation: observation_record | None
    reward: float
    done: bool
    answer: str | None
    correct: bool | None
    action: action_type
    info: dict[str, any_type]

    __match_args__ = (
        "observation",
        "reward",
        "done",
        "answer",
        "correct",
        "action",
        "info",
    )

    def __init__(
        self,
        observation: observation_record | None,
        reward: float,
        done: bool,
        answer: str | None,
        correct: bool | None,
        action: action_type,
        info: dict[str, any_type] = missing,
    ) -> None:
        object.__setattr__(self, "observation", observation)
        object.__setattr__(self, "reward", reward)
        object.__setattr__(self, "done", done)
        object.__setattr__(self, "answer", answer)
        object.__setattr__(self, "correct", correct)
        object.__setattr__(self, "action", action)
        object.__setattr__(self, "info", {} if info is missing else info)


def normalize_answer(text: str) -> str:
    """normalize short answers for deterministic benchmark scoring."""

    lowered = text.casefold().replace("°", "")
    cleaned = "".join(
        character if character.isalnum() or character in "_#.-" else " "
        for character in lowered
    )
    normalized = " ".join(cleaned.split())
    # keep leading signs and decimal points: -5 and .5 are not 5.
    return normalized.strip().rstrip(".")


def is_correct(answer: str, acceptable_answers: tuple[str, ...]) -> bool:
    normalized = normalize_answer(answer)
    return bool(normalized) and any(
        normalized == normalize_answer(candidate) for candidate in acceptable_answers
    )
