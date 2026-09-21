from __future__ import annotations

import math

from .records import immutable_record
from .human import human_reviewer
from .llm import language_model
from .retrieval import bm25_retriever, retrieved_document
from .tools import calculator_error, safe_calculator
from .types import action_type, observation_record, resource_usage, step_result, task_record, is_correct


feature_names = (
    "direct_confidence",
    "math_signal",
    "ambiguity_signal",
    "query_length",
    "has_retrieval",
    "has_tool_output",
    "step_progress",
)


class reward_config_type(immutable_record):
    success_reward: float = 1.0
    failure_penalty: float = -0.25
    token_cost: float = 0.002
    retrieval_cost: float = 0.08
    tool_cost: float = 0.04
    human_cost: float = 0.45
    latency_cost: float = 0.02

    __match_args__ = (
        "success_reward",
        "failure_penalty",
        "token_cost",
        "retrieval_cost",
        "tool_cost",
        "human_cost",
        "latency_cost",
    )

    def __init__(
        self,
        success_reward: float = 1.0,
        failure_penalty: float = -0.25,
        token_cost: float = 0.002,
        retrieval_cost: float = 0.08,
        tool_cost: float = 0.04,
        human_cost: float = 0.45,
        latency_cost: float = 0.02,
    ) -> None:
        object.__setattr__(self, "success_reward", success_reward)
        object.__setattr__(self, "failure_penalty", failure_penalty)
        object.__setattr__(self, "token_cost", token_cost)
        object.__setattr__(self, "retrieval_cost", retrieval_cost)
        object.__setattr__(self, "tool_cost", tool_cost)
        object.__setattr__(self, "human_cost", human_cost)
        object.__setattr__(self, "latency_cost", latency_cost)
        self._validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "success_reward": self.success_reward,
            "failure_penalty": self.failure_penalty,
            "token_cost": self.token_cost,
            "retrieval_cost": self.retrieval_cost,
            "tool_cost": self.tool_cost,
            "human_cost": self.human_cost,
            "latency_cost": self.latency_cost,
        }

    def _validate(self) -> None:
        for name, value in self.as_dict().items():
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite")
            if name.endswith("_cost") and value < 0:
                raise ValueError(f"{name} cannot be negative")


class assistance_environment:
    """a routing episode: gather evidence, then answer or request review.

    each resource action can run once. the last step is reserved for an answer
    or human review, so even an empty retrieval result cannot strand an episode.
    """

    _ambiguity_terms = (
        "ambiguous",
        "approve",
        "choose for me",
        "missing",
        "no criteria",
        "no preference",
        "personal judgment",
        "sensitive",
        "should i",
        "which one should",
    )

    def __init__(
        self,
        task: task_record,
        retriever: bm25_retriever,
        language_model: language_model,
        calculator: safe_calculator,
        human_reviewer: human_reviewer,
        reward_config: reward_config_type | None = None,
        max_steps: int = 3,
        retrieval_top_k: int = 2,
    ) -> None:
        if max_steps < 1:
            raise ValueError("max_steps must be at least 1")
        if retrieval_top_k < 1:
            raise ValueError("retrieval_top_k must be at least 1")

        self.task = task
        self.retriever = retriever
        self.language_model = language_model
        self.calculator = calculator
        self.human_reviewer = human_reviewer
        self.reward_config = reward_config or reward_config_type()
        self.max_steps = max_steps
        self.retrieval_top_k = retrieval_top_k

        self.documents: list[retrieved_document] = []
        self.tool_output: str | None = None
        self.usage = resource_usage()
        self.steps = 0
        self.done = False
        self.answer: str | None = None
        self.action_history: list[action_type] = []

    def observe(self) -> observation_record:
        direct_confidence = self.language_model.confidence(
            self.task.question,
            self.documents,
            self.tool_output,
        )
        lowered = self.task.question.casefold()
        # whitespace is allowed between the digits and a subtraction sign.
        compact = "".join(character for character in lowered if not character.isspace())
        has_subtraction = any(
            compact[index] == "-"
            and compact[index - 1].isdecimal()
            and compact[index + 1].isdecimal()
            for index in range(1, len(compact) - 1)
        )
        math_signal = float(
            any(character.isdecimal() for character in lowered)
            and (any(symbol in lowered for symbol in "+*/%") or has_subtraction)
        )
        ambiguity_signal = float(any(term in lowered for term in self._ambiguity_terms))
        query_length = min(1.0, len(self.task.question.split()) / 20.0)
        step_progress = self.steps / self.max_steps
        features = (
            direct_confidence,
            math_signal,
            ambiguity_signal,
            query_length,
            float(bool(self.documents)),
            float(self.tool_output is not None),
            step_progress,
        )
        return observation_record(
            features=features,
            feature_names=feature_names,
            direct_confidence=direct_confidence,
            math_signal=math_signal,
            ambiguity_signal=ambiguity_signal,
            has_retrieval=bool(self.documents),
            has_tool_output=self.tool_output is not None,
            step_index=self.steps,
        )

    def available_actions(self) -> tuple[action_type, ...]:
        if self.done:
            return ()

        actions = [action_type.answer_directly, action_type.ask_human]
        if self.steps < self.max_steps - 1:
            if self.usage.retrieval_calls == 0:
                actions.append(action_type.retrieve)
            if self.usage.tool_calls == 0:
                actions.append(action_type.use_tool)
        return tuple(sorted(actions, key=int))

    def step(self, action: action_type) -> step_result:
        if self.done:
            raise RuntimeError("the episode has already terminated")
        action = action_type(action)
        if action not in self.available_actions():
            raise ValueError(f"action_type {action.name} is not available in the current state")

        before = self.usage.copy()
        self.steps += 1
        self.action_history.append(action)
        info: dict[str, object] = {}

        if action is action_type.retrieve:
            self.documents = self.retriever.retrieve(
                self.task.question,
                top_k=self.retrieval_top_k,
            )
            self.usage.tokens += sum(
                len(item.document.text.split()) for item in self.documents
            )
            self.usage.retrieval_calls += 1
            self.usage.latency_units += 1.0
            info["documents"] = [item.document.doc_id for item in self.documents]

        elif action is action_type.use_tool:
            try:
                self.tool_output = self.calculator.calculate(self.task.question)
            except calculator_error as error:
                self.tool_output = "tool_error"
                info["tool_error"] = str(error)
            self.usage.tokens += 4
            self.usage.tool_calls += 1
            self.usage.latency_units += 0.5
            info["tool_output"] = self.tool_output

        elif action is action_type.ask_human:
            self.answer = self.human_reviewer.answer(self.task)
            self.usage.tokens += 5
            self.usage.human_calls += 1
            self.usage.latency_units += 5.0
            self.done = True

        elif action is action_type.answer_directly:
            model_answer = self.language_model.answer(
                self.task.question,
                self.documents,
                self.tool_output,
            )
            self.answer = model_answer.text
            self.usage.tokens += (
                len(self.task.question.split())
                + max(1, len(model_answer.text.split()))
                + 2
            )
            self.usage.latency_units += 0.5
            info["model_confidence"] = model_answer.confidence
            self.done = True

        if not self.done and self.steps >= self.max_steps:
            raise RuntimeError("the maximum step count was reached without a terminal action")

        delta = self.usage.delta(before)
        reward = -self._resource_penalty(delta)
        correct: bool | None = None
        if self.done:
            assert self.answer is not None
            correct = is_correct(self.answer, self.task.acceptable_answers)
            reward += (
                self.reward_config.success_reward
                if correct
                else self.reward_config.failure_penalty
            )

        return step_result(
            observation=None if self.done else self.observe(),
            reward=reward,
            done=self.done,
            answer=self.answer,
            correct=correct,
            action=action,
            info=info,
        )

    def _resource_penalty(self, usage: resource_usage) -> float:
        config = self.reward_config
        return (
            config.token_cost * usage.tokens
            + config.retrieval_cost * usage.retrieval_calls
            + config.tool_cost * usage.tool_calls
            + config.human_cost * usage.human_calls
            + config.latency_cost * usage.latency_units
        )

    @property
    def total_resource_penalty(self) -> float:
        return self._resource_penalty(self.usage)
