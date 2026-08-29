from __future__ import annotations

import re
from dataclasses import dataclass

from .llm import LanguageModel
from .retrieval import BM25Retriever, RetrievedDocument
from .tools import CalculatorError, SafeCalculator
from .types import (
    Action,
    Observation,
    ResourceUsage,
    StepResult,
    Task,
    is_correct,
)


FEATURE_NAMES = (
    "direct_confidence",
    "retrieval_signal",
    "math_signal",
    "ambiguity_signal",
    "query_length",
    "has_retrieval",
    "has_tool_output",
    "step_progress",
    "remaining_budget",
)


@dataclass(frozen=True)
class RewardConfig:
    success_reward: float = 1.0
    failure_penalty: float = -0.25
    token_cost: float = 0.002
    retrieval_cost: float = 0.08
    tool_cost: float = 0.04
    human_cost: float = 0.45
    latency_cost: float = 0.02


class AssistanceEnvironment:
    """A small MDP for resource-aware assistance routing.

    Retrieval and tool calls enrich the state. Direct answers and human
    escalation terminate the episode. This allows trajectories such as
    RETRIEVE -> ANSWER_DIRECTLY and USE_TOOL -> ANSWER_DIRECTLY.
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
        task: Task,
        retriever: BM25Retriever,
        language_model: LanguageModel,
        calculator: SafeCalculator,
        reward_config: RewardConfig | None = None,
        max_steps: int = 3,
        retrieval_top_k: int = 2,
    ) -> None:
        if max_steps < 3:
            raise ValueError("max_steps must be at least 3")
        self.task = task
        self.retriever = retriever
        self.language_model = language_model
        self.calculator = calculator
        self.reward_config = reward_config or RewardConfig()
        self.max_steps = max_steps
        self.retrieval_top_k = retrieval_top_k
        self.documents: list[RetrievedDocument] = []
        self.tool_output: str | None = None
        self.usage = ResourceUsage()
        self.steps = 0
        self.done = False
        self.answer: str | None = None
        self.action_history: list[Action] = []

    def observe(self) -> Observation:
        direct_confidence = self.language_model.confidence(
            self.task.question,
            self.documents,
            self.tool_output,
        )
        retrieval_signal = self.retriever.signal(self.task.question)
        lowered = self.task.question.casefold()
        math_signal = float(
            bool(re.search(r"\d", lowered))
            and bool(re.search(r"[+*/%]|\d\s*-\s*\d", lowered))
        )
        ambiguity_signal = float(any(term in lowered for term in self._ambiguity_terms))
        query_length = min(1.0, len(self.task.question.split()) / 20.0)
        remaining_budget = max(0.0, 1.0 - self.steps / self.max_steps)
        features = (
            direct_confidence,
            retrieval_signal,
            math_signal,
            ambiguity_signal,
            query_length,
            float(bool(self.documents)),
            float(self.tool_output is not None),
            self.steps / self.max_steps,
            remaining_budget,
        )
        return Observation(
            features=features,
            feature_names=FEATURE_NAMES,
            direct_confidence=direct_confidence,
            retrieval_signal=retrieval_signal,
            math_signal=math_signal,
            ambiguity_signal=ambiguity_signal,
            has_retrieval=bool(self.documents),
            has_tool_output=self.tool_output is not None,
            step_index=self.steps,
        )

    def available_actions(self) -> tuple[Action, ...]:
        if self.done:
            return ()
        actions = [Action.ANSWER_DIRECTLY, Action.ASK_HUMAN]
        if not self.documents:
            actions.append(Action.RETRIEVE)
        if self.tool_output is None:
            actions.append(Action.USE_TOOL)
        return tuple(sorted(actions, key=int))

    def step(self, action: Action) -> StepResult:
        if self.done:
            raise RuntimeError("The episode has already terminated")
        if action not in self.available_actions():
            raise ValueError(f"Action {action.name} is not available in the current state")

        before = self.usage.copy()
        self.steps += 1
        self.action_history.append(action)
        info: dict[str, object] = {}

        if action is Action.RETRIEVE:
            self.documents = self.retriever.retrieve(
                self.task.question,
                top_k=self.retrieval_top_k,
            )
            context_tokens = sum(
                len(item.document.text.split()) for item in self.documents
            )
            self.usage.tokens += context_tokens
            self.usage.retrieval_calls += 1
            self.usage.latency_units += 1.0
            info["documents"] = [item.document.doc_id for item in self.documents]
        elif action is Action.USE_TOOL:
            try:
                self.tool_output = self.calculator.calculate(self.task.question)
            except CalculatorError as error:
                self.tool_output = "TOOL_ERROR"
                info["tool_error"] = str(error)
            self.usage.tokens += 4
            self.usage.tool_calls += 1
            self.usage.latency_units += 0.5
            info["tool_output"] = self.tool_output
        elif action is Action.ASK_HUMAN:
            self.answer = self.task.reference_answer
            self.usage.tokens += 5
            self.usage.human_calls += 1
            self.usage.latency_units += 5.0
            self.done = True
        elif action is Action.ANSWER_DIRECTLY:
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
            raise RuntimeError("The maximum step count was reached without a terminal action")

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

        return StepResult(
            observation=None if self.done else self.observe(),
            reward=reward,
            done=self.done,
            answer=self.answer,
            correct=correct,
            action=action,
            info=info,
        )

    def _resource_penalty(self, usage: ResourceUsage) -> float:
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
