from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, Sequence

from .dqn import DQNPolicy
from .types import Action, Observation


class RoutingPolicy(Protocol):
    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
    ) -> Action:
        ...


class DirectOnlyPolicy:
    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
    ) -> Action:
        return Action.ANSWER_DIRECTLY


@dataclass(frozen=True)
class ConfidenceGatedRAGPolicy:
    """Retrieve once when direct-model confidence falls below a fixed threshold."""

    confidence_threshold: float = 0.5

    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
    ) -> Action:
        if (
            observation.direct_confidence < self.confidence_threshold
            and not observation.has_retrieval
            and Action.RETRIEVE in available_actions
        ):
            return Action.RETRIEVE
        return Action.ANSWER_DIRECTLY


class AlwaysRetrievePolicy:
    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
    ) -> Action:
        if not observation.has_retrieval and Action.RETRIEVE in available_actions:
            return Action.RETRIEVE
        return Action.ANSWER_DIRECTLY


@dataclass(frozen=True)
class HeuristicRoutingPolicy:
    confidence_threshold: float = 0.5

    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
    ) -> Action:
        if (
            observation.ambiguity_signal >= 0.5
            and Action.ASK_HUMAN in available_actions
        ):
            return Action.ASK_HUMAN
        if (
            observation.math_signal >= 0.5
            and not observation.has_tool_output
            and Action.USE_TOOL in available_actions
        ):
            return Action.USE_TOOL
        if (
            observation.direct_confidence < self.confidence_threshold
            and not observation.has_retrieval
            and Action.RETRIEVE in available_actions
        ):
            return Action.RETRIEVE
        return Action.ANSWER_DIRECTLY


@dataclass
class NoHumanPolicy:
    """Ablation: execute the learned policy with human escalation masked out."""

    learned_policy: DQNPolicy

    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
    ) -> Action:
        allowed = [
            action for action in available_actions if action is not Action.ASK_HUMAN
        ]
        return self.learned_policy.select_action(observation, allowed)
