from __future__ import annotations

from typing import Protocol as protocol, Sequence as sequence

from .records import immutable_record, mutable_record
from .types import action_type, observation_record


class routing_policy(protocol):
    def select_action(
        self,
        observation: observation_record,
        available_actions: sequence[action_type],
    ) -> action_type: ...


class direct_only_policy:
    def select_action(
        self,
        observation: observation_record,
        available_actions: sequence[action_type],
    ) -> action_type:
        return action_type.answer_directly


class confidence_gated_rag_policy(immutable_record):
    confidence_threshold: float = 0.5

    __match_args__ = (
        "confidence_threshold",
    )

    def __init__(
        self,
        confidence_threshold: float = 0.5,
    ) -> None:
        object.__setattr__(self, "confidence_threshold", confidence_threshold)

    def select_action(
        self,
        observation: observation_record,
        available_actions: sequence[action_type],
    ) -> action_type:
        if (
            observation.direct_confidence < self.confidence_threshold
            and not observation.has_retrieval
            and action_type.retrieve in available_actions
        ):
            return action_type.retrieve

        return action_type.answer_directly


class always_retrieve_policy:
    def select_action(
        self,
        observation: observation_record,
        available_actions: sequence[action_type],
    ) -> action_type:
        if (
            not observation.has_retrieval
            and action_type.retrieve in available_actions
        ):
            return action_type.retrieve

        return action_type.answer_directly


class heuristic_routing_policy(immutable_record):
    confidence_threshold: float = 0.5

    __match_args__ = (
        "confidence_threshold",
    )

    def __init__(
        self,
        confidence_threshold: float = 0.5,
    ) -> None:
        object.__setattr__(self, "confidence_threshold", confidence_threshold)

    def select_action(
        self,
        observation: observation_record,
        available_actions: sequence[action_type],
    ) -> action_type:
        if (
            observation.ambiguity_signal >= 0.5
            and action_type.ask_human in available_actions
        ):
            return action_type.ask_human

        if (
            observation.math_signal >= 0.5
            and not observation.has_tool_output
            and action_type.use_tool in available_actions
        ):
            return action_type.use_tool

        if (
            observation.direct_confidence < self.confidence_threshold
            and not observation.has_retrieval
            and action_type.retrieve in available_actions
        ):
            return action_type.retrieve

        return action_type.answer_directly


class no_human_policy(mutable_record):
    learned_policy: routing_policy

    __match_args__ = (
        "learned_policy",
    )

    def __init__(
        self,
        learned_policy: routing_policy,
    ) -> None:
        self.learned_policy = learned_policy

    def select_action(
        self,
        observation: observation_record,
        available_actions: sequence[action_type],
    ) -> action_type:
        allowed = [
            action
            for action in available_actions
            if action is not action_type.ask_human
        ]
        return self.learned_policy.select_action(
            observation,
            allowed,
        )
