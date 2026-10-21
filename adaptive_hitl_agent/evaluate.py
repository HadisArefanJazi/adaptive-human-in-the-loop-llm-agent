"""Baseline policies, evaluation metrics, and reproducible experiments."""

from __future__ import annotations

import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path as path_type
from typing import Callable as callable_type, Protocol as protocol, Sequence as sequence

import torch

from .core import (
    action_type,
    observation_record,
    assistance_environment,
    reward_config_type,
    bm25_retriever,
    human_reviewer,
    language_model,
    normalize_answer,
    oracle_human_reviewer,
    rule_based_language_model,
    safe_calculator,
    task_record,
)
from .dqn import (
    dqn_policy,
    training_config,
    training_result,
    train_dqn,
)


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


@dataclass(frozen=True)
class confidence_gated_rag_policy:
    confidence_threshold: float = 0.5

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
        if not observation.has_retrieval and action_type.retrieve in available_actions:
            return action_type.retrieve

        return action_type.answer_directly


@dataclass(frozen=True)
class heuristic_routing_policy:
    confidence_threshold: float = 0.5

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


@dataclass
class no_human_policy:
    learned_policy: routing_policy

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


abstention_answers = {
    normalize_answer(answer) for answer in ("", "i don't know", "tool_error", "none")
}


def is_autonomous_answer(answer: str | None, human_intervention: bool) -> bool:
    """whether a final response is a substantive answer without human review."""
    return (
        not human_intervention
        and answer is not None
        and normalize_answer(answer) not in abstention_answers
    )


@dataclass(frozen=True)
class policy_metrics:
    system_success: float
    autonomous_success: float
    autonomous_answer_precision: float
    autonomous_answer_coverage: float
    average_reward: float
    average_tokens: float
    average_retrieval_calls: float
    average_tool_calls: float
    average_latency: float
    intervention_frequency: float
    average_steps: float
    success_by_category: dict[str, float]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def evaluate_policy(
    name: str,
    policy: routing_policy,
    tasks: sequence[task_record],
    environment_factory: callable_type[[task_record], assistance_environment],
) -> tuple[policy_metrics, list[dict[str, object]]]:
    if not tasks:
        raise ValueError("evaluation requires at least one task")

    traces: list[dict[str, object]] = []
    for task in tasks:
        environment = environment_factory(task)
        observation = environment.observe()
        total_reward = 0.0

        while True:
            action = policy.select_action(
                observation,
                environment.available_actions(),
            )
            result = environment.step(action)
            total_reward += result.reward
            if result.done:
                break
            assert result.observation is not None
            observation = result.observation

        traces.append(
            {
                "policy": name,
                "task_id": task.task_id,
                "category": task.category,
                "question": task.question,
                "answer": environment.answer,
                "correct": bool(result.correct),
                "actions": [action.name for action in environment.action_history],
                "human_intervention": environment.usage.human_calls > 0,
                "reward": total_reward,
                **environment.usage.as_dict(),
            }
        )

    count = len(traces)
    correct_count = sum(bool(trace["correct"]) for trace in traces)
    autonomous_correct_count = sum(
        bool(trace["correct"]) and not bool(trace["human_intervention"])
        for trace in traces
    )
    autonomous_answers = [
        trace
        for trace in traces
        if is_autonomous_answer(trace["answer"], bool(trace["human_intervention"]))
    ]
    autonomous_answer_precision = (
        sum(bool(trace["correct"]) for trace in autonomous_answers)
        / len(autonomous_answers)
        if autonomous_answers
        else 0.0
    )

    categories = sorted({str(trace["category"]) for trace in traces})
    success_by_category = {}
    for category in categories:
        category_traces = [
            trace for trace in traces if str(trace["category"]) == category
        ]
        success_by_category[category] = sum(
            bool(trace["correct"]) for trace in category_traces
        ) / len(category_traces)

    metrics = policy_metrics(
        system_success=correct_count / count,
        autonomous_success=autonomous_correct_count / count,
        autonomous_answer_precision=autonomous_answer_precision,
        autonomous_answer_coverage=len(autonomous_answers) / count,
        average_reward=sum(float(trace["reward"]) for trace in traces) / count,
        average_tokens=sum(int(trace["tokens"]) for trace in traces) / count,
        average_retrieval_calls=sum(int(trace["retrieval_calls"]) for trace in traces)
        / count,
        average_tool_calls=sum(int(trace["tool_calls"]) for trace in traces) / count,
        average_latency=sum(float(trace["latency_units"]) for trace in traces) / count,
        intervention_frequency=sum(int(trace["human_calls"]) for trace in traces)
        / count,
        average_steps=sum(len(trace["actions"]) for trace in traces) / count,
        success_by_category=success_by_category,
    )
    return metrics, traces


def format_metrics(metrics: dict[str, policy_metrics]) -> str:
    headers = (
        "policy",
        "system",
        "autonomous",
        "precision",
        "coverage",
        "reward",
        "retrievals",
        "tools",
        "human",
    )
    rows = []
    for name, item in metrics.items():
        rows.append(
            (
                name,
                f"{item.system_success:.1%}",
                f"{item.autonomous_success:.1%}",
                f"{item.autonomous_answer_precision:.1%}",
                f"{item.autonomous_answer_coverage:.1%}",
                f"{item.average_reward:.3f}",
                f"{item.average_retrieval_calls:.2f}",
                f"{item.average_tool_calls:.2f}",
                f"{item.intervention_frequency:.1%}",
            )
        )

    widths = [
        max([len(headers[index]), *(len(row[index]) for row in rows)])
        for index in range(len(headers))
    ]
    lines = [
        "  ".join(
            heading.ljust(widths[index]) for index, heading in enumerate(headers)
        ),
        "  ".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(row))
        )
    return "\n".join(lines)


@dataclass(frozen=True)
class experiment_config:
    training_episodes: int = 3000
    seed: int = 7
    max_steps: int = 3
    retrieval_top_k: int = 2
    reward: reward_config_type = reward_config_type()

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    def __post_init__(self) -> None:
        if self.training_episodes < 1 or self.max_steps < 1 or self.retrieval_top_k < 1:
            raise ValueError(
                "training_episodes, max_steps, and retrieval_top_k must be positive"
            )


@dataclass(frozen=True)
class agent_components:
    retriever: bm25_retriever
    language_model: language_model
    calculator: safe_calculator
    human_reviewer: human_reviewer


@dataclass(frozen=True)
class experiment_result:
    metrics: dict[str, policy_metrics]
    traces: tuple[dict[str, object], ...]
    training: training_result
    metadata: dict[str, object]


def load_tasks() -> list[task_record]:
    path = files("adaptive_hitl_agent.data").joinpath("tasks.jsonl")
    tasks = [
        task_record.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("task_record ids must be unique")
    questions = [task.question.casefold().strip() for task in tasks]
    if len(set(questions)) != len(questions):
        raise ValueError("benchmark questions must be unique across splits")
    return tasks


def make_environment_factory(
    components: agent_components,
    config: experiment_config,
) -> callable_type[[task_record], assistance_environment]:
    def factory(task: task_record) -> assistance_environment:
        return assistance_environment(
            task=task,
            retriever=components.retriever,
            language_model=components.language_model,
            calculator=components.calculator,
            human_reviewer=components.human_reviewer,
            reward_config=config.reward,
            max_steps=config.max_steps,
            retrieval_top_k=config.retrieval_top_k,
        )

    return factory


def run_experiment(
    config: experiment_config | None = None,
    output_dir: str | path_type | None = None,
    language_model: language_model | None = None,
    human_reviewer: human_reviewer | None = None,
) -> experiment_result:
    config = config or experiment_config()
    tasks = load_tasks()
    train_tasks = [task for task in tasks if task.split == "train"]
    test_tasks = [task for task in tasks if task.split == "test"]
    if not train_tasks or not test_tasks:
        raise ValueError("benchmark data must contain non-empty train and test splits")

    components = agent_components(
        retriever=bm25_retriever.from_package_data(),
        language_model=language_model or rule_based_language_model(),
        calculator=safe_calculator(),
        human_reviewer=human_reviewer or oracle_human_reviewer(),
    )
    environment_factory = make_environment_factory(components, config)
    training = train_dqn(
        train_tasks,
        environment_factory,
        training_config(episodes=config.training_episodes, seed=config.seed),
    )

    policies: dict[str, routing_policy | dqn_policy] = {
        "adaptive_rl": training.policy,
        "confidence_rag": confidence_gated_rag_policy(),
        "always_retrieve": always_retrieve_policy(),
        "heuristic_router": heuristic_routing_policy(),
        "no_human": no_human_policy(training.policy),
        "direct_only": direct_only_policy(),
    }
    metrics: dict[str, policy_metrics] = {}
    all_traces: list[dict[str, object]] = []
    for name, policy in policies.items():
        policy_metrics, traces = evaluate_policy(
            name,
            policy,
            test_tasks,
            environment_factory,
        )
        metrics[name] = policy_metrics
        all_traces.extend(traces)

    result = experiment_result(
        metrics=metrics,
        traces=tuple(all_traces),
        training=training,
        metadata={
            "package_version": version("adaptive-hitl-agent"),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "training_device": "cpu",
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "language_model": type(components.language_model).__name__,
            "model_name": getattr(components.language_model, "model_name", None),
            "human_reviewer": type(components.human_reviewer).__name__,
            "train_tasks": len(train_tasks),
            "test_tasks": len(test_tasks),
            "data_sha256": {
                name: hashlib.sha256(
                    files("adaptive_hitl_agent.data").joinpath(name).read_bytes()
                ).hexdigest()
                for name in ("tasks.jsonl", "knowledge_base.json")
            },
        },
    )
    if output_dir is not None:
        write_artifacts(result, config, path_type(output_dir))
    return result


def write_artifacts(
    result: experiment_result,
    config: experiment_config,
    output_dir: path_type,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "metadata": result.metadata,
        "config": config.as_dict(),
        "metrics": {
            name: metrics.as_dict() for name, metrics in result.metrics.items()
        },
        "training": {
            "config": result.training.config.as_dict(),
            "episodes": len(result.training.episode_rewards),
            "mean_last_50_reward": (
                sum(result.training.episode_rewards[-50:])
                / min(50, len(result.training.episode_rewards))
            ),
            "final_loss": result.training.losses[-1]
            if result.training.losses
            else None,
        },
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_dir / "traces.jsonl").open("w", encoding="utf-8") as handle:
        for trace in result.traces:
            handle.write(json.dumps(trace) + "\n")
    result.training.policy.save(output_dir / "dqn_policy.pt")
