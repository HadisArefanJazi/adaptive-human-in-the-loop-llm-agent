from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from importlib.resources import files
from pathlib import Path as path_type
from typing import Callable as callable_type

import torch

from .records import immutable_record
from .baselines import (
    always_retrieve_policy,
    confidence_gated_rag_policy,
    direct_only_policy,
    heuristic_routing_policy,
    no_human_policy,
    routing_policy,
)
from .dqn import dqn_policy, training_config, training_result, train_dqn
from .environment import assistance_environment, reward_config_type
from .evaluation import policy_metrics, evaluate_policy
from .human import human_reviewer, oracle_human_reviewer
from .llm import language_model, rule_based_language_model
from .retrieval import bm25_retriever
from .tools import safe_calculator
from .types import task_record


class experiment_config(immutable_record):
    training_episodes: int = 600
    seed: int = 7
    max_steps: int = 3
    retrieval_top_k: int = 2
    reward: reward_config_type = reward_config_type()

    __match_args__ = (
        "training_episodes",
        "seed",
        "max_steps",
        "retrieval_top_k",
        "reward",
    )

    def __init__(
        self,
        training_episodes: int = 600,
        seed: int = 7,
        max_steps: int = 3,
        retrieval_top_k: int = 2,
        reward: reward_config_type = reward,
    ) -> None:
        object.__setattr__(self, "training_episodes", training_episodes)
        object.__setattr__(self, "seed", seed)
        object.__setattr__(self, "max_steps", max_steps)
        object.__setattr__(self, "retrieval_top_k", retrieval_top_k)
        object.__setattr__(self, "reward", reward)
        self._validate()

    def as_dict(self) -> dict[str, object]:
        return {
            "training_episodes": self.training_episodes,
            "seed": self.seed,
            "max_steps": self.max_steps,
            "retrieval_top_k": self.retrieval_top_k,
            "reward": self.reward.as_dict(),
        }

    def _validate(self) -> None:
        if self.training_episodes < 1 or self.max_steps < 1 or self.retrieval_top_k < 1:
            raise ValueError("training_episodes, max_steps, and retrieval_top_k must be positive")


class agent_components(immutable_record):
    retriever: bm25_retriever
    language_model: language_model
    calculator: safe_calculator
    human_reviewer: human_reviewer

    __match_args__ = (
        "retriever",
        "language_model",
        "calculator",
        "human_reviewer",
    )

    def __init__(
        self,
        retriever: bm25_retriever,
        language_model: language_model,
        calculator: safe_calculator,
        human_reviewer: human_reviewer,
    ) -> None:
        object.__setattr__(self, "retriever", retriever)
        object.__setattr__(self, "language_model", language_model)
        object.__setattr__(self, "calculator", calculator)
        object.__setattr__(self, "human_reviewer", human_reviewer)


class experiment_result(immutable_record):
    metrics: dict[str, policy_metrics]
    traces: tuple[dict[str, object], ...]
    training: training_result
    metadata: dict[str, object]

    __match_args__ = (
        "metrics",
        "traces",
        "training",
        "metadata",
    )

    def __init__(
        self,
        metrics: dict[str, policy_metrics],
        traces: tuple[dict[str, object], ...],
        training: training_result,
        metadata: dict[str, object],
    ) -> None:
        object.__setattr__(self, "metrics", metrics)
        object.__setattr__(self, "traces", traces)
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "metadata", metadata)


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
        "metrics": {name: metrics.as_dict() for name, metrics in result.metrics.items()},
        "training": {
            "config": result.training.config.as_dict(),
            "episodes": len(result.training.episode_rewards),
            "mean_last_50_reward": (
                sum(result.training.episode_rewards[-50:])
                / min(50, len(result.training.episode_rewards))
            ),
            "final_loss": result.training.losses[-1] if result.training.losses else None,
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
