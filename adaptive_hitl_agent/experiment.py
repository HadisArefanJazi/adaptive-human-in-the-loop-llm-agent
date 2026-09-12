from __future__ import annotations

import hashlib
import json
import platform

import torch
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable

from .baselines import (
    AlwaysRetrievePolicy,
    ConfidenceGatedRAGPolicy,
    DirectOnlyPolicy,
    HeuristicRoutingPolicy,
    NoHumanPolicy,
    RoutingPolicy,
)
from .dqn import DQNPolicy, TrainingConfig, TrainingResult, train_dqn
from .environment import AssistanceEnvironment, RewardConfig
from .evaluation import PolicyMetrics, evaluate_policy
from .human import HumanReviewer, OracleHumanReviewer
from .llm import LanguageModel, RuleBasedLanguageModel
from .retrieval import BM25Retriever
from .tools import SafeCalculator
from .types import Task


@dataclass(frozen=True)
class ExperimentConfig:
    training_episodes: int = 600
    seed: int = 7
    max_steps: int = 3
    retrieval_top_k: int = 2
    reward: RewardConfig = RewardConfig()

    def __post_init__(self) -> None:
        if self.training_episodes < 1 or self.max_steps < 1 or self.retrieval_top_k < 1:
            raise ValueError("training_episodes, max_steps, and retrieval_top_k must be positive")


@dataclass(frozen=True)
class AgentComponents:
    retriever: BM25Retriever
    language_model: LanguageModel
    calculator: SafeCalculator
    human_reviewer: HumanReviewer


@dataclass(frozen=True)
class ExperimentResult:
    metrics: dict[str, PolicyMetrics]
    traces: tuple[dict[str, object], ...]
    training: TrainingResult
    metadata: dict[str, object]


def load_tasks() -> list[Task]:
    path = files("adaptive_hitl_agent.data").joinpath("tasks.jsonl")
    tasks = [
        Task.from_dict(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len({task.task_id for task in tasks}) != len(tasks):
        raise ValueError("Task ids must be unique")
    questions = [task.question.casefold().strip() for task in tasks]
    if len(set(questions)) != len(questions):
        raise ValueError("Benchmark questions must be unique across splits")
    return tasks


def make_environment_factory(
    components: AgentComponents,
    config: ExperimentConfig,
) -> Callable[[Task], AssistanceEnvironment]:
    def factory(task: Task) -> AssistanceEnvironment:
        return AssistanceEnvironment(
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
    config: ExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    language_model: LanguageModel | None = None,
    human_reviewer: HumanReviewer | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    tasks = load_tasks()
    train_tasks = [task for task in tasks if task.split == "train"]
    test_tasks = [task for task in tasks if task.split == "test"]
    if not train_tasks or not test_tasks:
        raise ValueError("Benchmark data must contain non-empty train and test splits")

    components = AgentComponents(
        retriever=BM25Retriever.from_package_data(),
        language_model=language_model or RuleBasedLanguageModel(),
        calculator=SafeCalculator(),
        human_reviewer=human_reviewer or OracleHumanReviewer(),
    )
    environment_factory = make_environment_factory(components, config)
    training = train_dqn(
        train_tasks,
        environment_factory,
        TrainingConfig(episodes=config.training_episodes, seed=config.seed),
    )

    policies: dict[str, RoutingPolicy | DQNPolicy] = {
        "adaptive_rl": training.policy,
        "confidence_rag": ConfidenceGatedRAGPolicy(),
        "always_retrieve": AlwaysRetrievePolicy(),
        "heuristic_router": HeuristicRoutingPolicy(),
        "no_human": NoHumanPolicy(training.policy),
        "direct_only": DirectOnlyPolicy(),
    }
    metrics: dict[str, PolicyMetrics] = {}
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

    result = ExperimentResult(
        metrics=metrics,
        traces=tuple(all_traces),
        training=training,
        metadata={
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
        write_artifacts(result, config, Path(output_dir))
    return result


def write_artifacts(
    result: ExperimentResult,
    config: ExperimentConfig,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "metadata": result.metadata,
        "config": asdict(config),
        "metrics": {name: asdict(metrics) for name, metrics in result.metrics.items()},
        "training": {
            "config": asdict(result.training.config),
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
