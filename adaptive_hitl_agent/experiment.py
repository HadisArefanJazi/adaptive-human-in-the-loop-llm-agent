from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Callable, Sequence

from .baselines import (
    AlwaysRetrievePolicy,
    DirectOnlyPolicy,
    FixedRAGPolicy,
    HeuristicRoutingPolicy,
    NoHumanPolicy,
    RoutingPolicy,
)
from .environment import AssistanceEnvironment, RewardConfig
from .llm import LanguageModel, RuleBasedLanguageModel
from .policy import DQNPolicy, TrainingConfig, TrainingResult, train_dqn
from .retrieval import BM25Retriever
from .tools import SafeCalculator
from .types import Action, Task


@dataclass(frozen=True)
class ExperimentConfig:
    training_episodes: int = 600
    seed: int = 7
    max_steps: int = 3
    retrieval_top_k: int = 2
    reward: RewardConfig = RewardConfig()


@dataclass(frozen=True)
class AgentComponents:
    retriever: BM25Retriever
    language_model: LanguageModel
    calculator: SafeCalculator


@dataclass(frozen=True)
class PolicyMetrics:
    task_success: float
    factuality: float
    average_reward: float
    average_tokens: float
    average_retrieval_calls: float
    average_tool_calls: float
    average_latency: float
    intervention_frequency: float
    average_steps: float


@dataclass(frozen=True)
class ExperimentResult:
    metrics: dict[str, PolicyMetrics]
    traces: tuple[dict[str, object], ...]
    training: TrainingResult


def load_tasks() -> list[Task]:
    path = files("adaptive_hitl_agent.data").joinpath("tasks.jsonl")
    tasks = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            tasks.append(Task.from_dict(json.loads(line)))
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
            reward_config=config.reward,
            max_steps=config.max_steps,
            retrieval_top_k=config.retrieval_top_k,
        )

    return factory


def evaluate_policy(
    name: str,
    policy: RoutingPolicy | DQNPolicy,
    tasks: Sequence[Task],
    environment_factory: Callable[[Task], AssistanceEnvironment],
) -> tuple[PolicyMetrics, list[dict[str, object]]]:
    traces: list[dict[str, object]] = []
    for task in tasks:
        environment = environment_factory(task)
        observation = environment.observe()
        total_reward = 0.0
        while True:
            action = policy.select_action(observation, environment.available_actions())
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
                "reward": round(total_reward, 6),
                **environment.usage.as_dict(),
            }
        )

    count = len(traces)
    correct_count = 0
    substantive_count = 0
    substantive_correct_count = 0
    total_reward = 0.0
    total_tokens = 0
    total_retrieval_calls = 0
    total_tool_calls = 0
    total_latency = 0.0
    total_human_calls = 0
    total_steps = 0

    for trace in traces:
        is_correct = bool(trace["correct"])
        if is_correct:
            correct_count += 1

        answer = str(trace["answer"])
        if answer not in {"I don't know", "TOOL_ERROR", "None"}:
            substantive_count += 1
            if is_correct:
                substantive_correct_count += 1

        total_reward += float(trace["reward"])
        total_tokens += int(trace["tokens"])
        total_retrieval_calls += int(trace["retrieval_calls"])
        total_tool_calls += int(trace["tool_calls"])
        total_latency += float(trace["latency_units"])
        total_human_calls += int(trace["human_calls"])
        total_steps += len(trace["actions"])

    factuality = 0.0
    if substantive_count > 0:
        factuality = substantive_correct_count / substantive_count

    metrics = PolicyMetrics(
        task_success=correct_count / count,
        factuality=factuality,
        average_reward=total_reward / count,
        average_tokens=total_tokens / count,
        average_retrieval_calls=total_retrieval_calls / count,
        average_tool_calls=total_tool_calls / count,
        average_latency=total_latency / count,
        intervention_frequency=total_human_calls / count,
        average_steps=total_steps / count,
    )
    return metrics, traces


def run_experiment(
    config: ExperimentConfig | None = None,
    output_dir: str | Path | None = None,
    language_model: LanguageModel | None = None,
) -> ExperimentResult:
    config = config or ExperimentConfig()
    random.seed(config.seed)
    tasks = load_tasks()
    train_tasks = []
    test_tasks = []
    for task in tasks:
        if task.split == "train":
            train_tasks.append(task)
        elif task.split == "test":
            test_tasks.append(task)

    components = AgentComponents(
        retriever=BM25Retriever.from_package_data(),
        language_model=language_model or RuleBasedLanguageModel(),
        calculator=SafeCalculator(),
    )
    environment_factory = make_environment_factory(components, config)
    training = train_dqn(
        train_tasks,
        environment_factory,
        TrainingConfig(episodes=config.training_episodes, seed=config.seed),
    )

    policies: dict[str, RoutingPolicy | DQNPolicy] = {
        "adaptive_rl": training.policy,
        "fixed_rag": FixedRAGPolicy(),
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
        "config": asdict(config),
        "metrics": {name: asdict(metrics) for name, metrics in result.metrics.items()},
        "training": {
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
    result.training.policy.save(str(output_dir / "dqn_policy.pt"))


def format_metrics(metrics: dict[str, PolicyMetrics]) -> str:
    headers = (
        "policy",
        "success",
        "factuality",
        "reward",
        "tokens",
        "retrievals",
        "tools",
        "human",
        "latency",
    )
    rows = []
    for name, item in metrics.items():
        row = (
            name,
            f"{item.task_success:.1%}",
            f"{item.factuality:.1%}",
            f"{item.average_reward:.3f}",
            f"{item.average_tokens:.1f}",
            f"{item.average_retrieval_calls:.2f}",
            f"{item.average_tool_calls:.2f}",
            f"{item.intervention_frequency:.1%}",
            f"{item.average_latency:.2f}",
        )
        rows.append(row)

    widths = []
    for index, heading in enumerate(headers):
        width = len(heading)
        for row in rows:
            width = max(width, len(row[index]))
        widths.append(width)

    lines = []
    header_cells = []
    separator_cells = []
    for index, heading in enumerate(headers):
        header_cells.append(heading.ljust(widths[index]))
        separator_cells.append("-" * widths[index])
    lines.append("  ".join(header_cells))
    lines.append("  ".join(separator_cells))

    for row in rows:
        cells = []
        for index, value in enumerate(row):
            cells.append(value.ljust(widths[index]))
        lines.append("  ".join(cells))

    return "\n".join(lines)
