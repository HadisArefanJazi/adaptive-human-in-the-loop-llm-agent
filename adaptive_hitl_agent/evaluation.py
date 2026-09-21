from __future__ import annotations

from typing import Callable as callable_type, Sequence as sequence

from .records import immutable_record
from .baselines import routing_policy
from .environment import assistance_environment
from .types import task_record, normalize_answer


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


class policy_metrics(immutable_record):
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

    __match_args__ = (
        "system_success",
        "autonomous_success",
        "autonomous_answer_precision",
        "autonomous_answer_coverage",
        "average_reward",
        "average_tokens",
        "average_retrieval_calls",
        "average_tool_calls",
        "average_latency",
        "intervention_frequency",
        "average_steps",
        "success_by_category",
    )

    def __init__(
        self,
        system_success: float,
        autonomous_success: float,
        autonomous_answer_precision: float,
        autonomous_answer_coverage: float,
        average_reward: float,
        average_tokens: float,
        average_retrieval_calls: float,
        average_tool_calls: float,
        average_latency: float,
        intervention_frequency: float,
        average_steps: float,
        success_by_category: dict[str, float],
    ) -> None:
        object.__setattr__(self, "system_success", system_success)
        object.__setattr__(self, "autonomous_success", autonomous_success)
        object.__setattr__(self, "autonomous_answer_precision", autonomous_answer_precision)
        object.__setattr__(self, "autonomous_answer_coverage", autonomous_answer_coverage)
        object.__setattr__(self, "average_reward", average_reward)
        object.__setattr__(self, "average_tokens", average_tokens)
        object.__setattr__(self, "average_retrieval_calls", average_retrieval_calls)
        object.__setattr__(self, "average_tool_calls", average_tool_calls)
        object.__setattr__(self, "average_latency", average_latency)
        object.__setattr__(self, "intervention_frequency", intervention_frequency)
        object.__setattr__(self, "average_steps", average_steps)
        object.__setattr__(self, "success_by_category", success_by_category)

    def as_dict(self) -> dict[str, object]:
        return {
            "system_success": self.system_success,
            "autonomous_success": self.autonomous_success,
            "autonomous_answer_precision": self.autonomous_answer_precision,
            "autonomous_answer_coverage": self.autonomous_answer_coverage,
            "average_reward": self.average_reward,
            "average_tokens": self.average_tokens,
            "average_retrieval_calls": self.average_retrieval_calls,
            "average_tool_calls": self.average_tool_calls,
            "average_latency": self.average_latency,
            "intervention_frequency": self.intervention_frequency,
            "average_steps": self.average_steps,
            "success_by_category": self.success_by_category.copy(),
        }


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
        average_retrieval_calls=sum(
            int(trace["retrieval_calls"]) for trace in traces
        )
        / count,
        average_tool_calls=sum(int(trace["tool_calls"]) for trace in traces) / count,
        average_latency=sum(float(trace["latency_units"]) for trace in traces) / count,
        intervention_frequency=sum(
            int(trace["human_calls"]) for trace in traces
        )
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
