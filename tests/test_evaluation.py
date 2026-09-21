import pytest

from adaptive_hitl_agent.baselines import direct_only_policy, heuristic_routing_policy
from adaptive_hitl_agent.evaluation import evaluate_policy, format_metrics
from adaptive_hitl_agent.experiment import (
    agent_components,
    experiment_config,
    load_tasks,
    make_environment_factory,
)
from adaptive_hitl_agent.human import oracle_human_reviewer
from adaptive_hitl_agent.llm import rule_based_language_model
from adaptive_hitl_agent.retrieval import bm25_retriever
from adaptive_hitl_agent.tools import safe_calculator


def factory():
    components = agent_components(
        retriever=bm25_retriever.from_package_data(),
        language_model=rule_based_language_model(),
        calculator=safe_calculator(),
        human_reviewer=oracle_human_reviewer(),
    )
    return make_environment_factory(components, experiment_config())


def test_metrics_separate_system_success_from_autonomous_success() -> None:
    tasks = [task for task in load_tasks() if task.split == "test"]
    metrics, _ = evaluate_policy("heuristic", heuristic_routing_policy(), tasks, factory())

    assert metrics.system_success == 1.0
    assert metrics.autonomous_success == 0.75
    assert metrics.autonomous_answer_precision == 1.0
    assert metrics.autonomous_answer_coverage == 0.75
    assert metrics.intervention_frequency == 0.25
    assert metrics.success_by_category == {
        "direct": 1.0,
        "human": 1.0,
        "retrieve": 1.0,
        "tool": 1.0,
    }


def test_direct_only_reports_precision_alongside_coverage() -> None:
    tasks = [task for task in load_tasks() if task.split == "test"]
    metrics, _ = evaluate_policy("direct", direct_only_policy(), tasks, factory())

    assert metrics.system_success == 0.25
    assert metrics.autonomous_success == 0.25
    assert metrics.autonomous_answer_precision == 1.0
    assert metrics.autonomous_answer_coverage == 0.25
    assert metrics.intervention_frequency == 0.0


def test_evaluation_requires_tasks() -> None:
    with pytest.raises(ValueError):
        evaluate_policy("direct", direct_only_policy(), [], factory())


def test_format_metrics_contains_explicit_metric_names() -> None:
    tasks = [task for task in load_tasks() if task.split == "test"]
    metrics, _ = evaluate_policy("direct", direct_only_policy(), tasks, factory())
    table = format_metrics({"direct": metrics})
    assert "autonomous" in table
    assert "coverage" in table
    assert "direct" in table


def test_empty_metrics_table_is_valid() -> None:
    assert "coverage" in format_metrics({})


@pytest.mark.parametrize("answer", ["I DON'T KNOW.", "tool_error", "", "None"])
def test_abstentions_do_not_count_as_autonomous_answers(answer) -> None:
    from adaptive_hitl_agent.types import model_answer

    class abstaining_model(rule_based_language_model):
        def answer(self, *args):
            return model_answer(answer, 0.0)

    components = agent_components(
        bm25_retriever.from_package_data(), abstaining_model(),
        safe_calculator(), oracle_human_reviewer(),
    )
    environment_factory = make_environment_factory(components, experiment_config())
    metrics, _ = evaluate_policy(
        "direct", direct_only_policy(), load_tasks()[:1], environment_factory,
    )
    assert metrics.autonomous_answer_coverage == 0
    assert metrics.autonomous_answer_precision == 0


def test_wrong_autonomous_answer_is_in_precision_denominator() -> None:
    from adaptive_hitl_agent.types import model_answer

    class wrong_model(rule_based_language_model):
        def answer(self, *args):
            return model_answer("wrong answer", 0.9)

    components = agent_components(
        bm25_retriever.from_package_data(), wrong_model(),
        safe_calculator(), oracle_human_reviewer(),
    )
    metrics, _ = evaluate_policy(
        "direct", direct_only_policy(), load_tasks()[:1],
        make_environment_factory(components, experiment_config()),
    )
    assert metrics.autonomous_answer_coverage == 1
    assert metrics.autonomous_answer_precision == 0
    assert metrics.system_success == 0


@pytest.mark.parametrize(("answer", "human", "expected"), [
    (None, False, False), ("I don't know.", False, False),
    ("paris", True, False), ("paris", False, True),
])
def test_autonomous_answer_classification(answer, human, expected) -> None:
    from adaptive_hitl_agent.evaluation import is_autonomous_answer

    assert is_autonomous_answer(answer, human) is expected
