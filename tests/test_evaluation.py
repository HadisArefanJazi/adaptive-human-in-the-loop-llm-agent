import pytest

from adaptive_hitl_agent.baselines import DirectOnlyPolicy, HeuristicRoutingPolicy
from adaptive_hitl_agent.evaluation import evaluate_policy, format_metrics
from adaptive_hitl_agent.experiment import (
    AgentComponents,
    ExperimentConfig,
    load_tasks,
    make_environment_factory,
)
from adaptive_hitl_agent.human import OracleHumanReviewer
from adaptive_hitl_agent.llm import RuleBasedLanguageModel
from adaptive_hitl_agent.retrieval import BM25Retriever
from adaptive_hitl_agent.tools import SafeCalculator


def factory():
    components = AgentComponents(
        retriever=BM25Retriever.from_package_data(),
        language_model=RuleBasedLanguageModel(),
        calculator=SafeCalculator(),
        human_reviewer=OracleHumanReviewer(),
    )
    return make_environment_factory(components, ExperimentConfig())


def test_metrics_separate_system_success_from_autonomous_success() -> None:
    tasks = [task for task in load_tasks() if task.split == "test"]
    metrics, _ = evaluate_policy("heuristic", HeuristicRoutingPolicy(), tasks, factory())

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
    metrics, _ = evaluate_policy("direct", DirectOnlyPolicy(), tasks, factory())

    assert metrics.system_success == 0.25
    assert metrics.autonomous_success == 0.25
    assert metrics.autonomous_answer_precision == 1.0
    assert metrics.autonomous_answer_coverage == 0.25
    assert metrics.intervention_frequency == 0.0


def test_evaluation_requires_tasks() -> None:
    with pytest.raises(ValueError):
        evaluate_policy("direct", DirectOnlyPolicy(), [], factory())


def test_format_metrics_contains_explicit_metric_names() -> None:
    tasks = [task for task in load_tasks() if task.split == "test"]
    metrics, _ = evaluate_policy("direct", DirectOnlyPolicy(), tasks, factory())
    table = format_metrics({"direct": metrics})
    assert "autonomous" in table
    assert "coverage" in table
    assert "direct" in table


def test_empty_metrics_table_is_valid() -> None:
    assert "coverage" in format_metrics({})


@pytest.mark.parametrize("answer", ["I DON'T KNOW.", "tool_error", "", "None"])
def test_abstentions_do_not_count_as_autonomous_answers(answer) -> None:
    from adaptive_hitl_agent.types import ModelAnswer

    class AbstainingModel(RuleBasedLanguageModel):
        def answer(self, *args):
            return ModelAnswer(answer, 0.0)

    components = AgentComponents(
        BM25Retriever.from_package_data(), AbstainingModel(),
        SafeCalculator(), OracleHumanReviewer(),
    )
    environment_factory = make_environment_factory(components, ExperimentConfig())
    metrics, _ = evaluate_policy(
        "direct", DirectOnlyPolicy(), load_tasks()[:1], environment_factory,
    )
    assert metrics.autonomous_answer_coverage == 0
    assert metrics.autonomous_answer_precision == 0


def test_wrong_autonomous_answer_is_in_precision_denominator() -> None:
    from adaptive_hitl_agent.types import ModelAnswer

    class WrongModel(RuleBasedLanguageModel):
        def answer(self, *args):
            return ModelAnswer("Wrong answer", 0.9)

    components = AgentComponents(
        BM25Retriever.from_package_data(), WrongModel(),
        SafeCalculator(), OracleHumanReviewer(),
    )
    metrics, _ = evaluate_policy(
        "direct", DirectOnlyPolicy(), load_tasks()[:1],
        make_environment_factory(components, ExperimentConfig()),
    )
    assert metrics.autonomous_answer_coverage == 1
    assert metrics.autonomous_answer_precision == 0
    assert metrics.system_success == 0
