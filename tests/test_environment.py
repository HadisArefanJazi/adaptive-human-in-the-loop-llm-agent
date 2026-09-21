from adaptive_hitl_agent.environment import assistance_environment, feature_names
from adaptive_hitl_agent.experiment import load_tasks
from adaptive_hitl_agent.human import oracle_human_reviewer
from adaptive_hitl_agent.llm import rule_based_language_model
from adaptive_hitl_agent.retrieval import bm25_retriever
from adaptive_hitl_agent.tools import safe_calculator
from adaptive_hitl_agent.types import action_type


class counting_retriever(bm25_retriever):
    def __init__(self) -> None:
        base = bm25_retriever.from_package_data()
        super().__init__(base.documents)
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 2):
        self.calls += 1
        return super().retrieve(query, top_k)


def make_environment(task_id: str, retriever: bm25_retriever | None = None) -> assistance_environment:
    task = next(task for task in load_tasks() if task.task_id == task_id)
    return assistance_environment(
        task=task,
        retriever=retriever or bm25_retriever.from_package_data(),
        language_model=rule_based_language_model(),
        calculator=safe_calculator(),
        human_reviewer=oracle_human_reviewer(),
    )


def test_observe_does_not_perform_free_retrieval() -> None:
    retriever = counting_retriever()
    environment = make_environment("retrieve-06", retriever)
    observation = environment.observe()

    assert retriever.calls == 0
    assert len(observation.features) == len(feature_names) == 7

    environment.step(action_type.retrieve)
    assert retriever.calls == 1


def test_retrieve_then_answer_is_a_successful_two_step_episode() -> None:
    environment = make_environment("retrieve-06")
    retrieval = environment.step(action_type.retrieve)
    assert not retrieval.done
    assert retrieval.observation is not None

    answer = environment.step(action_type.answer_directly)
    assert answer.done and answer.correct
    assert answer.answer == "Albany, New York"
    assert environment.usage.retrieval_calls == 1


def test_tool_then_answer_is_a_successful_two_step_episode() -> None:
    environment = make_environment("tool-06")
    tool_call = environment.step(action_type.use_tool)
    assert tool_call.info["tool_output"] == "36"

    answer = environment.step(action_type.answer_directly)
    assert answer.correct
    assert environment.usage.tool_calls == 1


def test_human_intervention_uses_explicit_reviewer_and_is_costly() -> None:
    environment = make_environment("human-06")
    result = environment.step(action_type.ask_human)

    assert result.done and result.correct
    assert result.answer == "Use a neutral tone"
    assert environment.usage.human_calls == 1
    assert environment.total_resource_penalty > 0.5


def test_episode_return_matches_outcome_minus_all_costs() -> None:
    import pytest

    environment = make_environment("retrieve-06")
    results = [environment.step(action_type.retrieve), environment.step(action_type.answer_directly)]
    assert sum(result.reward for result in results) == pytest.approx(
        environment.reward_config.success_reward - environment.total_resource_penalty
    )
    assert environment.available_actions() == ()
    with pytest.raises(RuntimeError):
        environment.step(action_type.answer_directly)


def test_empty_retrieval_cannot_be_repeated() -> None:
    import pytest

    class empty_retriever(counting_retriever):
        def retrieve(self, query, top_k=2):
            return []

    environment = make_environment("retrieve-06", empty_retriever())
    environment.step(action_type.retrieve)
    assert action_type.retrieve not in environment.available_actions()
    with pytest.raises(ValueError):
        environment.step(action_type.retrieve)
    environment.step(action_type.use_tool)
    assert environment.available_actions() == (action_type.answer_directly, action_type.ask_human)
    assert environment.step(action_type.answer_directly).done


def test_last_step_is_reserved_for_a_terminal_action() -> None:
    environment = make_environment("tool-06")
    environment.max_steps = 1
    assert environment.available_actions() == (action_type.answer_directly, action_type.ask_human)
    assert environment.step(0).done


def test_reviewer_response_is_scored_without_substituting_reference() -> None:
    class wrong_reviewer:
        def answer(self, task):
            return "a different answer"

    environment = make_environment("human-06")
    environment.human_reviewer = wrong_reviewer()
    result = environment.step(action_type.ask_human)
    assert result.answer == "a different answer"
    assert result.correct is False
    assert result.reward < 0


def test_reward_config_rejects_negative_costs_and_nonfinite_values() -> None:
    import pytest
    from adaptive_hitl_agent.environment import reward_config_type

    for kwargs in ({"human_cost": -1}, {"token_cost": float("nan")}):
        with pytest.raises(ValueError):
            reward_config_type(**kwargs)


def test_math_signal_preserves_subtraction_and_unicode_rules():
    from adaptive_hitl_agent.types import task_record

    environment = make_environment("tool-06")
    for question, expected in [
        ("5 - 2", 1.0), ("٥\t-\n٢", 1.0), ("5\u00a0-\u20032", 1.0),
        ("-5", 0.0), ("release-2", 0.0), ("5 - x 2", 0.0),
        ("5 -- 2", 0.0), ("² + ³", 0.0), ("١ + ٢", 1.0),
        ("a+b", 0.0), ("no math", 0.0),
    ]:
        environment.task = task_record(
            "signal", question, ("unused",), "test", "tool"
        )
        assert environment.observe().math_signal == expected, question
