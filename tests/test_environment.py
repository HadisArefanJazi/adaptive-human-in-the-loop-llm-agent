from adaptive_hitl_agent.environment import AssistanceEnvironment, FEATURE_NAMES
from adaptive_hitl_agent.experiment import load_tasks
from adaptive_hitl_agent.human import OracleHumanReviewer
from adaptive_hitl_agent.llm import RuleBasedLanguageModel
from adaptive_hitl_agent.retrieval import BM25Retriever
from adaptive_hitl_agent.tools import SafeCalculator
from adaptive_hitl_agent.types import Action


class CountingRetriever(BM25Retriever):
    def __init__(self) -> None:
        base = BM25Retriever.from_package_data()
        super().__init__(base.documents)
        self.calls = 0

    def retrieve(self, query: str, top_k: int = 2):
        self.calls += 1
        return super().retrieve(query, top_k)


def make_environment(task_id: str, retriever: BM25Retriever | None = None) -> AssistanceEnvironment:
    task = next(task for task in load_tasks() if task.task_id == task_id)
    return AssistanceEnvironment(
        task=task,
        retriever=retriever or BM25Retriever.from_package_data(),
        language_model=RuleBasedLanguageModel(),
        calculator=SafeCalculator(),
        human_reviewer=OracleHumanReviewer(),
    )


def test_observe_does_not_perform_free_retrieval() -> None:
    retriever = CountingRetriever()
    environment = make_environment("retrieve-06", retriever)
    observation = environment.observe()

    assert retriever.calls == 0
    assert len(observation.features) == len(FEATURE_NAMES) == 7

    environment.step(Action.RETRIEVE)
    assert retriever.calls == 1


def test_retrieve_then_answer_is_a_successful_two_step_episode() -> None:
    environment = make_environment("retrieve-06")
    retrieval = environment.step(Action.RETRIEVE)
    assert not retrieval.done
    assert retrieval.observation is not None

    answer = environment.step(Action.ANSWER_DIRECTLY)
    assert answer.done and answer.correct
    assert answer.answer == "Albany, New York"
    assert environment.usage.retrieval_calls == 1


def test_tool_then_answer_is_a_successful_two_step_episode() -> None:
    environment = make_environment("tool-06")
    tool_call = environment.step(Action.USE_TOOL)
    assert tool_call.info["tool_output"] == "36"

    answer = environment.step(Action.ANSWER_DIRECTLY)
    assert answer.correct
    assert environment.usage.tool_calls == 1


def test_human_intervention_uses_explicit_reviewer_and_is_costly() -> None:
    environment = make_environment("human-06")
    result = environment.step(Action.ASK_HUMAN)

    assert result.done and result.correct
    assert result.answer == "Use a neutral tone"
    assert environment.usage.human_calls == 1
    assert environment.total_resource_penalty > 0.5


def test_episode_return_matches_outcome_minus_all_costs() -> None:
    import pytest

    environment = make_environment("retrieve-06")
    results = [environment.step(Action.RETRIEVE), environment.step(Action.ANSWER_DIRECTLY)]
    assert sum(result.reward for result in results) == pytest.approx(
        environment.reward_config.success_reward - environment.total_resource_penalty
    )
    assert environment.available_actions() == ()
    with pytest.raises(RuntimeError):
        environment.step(Action.ANSWER_DIRECTLY)


def test_empty_retrieval_cannot_be_repeated() -> None:
    import pytest

    class EmptyRetriever(CountingRetriever):
        def retrieve(self, query, top_k=2):
            return []

    environment = make_environment("retrieve-06", EmptyRetriever())
    environment.step(Action.RETRIEVE)
    assert Action.RETRIEVE not in environment.available_actions()
    with pytest.raises(ValueError):
        environment.step(Action.RETRIEVE)
    environment.step(Action.USE_TOOL)
    assert environment.available_actions() == (Action.ANSWER_DIRECTLY, Action.ASK_HUMAN)
    assert environment.step(Action.ANSWER_DIRECTLY).done


def test_last_step_is_reserved_for_a_terminal_action() -> None:
    environment = make_environment("tool-06")
    environment.max_steps = 1
    assert environment.available_actions() == (Action.ANSWER_DIRECTLY, Action.ASK_HUMAN)
    assert environment.step(0).done


def test_reviewer_response_is_scored_without_substituting_reference() -> None:
    class WrongReviewer:
        def answer(self, task):
            return "A different answer"

    environment = make_environment("human-06")
    environment.human_reviewer = WrongReviewer()
    result = environment.step(Action.ASK_HUMAN)
    assert result.answer == "A different answer"
    assert result.correct is False
    assert result.reward < 0


def test_reward_config_rejects_negative_costs_and_nonfinite_values() -> None:
    import pytest
    from adaptive_hitl_agent.environment import RewardConfig

    for kwargs in ({"human_cost": -1}, {"token_cost": float("nan")}):
        with pytest.raises(ValueError):
            RewardConfig(**kwargs)
