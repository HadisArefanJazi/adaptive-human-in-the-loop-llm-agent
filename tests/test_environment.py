from adaptive_hitl_agent.environment import AssistanceEnvironment
from adaptive_hitl_agent.experiment import load_tasks
from adaptive_hitl_agent.llm import RuleBasedLanguageModel
from adaptive_hitl_agent.retrieval import BM25Retriever
from adaptive_hitl_agent.tools import SafeCalculator
from adaptive_hitl_agent.types import Action


def make_environment(task_id: str) -> AssistanceEnvironment:
    task = next(task for task in load_tasks() if task.task_id == task_id)
    return AssistanceEnvironment(
        task=task,
        retriever=BM25Retriever.from_package_data(),
        language_model=RuleBasedLanguageModel(),
        calculator=SafeCalculator(),
    )


def test_retrieve_then_answer_is_a_successful_two_step_episode() -> None:
    environment = make_environment("retrieve-06")
    retrieval = environment.step(Action.RETRIEVE)
    assert not retrieval.done
    assert retrieval.observation is not None
    answer = environment.step(Action.ANSWER_DIRECTLY)
    assert answer.done
    assert answer.correct
    assert answer.answer == "Albany, New York"
    assert environment.usage.retrieval_calls == 1


def test_tool_then_answer_is_a_successful_two_step_episode() -> None:
    environment = make_environment("tool-06")
    tool_call = environment.step(Action.USE_TOOL)
    assert tool_call.info["tool_output"] == "36"
    answer = environment.step(Action.ANSWER_DIRECTLY)
    assert answer.correct
    assert environment.usage.tool_calls == 1


def test_human_intervention_is_correct_but_costly() -> None:
    environment = make_environment("human-06")
    result = environment.step(Action.ASK_HUMAN)
    assert result.done and result.correct
    assert environment.usage.human_calls == 1
    assert environment.total_resource_penalty > 0.5
