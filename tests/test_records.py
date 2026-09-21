import pytest

from adaptive_hitl_agent.baselines import confidence_gated_rag_policy, no_human_policy
from adaptive_hitl_agent.dqn import training_config, transition_record
from adaptive_hitl_agent.environment import reward_config_type
from adaptive_hitl_agent.evaluation import policy_metrics
from adaptive_hitl_agent.experiment import experiment_config
from adaptive_hitl_agent.retrieval import document_record, retrieved_document
from adaptive_hitl_agent.types import (
    action_type, model_answer, resource_usage, step_result, task_record,
)


@pytest.mark.parametrize("make_record", [
    lambda: task_record("id", "question", ("answer",), "test", "direct"),
    lambda: model_answer("answer", 0.8),
    lambda: training_config(),
    lambda: experiment_config(),
    lambda: reward_config_type(),
    lambda: confidence_gated_rag_policy(),
    lambda: document_record("id", "title", "text", "answer"),
    lambda: retrieved_document(document_record("id", "title", "text", "answer"), 0.5),
    lambda: transition_record((1.0,), 0, 0.5, (0.0,), (False,), True),
])
def test_immutable_records_have_value_equality_hashes_and_named_representations(make_record):
    record = make_record()
    equal_record = make_record()
    assert record == equal_record
    assert hash(record) == hash(equal_record)
    assert len({record, equal_record}) == 1
    assert record != object()
    assert record.__eq__(object()) is NotImplemented
    assert repr(record).startswith(type(record).__name__ + "(")
    assert all(name + "=" in repr(record) for name in record.__match_args__)
    name = record.__match_args__[0]
    with pytest.raises(AttributeError):
        setattr(record, name, None)
    with pytest.raises(AttributeError):
        delattr(record, name)
    with pytest.raises(AttributeError):
        record.extra = True


def test_value_comparison_uses_record_type_and_values():
    assert model_answer("answer", 0.8) != model_answer("answer", 0.9)
    assert model_answer("answer", 0.8) != ("answer", 0.8)
    assert repr(model_answer("answer", 0.8)) == "model_answer(text='answer', confidence=0.8)"


def test_mutable_records_allow_assignment_and_are_unhashable():
    usage = resource_usage()
    usage.tokens = 3
    assert usage == resource_usage(tokens=3)
    assert usage.copy() == usage
    assert usage.copy() is not usage
    with pytest.raises(TypeError):
        hash(usage)
    policy = no_human_policy(None)
    policy.learned_policy = "policy"
    assert policy == no_human_policy("policy")
    with pytest.raises(TypeError):
        hash(policy)


def test_step_info_defaults_are_independent_and_explicit_values_are_preserved():
    args = (None, 0.5, True, "answer", True, action_type.answer_directly)
    first, second = step_result(*args), step_result(*args)
    first.info["note"] = "details"
    assert second.info == {}
    info = {"key": "value"}
    assert step_result(*args, info=info).info is info
    assert step_result(*args, info=None).info is None
    with pytest.raises(TypeError):
        hash(first)
    with pytest.raises(AttributeError):
        first.info = {}


def test_configuration_exports_are_detached_and_include_nested_reward_values():
    config = experiment_config(reward=reward_config_type(tool_cost=0.1))
    payload = config.as_dict()
    assert payload == {
        "training_episodes": 600, "seed": 7, "max_steps": 3, "retrieval_top_k": 2,
        "reward": {
            "success_reward": 1.0, "failure_penalty": -0.25, "token_cost": 0.002,
            "retrieval_cost": 0.08, "tool_cost": 0.1, "human_cost": 0.45,
            "latency_cost": 0.02,
        },
    }
    payload["reward"]["tool_cost"] = 9
    assert config.reward.tool_cost == 0.1
    defaults = experiment_config()
    assert defaults.reward is experiment_config.reward
    assert training_config(seed=3).as_dict()["seed"] == 3


def test_metric_exports_copy_category_counts():
    metrics = policy_metrics(*(0.0,) * 11, success_by_category={"direct": 1.0})
    exported = metrics.as_dict()
    exported["success_by_category"]["direct"] = 0.0
    assert metrics.success_by_category == {"direct": 1.0}
    assert set(exported) == set(metrics.__match_args__)


@pytest.mark.parametrize("name", [
    "success_reward", "failure_penalty", "token_cost", "retrieval_cost",
    "tool_cost", "human_cost", "latency_cost",
])
def test_reward_fields_reject_nonfinite_values(name):
    with pytest.raises(ValueError, match="finite"):
        reward_config_type(**{name: float("nan")})
