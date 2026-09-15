import random

import pytest
import torch
from torch import nn

from adaptive_hitl_agent.dqn import dqn_policy, replay_buffer, training_config, transition_record
from adaptive_hitl_agent.environment import feature_names
from adaptive_hitl_agent.types import action_type, observation_record


class fixed_network(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        values = torch.tensor([1.0, 100.0, 50.0, 2.0])
        return values.repeat(state.shape[0], 1)


def observation() -> observation_record:
    features = tuple(0.0 for _ in feature_names)
    return observation_record(
        features=features,
        feature_names=feature_names,
        direct_confidence=0.0,
        math_signal=0.0,
        ambiguity_signal=0.0,
        has_retrieval=False,
        has_tool_output=False,
        step_index=0,
    )


def transition(index: int) -> transition_record:
    state = tuple(float(index) for _ in feature_names)
    return transition_record(
        state=state,
        action=int(action_type.answer_directly),
        reward=float(index),
        next_state=state,
        next_mask=tuple(True for _ in action_type),
        done=False,
    )


def test_policy_masks_unavailable_actions() -> None:
    policy = dqn_policy(fixed_network())
    selected = policy.select_action(
        observation(),
        [action_type.answer_directly, action_type.ask_human],
    )
    assert selected is action_type.ask_human


def test_replay_buffer_respects_capacity_and_sampling() -> None:
    replay = replay_buffer(capacity=2)
    replay.add(transition(1))
    replay.add(transition(2))
    replay.add(transition(3))

    assert len(replay) == 2
    sampled = replay.sample(2, random.Random(7))
    assert {item.reward for item in sampled} == {2.0, 3.0}


def test_training_config_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        training_config(episodes=0)
    with pytest.raises(ValueError):
        training_config(epsilon_start=0.1, epsilon_end=0.2)


def test_exploration_samples_only_available_actions() -> None:
    policy = dqn_policy(fixed_network())
    allowed = [action_type.answer_directly, action_type.ask_human]
    rng = random.Random(7)
    actions = {policy.select_action(observation(), allowed, 1.0, rng) for _ in range(50)}
    assert actions == set(allowed)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, float("nan")])
def test_policy_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError):
        dqn_policy(fixed_network()).select_action(observation(), [action_type.answer_directly], epsilon)


def test_policy_rejects_empty_action_set() -> None:
    with pytest.raises(ValueError):
        dqn_policy(fixed_network()).select_action(observation(), [])


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"replay_capacity": 1}, {"gamma": -0.1},
    {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"target_update_interval": 0},
])
def test_training_rejects_invalid_hyperparameters(kwargs) -> None:
    with pytest.raises(ValueError):
        training_config(**kwargs)


def test_replay_rejects_invalid_capacity_and_sample_sizes() -> None:
    with pytest.raises(ValueError):
        replay_buffer(0)
    replay = replay_buffer(2)
    for size in (0, 1, -1):
        with pytest.raises(ValueError):
            replay.sample(size, random.Random(0))


def test_bellman_update_masks_actions_and_stops_at_terminal_states() -> None:
    from adaptive_hitl_agent.dqn import _optimize

    online = nn.Linear(len(feature_names), len(action_type))
    target = nn.Linear(len(feature_names), len(action_type))
    with torch.no_grad():
        online.weight.zero_()
        online.bias.zero_()
        target.weight.zero_()
        target.bias.copy_(torch.tensor([2.0, 500.0, 4.0, 6.0]))
    state = tuple(0.0 for _ in feature_names)
    batch = [
        transition_record(state, 0, 2.0, state, (False,) * 4, True),
        transition_record(state, 0, 1.0, state, (True, False, True, False), False),
    ]
    before = {name: value.clone() for name, value in target.state_dict().items()}
    loss = _optimize(online, target, torch.optim.SGD(online.parameters(), lr=0.1), batch, 0.5)
    # targets are 2 and 1 + 0.5 * 4 = 3. their huber losses average to 2.
    assert loss == pytest.approx(2.0)
    assert online.bias[0].item() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(torch.equal(before[name], value) for name, value in target.state_dict().items())


def test_non_terminal_transition_requires_a_next_action() -> None:
    from adaptive_hitl_agent.dqn import q_network, _optimize

    online = q_network(len(feature_names), len(action_type))
    target = q_network(len(feature_names), len(action_type))
    state = tuple(0.0 for _ in feature_names)
    batch = [transition_record(state, 0, 0.0, state, (False,) * 4, False)]
    with pytest.raises(ValueError):
        _optimize(online, target, torch.optim.Adam(online.parameters()), batch, 0.95)


def test_checkpoint_round_trip_and_schema_validation(tmp_path) -> None:
    policy = dqn_policy()
    path = tmp_path / "policy.pt"
    policy.save(path)
    restored = dqn_policy.load(path)
    state = torch.zeros(1, len(feature_names))
    assert torch.equal(policy.network(state), restored.network(state))
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["feature_names"] = ("invalid_state",)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="schema"):
        dqn_policy.load(path)


def test_training_synchronizes_target_and_restores_rng(monkeypatch) -> None:
    from adaptive_hitl_agent.dqn import q_network, train_dqn
    from adaptive_hitl_agent.experiment import (
        agent_components, experiment_config, load_tasks, make_environment_factory,
    )
    from adaptive_hitl_agent.human import oracle_human_reviewer
    from adaptive_hitl_agent.llm import rule_based_language_model
    from adaptive_hitl_agent.retrieval import bm25_retriever
    from adaptive_hitl_agent.tools import safe_calculator

    copies = []
    original = q_network.load_state_dict

    def record_copy(self, state_dict, *args, **kwargs):
        copies.append(self)
        return original(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(q_network, "load_state_dict", record_copy)
    components = agent_components(
        bm25_retriever.from_package_data(), rule_based_language_model(),
        safe_calculator(), oracle_human_reviewer(),
    )
    factory = make_environment_factory(components, experiment_config())
    tasks = [task for task in load_tasks() if task.split == "train"]
    rng_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    config = training_config(episodes=4, batch_size=1, target_update_interval=2)
    result = train_dqn(tasks, factory, config)
    assert len(copies) == 3  # initial copy, then after episodes 2 and 4.
    assert len({id(network) for network in copies}) == 1
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.get_num_threads() == threads
    assert result.losses and all(torch.isfinite(torch.tensor(result.losses)))
    with pytest.raises(ValueError):
        train_dqn([], factory)


def test_checkpoint_action_names_are_case_insensitive(tmp_path):
    policy = dqn_policy()
    path = tmp_path / "uppercase.pt"
    policy.save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["action_names"] = tuple(name.upper() for name in checkpoint["action_names"])
    torch.save(checkpoint, path)
    restored = dqn_policy.load(path)
    state = torch.zeros(1, len(feature_names))
    assert torch.equal(policy.network(state), restored.network(state))


@pytest.mark.parametrize("names", [None, (1, 2, 3, 4), ("unknown",) * 4])
def test_checkpoint_rejects_invalid_action_names(tmp_path, names):
    path = tmp_path / "invalid.pt"
    dqn_policy().save(path)
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["action_names"] = names
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="schema"):
        dqn_policy.load(path)
