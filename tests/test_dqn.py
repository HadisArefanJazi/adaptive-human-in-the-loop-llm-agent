import random

import pytest
import torch
from torch import nn

from adaptive_hitl_agent.dqn import DQNPolicy, ReplayBuffer, TrainingConfig, Transition
from adaptive_hitl_agent.environment import FEATURE_NAMES
from adaptive_hitl_agent.types import Action, Observation


class FixedNetwork(nn.Module):
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        values = torch.tensor([1.0, 100.0, 50.0, 2.0])
        return values.repeat(state.shape[0], 1)


def observation() -> Observation:
    features = tuple(0.0 for _ in FEATURE_NAMES)
    return Observation(
        features=features,
        feature_names=FEATURE_NAMES,
        direct_confidence=0.0,
        math_signal=0.0,
        ambiguity_signal=0.0,
        has_retrieval=False,
        has_tool_output=False,
        step_index=0,
    )


def transition(index: int) -> Transition:
    state = tuple(float(index) for _ in FEATURE_NAMES)
    return Transition(
        state=state,
        action=int(Action.ANSWER_DIRECTLY),
        reward=float(index),
        next_state=state,
        next_mask=tuple(True for _ in Action),
        done=False,
    )


def test_policy_masks_unavailable_actions() -> None:
    policy = DQNPolicy(FixedNetwork())
    selected = policy.select_action(
        observation(),
        [Action.ANSWER_DIRECTLY, Action.ASK_HUMAN],
    )
    assert selected is Action.ASK_HUMAN


def test_replay_buffer_respects_capacity_and_sampling() -> None:
    replay = ReplayBuffer(capacity=2)
    replay.add(transition(1))
    replay.add(transition(2))
    replay.add(transition(3))

    assert len(replay) == 2
    sampled = replay.sample(2, random.Random(7))
    assert {item.reward for item in sampled} == {2.0, 3.0}


def test_training_config_rejects_invalid_ranges() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(episodes=0)
    with pytest.raises(ValueError):
        TrainingConfig(epsilon_start=0.1, epsilon_end=0.2)


def test_exploration_samples_only_available_actions() -> None:
    policy = DQNPolicy(FixedNetwork())
    allowed = [Action.ANSWER_DIRECTLY, Action.ASK_HUMAN]
    rng = random.Random(7)
    actions = {policy.select_action(observation(), allowed, 1.0, rng) for _ in range(50)}
    assert actions == set(allowed)


@pytest.mark.parametrize("epsilon", [-0.1, 1.1, float("nan")])
def test_policy_rejects_invalid_epsilon(epsilon: float) -> None:
    with pytest.raises(ValueError):
        DQNPolicy(FixedNetwork()).select_action(observation(), [Action.ANSWER_DIRECTLY], epsilon)


def test_policy_rejects_empty_action_set() -> None:
    with pytest.raises(ValueError):
        DQNPolicy(FixedNetwork()).select_action(observation(), [])


@pytest.mark.parametrize("kwargs", [
    {"batch_size": 0}, {"replay_capacity": 1}, {"gamma": -0.1},
    {"learning_rate": 0}, {"learning_rate": float("nan")},
    {"target_update_interval": 0},
])
def test_training_rejects_invalid_hyperparameters(kwargs) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**kwargs)


def test_replay_rejects_invalid_capacity_and_sample_sizes() -> None:
    with pytest.raises(ValueError):
        ReplayBuffer(0)
    replay = ReplayBuffer(2)
    for size in (0, 1, -1):
        with pytest.raises(ValueError):
            replay.sample(size, random.Random(0))


def test_bellman_update_masks_actions_and_stops_at_terminal_states() -> None:
    from adaptive_hitl_agent.dqn import _optimize

    online = nn.Linear(len(FEATURE_NAMES), len(Action))
    target = nn.Linear(len(FEATURE_NAMES), len(Action))
    with torch.no_grad():
        online.weight.zero_()
        online.bias.zero_()
        target.weight.zero_()
        target.bias.copy_(torch.tensor([2.0, 500.0, 4.0, 6.0]))
    state = tuple(0.0 for _ in FEATURE_NAMES)
    batch = [
        Transition(state, 0, 2.0, state, (False,) * 4, True),
        Transition(state, 0, 1.0, state, (True, False, True, False), False),
    ]
    before = {name: value.clone() for name, value in target.state_dict().items()}
    loss = _optimize(online, target, torch.optim.SGD(online.parameters(), lr=0.1), batch, 0.5)
    # Targets are 2 and 1 + 0.5 * 4 = 3. Their Huber losses average to 2.
    assert loss == pytest.approx(2.0)
    assert online.bias[0].item() > 0
    assert all(parameter.grad is None for parameter in target.parameters())
    assert all(torch.equal(before[name], value) for name, value in target.state_dict().items())


def test_non_terminal_transition_requires_a_next_action() -> None:
    from adaptive_hitl_agent.dqn import QNetwork, _optimize

    online = QNetwork(len(FEATURE_NAMES), len(Action))
    target = QNetwork(len(FEATURE_NAMES), len(Action))
    state = tuple(0.0 for _ in FEATURE_NAMES)
    batch = [Transition(state, 0, 0.0, state, (False,) * 4, False)]
    with pytest.raises(ValueError):
        _optimize(online, target, torch.optim.Adam(online.parameters()), batch, 0.95)


def test_checkpoint_round_trip_and_schema_validation(tmp_path) -> None:
    policy = DQNPolicy()
    path = tmp_path / "policy.pt"
    policy.save(path)
    restored = DQNPolicy.load(path)
    state = torch.zeros(1, len(FEATURE_NAMES))
    assert torch.equal(policy.network(state), restored.network(state))
    checkpoint = torch.load(path, weights_only=True)
    checkpoint["feature_names"] = ("old_state",)
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match="schema"):
        DQNPolicy.load(path)


def test_training_synchronizes_target_and_restores_rng(monkeypatch) -> None:
    from adaptive_hitl_agent.dqn import QNetwork, train_dqn
    from adaptive_hitl_agent.experiment import (
        AgentComponents, ExperimentConfig, load_tasks, make_environment_factory,
    )
    from adaptive_hitl_agent.human import OracleHumanReviewer
    from adaptive_hitl_agent.llm import RuleBasedLanguageModel
    from adaptive_hitl_agent.retrieval import BM25Retriever
    from adaptive_hitl_agent.tools import SafeCalculator

    copies = []
    original = QNetwork.load_state_dict

    def record_copy(self, state_dict, *args, **kwargs):
        copies.append(self)
        return original(self, state_dict, *args, **kwargs)

    monkeypatch.setattr(QNetwork, "load_state_dict", record_copy)
    components = AgentComponents(
        BM25Retriever.from_package_data(), RuleBasedLanguageModel(),
        SafeCalculator(), OracleHumanReviewer(),
    )
    factory = make_environment_factory(components, ExperimentConfig())
    tasks = [task for task in load_tasks() if task.split == "train"]
    rng_state = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    config = TrainingConfig(episodes=4, batch_size=1, target_update_interval=2)
    result = train_dqn(tasks, factory, config)
    assert len(copies) == 3  # Initial copy, then after episodes 2 and 4.
    assert len({id(network) for network in copies}) == 1
    assert torch.equal(torch.get_rng_state(), rng_state)
    assert torch.get_num_threads() == threads
    assert result.losses and all(torch.isfinite(torch.tensor(result.losses)))
    with pytest.raises(ValueError):
        train_dqn([], factory)
