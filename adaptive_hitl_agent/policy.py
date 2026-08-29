from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

import torch
from torch import nn

from .environment import AssistanceEnvironment, FEATURE_NAMES
from .types import Action, Observation, Task


class QNetwork(nn.Module):
    def __init__(self, state_size: int, action_size: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(state_size, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, action_size),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.layers(state)


@dataclass(frozen=True)
class TrainingConfig:
    episodes: int = 600
    batch_size: int = 32
    replay_capacity: int = 4_000
    gamma: float = 0.95
    learning_rate: float = 0.002
    epsilon_start: float = 0.9
    epsilon_end: float = 0.05
    target_update_interval: int = 40
    seed: int = 7


@dataclass(frozen=True)
class Transition:
    state: tuple[float, ...]
    action: int
    reward: float
    next_state: tuple[float, ...]
    next_mask: tuple[bool, ...]
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self._items: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, size: int, rng: random.Random) -> list[Transition]:
        return rng.sample(list(self._items), size)

    def __len__(self) -> int:
        return len(self._items)


class DQNPolicy:
    def __init__(self, network: QNetwork | None = None) -> None:
        self.network = network or QNetwork(len(FEATURE_NAMES), len(Action))

    def select_action(
        self,
        observation: Observation,
        available_actions: Sequence[Action],
        epsilon: float = 0.0,
        rng: random.Random | None = None,
    ) -> Action:
        if not available_actions:
            raise ValueError("At least one available action is required")

        rng = rng or random.Random()
        if rng.random() < epsilon:
            return rng.choice(list(available_actions))

        state = torch.tensor(observation.features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.network(state).squeeze(0)

        # Unavailable actions receive negative infinity, so they cannot be chosen.
        allowed_values = torch.full_like(action_values, float("-inf"))
        for action in available_actions:
            allowed_values[int(action)] = action_values[int(action)]

        best_action = torch.argmax(allowed_values).item()
        return Action(int(best_action))

    def save(self, path: str) -> None:
        torch.save(self.network.state_dict(), path)


@dataclass(frozen=True)
class TrainingResult:
    policy: DQNPolicy
    episode_rewards: tuple[float, ...]
    losses: tuple[float, ...]


def _action_mask(actions: Iterable[Action]) -> tuple[bool, ...]:
    available = set(actions)
    mask = []
    for action in Action:
        mask.append(action in available)
    return tuple(mask)


def _optimize(
    online: QNetwork,
    target: QNetwork,
    optimizer: torch.optim.Optimizer,
    transitions: list[Transition],
    gamma: float,
) -> float:
    states = torch.tensor([item.state for item in transitions], dtype=torch.float32)
    actions = torch.tensor([item.action for item in transitions], dtype=torch.int64)
    rewards = torch.tensor([item.reward for item in transitions], dtype=torch.float32)
    next_states = torch.tensor(
        [item.next_state for item in transitions], dtype=torch.float32
    )
    next_masks = torch.tensor(
        [item.next_mask for item in transitions], dtype=torch.bool
    )
    dones = torch.tensor([item.done for item in transitions], dtype=torch.bool)

    # Q(s, a): the value predicted for each action that was actually taken.
    predictions = online(states)
    predictions = predictions.gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bellman target: immediate reward plus the best future value.
    with torch.no_grad():
        next_values = target(next_states)
        next_values = next_values.masked_fill(~next_masks, -1e9)
        next_values = next_values.max(dim=1).values
        next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
        targets = rewards + gamma * next_values

    loss = nn.functional.smooth_l1_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


def train_dqn(
    tasks: Sequence[Task],
    environment_factory: Callable[[Task], AssistanceEnvironment],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a DQN router on complete multi-step assistance episodes."""

    if not tasks:
        raise ValueError("Training requires at least one task")
    config = config or TrainingConfig()
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.set_num_threads(1)
    rng = random.Random(config.seed)
    task_list = list(tasks)

    online_network = QNetwork(len(FEATURE_NAMES), len(Action))
    target_network = QNetwork(len(FEATURE_NAMES), len(Action))
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    optimizer = torch.optim.Adam(
        online_network.parameters(),
        lr=config.learning_rate,
    )
    replay = ReplayBuffer(config.replay_capacity)
    policy = DQNPolicy(online_network)
    losses: list[float] = []
    episode_rewards: list[float] = []

    decay_denominator = max(1, config.episodes - 1)
    for episode in range(config.episodes):
        task = rng.choice(task_list)
        environment = environment_factory(task)
        observation = environment.observe()
        total_reward = 0.0
        progress = episode / decay_denominator
        epsilon_range = config.epsilon_end - config.epsilon_start
        epsilon = config.epsilon_start + progress * epsilon_range

        while True:
            available = environment.available_actions()
            action = policy.select_action(observation, available, epsilon, rng)
            result = environment.step(action)
            next_observation = result.observation

            if result.done:
                next_state = tuple(0.0 for _ in FEATURE_NAMES)
                next_mask = tuple(False for _ in Action)
            else:
                assert next_observation is not None
                next_state = next_observation.features
                next_mask = _action_mask(environment.available_actions())

            replay.add(
                Transition(
                    state=observation.features,
                    action=int(action),
                    reward=result.reward,
                    next_state=next_state,
                    next_mask=next_mask,
                    done=result.done,
                )
            )
            total_reward += result.reward

            if len(replay) >= config.batch_size:
                batch = replay.sample(config.batch_size, rng)
                loss = _optimize(
                    online_network,
                    target_network,
                    optimizer,
                    batch,
                    config.gamma,
                )
                losses.append(loss)

            if result.done:
                break
            assert next_observation is not None
            observation = next_observation

        episode_rewards.append(total_reward)
        if (episode + 1) % config.target_update_interval == 0:
            target_network.load_state_dict(online_network.state_dict())

    return TrainingResult(
        policy=policy,
        episode_rewards=tuple(episode_rewards),
        losses=tuple(losses),
    )
