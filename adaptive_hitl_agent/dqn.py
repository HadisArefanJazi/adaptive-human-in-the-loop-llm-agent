from __future__ import annotations

import math
import random
from collections import deque
from dataclasses import dataclass
from pathlib import Path
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

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.replay_capacity < self.batch_size:
            raise ValueError("replay_capacity must be at least batch_size")
        if not 0 <= self.gamma <= 1:
            raise ValueError("gamma must be between 0 and 1")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if not 0 <= self.epsilon_end <= self.epsilon_start <= 1:
            raise ValueError("epsilon must satisfy 0 <= end <= start <= 1")
        if self.target_update_interval < 1:
            raise ValueError("target_update_interval must be at least 1")


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
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._items: deque[Transition] = deque(maxlen=capacity)

    def add(self, transition: Transition) -> None:
        self._items.append(transition)

    def sample(self, size: int, rng: random.Random) -> list[Transition]:
        if size < 1 or size > len(self._items):
            raise ValueError("sample size must be between 1 and the replay buffer length")
        return rng.sample(list(self._items), size)

    def __len__(self) -> int:
        return len(self._items)


def _action_mask(actions: Iterable[Action]) -> tuple[bool, ...]:
    available = set(actions)
    return tuple(action in available for action in Action)


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
        if not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be between 0 and 1")

        if epsilon > 0:
            rng = rng or random.Random()
            if rng.random() < epsilon:
                return rng.choice(list(available_actions))

        state = torch.tensor(observation.features, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            action_values = self.network(state).squeeze(0)

        allowed_values = torch.full_like(action_values, float("-inf"))
        for action in available_actions:
            allowed_values[int(action)] = action_values[int(action)]
        return Action(int(torch.argmax(allowed_values).item()))

    def save(self, path: str | Path) -> None:
        torch.save(
            {
                "feature_names": FEATURE_NAMES,
                "action_names": tuple(action.name for action in Action),
                "state_dict": self.network.state_dict(),
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "DQNPolicy":
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint.get("feature_names") != FEATURE_NAMES
            or checkpoint.get("action_names") != tuple(action.name for action in Action)
        ):
            raise ValueError("Checkpoint feature or action schema does not match this policy")
        policy = cls()
        policy.network.load_state_dict(checkpoint["state_dict"])
        policy.network.eval()
        return policy


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
    if torch.any(~dones & ~next_masks.any(dim=1)):
        raise ValueError("A non-terminal transition needs an available next action")

    predictions = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    with torch.no_grad():
        next_values = target(next_states)
        # Mask unavailable actions before the max; terminal states never bootstrap.
        next_values = next_values.masked_fill(~next_masks, float("-inf"))
        next_values = next_values.max(dim=1).values
        next_values = torch.where(dones, torch.zeros_like(next_values), next_values)
        targets = rewards + gamma * next_values

    loss = nn.functional.smooth_l1_loss(predictions, targets)
    optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(online.parameters(), max_norm=5.0)
    optimizer.step()
    return float(loss.item())


@dataclass(frozen=True)
class TrainingResult:
    policy: DQNPolicy
    episode_rewards: tuple[float, ...]
    losses: tuple[float, ...]
    config: TrainingConfig


def train_dqn(
    tasks: Sequence[Task],
    environment_factory: Callable[[Task], AssistanceEnvironment],
    config: TrainingConfig | None = None,
) -> TrainingResult:
    """Train a DQN router over complete multi-step assistance episodes."""

    if not tasks:
        raise ValueError("Training requires at least one task")
    config = config or TrainingConfig()

    previous_threads = torch.get_num_threads()
    try:
        # Tiny CPU batches are faster on one thread; do not change the caller's settings.
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(config.seed)
            return _train_dqn(tasks, environment_factory, config)
    finally:
        torch.set_num_threads(previous_threads)


def _train_dqn(
    tasks: Sequence[Task],
    environment_factory: Callable[[Task], AssistanceEnvironment],
    config: TrainingConfig,
) -> TrainingResult:
    rng = random.Random(config.seed)
    task_list = list(tasks)

    online_network = QNetwork(len(FEATURE_NAMES), len(Action))
    target_network = QNetwork(len(FEATURE_NAMES), len(Action))
    target_network.load_state_dict(online_network.state_dict())
    target_network.eval()
    target_network.requires_grad_(False)
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
        epsilon = config.epsilon_start + progress * (
            config.epsilon_end - config.epsilon_start
        )

        while True:
            action = policy.select_action(
                observation,
                environment.available_actions(),
                epsilon=epsilon,
                rng=rng,
            )
            result = environment.step(action)

            if result.done:
                next_state = tuple(0.0 for _ in FEATURE_NAMES)
                next_mask = tuple(False for _ in Action)
            else:
                assert result.observation is not None
                next_state = result.observation.features
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
                losses.append(
                    _optimize(
                        online_network,
                        target_network,
                        optimizer,
                        replay.sample(config.batch_size, rng),
                        config.gamma,
                    )
                )

            if result.done:
                break
            assert result.observation is not None
            observation = result.observation

        episode_rewards.append(total_reward)
        # The interval is measured in completed episodes, not optimizer steps.
        if (episode + 1) % config.target_update_interval == 0:
            target_network.load_state_dict(online_network.state_dict())

    return TrainingResult(
        policy=policy,
        episode_rewards=tuple(episode_rewards),
        losses=tuple(losses),
        config=config,
    )
