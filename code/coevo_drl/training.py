import logging
import random
from dataclasses import dataclass

import numpy as np
import torch

from .agent import PPOAgent
from .environment import CoevolutionEnvironment

LOGGER = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class RolloutBuffer:
    observations: list[np.ndarray]
    actions: list[int]
    rewards: list[float]
    dones: list[bool]
    log_probabilities: list[float]
    values: list[float]

    @classmethod
    def empty(cls) -> "RolloutBuffer":
        return cls([], [], [], [], [], [])

    def append(self, observation: np.ndarray, action: int, reward: float, done: bool, log_probability: float, value: float) -> None:
        self.observations.append(observation.copy())
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probabilities.append(log_probability)
        self.values.append(value)

    def prepare(self, last_value: float, discount: float, gae_lambda: float) -> dict[str, np.ndarray]:
        advantages = np.zeros(len(self.rewards), dtype=np.float32)
        accumulator = 0.0
        next_value = last_value
        for index in range(len(self.rewards) - 1, -1, -1):
            continuation = 1.0 - float(self.dones[index])
            delta = self.rewards[index] + discount * next_value * continuation - self.values[index]
            accumulator = delta + discount * gae_lambda * continuation * accumulator
            advantages[index] = accumulator
            next_value = self.values[index]
        returns = advantages + np.asarray(self.values, dtype=np.float32)
        return {"observations": np.asarray(self.observations), "actions": np.asarray(self.actions), "log_probabilities": np.asarray(self.log_probabilities), "advantages": advantages, "returns": returns}


def train(total_steps: int = 1_000_000, rollout_steps: int = 2048, seed: int = 42, device: str = "cpu") -> PPOAgent:
    set_seed(seed)
    environment = CoevolutionEnvironment(seed=seed)
    agent = PPOAgent(device=device)
    observation = environment.reset()
    completed = 0
    while completed < total_steps:
        buffer = RolloutBuffer.empty()
        target = min(rollout_steps, total_steps - completed)
        for _ in range(target):
            action, log_probability, value = agent.act(observation)
            next_observation, reward, done, _ = environment.step(action)
            buffer.append(observation, action, reward, done, log_probability, value)
            observation = environment.reset() if done else next_observation
            completed += 1
        _, _, last_value = agent.act(observation)
        batch = buffer.prepare(last_value, agent.settings.discount, agent.settings.gae_lambda)
        summary = agent.update(batch)
        LOGGER.info("step=%d loss=%.6f entropy=%.6f", completed, summary["loss"], summary["entropy"])
    return agent
