from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    def __init__(self, observation_size: int = 15, action_size: int = 5) -> None:
        super().__init__()
        self.features = nn.Sequential(nn.Linear(observation_size, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.policy = nn.Linear(128, action_size)
        self.value = nn.Linear(128, 1)

    def forward(self, observations: Tensor) -> tuple[Tensor, Tensor]:
        features = self.features(observations)
        return self.policy(features), self.value(features).squeeze(-1)


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float = 3e-4
    clipping: float = 0.2
    entropy_coefficient: float = 0.01
    value_coefficient: float = 0.5
    discount: float = 0.99
    gae_lambda: float = 0.95
    max_gradient_norm: float = 0.5
    update_epochs: int = 10
    minibatch_size: int = 64


class PPOAgent:
    def __init__(self, settings: PPOSettings | None = None, device: str = "cpu") -> None:
        self.settings = settings or PPOSettings()
        self.device = torch.device(device)
        self.network = ActorCritic().to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.settings.learning_rate)

    def act(self, observation: np.ndarray, deterministic: bool = False) -> tuple[int, float, float]:
        tensor = torch.as_tensor(observation, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            logits, value = self.network(tensor)
            distribution = Categorical(logits=logits)
            action = torch.argmax(logits, dim=-1) if deterministic else distribution.sample()
            log_probability = distribution.log_prob(action)
        return int(action.item()), float(log_probability.item()), float(value.item())

    def evaluate(self, observations: Tensor, actions: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        logits, values = self.network(observations)
        distribution = Categorical(logits=logits)
        return distribution.log_prob(actions), values, distribution.entropy()

    def update(self, batch: dict[str, np.ndarray]) -> dict[str, float]:
        observations = torch.as_tensor(batch["observations"], dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch["actions"], dtype=torch.long, device=self.device)
        old_log_probabilities = torch.as_tensor(batch["log_probabilities"], dtype=torch.float32, device=self.device)
        advantages = torch.as_tensor(batch["advantages"], dtype=torch.float32, device=self.device)
        returns = torch.as_tensor(batch["returns"], dtype=torch.float32, device=self.device)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        losses: list[float] = []
        policy_losses: list[float] = []
        value_losses: list[float] = []
        entropy_values: list[float] = []
        count = observations.shape[0]
        for _ in range(self.settings.update_epochs):
            indices = torch.randperm(count, device=self.device)
            for start in range(0, count, self.settings.minibatch_size):
                selected = indices[start : start + self.settings.minibatch_size]
                log_probabilities, values, entropy = self.evaluate(observations[selected], actions[selected])
                ratios = torch.exp(log_probabilities - old_log_probabilities[selected])
                raw = ratios * advantages[selected]
                clipped = torch.clamp(ratios, 1.0 - self.settings.clipping, 1.0 + self.settings.clipping) * advantages[selected]
                policy_loss = -torch.minimum(raw, clipped).mean()
                value_loss = torch.square(values - returns[selected]).mean()
                entropy_mean = entropy.mean()
                loss = policy_loss + self.settings.value_coefficient * value_loss - self.settings.entropy_coefficient * entropy_mean
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.settings.max_gradient_norm)
                self.optimizer.step()
                losses.append(float(loss.item()))
                policy_losses.append(float(policy_loss.item()))
                value_losses.append(float(value_loss.item()))
                entropy_values.append(float(entropy_mean.item()))
        return {"loss": float(np.mean(losses)), "policy_loss": float(np.mean(policy_losses)), "value_loss": float(np.mean(value_losses)), "entropy": float(np.mean(entropy_values))}

    def save(self, path: str, seed: int, step: int) -> None:
        temporary = f"{path}.tmp"
        torch.save({"model": self.network.state_dict(), "optimizer": self.optimizer.state_dict(), "seed": seed, "step": step}, temporary)
        import os

        os.replace(temporary, path)

    def load(self, path: str) -> tuple[int, int]:
        payload = torch.load(path, map_location=self.device, weights_only=True)
        self.network.load_state_dict(payload["model"])
        self.optimizer.load_state_dict(payload["optimizer"])
        return int(payload["seed"]), int(payload["step"])
