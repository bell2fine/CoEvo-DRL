import numpy as np

from .model import CoevolutionModel
from .types import Action, EpisodeSettings, FloatArray, RewardWeights


class CoevolutionEnvironment:
    observation_size = 15
    action_size = 5

    def __init__(self, model: CoevolutionModel | None = None, settings: EpisodeSettings | None = None, rewards: RewardWeights | None = None, seed: int = 42) -> None:
        self.model = model or CoevolutionModel()
        self.settings = settings or EpisodeSettings()
        self.rewards = rewards or RewardWeights()
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(10, dtype=np.float64)
        self.previous_state = np.zeros(10, dtype=np.float64)
        self.time = 0.0
        self.response_days = 0.0
        self.neoantigen_history: list[float] = []
        self.ratio_history: list[float] = []

    def reset(self, initial_state: FloatArray | None = None) -> FloatArray:
        if initial_state is None:
            initial_state = self.sample_initial_state()
        self.state = np.asarray(initial_state, dtype=np.float64).copy()
        self.state[9] = self.state[0] + self.state[1]
        self.previous_state = self.state.copy()
        self.time = 0.0
        self.response_days = 0.0
        self.neoantigen_history = [float(self.state[5])]
        self.ratio_history = [float(self.state[3] / max(self.state[2], 1e-8))]
        return self.observation()

    def sample_initial_state(self) -> FloatArray:
        tumor = float(self.rng.beta(2.5, 5.0) * 0.65 + 0.08)
        resistant_fraction = float(self.rng.beta(1.5, 9.0) * 0.35)
        ts = tumor * (1.0 - resistant_fraction)
        tr = tumor * resistant_fraction
        e = float(self.rng.beta(2.3, 3.8) * 0.7 + 0.05)
        r = float(self.rng.beta(1.8, 5.0) * 0.35 + 0.01)
        d = float(self.rng.beta(2.0, 4.5) * 0.4 + 0.02)
        n = float(self.rng.beta(2.2, 2.8) * 0.75 + 0.05)
        pdl1 = float(self.rng.beta(2.0, 2.5) * 0.8 + 0.05)
        return np.asarray([ts, tr, e, r, d, n, pdl1, 0.0, 0.0, tumor], dtype=np.float64)

    def observation(self) -> FloatArray:
        interval = max(self.settings.decision_interval_days, 1e-8)
        n_velocity = (self.state[5] - self.previous_state[5]) / interval
        p_velocity = (self.state[6] - self.previous_state[6]) / interval
        diversity = float(np.std(self.neoantigen_history[-5:])) if len(self.neoantigen_history) > 1 else 0.0
        escape_velocity = p_velocity - n_velocity
        ratio = self.state[3] / max(self.state[2], 1e-8)
        previous_ratio = self.previous_state[3] / max(self.previous_state[2], 1e-8)
        ratio_velocity = (ratio - previous_ratio) / interval
        burden_velocity = (self.state[9] - self.previous_state[9]) / interval
        normalized_growth = burden_velocity / max(self.previous_state[9], 1e-8)
        derived = np.asarray([diversity, escape_velocity, ratio, ratio_velocity, normalized_growth], dtype=np.float64)
        return np.concatenate((np.clip(self.state, 0.0, 1.0), np.clip(derived, -1.0, 1.0)))

    def step(self, action_index: int) -> tuple[FloatArray, float, bool, dict[str, float]]:
        action = Action(action_index)
        old_burden = float(self.state[9])
        old_ratio = float(self.state[3] / max(self.state[2], 1e-8))
        self.previous_state = self.state.copy()
        self.state = self.model.integrate(self.state, action, self.settings.decision_interval_days, self.settings.integration_step_days)
        self.time += self.settings.decision_interval_days
        self.neoantigen_history.append(float(self.state[5]))
        self.ratio_history.append(float(self.state[3] / max(self.state[2], 1e-8)))
        burden_change = float(self.state[9]) - old_burden
        active = float(action != Action.HOLIDAY)
        holiday_bonus = float(action == Action.HOLIDAY) * max(0.0, 1.0 - old_ratio / self.rewards.treg_ratio_threshold)
        reward = -self.rewards.burden * burden_change - self.rewards.treatment * active + self.rewards.holiday * holiday_bonus
        self.response_days = self.response_days + self.settings.decision_interval_days if self.state[9] < self.settings.response_burden else 0.0
        done = self.state[9] > self.settings.death_burden or self.response_days >= self.settings.response_duration_days or self.time >= self.settings.horizon_days
        info = {"burden": float(self.state[9]), "time": self.time, "phase": float(self.model.phase(self.state)), "treg_effector_ratio": float(self.ratio_history[-1])}
        return self.observation(), reward, done, info
