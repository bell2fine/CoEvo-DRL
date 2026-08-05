from abc import ABC, abstractmethod

import numpy as np

from .types import Action, FloatArray


class Policy(ABC):
    @abstractmethod
    def select(self, state: FloatArray) -> int:
        raise NotImplementedError


class FixedPolicy(Policy):
    def __init__(self, action: Action) -> None:
        self.action = action

    def select(self, state: FloatArray) -> int:
        del state
        return int(self.action)


class PDL1GuidedPolicy(Policy):
    def select(self, state: FloatArray) -> int:
        return int(Action.PD1 if state[6] >= 0.50 else Action.ICI_CHEMOTHERAPY)


class TMBGuidedPolicy(Policy):
    def select(self, state: FloatArray) -> int:
        return int(Action.PD1 if state[5] >= 0.33 else Action.ICI_CHEMOTHERAPY)


class CD8TregPolicy(Policy):
    def select(self, state: FloatArray) -> int:
        ratio = state[3] / max(state[2], 1e-8)
        if ratio < 0.30 and state[9] < 0.50:
            return int(Action.HOLIDAY)
        if ratio > 0.40:
            return int(Action.COMBINATION)
        return int(Action.PD1)


class AT50Policy(Policy):
    def __init__(self) -> None:
        self.reference: float | None = None

    def select(self, state: FloatArray) -> int:
        if self.reference is None:
            self.reference = float(state[9])
        return int(Action.HOLIDAY if state[9] <= 0.5 * self.reference else Action.PD1)


class ThompsonBanditPolicy(Policy):
    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)
        self.alpha = np.ones(5)
        self.beta = np.ones(5)

    def select(self, state: FloatArray) -> int:
        del state
        return int(np.argmax(self.rng.beta(self.alpha, self.beta)))

    def update(self, action: int, reward: float) -> None:
        scaled = 1.0 / (1.0 + np.exp(-reward))
        self.alpha[action] += scaled
        self.beta[action] += 1.0 - scaled


def registry() -> dict[str, Policy]:
    return {
        "keynote_024": FixedPolicy(Action.PD1),
        "keynote_189": FixedPolicy(Action.ICI_CHEMOTHERAPY),
        "checkmate_227": FixedPolicy(Action.COMBINATION),
        "pdl1_guided": PDL1GuidedPolicy(),
        "tmb_guided": TMBGuidedPolicy(),
        "qsp_cd8_treg": CD8TregPolicy(),
        "at50": AT50Policy(),
        "thompson_bandit": ThompsonBanditPolicy(),
    }
