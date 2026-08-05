from dataclasses import dataclass
from enum import IntEnum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class Action(IntEnum):
    PD1 = 0
    CTLA4 = 1
    COMBINATION = 2
    ICI_CHEMOTHERAPY = 3
    HOLIDAY = 4


class Phase(IntEnum):
    ELIMINATION = 0
    EQUILIBRIUM = 1
    ESCAPE = 2


@dataclass(frozen=True)
class ODEParameters:
    sensitive_growth: float = 0.18
    resistant_growth: float = 0.14
    carrying_capacity: float = 1.0
    sensitive_killing: float = 0.42
    resistant_killing: float = 0.16
    treg_suppression: float = 0.35
    effector_influx: float = 0.025
    effector_stimulation: float = 0.30
    antigen_half_saturation: float = 0.20
    effector_decay: float = 0.08
    treg_effector_suppression: float = 0.12
    ici_effector_restoration: float = 0.22
    pdl1_half_saturation: float = 0.25
    treg_influx: float = 0.012
    tumor_treg_recruitment: float = 0.08
    treg_decay: float = 0.05
    ctla4_treg_depletion: float = 0.18
    dendritic_influx: float = 0.02
    dendritic_stimulation: float = 0.24
    dendritic_half_saturation: float = 0.20
    dendritic_decay: float = 0.06
    neoantigen_generation: float = 0.09
    neoantigen_editing: float = 0.16
    pdl1_induction: float = 0.25
    effector_half_saturation: float = 0.20
    resistant_pdl1: float = 0.15
    pdl1_decay: float = 0.08
    ici_pdl1_blockade: float = 0.22
    resistance_transition: float = 0.003
    chemotherapy_killing: float = 0.28
    ici_elimination: float = 0.10
    chemotherapy_elimination: float = 0.22
    numerical_epsilon: float = 1e-8
    growth_threshold: float = 0.003
    effector_threshold: float = 0.15


@dataclass(frozen=True)
class RewardWeights:
    burden: float = 1.0
    treatment: float = 0.03
    holiday: float = 0.08
    treg_ratio_threshold: float = 0.30


@dataclass(frozen=True)
class EpisodeSettings:
    decision_interval_days: float = 21.0
    horizon_days: float = 730.0
    integration_step_days: float = 0.25
    death_burden: float = 0.98
    response_burden: float = 0.02
    response_duration_days: float = 90.0


@dataclass(frozen=True)
class Transition:
    state: FloatArray
    action: int
    reward: float
    next_state: FloatArray
    done: bool
    log_probability: float
    value: float
