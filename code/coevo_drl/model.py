from dataclasses import replace

import numpy as np
from scipy.integrate import solve_ivp

from .types import Action, FloatArray, ODEParameters, Phase


class CoevolutionModel:
    state_size = 10

    def __init__(self, parameters: ODEParameters | None = None) -> None:
        self.parameters = parameters or ODEParameters()

    def derivatives(self, time: float, state: FloatArray, inputs: tuple[float, float, float]) -> FloatArray:
        del time
        p = self.parameters
        ts, tr, e, r, d, n, pdl1, c_ici, c_chemo, _ = state
        burden = max(ts + tr, p.numerical_epsilon)
        ici_input, chemo_input, ctla4_factor = inputs
        competition = max(0.0, 1.0 - burden / p.carrying_capacity)
        immune_denominator = 1.0 + p.treg_suppression * r
        sensitive_kill = p.sensitive_killing * n * (1.0 - pdl1) * e * ts / immune_denominator
        resistant_kill = p.resistant_killing * n * (1.0 - pdl1) * e * tr / immune_denominator
        dts = p.sensitive_growth * ts * competition - sensitive_kill - p.chemotherapy_killing * c_chemo * ts
        dtr = p.resistant_growth * tr * competition - resistant_kill + p.resistance_transition * ts
        de = (
            p.effector_influx
            + p.effector_stimulation * burden * n * d / (p.antigen_half_saturation + burden)
            - p.effector_decay * e
            - p.treg_effector_suppression * r * e
            + p.ici_effector_restoration * c_ici * pdl1 * e / (p.pdl1_half_saturation + pdl1)
        )
        dr = p.treg_influx + p.tumor_treg_recruitment * burden - p.treg_decay * r - p.ctla4_treg_depletion * ctla4_factor * c_ici * r
        dd = p.dendritic_influx + p.dendritic_stimulation * burden * n / (p.dendritic_half_saturation + burden) - p.dendritic_decay * d
        dn = p.neoantigen_generation * ts / (burden + p.numerical_epsilon) - p.neoantigen_editing * e * n / immune_denominator
        dpdl1 = p.pdl1_induction * e / (p.effector_half_saturation + e) + p.resistant_pdl1 * tr - p.pdl1_decay * pdl1 - p.ici_pdl1_blockade * c_ici * pdl1
        dici = ici_input - p.ici_elimination * c_ici
        dchemo = chemo_input - p.chemotherapy_elimination * c_chemo
        dburden = dts + dtr
        return np.asarray([dts, dtr, de, dr, dd, dn, dpdl1, dici, dchemo, dburden], dtype=np.float64)

    def action_inputs(self, action: Action) -> tuple[float, float, float]:
        values = {
            Action.PD1: (0.55, 0.0, 0.0),
            Action.CTLA4: (0.38, 0.0, 1.0),
            Action.COMBINATION: (0.72, 0.0, 0.8),
            Action.ICI_CHEMOTHERAPY: (0.55, 0.65, 0.0),
            Action.HOLIDAY: (0.0, 0.0, 0.0),
        }
        return values[action]

    def integrate(self, state: FloatArray, action: Action, days: float, max_step: float = 0.25) -> FloatArray:
        inputs = self.action_inputs(action)
        result = solve_ivp(lambda t, y: self.derivatives(t, y, inputs), (0.0, days), state, method="RK45", rtol=1e-7, atol=1e-9, max_step=max_step)
        if not result.success:
            raise RuntimeError(result.message)
        next_state = np.clip(result.y[:, -1], 0.0, 1.5)
        next_state[9] = next_state[0] + next_state[1]
        return next_state

    def phase(self, state: FloatArray) -> Phase:
        growth = self.derivatives(0.0, state, (0.0, 0.0, 0.0))[9]
        if growth < -self.parameters.growth_threshold and state[2] > self.parameters.effector_threshold:
            return Phase.ELIMINATION
        if abs(growth) < self.parameters.growth_threshold:
            return Phase.EQUILIBRIUM
        return Phase.ESCAPE

    def perturbed(self, name: str, multiplier: float) -> "CoevolutionModel":
        if name not in ODEParameters.__dataclass_fields__:
            raise KeyError(name)
        value = getattr(self.parameters, name)
        return CoevolutionModel(replace(self.parameters, **{name: value * multiplier}))
