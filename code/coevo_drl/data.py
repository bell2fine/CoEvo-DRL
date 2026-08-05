import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .types import FloatArray


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    tumor_mutational_burden: float
    neoantigen_count: float
    cd8_fraction: float
    treg_fraction: float
    dendritic_fraction: float
    pdl1_expression: float
    survival_days: float
    event: bool


def read_patient_records(path: str | Path) -> list[PatientRecord]:
    records: list[PatientRecord] = []
    with Path(path).open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            records.append(
                PatientRecord(
                    patient_id=row["patient_id"],
                    tumor_mutational_burden=float(row["tumor_mutational_burden"]),
                    neoantigen_count=float(row["neoantigen_count"]),
                    cd8_fraction=float(row["cd8_fraction"]),
                    treg_fraction=float(row["treg_fraction"]),
                    dendritic_fraction=float(row["dendritic_fraction"]),
                    pdl1_expression=float(row["pdl1_expression"]),
                    survival_days=float(row["survival_days"]),
                    event=row["event"].lower() in {"1", "true", "yes"},
                )
            )
    return records


def record_to_initial_state(record: PatientRecord, targeted_panel: bool = False) -> FloatArray:
    tmb_scale = 30.0 if not targeted_panel else 20.0
    neoantigen = np.clip(record.neoantigen_count / 1200.0 if record.neoantigen_count > 0 else record.tumor_mutational_burden / tmb_scale, 0.02, 1.0)
    burden = np.clip(0.20 + 0.35 * record.pdl1_expression + 0.10 * record.treg_fraction, 0.05, 0.90)
    resistance = np.clip(0.04 + 0.18 * record.pdl1_expression, 0.01, 0.35)
    return np.asarray(
        [
            burden * (1.0 - resistance),
            burden * resistance,
            np.clip(record.cd8_fraction * 3.0, 0.02, 1.0),
            np.clip(record.treg_fraction * 4.0, 0.01, 1.0),
            np.clip(record.dendritic_fraction * 5.0, 0.01, 1.0),
            neoantigen,
            np.clip(record.pdl1_expression, 0.01, 1.0),
            0.0,
            0.0,
            burden,
        ],
        dtype=np.float64,
    )


def generate_virtual_cohort(records: list[PatientRecord], count: int, coefficient_of_variation: float = 0.20, seed: int = 42) -> list[FloatArray]:
    if not records:
        raise ValueError("at least one patient record is required")
    rng = np.random.default_rng(seed)
    base = np.stack([record_to_initial_state(record) for record in records])
    cohort: list[FloatArray] = []
    while len(cohort) < count:
        selected = base[int(rng.integers(0, len(base)))].copy()
        noise = rng.lognormal(-0.5 * np.log1p(coefficient_of_variation**2), np.sqrt(np.log1p(coefficient_of_variation**2)), selected.shape)
        candidate = np.clip(selected * noise, 0.0, 1.0)
        candidate[9] = candidate[0] + candidate[1]
        if 0.03 <= candidate[9] <= 0.95 and 0.01 <= candidate[2] <= 1.0:
            cohort.append(candidate)
    return cohort
