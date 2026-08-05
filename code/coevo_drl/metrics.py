from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import norm


@dataclass(frozen=True)
class Interval:
    estimate: float
    lower: float
    upper: float


def concordance_index(times: NDArray[np.float64], scores: NDArray[np.float64], events: NDArray[np.bool_]) -> float:
    concordant = 0.0
    comparable = 0
    for first in range(len(times)):
        for second in range(first + 1, len(times)):
            if times[first] == times[second]:
                continue
            earlier, later = (first, second) if times[first] < times[second] else (second, first)
            if not events[earlier]:
                continue
            comparable += 1
            if scores[earlier] > scores[later]:
                concordant += 1.0
            elif scores[earlier] == scores[later]:
                concordant += 0.5
    return concordant / comparable if comparable else float("nan")


def bootstrap_concordance(times: NDArray[np.float64], scores: NDArray[np.float64], events: NDArray[np.bool_], resamples: int = 1000, seed: int = 42) -> Interval:
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(resamples):
        indices = rng.integers(0, len(times), len(times))
        value = concordance_index(times[indices], scores[indices], events[indices])
        if np.isfinite(value):
            values.append(value)
    estimate = concordance_index(times, scores, events)
    lower, upper = np.quantile(values, [0.025, 0.975])
    return Interval(float(estimate), float(lower), float(upper))


def holm_bonferroni(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    count = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, float((count - rank) * p_values[index]))
        adjusted[index] = min(1.0, running)
    return adjusted


def benjamini_hochberg(p_values: NDArray[np.float64]) -> NDArray[np.float64]:
    order = np.argsort(p_values)[::-1]
    adjusted = np.empty_like(p_values)
    running = 1.0
    count = len(p_values)
    for reverse_rank, index in enumerate(order):
        rank = count - reverse_rank
        running = min(running, float(p_values[index] * count / rank))
        adjusted[index] = running
    return adjusted


def one_sided_normal_test(mean_a: float, std_a: float, mean_b: float, std_b: float, count: int) -> float:
    standard_error = np.sqrt((std_a * std_a + std_b * std_b) / count)
    return float(norm.sf((mean_a - mean_b) / standard_error))
