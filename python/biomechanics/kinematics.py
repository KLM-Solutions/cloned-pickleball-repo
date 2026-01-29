"""Velocities / accelerations helpers."""
from typing import List


def velocity(series: List[float], dt: float = 1.0) -> List[float]:
    if len(series) < 2:
        return [0.0] * len(series)
    return [0.0] + [(series[i] - series[i-1]) / dt for i in range(1, len(series))]
