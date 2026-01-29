"""Derived features (wrist-to-body, torso inclination, etc.)."""
from typing import Tuple


def norm_dist(dx_px: float, dy_px: float, diag_px: float) -> float:
    if diag_px <= 0:
        return 0.0
    return ((dx_px**2 + dy_px**2) ** 0.5) / diag_px
