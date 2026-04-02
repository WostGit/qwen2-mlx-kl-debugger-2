"""Metrics with explicit validity bookkeeping."""

from __future__ import annotations

import numpy as np

from models.common import check_vector_health, kl_divergence


def agreement(victim_probs: np.ndarray, student_probs: np.ndarray) -> float:
    return float(int(np.argmax(victim_probs) == np.argmax(student_probs)))


def safe_kl(victim_probs: np.ndarray, student_probs: np.ndarray) -> tuple[float, bool, str]:
    hv = check_vector_health(victim_probs)
    hs = check_vector_health(student_probs)
    if hv.has_nan or hs.has_nan:
        return np.nan, False, "nan_in_vector"
    if hv.has_inf or hs.has_inf:
        return np.nan, False, "inf_in_vector"
    if abs(hv.prob_sum - 1.0) > 1e-6:
        return np.nan, False, f"victim_not_normalized:{hv.prob_sum}"
    if abs(hs.prob_sum - 1.0) > 1e-6:
        return np.nan, False, f"student_not_normalized:{hs.prob_sum}"
    if victim_probs.shape != student_probs.shape:
        return np.nan, False, f"shape_mismatch:{victim_probs.shape}!={student_probs.shape}"
    return kl_divergence(victim_probs, student_probs), True, "ok"
