from __future__ import annotations

import numpy as np

from experiments.common import argmax_projection, topk_projection


def make_interface_target(victim_probs: np.ndarray, interface: str) -> np.ndarray:
    if interface == "argmax":
        return argmax_projection(victim_probs)
    if interface == "probs":
        return victim_probs.copy()
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        return topk_projection(victim_probs, k)
    raise ValueError(f"Unknown interface: {interface}")
