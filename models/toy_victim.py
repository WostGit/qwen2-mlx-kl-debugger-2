"""Toy victim model that always returns dense valid probability vectors."""

from __future__ import annotations

import numpy as np

from models.common import deterministic_seed_from_text, stable_softmax


class ToyVictimModel:
    def __init__(self, num_classes: int = 32, base_seed: int = 0):
        self.num_classes = num_classes
        self.base_seed = base_seed

    def full_probs(self, prompt: str) -> np.ndarray:
        rs = np.random.RandomState(deterministic_seed_from_text(prompt, self.base_seed))
        logits = rs.normal(loc=0.0, scale=1.0, size=self.num_classes)
        return stable_softmax(logits)

    def interface_view(self, prompt: str, interface: str, topk: int = 5) -> np.ndarray:
        probs = self.full_probs(prompt)
        if interface == "probs":
            return probs
        if interface == "argmax":
            out = np.zeros_like(probs)
            out[int(np.argmax(probs))] = 1.0
            return out
        if interface.startswith("top"):
            k = topk if interface == "topk" else int(interface.replace("top", ""))
            idx = np.argsort(probs)[-k:]
            out = np.zeros_like(probs)
            out[idx] = probs[idx]
            out /= out.sum()
            return out
        raise ValueError(f"Unknown interface: {interface}")
