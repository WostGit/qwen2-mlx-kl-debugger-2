from __future__ import annotations
import numpy as np


class FrequencyStudent:
    """Simple estimator: average target distribution over observed queries."""

    def __init__(self, vocab_size: int, eps: float = 1e-12):
        self.vocab_size = vocab_size
        self.counts = np.zeros(vocab_size, dtype=np.float64)
        self.eps = eps

    def fit(self, targets: np.ndarray):
        if targets.ndim != 2 or targets.shape[1] != self.vocab_size:
            raise ValueError(f"targets shape {targets.shape} incompatible with vocab_size={self.vocab_size}")
        self.counts = np.sum(targets, axis=0)

    def predict_proba(self) -> np.ndarray:
        probs = self.counts + self.eps
        probs = probs / np.sum(probs)
        return probs
