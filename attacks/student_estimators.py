"""Simple dense students used in toy and Qwen experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PromptTableStudent:
    vocab_size: int
    smoothing: float = 1e-6
    table: dict[int, np.ndarray] = field(default_factory=dict)
    counts: dict[int, int] = field(default_factory=dict)
    global_sum: np.ndarray | None = None
    global_count: int = 0

    def fit(self, prompt_id: int, target_probs: np.ndarray) -> None:
        if target_probs.shape[0] != self.vocab_size:
            raise ValueError("target_probs has wrong vocab size")
        if self.global_sum is None:
            self.global_sum = np.zeros(self.vocab_size, dtype=np.float64)
        self.global_sum += target_probs
        self.global_count += 1

        if prompt_id not in self.table:
            self.table[prompt_id] = np.zeros(self.vocab_size, dtype=np.float64)
            self.counts[prompt_id] = 0
        self.table[prompt_id] += target_probs
        self.counts[prompt_id] += 1

    def predict_dense(self, prompt_id: int) -> np.ndarray:
        if prompt_id in self.table and self.counts[prompt_id] > 0:
            probs = self.table[prompt_id] / self.counts[prompt_id]
        elif self.global_sum is not None and self.global_count > 0:
            probs = self.global_sum / self.global_count
        else:
            probs = np.ones(self.vocab_size, dtype=np.float64) / self.vocab_size

        probs = np.clip(probs + self.smoothing, 1e-12, None)
        probs /= np.sum(probs)
        return probs
