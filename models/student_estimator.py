from __future__ import annotations

import numpy as np


class PromptTableStudent:
    """Memorizes target distribution per prompt vector key; fallback is average."""

    def __init__(self, vocab_size: int) -> None:
        self.vocab_size = vocab_size
        self.table: dict[str, np.ndarray] = {}
        self.global_mean = np.ones(vocab_size, dtype=np.float64) / vocab_size

    @staticmethod
    def _key(prompt: str) -> str:
        return prompt.strip().lower()

    def fit(self, prompts: list[str], targets: list[np.ndarray]) -> None:
        if len(prompts) != len(targets):
            raise ValueError("prompts and targets length mismatch")
        accum = []
        for p, t in zip(prompts, targets):
            if t.shape[0] != self.vocab_size:
                raise ValueError("target vocab mismatch")
            self.table[self._key(p)] = t.astype(np.float64)
            accum.append(t)
        if accum:
            mean = np.mean(np.stack(accum), axis=0)
            s = mean.sum()
            self.global_mean = mean / s

    def predict(self, prompt: str) -> np.ndarray:
        return self.table.get(self._key(prompt), self.global_mean)
