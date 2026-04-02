from __future__ import annotations

import numpy as np

from experiments.common import safe_softmax


class ToyDenseVictim:
    """Simple deterministic dense probability victim."""

    def __init__(self, vocab_size: int = 16, seed: int = 0) -> None:
        self.vocab_size = vocab_size
        self.rng = np.random.default_rng(seed)
        self.W = self.rng.normal(0, 1, size=(vocab_size, vocab_size))

    def vectorize_prompt(self, prompt: str) -> np.ndarray:
        vec = np.zeros(self.vocab_size, dtype=np.float64)
        for ch in prompt:
            vec[ord(ch) % self.vocab_size] += 1.0
        norm = np.linalg.norm(vec)
        return vec / norm if norm > 0 else vec

    def full_probs(self, prompt: str) -> np.ndarray:
        x = self.vectorize_prompt(prompt)
        logits = self.W @ x
        return safe_softmax(logits)
