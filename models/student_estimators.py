"""Simple dense student estimators for imitation in constrained budget settings."""

from __future__ import annotations

import numpy as np

from models.common import stable_softmax


class DenseFrequencyStudent:
    def __init__(self, vocab_size: int, smoothing: float = 1e-3):
        self.vocab_size = vocab_size
        self.counts = np.full(vocab_size, smoothing, dtype=np.float64)

    def fit_targets(self, target_vectors: list[np.ndarray]) -> None:
        for vec in target_vectors:
            arr = np.asarray(vec, dtype=np.float64)
            if arr.shape[0] != self.vocab_size:
                raise ValueError(f"Student got vector size {arr.shape[0]} expected {self.vocab_size}")
            self.counts += arr

    def predict_dense(self, prompt: str | None = None) -> np.ndarray:
        del prompt
        probs = self.counts / self.counts.sum()
        return probs


class PromptLinearStudent:
    """A tiny full-distribution estimator for Qwen using prompt token bag-of-ids features."""

    def __init__(self, vocab_size: int, lr: float = 0.5, epochs: int = 3, seed: int = 0):
        self.vocab_size = vocab_size
        self.lr = lr
        self.epochs = epochs
        self.seed = seed
        rs = np.random.RandomState(seed)
        self.W = rs.normal(scale=0.01, size=(vocab_size, vocab_size))

    def _features(self, prompt_ids: list[int]) -> np.ndarray:
        x = np.zeros(self.vocab_size, dtype=np.float64)
        for tid in prompt_ids[-32:]:
            if 0 <= tid < self.vocab_size:
                x[tid] += 1.0
        s = x.sum()
        if s > 0:
            x /= s
        return x

    def fit(self, prompt_ids_batch: list[list[int]], targets: list[np.ndarray]) -> None:
        for _ in range(self.epochs):
            for ids, target in zip(prompt_ids_batch, targets):
                x = self._features(ids)
                logits = self.W @ x
                pred = stable_softmax(logits)
                grad = np.outer((pred - target), x)
                self.W -= self.lr * grad

    def predict_dense(self, prompt_ids: list[int]) -> np.ndarray:
        x = self._features(prompt_ids)
        logits = self.W @ x
        return stable_softmax(logits)
