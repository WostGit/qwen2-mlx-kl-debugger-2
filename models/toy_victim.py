"""Toy victim model for dense, always-valid next-class probabilities."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


@dataclass
class ToyVictim:
    num_classes: int = 20
    temperature: float = 1.0

    def _rng_for_prompt(self, prompt: str, seed: int) -> np.random.Generator:
        digest = hashlib.sha256(f"{prompt}|{seed}".encode("utf-8")).hexdigest()
        int_seed = int(digest[:16], 16)
        return np.random.default_rng(int_seed)

    def dense_probs(self, prompt: str, seed: int) -> np.ndarray:
        rng = self._rng_for_prompt(prompt, seed)
        logits = rng.normal(loc=0.0, scale=1.0, size=self.num_classes)
        logits = logits / max(self.temperature, 1e-8)
        logits -= np.max(logits)
        probs = np.exp(logits)
        probs /= np.sum(probs)
        return probs.astype(np.float64)

    def interface_view(self, prompt: str, seed: int, interface: str, topk: int = 5) -> np.ndarray:
        probs = self.dense_probs(prompt=prompt, seed=seed)
        return interface_from_dense_probs(probs=probs, interface=interface, topk=topk)


def interface_from_dense_probs(probs: np.ndarray, interface: str, topk: int = 5) -> np.ndarray:
    if interface == "probs":
        return probs.copy()
    if interface == "argmax":
        out = np.zeros_like(probs)
        out[np.argmax(probs)] = 1.0
        return out
    if interface in {"topk", "top2", "top3", "top5"}:
        if interface.startswith("top") and interface != "topk":
            topk = int(interface[3:])
        k = min(topk, probs.shape[0])
        idx = np.argpartition(probs, -k)[-k:]
        out = np.zeros_like(probs)
        out[idx] = probs[idx]
        denom = np.sum(out)
        if denom <= 0:
            out[idx] = 1.0 / len(idx)
        else:
            out /= denom
        return out
    raise ValueError(f"Unsupported interface: {interface}")
