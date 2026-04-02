"""Common math and validation utilities with intentionally verbose checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

EPS = 1e-12


def stable_softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    denom = np.sum(exps)
    if not np.isfinite(denom) or denom <= 0:
        raise ValueError(f"Invalid softmax denominator={denom}")
    probs = exps / denom
    return probs


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if p.shape != q.shape:
        raise ValueError(f"KL shape mismatch: p{p.shape} vs q{q.shape}")
    if np.any(p < 0) or np.any(q < 0):
        raise ValueError("KL received negative probabilities")
    sp = float(p.sum())
    sq = float(q.sum())
    if abs(sp - 1.0) > 1e-6 or abs(sq - 1.0) > 1e-6:
        raise ValueError(f"KL requires normalized vectors: sum(p)={sp}, sum(q)={sq}")
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        raise ValueError("KL received NaN or inf values")
    mask = p > 0
    return float(np.sum(p[mask] * np.log((p[mask] + EPS) / (q[mask] + EPS))))


def deterministic_seed_from_text(text: str, base_seed: int = 0) -> int:
    digest = hashlib.sha256(f"{base_seed}:{text}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) % (2**31 - 1)


@dataclass
class VectorHealth:
    prob_sum: float
    has_nan: bool
    has_inf: bool


def check_vector_health(vec: np.ndarray) -> VectorHealth:
    vec = np.asarray(vec)
    return VectorHealth(
        prob_sum=float(np.sum(vec)),
        has_nan=bool(np.any(np.isnan(vec))),
        has_inf=bool(np.any(np.isinf(vec))),
    )


def topn(vec: np.ndarray, n: int = 10) -> tuple[np.ndarray, np.ndarray]:
    idx = np.argsort(vec)[-n:][::-1]
    vals = vec[idx]
    return idx, vals
