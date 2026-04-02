from __future__ import annotations
import json
from pathlib import Path
import numpy as np


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape:
        raise ValueError(f"KL shape mismatch: {p.shape} vs {q.shape}")
    if not np.all(np.isfinite(p)) or not np.all(np.isfinite(q)):
        return np.nan
    if np.any(p < 0) or np.any(q < 0):
        return np.nan
    ps = np.sum(p)
    qs = np.sum(q)
    if ps <= 0 or qs <= 0:
        return np.nan
    p = p / ps
    q = q / qs
    if np.any(q <= 0):
        return np.nan
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def top1_agreement(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape:
        raise ValueError(f"Agreement shape mismatch: {p.shape} vs {q.shape}")
    return float(int(np.argmax(p) == np.argmax(q)))


def ensure_dir(path: str | Path):
    Path(path).mkdir(parents=True, exist_ok=True)


def write_jsonl(path: str | Path, rows: list[dict]):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
