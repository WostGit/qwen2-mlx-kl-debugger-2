from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_softmax(logits: np.ndarray) -> np.ndarray:
    if logits.ndim != 1:
        raise ValueError(f"Expected 1D logits, got shape={logits.shape}")
    shifted = logits - np.max(logits)
    exps = np.exp(shifted)
    total = np.sum(exps)
    if not np.isfinite(total) or total <= 0:
        raise ValueError(f"Invalid softmax denominator: {total}")
    probs = exps / total
    return probs


def topk_projection(probs: np.ndarray, k: int) -> np.ndarray:
    if k <= 0:
        raise ValueError("k must be > 0")
    if k > probs.shape[0]:
        raise ValueError(f"k={k} exceeds vocab size={probs.shape[0]}")
    idx = np.argpartition(probs, -k)[-k:]
    out = np.zeros_like(probs)
    out[idx] = probs[idx]
    mass = out.sum()
    if mass <= 0 or not np.isfinite(mass):
        raise ValueError(f"Invalid topk mass: {mass}")
    out /= mass
    return out


def argmax_projection(probs: np.ndarray) -> np.ndarray:
    out = np.zeros_like(probs)
    out[int(np.argmax(probs))] = 1.0
    return out


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-12) -> float:
    if p.shape != q.shape:
        raise ValueError(f"Shape mismatch: p{p.shape} vs q{q.shape}")
    p_safe = np.clip(p, eps, 1.0)
    q_safe = np.clip(q, eps, 1.0)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def summarize_results(raw_df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    keys = ["source", "interface", "budget"]
    for (source, interface, budget), grp in raw_df.groupby(keys):
        valid_kl = grp["kl_divergence"].notna()
        grouped.append(
            {
                "source": source,
                "interface": interface,
                "budget": int(budget),
                "n_rows": int(len(grp)),
                "n_valid_kl": int(valid_kl.sum()),
                "n_invalid_kl": int((~valid_kl).sum()),
                "agreement_mean": float(grp["agreement"].mean()),
                "agreement_std": float(grp["agreement"].std(ddof=0)),
                "kl_mean": float(grp.loc[valid_kl, "kl_divergence"].mean()) if valid_kl.any() else np.nan,
                "kl_std": float(grp.loc[valid_kl, "kl_divergence"].std(ddof=0)) if valid_kl.any() else np.nan,
            }
        )
    return pd.DataFrame(grouped)


def assert_summary_not_zero_filled(summary_df: pd.DataFrame) -> None:
    bad = summary_df[(summary_df["n_invalid_kl"] > 0) & (summary_df["kl_mean"] == 0.0)]
    if not bad.empty:
        raise RuntimeError("Detected forbidden zero-filled KL in summary")


def fail_if_too_many_invalid_qwen(raw_df: pd.DataFrame, threshold: float = 0.01) -> None:
    qwen = raw_df[raw_df["source"] == "qwen"]
    if qwen.empty:
        return
    ratio = qwen["kl_divergence"].isna().mean()
    if ratio > threshold:
        raise RuntimeError(f"Qwen invalid KL ratio {ratio:.4f} exceeds threshold {threshold:.4f}")
