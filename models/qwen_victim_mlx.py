"""Qwen2-0.5B victim wrapper via MLX / mlx-lm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

try:
    import mlx.core as mx
    from mlx_lm import load
except Exception as exc:  # pragma: no cover
    mx = None
    load = None
    IMPORT_ERROR = exc
else:
    IMPORT_ERROR = None


@dataclass
class QwenVictimMLX:
    model_name: str = "Qwen/Qwen2-0.5B-Instruct"

    def __post_init__(self) -> None:
        if IMPORT_ERROR is not None:
            raise RuntimeError(f"mlx/mlx-lm import failed: {IMPORT_ERROR}")
        self.model, self.tokenizer = load(self.model_name)

    def encode(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        return list(ids)

    def decode_token(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([int(token_id)])
        except Exception:
            return f"<tok_{token_id}>"

    def get_next_token_probs(self, prompt: str) -> tuple[np.ndarray | None, list[int], str | None]:
        token_ids = self.encode(prompt)
        if len(token_ids) == 0:
            return None, token_ids, "empty_tokenized_prompt"

        try:
            x = mx.array([token_ids], dtype=mx.int32)
            raw = self.model(x)
            logits = _extract_logits(raw)
            logits_np = np.array(logits)
            if logits_np.ndim == 3:
                next_logits = logits_np[0, -1, :]
            elif logits_np.ndim == 2:
                next_logits = logits_np[-1, :]
            else:
                return None, token_ids, f"unexpected_logits_rank_{logits_np.ndim}"
            probs = softmax(next_logits)
            return probs, token_ids, None
        except Exception as exc:
            return None, token_ids, f"mlx_forward_exception:{type(exc).__name__}:{exc}"


def _extract_logits(raw: Any) -> Any:
    if isinstance(raw, dict):
        if "logits" in raw:
            return raw["logits"]
        return _extract_logits(next(iter(raw.values())))
    if isinstance(raw, (tuple, list)):
        return _extract_logits(raw[0])
    return raw


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits.astype(np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    z = np.sum(e)
    if not np.isfinite(z) or z <= 0:
        return np.full_like(e, fill_value=np.nan, dtype=np.float64)
    return e / z
