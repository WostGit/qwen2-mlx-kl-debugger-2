"""Qwen2-0.5B victim adapter via MLX with explicit full-logit extraction."""

from __future__ import annotations

import numpy as np

import mlx.core as mx
from mlx_lm import load

from models.common import stable_softmax, topn


class QwenVictimMLX:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct"):
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)
        self.vocab_size = int(self.tokenizer.vocab_size)

    def tokenize(self, prompt: str) -> list[int]:
        ids = self.tokenizer.encode(prompt)
        return list(ids)

    def decode(self, token_ids: list[int]) -> list[str]:
        return [self.tokenizer.decode([int(t)]) for t in token_ids]

    def _mlx_logits_for_next_token(self, prompt_ids: list[int]) -> np.ndarray:
        x = mx.array([prompt_ids])
        out = self.model(x)
        # Common MLX LM shape is [batch, seq, vocab]; we only need final step.
        logits = np.array(out[:, -1, :])[0]
        if logits.shape[0] != self.vocab_size:
            raise ValueError(
                f"Logit vocab mismatch. logits={logits.shape[0]} tokenizer={self.vocab_size}"
            )
        return logits

    def dense_next_token_probs(self, prompt: str) -> dict:
        prompt_ids = self.tokenize(prompt)
        logits = self._mlx_logits_for_next_token(prompt_ids)
        probs = stable_softmax(logits)
        top_ids, top_probs = topn(probs, n=10)
        top_tokens = self.decode(top_ids.tolist())
        return {
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "logits": logits,
            "probs": probs,
            "top10_ids": top_ids.tolist(),
            "top10_tokens": top_tokens,
            "top10_probs": top_probs.tolist(),
            "vocab_size": self.vocab_size,
        }


def derive_interface_target(full_probs: np.ndarray, interface: str) -> np.ndarray:
    p = np.array(full_probs, dtype=np.float64)
    if interface == "probs":
        return p
    if interface == "argmax":
        out = np.zeros_like(p)
        out[int(np.argmax(p))] = 1.0
        return out
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argsort(p)[-k:]
        out = np.zeros_like(p)
        out[idx] = p[idx]
        out /= out.sum()
        return out
    raise ValueError(f"Unsupported interface: {interface}")
