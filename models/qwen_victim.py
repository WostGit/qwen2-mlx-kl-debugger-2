from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from mlx_lm import load

from experiments.common import safe_softmax


@dataclass
class QwenVictimOutput:
    token_ids: list[int]
    logits: np.ndarray
    probs: np.ndarray


class QwenMLXVictim:
    def __init__(self, model_name: str = "Qwen/Qwen2-0.5B-Instruct") -> None:
        self.model_name = model_name
        self.model, self.tokenizer = load(model_name)

    def next_token_distribution(self, prompt: str) -> QwenVictimOutput:
        token_ids = self.tokenizer.encode(prompt)
        arr = mx.array([token_ids])
        logits = self.model(arr)
        # logits: [batch, seq, vocab]
        last = np.array(logits[:, -1, :])[0].astype(np.float64)
        probs = safe_softmax(last)
        return QwenVictimOutput(token_ids=token_ids, logits=last, probs=probs)

    def decode_ids(self, ids: list[int]) -> list[str]:
        return [self.tokenizer.decode([int(i)]) for i in ids]
