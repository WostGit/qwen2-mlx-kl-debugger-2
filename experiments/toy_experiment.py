from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from attacks.train_student import make_interface_target
from experiments.common import kl_divergence, set_seed
from models.student_estimator import PromptTableStudent
from models.toy_victim import ToyDenseVictim


@dataclass
class ToyConfig:
    budgets: list[int]
    seeds: list[int]
    interfaces: list[str]


def build_toy_prompts(n: int = 1200) -> list[str]:
    return [f"toy prompt {i} value {i % 7}" for i in range(n)]


def run_toy(config: ToyConfig) -> pd.DataFrame:
    prompts = build_toy_prompts(max(config.budgets) + 200)
    rows: list[dict] = []
    for seed in config.seeds:
        set_seed(seed)
        victim = ToyDenseVictim(vocab_size=16, seed=seed)
        eval_prompts = prompts[-120:]
        victim_eval = {p: victim.full_probs(p) for p in eval_prompts}
        for budget in config.budgets:
            train_prompts = prompts[:budget]
            full_probs_train = [victim.full_probs(p) for p in train_prompts]
            for interface in config.interfaces:
                targets = [make_interface_target(v, interface) for v in full_probs_train]
                student = PromptTableStudent(vocab_size=16)
                student.fit(train_prompts, targets)
                for p in eval_prompts:
                    vfull = victim_eval[p]
                    sp = student.predict(p)
                    row = {
                        "source": "toy",
                        "interface": interface,
                        "budget": budget,
                        "seed": seed,
                        "agreement": float(np.argmax(vfull) == np.argmax(sp)),
                        "kl_divergence": float(kl_divergence(vfull, sp)),
                        "is_valid_kl": True,
                        "invalid_reason": "",
                    }
                    rows.append(row)
    return pd.DataFrame(rows)
