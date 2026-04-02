"""Toy baseline budget sweep with always-valid dense vectors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from attacks.metrics import agreement, safe_kl
from experiments.prompts import TOY_PROMPTS
from models.toy_victim import ToyVictimModel
from models.student_estimators import DenseFrequencyStudent


@dataclass
class ToyConfig:
    budgets: list[int]
    seeds: list[int]
    interfaces: list[str]
    num_classes: int = 32
    eval_examples: int = 80


def run_toy_budget_sweep(cfg: ToyConfig) -> pd.DataFrame:
    rows = []
    for seed in cfg.seeds:
        victim = ToyVictimModel(num_classes=cfg.num_classes, base_seed=seed)
        for budget in cfg.budgets:
            train_prompts = TOY_PROMPTS[:budget]
            eval_prompts = TOY_PROMPTS[2000 : 2000 + cfg.eval_examples]
            for interface in cfg.interfaces:
                student = DenseFrequencyStudent(vocab_size=cfg.num_classes)
                targets = [victim.interface_view(p, interface=interface) for p in train_prompts]
                student.fit_targets(targets)
                for i, prompt in enumerate(eval_prompts):
                    vp = victim.full_probs(prompt)
                    sp = student.predict_dense(prompt)
                    kl, kl_valid, kl_reason = safe_kl(vp, sp)
                    rows.append(
                        {
                            "source": "toy",
                            "prompt_id": f"toy_eval_{i}",
                            "prompt_text": prompt,
                            "interface": interface,
                            "budget": budget,
                            "seed": seed,
                            "agreement": agreement(vp, sp),
                            "kl_divergence": kl,
                            "kl_valid": kl_valid,
                            "kl_reason": kl_reason,
                            "vocab_alignment_passed": True,
                        }
                    )
    return pd.DataFrame(rows)
