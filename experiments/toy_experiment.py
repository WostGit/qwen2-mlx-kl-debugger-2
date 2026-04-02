from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.student_estimators import PromptTableStudent
from models.toy_victim import ToyVictim
from models.toy_victim import interface_from_dense_probs

TOY_PROMPTS = [f"toy_prompt_{i}" for i in range(40)]


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    if p.shape != q.shape:
        return np.nan
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(q)):
        return np.nan
    if np.any(p < 0) or np.any(q <= 0):
        return np.nan
    return float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))


def run_toy(budgets: list[int], seeds: list[int], out_csv: Path) -> pd.DataFrame:
    victim = ToyVictim(num_classes=20)
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        for budget in budgets:
            for interface in ["argmax", "topk", "probs"]:
                student = PromptTableStudent(vocab_size=victim.num_classes)
                for _ in range(budget):
                    pid = int(rng.integers(0, len(TOY_PROMPTS)))
                    prompt = TOY_PROMPTS[pid]
                    dense = victim.dense_probs(prompt, seed=seed)
                    target = interface_from_dense_probs(dense, interface=interface, topk=5)
                    student.fit(prompt_id=pid, target_probs=target)

                for pid, prompt in enumerate(TOY_PROMPTS):
                    victim_dense = victim.dense_probs(prompt, seed=seed)
                    student_dense = student.predict_dense(prompt_id=pid)
                    agreement = int(np.argmax(victim_dense) == np.argmax(student_dense))
                    kl = kl_divergence(victim_dense, student_dense)
                    rows.append(
                        {
                            "source": "toy",
                            "interface": interface,
                            "budget": budget,
                            "seed": seed,
                            "prompt_id": pid,
                            "prompt": prompt,
                            "agreement": agreement,
                            "kl_divergence": kl,
                            "valid_kl": int(np.isfinite(kl)),
                            "invalid_reason": "" if np.isfinite(kl) else "toy_invalid_kl",
                        }
                    )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=str, default="64,128,256,512,1024")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--out-csv", type=Path, default=Path("results/toy_results_raw.csv"))
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]
    run_toy(budgets=budgets, seeds=seeds, out_csv=args.out_csv)


if __name__ == "__main__":
    main()
