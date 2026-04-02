from __future__ import annotations
import argparse
import numpy as np
import pandas as pd

from models.toy_victim import ToyVictimModel
from attacks.student_estimators import FrequencyStudent
from experiments.common import kl_divergence, top1_agreement, ensure_dir


def interface_target(victim_probs: np.ndarray, interface: str) -> np.ndarray:
    t = np.zeros_like(victim_probs)
    if interface == "probs":
        return victim_probs.copy()
    if interface == "argmax":
        t[int(np.argmax(victim_probs))] = 1.0
        return t
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argsort(victim_probs)[::-1][:k]
        t[idx] = victim_probs[idx]
        t = t / np.sum(t)
        return t
    raise ValueError(f"Unknown interface: {interface}")


def run_toy(budgets, seeds, interfaces, out_csv: str):
    rows = []
    for seed in seeds:
        victim = ToyVictimModel(seed=seed)
        for budget in budgets:
            for interface in interfaces:
                targets = []
                for qid in range(budget):
                    vp = victim.probs(qid)
                    targets.append(interface_target(vp, interface))
                targets = np.vstack(targets)

                student = FrequencyStudent(vocab_size=victim.n_classes)
                student.fit(targets)
                student_p = student.predict_proba()

                for eval_qid in range(64):
                    victim_full = victim.probs(10_000 + eval_qid)
                    agreement = top1_agreement(victim_full, student_p)
                    kl = kl_divergence(victim_full, student_p)
                    rows.append(
                        {
                            "source": "toy",
                            "interface": interface,
                            "budget": budget,
                            "seed": seed,
                            "prompt_id": f"toy_{eval_qid}",
                            "agreement": agreement,
                            "kl_divergence": kl,
                            "kl_valid": int(np.isfinite(kl)),
                            "invalid_reason": "" if np.isfinite(kl) else "non_finite_kl",
                        }
                    )

    df = pd.DataFrame(rows)
    ensure_dir("results")
    df.to_csv(out_csv, index=False)
    print(f"[toy] wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/toy_raw.csv")
    args = ap.parse_args()
    run_toy(
        budgets=[64, 128, 256, 512, 1024],
        seeds=[0, 1, 2],
        interfaces=["argmax", "top3", "probs"],
        out_csv=args.out,
    )
