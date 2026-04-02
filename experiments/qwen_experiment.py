from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.student_estimators import PromptTableStudent
from models.qwen_victim_mlx import QwenVictimMLX
from models.toy_victim import interface_from_dense_probs

QWEN_PROMPTS = [
    "Explain gravity in one sentence.",
    "Name a prime number between 50 and 70.",
    "Complete: The capital of Japan is",
    "Write one word that rhymes with 'blue'.",
    "What is 7 multiplied by 8?",
    "Give a synonym for 'fast'.",
    "Finish: Machine learning is",
    "What color is the daytime sky?",
    "Provide one antonym of 'hot'.",
    "Complete: Water freezes at",
    "Name one planet in our solar system.",
    "Write one adjective for the ocean.",
]


def kl_divergence(p: np.ndarray, q: np.ndarray) -> tuple[float, str]:
    if p.shape != q.shape:
        return np.nan, "shape_mismatch"
    if np.any(~np.isfinite(p)):
        return np.nan, "victim_has_nan_or_inf"
    if np.any(~np.isfinite(q)):
        return np.nan, "student_has_nan_or_inf"
    if np.any(p < 0) or np.any(q < 0):
        return np.nan, "negative_probability"
    if not np.isclose(np.sum(p), 1.0, atol=1e-4):
        return np.nan, "victim_not_normalized"
    if not np.isclose(np.sum(q), 1.0, atol=1e-4):
        return np.nan, "student_not_normalized"
    if np.any(q <= 0):
        return np.nan, "student_contains_zero"
    value = float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
    if not np.isfinite(value):
        return np.nan, "kl_not_finite"
    return value, ""


def run_qwen(
    budgets: list[int],
    seeds: list[int],
    debug_llm: bool,
    out_csv: Path,
    debug_dir: Path,
    nan_report_csv: Path,
    vocab_check_csv: Path,
) -> pd.DataFrame:
    victim = QwenVictimMLX()
    rows = []
    nan_rows = []
    vocab_checks = []
    debug_dir.mkdir(parents=True, exist_ok=True)

    budget_interfaces = ["argmax", "topk", "probs"]
    fixed_topk_interfaces = ["argmax", "top2", "top3", "top5", "probs"]

    for seed in seeds:
        rng = np.random.default_rng(seed)

        for budget in budgets:
            for interface in budget_interfaces:
                rows.extend(
                    _run_one_setting(
                        victim=victim,
                        rng=rng,
                        seed=seed,
                        budget=budget,
                        interface=interface,
                        debug_llm=debug_llm,
                        debug_dir=debug_dir,
                        nan_rows=nan_rows,
                        vocab_checks=vocab_checks,
                    )
                )

        fixed_budget = 256
        for interface in fixed_topk_interfaces:
            rows.extend(
                _run_one_setting(
                    victim=victim,
                    rng=rng,
                    seed=seed,
                    budget=fixed_budget,
                    interface=interface,
                    debug_llm=debug_llm,
                    debug_dir=debug_dir,
                    nan_rows=nan_rows,
                    vocab_checks=vocab_checks,
                    is_topk_sweep=True,
                )
            )

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    pd.DataFrame(nan_rows).to_csv(nan_report_csv, index=False)
    pd.DataFrame(vocab_checks).to_csv(vocab_check_csv, index=False)

    qwen_rows = df[df["source"] == "qwen"]
    nan_rate = qwen_rows["kl_divergence"].isna().mean()
    if nan_rate > 0.01:
        raise RuntimeError(f"Qwen KL NaN rate too high: {nan_rate:.4f} > 0.01")

    return df


def _run_one_setting(
    victim: QwenVictimMLX,
    rng: np.random.Generator,
    seed: int,
    budget: int,
    interface: str,
    debug_llm: bool,
    debug_dir: Path,
    nan_rows: list[dict],
    vocab_checks: list[dict],
    is_topk_sweep: bool = False,
) -> list[dict]:
    first_probs, _, err = victim.get_next_token_probs(QWEN_PROMPTS[0])
    if err or first_probs is None:
        raise RuntimeError(f"Could not initialize Qwen victim probs: {err}")

    vocab_size = first_probs.shape[0]
    student = PromptTableStudent(vocab_size=vocab_size)

    for _ in range(budget):
        pid = int(rng.integers(0, len(QWEN_PROMPTS)))
        prompt = QWEN_PROMPTS[pid]
        victim_dense, _, error = victim.get_next_token_probs(prompt)
        if error or victim_dense is None:
            continue
        target = interface_from_dense_probs(victim_dense, interface=interface, topk=5)
        student.fit(prompt_id=pid, target_probs=target)

    setting_rows: list[dict] = []
    debug_records = []

    for pid, prompt in enumerate(QWEN_PROMPTS):
        victim_dense, token_ids, victim_error = victim.get_next_token_probs(prompt)
        row = {
            "source": "qwen",
            "interface": interface,
            "budget": budget,
            "seed": seed,
            "prompt_id": pid,
            "prompt": prompt,
            "is_topk_sweep": int(is_topk_sweep),
        }

        if victim_error or victim_dense is None:
            row.update(
                {
                    "agreement": np.nan,
                    "kl_divergence": np.nan,
                    "valid_kl": 0,
                    "invalid_reason": f"victim_error:{victim_error}",
                }
            )
            setting_rows.append(row)
            nan_rows.append({**row, "failure_reason": row["invalid_reason"]})
            continue

        student_dense = student.predict_dense(prompt_id=pid)
        vocab_ok = int(victim_dense.shape[0] == student_dense.shape[0])
        vocab_checks.append(
            {
                "seed": seed,
                "interface": interface,
                "budget": budget,
                "prompt_id": pid,
                "victim_vocab_size": int(victim_dense.shape[0]),
                "student_vocab_size": int(student_dense.shape[0]),
                "index_mapping_assumed_identical": 1,
                "alignment_pass": vocab_ok,
            }
        )

        if not vocab_ok:
            kl_value = np.nan
            reason = "vocab_mismatch"
            agreement = np.nan
        else:
            agreement = int(np.argmax(victim_dense) == np.argmax(student_dense))
            kl_value, reason = kl_divergence(victim_dense, student_dense)

        valid_kl = int(np.isfinite(kl_value))
        row.update(
            {
                "agreement": agreement,
                "kl_divergence": kl_value,
                "valid_kl": valid_kl,
                "invalid_reason": "" if valid_kl else reason,
            }
        )
        setting_rows.append(row)

        if not valid_kl:
            nan_rows.append({**row, "failure_reason": row["invalid_reason"]})

        if debug_llm and pid < 10:
            debug_records.append(
                _debug_record(
                    prompt_id=pid,
                    prompt=prompt,
                    token_ids=token_ids,
                    victim_dense=victim_dense,
                    student_dense=student_dense,
                    vocab_ok=bool(vocab_ok),
                    kl_value=kl_value,
                    kl_reason=reason,
                    victim=victim,
                )
            )

    if debug_llm and debug_records:
        stem = f"qwen_debug_seed{seed}_budget{budget}_{interface}{'_topk' if is_topk_sweep else ''}"
        jsonl = debug_dir / f"{stem}.jsonl"
        csv = debug_dir / f"{stem}.csv"
        with jsonl.open("w", encoding="utf-8") as f:
            for rec in debug_records:
                f.write(json.dumps(rec) + "\n")
        pd.DataFrame(debug_records).to_csv(csv, index=False)

    return setting_rows


def _debug_record(
    prompt_id: int,
    prompt: str,
    token_ids: list[int],
    victim_dense: np.ndarray,
    student_dense: np.ndarray,
    vocab_ok: bool,
    kl_value: float,
    kl_reason: str,
    victim: QwenVictimMLX,
) -> dict:
    v_idx = np.argsort(victim_dense)[-10:][::-1]
    s_idx = np.argsort(student_dense)[-10:][::-1]
    return {
        "prompt_id": prompt_id,
        "prompt": prompt,
        "tokenized_prompt_ids": token_ids,
        "victim_top10_token_ids": [int(x) for x in v_idx],
        "victim_top10_token_strings": [victim.decode_token(int(x)) for x in v_idx],
        "victim_top10_probabilities": [float(victim_dense[x]) for x in v_idx],
        "student_top10_token_ids": [int(x) for x in s_idx],
        "student_top10_token_strings": [victim.decode_token(int(x)) for x in s_idx],
        "student_top10_probabilities": [float(student_dense[x]) for x in s_idx],
        "victim_prob_sum": float(np.sum(victim_dense)),
        "student_prob_sum": float(np.sum(student_dense)),
        "victim_has_nan_or_inf": bool(np.any(~np.isfinite(victim_dense))),
        "student_has_nan_or_inf": bool(np.any(~np.isfinite(student_dense))),
        "full_vocab_size": int(victim_dense.shape[0]),
        "vocabulary_alignment_passed": bool(vocab_ok),
        "kl_divergence": float(kl_value) if np.isfinite(kl_value) else np.nan,
        "kl_skipped_reason": "" if np.isfinite(kl_value) else kl_reason,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=str, default="64,128,256,512,1024")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--out-csv", type=Path, default=Path("results/qwen_results_raw.csv"))
    ap.add_argument("--debug-dir", type=Path, default=Path("results/debug"))
    ap.add_argument("--nan-report-csv", type=Path, default=Path("results/qwen_nan_report.csv"))
    ap.add_argument(
        "--vocab-check-csv", type=Path, default=Path("results/qwen_vocab_alignment_check.csv")
    )
    ap.add_argument("--debug-llm", action="store_true")
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    run_qwen(
        budgets=budgets,
        seeds=seeds,
        debug_llm=args.debug_llm,
        out_csv=args.out_csv,
        debug_dir=args.debug_dir,
        nan_report_csv=args.nan_report_csv,
        vocab_check_csv=args.vocab_check_csv,
    )


if __name__ == "__main__":
    main()
