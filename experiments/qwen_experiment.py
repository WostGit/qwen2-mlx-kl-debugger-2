from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

from attacks.student_estimators import FrequencyStudent
from experiments.common import kl_divergence, top1_agreement, ensure_dir, write_jsonl


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    ex = np.exp(x)
    return ex / np.sum(ex)


def load_qwen_mlx(model_id: str):
    from mlx_lm import load
    model, tokenizer = load(model_id)
    return model, tokenizer


def next_token_logits(model, token_ids: list[int]) -> np.ndarray:
    import mlx.core as mx
    x = mx.array([token_ids], dtype=mx.int32)
    out = model(x)
    logits = np.array(out.logits if hasattr(out, "logits") else out)
    return logits[0, -1, :]


def derive_target(victim_full: np.ndarray, interface: str) -> np.ndarray:
    target = np.zeros_like(victim_full)
    if interface == "probs":
        return victim_full.copy()
    if interface == "argmax":
        target[int(np.argmax(victim_full))] = 1.0
        return target
    if interface.startswith("top"):
        k = int(interface.replace("top", ""))
        idx = np.argsort(victim_full)[::-1][:k]
        target[idx] = victim_full[idx]
        s = np.sum(target)
        if s <= 0:
            return target
        return target / s
    raise ValueError(f"Unknown interface: {interface}")


def top_tokens(vec: np.ndarray, tokenizer, k=10):
    idx = np.argsort(vec)[::-1][:k]
    toks = []
    for t in idx:
        try:
            toks.append(tokenizer.decode([int(t)]))
        except Exception:
            toks.append("<decode_error>")
    return idx.tolist(), toks, vec[idx].tolist()


def run_qwen(prompts_path: str, out_csv: str, debug_llm: bool = False):
    with open(prompts_path, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)["prompts"]

    ensure_dir("results")
    ensure_dir("results/debug")
    model, tokenizer = load_qwen_mlx("Qwen/Qwen2-0.5B-Instruct")

    budget_sweep = [64, 128, 256, 512, 1024]
    common_seeds = [0, 1, 2]
    interfaces_by_budget = ["argmax", "top3", "probs"]
    topk_interfaces = ["argmax", "top2", "top3", "top5", "probs"]

    rows = []
    nan_rows = []
    vocab_rows = []
    debug_rows_by_key = {}

    def run_block(budget: int, seed: int, interface: str):
        rng = np.random.default_rng(seed)
        train_targets = []
        train_full = []
        train_prompts = [prompts[i % len(prompts)] for i in range(budget)]
        for p in train_prompts:
            token_ids = tokenizer.encode(p["text"])
            logits = next_token_logits(model, token_ids)
            vfull = softmax_np(logits.astype(np.float64))
            train_full.append(vfull)
            train_targets.append(derive_target(vfull, interface))

        train_targets_arr = np.vstack(train_targets)
        student = FrequencyStudent(vocab_size=train_targets_arr.shape[1])
        student.fit(train_targets_arr)
        student_p = student.predict_proba()

        for i, p in enumerate(prompts):
            token_ids = tokenizer.encode(p["text"])
            logits = next_token_logits(model, token_ids)
            victim_full = softmax_np(logits.astype(np.float64))

            vocab_ok = victim_full.shape == student_p.shape
            reason = ""
            kl = np.nan
            if not vocab_ok:
                reason = "vocab_shape_mismatch"
            elif not np.all(np.isfinite(victim_full)):
                reason = "victim_non_finite"
            elif not np.all(np.isfinite(student_p)):
                reason = "student_non_finite"
            else:
                kl = kl_divergence(victim_full, student_p)
                if not np.isfinite(kl):
                    reason = "kl_non_finite"

            agree = top1_agreement(victim_full, student_p) if vocab_ok else np.nan
            row = {
                "source": "qwen",
                "interface": interface,
                "budget": budget,
                "seed": seed,
                "prompt_id": p["id"],
                "agreement": agree,
                "kl_divergence": kl,
                "kl_valid": int(np.isfinite(kl)),
                "invalid_reason": reason,
            }
            rows.append(row)

            vocab_rows.append(
                {
                    "source": "qwen",
                    "interface": interface,
                    "budget": budget,
                    "seed": seed,
                    "prompt_id": p["id"],
                    "victim_vocab_size": int(victim_full.shape[0]),
                    "student_vocab_size": int(student_p.shape[0]),
                    "index_alignment_passed": int(vocab_ok),
                }
            )
            if reason:
                nan_rows.append({**row})

            if debug_llm and i < 10:
                v_ids, v_toks, v_probs = top_tokens(victim_full, tokenizer, k=10)
                s_ids, s_toks, s_probs = top_tokens(student_p, tokenizer, k=10)
                dbg = {
                    "prompt_id": p["id"],
                    "prompt_text": p["text"],
                    "tokenized_prompt_ids": token_ids,
                    "victim_top10_token_ids": v_ids,
                    "victim_top10_token_strings": v_toks,
                    "victim_top10_probabilities": v_probs,
                    "student_top10_token_ids": s_ids,
                    "student_top10_token_strings": s_toks,
                    "student_top10_probabilities": s_probs,
                    "victim_prob_sum": float(np.sum(victim_full)),
                    "student_prob_sum": float(np.sum(student_p)),
                    "victim_has_nan": bool(np.isnan(victim_full).any()),
                    "victim_has_inf": bool(np.isinf(victim_full).any()),
                    "student_has_nan": bool(np.isnan(student_p).any()),
                    "student_has_inf": bool(np.isinf(student_p).any()),
                    "full_vocab_size": int(victim_full.shape[0]),
                    "vocabulary_alignment_passed": bool(vocab_ok),
                    "per_example_kl": None if not np.isfinite(kl) else float(kl),
                    "kl_skipped_reason": "" if np.isfinite(kl) else reason,
                }
                debug_rows_by_key.setdefault((interface, seed), []).append(dbg)

    for seed in common_seeds:
        for budget in budget_sweep:
            for interface in interfaces_by_budget:
                print(f"[qwen] running budget sweep seed={seed} budget={budget} interface={interface}")
                run_block(budget, seed, interface)

    for seed in common_seeds:
        for interface in topk_interfaces:
            print(f"[qwen] running topk sweep seed={seed} budget=256 interface={interface}")
            run_block(256, seed, interface)

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    pd.DataFrame(nan_rows).to_csv("results/qwen_nan_report.csv", index=False)
    pd.DataFrame(vocab_rows).to_csv("results/qwen_vocab_alignment_check.csv", index=False)

    if debug_llm:
        for (interface, seed), dbg_rows in debug_rows_by_key.items():
            jpath = Path(f"results/debug/qwen_debug_{interface}_seed{seed}.jsonl")
            cpath = Path(f"results/debug/qwen_debug_{interface}_seed{seed}.csv")
            write_jsonl(jpath, dbg_rows)
            pd.DataFrame(dbg_rows).to_csv(cpath, index=False)

    invalid_rate = float((~np.isfinite(df["kl_divergence"])).mean())
    if invalid_rate > 0.01:
        raise RuntimeError(f"Qwen KL NaN/invalid rate too high: {invalid_rate:.4f} > 0.01")

    print(f"[qwen] wrote {len(df)} rows -> {out_csv}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="prompts.yaml")
    ap.add_argument("--out", default="results/qwen_raw.csv")
    ap.add_argument("--debug-llm", action="store_true")
    args = ap.parse_args()
    run_qwen(prompts_path=args.prompts, out_csv=args.out, debug_llm=args.debug_llm)
