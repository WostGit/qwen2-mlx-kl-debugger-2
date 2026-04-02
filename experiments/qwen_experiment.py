"""Qwen next-token one-step extraction experiments with explicit debug artifacts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from attacks.metrics import agreement, safe_kl
from experiments.prompts import QWEN_PROMPTS
from models.common import check_vector_health, topn
from models.qwen_victim_mlx import QwenVictimMLX, derive_interface_target
from models.student_estimators import PromptLinearStudent


@dataclass
class QwenConfig:
    budgets: list[int]
    seeds: list[int]
    budget_interfaces: list[str]
    topk_sweep_interfaces: list[str]
    topk_sweep_budget: int
    eval_examples: int = 20
    debug_llm: bool = False


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _debug_record(prompt_id: str, prompt_text: str, prompt_ids: list[int], victim_probs: np.ndarray, student_probs: np.ndarray, vocab_alignment_passed: bool, kl: float, kl_valid: bool, kl_reason: str, victim: QwenVictimMLX) -> dict:
    v_ids, v_probs = topn(victim_probs, n=10)
    s_ids, s_probs = topn(student_probs, n=10)
    hv = check_vector_health(victim_probs)
    hs = check_vector_health(student_probs)
    return {
        "prompt_id": prompt_id,
        "raw_prompt_text": prompt_text,
        "tokenized_prompt_ids": prompt_ids,
        "victim_top10_token_ids": v_ids.tolist(),
        "victim_top10_token_strings": victim.decode(v_ids.tolist()),
        "victim_top10_probabilities": [float(x) for x in v_probs],
        "student_top10_token_ids": s_ids.tolist(),
        "student_top10_token_strings": victim.decode(s_ids.tolist()),
        "student_top10_probabilities": [float(x) for x in s_probs],
        "victim_prob_sum": hv.prob_sum,
        "student_prob_sum": hs.prob_sum,
        "victim_has_nan": hv.has_nan,
        "student_has_nan": hs.has_nan,
        "victim_has_inf": hv.has_inf,
        "student_has_inf": hs.has_inf,
        "full_vocabulary_size": int(victim_probs.shape[0]),
        "vocabulary_alignment_passed": vocab_alignment_passed,
        "per_example_kl": float(kl) if kl_valid else None,
        "kl_skipped_reason": None if kl_valid else kl_reason,
    }


def run_qwen_experiments(cfg: QwenConfig, results_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    nan_rows = []
    vocab_rows = []
    victim = QwenVictimMLX()

    def run_family(budget: int, interface: str, seed: int, family: str) -> None:
        train_prompts = QWEN_PROMPTS[: min(budget, len(QWEN_PROMPTS))]
        eval_prompts = QWEN_PROMPTS[-cfg.eval_examples :]

        prompt_ids_batch = []
        targets = []
        for prompt in train_prompts:
            info = victim.dense_next_token_probs(prompt)
            target = derive_interface_target(info["probs"], interface=interface)
            prompt_ids_batch.append(info["prompt_ids"])
            targets.append(target)

        student = PromptLinearStudent(vocab_size=victim.vocab_size, seed=seed)
        student.fit(prompt_ids_batch, targets)

        debug_rows = []
        for i, prompt in enumerate(eval_prompts):
            info = victim.dense_next_token_probs(prompt)
            vp = info["probs"]
            sp = student.predict_dense(info["prompt_ids"])
            vocab_ok = vp.shape == sp.shape == (victim.vocab_size,)
            vocab_rows.append(
                {
                    "source": "qwen",
                    "family": family,
                    "interface": interface,
                    "budget": budget,
                    "seed": seed,
                    "prompt_id": f"qwen_eval_{i}",
                    "victim_vocab_size": int(vp.shape[0]),
                    "student_vocab_size": int(sp.shape[0]),
                    "index_mapping_assumption": "shared tokenizer ids from same victim tokenizer",
                    "vocab_alignment_passed": vocab_ok,
                }
            )
            kl, kl_valid, kl_reason = safe_kl(vp, sp)
            row = {
                "source": "qwen",
                "family": family,
                "prompt_id": f"qwen_eval_{i}",
                "prompt_text": prompt,
                "interface": interface,
                "budget": budget,
                "seed": seed,
                "agreement": agreement(vp, sp),
                "kl_divergence": kl,
                "kl_valid": kl_valid and vocab_ok,
                "kl_reason": kl_reason if kl_valid and vocab_ok else ("vocab_alignment_failed" if not vocab_ok else kl_reason),
                "vocab_alignment_passed": vocab_ok,
            }
            rows.append(row)
            if not row["kl_valid"]:
                nan_rows.append({**row, "failure_reason": row["kl_reason"]})
            if cfg.debug_llm and i < 10:
                debug_rows.append(
                    _debug_record(
                        prompt_id=row["prompt_id"],
                        prompt_text=prompt,
                        prompt_ids=info["prompt_ids"],
                        victim_probs=vp,
                        student_probs=sp,
                        vocab_alignment_passed=vocab_ok,
                        kl=kl,
                        kl_valid=row["kl_valid"],
                        kl_reason=row["kl_reason"],
                        victim=victim,
                    )
                )

        if cfg.debug_llm:
            tag = f"{family}_{interface}_budget{budget}_seed{seed}"
            _write_jsonl(results_dir / "debug" / f"{tag}.jsonl", debug_rows)
            pd.DataFrame(debug_rows).to_csv(results_dir / "debug" / f"{tag}.csv", index=False)

    for seed in cfg.seeds:
        for budget in cfg.budgets:
            for interface in cfg.budget_interfaces:
                run_family(budget=budget, interface=interface, seed=seed, family="budget_sweep")

    for seed in cfg.seeds:
        for interface in cfg.topk_sweep_interfaces:
            run_family(
                budget=cfg.topk_sweep_budget,
                interface=interface,
                seed=seed,
                family="topk_sweep",
            )

    return pd.DataFrame(rows), pd.DataFrame(nan_rows), pd.DataFrame(vocab_rows)
