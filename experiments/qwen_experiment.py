from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from attacks.train_student import make_interface_target
from experiments.common import ensure_dir, kl_divergence, write_jsonl
from models.qwen_victim import QwenMLXVictim
from models.student_estimator import PromptTableStudent


@dataclass
class QwenConfig:
    budgets: list[int]
    seeds: list[int]
    interfaces: list[str]
    debug_llm: bool
    prompt_file: str = "configs/prompts.yaml"


def load_prompts(path: str) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data["prompts"]


def run_qwen(config: QwenConfig, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    prompts = load_prompts(config.prompt_file)
    max_budget = max(config.budgets)
    expanded = [prompts[i % len(prompts)] for i in range(max_budget + 20)]
    eval_prompts = prompts

    victim = QwenMLXVictim()
    rows: list[dict] = []
    nan_rows: list[dict] = []
    align_rows: list[dict] = []

    for seed in config.seeds:
        train_items = expanded[:max_budget]
        cache = {}
        for item in train_items + eval_prompts:
            if item["id"] not in cache:
                cache[item["id"]] = victim.next_token_distribution(item["text"])

        vocab_size = next(iter(cache.values())).probs.shape[0]
        for budget in config.budgets:
            current_train = train_items[:budget]
            for interface in config.interfaces:
                train_targets = [
                    make_interface_target(cache[it["id"]].probs, interface) for it in current_train
                ]
                student = PromptTableStudent(vocab_size=vocab_size)
                student.fit([it["text"] for it in current_train], train_targets)
                debug_rows = []

                for idx, item in enumerate(eval_prompts):
                    out = cache[item["id"]]
                    victim_full = out.probs
                    student_full = student.predict(item["text"])
                    vocab_ok = victim_full.shape == student_full.shape
                    kl = np.nan
                    reason = ""
                    is_valid = True
                    has_nan = bool(np.isnan(victim_full).any() or np.isnan(student_full).any())
                    has_inf = bool(np.isinf(victim_full).any() or np.isinf(student_full).any())

                    if not vocab_ok:
                        is_valid = False
                        reason = "vocabulary_shape_mismatch"
                    elif has_nan:
                        is_valid = False
                        reason = "nan_detected"
                    elif has_inf:
                        is_valid = False
                        reason = "inf_detected"
                    else:
                        try:
                            kl = kl_divergence(victim_full, student_full)
                        except Exception as e:
                            is_valid = False
                            reason = f"kl_exception:{e}"

                    if not np.isfinite(kl):
                        is_valid = False
                        if not reason:
                            reason = "kl_not_finite"

                    row = {
                        "source": "qwen",
                        "interface": interface,
                        "budget": budget,
                        "seed": seed,
                        "agreement": float(np.argmax(victim_full) == np.argmax(student_full)),
                        "kl_divergence": float(kl) if np.isfinite(kl) else np.nan,
                        "is_valid_kl": is_valid,
                        "invalid_reason": reason,
                    }
                    rows.append(row)
                    align_rows.append(
                        {
                            "seed": seed,
                            "interface": interface,
                            "budget": budget,
                            "prompt_id": item["id"],
                            "victim_vocab_size": victim_full.shape[0],
                            "student_vocab_size": student_full.shape[0],
                            "vocab_alignment_passed": vocab_ok,
                        }
                    )
                    if not is_valid:
                        nan_rows.append({**row, "prompt_id": item["id"]})

                    if config.debug_llm and idx < 10:
                        vt = np.argpartition(victim_full, -10)[-10:]
                        st = np.argpartition(student_full, -10)[-10:]
                        vt = vt[np.argsort(victim_full[vt])[::-1]]
                        st = st[np.argsort(student_full[st])[::-1]]
                        debug_rows.append(
                            {
                                "prompt_id": item["id"],
                                "prompt_text": item["text"],
                                "tokenized_prompt_ids": out.token_ids,
                                "victim_top10_token_ids": vt.tolist(),
                                "victim_top10_token_strings": victim.decode_ids(vt.tolist()),
                                "victim_top10_probabilities": victim_full[vt].tolist(),
                                "student_top10_token_ids": st.tolist(),
                                "student_top10_token_strings": victim.decode_ids(st.tolist()),
                                "student_top10_probabilities": student_full[st].tolist(),
                                "victim_prob_sum": float(victim_full.sum()),
                                "student_prob_sum": float(student_full.sum()),
                                "has_nan": has_nan,
                                "has_inf": has_inf,
                                "full_vocab_size": int(vocab_size),
                                "vocabulary_alignment_passed": vocab_ok,
                                "per_example_kl": float(kl) if np.isfinite(kl) else None,
                                "kl_skipped_reason": reason,
                            }
                        )
                if config.debug_llm:
                    debug_dir = ensure_dir(out_dir / "debug")
                    stem = f"qwen_seed{seed}_{interface}_b{budget}"
                    write_jsonl(debug_dir / f"{stem}.jsonl", debug_rows)
                    pd.DataFrame(debug_rows).to_csv(debug_dir / f"{stem}.csv", index=False)

    raw_df = pd.DataFrame(rows)
    return raw_df, pd.DataFrame(nan_rows), pd.DataFrame(align_rows)
