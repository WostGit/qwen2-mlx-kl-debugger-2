from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from experiments.qwen_experiment import run_qwen
from experiments.toy_experiment import run_toy
from scripts.plot_results import plot
from scripts.summarize_results import summarize


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budgets", type=str, default="64,128,256,512,1024")
    ap.add_argument("--seeds", type=str, default="0,1")
    ap.add_argument("--debug-llm", action="store_true")
    ap.add_argument("--results-dir", type=Path, default=Path("results"))
    args = ap.parse_args()

    budgets = [int(x) for x in args.budgets.split(",")]
    seeds = [int(x) for x in args.seeds.split(",")]

    args.results_dir.mkdir(parents=True, exist_ok=True)

    toy_raw = args.results_dir / "toy_results_raw.csv"
    qwen_raw = args.results_dir / "qwen_results_raw.csv"
    merged_raw = args.results_dir / "results_raw.csv"

    run_toy(budgets=budgets, seeds=seeds, out_csv=toy_raw)
    run_qwen(
        budgets=budgets,
        seeds=seeds,
        debug_llm=args.debug_llm,
        out_csv=qwen_raw,
        debug_dir=args.results_dir / "debug",
        nan_report_csv=args.results_dir / "qwen_nan_report.csv",
        vocab_check_csv=args.results_dir / "qwen_vocab_alignment_check.csv",
    )

    df = pd.concat([pd.read_csv(toy_raw), pd.read_csv(qwen_raw)], ignore_index=True)
    df.to_csv(merged_raw, index=False)

    summarize(raw_csv=merged_raw, out_csv=args.results_dir / "results_summary.csv")
    plot(raw_csv=merged_raw, out_dir=args.results_dir)


if __name__ == "__main__":
    main()
