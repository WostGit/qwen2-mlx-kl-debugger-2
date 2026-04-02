from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import numpy as np
import pandas as pd


def main() -> None:
    outdir = Path("results")
    toy = pd.read_csv(outdir / "toy_results_raw.csv")
    qwen = pd.read_csv(outdir / "qwen_results_raw.csv")
    raw = pd.concat([toy, qwen], ignore_index=True)
    raw.to_csv(outdir / "results_raw.csv", index=False)

    # Strict protections against hidden KL replacement.
    if "kl_divergence" not in raw.columns:
        raise RuntimeError("Missing kl_divergence in raw results")
    if (raw["source"].eq("qwen") & raw["kl_valid"].eq(False) & raw["kl_divergence"].eq(0)).any():
        raise RuntimeError("Detected invalid Qwen KL rows with zero value; zero-fill is forbidden")

    qwen_kl_nan_rate = float(qwen["kl_divergence"].isna().mean())
    if qwen_kl_nan_rate > 0.01:
        raise RuntimeError(f"Qwen KL NaN rate too high: {qwen_kl_nan_rate:.4%}")

    grouped = (
        raw.groupby(["source", "family", "interface", "budget", "seed"], dropna=False)
        .agg(
            agreement_mean=("agreement", "mean"),
            kl_mean=("kl_divergence", "mean"),
            valid_kl_rows=("kl_valid", "sum"),
            total_rows=("kl_valid", "count"),
        )
        .reset_index()
    )
    grouped["invalid_kl_rows"] = grouped["total_rows"] - grouped["valid_kl_rows"]

    multi_seed = (
        grouped.groupby(["source", "family", "interface", "budget"], dropna=False)
        .agg(
            agreement_mean=("agreement_mean", "mean"),
            agreement_std=("agreement_mean", "std"),
            kl_mean=("kl_mean", "mean"),
            kl_std=("kl_mean", "std"),
            valid_kl_rows=("valid_kl_rows", "sum"),
            invalid_kl_rows=("invalid_kl_rows", "sum"),
        )
        .reset_index()
    )

    grouped.to_csv(outdir / "results_summary_by_seed.csv", index=False)
    multi_seed.to_csv(outdir / "results_summary.csv", index=False)

    print("[summary] wrote results_raw.csv, results_summary_by_seed.csv, results_summary.csv")


if __name__ == "__main__":
    main()
