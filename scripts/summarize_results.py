from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def summarize(raw_csv: Path, out_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(raw_csv)

    if "kl_divergence" not in df.columns:
        raise ValueError("raw results missing kl_divergence column")

    grp = ["source", "interface", "budget"]
    summary = (
        df.groupby(grp, dropna=False)
        .agg(
            mean_agreement=("agreement", "mean"),
            std_agreement=("agreement", "std"),
            mean_kl=("kl_divergence", "mean"),
            std_kl=("kl_divergence", "std"),
            valid_kl_rows=("valid_kl", "sum"),
            total_rows=("valid_kl", "count"),
        )
        .reset_index()
    )
    summary["invalid_kl_rows"] = summary["total_rows"] - summary["valid_kl_rows"]

    # Strict anti-silent-fix guard.
    raw_nan = df["kl_divergence"].isna().sum()
    if raw_nan > 0:
        suspect_zero = (summary["invalid_kl_rows"] > 0) & (summary["mean_kl"] == 0)
        if suspect_zero.any():
            bad = summary[suspect_zero]
            raise RuntimeError(f"Summary appears to replace missing KL with zero:\n{bad}")

    qwen = df[df["source"] == "qwen"]
    qwen_nan_rate = float(qwen["kl_divergence"].isna().mean()) if len(qwen) else 0.0
    if qwen_nan_rate > 0.01:
        raise RuntimeError(f"Qwen KL NaN rate too high at summary stage: {qwen_nan_rate:.4f}")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_csv, index=False)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", type=Path, default=Path("results/results_raw.csv"))
    ap.add_argument("--out-csv", type=Path, default=Path("results/results_summary.csv"))
    args = ap.parse_args()
    summarize(raw_csv=args.raw_csv, out_csv=args.out_csv)


if __name__ == "__main__":
    main()
