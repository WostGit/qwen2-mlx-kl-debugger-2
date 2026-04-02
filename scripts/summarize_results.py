from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
import numpy as np
import pandas as pd


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    grouped = []
    for keys, g in df.groupby(["source", "interface", "budget", "seed"], dropna=False):
        valid_kl = g["kl_divergence"].dropna()
        grouped.append(
            {
                "source": keys[0],
                "interface": keys[1],
                "budget": keys[2],
                "seed": keys[3],
                "n_rows": len(g),
                "n_valid_kl": int(g["kl_divergence"].notna().sum()),
                "n_invalid_kl": int(g["kl_divergence"].isna().sum()),
                "agreement_mean": float(g["agreement"].mean()),
                "agreement_std": float(g["agreement"].std(ddof=0)),
                "kl_mean": float(valid_kl.mean()) if len(valid_kl) else np.nan,
                "kl_std": float(valid_kl.std(ddof=0)) if len(valid_kl) else np.nan,
            }
        )
    return pd.DataFrame(grouped)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--toy", default="results/toy_raw.csv")
    ap.add_argument("--qwen", default="results/qwen_raw.csv")
    ap.add_argument("--raw-out", default="results/results_raw.csv")
    ap.add_argument("--summary-out", default="results/results_summary.csv")
    args = ap.parse_args()

    toy = pd.read_csv(args.toy)
    qwen = pd.read_csv(args.qwen)
    raw = pd.concat([toy, qwen], ignore_index=True)

    if "kl_divergence" in raw and raw["kl_divergence"].fillna(0).equals(raw["kl_divergence"]):
        raise RuntimeError("Detected forbidden replacement of missing KL with zero.")

    raw.to_csv(args.raw_out, index=False)
    summary = summarize(raw)
    summary.to_csv(args.summary_out, index=False)

    qwen_invalid_rate = float((raw[raw["source"] == "qwen"]["kl_divergence"].isna()).mean())
    if qwen_invalid_rate > 0.01:
        raise RuntimeError(f"Qwen KL invalid rate {qwen_invalid_rate:.4f} exceeds 1% threshold")

    print(f"wrote {args.raw_out} and {args.summary_out}")
