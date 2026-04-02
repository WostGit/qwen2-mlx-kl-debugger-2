from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(raw_csv: Path, out_dir: Path) -> None:
    df = pd.read_csv(raw_csv)
    out_dir.mkdir(parents=True, exist_ok=True)

    budget_df = df[df["is_topk_sweep"] == 0] if "is_topk_sweep" in df.columns else df

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), g in budget_df.groupby(["source", "interface"]):
        m = g.groupby("budget", as_index=False)["agreement"].mean()
        ax.plot(m["budget"], m["agreement"], marker="o", label=f"{source}:{interface}")
    ax.set_title("Agreement vs Budget")
    ax.set_xlabel("Budget")
    ax.set_ylabel("Agreement")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "agreement_vs_budget.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    for (source, interface), g in budget_df.groupby(["source", "interface"]):
        m = g.groupby("budget", as_index=False)["kl_divergence"].mean()
        ax.plot(m["budget"], m["kl_divergence"], marker="o", label=f"{source}:{interface}")
    ax.set_title("KL Divergence vs Budget")
    ax.set_xlabel("Budget")
    ax.set_ylabel("KL(victim||student)")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out_dir / "kl_vs_budget.png", dpi=160)
    plt.close(fig)

    qwen_topk = df[(df["source"] == "qwen") & (df.get("is_topk_sweep", 0) == 1)]
    if len(qwen_topk):
        fig, ax = plt.subplots(figsize=(8, 5))
        m = qwen_topk.groupby("interface", as_index=False)["kl_divergence"].mean()
        ax.bar(m["interface"], m["kl_divergence"])
        ax.set_title("Qwen Fixed-Budget KL by Interface")
        ax.set_ylabel("KL(victim||student)")
        fig.tight_layout()
        fig.savefig(out_dir / "qwen_topk_kl.png", dpi=160)
        plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-csv", type=Path, default=Path("results/results_raw.csv"))
    ap.add_argument("--out-dir", type=Path, default=Path("results"))
    args = ap.parse_args()
    plot(raw_csv=args.raw_csv, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
