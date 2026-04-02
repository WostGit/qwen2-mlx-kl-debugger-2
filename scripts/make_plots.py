from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    outdir = Path("results")
    summary = pd.read_csv(outdir / "results_summary.csv")

    budget = summary[summary["family"].fillna("budget_sweep").eq("budget_sweep")]

    plt.figure(figsize=(8, 5))
    for (source, interface), grp in budget.groupby(["source", "interface"]):
        plt.plot(grp["budget"], grp["agreement_mean"], marker="o", label=f"{source}:{interface}")
    plt.xscale("log", base=2)
    plt.xlabel("Budget")
    plt.ylabel("Agreement")
    plt.title("Agreement vs Budget")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "agreement_vs_budget.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for (source, interface), grp in budget.groupby(["source", "interface"]):
        plt.plot(grp["budget"], grp["kl_mean"], marker="o", label=f"{source}:{interface}")
    plt.xscale("log", base=2)
    plt.xlabel("Budget")
    plt.ylabel("KL(victim||student)")
    plt.title("KL vs Budget")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(outdir / "kl_vs_budget.png", dpi=150)
    plt.close()

    topk = summary[(summary["source"] == "qwen") & (summary["family"] == "topk_sweep")]
    plt.figure(figsize=(8, 5))
    order = ["argmax", "top2", "top3", "top5", "probs"]
    topk = topk.set_index("interface").reindex(order).reset_index()
    plt.plot(topk["interface"], topk["kl_mean"], marker="o")
    plt.xlabel("Interface")
    plt.ylabel("KL(victim||student)")
    plt.title("Qwen Top-k Interface Sweep KL")
    plt.tight_layout()
    plt.savefig(outdir / "qwen_topk_kl.png", dpi=150)
    plt.close()

    print("[plots] wrote agreement_vs_budget.png, kl_vs_budget.png, qwen_topk_kl.png")


if __name__ == "__main__":
    main()
