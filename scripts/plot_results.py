from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd


def main():
    raw = pd.read_csv("results/results_raw.csv")
    raw_budget = raw[raw["budget"].isin([64, 128, 256, 512, 1024])]

    plt.figure(figsize=(8, 5))
    for (source, interface), g in raw_budget.groupby(["source", "interface"]):
        means = g.groupby("budget")["agreement"].mean().reset_index()
        plt.plot(means["budget"], means["agreement"], marker="o", label=f"{source}:{interface}")
    plt.xlabel("budget")
    plt.ylabel("agreement")
    plt.title("Agreement vs Budget")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/agreement_vs_budget.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    for (source, interface), g in raw_budget.groupby(["source", "interface"]):
        means = g.groupby("budget")["kl_divergence"].mean().reset_index()
        plt.plot(means["budget"], means["kl_divergence"], marker="o", label=f"{source}:{interface}")
    plt.xlabel("budget")
    plt.ylabel("kl_divergence")
    plt.title("KL vs Budget")
    plt.legend(fontsize=7)
    plt.tight_layout()
    plt.savefig("results/kl_vs_budget.png", dpi=150)
    plt.close()

    qwen = raw[(raw["source"] == "qwen") & (raw["budget"] == 256)]
    topk_ifaces = ["argmax", "top2", "top3", "top5", "probs"]
    qwen = qwen[qwen["interface"].isin(topk_ifaces)]
    means = qwen.groupby("interface")["kl_divergence"].mean().reindex(topk_ifaces)

    plt.figure(figsize=(8, 5))
    means.plot(kind="bar")
    plt.ylabel("mean kl_divergence")
    plt.title("Qwen top-k interface KL (budget=256)")
    plt.tight_layout()
    plt.savefig("results/qwen_topk_kl.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
