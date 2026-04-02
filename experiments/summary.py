from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from experiments.common import assert_summary_not_zero_filled, fail_if_too_many_invalid_qwen, summarize_results


def make_plots(raw: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    for metric, filename in [("agreement", "agreement_vs_budget.png"), ("kl_divergence", "kl_vs_budget.png")]:
        fig, ax = plt.subplots(figsize=(8, 5))
        for (source, interface), grp in raw.groupby(["source", "interface"]):
            stats = grp.groupby("budget")[metric].mean().reset_index()
            ax.plot(stats["budget"], stats[metric], marker="o", label=f"{source}:{interface}")
        ax.set_xlabel("budget")
        ax.set_ylabel(metric)
        ax.set_title(f"{metric} vs budget")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out_dir / filename)
        plt.close(fig)

    qwen_topk = raw[(raw["source"] == "qwen") & (raw["budget"] == raw["budget"].max())]
    fig, ax = plt.subplots(figsize=(7, 4))
    qstats = qwen_topk.groupby("interface")["kl_divergence"].mean().sort_values()
    ax.bar(qstats.index, qstats.values)
    ax.set_title("Qwen top-k sweep KL (fixed budget)")
    ax.set_ylabel("KL")
    fig.tight_layout()
    fig.savefig(out_dir / "qwen_topk_kl.png")
    plt.close(fig)


def build_summary(raw: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    summary = summarize_results(raw)
    assert_summary_not_zero_filled(summary)
    fail_if_too_many_invalid_qwen(raw)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw.to_csv(out_dir / "results_raw.csv", index=False)
    summary.to_csv(out_dir / "results_summary.csv", index=False)
    make_plots(raw, out_dir)
    return summary
