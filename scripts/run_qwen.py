from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from experiments.qwen_experiment import QwenConfig, run_qwen_experiments


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--debug-llm", action="store_true", help="Write verbose per-example artifacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = QwenConfig(
        budgets=[64, 128, 256, 512, 1024],
        seeds=[0, 1, 2],
        budget_interfaces=["argmax", "top5", "probs"],
        topk_sweep_interfaces=["argmax", "top2", "top3", "top5", "probs"],
        topk_sweep_budget=256,
        debug_llm=args.debug_llm,
    )
    df, nan_df, vocab_df = run_qwen_experiments(cfg, outdir)
    df.to_csv(outdir / "qwen_results_raw.csv", index=False)
    nan_df.to_csv(outdir / "qwen_nan_report.csv", index=False)
    vocab_df.to_csv(outdir / "qwen_vocab_alignment_check.csv", index=False)
    print(f"[qwen] raw rows={len(df)} nan_rows={len(nan_df)}")


if __name__ == "__main__":
    main()
