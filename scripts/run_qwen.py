import argparse
from pathlib import Path

from experiments.qwen_experiment import QwenConfig, run_qwen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-llm", action="store_true")
    args = parser.parse_args()

    cfg = QwenConfig(
        budgets=[64, 128, 256, 512, 1024],
        seeds=[0, 1, 2],
        interfaces=["argmax", "top2", "top3", "top5", "probs"],
        debug_llm=args.debug_llm,
    )
    raw, nan_report, align = run_qwen(cfg, out_dir=Path("results"))
    raw.to_csv("results/qwen_results_raw.csv", index=False)
    nan_report.to_csv("results/qwen_nan_report.csv", index=False)
    align.to_csv("results/qwen_vocab_alignment_check.csv", index=False)
    print(raw.head())


if __name__ == "__main__":
    main()
