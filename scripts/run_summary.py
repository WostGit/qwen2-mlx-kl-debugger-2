from pathlib import Path

import pandas as pd

from experiments.summary import build_summary


def main() -> None:
    toy = pd.read_csv("results/toy_results_raw.csv")
    qwen = pd.read_csv("results/qwen_results_raw.csv")
    raw = pd.concat([toy, qwen], ignore_index=True)
    build_summary(raw, Path("results"))
    print("Summary complete")


if __name__ == "__main__":
    main()
