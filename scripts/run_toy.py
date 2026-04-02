from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

from experiments.toy_experiment import run_toy

if __name__ == "__main__":
    run_toy(
        budgets=[64, 128, 256, 512, 1024],
        seeds=[0, 1, 2],
        interfaces=["argmax", "top3", "probs"],
        out_csv="results/toy_raw.csv",
    )
