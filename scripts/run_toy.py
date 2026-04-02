from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))


from experiments.toy_experiment import ToyConfig, run_toy_budget_sweep


def main() -> None:
    outdir = Path("results")
    outdir.mkdir(parents=True, exist_ok=True)
    cfg = ToyConfig(
        budgets=[64, 128, 256, 512, 1024],
        seeds=[0, 1, 2],
        interfaces=["argmax", "top5", "probs"],
    )
    df = run_toy_budget_sweep(cfg)
    df.to_csv(outdir / "toy_results_raw.csv", index=False)
    print(f"[toy] wrote {len(df)} rows to {outdir / 'toy_results_raw.csv'}")


if __name__ == "__main__":
    main()
