from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))

import argparse
from experiments.qwen_experiment import run_qwen

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-llm", action="store_true")
    args = ap.parse_args()
    run_qwen(prompts_path="prompts.yaml", out_csv="results/qwen_raw.csv", debug_llm=args.debug_llm)
