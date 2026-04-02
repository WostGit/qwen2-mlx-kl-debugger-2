"""Single entrypoint for Day 1 experiments."""

from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"[run_all] running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--debug-llm", action="store_true", help="Write verbose Qwen debug artifacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run([sys.executable, "scripts/run_toy.py"])
    qwen_cmd = [sys.executable, "scripts/run_qwen.py"]
    if args.debug_llm:
        qwen_cmd.append("--debug-llm")
    run(qwen_cmd)
    run([sys.executable, "scripts/generate_summary.py"])
    run([sys.executable, "scripts/make_plots.py"])


if __name__ == "__main__":
    main()
