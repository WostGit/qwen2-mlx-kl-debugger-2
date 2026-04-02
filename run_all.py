from __future__ import annotations

import argparse
import subprocess
import sys


def run(cmd: list[str]) -> None:
    print(f"[RUN] {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run toy + qwen + summary pipeline")
    parser.add_argument("--debug-llm", action="store_true", help="write per-example debug CSV/JSONL for first 10 eval prompts")
    args = parser.parse_args()

    run([sys.executable, "scripts/run_toy.py"])
    qcmd = [sys.executable, "scripts/run_qwen.py"]
    if args.debug_llm:
        qcmd.append("--debug-llm")
    run(qcmd)
    run([sys.executable, "scripts/run_summary.py"])


if __name__ == "__main__":
    main()
