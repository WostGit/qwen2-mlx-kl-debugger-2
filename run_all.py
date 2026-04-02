import argparse
import subprocess
import sys


def run(cmd):
    print(f"[run_all] executing: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--debug-llm", action="store_true")
    args = ap.parse_args()

    run([sys.executable, "scripts/run_toy.py"])
    qcmd = [sys.executable, "scripts/run_qwen.py"]
    if args.debug_llm:
        qcmd.append("--debug-llm")
    run(qcmd)
    run([sys.executable, "scripts/summarize_results.py"])
    run([sys.executable, "scripts/plot_results.py"])
