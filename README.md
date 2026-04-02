# qwen2-mlx-kl-debugger

A self-contained research repo for **black-box model extraction** experiments on macOS GitHub Actions, focused on making KL/metric bugs obvious.

## Goal

This repository has exactly two experiment families:

1. **Toy baseline** (`source=toy`)
   - Uses a deterministic toy victim that always emits dense, normalized next-class probabilities.
   - Always has aligned vocab support and valid KL in normal conditions.

2. **Qwen2-0.5B MLX next-token experiment** (`source=qwen`)
   - Uses `Qwen/Qwen2-0.5B-Instruct` through `mlx` + `mlx-lm`.
   - Only does one-step next-token prediction on a fixed prompt set.
   - Always gets full next-token logits, converts to full softmax, then derives interface-specific targets from that same dense distribution.

## Repository layout

- `models/`: victim model wrappers (`toy_victim.py`, `qwen_victim_mlx.py`)
- `attacks/`: student estimators (`student_estimators.py`)
- `experiments/`: experiment runners (`toy_experiment.py`, `qwen_experiment.py`)
- `scripts/`: stage scripts (`run_toy.py`, `run_qwen.py`, `summarize_results.py`, `plot_results.py`)
- `results/`: output CSV, JSONL, PNG artifacts
- `.github/workflows/`: macOS CI workflow
- `run_all.py`: one top-level entrypoint for the full pipeline

## Interfaces and what they return

All interfaces are derived from a dense victim vector over the full vocabulary/classes:

- `probs`: full dense softmax unchanged.
- `argmax`: one-hot at the top token/class.
- `topk` (and fixed sweep variants `top2`, `top3`, `top5`): keep only top-k probabilities, set others to 0, renormalize retained mass to 1.

## Student training

Students are dense estimators that output a full probability vector over the same support:

- Training data = black-box queries to the victim interface.
- Student stores running averages per prompt id + global fallback average.
- Predictions are dense, smoothed, and normalized before metric computation.

## Metrics

Per evaluation row:

- **agreement**: `argmax(victim_full_softmax) == argmax(student_full_distribution)`
- **kl_divergence**: `KL(victim_full_softmax || student_full_distribution)`

Hard rules enforced in code:

- Never compute KL from label-only targets.
- Never compare mismatched vocab sizes.
- Never silently zero-fill invalid rows.
- Keep raw NaNs in `results_raw.csv`.

## Invalid Qwen row conditions

A Qwen row is invalid (KL = NaN with explicit reason) if any occur:

- Missing logits / MLX forward errors.
- Victim or student has NaN/inf probabilities.
- Negative probabilities.
- Victim or student vector not normalized.
- Student contains exact zero (unsafe for KL denominator support).
- Vocab mismatch.

## Debugging artifacts

Use `--debug-llm` to emit per-example logs (first 10 eval prompts for each interface/seed/setting):

- `results/debug/*.jsonl` and `results/debug/*.csv`
- Includes prompt id/text, token IDs, top-10 ids/strings/probs for victim and student,
  full-vector sums, NaN/inf flags, vocab size/alignment, per-example KL, and skip reason.

Additional strict diagnostics:

- `results/qwen_nan_report.csv`: every invalid Qwen KL row + failure reason.
- `results/qwen_vocab_alignment_check.csv`: vocab alignment checks before KL.

## Day 1 plan implemented

- Budget sweep over `64,128,256,512,1024` for toy + Qwen.
- Qwen fixed-budget (256) top-k sweep over `argmax,top2,top3,top5,probs`.
- Multi-seed summary (mean/std).
- Required plots:
  - `results/agreement_vs_budget.png`
  - `results/kl_vs_budget.png`
  - `results/qwen_topk_kl.png`

## Strict summary behavior

- `results/results_raw.csv`: raw rows preserved (including NaN KL).
- `results/results_summary.csv`: includes valid/invalid KL counts.
- Pipeline fails if Qwen KL NaN rate > 1%.
- Pipeline fails if summary appears to replace missing KL with zero.

## Run locally (macOS + MLX)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --budgets 64,128,256,512,1024 --seeds 0,1 --debug-llm
```

Or run stages individually:

```bash
python scripts/run_toy.py --budgets 64,128,256,512,1024 --seeds 0,1,2
python scripts/run_qwen.py --budgets 64,128,256,512,1024 --seeds 0,1 --debug-llm
python scripts/summarize_results.py --raw-csv results/results_raw.csv --out-csv results/results_summary.csv
python scripts/plot_results.py --raw-csv results/results_raw.csv --out-dir results
```

## GitHub Actions

Workflow: `.github/workflows/macos_experiments.yml`

- macOS runner only
- single `requirements.txt`
- caches pip + Hugging Face model downloads
- stage order: toy -> Qwen -> summary/plots -> artifact upload

## Dependencies

Only uses:

- numpy
- pandas
- matplotlib
- pyyaml
- mlx
- mlx-lm
- huggingface_hub

No `transformers` dependency is used.
