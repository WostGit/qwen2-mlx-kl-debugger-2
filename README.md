# qwen2-mlx-kl-debugger

A self-contained research repository for **black-box model extraction diagnostics** on GitHub Actions macOS runners, using only Python + MLX + `mlx-lm` for the real LLM experiment.

This repo is intentionally verbose: it stores row-level raw outputs and explicit invalidity reasons so KL bugs are visible instead of being hidden by summaries.

## Repository layout

- `models/`: victim adapters and student estimators.
- `attacks/`: metric implementations (`agreement`, strict KL checks).
- `experiments/`: toy and Qwen Day-1 experiment families.
- `scripts/`: stage-by-stage workflow scripts.
- `results/`: generated CSV/JSONL/PNG outputs.
- `.github/workflows/`: macOS workflow.
- `run_all.py`: one command entrypoint.

## Two experiment families (exactly)

1. **Toy baseline (`source=toy`)**
   - Victim always emits dense valid probabilities.
   - Interface views: `argmax`, `top5`, `probs`.
   - Student learns dense distribution; metrics are always well-defined.

2. **Qwen2-0.5B one-step next-token (`source=qwen`)**
   - Victim is `Qwen/Qwen2-0.5B-Instruct` loaded by `mlx-lm`.
   - For each prompt, code reads full next-token logits from MLX and computes full dense softmax.
   - Interface targets are derived from that same dense vector:
     - `argmax`: one-hot on top token.
     - `topk`: keep top-k probabilities and renormalize (used as `top2`, `top3`, `top5`).
     - `probs`: unchanged full softmax.
   - Student always predicts full dense distribution over the same vocabulary.
   - Evaluation always compares student full distribution vs victim full softmax on same token support.

## Interface definitions

- `argmax`: returns only the most likely token identity (represented as one-hot vector in training).
- `topk`: returns top-k tokens with probabilities, renormalized to sum to 1 over retained support.
- `probs`: returns full dense probability vector over full vocabulary.

## Student training (plain language)

- Toy: student is a smoothed frequency estimator over observed interface vectors.
- Qwen: student is a tiny linear dense estimator trained on prompt token-id features and interface-derived targets.
- In both cases, student output is always a full dense probability distribution.

## Metrics

- `agreement`: 1 if `argmax(student) == argmax(victim_full_softmax)`, else 0.
- `kl_divergence`: `KL(victim_full_softmax || student_full_distribution)`.

KL is never computed from labels only, never on mismatched vector shapes, and never on non-normalized vectors.

## Invalid Qwen rows

A Qwen metric row becomes invalid if any of the following holds:

- NaN or inf exists in victim/student vectors.
- Probability vectors do not sum to ~1.
- Vector shape mismatch.
- Vocabulary alignment check fails.

Invalid rows keep `kl_divergence=NaN`, `kl_valid=False`, and a human-readable `kl_reason`.

## Debugging outputs to inspect first

- `results/results_raw.csv`: all row-level values (including NaN KL).
- `results/results_summary.csv`: summary with valid/invalid KL counts.
- `results/qwen_nan_report.csv`: every invalid Qwen KL row and reason.
- `results/qwen_vocab_alignment_check.csv`: explicit vocab-alignment checks before KL.
- `results/debug/*.jsonl` and `results/debug/*.csv` (with `--debug-llm`): per-example diagnostics for first 10 eval prompts per interface/seed.

Debug rows include prompt text/ids, victim and student top-10 ids/tokens/probs, prob sums, NaN/inf flags, vocabulary size, alignment pass/fail, per-example KL, and skip reason.

## Day 1 plan implemented

- Budget sweep for **64, 128, 256, 512, 1024** on both toy and Qwen.
- Qwen fixed-budget interface sweep at budget=256 over: `argmax`, `top2`, `top3`, `top5`, `probs`.
- Multi-seed summary (mean/std over seeds).
- Plots:
  - `agreement_vs_budget.png`
  - `kl_vs_budget.png`
  - `qwen_topk_kl.png`

## Strict summary behavior

- `results_raw.csv` preserves raw NaN KL values.
- `results_summary.csv` reports valid and invalid KL row counts.
- Pipeline fails if:
  - More than 1% of Qwen KL values are NaN.
  - Any invalid Qwen KL appears zero-filled.

## Run locally on macOS (MLX)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python run_all.py --debug-llm
```

## GitHub Actions

Workflow is macOS-only and performs:

1. Install deps from `requirements.txt`.
2. Cache pip and HF/MLX downloads.
3. Run toy experiment.
4. Run Qwen experiment.
5. Build summary + plots.
6. Upload CSV/JSONL/PNG artifacts.

## Dependencies

Only:
- `numpy`
- `pandas`
- `matplotlib`
- `pyyaml`
- `mlx`
- `mlx-lm`
- `huggingface_hub`

`transformers` is not used.
