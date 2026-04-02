# qwen2-mlx-kl-debugger

A fully self-contained, explicit, and verbose research repo for black-box model extraction experiments on GitHub Actions **macOS** runners using **Python + MLX + mlx-lm**, with **Qwen2-0.5B** as the tiny-LLM victim.

## Repository layout

- `models/`: victim model implementations (toy victim).
- `attacks/`: student estimators.
- `experiments/`: experiment logic.
- `scripts/`: workflow stage scripts.
- `results/`: generated CSV/JSONL/PNG outputs.
- `.github/workflows/`: macOS GitHub Actions workflow.
- `run_all.py`: single top-level entrypoint.

## Exact experiment families (only two)

1. **Toy baseline (always dense valid probabilities)**
   - Victim always emits a dense next-class probability vector.
   - Interface views:
     - `argmax`: one-hot on highest-probability class.
     - `topk` (toy uses `top3`): keep top-k probabilities and renormalize.
     - `probs`: full dense probability vector.
   - Student is fit on interface targets and always outputs a dense distribution.
   - Metrics are always computed on full same-support vectors:
     - `agreement`: top-1 token/class match.
     - `kl_divergence`: KL(victim_full || student_full).

2. **Qwen2-0.5B next-token (one-step only)**
   - Victim: `Qwen/Qwen2-0.5B-Instruct` via `mlx_lm`.
   - For each prompt, code obtains full next-token logits from MLX, converts to a **full dense softmax**.
   - Interface targets are derived **from that same full victim vector**:
     - `argmax`: one-hot top token.
     - `topk`: keep top-k victim probabilities and renormalize.
     - `probs`: unchanged full victim softmax.
   - Student estimator still outputs a **full dense distribution over same vocabulary**.
   - Evaluation compares student full distribution vs victim full softmax on exactly matching support.

## Day 1 plan implemented

- Budget sweep: `64, 128, 256, 512, 1024` for toy and Qwen.
- Logs columns: `source, interface, budget, seed, agreement, kl_divergence, validity flags`.
- Fixed-budget Qwen top-k sweep at budget `256` with: `argmax, top2, top3, top5, probs`.
- Small multi-seed summary with mean/std.
- Plots produced:
  - `results/agreement_vs_budget.png`
  - `results/kl_vs_budget.png`
  - `results/qwen_topk_kl.png`

## Strict metric behavior

- Never computes KL from labels only.
- Never compares vectors with mismatched vocab sizes.
- Never silently fills missing KL with zero.
- `results/results_raw.csv` preserves raw values including NaN.
- `results/results_summary.csv` includes valid/invalid KL counts.
- Run fails if:
  - Qwen KL invalid rate exceeds 1%.
  - summary stage detects missing KL replaced by zero.

## Debug artifacts (`--debug-llm`)

For at least first 10 evaluation prompts per interface and seed, outputs JSONL + CSV in `results/debug/` including:

- prompt id and raw prompt text
- tokenized prompt ids
- victim top-10 ids/strings/probabilities
- student top-10 ids/strings/probabilities
- full vector probability sums
- NaN/inf flags
- full vocabulary size
- vocabulary alignment check
- per-example KL (when valid)
- skip reason (when KL invalid)

Additional debug reports:

- `results/qwen_nan_report.csv`: every invalid Qwen KL row + reason
- `results/qwen_vocab_alignment_check.csv`: explicit vocab/index alignment checks before KL

## Dependencies

Only these are used in `requirements.txt`:
- `numpy`, `pandas`, `matplotlib`, `pyyaml`, `mlx`, `mlx-lm`, `huggingface_hub`

No direct `transformers` dependency is added.

## Local run on macOS (same flow)

```bash
python -m pip install -r requirements.txt
python run_all.py --debug-llm
```

Stage-by-stage:

```bash
python scripts/run_toy.py
python scripts/run_qwen.py --debug-llm
python scripts/summarize_results.py
python scripts/plot_results.py
```

## GitHub Actions flow

Workflow: `.github/workflows/macos_mlx.yml`

1. install from single `requirements.txt`
2. run toy first
3. run Qwen second
4. run summary + plots
5. upload all CSV/JSONL/PNG artifacts
