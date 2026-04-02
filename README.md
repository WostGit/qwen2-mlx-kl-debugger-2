# qwen2-mlx-kl-debugger

A self-contained research repository for black-box model extraction experiments on **GitHub Actions macOS runners** using **Python + MLX + mlx-lm** with **Qwen2-0.5B** as the real tiny-LLM victim.

## Repository layout

- `models/`: Victim and student implementations.
- `attacks/`: Interface target conversion logic.
- `experiments/`: Toy and Qwen experiment drivers + strict summary checks.
- `scripts/`: Stage-specific runnable scripts.
- `results/`: CSV, JSONL, and plot outputs.
- `.github/workflows/`: macOS CI pipeline.
- `run_all.py`: single top-level entrypoint.

## Exact experiment families

This repo contains exactly two experiment families:

1. **Toy baseline**
   - Victim emits dense probability vectors that always sum to 1.
   - Interface views:
     - `argmax`: one-hot at top token.
     - `topk` (`top2`, `top3`, `top5`): keep top-k mass and renormalize.
     - `probs`: full dense probability unchanged.
   - Student predicts a full dense distribution.
   - Metrics are always valid: `agreement` and `kl_divergence`.

2. **Qwen2-0.5B one-step next-token**
   - MLX victim computes full next-token logits for each prompt.
   - Logits are converted to **full-vocabulary dense softmax**.
   - Interface targets are derived from that same dense vector.
   - Student always predicts full dense distribution over exactly the same vocabulary.
   - Evaluation always compares against victim full softmax on identical support.

## Day-1 plan implemented

- Budget sweep for both families: `64, 128, 256, 512, 1024`.
- Interfaces: `argmax`, `top2`, `top3`, `top5`, `probs`.
- Multi-seed run: seeds `0,1,2` with mean/std summary.
- Required logging columns in raw rows:
  - `source, interface, budget, seed, agreement, kl_divergence, is_valid_kl, invalid_reason`

## Strict metric rules

- KL is computed only as `KL(victim_full_softmax || student_full_distribution)`.
- No KL from label-only targets.
- No mismatched vocabulary KL.
- No silent drop/repair/zero-fill of invalid rows.
- `results_raw.csv` preserves NaNs.
- `results_summary.csv` reports valid/invalid KL counts.
- Pipeline fails when:
  - Qwen invalid KL ratio is > 1%.
  - Summary appears to replace missing KL with `0`.

## Debugging artifacts (`--debug-llm`)

For the first 10 eval prompts per interface per seed (per budget), writes JSONL + CSV with:

- prompt id/text
- tokenized prompt ids
- victim top-10 token ids/strings/probabilities
- student top-10 token ids/strings/probabilities
- full-probability sums for victim and student
- NaN/Inf flags
- full vocab size
- vocabulary alignment pass/fail
- exact per-example KL (if valid)
- human-readable skip reason (if invalid)

Additional mandatory diagnostics:

- `results/qwen_nan_report.csv`: every invalid Qwen KL row + reason.
- `results/qwen_vocab_alignment_check.csv`: shape/index alignment checks before KL.

## How interfaces map from full victim softmax

- `argmax`: one-hot vector at highest-probability token.
- `topk`: sparse top-k vector, then renormalized to sum to 1.
- `probs`: unchanged full softmax.

## How the student is trained

A deterministic prompt-table student is used:

- During training it stores interface-derived target distributions per prompt.
- At inference it returns stored distribution for known prompt, otherwise global mean.
- Output is always full dense probability over victim vocabulary.

## How metrics are computed

- `agreement`: `argmax(victim_full_softmax) == argmax(student_distribution)`.
- `kl_divergence`: `sum_i p_i * (log p_i - log q_i)` with tiny clipping epsilon for numerical safety.

## Invalid Qwen row conditions

A Qwen row is invalid if any of the following occurs:

- Victim/student vocabulary shape mismatch.
- Any NaN in victim or student distribution.
- Any Inf in victim or student distribution.
- KL computation exception.
- KL non-finite output.

## Local macOS rerun (same workflow stages)

```bash
python -m pip install -r requirements.txt
python scripts/run_toy.py
python scripts/run_qwen.py --debug-llm
python scripts/run_summary.py
# OR single command:
python run_all.py --debug-llm
```

## Dependency policy

Only these dependencies are used:

- `numpy`, `pandas`, `matplotlib`, `pyyaml`, `mlx`, `mlx-lm`, `huggingface_hub`

`transformers` is intentionally not used.
