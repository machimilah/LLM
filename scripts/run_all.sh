#!/usr/bin/env bash
# Reproduce every CPU-runnable result in order. Hours on 12 CPU threads.
set -euo pipefail
cd "$(dirname "$0")"
python -m pytest -q

# P1: historical experiments
python experiments/synthetic/learned_compiler.py
python experiments/synthetic/ablation.py
python experiments/synthetic/report.py

# P2: CED baseline (must be healthy before anything is compared to it)
python experiments/synthetic/end_to_end.py --config ced_baseline_gen.yaml --seeds 1 2 3

# Decisive comparison + mixed-length compiler
python experiments/synthetic/snced_compare.py --seeds 1 2 3 --tag mixed
python experiments/synthetic/compare_report.py --experiment snced_compare_mixed --out snced_vs_ced_mixed

# Memory-representation ablations, router calibration, adversarial suite
python experiments/synthetic/ablations.py --seeds 1 2 3
python experiments/synthetic/ablations_report.py
python experiments/synthetic/router_calibration.py --seeds 1 2 3
python experiments/synthetic/adversarial_eval.py --seeds 1 2 3

# Real text (downloads models and datasets)
python experiments/real/train_note_compiler.py --train-docs 3000 --epochs 6
python experiments/real/span_compiler.py --train-docs 3000 --epochs 4
