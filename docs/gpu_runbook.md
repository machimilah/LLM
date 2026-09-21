# GPU runbook (Stage 0 and Stage 1)

Everything in this repository runs on CPU or GPU unchanged: training scripts take `--device auto|cpu|cuda`,
and `peak_gpu_memory_mb` is recorded automatically when CUDA is present. This file is the order to run
things in on a rented machine, with the gates that decide whether to continue.

## Before renting

```bash
pip install -e ".[dev]"
python -m pytest -q                      # 21 tests, ~10 s
python experiments/synthetic/end_to_end.py --config ced_baseline_gen.yaml --max-steps 5 --device cuda
```
If that last command prints `device: cuda` and finishes, the environment is good.

## Stage 0 - fix the compiler (1 GPU, hours, ~$20-50)

The blocker: open-vocabulary note extraction is 51-66% on entity strings never seen in training
(results/tables/span_compiler.md), against 88-92% for the closed-vocabulary version.

```bash
# 1. Fine-tune the encoder instead of freezing MiniLM (the frozen features are the suspected limit)
python experiments/real/span_compiler.py --train-docs 8000 --epochs 4 \
    --finetune-encoder --encoder-lr 2e-5 --tag ft --limit 200

# 2. If that is not enough, swap the backbone for a stronger encoder (deberta-v3-base, e5-base)
#    by changing ENCODER_ID; the heads and data pipeline are unchanged.
```

**Gate:** >= 95% retention on held-out entity strings for qa1, qa2 and qa9, with compression under 15%.
If this fails, stop: every downstream stage depends on reliable compilation.

## Stage 0b - the 7B comparison the laptop could not run

```bash
python experiments/real/babilong_pilot.py --model Qwen/Qwen2.5-7B-Instruct \
    --tasks qa1 qa2 qa9 --length 1k --limit 100 --conditions full_context \
    --device cuda        # baseline gate first
```
**Gate:** full-context accuracy >= 70-80%. Below that the QA model cannot support a memory comparison
(Rule 2) - this is exactly why the 0.5B/1.5B runs were stopped.

Then run all five arms (`full_context rag_topk summary snm_notes snm_trained`) and record accuracy,
prompt tokens, KV bytes, latency and cost per successful task.

## Stage 1 - native SN-CED at scale (8-32 GPUs, days)

```bash
python experiments/synthetic/end_to_end.py --config ced_baseline_gen.yaml --seeds 1 2 3 --device cuda
python experiments/synthetic/snced_compare.py --seeds 1 2 3 --tag mixed --device cuda
python experiments/synthetic/compare_report.py --experiment snced_compare_mixed --out snced_vs_ced_mixed
python experiments/synthetic/router_calibration.py --seeds 1 2 3
python experiments/synthetic/adversarial_eval.py --seeds 1 2 3
```

Then the parts that need real data and real length, which are NOT yet implemented:

1. Replace the synthetic corpus with real text and a real tokenizer.
2. Train the compiler with the sufficiency loss (plan section 10) so notes need no gold labels:
   minimise `KL(P(y | full context) || P(y | notes)) + lambda * note tokens`.
3. Add the training distribution the adversarial suite exposed as missing: restated facts
   (temporal updates) and multi-token answers.
4. Context-length sweep 8K / 32K / 128K with measured HBM, prefill and decode latency.

**Gate (plan section 25):** 95-99% of baseline quality, >= 50% less active memory, no total-compute
increase, on more than one benchmark, before considering anything larger than 7B.

## Cost estimates

| Stage | Hardware | Wall time | Rented cost |
|---|---|---|---|
| 0 - compiler fix | 1x A100/H100 | 2-6 h | $20-50 |
| 0b - 7B comparison | 1x H100 (80GB) | 4-8 h | $30-80 |
| 1 - native 300M-1.5B | 8-32 GPUs | 2-5 days | $2-10k |
| 2 - native 7B, 1T tokens | 64 GPUs | ~2.5 weeks | $75-150k |
