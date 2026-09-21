# SN-CED

Semantic Note Memory for Causal Encoder-Decoder LLMs. Research prototype.

The full architecture and test plan is in
[SN_CED_ARCHITECTURE_AND_TEST_PLAN.md](SN_CED_ARCHITECTURE_AND_TEST_PLAN.md).
Nothing here demonstrates superiority over CED, DeepSeek V4.1 or RAG (plan section 12).

## Layout

```text
src/snced/          library: notes, synthetic data, compiler, router, fallback, models
experiments/        runnable experiments (synthetic/ now; ruler/, babilong/, ... later)
configs/            YAML configs for every experiment
tests/              unit tests (pytest)
results/raw/        one JSON per run, schema from plan section 28 (never delete)
results/logs/       training curves and console logs
results/tables/     generated reports
```

## Running

```bash
python -m pytest                                   # unit tests
python experiments/synthetic/learned_compiler.py   # Experiment B, seeds 11 22 33
python experiments/synthetic/ablation.py           # Experiment A, seeds 123 456 789
python experiments/synthetic/report.py             # -> results/tables/reproduction.md
python experiments/synthetic/end_to_end.py --config ced_baseline_gen.yaml   # P2 CED baseline
```

All scripts are CPU-only and use repository-relative paths.

## Status

| Task | State |
|---|---|
| P0 repository preparation | done |
| P1 reproduce historical experiments | done: reimplemented from the plan (originals unavailable), all four acceptance criteria pass; see results/tables/reproduction.md |
| P2 CED baseline at >= 95% | done: all 3 seeds healthy (100% direct / one-hop / two-hop) |
| Decisive CED vs SN-CED test | done, 3 seeds. At the trained length notes-only keeps 99.42 ± 0.58% vs 100% baseline on 4.4% of the memory and 57% of the estimated FLOPs. At unseen longer contexts notes-only degrades (97.4%, 93.9%) while gold notes hold ~99.9%, so the compiler is the weak link. The confidence router is badly calibrated (38-63% fallback). See results/tables/snced_vs_ced.md |
| Memory-representation ablations | done, 3 seeds, 10 conditions. At a matched 72-token budget SNM wins by a wide margin (99.4% vs 44% RAG top-k, 20% summary, 18% sliding window). Anchor-only 12% and phrase-only 27% confirm both halves of a note are needed. Uncompressed rivals (extractive summary, iterative RAG) still match full-context quality at 5-9x more memory. See results/tables/ablations.md |
| Mixed-length compiler (3 seeds) | done: learned SNM holds 96.4-100% of full-context accuracy at 547-2705 tokens on 24 slots; compiler extraction no longer decays with length. See results/tables/snced_vs_ced_mixed.md |
| Real-model pilot (BABILong, 0.5B/1.5B) | stopped at the baseline gate: full context only 48%/32%, below the agreed 70-80%. Notes were 7.4x cheaper but lost on quality. See results/logs/FAILED_RUNS.md |
| Trained note compiler on real text | done: 0.2M trained params over a frozen MiniLM, trained with WikiText noise, evaluated on BABILong book-text noise. Retention 88/90/92% on qa1/qa2/qa9 vs 100% for gold notes, at 4-10% of the document. Beats a prompted 0.5B compiler (52 vs 36 on qa1, 32 vs 4 on qa2) and builds memory ~50x faster. See results/tables/real_text_compiler.md |
| Timing + cost per successful task | done, 3 seeds, idle machine: notes cost 40 ms vs 55 ms per successful task at 547 tokens, and 54 ms vs 111 ms at 1528 tokens, with 39-42% fewer FLOPs. See results/tables/timing.md |
| Router calibration | done: question-level routing at t=0.5 gives 96.9% on 79 slots where the old lecture-level policy needed 996 slots for 97.7%. Now the default. See results/tables/router_calibration.md |
| Adversarial suite (plan 20) | done, 3 seeds: temporal updates cost every condition (full context 84.9%, notes 78.8%); exact multi-token values are inconclusive because the baseline fails too. See results/tables/adversarial.md |
| Open-vocabulary span compiler | done, gate not met: 82/73/82% retention on the bAbI cast, 66/51/61% on held-out entity strings, vs 88/90/92% for the closed-vocabulary version. Compression 4-11%. 7B comparison not run (gate + no GPU). See results/tables/span_compiler.md |
| Stage 0 on GPU (Kaggle T4) | gate not met and now well characterised: 225k training sentences, 20k steps, fine-tuned encoder -> 85/78/84% retention (worst case 0.82 vs gate 0.95) with training loss 0.0000. The limit is generalisation, not compute. See results/tables/stage0_gpu.md |
| General architecture v2 | notes 100% = full context 100% on 7.1% of the memory, no gold note labels anywhere. L0 gates, learned router, targeted span reread. See results/tables/snced_general.md |
| P8+ real benchmarks | blocked: needs a 7B+ QA model and a GPU (the 0.5B/1.5B QA baseline is too weak) |

## Documentation

- [docs/architecture.md](docs/architecture.md) - what is built and why, with the cost of each decision
- [docs/gpu_runbook.md](docs/gpu_runbook.md) - the order to run things on rented GPUs, with gates and costs
- [scripts/run_all.sh](scripts/run_all.sh) - reproduce every CPU result
- `--device auto|cpu|cuda` on the training scripts; GPU memory is recorded automatically

## P2 baseline recipe

Four attempts; the failed ones are documented in results/logs/FAILED_RUNS.md.
The working configuration (configs/ced_baseline_gen.yaml, 243k parameters):

- causal encoder with RoPE attention plus a short causal depthwise convolution (kernel 6)
- decoder memory projected from the final encoder states
- multi-query training: every lecture is encoded once and asked about every entity
- generative decoder that emits the reasoning chain (`e_1 ... e_h value`) and is scored
  only on its final token
- adaptive curriculum: direct -> one-hop -> two-hop -> 6 / 12 / 18 filler chunks
