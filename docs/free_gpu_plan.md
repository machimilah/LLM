# Running the next stages on free GPUs

Everything needed is in this repository and in [kaggle/snced_stage0.ipynb](../kaggle/snced_stage0.ipynb).
The notebook is the whole plan; this file explains what it does, what it cannot do, and how to read
the result.

## What free hardware can and cannot do

Kaggle gives 30 h/week of 2x T4 (16 GB each), sessions up to ~9 h. Colab free gives less and
disconnects sooner.

| Work | Free tier | Why |
|---|---|---|
| Stage 0: train the note compiler on real text | yes, hours | a 100-300M encoder trains comfortably on one T4 |
| Stage 0b: 7B comparison on BABILong | yes, ~10-20 GPU-h | 7B in 4-bit is ~5 GB, inference only |
| Stage 1-2: note branch on a 1.5B with LoRA, 2-4k context | yes, slowly | weeks of sessions |
| Same at 7B with long context | no | training activations exceed 16 GB |
| RULER at 32k-128k | no | context length, not parameters, is the wall |
| Pretraining anything | no | orders of magnitude away |

T4s have no bf16 and modest throughput, so expect 5-10x slower than an A100. Fine for evaluation,
painful for training. About $50 of rented A100 time replaces roughly a week of Kaggle.

## Session survival

Free sessions die without warning, so every long run here:

- **checkpoints atomically** (`snced/training.py`), including optimizer, step, curriculum stage and
  RNG streams, so the data stream continues identically after a restart
- **resumes** with `--resume` (on by default)
- **stops itself** before the platform's limit with `--session-hours`, rather than being killed
  mid-write

Rerunning the same notebook cell after a session ends continues from the last checkpoint.

## The two gates

**Stage 0 - compiler quality.** Trains on bAbI facts inside WikiText prose with substituted
entities; evaluates on real BABILong documents whose noise is book text and whose entities were
held out. Gate: >= 95% retention on qa1, qa2 and qa9. Current CPU result is 82% (74% on qa2), and
the suspected limit is the frozen MiniLM backbone, so the notebook fine-tunes the encoder and, if
that is not enough, the next thing to try is a stronger backbone (`ENCODER_ID` in
`experiments/real/span_compiler.py`: deberta-v3-base, e5-base).

**Stage 0b - does it help a competent model.** Runs the full-context baseline FIRST and refuses to
interpret anything if it scores below 70%: comparing memory formats under a model that cannot do
the task is what invalidated the 0.5B and 1.5B runs (Rule 2). Only then does it run all five arms
at a matched token budget.

## Reading the result

- Stage 0 writes `results/checkpoints/kaggle_compiler_seed0.json` with `best_retention` and
  `passed`.
- Stage 0b writes one record per condition to `results/raw/kaggle_qa/`, and every individual answer
  to `results/logs/kaggle_qa_1k.jsonl` so nothing is lost if a session dies.
- The last notebook cell zips both into `/kaggle/working/snced_results.zip`. Download it before the
  session closes.

## If both gates pass

Then, and only then, is money worth spending: rent a single A100 for Stages 1-2 (attach the note
branch to a 7B with LoRA, measure real HBM and latency on RULER/BABILong/LongMemEval). See
[gpu_runbook.md](gpu_runbook.md) for that sequence and its costs.

## If Stage 0 fails

The architecture is not refuted - the note *representation* answers 100% of these questions when
the notes are correct (results/tables/real_text_compiler.md). What fails is writing them. The next
lever is a bigger or better-pretrained encoder, not a redesign of the memory.
