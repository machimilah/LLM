# SN-CED: complete project state

**Last updated:** 2026-09-23
**Purpose:** everything needed to continue this project on another machine, or to hand it to
another person, with no context lost. Written to be read top to bottom.

---

# Part 1 — What this project is

## 1.1 The idea in plain words

A student who reads a lecture does not re-read the whole lecture before answering each exam
question. They took notes, and they reason from the notes, returning to the source only when the
notes are insufficient.

Large language models do the opposite. Every question re-reads the entire context. Cost and memory
grow with everything the model has ever seen, which is why long agent sessions get progressively
more expensive and why context windows are rationed.

SN-CED (Semantic Note Causal Encoder-Decoder) asks: after a model has understood some text, can it
keep a small set of explicit notes and reason from those instead?

A note is:

```
anchor  ->  micro-context        + source pointer + confidence + order
Maria   ->  sent contract Tokyo    sentence 47      0.98         2
```

The claim being tested is not that text can be summarised. It is that **a sparse set of explicit
anchor-plus-micro-context notes preserves enough for future reasoning that most of the original
token-level context no longer needs to stay active.**

## 1.2 Why it might matter commercially

Cost per question stops scaling with how much has been read. That matters wherever the same
material is queried repeatedly: agents with long histories, codebases, customer records, case
files, research corpora.

Second, less obvious advantage: the memory is readable text with a pointer to its source. It can be
audited, corrected, versioned, and a single fact can be deleted on request. A KV cache can do none
of that. For regulated buyers this is not cosmetic.

Honest counterpoint, measured in our own ablations: when a document is asked a **single** question,
ordinary retrieval is cheaper. The economics favour this architecture only under reuse.

## 1.3 Origin

The research plan is `SN_CED_ARCHITECTURE_AND_TEST_PLAN.md` (in this repo, written before the work
started). It specifies the architecture, the experiments, the gates, and twelve rules for the
implementing agent — including "never compare memory systems with a baseline that cannot solve the
task" (Rule 2) and "report negative results" (Rule 11). Those rules were followed and they stopped
work twice; see Part 5.

---

# Part 2 — The architecture

## 2.1 The target design

```
 document
    |
    v
 causal encoder (one pass)
    |                         |
    |-> detailed memory       |-> note compiler -> note encoder -> Semantic Note Memory
    |   (token level,         |     anchor, micro-context, source pointer,
    |    compressed)          |     confidence, order/supersession
    |                         |
    |-> local memory          v
    |   (recent tokens)   indexers (top-k retrieval over notes and detail)
    |                         |
    +--------> memory router <+       "cheapest sufficient memory"
                    |
                    v
                 decoder    reads notes by default; opens detail only when asked
```

## 2.2 Two implementations in this repository

**v1 — `src/snced/models/snced.py`.** The benchmark-fitted prototype that produced most of the
published synthetic results. Its compiler picks from closed sets (24 entities, 8 values, 2
relations) and its router walks the question's chain symbolically. Fast to train, but the design
only works because the synthetic task has known structure. Kept so earlier results stay
reproducible.

**v2 — `src/snced/models/snced_general.py`.** The general architecture. Differences that matter:

| | v1 | v2 |
|---|---|---|
| Compiler | closed-set classifiers | **soft pointers into the source** — any token can be an anchor |
| Note fields | anchor, relation, target | + **source span, confidence, order, supersession** |
| Router | symbolic chain walk, hand-set threshold | **learned**, trained with an explicit access price |
| Fallback | whole document | **only the source spans of the notes this question needs** |
| Note labels | gold labels required | **none** — notes emerge from the task and sufficiency losses |
| Sparsity | none | **L0 (hard-concrete) gates** on note count |

v2 is the one that tests the actual hypothesis. Because its pointers are soft (attention over
segment states), the whole compiler is differentiable, so the notebook can be learned end to end
with no supervision about what to write. Taking the argmax of the same distributions recovers the
note as readable text.

## 2.3 Composite loss (v2, plan section 10)

```
L = L_task(notes)                     reason from notes by default
  + route       * L_task(routed)      the path actually served
  + detail_task * L_task(detailed)    keep the reference path competent
  + sufficiency * KL(detail || notes) notes should behave like full context
  + notes       * E[number of notes]  compression pressure (L0, ramped in)
  + separation  * pointer overlap     anchor and micro-context must differ
  + detail      * E[slots reread]     price of touching detailed memory
```

## 2.4 Component status against the target diagram

| Component | Status |
|---|---|
| Semantic Note Compiler (5 fields) | built, v2 |
| Note Encoder | built (separate module — this matters, see 5.4) |
| Semantic Note Memory | built |
| Memory Router | built, learned, with access price |
| Local Memory (recent tokens) | built — `memory_tiers.LocalMemory` |
| Detailed Memory, compressed | built — `memory_tiers.CompressedDetailMemory` |
| Semantic / Detail Indexers | built — `memory_tiers.Indexer`, top-k retrieval |
| MoE feed-forward | built and unit-tested — `layers.MoEFeedForward`; **not used in trained models** |
| Sliding-window attention | built and unit-tested; not used in trained models |
| Sparse (top-k) attention | built and unit-tested; not used in trained models |
| Vision encoder / embedding | **not built** — no image data or multimodal task exists here |
| Engram | **not built** — no definition to implement against |
| DSpark | **not built** — decoder output head is a plain projection |

The trained models use dense attention and dense feed-forward. MoE and sparse attention are scale
infrastructure: they change cost, not the memory behaviour being measured.

## 2.5 Implementation choices that are NOT architecture

Two things were needed to make the synthetic benchmark learnable and are kept as flags, not as
components:

- **short causal depthwise convolution** in encoder blocks (`conv_kernel: 6`)
- **chain decoding** — the decoder emits `e5 -> e9 -> v3` rather than classifying the answer

Both are documented in `docs/architecture.md` with the evidence for why they were needed.

---

# Part 3 — Results

Every number below is reproducible from this repository. Raw per-run records are in
`results/raw/<experiment>/*.json` using the schema in plan section 28 (accuracy, note count, memory
slots, KV bytes, FLOPs, fallback rate, timings, seed, git commit).

## 3.1 Headline: memory stops growing with document length

`results/tables/scaling_benchmark.md` — 2 seeds, 40 documents per length, 36 questions per document.

| Document | Full context | Notes | Memory kept | KV memory | Compute | Time per task |
|---|---:|---:|---:|---:|---:|---:|
| 547 tok | 100.0% | 98.5% | 24 vs 547 (4.4%) | 36 KB vs 820 KB | 0.28 vs 0.49 GF | 38 vs 46 ms |
| 939 tok | 99.8% | 97.9% | 24 vs 939 (2.6%) | 36 KB vs 1,409 KB | 0.48 vs 0.85 GF | 41 vs 63 ms |
| 1,527 tok | 96.7% | 96.2% | 24 vs 1,527 (1.6%) | 36 KB vs 2,291 KB | 0.93 vs 1.54 GF | 48 vs 88 ms |
| 2,704 tok | 92.7% | 91.9% | 24 vs 2,704 (0.9%) | 37 KB vs 4,056 KB | 2.36 vs 3.45 GF | 64 vs 155 ms |

The document grows 5x; the notebook does not move. Time per task grows 1.7x for notes against 3.4x
for full context.

## 3.2 The note format against every alternative at a matched budget

`results/tables/ablations.md` — 3 seeds, 10 memory formats, same encoder, decoder and harness,
72 tokens of memory each.

| Memory format | Accuracy | Direct | One-hop | Two-hop |
|---|---:|---:|---:|---:|
| **Anchor + micro-context notes** | **99.42 ± 0.58%** | 99.5 | 99.2 | 99.5 |
| RAG top-k (query-conditioned) | 44.08 ± 1.54% | 98.9 | 13.4 | 19.9 |
| Extractive summary, truncated | 20.50 ± 0.87% | 28.1 | 17.4 | 16.0 |
| Sliding window | 18.28 ± 1.40% | 23.4 | 16.1 | 15.3 |
| SNM: phrase only (no anchor) | 26.89 ± 0.51% | 25.7 | 27.2 | 27.8 |
| SNM: keyword (anchor) only | 12.31 ± 0.53% | 11.8 | 11.8 | 13.2 |

Two readings matter. RAG scores 98.9% on single-fact questions and 13.4% on two-fact ones: a single
retrieval cannot chain. And **both halves of a note are necessary** — keyword alone is chance
(12.5%), context without its keyword is 26.9%, the two together are 99.4%.

Unbudgeted rivals still match full-context quality but cost 5-9x more memory: extractive summary of
fact sentences 100% at 351 slots, iterative RAG 99.97% at 185 slots.

## 3.3 v1 decisive comparison, mixed-length compiler

`results/tables/snced_vs_ced_mixed.md` — 3 seeds.

| Context | Baseline | Notes only | Ratio to baseline | Memory |
|---|---:|---:|---:|---:|
| 547 tok | 100.00% | 98.39 ± 1.55% | 98.4% | 24 vs 548 (4.4%) |
| 939 tok | 99.86% | 97.61 ± 1.86% | 97.7% | 24 vs 939 (2.6%) |
| 1,527 tok | 98.17% | 96.1% | 97.9% | 24 vs 1,528 (1.6%) |
| 2,704 tok | 93.97% | 93.89% | 100.1% | 24 vs 2,705 (0.9%) |

Every seed at every length retained at least 96.4% of full-context performance. At 2,704 tokens on
the weakest seed, notes **beat** full context (85.8% vs 82.9%) — the baseline degrades with length
while the notebook does not.

## 3.4 v2 general architecture: notes learned with no labels

`results/tables/snced_general.md` — seed 2, warm-started from a trained backbone (plan Phase 2).

| Path | Accuracy | Memory slots |
|---|---:|---:|
| Full context (reference) | 100% | 448 |
| **Semantic Note Memory only** | **100%** | **30 (7.1%)** |
| Router's choice | 100% | 30 |

No gold note labels anywhere. The notebook was shaped only by the task loss, the sufficiency term,
the L0 compression penalty and the pointer-separation penalty. At full length (18 filler, ~547
tokens) the same recipe also reached 100% on all three paths.

**Training recipe that made it work:** warm-start the encoder and decoder from a trained CED
baseline, freeze the encoder, hold the decoder still for 800 steps while the compiler learns, then
release it. Training everything from scratch at once left both paths stuck near 30%.

## 3.5 Router calibration

`results/tables/router_calibration.md` — 3 seeds, two policies, 8 thresholds.

At 1,528-token contexts:

| Policy | Accuracy | Memory slots | Fallback rate |
|---|---:|---:|---:|
| Notes only, no fallback | 96.33% | 24 | 0% |
| Lecture-level, t=0.9 (the old default) | 97.72% | **996** | 64.7% |
| **Question-level, t=0.5 (calibrated)** | 96.94% | **79** | 3.7% |

The old policy spent 996 slots to buy 1.4 points. Routing per question buys 0.6 points for 79
slots — 12.6x cheaper. Question-level is now the default in `src/snced/router.py`.

## 3.6 Adversarial cases (plan section 20)

`results/tables/adversarial.md` — 3 seeds, zero-shot.

| Variant | Full context | Predicted notes | Gold notes |
|---|---:|---:|---:|
| baseline | 100.0% | 97.6% | 99.6% |
| temporal_update (a fact restated later) | 84.9% | 78.8% | 80.3% |
| exact_value (two-token answers) | 10.6% | 3.6% | 1.1% |

Temporal updates hurt everything zero-shot — no model was trained on restated facts — and notes are
~6 points worse. `exact_value` is **inconclusive**: the decoder was trained to emit one value token,
so full context fails too (Rule 2). Both are training-distribution gaps, not verdicts.

Not testable in the synthetic setting: confusable entity names (every token is equally distinct in a
word-level vocabulary) and exact quotation (no free text to quote).

## 3.7 Timing and cost per successful task

`results/tables/timing.md` — 3 seeds, one condition at a time on an idle machine.

| Context | Condition | ms per successful task | GFLOPs per successful task |
|---|---|---:|---:|
| 547 tok | full context | 55 | 0.490 |
| | notes | **40 (-27%)** | **0.283 (-42%)** |
| 1,528 tok | full context | 111 | 1.578 |
| | notes | **54 (-51%)** | **0.970 (-39%)** |

A "task" is one document plus 36 questions, so note-building is amortised. Caveat recorded in the
table: the fallback condition builds both memories in this implementation, so its wall-clock is
pessimistic.

## 3.8 Real text — the open problem

### Gold notes are sufficient

`results/tables/real_text_compiler.md` — BABILong 1k (bAbI facts hidden in PG19 book prose), n=100.

| Task | Gold-note retention | Notes vs document |
|---|---:|---:|
| qa1 (one supporting fact) | **100%** | 17 tok vs 485 (3.6%) |
| qa2 (two supporting facts, order) | **100%** | 47 tok vs 495 (9.6%) |
| qa9 (negation) | **100%** | 20 tok vs 479 (4.1%) |

So the note *format* handles real-text relational structure: ordered events, several relations per
anchor, negation. Measured with a deterministic reader (no LLM) that replays notes in order.

### The compiler does not reach that

| Compiler | qa1 | qa2 | qa9 | Notes |
|---|---:|---:|---:|---|
| Closed-vocabulary heads, frozen MiniLM | 88% | 90% | 92% | but a closed entity list |
| Open-vocabulary span tagger, frozen MiniLM | 82% | 73% | 82% | 66/51/61% on held-out entities |
| + fine-tuned encoder | 82% | 74% | 82% | held-out gap **closed**: 82/74/82% |
| + 225k training sentences, 20k GPU steps | 85% | 78% | 84% | training loss 0.0000 |
| + E5-base-v2 encoder (109M vs 22M) | **90%** | **82%** | **88%** | best worst-case **0.86** |

Gate is 0.95. Not met.

**What this establishes:** training loss is 0.0000 while real-document retention oscillates between
0.73 and 0.90 across evaluations. It fits its training data perfectly. 28x more data moved the
worst case 8 points; a 5x bigger encoder moved it 4 more. **This is a distribution gap, not a
compute or capacity limit.** Training uses bAbI facts inside WikiText encyclopedia prose;
evaluation uses the same facts inside 19th-century book prose.

### Reference-model gate

`results/raw/kaggle_qa/` — Qwen2.5-7B-Instruct in 4-bit, full context, 50 questions per task:

| Task | Accuracy | Gate (70%) |
|---|---:|---|
| qa1 | 84% | pass |
| qa2 | 54% | **fail** |
| qa9 | 92% | pass |

A 7B model is a fair reference for single-fact and negation tasks but cannot reliably chain two
facts at 1k context. Comparisons at that scale are only meaningful on some tasks today.

Earlier attempts at 0.5B and 1.5B were **stopped** because full-context accuracy was 48%/32% —
below the agreed 70% bar (Rule 2).

## 3.9 Reproduction of the historical experiments

`results/tables/reproduction.md`. The original prototype scripts were not in the folder, so both
experiments were rebuilt from the plan's description. All four Phase-0 acceptance criteria pass:
SNM accuracy ~100%, word-only ~chance (11.74% vs 12.15% reported), compression 86.84% (vs 87.45%),
information recovery 100% (vs 99.93%).

Caveat recorded in that table: the reconstruction appears slightly easier than the original, so the
SNM-vs-RAG token gap is larger here (58% vs the plan's 44%). Do not read that as an improvement.

---

# Part 4 — Repository

## 4.1 Layout

```
SN_CED_ARCHITECTURE_AND_TEST_PLAN.md   the original research plan
HANDOFF.md                             this file
README.md                              status table and quick start
docs/
  architecture.md                      what is built, why, and what each decision cost
  gpu_runbook.md                       order of operations on rented GPUs, with gates and costs
  free_gpu_plan.md                     what free hardware can and cannot do
src/snced/
  data.py            synthetic lecture generator, vocabulary, questions
  notes.py           SemanticNote, Notebook, deterministic traversal
  memory.py          memory conditions for the representation ablation
  compiler.py        v1 GRU note compiler
  router.py          rule-based routing, question-level (calibrated default)
  fallback.py        source store and fetch accounting
  adversarial.py     plan section 20 lecture variants
  metrics.py         aggregation and analytic FLOPs
  runlog.py          run records (plan section 28), seeding, device helpers
  training.py        checkpoint/resume, session budget, 4-bit loading, LoRA
  models/
    ced.py                 causal encoder-decoder backbone (RoPE, optional conv, generative head)
    snced.py               v1 SN-CED
    snced_general.py       v2 general architecture
    note_compiler.py       open-vocabulary pointer compiler
    note_memory.py         note encoder (fields -> memory slots)
    sufficiency_router.py  learned router + access cost
    memory_tiers.py        local memory, compressed detail, indexers
    gates.py               L0 hard-concrete gates
    layers.py              MoE, sliding-window and sparse attention
experiments/
  synthetic/    ablation, learned_compiler, end_to_end, snced_compare, ablations,
                router_calibration, adversarial_eval, timing_pass, general_train,
                scaling_benchmark, and their report scripts
  real/         notes_data, span_data, train_note_compiler, span_compiler,
                babilong_pilot, kaggle_stage0
kaggle/snced_stage0.ipynb              ready-to-run notebook
scripts/run_all.sh                     reproduce every CPU result
results/
  raw/<experiment>/*.json              one record per run, never deleted
  logs/                                training curves, console logs, FAILED_RUNS.md
  tables/                              generated reports
  figures/                             Plot A, C, F
tests/                                 39 tests
```

## 4.2 Environment

Python 3.14, torch 2.12 (CPU), transformers 5.10.2, datasets 5.0.0. `requirements.txt` is pinned to
what produced these results.

```bash
pip install -e ".[dev]"
python -m pytest -q                      # 39 tests, ~10 s
python experiments/synthetic/end_to_end.py --config ced_baseline_gen.yaml --device cuda
```

All training scripts take `--device auto|cpu|cuda`; GPU memory is recorded automatically when CUDA
is present.

## 4.3 Machine-specific gotchas discovered the hard way

- **bfloat16 is ~26x slower than float32 on this CPU** (no AMX). Model size on the laptop is capped
  by RAM (1.5B fp32 = 6.0 GB resident), not by patience.
- **The laptop sleeps during long runs.** One epoch took 70 minutes of wall time for 7 minutes of
  compute. Wall-clock numbers from interrupted runs are meaningless; check CPU-time vs wall-time.
- **Windows console is cp1252.** Set `PYTHONIOENCODING=utf-8` or printing `±` crashes the script.
- **Do not write regexes or escapes through nested heredocs.** `\b` became a literal backspace byte
  and `\n` became a real newline, twice. Use the editor.

---

# Part 5 — Everything that failed, and what it taught

Full detail with diagnoses in `results/logs/FAILED_RUNS.md`. This section exists because the
failures are where the design knowledge is.

## 5.1 The baseline took four attempts

The synthetic task looks trivial and was not. Attempts 1-2 never learned to bind an entity to its
value: the model answered with *any* value present in the context and sat at that plateau for
thousands of steps. Changing width, depth, learning rate and batch size did nothing. A 24-token
probe showed the same failure, so it was not sequence length.

What fixed it:
1. **Short causal convolution** in encoder blocks — a value token can see its entity from step one
   instead of having to learn the binding at every offset.
2. **Multi-query training** — encode a lecture once, ask about every entity. Contrasting targets in
   the same context break the "answer any value" attractor. 12 entities learned *faster* than 2.
3. **Generative chain decoding** — emit `e5 -> e9 -> v3`. Each hop becomes a single lookup, which
   the model already did perfectly. One-shot classification never learned two hops (stuck at 30%).

## 5.2 The notes path took three attempts

- **Re-encoding note text with the lecture encoder: 32%.** The encoder was trained on prose and
  gives packed note sequences unusable states. Fixing this required a **separate note encoder**
  producing one memory slot per note: 99.4%.
- **Compiler under-trained**: stopped before comparing anything, and a gate added (held-out
  extraction >= 99%) so a weak compiler cannot silently invalidate a comparison.

## 5.3 Four measurement bugs that would have produced wrong claims

- **The reference path was never trained** in v2: the detailed branch appeared only under
  `no_grad` and inside a routed path the router never opened, so it sat at chance while the
  sufficiency term trained notes to imitate noise.
- **Build cost divided by the wrong number** in the timing harness (n/6 instead of n), inflating
  every condition's memory-building time 6x and making notes look slower than full context. Notes
  are in fact 27-51% cheaper per successful task.
- **Query-conditioned memory charged once per document** instead of once per question, which
  flattered RAG in the FLOPs column.
- **"Unseen entities" that were not unseen**: the span compiler's evaluation renamed entities using
  the *same pools* used in training, which reported 89% where the true number on genuinely unseen
  strings was 10%.

## 5.4 Two compression penalties that were gamed

- **Sigmoid gates**: the model drove every gate to ~0.44 (spread 0.007) and scaled the note encoder
  up to compensate. The penalty fell; nothing was pruned. Replaced with **L0 hard-concrete gates**,
  where the cost is P(gate > 0) rather than gate magnitude.
- **Saturated L0 gates**: after a warm-up with the penalty off, every gate sat at exactly 1.0 where
  the sigmoid gradient vanishes, so no compression weight could close them. Fixed by **clamping the
  gate logit** to ±4.

Current status of sparsity: mechanically honest, but on this task the model keeps all notes at
lambda <= 16 once the task is solved. Pruning appeared only in an under-trained run. A proper
lambda sweep at convergence would draw the quality/compression curve.

## 5.5 Work stopped by the plan's own rules

- **BABILong at 0.5B and 1.5B**: full-context accuracy 48%/32%, below the agreed 70% bar. Stopped
  rather than comparing memory formats under a model that cannot do the task.
- **Stage 0b five-arm comparison**: not run, because the compiler missed its 95% retention gate.

## 5.6 A checkpoint destroyed by a smoke test

`end_to_end.py` saved a checkpoint at the end of every run, so a 2-step `--device` test overwrote
the healthy seed-1 baseline. Detected when warm-starting produced 0.000 accuracy with everything
frozen. Seeds 2-3 were intact; seed 1 was retrained; a guard now refuses to overwrite an existing
checkpoint from a run of under 1000 steps. No published result was affected.

---

# Part 6 — Infrastructure

## 6.1 Kaggle

Working setup, on the account **macs26**:

- Token format is the new `KGAT_...` style, stored at `~/.kaggle/access_token` (chmod 600), **not**
  `kaggle.json`. Set it up with:
  `mkdir -p ~/.kaggle && printf '%s' '<token>' > ~/.kaggle/access_token && chmod 600 ~/.kaggle/access_token`
- Install the real client: `python -m pip install kaggle` (a namespace-only `kaggle` directory on
  the path will import but expose nothing).
- **GPU and Internet require phone verification.** Without it a kernel silently runs CPU-only with
  no network and fails on the first model download, while still reporting COMPLETE.

Assets created (all private):

| Asset | Purpose |
|---|---|
| `macs26/snced-code` | dataset holding the repository (zipped per folder) |
| `macs26/sn-ced-stage-0-note-compiler` | Stage 0 with MiniLM — completed, 0.82 |
| `macs26/sn-ced-stage-0-e5` | Stage 0 with E5-base — completed, 0.86 |
| `macs26/sn-ced-stage-0b-comparison` | 7B baseline gate — completed |

Push workflow: update the dataset with `dataset_create_version`, then `kernels_push`. Kernel ids
must match the slug Kaggle derives from the title, or the push 409s. Kaggle publishes **no logs
until a kernel finishes**.

`experiments/real/kaggle_stage0.py` runs both stages with `--resume` (atomic checkpointing) and
`--session-hours` so a run stops cleanly before the platform kills it.

**Security note:** an API token was pasted into a chat transcript during this work and should be
expired if it has not been already (Kaggle -> Settings -> API -> Expire Token).

## 6.2 GitHub — unresolved

Remote is set to `https://github.com/machimilah/LLM.git`. Two commits exist locally. **The push is
blocked**: the credentials stored on the old machine belong to `machidevelop`, which lacks write
access (403). Resolve by adding that account as a collaborator, using a fine-grained token for
`machimilah`, or pushing from an authenticated machine:

```bash
git push -u origin main
```

---

# Part 7 — Where the project stands

## 7.1 What is established

1. **The note representation works.** Anchor + micro-context preserves multi-hop reasoning at 1-7%
   of the memory, beats every alternative at a matched budget, and both halves are necessary.
2. **A model can learn what to remember with no supervision.** 100% from notes, equal to full
   context, on 7% of the memory, shaped only by task loss, sufficiency, compression and an access
   price.
3. **Memory stops growing with document length**, which is the commercial claim, measured across a
   5x range.
4. **The format handles real-text structure** — ordered events, multiple relations per anchor,
   negation — with gold notes answering 100% of real BABILong questions at 4-10% of the document.

## 7.2 What is not established

- Anything above 2,704 tokens or 445k parameters.
- Real HBM or latency on a GPU-served model (KV figures are computed from tensor shapes).
- Any comparison against a production system, or any real-world cost reduction.
- That the compiler can write reliable notes about real prose. **This is the blocker.**

## 7.3 The immediate next experiment

Retention plateaus at 0.86 while training loss is 0.0000, across two encoder sizes and a 28x data
increase. The remaining gap is a **distribution mismatch**: training uses WikiText encyclopedia
prose, evaluation uses 19th-century book prose.

**Experiment:** train the compiler on book prose instead of WikiText, everything else unchanged.

**Methodological catch that must be handled:** BABILong's noise comes from PG19, which is Gutenberg
books before 1919. Training on Gutenberg risks overlapping with the exact books in the test
documents. That would not be label leakage — the bAbI facts are inserted separately — but it turns
"works on real prose" into "works when trained on the deployment distribution".

- **Fast version**: train on Gutenberg directly. Answers the diagnostic question today; the number
  must be labelled distribution-matched, not general.
- **Clean version**: train on literary prose from a different source or outside PG19's date range.
  Slower to set up, defensible as a general claim.

Run the fast one as a diagnostic first. Note that the weaker scientific claim is a perfectly good
commercial one: in a deployment you fine-tune the compiler on the customer's own documents.

## 7.4 Roadmap and cost

| Stage | Question | Gate | Cost |
|---|---|---|---|
| **0. Compiler** | Reliable notes on real prose? | >= 95% retention on unseen documents | free tier or ~$50 |
| **0b. Reference** | Is a 7B good enough to compare against? | >= 70% full context | done: passes on qa1/qa9, fails qa2 |
| **1. Integration** | Does a competent model do better with notes than full context / RAG / summaries? | 95-99% of baseline quality, >= 50% less memory, no compute increase, >1 benchmark | ~$100k, 2-3 months |
| **2. Native** | Does the advantage survive in a model trained with this memory from the start? | same, at scale | $1M+ |

Do not pretrain before Stage 1 passes. A bolt-on that works is strong evidence a native version
works better; a bolt-on that fails means the native version would have failed expensively.

## 7.5 Still unbuilt, from the plan

- **Note consolidation and eviction** (section 21) — merging repeated notes, retiring stale ones.
  A long-lived agent needs this; nothing merges or retires notes today.
- **Versioned notes** (section 22) — "what was the launch date *last week*". Order and a
  supersession score are carried in each note, but no version query exists.
- **Learned routing with a swept access cost** to draw the quality-vs-cost curve (Plot D/E at
  convergence).
- **Real benchmarks**: RULER, LongBench v2, LongMemEval. BABILong is partially done.

---

# Part 8 — How to talk about this

## 8.1 Positioning

Lead with **the memory layer for AI agents**: agents re-read everything they have ever seen; this
makes them keep notes and answer from those. It sells on top of any model, before anything is
trained, and the buyers are people querying long-lived material repeatedly.

Do **not** lead with "a new LLM architecture". Nothing is tested above 2,704 tokens or 445k
parameters, and that framing invites a comparison to DeepSeek that loses immediately. It is the
long-term prize, not the opening claim.

The second argument is underrated: notes are readable, correctable, versionable, deletable. A KV
cache is none of those, and for regulated buyers that is not cosmetic.

## 8.2 What is safe to show

`results/tables/investor_summary.md` states the results with their scale limits on the same page:
the scaling table, the matched-budget comparison, the label-free result, and explicitly the 82-86%
real-text ceiling, the 445k parameter count, and the workload dependence.

**Never claim:** superiority over any existing model; anything at 128k context; measured HBM
savings; real-world cost reduction.

**Show `FAILED_RUNS.md`.** A lab that kills its own results — twice, by its own pre-agreed rules —
is why the surviving numbers are believable.

## 8.3 The honest moat assessment

The architecture is publishable and reproducible by any competent lab in months. What is defensible
is compilation quality on messy real text, and the data from real agent workloads about what is
worth remembering. That argues for shipping the runtime early, for the data as much as the revenue.

---

# Part 9 — If you are picking this up cold

1. Read `SN_CED_ARCHITECTURE_AND_TEST_PLAN.md` sections 1-10 for the idea and the loss design.
2. Read `docs/architecture.md` for what exists and why each decision was made.
3. Read `results/logs/FAILED_RUNS.md` — it is the fastest way to avoid repeating a week of work.
4. Run `python -m pytest -q` (39 tests, ~10 s) to confirm the environment.
5. Run `python experiments/synthetic/scaling_benchmark.py` to reproduce the headline table.
6. The open problem is `experiments/real/span_compiler.py` and its 0.86 ceiling. Start at Part 7.3.

---

# Part 10 — Appendices: the details that reproduce the work

## A. The synthetic benchmark, exactly

Defined in `src/snced/data.py`. Reconstructed from the plan's description because the original
prototype scripts were not in the folder.

**A document ("lecture")** has 42 chunks in shuffled order:

- 12 **value facts**, one per entity: *"the lecture explains that e17 has value v4 and this detail
  is relevant to the topic ."* — 4 template variants
- 12 **link facts**, one per entity: *"in the framework e17 connects to e3 and this relation
  matters later ."* — 4 template variants
- 18 **filler** sentences — 10 variants, some mentioning entities or the words "value" and
  "connects", so chunk type cannot be read off surface keywords

Entity pool `e0..e23` (12 sampled per document), values `v0..v7` (chance = 12.5%), vocabulary of
156 word-level tokens, no UNK. Lecture length ~547 tokens at 18 filler; the length sweep uses 54,
108 and 216 filler (939 / 1,528 / 2,705 tokens).

**Three question types**, all answered by a single value token:

- direct: *"what is the value of e5 ?"*
- one-hop: *"what is the value of the entity that e5 connects to ?"*
- two-hop: *"... reached from e5 after two connections ?"*

**Adversarial variants** (`src/snced/adversarial.py`): `temporal_update` restates an entity's value
later so the last statement wins; `exact_value` makes answers two tokens ("v3 v7"), which a
three-token note cannot hold. Both use only in-vocabulary text, so results measure memory rather
than unknown-token handling.

## B. Historical experiments reproduced (P1)

`results/tables/reproduction.md`, 3 seeds each (123/456/789 and 11/22/33).

**Experiment A — representation and fallback ablation.** The same 42k-parameter GRU reasoner is
trained on every memory condition; memory is assembled per question (query-conditioned, which is
why this tests the representation rather than autonomous compilation):

| Condition | Accuracy (rebuilt) | Plan | Active tokens/q | Fallback |
|---|---:|---:|---:|---:|
| iterative RAG | 100.00 ± 0.00% | 100.00% | 40.23 | — |
| SNM | 100.00 ± 0.00% | 100.00% | 17.00 | — |
| SNM, 20% of notes damaged | 80.76 ± 2.21% | 82.22% | 16.58 | — |
| SNM + selective fallback | 100.00 ± 0.00% | 100.00% | 19.53 | 21.0% |
| word only (anchors, no context) | 11.74 ± 1.68% | 12.15% | 13.00 | — |

**Experiment B — learned query-independent compiler.** A 22,823-parameter causal GRU (the plan
reports 22,923) reads every chunk before any question exists:

| Metric | Rebuilt | Plan |
|---|---:|---:|
| Raw lecture tokens | 546.97 | 573.44 |
| Note tokens | 72.00 | 71.98 |
| Text reduction | 86.84% | 87.45% |
| Memory-slot reduction | 42.86% | 42.87% |
| Information recovery (direct / 1-hop / 2-hop) | 100 / 100 / 100% | 100 / 99.79 / 100% |

The reader is deterministic graph traversal over predicted notes, **not** an LLM decoder — the same
limitation the plan flags in section 11.2.

## C. BABILong five-arm comparison

`results/tables/real_text_compiler.md`. Qwen2.5-0.5B-Instruct answers every arm; 25 questions per
cell, ±~19pp at 95% confidence:

| Memory | qa1 | qa2 | Prompt tokens | Memory build |
|---|---:|---:|---:|---:|
| Full context | 48.0% | 36.0% | 710 | — |
| RAG top-k (query-conditioned) | 48.0% | 16.0% | 268 | — |
| LLM-generated summary | 20.0% | 16.0% | 153 | 16–18 s |
| LLM-prompted notes | 36.0% | 4.0% | 96 | 7–11 s |
| **Trained compiler notes** | **52.0%** | **32.0%** | **96** | **0.16–0.18 s** |

Three things this shows: a **trained** compiler beats a **prompted** LLM compiler decisively
(52 vs 36, and 32 vs 4) at the same prompt size and ~50x faster memory building; notes match or
beat full context on qa1 with 7.4x fewer prompt tokens; and the QA model is the bottleneck, since
the same notes answer 88–90% under a deterministic reader.

**Caveats recorded with that table:** qa9 is omitted because the model answers in sentences rather
than yes/no, making word-containment scoring unreliable in both directions. Both LLM-memory arms
were given a task-aware prompt hint (that the text hides short statements about people and places)
after a generic prompt made the 0.5B summarise the literary filler instead. The hint was applied
equally to the summary and prompted-notes arms and is a deviation from zero-shot use.

## D. Sparsity investigation, in full

The compression penalty has been through three designs:

1. **Sigmoid gates, penalty on gate magnitude** — gamed. Every gate settled at 0.435–0.442 (spread
   0.007) while the note encoder scaled its outputs up to compensate. The reported "18.5 notes" was
   really 42 x 0.44; hard pruning at any threshold up to 0.3 removed nothing.
2. **L0 hard-concrete gates, penalty on P(gate > 0)** — honest, but gates saturated at exactly 1.0
   during the penalty warm-up, where the sigmoid gradient vanishes. Sweeping lambda to 4 and 16
   pruned nothing: both finished at 100% accuracy keeping all 30 notes.
3. **L0 with the logit clamped to ±4** — gradients survive. A 3,000-step run pruned 30 → 23.9 notes
   (the fast config has exactly 24 fact sentences and 6 filler), but had not finished the curriculum
   and sat at 81.7%. Given 8,000 steps it reached 100% on all paths and re-opened to 29.9 notes.

**Current state:** sparsity works mechanically; on this task the optimum keeps every note at
lambda <= 16 once the task is solved, because the penalty (~0.28 at lambda=4) is cheaper than any
accuracy risk. Drawing the real quality-vs-compression curve needs a lambda sweep run **to
convergence** at each point — that is the missing Plot F for v2.

## E. Performance engineering findings

- `SNCED.compile` originally indexed head tensors chunk by chunk: roughly 0.9 ms of dispatch
  overhead per chunk, about 1.9 s per 48-document batch. Moving every head to Python lists in one
  pass fixed it. Steady-state breakdown per document afterwards: encoder 2.33 ms, compile 2.96 ms
  (of which chunk-span packing is 2.04 ms), note memory 0.02 ms.
- CPU timings are comparable only when one condition runs at a time on an idle machine; earlier
  numbers taken under contention included a 23.7-second outlier.

## F. Exact commands

```bash
# Historical reproduction (P1)
python experiments/synthetic/learned_compiler.py            # seeds 11 22 33
python experiments/synthetic/ablation.py                    # seeds 123 456 789
python experiments/synthetic/report.py                      # -> reproduction.md

# CED baseline (P2). MUST reach >=95% before anything is compared against it
python experiments/synthetic/end_to_end.py --config ced_baseline_gen.yaml --seeds 1 2 3

# v1 decisive comparison, mixed-length compiler
python experiments/synthetic/snced_compare.py --seeds 1 2 3 --tag mixed
python experiments/synthetic/compare_report.py --experiment snced_compare_mixed --out snced_vs_ced_mixed

# Memory-format ablations, router calibration, adversarial, timing, scaling
python experiments/synthetic/ablations.py --seeds 1 2 3
python experiments/synthetic/ablations_report.py
python experiments/synthetic/router_calibration.py --seeds 1 2 3
python experiments/synthetic/adversarial_eval.py --seeds 1 2 3
python experiments/synthetic/timing_pass.py --seeds 1 2 3          # idle machine only
python experiments/synthetic/scaling_benchmark.py --seeds 2 3 --fillers 18 54 108 216

# v2 general architecture - the warm-start flags matter, see 3.4
python experiments/synthetic/general_train.py --config snced_general.yaml --seeds 2 \
    --init-backbone ced_baseline_seed{seed}.pt --freeze-encoder --decoder-frozen-steps 800
python experiments/synthetic/general_report.py

# Real text
python experiments/real/train_note_compiler.py --train-docs 3000 --epochs 6   # closed vocabulary
python experiments/real/span_compiler.py --train-docs 3000 --epochs 4         # open vocabulary
SNCED_ENCODER=intfloat/e5-base-v2 python experiments/real/span_compiler.py --finetune-encoder
python experiments/real/babilong_pilot.py --tasks qa1 qa2 --limit 25          # five-arm QA

# On a GPU box / Kaggle
python experiments/real/kaggle_stage0.py --stage compiler --session-hours 8 --resume
python experiments/real/kaggle_stage0.py --stage qa --gate-only \
    --model Qwen/Qwen2.5-7B-Instruct --bits 4
```

`scripts/run_all.sh` chains the CPU-runnable set in order.

## G. Checkpoint inventory (`results/checkpoints/`, gitignored)

| File | What it is |
|---|---|
| `ced_baseline_seed{1,2,3}.pt` | healthy CED baselines, 100% on all question types. **Seed 1 was destroyed by a smoke test and retrained** |
| `snced_mixed_seed{1,2,3}.pt` | v1 SN-CED, mixed-length compiler — the checkpoints behind the headline v1 tables |
| `snced_seed{2,3}.pt` | v1 SN-CED, fixed-length compiler (superseded) |
| `snced_attempt2_seed1.pt` | v1 attempt 2, shared-encoder note memory — the 32% failure, kept for the record |
| `snced_general_seed2.pt` | v2 general architecture |
| `note_compiler_real_seed0.pt` | closed-vocabulary real-text compiler (88/90/92%) |
| `span_compiler_seed0.pt` | open-vocabulary span tagger, frozen MiniLM |
| `span_compiler_ft_seed0.pt` + `span_encoder_ft_seed0.pt` | span tagger with fine-tuned encoder (87 MB) |

## H. External models and datasets used

| Asset | Role |
|---|---|
| `Qwen/Qwen2.5-0.5B-Instruct`, `-1.5B-`, `-7B-` | QA models for the real-text pilots |
| `sentence-transformers/all-MiniLM-L6-v2` (22M) | default compiler backbone |
| `intfloat/e5-base-v2` (109M) | stronger backbone, selected via `SNCED_ENCODER` |
| `microsoft/deberta-v3-base` | **untried** — needs `tiktoken` / sentencepiece installed |
| `RMT-team/babilong`, config `1k`, splits qa1..qa10 | real-text evaluation |
| `Muennighoff/babi` | bAbI training stories (tasks 1, 2, 9) |
| `Salesforce/wikitext`, `wikitext-103-raw-v1` | training noise, and source of the mined entity vocabulary |
| `pg19` | **not loadable** — script-based dataset, removed from `datasets` |

## I. Real-text data pipeline

`experiments/real/notes_data.py` and `span_data.py`:

1. bAbI stories provide the relational structure (five relations: move, take, drop, state_in,
   state_not_in). Templates are parsed with closed-vocabulary regexes to produce **gold labels
   only**; the compiler itself learns open-vocabulary extraction.
2. Entities are substituted from a vocabulary **mined from WikiText** (2,626 names / 4,723 nouns),
   split into disjoint train and eval halves, with 15% of documents keeping the original cast. This
   replaced fixed 40–70 word pools, which the tagger simply memorised.
3. Fact sentences are interleaved into WikiText prose, and 30% of the time glued onto a neighbouring
   sentence, because BABILong does the same.
4. Evaluation uses real BABILong documents, optionally renamed with the held-out vocabulary half.

The **deterministic reader** (`answer_from_notes`) replays notes in order — locations, holders,
dropped objects, negations — and answers with no LLM, so retention measures the notebook rather than
a reader. Gold notes score 100% on all three tasks, which is what makes it a valid ceiling.

## J. Test coverage (39 tests)

| File | What it protects |
|---|---|
| `test_data.py` | lecture structure (42 chunks, 12/12/18), required chunks support the answer, no UNK |
| `test_notes.py` | note text and fields round-trip, notebook traversal, missing-note behaviour |
| `test_router.py`, `test_router_policies.py` | sufficiency rules; per-question confidence uses only the notes a question needs; a missing chain forces fallback |
| `test_fallback.py` | source-store accounting; fallback restores the answer exactly when corrupted; word-only hides the answer |
| `test_compiler.py` | v1 compiler shapes, loss, query-independence |
| `test_ced.py` | encoder causality, right-padding invariance, generative decoding shapes |
| `test_snced_general.py` | segmentation modes, query-independent compilation, pointers resolve to real document tokens, router probability bounds, **targeted reread costs <=40 tokens not 120**, three memory tiers, indexer top-k, composite loss reaches all four learned components, L0 gate cannot be gamed, compression ramp schedule |
| `test_adversarial.py` | variants stay in vocabulary, temporal answer is the later value, exact_value needs two tokens |
| `test_layers.py` | MoE routes to k of n experts and balances load, sliding window forgets, top-k sparse keeps k keys |
| `test_training_utils.py` | checkpoint round-trip restores weights/optimizer/step/RNG, atomic save, session budget |

## K. Figures

`results/figures/`: Plot A (quality vs active memory), Plot C (quality vs context length) for both
the fixed- and mixed-length compilers, Plot F (compression vs quality across all ten memory
formats), plus `*_attempt2.png` from the superseded shared-encoder run.

## L. Configuration files

| Config | Drives |
|---|---|
| `synthetic_small.yaml` | the two historical experiments |
| `ced_baseline.yaml` | P2 baseline, classification decoder (superseded) |
| `ced_baseline_gen.yaml` | **P2 baseline in use** — generative chain decoder, conv kernel 6 |
| `snced.yaml` | v1 comparison: compiler gate 0.99, decoder gate 0.97, router policy `question` at t=0.5, eval at 18/54/108/216 filler |
| `snced_general.yaml` | v2 full length |
| `snced_general_fast.yaml` | v2 quick signal (6 filler, 30 eval documents) |
| `ablations.yaml` | 72-token budget, RAG top-k 6, iterative RAG 3 rounds |
| `tmp_lambda{4,16}.yaml` | leftovers from the sparsity sweep; safe to delete |

## M. Hardware these results came from

Windows 11, 16 CPU threads, 15.2 GB RAM, **no GPU**. Kaggle T4 (16 GB) for the two GPU runs. That
is why model sizes are what they are, and why some measurements carry contention caveats.
