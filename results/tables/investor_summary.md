# SN-CED: evidence summary

**What this is.** A research prototype testing one idea: after a model has read something, can it
keep a small set of explicit notes instead of the whole text, and reason from those?

**Scale, stated up front.** Every number below comes from a **445,000-parameter** model on
**synthetic documents of 547-2,705 tokens**, plus one real-text experiment on BABILong. For
comparison, a small production LLM is ~7 billion parameters and useful context is 128,000 tokens.
This is a mechanism demonstration. It is not a product benchmark, and it does not show that SN-CED
beats any existing model.

---

## 1. The core result: memory stops growing with document length

Same trained model, answering the same questions, from notes vs from the full text:

| Document | Accuracy, full text | Accuracy, notes | Memory kept | KV memory | Compute | Time per task |
|---|---:|---:|---:|---:|---:|---:|
| 547 tokens | 100.0% | 98.5% | 24 vs 547 slots (4.4%) | 36 KB vs 820 KB | 0.28 vs 0.49 GF | 38 vs 46 ms |
| 939 tokens | 99.8% | 97.9% | 24 vs 939 (2.6%) | 36 KB vs 1,409 KB | 0.48 vs 0.85 GF | 41 vs 63 ms |
| 1,527 tokens | 96.7% | 96.2% | 24 vs 1,527 (1.6%) | 36 KB vs 2,291 KB | 0.93 vs 1.54 GF | 48 vs 88 ms |
| 2,704 tokens | 92.7% | 91.9% | 24 vs 2,704 (0.9%) | 37 KB vs 4,056 KB | 2.36 vs 3.45 GF | 64 vs 155 ms |

Read the memory column downwards. The document grows 5x; the notebook does not move. Accuracy stays
within 0.8-2.0 points of reading everything, and time per task grows 1.7x for notes against 3.4x for
full context. A workload here is one document plus 36 questions, which is what an agent re-querying
a document looks like.

## 2. The note format beats the alternatives at the same budget

Give every method the same 72 tokens of memory and ask the same questions (3 seeds):

| Memory format | Accuracy |
|---|---:|
| **Anchor + micro-context notes** | **99.4%** |
| RAG top-k (retrieval, query-aware) | 44.1% |
| Extractive summary | 20.5% |
| Sliding window (most recent text) | 18.3% |

And both halves of a note are necessary: keywords alone score 12.3% (chance is 12.5%), context
without its keyword 26.9%, the two together 99.4%.

## 3. The notes are learned, not hand-written

The strongest version writes its own notes with no examples to copy: it is trained only to answer
correctly from its notes, keep them few, behave like full context, and avoid rereading. Under those
pressures it reaches **100% accuracy from notes, equal to reading the full text, on 7% of the
memory**. The notes are inspectable text with a pointer back to the source sentence.

## 4. What is not solved

- **Writing notes about real prose.** On real documents, notes written by hand answer 100% of
  questions from 4-10% of the text - the format is sufficient. The trained note-writer reaches only
  **82%**. Going from 8k to 225k training examples on a GPU moved this by 8 points and then
  flattened, with training loss at 0.0000: it is a generalisation limit, not a compute limit. This
  is the main open problem and the next thing to fix.
- **Scale.** Nothing has been tested above 2,705 tokens or 445k parameters. Claims at 128k context
  or 7B parameters would be unfounded.
- **Reference models.** A 7B model in 4-bit clears our quality bar on single-fact (84%) and negation
  (92%) tasks but not on two-fact reasoning (54%), so comparisons at that scale are only meaningful
  on some tasks today.
- **Workload dependence.** Notes win when a document is asked many questions. For one question per
  document, ordinary retrieval is cheaper, as our own ablation shows.

## 5. Where the value would be, if it holds at scale

Cost per question stops scaling with how much the system has read. That matters for agents with long
histories, codebases, customer records and case files - anywhere the same material is queried
repeatedly. The memory is also plain text: inspectable, correctable, versionable and deletable,
which a conventional attention cache is not.

## 6. What it would take to find out

| Stage | Question | Cost |
|---|---|---|
| Fix the note-writer | Can it reach 95% on real prose with a stronger encoder? | ~$50 or free-tier GPU |
| 7B integration | Does a competent model do better with notes than with full context, RAG or summaries? | ~$100k, 2-3 months |
| Native training | Does the advantage survive in a model trained with this memory from the start? | $1M+ |

Each stage has a pre-agreed pass mark, and the work stops if one fails. Two have already been
stopped this way: a comparison under a model too weak to do the task, and a compiler that missed its
retention gate.

---

*Reproduce anything here: `scripts/run_all.sh`. Failed and abandoned runs, with their diagnoses, are
kept in `results/logs/FAILED_RUNS.md`.*
