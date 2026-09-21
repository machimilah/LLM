# SN-CED implementation notes

> Two implementations live here. **v1** (`models/snced.py`) is the benchmark-fitted prototype that
> produced the published results: closed-set compiler heads and a symbolic chain-walking router.
> **v2, the general architecture** (`models/snced_general.py` with `note_compiler.py`,
> `note_memory.py`, `sufficiency_router.py`) is the one that matches the plan's intent:
> open-vocabulary pointer notes, a learned sufficiency router with an access price, and no gold
> note labels. v1 is kept so earlier results stay reproducible.

What is actually built, why it is built that way, and what each decision cost.
The research plan is [../SN_CED_ARCHITECTURE_AND_TEST_PLAN.md](../SN_CED_ARCHITECTURE_AND_TEST_PLAN.md);
this file records the implementation that resulted.

## 1. Synthetic native architecture (`src/snced/models/`)

```
lecture tokens
   |
   v
causal encoder  (RoPE self-attention + short causal depthwise conv, kernel 6)
   |                                   |
   | final states                      | final states per sentence
   v                                   v
mem_proj -> detailed memory       note compiler (attention pooling + heads)
   |                                   |
   |                                   v  anchor / relation / target / confidence
   |                              note encoder: one memory slot per note
   |                                   |
   +------------> router <-------------+
                    |
                    v
      decoder (causal self-attn + cross-attn), generates the reasoning chain
```

### Decisions that mattered

| Decision | Why | Evidence |
|---|---|---|
| Short causal conv in encoder blocks | Attention-only models never learned entity->value binding; they sat at the "answer any value in context" plateau for thousands of steps | FAILED_RUNS.md, attempts 1-3 |
| Multi-query training (one lecture, all entities asked) | Gives contrasting targets in the same context, which breaks the plateau | 12 entities learned *faster* than 2 |
| Generative chain decoding (`e5 -> e9 -> v3`) | Each hop becomes a single lookup, which the model already does perfectly; one-shot classification never learned two hops | attempt 3 stuck at 30% one-hop, attempt 4 reached 100% |
| One memory slot per note (dedicated note encoder) | Re-encoding packed note text with the lecture encoder produced unusable memory | 32% vs 99.4% |
| Compiler trained on mixed context lengths | Fixed-length training made extraction decay on longer documents | 88.8% -> 95.8% at 1528 tokens |
| Question-level routing, threshold 0.5 | Lecture-level routing spent ~996 memory slots for +1.4 points | results/tables/router_calibration.md |

## 2. Real-text compilers (`experiments/real/`)

- `train_note_compiler.py` - closed-vocabulary heads over a frozen MiniLM. 88/90/92% retention on
  BABILong qa1/qa2/qa9 at 4-10% of the document.
- `span_compiler.py` - open-vocabulary BIO span tagger (ANCHOR / RELATION / TARGET) plus fact and
  negation heads. 82/73/82% on the familiar cast, 51-66% on held-out entity strings. `--finetune-encoder`
  additionally trains the encoder.
- Notes carry a source pointer (sentence index) and keep document order; several notes may share an
  anchor, and negation is a distinct relation.

The deterministic reader in `notes_data.answer_from_notes` replays notes in order to answer a
question without any LLM. Gold notes score 100% on all three tasks, so it measures the notebook,
not the reader.

## 3. What is measured

Every run writes `results/raw/<experiment>/*.json` with the schema in plan section 28: accuracy,
note tokens and count, memory slots, KV bytes, estimated FLOPs, fallback rate, timings, seed and
git commit. `peak_gpu_memory_mb` is populated automatically when CUDA is present.

FLOPs are analytic (`snced/metrics.py`): encoder, per-lecture memory projection, and per-question
decoder cost, with query-conditioned memories charged once per question and query-independent
memories once per document.

## 3b. General architecture (v2)

```
input -> ONE causal encoder pass
           |-> detailed memory (projected states, one slot per token)
           |-> note compiler: per segment, soft pointers into the source pick an
           |                  anchor and a k-head micro-context, plus importance,
           |                  confidence, supersession, order and source pointer
           |                          |
           |                          v  note encoder (separate module)
           |                     one memory slot per note
           v                          |
        learned sufficiency router <--+   sees the query and the notebook
                    |                     outputs P(need detailed memory)
                    v
             decoder reads SNM, plus detailed memory only when the router asks
```

Composite loss (plan section 10), with no gold note labels anywhere:

```
L = L_task(notes)                     reason from notes by default
  + route       * L_task(routed)      the path actually served
  + sufficiency * KL(detail || notes) notes should behave like full context
  + notes       * E[note count]       compression pressure
  + detail      * E[detailed slots]   price of touching detailed memory
```

Pointers are soft (attention over segment states) so gradients reach the compiler; their argmax
recovers the note as text, which is what `OpenVocabNoteCompiler.readable` prints. Segmentation is
by boundary token or fixed window, so the model does not assume sentence structure.

## 4. Known gaps

- The synthetic vocabulary is word-level and closed, so surface confusability and exact quotations
  cannot be expressed; those failure cases need the real-text setting.
- `exact_value` in the adversarial suite is inconclusive: the decoder was trained to emit one value
  token, so full context fails too (Rule 2).
- Learned routing with an access cost exists in v2 only; v1 and every published result so far used
  the rule-based router.
- v2's note consolidation and eviction (plan section 21) and versioned notes for "what was true
  last week" queries (section 22) are still absent; order and a supersession score are carried in
  the note, but nothing merges or retires notes yet.
- No GPU measurements: KV bytes are computed analytically, and wall-clock timings were taken on a
  shared CPU.
