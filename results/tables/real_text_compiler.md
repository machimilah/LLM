# Trained Semantic Note Compiler on real text (BABILong 1k)

Compiler: frozen MiniLM-L6 sentence encoder (22M, not fine-tuned) + trained heads (0.2M).
Query-independent: it reads the document and freezes its notes before any question exists.
Trained on bAbI stories embedded in **WikiText** noise; evaluated on **BABILong** documents,
whose noise is book text from a different corpus (PG19).

A note is (anchor, relation, target, source sentence index). Notes keep document order and are
never collapsed, so later events override earlier ones, an anchor can carry several relations,
and negation is its own relation.

## 1. Retention: does the notebook still support the task? (deterministic reader, no LLM, n=100)

| Task | Trained compiler | Gold notes | Fact precision | Fact recall | Notes vs document |
|---|---:|---:|---:|---:|---:|
| qa1 (one supporting fact) | 88.0% | 100.0% | 83.1% | 96.0% | 20 / 485 tok (4.2%) |
| qa2 (two supporting facts, order) | 90.0% | 100.0% | 93.3% | 97.1% | 49 / 495 tok (10.0%) |
| qa9 (negation) | 92.0% | 100.0% | 83.3% | 93.4% | 22 / 479 tok (4.6%) |

Gold notes answer every question, so the note schema is sufficient for real-text relational
structure; the 8-12 point gap is compiler error, mostly false positives from prose.

## 2. QA with the same model (Qwen2.5-0.5B-Instruct, greedy, n=25 per cell, +-~19pp at 95%)

| Memory | qa1 | qa2 | Prompt tokens | Memory build |
|---|---:|---:|---:|---:|
| Full context | 48.0% | 36.0% | 710 | - |
| RAG top-k (query-conditioned) | 48.0% | 16.0% | 268 | - |
| LLM summary | 20.0% | 16.0% | 153 | 16-18 s |
| LLM-prompted notes | 36.0% | 4.0% | 96 | 7-11 s |
| **Trained compiler notes** | **52.0%** | **32.0%** | **96** | **0.16-0.18 s** |

qa9 is omitted here: the model answers with sentences rather than yes/no, so word-containment
scoring is unreliable for that task in both directions.

## 3. Reading

- The trained compiler beats the prompted LLM compiler by a wide margin (52 vs 36, 32 vs 4) and
  builds its memory ~50x faster, while using the same number of prompt tokens.
- It matches or beats full context on qa1 with 7.4x fewer prompt tokens, and stays within 4 points
  on qa2.
- The QA model, not the memory, is now the bottleneck: the same notes answer 88-90% under a
  deterministic reader but only 52%/32% through the 0.5B model, whose full-context score is itself
  only 48%/36%. Rule 2 still applies - this QA comparison sits on a weak baseline.

## Limits

- bAbI's closed vocabulary (4 people, 9 places/objects) is used for the gold labels and for the
  compiler's output heads. A general compiler needs span extraction, not classification.
- One seed, n=25 per QA cell, single context length (1k), three tasks.
- Task-aware prompts were used for the summary and prompted-notes arms (documented in the script).
