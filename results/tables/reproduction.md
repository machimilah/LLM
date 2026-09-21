# Reproduction report: historical SNM experiments

The original prototype scripts were not available, so both experiments were
reimplemented from the plan's description (src/snced/data.py documents the
reconstruction). Exact token counts depend on template wording and are not
expected to match; the acceptance criteria concern the qualitative pattern.

Caveat: the reconstruction appears slightly easier than the original. The
compiler reaches 100% extraction on every seed (original seed 33: 99.94%),
and notes/questions are shorter, so the SNM-vs-RAG active-token gap is larger
than recorded in the plan. Do not read the larger gap as an improvement.

## Experiment A: representation and fallback ablation

| Condition | Seeds | Accuracy (new) | Accuracy (plan) | Active tokens/q (new) | Active tokens/q (plan) | Fallback rate | Exposure, 100 q |
|---|---:|---:|---:|---:|---:|---:|---:|
| iterative_rag | 123, 456, 789 | 100.00% ± 0.00 | 100.00% | 40.23 ± 0.02 | 43.48 | 0.00% ± 0.00 | 4570 ± 1 |
| snm | 123, 456, 789 | 100.00% ± 0.00 | 100.00% | 17.00 ± 0.00 | 24.41 | 0.00% ± 0.00 | 2247 ± 1 |
| snm_noisy | 123, 456, 789 | 80.76% ± 2.21 | 82.22% | 16.58 ± 0.05 | 23.99 | 0.00% ± 0.00 | 2205 ± 5 |
| snm_fallback | 123, 456, 789 | 100.00% ± 0.00 | 100.00% | 19.53 ± 0.27 | 27.70 | 21.04% ± 2.37 | 2500 ± 27 |
| word_only | 123, 456, 789 | 11.74% ± 1.68 | 12.15% | 13.00 ± 0.00 | 16.36 | 0.00% ± 0.00 | 1847 ± 1 |

SNM uses 57.7% fewer active tokens than iterative RAG (plan: 43.9%).

## Experiment B: learned query-independent note compiler

Seeds: 11, 22, 33. Compiler parameters: 22,823 (plan: 22,923). Reader: deterministic graph traversal over predicted notes, not an LLM decoder.

| Metric | New (mean ± std) | Plan |
|---|---:|---:|
| raw_tokens | 546.97 ± 0.24 | 573.44 |
| note_tokens | 72.00 ± 0.00 | 71.98 |
| raw_slots | 42.00 ± 0.00 | 42.00 |
| note_slots | 24.00 ± 0.00 | 23.99 |
| text_token_reduction | 86.84% ± 0.01 | 87.45% |
| slot_reduction | 42.86% ± 0.00 | 42.87% |
| direct | 100.00% ± 0.00 | 100.00% |
| one_hop | 100.00% ± 0.00 | 99.79% |
| two_hop | 100.00% ± 0.00 | 100.00% |
| overall | 100.00% ± 0.00 | 99.93% |

Exact chunk extraction: seed 11: 100.00%, seed 22: 100.00%, seed 33: 100.00%.

## Phase 0 acceptance criteria

| Criterion | Result |
|---|---|
| A: SNM accuracy near 100% | PASS |
| A: word-only accuracy near chance (12.5%) | PASS |
| B: textual compression near 87% | PASS |
| B: information recovery near 99.9% | PASS |
