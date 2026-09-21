# Timing and cost per successful task

Seeds: [1, 2, 3]. One condition at a time on an otherwise idle machine, after warm-up.
A task = one document + 36 questions, so memory building is amortised.
Device: cpu.

Caveat: the fallback condition builds BOTH memories every time in this implementation, so its
wall-clock is pessimistic; a real system would encode the full context only for the questions that
actually fall back. FLOPs are analytic and do not have this problem.

## 18 filler chunks

| Condition | Accuracy | Slots | Build ms/doc | Answer ms/q | Task ms | ms / successful task | GFLOPs / successful task |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_context | 99.91% | 546.8 | 2.5 | 1.44 | 54 | 55 | 0.490 |
| snm_notes | 98.52% | 24.0 | 6.2 | 0.92 | 39 | 40 | 0.283 |
| snm_fallback | 98.89% | 30.8 | 5.7 | 1.48 | 59 | 60 | 0.285 |

## 108 filler chunks

| Condition | Accuracy | Slots | Build ms/doc | Answer ms/q | Task ms | ms / successful task | GFLOPs / successful task |
|---|---:|---:|---:|---:|---:|---:|---:|
| full_context | 97.78% | 1527.8 | 9.5 | 2.74 | 108 | 111 | 1.578 |
| snm_notes | 95.93% | 24.1 | 17.5 | 0.94 | 52 | 54 | 0.970 |
| snm_fallback | 96.76% | 85.3 | 17.8 | 2.45 | 106 | 110 | 0.995 |

