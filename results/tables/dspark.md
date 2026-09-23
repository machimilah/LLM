# DSpark benchmark

Seeds [2], draft block 3, 3 generated tokens.

| Seed | Output identical to greedy | Accepted/step | Acceptance rate | Tokens/s plain | Tokens/s speculative | Speedup |
|---|:--:|---:|---:|---:|---:|---:|
| 2 | yes | 0.50 | 0.6% | 91 | 42 | 0.46x |

**Correctness is the primary result:** identical output means throughput work
cannot change what the model says.

This benchmark generates 1-3 token chains, so there is little to speculate about and
no speedup should be expected here - the draft costs a pass that the short generation
cannot amortise. Acceptance rate is the number that transfers to long generations.
