# SN-CED general architecture (v2): results

Seeds: 2. Context ~448 tokens.

The compiler writes open-vocabulary notes by pointing into the source, the router is
learned, and **no gold note labels are used anywhere**: the notebook is shaped only by the
task loss, a sufficiency term against the full-context path, an L0 penalty on the number of
notes, and a price on rereading.

| Path | Accuracy | Memory slots | Notes kept | Router fired |
|---|---:|---:|---:|---:|
| Full context (reference) | 98.10 ± 5.04% | 437.2 | 31.7 | 0.0% |
| Semantic Note Memory only | 97.38 ± 6.93% | 31.7 | 31.7 | 0.0% |
| Router's choice | 97.38 ± 6.93% | 31.7 | 31.7 | 0.0% |

Note memory is 7.1% of the context length.

Sample of the notebook, as text (the pointers' argmax):

| anchor | micro-context | source | order | gate |
|---|---|---:|---:|---:|
| value | v2 v2 | 0 | 0 | 1.00 |
| lecture | v6 v6 | 14 | 1 | 1.00 |
| the | e12 e12 | 31 | 2 | 1.00 |
| speaker | e7 e7 | 45 | 3 | 1.00 |
| the | of of | 61 | 4 | 1.00 |
| than | of lower | 72 | 5 | 1.00 |

## Caveats

- Warm-started from a trained CED backbone with the encoder frozen (plan Phase 2).
  Training all of it from scratch at once left both paths near 30%.
- The router never had to fire: notes were sufficient, so its value is untested here.
- Synthetic benchmark only; the real-text compiler is a separate, weaker result
  (results/tables/span_compiler.md).
