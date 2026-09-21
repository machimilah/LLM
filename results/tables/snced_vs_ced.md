# CED vs SN-CED: controlled comparison (synthetic lectures)

Seeds: 1, 2, 3. Values are mean ± std across seeds.
Trained at 18 filler chunks; 54 and 108 are longer, unseen contexts.
Active memory = decoder cross-attention slots per lecture. KV = fp32 K and V for all decoder layers.
GFLOPs = analytic estimate per lecture for 36 questions, including the full-lecture encoder pass
for every condition and note re-encoding for note conditions (Rules 4 and 5).

Caveats:
- The note compiler is trained with gold chunk labels, supervision the baseline does not get.
- Wall-clock timings were measured on a shared CPU with other runs in flight and include one
  outlier of ~23.7 s/question (SN-CED full memory, 939 tokens, seed 1). Use the FLOPs column;
  treat decode-ms means, especially their standard deviations, as unreliable.
- Notes-only degrades at unseen context lengths while gold notes do not: the compiler, not the
  note representation, is what fails to generalize.

## 18 filler chunks (~547 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 100.00 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 547.5 ± 0.9 | 100.0% | 840,911 | 0.490 ± 0.001 | 2.50 ± 0.13 | 0.0 ± 0.0% |
| SN-CED, full memory | 100.00 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 547.5 ± 0.9 | 100.0% | 840,911 | 0.490 ± 0.001 | 2.32 ± 0.12 | 0.0 ± 0.0% |
| SN-CED, notes only | 99.42 ± 0.58% | 99.5 ± 0.7 | 99.2 ± 0.9 | 99.5 ± 0.5 | 24.0 ± 0.0 | 4.4% | 36,866 | 0.279 ± 0.000 | 1.37 ± 0.10 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.86 ± 0.17% | 99.9 ± 0.1 | 99.8 ± 0.3 | 99.9 ± 0.1 | 24.0 ± 0.0 | 4.4% | 36,864 | 0.279 ± 0.000 | 1.39 ± 0.14 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 99.92 ± 0.14% | 99.9 ± 0.1 | 99.8 ± 0.3 | 100.0 ± 0.0 | 223.5 ± 156.9 | 40.8% | 343,324 | 0.360 ± 0.064 | 2.31 ± 0.25 | 38.2 ± 30.0% |

Compiler exact chunk extraction: 99.79 ± 0.23%.

## 54 filler chunks (~939 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 99.86 ± 0.13% | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.6 ± 0.4 | 939.2 ± 1.7 | 100.0% | 1,442,539 | 0.851 ± 0.002 | 3.83 ± 0.62 | 0.0 ± 0.0% |
| SN-CED, full memory | 99.81 ± 0.17% | 100.0 ± 0.0 | 99.8 ± 0.1 | 99.6 ± 0.4 | 939.2 ± 1.7 | 100.0% | 1,442,539 | 0.851 ± 0.002 | 7890.44 ± 13660.87 | 0.0 ± 0.0% |
| SN-CED, notes only | 97.39 ± 3.82% | 98.0 ± 3.5 | 98.0 ± 2.8 | 96.2 ± 5.2 | 24.1 ± 0.2 | 2.6% | 37,084 | 0.480 ± 0.001 | 1.96 ± 0.59 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.94 ± 0.05% | 100.0 ± 0.0 | 99.9 ± 0.1 | 99.9 ± 0.1 | 24.0 ± 0.0 | 2.6% | 36,864 | 0.480 ± 0.001 | 1.78 ± 0.57 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 99.75 ± 0.25% | 100.0 ± 0.0 | 99.8 ± 0.3 | 99.4 ± 0.5 | 499.5 ± 398.5 | 53.2% | 767,260 | 0.674 ± 0.162 | 3.38 ± 0.90 | 52.0 ± 43.6% |

Compiler exact chunk extraction: 99.31 ± 0.95%.

## 108 filler chunks (~1528 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 98.17 ± 2.69% | 100.0 ± 0.0 | 98.1 ± 2.7 | 96.4 ± 5.4 | 1527.8 ± 2.3 | 100.0% | 2,346,754 | 1.542 ± 0.003 | 4.56 ± 0.35 | 0.0 ± 0.0% |
| SN-CED, full memory | 98.06 ± 2.74% | 100.0 ± 0.0 | 97.8 ± 2.9 | 96.3 ± 5.3 | 1527.8 ± 2.3 | 100.0% | 2,346,754 | 1.542 ± 0.003 | 4.36 ± 0.46 | 0.0 ± 0.0% |
| SN-CED, notes only | 93.86 ± 5.67% | 97.1 ± 4.2 | 94.2 ± 5.5 | 90.3 ± 8.4 | 24.4 ± 0.6 | 1.6% | 37,545 | 0.930 ± 0.002 | 1.74 ± 0.35 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.89 ± 0.10% | 99.8 ± 0.1 | 99.9 ± 0.1 | 99.9 ± 0.1 | 24.0 ± 0.0 | 1.6% | 36,864 | 0.930 ± 0.002 | 1.51 ± 0.18 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 97.94 ± 2.80% | 99.9 ± 0.1 | 97.8 ± 3.1 | 96.2 ± 5.4 | 963.5 ± 800.7 | 63.1% | 1,479,899 | 1.314 ± 0.326 | 4.25 ± 0.54 | 62.5 ± 53.3% |

Compiler exact chunk extraction: 99.09 ± 0.92%.

