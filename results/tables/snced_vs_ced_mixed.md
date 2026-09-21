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
| CED baseline (full memory) | 100.00 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 547.5 ± 0.9 | 100.0% | 840,911 | 0.490 ± 0.001 | 1.78 ± 0.53 | 0.0 ± 0.0% |
| SN-CED, full memory | 99.97 ± 0.05% | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.9 ± 0.1 | 547.5 ± 0.9 | 100.0% | 840,911 | 0.490 ± 0.001 | 1.68 ± 0.48 | 0.0 ± 0.0% |
| SN-CED, notes only | 98.39 ± 1.55% | 98.9 ± 1.2 | 98.8 ± 1.4 | 97.5 ± 2.1 | 24.0 ± 0.0 | 4.4% | 36,864 | 0.279 ± 0.000 | 1.00 ± 0.18 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.42 ± 0.38% | 99.4 ± 0.3 | 99.7 ± 0.6 | 99.2 ± 0.5 | 24.0 ± 0.0 | 4.4% | 36,864 | 0.279 ± 0.000 | 0.98 ± 0.18 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 99.92 ± 0.08% | 99.9 ± 0.1 | 99.9 ± 0.1 | 99.9 ± 0.1 | 340.5 ± 236.7 | 62.2% | 523,013 | 0.408 ± 0.096 | 1.63 ± 0.42 | 60.5 ± 45.3% |

Compiler exact chunk extraction: 99.50 ± 0.52%.

## 54 filler chunks (~939 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 99.86 ± 0.13% | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.6 ± 0.4 | 939.2 ± 1.7 | 100.0% | 1,442,539 | 0.851 ± 0.002 | 2.14 ± 0.53 | 0.0 ± 0.0% |
| SN-CED, full memory | 99.78 ± 0.21% | 100.0 ± 0.0 | 99.8 ± 0.1 | 99.5 ± 0.5 | 939.2 ± 1.7 | 100.0% | 1,442,539 | 0.851 ± 0.002 | 2.07 ± 0.51 | 0.0 ± 0.0% |
| SN-CED, notes only | 97.61 ± 1.86% | 98.5 ± 1.4 | 97.7 ± 1.9 | 96.7 ± 2.6 | 24.0 ± 0.0 | 2.6% | 36,887 | 0.480 ± 0.001 | 0.99 ± 0.16 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.50 ± 0.72% | 99.8 ± 0.4 | 99.3 ± 1.2 | 99.4 ± 0.6 | 24.0 ± 0.0 | 2.6% | 36,864 | 0.480 ± 0.001 | 0.98 ± 0.17 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 99.75 ± 0.17% | 100.0 ± 0.0 | 99.8 ± 0.1 | 99.4 ± 0.4 | 589.3 ± 446.7 | 62.7% | 905,180 | 0.711 ± 0.182 | 2.11 ± 0.55 | 61.8 ± 48.8% |

Compiler exact chunk extraction: 99.55 ± 0.38%.

## 108 filler chunks (~1528 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 98.17 ± 2.69% | 100.0 ± 0.0 | 98.1 ± 2.7 | 96.4 ± 5.4 | 1527.8 ± 2.3 | 100.0% | 2,346,754 | 1.542 ± 0.003 | 3.23 ± 0.87 | 0.0 ± 0.0% |
| SN-CED, full memory | 98.17 ± 2.75% | 100.0 ± 0.0 | 98.2 ± 2.8 | 96.3 ± 5.5 | 1527.8 ± 2.3 | 100.0% | 2,346,754 | 1.542 ± 0.003 | 3.15 ± 0.94 | 0.0 ± 0.0% |
| SN-CED, notes only | 96.11 ± 3.76% | 97.4 ± 2.5 | 96.3 ± 3.6 | 94.6 ± 5.1 | 24.1 ± 0.1 | 1.6% | 37,005 | 0.930 ± 0.002 | 1.02 ± 0.20 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.58 ± 0.52% | 99.7 ± 0.6 | 99.5 ± 0.9 | 99.6 ± 0.4 | 24.0 ± 0.0 | 1.6% | 36,864 | 0.930 ± 0.002 | 1.00 ± 0.19 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 98.17 ± 2.69% | 100.0 ± 0.0 | 98.2 ± 2.8 | 96.3 ± 5.3 | 1006.2 ± 735.3 | 65.9% | 1,545,515 | 1.331 ± 0.299 | 2.85 ± 0.73 | 65.3 ± 48.9% |

Compiler exact chunk extraction: 99.48 ± 0.53%.

## 216 filler chunks (~2704 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 93.97 ± 9.59% | 100.0 ± 0.0 | 93.6 ± 10.7 | 88.3 ± 18.1 | 2705.4 ± 0.9 | 100.0% | 4,155,463 | 3.455 ± 0.002 | 4.65 ± 0.44 | 0.0 ± 0.0% |
| SN-CED, full memory | 93.39 ± 9.97% | 100.0 ± 0.0 | 92.2 ± 11.9 | 88.0 ± 18.0 | 2705.4 ± 0.9 | 100.0% | 4,155,463 | 3.455 ± 0.002 | 4.43 ± 1.07 | 0.0 ± 0.0% |
| SN-CED, notes only | 93.89 ± 7.26% | 97.4 ± 3.4 | 92.3 ± 9.4 | 91.9 ± 9.0 | 24.4 ± 0.6 | 0.9% | 37,409 | 2.362 ± 0.002 | 1.01 ± 0.19 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 99.67 ± 0.30% | 99.8 ± 0.4 | 99.7 ± 0.1 | 99.6 ± 0.5 | 24.0 ± 0.0 | 0.9% | 36,864 | 2.362 ± 0.001 | 1.04 ± 0.23 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 93.36 ± 9.95% | 100.0 ± 0.0 | 92.1 ± 11.8 | 88.0 ± 18.0 | 1816.5 ± 1314.8 | 67.1% | 2,790,069 | 3.094 ± 0.538 | 4.60 ± 1.31 | 66.8 ± 49.0% |

Compiler exact chunk extraction: 99.44 ± 0.73%.

