# CED vs SN-CED: controlled comparison (synthetic lectures)

Seeds: 1. Values are mean ± std across seeds.
Trained at 18 filler chunks; 54 and 108 are longer, unseen contexts.
Active memory = decoder cross-attention slots per lecture. KV = fp32 K and V for all decoder layers.
GFLOPs = analytic estimate per lecture for 36 questions, including the full-lecture encoder pass
for every condition and note re-encoding for note conditions (Rules 4 and 5).

Caveat: the note compiler is trained with gold chunk labels, supervision the baseline does not get.

## 18 filler chunks (~547 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 100.00 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 547.8 ± 0.0 | 100.0% | 841,482 | 0.491 ± 0.000 | 5.86 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, full memory | 99.92 ± 0.00% | 100.0 ± 0.0 | 99.8 ± 0.0 | 100.0 ± 0.0 | 547.8 ± 0.0 | 100.0% | 841,482 | 0.491 ± 0.000 | 5.77 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, notes only | 32.25 ± 0.00% | 53.2 ± 0.0 | 21.8 ± 0.0 | 21.8 ± 0.0 | 72.0 ± 0.0 | 13.1% | 110,592 | 0.307 ± 0.000 | 4.39 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 32.33 ± 0.00% | 53.2 ± 0.0 | 22.0 ± 0.0 | 21.8 ± 0.0 | 72.0 ± 0.0 | 13.1% | 110,592 | 0.307 ± 0.000 | 4.65 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 73.50 ± 0.00% | 82.5 ± 0.0 | 68.5 ± 0.0 | 69.5 ± 0.0 | 350.6 ± 0.0 | 64.0% | 538,467 | 0.421 ± 0.000 | 6.26 ± 0.00 | 58.5 ± 0.0% |

Compiler exact chunk extraction: 99.73 ± 0.00%.

## 54 filler chunks (~939 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 99.83 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.5 ± 0.0 | 939.2 ± 0.0 | 100.0% | 1,442,618 | 0.851 ± 0.000 | 6.72 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, full memory | 99.42 ± 0.00% | 100.0 ± 0.0 | 99.5 ± 0.0 | 98.8 ± 0.0 | 939.2 ± 0.0 | 100.0% | 1,442,618 | 0.851 ± 0.000 | 6.71 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, notes only | 30.83 ± 0.00% | 46.2 ± 0.0 | 28.2 ± 0.0 | 18.0 ± 0.0 | 72.9 ± 0.0 | 7.8% | 111,951 | 0.509 ± 0.000 | 3.70 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 32.33 ± 0.00% | 48.2 ± 0.0 | 29.8 ± 0.0 | 19.0 ± 0.0 | 72.0 ± 0.0 | 7.7% | 110,592 | 0.508 ± 0.000 | 3.69 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 83.92 ± 0.00% | 88.5 ± 0.0 | 83.0 ± 0.0 | 80.2 ± 0.0 | 739.0 ± 0.0 | 78.7% | 1,135,104 | 0.781 ± 0.000 | 6.18 ± 0.00 | 77.0 ± 0.0% |

Compiler exact chunk extraction: 98.56 ± 0.00%.

## 108 filler chunks (~1528 context tokens)

| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CED baseline (full memory) | 99.42 ± 0.00% | 100.0 ± 0.0 | 99.2 ± 0.0 | 99.0 ± 0.0 | 1528.8 ± 0.0 | 100.0% | 2,348,236 | 1.543 ± 0.000 | 7.62 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, full memory | 99.08 ± 0.00% | 99.8 ± 0.0 | 98.8 ± 0.0 | 98.8 ± 0.0 | 1528.8 ± 0.0 | 100.0% | 2,348,236 | 1.543 ± 0.000 | 8.16 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, notes only | 30.92 ± 0.00% | 48.2 ± 0.0 | 22.0 ± 0.0 | 22.5 ± 0.0 | 74.2 ± 0.0 | 4.9% | 114,001 | 0.960 ± 0.000 | 4.28 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, gold notes (upper bound) | 30.42 ± 0.00% | 48.5 ± 0.0 | 21.8 ± 0.0 | 21.0 ± 0.0 | 72.0 ± 0.0 | 4.7% | 110,592 | 0.959 ± 0.000 | 4.12 ± 0.00 | 0.0 ± 0.0% |
| SN-CED, notes + fallback | 92.00 ± 0.00% | 95.0 ± 0.0 | 90.2 ± 0.0 | 90.8 ± 0.0 | 1361.9 ± 0.0 | 89.1% | 2,091,886 | 1.486 ± 0.000 | 8.58 ± 0.00 | 88.5 ± 0.0% |

Compiler exact chunk extraction: 98.58 ± 0.00%.

