# Memory-representation ablations

Seeds: 1, 2, 3. Mean ± std. Every condition uses the same encoder,
decoder and evaluation harness; only the decoder's memory differs.
Budget conditions keep 72 tokens, which is what SNM's 24 notes cost in text.
GFLOPs = analytic estimate per lecture for 36 questions, including building the memory
(the full-lecture pass needed for compilation or fact selection is charged to those conditions).
Query-conditioned memory (RAG) is rebuilt for every question; query-independent memory is built
once per lecture and reused across the 36 questions.

Keyword-only and phrase-only each got their own short fine-tune, so they are not penalised
for being unfamiliar to the decoder.

## 18 filler chunks (~547 context tokens)

| Memory | Query-independent | Accuracy | Direct | One-hop | Two-hop | Slots | Tokens kept | GFLOPs/lecture |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|
| Full context | yes | 100.00 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 547.5 ± 0.9 | 547.5 ± 0.9 | 0.507 ± 0.001 |
| Extractive summary (fact sentences) | yes | 100.00 ± 0.00% | 100.0 ± 0.0 | 100.0 ± 0.0 | 100.0 ± 0.0 | 351.2 ± 0.4 | 351.2 ± 0.4 | 0.501 ± 0.001 |
| Iterative RAG (query-conditioned) | no | 99.97 ± 0.05% | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.9 ± 0.1 | 185.2 ± 1.5 | 185.2 ± 1.5 | 1.722 ± 0.016 |
| SNM + source fallback | yes | 99.92 ± 0.14% | 99.9 ± 0.1 | 99.8 ± 0.3 | 100.0 ± 0.0 | 223.5 ± 156.9 | 253.2 ± 142.5 | 0.391 ± 0.082 |
| SNM: anchor + micro-context | yes | 99.42 ± 0.58% | 99.5 ± 0.7 | 99.2 ± 0.9 | 99.5 ± 0.5 | 24.0 ± 0.0 | 72.0 ± 0.0 | 0.286 ± 0.000 |
| RAG top-k, 72-token budget (query-conditioned) | no | 44.08 ± 1.54% | 98.9 ± 0.1 | 13.4 ± 2.3 | 19.9 ± 4.0 | 44.3 ± 0.1 | 44.3 ± 0.1 | 0.449 ± 0.001 |
| Extractive summary, 72-token budget | yes | 20.50 ± 0.87% | 28.1 ± 1.6 | 17.4 ± 1.4 | 16.0 ± 0.4 | 64.4 ± 0.2 | 64.4 ± 0.2 | 0.310 ± 0.000 |
| Sliding window, 72 tokens | yes | 18.28 ± 1.40% | 23.4 ± 2.4 | 16.1 ± 1.4 | 15.3 ± 0.8 | 72.0 ± 0.0 | 72.0 ± 0.0 | 0.165 ± 0.000 |
| SNM: phrase only (no anchor) | yes | 26.89 ± 0.51% | 25.7 ± 2.1 | 27.2 ± 1.3 | 27.8 ± 1.7 | 24.0 ± 0.0 | 72.0 ± 0.0 | 0.286 ± 0.000 |
| SNM: keyword (anchor) only | yes | 12.31 ± 0.53% | 11.8 ± 1.4 | 11.8 ± 0.4 | 13.2 ± 1.7 | 24.0 ± 0.0 | 72.0 ± 0.0 | 0.286 ± 0.000 |

## 108 filler chunks (~1528 context tokens)

| Memory | Query-independent | Accuracy | Direct | One-hop | Two-hop | Slots | Tokens kept | GFLOPs/lecture |
|---|:--:|---:|---:|---:|---:|---:|---:|---:|
| Full context | yes | 98.06 ± 2.74% | 100.0 ± 0.0 | 97.8 ± 2.9 | 96.3 ± 5.3 | 1527.8 ± 2.3 | 1527.8 ± 2.3 | 1.578 ± 0.003 |
| Extractive summary (fact sentences) | yes | 99.97 ± 0.05% | 100.0 ± 0.0 | 99.9 ± 0.1 | 100.0 ± 0.0 | 356.3 ± 6.5 | 356.3 ± 6.5 | 1.156 ± 0.005 |
| Iterative RAG (query-conditioned) | no | 99.97 ± 0.05% | 100.0 ± 0.0 | 100.0 ± 0.0 | 99.9 ± 0.1 | 228.1 ± 0.8 | 228.1 ± 0.8 | 2.182 ± 0.009 |
| SNM + source fallback | yes | 97.94 ± 2.80% | 99.9 ± 0.1 | 97.8 ± 3.1 | 96.2 ± 5.4 | 963.5 ± 800.7 | 981.5 ± 775.2 | 1.431 ± 0.420 |
| SNM: anchor + micro-context | yes | 93.86 ± 5.67% | 97.1 ± 4.2 | 94.2 ± 5.5 | 90.3 ± 8.4 | 24.4 ± 0.6 | 73.3 ± 1.7 | 0.936 ± 0.002 |
| RAG top-k, 72-token budget (query-conditioned) | no | 42.72 ± 0.95% | 97.3 ± 0.8 | 11.4 ± 0.1 | 19.4 ± 2.1 | 50.7 ± 0.2 | 50.7 ± 0.2 | 0.499 ± 0.002 |
| Extractive summary, 72-token budget | yes | 21.36 ± 1.99% | 31.4 ± 2.9 | 18.1 ± 2.4 | 14.6 ± 1.2 | 64.6 ± 0.2 | 64.6 ± 0.2 | 0.961 ± 0.002 |
| Sliding window, 72 tokens | yes | 13.22 ± 1.12% | 15.2 ± 1.8 | 12.9 ± 1.4 | 11.6 ± 0.9 | 72.0 ± 0.0 | 72.0 ± 0.0 | 0.165 ± 0.000 |
| SNM: phrase only (no anchor) | yes | 27.64 ± 0.55% | 28.8 ± 1.5 | 27.4 ± 2.8 | 26.8 ± 0.3 | 24.4 ± 0.6 | 73.3 ± 1.7 | 0.936 ± 0.002 |
| SNM: keyword (anchor) only | yes | 13.25 ± 1.64% | 13.5 ± 2.0 | 13.1 ± 2.7 | 13.2 ± 1.3 | 24.4 ± 0.6 | 73.3 ± 1.7 | 0.936 ± 0.002 |

