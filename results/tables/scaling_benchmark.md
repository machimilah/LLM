# Scaling benchmark

Seeds [2, 3], 40 documents per length, 36 questions per document.

| Context | Full context | Notes | Memory | KV | GFLOPs | Time per task |
|---|---:|---:|---:|---:|---:|---:|
| 547 tok | 100.0% | 98.5% | 24 vs 547 (4.4%) | 36KB vs 820KB | 0.28 vs 0.49 | 38ms vs 46ms |
| 939 tok | 99.8% | 97.9% | 24 vs 939 (2.6%) | 36KB vs 1409KB | 0.48 vs 0.85 | 41ms vs 63ms |
| 1527 tok | 96.7% | 96.2% | 24 vs 1527 (1.6%) | 36KB vs 2291KB | 0.93 vs 1.54 | 48ms vs 88ms |
| 2704 tok | 92.7% | 91.9% | 24 vs 2704 (0.9%) | 37KB vs 4056KB | 2.36 vs 3.45 | 64ms vs 155ms |
