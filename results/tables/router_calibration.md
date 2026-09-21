# Router calibration (Plot E)

Seeds: [1, 2, 3]. 100 documents per length, 6 questions each.
`lecture` = fall back if any note in the lecture is uncertain (the policy used so far).
`question` = fall back only if a note the question actually needs is uncertain.

## 18 filler chunks

| Policy | Threshold | Accuracy | Mean slots | Fallback rate |
|---|---:|---:|---:|---:|
| lecture | 0.0 | 98.56 ± 0.98% | 24.0 | 0.0 ± 0.0% |
| lecture | 0.5 | 98.78 ± 0.84% | 60.7 | 7.0 ± 6.2% |
| lecture | 0.8 | 99.61 ± 0.10% | 250.7 | 43.3 ± 39.4% |
| lecture | 0.9 | 99.94 ± 0.10% | 341.7 | 60.7 ± 46.5% |
| lecture | 0.95 | 99.94 ± 0.10% | 408.1 | 73.3 ± 40.3% |
| lecture | 0.99 | 99.94 ± 0.10% | 500.7 | 91.0 ± 15.6% |
| lecture | 0.999 | 99.94 ± 0.10% | 547.9 | 100.0 ± 0.0% |
| lecture | 1.01 | 99.94 ± 0.10% | 547.9 | 100.0 ± 0.0% |
| question | 0.0 | 98.56 ± 0.98% | 24.0 | 0.0 ± 0.0% |
| question | 0.5 | 99.00 ± 0.67% | 31.0 | 1.3 ± 1.2% |
| question | 0.8 | 99.33 ± 0.33% | 52.2 | 5.4 ± 5.2% |
| question | 0.9 | 99.33 ± 0.33% | 80.1 | 10.7 ± 10.2% |
| question | 0.95 | 99.33 ± 0.33% | 112.1 | 16.8 ± 14.3% |
| question | 0.99 | 99.44 ± 0.19% | 225.0 | 38.4 ± 24.5% |
| question | 0.999 | 99.78 ± 0.10% | 406.1 | 72.9 ± 22.5% |
| question | 1.01 | 99.94 ± 0.10% | 547.9 | 100.0 ± 0.0% |

## 108 filler chunks

| Policy | Threshold | Accuracy | Mean slots | Fallback rate |
|---|---:|---:|---:|---:|
| lecture | 0.0 | 96.33 ± 3.59% | 24.1 | 0.0 ± 0.0% |
| lecture | 0.5 | 95.94 ± 4.26% | 229.1 | 13.7 ± 17.2% |
| lecture | 0.8 | 97.44 ± 3.34% | 800.0 | 51.7 ± 44.0% |
| lecture | 0.9 | 97.72 ± 3.25% | 995.7 | 64.7 ± 51.0% |
| lecture | 0.95 | 97.72 ± 3.25% | 1075.6 | 70.0 ± 49.4% |
| lecture | 0.99 | 97.78 ± 3.29% | 1397.0 | 91.3 ± 15.0% |
| lecture | 0.999 | 97.78 ± 3.29% | 1527.4 | 100.0 ± 0.0% |
| lecture | 1.01 | 97.78 ± 3.29% | 1527.4 | 100.0 ± 0.0% |
| question | 0.0 | 96.33 ± 3.59% | 24.1 | 0.0 ± 0.0% |
| question | 0.5 | 96.94 ± 3.58% | 79.2 | 3.7 ± 3.8% |
| question | 0.8 | 96.94 ± 3.58% | 156.8 | 8.8 ± 8.4% |
| question | 0.9 | 96.83 ± 3.88% | 230.4 | 13.7 ± 11.9% |
| question | 0.95 | 97.28 ± 3.50% | 336.2 | 20.8 ± 17.6% |
| question | 0.99 | 97.39 ± 3.43% | 650.9 | 41.7 ± 28.1% |
| question | 0.999 | 97.78 ± 3.29% | 1131.3 | 73.7 ± 24.2% |
| question | 1.01 | 97.78 ± 3.29% | 1527.4 | 100.0 ± 0.0% |

## Calibrated thresholds (cheapest within 0.5pp of the best accuracy)

| Filler | Policy | Threshold |
|---|---|---:|
| 18 | lecture | 0.8 |
| 18 | question | 0.99 |
| 108 | lecture | 0.8 |
| 108 | question | 0.95 |
