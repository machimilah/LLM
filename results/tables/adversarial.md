# Adversarial evaluation (plan section 20)

Seeds: [1, 2, 3]. 40 documents per variant, 6 questions each, zero-shot:
the models were trained only on the standard lectures.

| Variant | Full context | Predicted notes | Gold notes | Note slots |
|---|---:|---:|---:|---:|
| baseline | 100.0 ± 0.0% | 97.6 ± 3.7% | 99.6 ± 0.7% | 24.0 |
| temporal_update | 84.9 ± 0.9% | 78.8 ± 1.8% | 80.3 ± 1.3% | 28.0 |
| exact_value | 10.6 ± 6.1% | 3.6 ± 1.9% | 1.1 ± 1.2% | 23.7 |

Reading: a gap between gold notes and full context is a limit of the note schema;
a gap between predicted and gold notes is compiler error.
