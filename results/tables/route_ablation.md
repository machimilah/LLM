# Why the routed path underperformed the notes path

Reproduce: `python experiments/synthetic/route_ablation.py` and
`python experiments/synthetic/route_topk_sweep.py` against
`results/checkpoints/snced_general_seed{1,3}.pt`. Evaluation only - no retraining.

## The symptom

Three seeds of the v2 general architecture answered from Semantic Note Memory at 97.9-100%,
but the path the **router** actually assembles scored far worse:

| Path | seed 1 | seed 3 |
|---|---:|---:|
| notes (all 30 slots) | 99.3% | 97.9% |
| routed (what the router builds) | 70.8% | 52.1% |
| detailed (full context reference) | 100% | 100% |

A router that makes the model worse than ignoring it is not a routing bug - it is the only
path that hands the decoder a *subset* of the notebook.

## Ablation: which component costs the accuracy

| Arm | seed 1 | seed 3 | slots |
|---|---:|---:|---:|
| notes | 99.3% | 97.9% | 30 |
| routed | 70.8% | 52.1% | 91 |
| routed, local tier removed | 60.4% | 49.3% | 27 |
| **routed, semantic indexer off (all notes)** | **99.3%** | **75.7%** | 105 |
| routed, detail gate forced shut | 68.1% | 52.1% | 80 |
| indexed notes alone | 58.3% | 49.3% | 16 |

Turning the indexer off restores seed 1 completely. Neither the local tier nor the detail gate
explains the loss.

## The mechanism: accuracy tracks *how many* notes, not *which*

| top-k kept | indexer scorer | router scorer |
|---|---|---|
| | seed 1 / seed 3 | seed 1 / seed 3 |
| 8 | 31.9% / 27.1% | 37.5% / 10.4% |
| 16 | 58.3% / 49.3% | 49.3% / 31.9% |
| 24 | 76.4% / 71.5% | 67.4% / 63.9% |
| 30 (all) | 99.3% / 97.9% | 99.3% / 97.9% |

Substituting the router's own trained relevance head for the indexer's scorer did not help and
was mostly worse. The indexer is not choosing badly; **there is nothing good to choose.**

Accuracy falls roughly in proportion to the fraction of notes withheld. That is the signature of
a *distributed* representation: each fact is smeared across the whole notebook rather than written
into the notes about it. Retrieval cannot work against a distributed code, because every subset
loses part of every answer.

## Why the training objective allowed this

Nothing priced it. `l_notes` always decoded the complete notebook, so spreading a fact across all
30 slots cost the model nothing and probably helped optimisation. The routed arm did see a subset,
but at weight 1.0 against a full-notebook term that was far easier to satisfy.

This was invisible for as long as only the full-notebook path was measured - a reminder of plan
Rule 2 in a new form: a memory system must be evaluated on the access pattern it claims, not on
the most generous one.

## The fix under test

`SNCEDGeneral(note_dropout=...)` adds a training arm that hides a random fraction of notes
(0.4) and still requires the correct answer, weighted by `loss.note_dropout`. A fact can no
longer live everywhere; it has to live in a slot, which is the precondition for the indexer's
claim that decoder cost stops growing with the size of the notebook.

Status: retraining seeds 1 and 3 on `configs/snced_general_quick.yaml`. The result that matters
is whether top-16 retrieval matches the full notebook. If it does not, the scaling story needs
rethinking rather than tuning.
