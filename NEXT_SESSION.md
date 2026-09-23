# Prompt for continuing on another machine

Clone `https://github.com/machimilah/LLM.git`, then paste everything below the line
into a fresh Claude Code session opened in the repo root.

---

I am continuing work on SN-CED (Semantic Note Memory for Causal Encoder-Decoder LLMs). This
repository is the complete state of the project. Before doing anything else, read `HANDOFF.md`
in full - it is the canonical record of every experiment, result, failed run and design decision -
then read `docs/target_architecture.md` for the architecture we are building toward and
`results/tables/route_ablation.md` for the open problem described below.

**Where things stand.** The v2 general architecture (`src/snced/models/snced_general.py`,
trained by `experiments/synthetic/general_train.py`) answers from Semantic Note Memory alone at
97.9-100% across three seeds, using 30 note slots for a ~430-token document, against a
full-context reference that also scores 100%. That headline result is solid and reproducible.

**The open problem.** The path the memory router actually assembles scores far worse than the
notes path - 70.8% and 52.1% versus 99.3% and 97.9%. I diagnosed this on trained checkpoints
without retraining (`experiments/synthetic/route_ablation.py` and
`experiments/synthetic/route_topk_sweep.py`). The cause is not the router and not the indexer's
ranking: accuracy tracks how MANY notes are kept rather than which ones (8/16/24/30 notes give
32/58/76/99%), and swapping in the router's trained relevance head does not help. Each fact is
smeared across the whole notebook instead of living in the notes about it, so every subset loses
part of every answer and retrieval cannot work. Nothing in the objective priced this, because the
notes loss term always decoded the complete notebook.

**The fix under test, which you need to finish.** I added `note_dropout` to `SNCEDGeneral`: a
training arm that hides a random fraction of notes (0.4) and still requires the correct answer,
weighted by `loss.note_dropout` in the config. It is committed and the test suite passes, but the
training runs were still in progress on the old machine and their results were never recorded.

Please rerun them, two seeds in parallel, roughly 15-25 minutes on CPU:

```
python experiments/synthetic/general_train.py --config snced_general_quick.yaml --seeds 1 \
  --threads 6 --init-backbone "ced_baseline_seed{seed}.pt" --freeze-encoder --decoder-frozen-steps 300
python experiments/synthetic/general_train.py --config snced_general_quick.yaml --seeds 3 \
  --threads 6 --init-backbone "ced_baseline_seed{seed}.pt" --freeze-encoder --decoder-frozen-steps 300
```

Then rerun both diagnostics against the new checkpoints. **The result that matters is whether
top-16 retrieval now matches the full notebook.** Report it honestly either way:

- If it does, the claim that decoder cost stops growing with the size of the notebook holds, and
  `results/tables/route_ablation.md` should be updated with the before/after.
- If it does not, say so plainly. The scaling story then needs rethinking rather than tuning, and
  that belongs in `results/logs/FAILED_RUNS.md` per plan Rule 11. Do not quietly raise `index_notes`
  to 30 and call the routed path fixed - that abandons the scaling claim while appearing to confirm it.

**How I work on this project.** Plan Rule 2: never compare memory systems against a baseline that
cannot solve the task. Rule 9: never delete failed experiments. Rule 11: report negative results.
Every measurement bug found so far came from a path that was flattered by how it was evaluated, so
be suspicious of good numbers and check what access pattern produced them. Keep `HANDOFF.md`,
`results/tables/` and `results/logs/FAILED_RUNS.md` current as you go, and commit with a message
that explains the finding rather than the diff.

**Known open items beyond this one**, in `HANDOFF.md`: the compiler's ceiling on real book prose
(0.86 against a 0.95 gate, diagnosed as a WikiText/BABILong distribution mismatch), sparsity that
does not actually prune at convergence, a v2 scaling benchmark across 547-2704 tokens that only
exists for v1, an unfinished Engram matched ablation, and a vision encoder blocked on the absence
of a multimodal benchmark.

Start by reading `HANDOFF.md`, then launch the two training runs, and tell me what you find.
