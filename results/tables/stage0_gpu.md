# Stage 0 on GPU: the compiler's ceiling

Kaggle T4, `macs26/sn-ced-stage-0-note-compiler`, 20,000 steps over 225,320 sentences
(72,631 labelled facts), sentence encoder fine-tuned (lr 2e-5), ~30 minutes.

| | CPU run (2,000 docs) | GPU run (8,000 docs, 20k steps) |
|---|---:|---:|
| qa1 retention | 82% | **85%** |
| qa2 retention (two facts) | 74% | **78%** |
| qa9 retention (negation) | 82% | **84%** |
| worst-case (the gate) | 0.74 | **0.82** |
| training loss | 0.0002 | **0.0000** |

Gate was >= 0.95. Not met.

## What this settles

The compiler fits its training data perfectly (loss 0.0000) and still reaches only ~82% on real
BABILong documents. 28x more data, a fine-tuned encoder and 20,000 GPU steps moved the worst-case
number by 8 points and then flattened. **The limit is generalisation, not compute.**

Training uses bAbI facts inside WikiText prose with entities substituted from a mined vocabulary;
evaluation uses BABILong, whose noise is book text (PG19) and whose entities are held out. The gap
between those two distributions is what the model fails to cross.

## What to try next, in order

1. **A stronger pretrained backbone.** MiniLM-L6 (22M, 384-dim) is the smallest sentence encoder in
   common use; DeBERTa-v3-base or E5-base would test whether the representation is the bottleneck.
   One line: `ENCODER_ID` in `experiments/real/span_compiler.py`.
2. **Training noise closer to the target.** WikiText is encyclopaedic; BABILong's is 19th-century
   prose. Training on book text (Gutenberg) removes a distribution shift the compiler should not
   have to cross.
3. **Token-level rather than sentence-level decisions.** The current design commits to one fact per
   sentence; glued sentences and multi-fact passages are lost by construction.

Nothing here contradicts the note *representation*: gold notes still answer 100% of these questions
on 4-10% of the document (results/tables/real_text_compiler.md). What fails is writing them.
