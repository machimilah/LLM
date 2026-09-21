# Open-vocabulary span-based Semantic Note Compiler (BABILong 1k)

Compiler: frozen MiniLM-L6 (22M, not fine-tuned) + BIO span tagger and two sentence heads
(0.5M trained parameters). Per sentence it tags ANCHOR / RELATION / TARGET spans and predicts
"is this a fact" and "is it negated". Notes are the extracted strings, so no entity list is
built in. It compiles before any question is known, keeps document order and a source pointer.

Training: bAbI stories inside WikiText prose. Entity strings are substituted from a vocabulary
mined from real prose (2,626 names / 4,723 nouns), split into disjoint train and eval halves;
15% of documents keep bAbI's original cast. Evaluation is on real BABILong documents (book prose).

## Fact retention (deterministic reader over the notes, no LLM, n=100 per task)

| Evaluation | qa1 | qa2 | qa9 | Precision | Recall | Notes vs document |
|---|---:|---:|---:|---:|---:|---:|
| Original bAbI cast | 82.0% | 73.0% | 82.0% | 96.5-99.0% | 84.6-88.9% | 4.4-11.1% |
| Held-out entity strings | 66.0% | 51.0% | 61.0% | 82.2-86.6% | 65.6-74.3% | 4.1-10.6% |
| Gold notes (ceiling) | 100% | 100% | 100% | - | - | 3.6-9.6% |

Closed-vocabulary compiler, same tasks, for comparison: 88 / 90 / 92%.

## Reading

- Open-vocabulary extraction works but costs accuracy: 82/73/82% on the familiar cast versus
  88/90/92% for the closed-vocabulary classifier, and 51-66% on entity strings never seen.
- Precision stays high (82-99%): emitted notes are usually correct. Recall is the weak point
  (66-74% on unseen entities), so facts are missed rather than invented.
- qa2 degrades most (51%) because two facts must both survive; per-fact errors compound.
- Compression is unaffected by any of this: 4-11% of the document throughout.
- The bottleneck looks like the frozen MiniLM features plus a linear tagger, not the note schema:
  gold notes still answer 100% of every task.

## Fine-tuned encoder (`--finetune-encoder`, 2 epochs)

Unfreezing the sentence encoder removes the vocabulary dependence entirely:

| Evaluation | qa1 | qa2 | qa9 | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| Original bAbI cast | 82.0% | 73.0% | 82.0% | 97.7-99.4% | 84.8-89.0% |
| Held-out entity strings | 82.0% | 74.0% | 82.0% | 97.0-99.1% | 84.3-88.7% |

Held-out entities now score the same as familiar ones (66/51/61% -> 82/74/82%) and precision rises
to 97-99%. The residual loss is recall (~85-89% of facts found), uniform across vocabularies, which
is an extraction-quality problem rather than a generalisation one.

## Gate decision

The pre-agreed gate was retention near 90%+ before testing the notes against full context, RAG and
summaries with a 7B-class model. Best open-vocabulary retention is 82% (73% on qa2), on both familiar and
held-out entities after encoder fine-tuning, so the 7B comparison was NOT run. That stage also needs a GPU: 7B in fp32 is
~28 GB against 15 GB of RAM here, and bfloat16 is ~26x slower on this CPU.

## Limits

- One seed, n=100 per task, single context length (1k), three bAbI tasks.
- Relation canonicalisation for the reader uses a verb lexicon (evaluation side only).
- Training wall-clock is not comparable: the machine slept during epochs 3-4.
