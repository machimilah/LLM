"""Adversarial lecture variants (plan section 20).

Each variant stresses a different failure mode of lossy semantic memory, using
only the existing vocabulary so that results measure memory rather than unknown
tokens:

  temporal_update  an entity's value is restated later in the document with the
                   same surface form; the later statement wins, so notes must
                   preserve order instead of collapsing to one fact per anchor
  exact_value      the value needs two tokens ("v3 v7"); a three-token note can
                   hold only one of them, so this probes a limit of the note
                   schema itself rather than of the compiler

Not covered here: confusable entity names (every token is equally distinct in a
word-level vocabulary, so surface confusability cannot be expressed) and exact
quotations (there is no free text to quote). Both need the real-text setting.

These are distribution shifts: models trained on standard lectures are evaluated
zero-shot, to find where fallback ought to fire.
"""

from __future__ import annotations

import random

from .data import VALUE_TEMPLATES, VALUES, Chunk, Lecture, Question, make_lecture

VARIANTS = ("baseline", "temporal_update", "exact_value")

EXACT_TEMPLATES = (
    "it was stated that the value of {e} is {v} {w} in this particular setting .",
    "the lecture explains that {e} has value {v} {w} and this detail is relevant to the topic .",
)


def make_variant(rng: random.Random, variant: str, n_entities: int = 12, n_filler: int = 18) -> Lecture:
    lec = make_lecture(rng, n_entities, n_filler)
    if variant == "baseline":
        return lec
    if variant == "temporal_update":
        for e in rng.sample(lec.entities, max(1, n_entities // 3)):
            new_value = rng.choice([v for v in VALUES if v != lec.values[e]])
            text = rng.choice(VALUE_TEMPLATES).format(e=e, v=new_value)
            # Insert after the original statement, in the second half of the lecture.
            original = lec.fact_index("value", e)
            pos = rng.randrange(max(original + 1, len(lec.chunks) // 2), len(lec.chunks) + 1)
            lec.chunks.insert(pos, Chunk(text, "value", e, new_value))
            lec.values[e] = new_value
        return lec
    if variant == "exact_value":
        for i, c in enumerate(list(lec.chunks)):
            if c.kind == "value":
                second = rng.choice(VALUES)
                text = rng.choice(EXACT_TEMPLATES).format(e=c.anchor, v=c.target, w=second)
                lec.chunks[i] = Chunk(text, "value", c.anchor, f"{c.target} {second}")
                lec.values[c.anchor] = f"{c.target} {second}"
        return lec
    raise ValueError(variant)


def sample_variant_questions(rng: random.Random, lec: Lecture, per_type: int) -> list[Question]:
    out = []
    for hops in (0, 1, 2):
        for e in rng.sample(lec.entities, min(per_type, len(lec.entities))):
            out.append(Question(e, hops, lec.answer(e, hops)))
    return out


def answer_tokens(answer: str) -> list[str]:
    """Answer as a token list (two tokens in the exact_value variant)."""
    return answer.split()
