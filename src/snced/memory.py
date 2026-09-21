"""Memory conditions for the representation ablation (plan section 11.1).

As in the original Experiment A, the memory is assembled per question from the
required reasoning chain (query-conditioned). This tests the representation and
fallback, not autonomous note construction (that is Experiment B).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .data import Lecture, Question
from .fallback import SourceStore, chunk_ref
from .notes import SemanticNote, link_note, value_note
from .router import note_sufficient

CONDITIONS = ("iterative_rag", "snm", "snm_noisy", "snm_fallback", "word_only")


@dataclass
class Memory:
    tokens: list[str]
    fallback: bool = False
    corrupted: bool = False


def gold_note(lecture: Lecture, idx: int) -> SemanticNote:
    c = lecture.chunks[idx]
    make = value_note if c.kind == "value" else link_note
    return make(c.anchor, c.target, chunk_ref(idx))


def build_memory(
    lecture: Lecture, q: Question, condition: str, rng: random.Random, noise_p: float = 0.2
) -> Memory:
    required = lecture.required_chunks(q.entity, q.hops)

    if condition == "iterative_rag":
        return Memory([t for i in required for t in lecture.chunks[i].tokens])

    notes = [gold_note(lecture, i) for i in required]

    if condition == "word_only":
        return Memory([n.anchor for n in notes])
    if condition == "snm":
        return Memory([t for n in notes for t in n.tokens])

    # Noisy notes: with probability noise_p the answer-bearing note loses its
    # micro-context (only the anchor survives) and its confidence drops to 0.
    corrupted = rng.random() < noise_p
    if corrupted:
        last = notes[-1]
        notes[-1] = SemanticNote(last.anchor, "", last.source_ref, confidence=0.0)

    if condition == "snm_noisy":
        return Memory([t for n in notes for t in n.tokens], corrupted=corrupted)

    if condition == "snm_fallback":
        store = SourceStore.from_lecture(lecture)
        tokens: list[str] = []
        for n in notes:
            if note_sufficient(n):
                tokens += n.tokens
            else:
                tokens += store.fetch(n.source_ref).split()
        return Memory(tokens, fallback=store.fetches > 0, corrupted=corrupted)

    raise ValueError(condition)
