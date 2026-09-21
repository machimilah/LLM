"""Synthetic lecture benchmark (plan sections 11.1 and 11.2).

A lecture contains a set of entities. Each entity has one value fact and one
directed link fact, and the facts are shuffled among irrelevant filler chunks.
Questions ask for the value reached after 0, 1 or 2 link hops.

The original prototype scripts were not available, so this generator is a
reconstruction from the plan's description: 12 entities, 18 filler chunks,
42 raw chunks, 8 answer values.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

N_ENTITY_POOL = 24
N_VALUES = 8
ENTITIES = [f"e{i}" for i in range(N_ENTITY_POOL)]
VALUES = [f"v{i}" for i in range(N_VALUES)]

VALUE_TEMPLATES = (
    "the lecture explains that {e} has value {v} and this detail is relevant to the topic .",
    "according to the notes {e} is assigned value {v} which students should remember .",
    "it was stated that the value of {e} is {v} in this particular setting .",
    "remember that {e} takes value {v} whenever the model is applied in practice .",
)
LINK_TEMPLATES = (
    "in the framework {e} connects to {t} and this relation matters later .",
    "the speaker noted that {e} points to {t} in the causal diagram on the slide .",
    "one key relation is that {e} leads to {t} according to the slides .",
    "note that {e} is linked with {t} throughout the rest of the discussion .",
)
# Some filler mentions an entity or relation words without stating a fact, so
# the compiler cannot classify chunks from surface keywords alone.
FILLER_TEMPLATES = (
    "the professor paused to discuss the history of the course .",
    "several students asked questions about the homework deadline .",
    "the professor stressed that every value in the table must be checked twice .",
    "this part of the lecture is mostly background and can be skimmed .",
    "a short video was shown before the break .",
    "students often confuse {e} with other symbols in the exam .",
    "the slides about {e} will be uploaded after class .",
    "the diagram connects many ideas but none of them are tested .",
    "attendance was lower than usual because of the weather .",
    "the next section repeats material from the previous week .",
)

QUESTION_TEMPLATES = {
    0: "what is the value of {e} ?",
    1: "what is the value of the entity that {e} connects to ?",
    2: "what is the value of the entity reached from {e} after two connections ?",
}
QTYPE_NAMES = {0: "direct", 1: "one_hop", 2: "two_hop"}

SPECIALS = ("[PAD]", "[UNK]", "[SEP]", "[CLS]")
PAD, UNK, SEP, CLS = range(4)


@dataclass(frozen=True)
class Chunk:
    text: str
    kind: str  # "value" | "link" | "filler"
    anchor: str | None = None
    target: str | None = None  # value token for value facts, entity for links

    @property
    def tokens(self) -> list[str]:
        return self.text.split()


@dataclass
class Lecture:
    chunks: list[Chunk]
    values: dict[str, str]
    links: dict[str, str]
    entities: list[str] = field(default_factory=list)

    def tokens(self) -> list[str]:
        return [t for c in self.chunks for t in c.tokens]

    def chain_entities(self, entity: str, hops: int) -> list[str]:
        chain = [entity]
        for _ in range(hops):
            chain.append(self.links[chain[-1]])
        return chain

    def answer(self, entity: str, hops: int) -> str:
        return self.values[self.chain_entities(entity, hops)[-1]]

    def fact_index(self, kind: str, anchor: str) -> int:
        for i, c in enumerate(self.chunks):
            if c.kind == kind and c.anchor == anchor:
                return i
        raise KeyError((kind, anchor))

    def required_chunks(self, entity: str, hops: int) -> list[int]:
        """Indices of the source chunks needed to answer, in reasoning order."""
        chain = self.chain_entities(entity, hops)
        idx = [self.fact_index("link", e) for e in chain[:-1]]
        idx.append(self.fact_index("value", chain[-1]))
        return idx


@dataclass(frozen=True)
class Question:
    entity: str
    hops: int
    answer: str

    @property
    def text(self) -> str:
        return QUESTION_TEMPLATES[self.hops].format(e=self.entity)


def make_lecture(rng: random.Random, n_entities: int = 12, n_filler: int = 18) -> Lecture:
    ents = rng.sample(ENTITIES, n_entities)
    values = {e: rng.choice(VALUES) for e in ents}
    links = {e: rng.choice([t for t in ents if t != e]) for e in ents}
    chunks: list[Chunk] = []
    for e in ents:
        chunks.append(Chunk(rng.choice(VALUE_TEMPLATES).format(e=e, v=values[e]), "value", e, values[e]))
        chunks.append(Chunk(rng.choice(LINK_TEMPLATES).format(e=e, t=links[e]), "link", e, links[e]))
    for _ in range(n_filler):
        chunks.append(Chunk(rng.choice(FILLER_TEMPLATES).format(e=rng.choice(ENTITIES)), "filler"))
    rng.shuffle(chunks)
    return Lecture(chunks, values, links, ents)


def sample_questions(rng: random.Random, lecture: Lecture, per_type: int) -> list[Question]:
    qs = []
    for hops in (0, 1, 2):
        for e in rng.sample(lecture.entities, per_type):
            qs.append(Question(e, hops, lecture.answer(e, hops)))
    return qs


class Vocab:
    """Closed word-level vocabulary over every template, entity and value."""

    def __init__(self) -> None:
        words: list[str] = list(SPECIALS)
        extra = ["value", "links"]  # note micro-context words
        templates = VALUE_TEMPLATES + LINK_TEMPLATES + FILLER_TEMPLATES + tuple(QUESTION_TEMPLATES.values())
        for tpl in templates:
            for w in tpl.split():
                if not w.startswith("{"):
                    extra.append(w)
        for w in ENTITIES + VALUES + extra:
            if w not in words:
                words.append(w)
        self.itos = words
        self.stoi = {w: i for i, w in enumerate(words)}
        self.entity_ids = [self.stoi[e] for e in ENTITIES]
        self.value_ids = [self.stoi[v] for v in VALUES]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(t, UNK) for t in tokens]
