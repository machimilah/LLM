"""Span-labelled training data for an open-vocabulary note compiler.

Notes are extracted as character spans, so the compiler is not tied to any
entity list. To stop it memorising bAbI's four names and nine places, every
training document substitutes fresh names, places and objects drawn from large
pools; evaluation can then use entities the compiler never saw.

A training label is (anchor span, relation span, target span, negated).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

from notes_data import DROP_VERBS, MOVE_VERBS, TAKE_VERBS, split_sentences

# Surface pools for substitution. Training never sees the bAbI cast unchanged.
NAME_POOL = """Alice Boris Carmen Dmitri Elena Farid Greta Hassan Ingrid Jonas Kavita Lukas Maya Noor
Omar Priya Quentin Rosa Stefan Tomas Ulrike Viktor Wanda Xiomara Yusuf Zara Anneke Bruno Celia Dov
Esme Felix Gita Hugo Ilse Jarek Kenji Lena Milo Nadia Oskar Paloma Rafael Sonia Tarek Ulla Vera
Wiktor Yara Zeno Agnes Bertil Chiara Darius Edda Fabio Gunnar Hedda Ivar Jolanta Kasper Liesel
Mattis Nuria Otto Pernille Rurik Saskia Thilo Ursula Valter Wilhelmina Yannick Zsofia""".split()
PLACE_POOL = """attic balcony basement cellar chapel courtyard dairy forge granary greenhouse hangar
infirmary laundry library nursery observatory orchard pantry quarry refectory scullery stable
studio terrace vestibule workshop armoury bakery boathouse cloister conservatory dovecote
foundry gallery henhouse icehouse kiln lodge mill parlour pumphouse rookery smithy tannery
vault warehouse""".split()
OBJECT_POOL = """anvil basket candle compass drum easel flute goblet harp inkwell kettle lantern
mallet netting oilcan parcel quiver rucksack satchel telescope urn violin whistle yardstick zither
abacus bellows canteen decanter ewer funnel gourd hourglass ladle mortar""".split()

MOVE_RE = re.compile(rf"\b(\w+) ({MOVE_VERBS}) the (\w+)\b", re.I)
TAKE_RE = re.compile(rf"\b(\w+) ({TAKE_VERBS}) the (\w+)\b", re.I)
DROP_RE = re.compile(rf"\b(\w+) ({DROP_VERBS}) the (\w+)\b", re.I)
NEG_RE = re.compile(r"\b(\w+) (is no longer in|is not in) the (\w+)\b", re.I)
POS_RE = re.compile(r"\b(\w+) (is in) the (\w+)\b", re.I)
FACT_RES = (NEG_RE, MOVE_RE, TAKE_RE, DROP_RE, POS_RE)

# Relation surface -> semantics, used only by the evaluation reader.
MOVE_WORDS = ("moved", "went", "journeyed", "travelled", "traveled")
TAKE_WORDS = ("got", "grabbed", "took", "picked")
DROP_WORDS = ("dropped", "left", "discarded", "put")


def canonical_relation(rel_text: str, negated: bool) -> str:
    r = rel_text.lower()
    if negated or "no longer" in r or "not in" in r:
        return "state_not_in"
    if any(w in r for w in TAKE_WORDS):
        return "take"
    if any(w in r for w in DROP_WORDS):
        return "drop"
    if any(w in r for w in MOVE_WORDS):
        return "move"
    return "state_in"


@dataclass(frozen=True)
class SpanFact:
    anchor: tuple[int, int]
    relation: tuple[int, int]
    target: tuple[int, int]
    negated: bool

    def read(self, sentence: str) -> tuple[str, str, str]:
        return (sentence[slice(*self.anchor)], sentence[slice(*self.relation)], sentence[slice(*self.target)])


def find_span_facts(sentence: str) -> list[SpanFact]:
    """Character spans for every fact stated in a sentence, open vocabulary."""
    out: list[SpanFact] = []
    claimed: list[tuple[int, int]] = []
    for rx in FACT_RES:
        for m in rx.finditer(sentence):
            if any(a <= m.start() < b for a, b in claimed):
                continue
            claimed.append((m.start(), m.end()))
            out.append(SpanFact(m.span(1), m.span(2), m.span(3), rx is NEG_RE))
    return sorted(out, key=lambda f: f.anchor[0])


def mined_vocab(noise: list[str], seed: int = 0) -> dict[str, list[str]]:
    """Entity strings mined from real prose, split into disjoint train / eval halves.

    Attempt 1 substituted from small fixed pools and the tagger simply memorised
    them (results/logs/FAILED_RUNS.md). Sampling from a large, dynamic vocabulary
    and holding half of it out forces extraction to rely on syntax instead.
    """
    names, nouns = set(), set()
    for s in noise:
        for w in re.findall(r"\b[A-Z][a-z]{3,11}\b", s):
            names.add(w)
        for w in re.findall(r"\b[a-z]{4,10}\b", s):
            nouns.add(w)
    rng = random.Random(seed)
    names, nouns = sorted(names), sorted(nouns)
    rng.shuffle(names)
    rng.shuffle(nouns)
    half_n, half_o = len(names) // 2, len(nouns) // 2
    return {"train_names": names[:half_n], "eval_names": names[half_n:],
            "train_nouns": nouns[:half_o], "eval_nouns": nouns[half_o:]}


def substitute(rng: random.Random, story: dict, names: list[str] | None = None,
               nouns: list[str] | None = None, keep_original_p: float = 0.0) -> dict:
    """Swap bAbI's cast and nouns for fresh ones, consistently across the story.

    `names` / `nouns` default to the small fixed pools; pass mined vocabulary
    halves for open-vocabulary training and held-out evaluation. With
    `keep_original_p` the story is sometimes left untouched, so the original
    cast stays in the training mix.
    """
    if rng.random() < keep_original_p:
        return story
    text = story["passage"] + "\n" + story["question"] + "\n" + story["answer"]
    found_names = {n for n in re.findall(r"\b(Mary|John|Daniel|Sandra)\b", text)}
    places = {p for p in re.findall(r"\b(bathroom|bedroom|garden|hallway|kitchen|office)\b", text)}
    objects = {o for o in re.findall(r"\b(apple|football|milk)\b", text)}
    name_pool = names if names is not None else NAME_POOL
    place_pool = nouns if nouns is not None else PLACE_POOL
    object_pool = nouns if nouns is not None else OBJECT_POOL
    mapping = {}
    used: set[str] = set()
    for group, pool in ((found_names, name_pool), (places, place_pool), (objects, object_pool)):
        picks = [p for p in rng.sample(pool, min(len(pool), len(group) + 6)) if p not in used][: len(group)]
        used.update(picks)
        mapping.update(dict(zip(sorted(group), picks)))
    if not mapping or len(mapping) < len(found_names) + len(places) + len(objects):
        return story
    rx = re.compile(r"\b(" + "|".join(map(re.escape, mapping)) + r")\b")
    sub = lambda s: rx.sub(lambda m: mapping[m.group(1)], s)  # noqa: E731
    return {**story, "passage": sub(story["passage"]), "question": sub(story["question"]),
            "answer": sub(story["answer"])}


def rename_document(rng: random.Random, doc_text: str, question: str, answer: str,
                    names: list[str] | None = None, nouns: list[str] | None = None):
    """Substitution applied to a real BABILong document, for held-out-entity evaluation."""
    story = {"passage": doc_text, "question": question, "answer": answer, "task": 0}
    out = substitute(rng, story, names, nouns)
    return out["passage"], out["question"], out["answer"]


def build_span_document(rng: random.Random, story: dict, noise: list[str], target_words: int,
                        glue_p: float = 0.3, names: list[str] | None = None,
                        nouns: list[str] | None = None, keep_original_p: float = 0.15):
    """bAbI sentences (with substituted entities) interleaved into real prose."""
    story = substitute(rng, story, names, nouns, keep_original_p)
    facts = [s for s in story["passage"].strip().split("\n") if s.strip()]
    n_noise = max(1, (target_words - sum(len(s.split()) for s in facts)) // 18)
    picked = [rng.choice(noise) for _ in range(n_noise)]
    slots = sorted(rng.sample(range(len(picked) + len(facts)), len(facts)))
    sentences, fi, ni = [], 0, 0
    next_noise = lambda: picked[ni] if ni < len(picked) else rng.choice(noise)  # noqa: E731
    for i in range(len(picked) + len(facts)):
        if fi < len(slots) and i == slots[fi]:
            s = facts[fi]
            # Sometimes glue the fact onto the next prose sentence, as BABILong does.
            if rng.random() < glue_p:
                s = s + " " + next_noise()
                ni += 1
            sentences.append(s)
            fi += 1
        else:
            sentences.append(next_noise())
            ni += 1
    labels = [find_span_facts(s) for s in sentences]
    return sentences, labels, story


def load_noise(limit_paragraphs: int = 4000) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out = []
    for row in ds.take(limit_paragraphs):
        for s in split_sentences(row["text"]):
            if 8 <= len(s.split()) <= 40 and not s.startswith("="):
                out.append(s)
    return out
