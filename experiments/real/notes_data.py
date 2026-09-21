"""Real-text data for the Semantic Note Compiler (plan sections 6, 20, P7).

bAbI stories carry the task-relevant structure (ordered events, several
relations per anchor, negation, temporal updates); the surrounding prose is
real text. Training documents use WikiText noise, while evaluation uses the
real BABILong documents, whose noise is book text the compiler never saw.

A note is (anchor, relation, target, polarity, source sentence index). Notes
keep document order and are never collapsed, so later events override earlier
ones and both supporting facts of a two-fact question survive.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass

MOVE_VERBS = r"(?:moved to|went to|went back to|journeyed to|travelled to|traveled to)"
TAKE_VERBS = r"(?:got|grabbed|took|picked up)"
DROP_VERBS = r"(?:dropped|left|discarded|put down)"

RELATIONS = ("move", "take", "drop", "state_in", "state_not_in")
NONE_REL = len(RELATIONS)  # label for "not a fact"

# bAbI uses a closed cast and set of places/objects (see babi_vocab.json). The
# patterns are not anchored to sentence boundaries because a fact sentence is
# sometimes glued onto surrounding prose; restricting the slots to the known
# vocabulary keeps prose from producing false matches. This closed-world parser
# only produces the GOLD labels - the compiler itself learns extraction.
PERSONS = ("daniel", "john", "mary", "sandra")
TARGETS = ("apple", "bathroom", "bedroom", "football", "garden", "hallway", "kitchen", "milk", "office")
_P = rf"({'|'.join(PERSONS)})"
_T = rf"({'|'.join(TARGETS)})"

PATTERNS = [
    ("move", re.compile(rf"\b{_P} {MOVE_VERBS} the {_T}\b", re.I)),
    ("take", re.compile(rf"\b{_P} {TAKE_VERBS} the {_T}\b", re.I)),
    ("drop", re.compile(rf"\b{_P} {DROP_VERBS} the {_T}\b", re.I)),
    ("state_not_in", re.compile(rf"\b{_P} is (?:no longer|not) in the {_T}\b", re.I)),
    ("state_in", re.compile(rf"\b{_P} is in the {_T}\b", re.I)),
]


@dataclass(frozen=True)
class Note:
    anchor: str
    relation: str
    target: str
    source: int  # sentence index in the document

    @property
    def polarity(self) -> str:
        return "not" if self.relation == "state_not_in" else ""

    @property
    def text(self) -> str:
        verb = {"move": "in", "take": "has", "drop": "dropped", "state_in": "in", "state_not_in": "not in"}
        return f"{self.anchor} {verb[self.relation]} {self.target}"

    @property
    def tokens(self) -> list[str]:
        return self.text.split()


def find_facts(sentence: str) -> list[tuple[str, str, str]]:
    """Every (anchor, relation, target) stated in a sentence, in order.

    `state_in` is only reported when no stronger relation matched the same
    span, so "Mary is no longer in the kitchen" is not also read as "is in".
    """
    found: list[tuple[int, str, str, str]] = []
    claimed: list[tuple[int, int]] = []
    for rel, pat in PATTERNS:
        for m in pat.finditer(sentence):
            if any(a <= m.start() < b for a, b in claimed):
                continue
            claimed.append((m.start(), m.end()))
            found.append((m.start(), m.group(1).lower(), rel, m.group(2).lower()))
    return [(a, r, t) for _, a, r, t in sorted(found)]


def parse_fact(sentence: str) -> tuple[str, str, str] | None:
    facts = find_facts(sentence)
    return facts[0] if facts else None


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [p.strip() for p in parts if p.strip()]


def notes_from_sentences(sentences: list[str]) -> list[Note]:
    """Gold notes: every fact stated anywhere in the document, in order."""
    return [Note(a, r, t, i) for i, s in enumerate(sentences) for a, r, t in find_facts(s)]


# --------------------------------------------------------------------------
# Deterministic reader: answers a question from notes alone (no LLM).
# This is the retention metric - does the notebook still support the task?
# --------------------------------------------------------------------------

def answer_from_notes(notes: list[Note], question: str) -> str | None:
    q = question.strip().rstrip("?").strip()
    location: dict[str, str] = {}
    holder: dict[str, str] = {}  # object -> person
    dropped_at: dict[str, str] = {}
    not_in: dict[str, set[str]] = {}
    for n in notes:
        if n.relation in ("move", "state_in"):
            location[n.anchor] = n.target
            not_in.setdefault(n.anchor, set()).discard(n.target)
        elif n.relation == "state_not_in":
            not_in.setdefault(n.anchor, set()).add(n.target)
            if location.get(n.anchor) == n.target:
                location.pop(n.anchor)
        elif n.relation == "take":
            holder[n.target] = n.anchor
            dropped_at.pop(n.target, None)
        elif n.relation == "drop":
            holder.pop(n.target, None)
            if n.anchor in location:
                dropped_at[n.target] = location[n.anchor]

    m = re.match(r"(?:Where is|where is) the (\w+)", q)
    if m:  # object question (qa2)
        obj = m.group(1).lower()
        if obj in holder:
            return location.get(holder[obj])
        return dropped_at.get(obj)
    m = re.match(r"(?:Where is|where is) (\w+)", q)
    if m:  # person question (qa1)
        return location.get(m.group(1).lower())
    m = re.match(r"(?:Is|is) (\w+) in the (\w+)", q)
    if m:  # yes/no question (qa9)
        person, place = m.group(1).lower(), m.group(2).lower()
        if place in not_in.get(person, set()) and location.get(person) != place:
            return "no"
        if person in location:
            return "yes" if location[person] == place else "no"
        return None
    return None


# --------------------------------------------------------------------------
# Training documents: bAbI story sentences inserted into real prose.
# --------------------------------------------------------------------------

@dataclass
class Document:
    sentences: list[str]
    labels: list[list[tuple[str, str, str]]]  # facts per sentence (usually 0 or 1)
    question: str
    answer: str
    task: int

    @property
    def text(self) -> str:
        return " ".join(self.sentences)

    def gold_notes(self) -> list[Note]:
        return [Note(a, r, t, i) for i, fs in enumerate(self.labels) for a, r, t in fs]


def build_document(rng: random.Random, story: dict, noise_sentences: list[str], target_words: int) -> Document:
    fact_sentences = [s for s in story["passage"].strip().split("\n") if s.strip()]
    n_noise = max(0, (target_words - sum(len(s.split()) for s in fact_sentences)) // 18)
    picked = [rng.choice(noise_sentences) for _ in range(n_noise)]
    # Interleave: facts keep their relative order, noise fills the gaps.
    slots = sorted(rng.sample(range(len(picked) + len(fact_sentences)), len(fact_sentences)))
    sentences, labels, fi, ni = [], [], 0, 0
    for i in range(len(picked) + len(fact_sentences)):
        if fi < len(slots) and i == slots[fi]:
            s = fact_sentences[fi]
            sentences.append(s)
            labels.append(find_facts(s))
            fi += 1
        else:
            sentences.append(picked[ni])
            labels.append([])
            ni += 1
    return Document(sentences, labels, story["question"], story["answer"], story["task"])


def load_noise_sentences(limit_paragraphs: int = 4000) -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    out: list[str] = []
    for row in ds.take(limit_paragraphs):
        for s in split_sentences(row["text"]):
            if 8 <= len(s.split()) <= 40 and not s.startswith("="):
                out.append(s)
    return out


def load_stories(tasks: tuple[int, ...] = (1, 2, 9)) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("Muennighoff/babi", split="train")
    return [r for r in ds if r["task"] in tasks]
