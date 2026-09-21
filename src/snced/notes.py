"""Semantic notes: anchor + micro-context (plan section 5)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class SemanticNote:
    anchor: str
    micro_context: str
    source_ref: str | None = None
    confidence: float = 1.0
    created_at: float | None = None
    valid_from: float | None = None
    valid_to: float | None = None
    embedding: list[float] | None = None

    @property
    def text(self) -> str:
        return f"{self.anchor} {self.micro_context}".strip()

    @property
    def tokens(self) -> list[str]:
        return self.text.split()

    @property
    def relation(self) -> str | None:
        parts = self.micro_context.split()
        return parts[0] if parts else None

    @property
    def target(self) -> str | None:
        parts = self.micro_context.split()
        return parts[1] if len(parts) > 1 else None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "SemanticNote":
        return cls(**d)


@dataclass
class Notebook:
    notes: list[SemanticNote] = field(default_factory=list)

    def add(self, note: SemanticNote) -> None:
        self.notes.append(note)

    def lookup(self, anchor: str, relation: str) -> SemanticNote | None:
        for n in self.notes:
            if n.anchor == anchor and n.relation == relation:
                return n
        return None

    @property
    def slots(self) -> int:
        return len(self.notes)

    @property
    def token_count(self) -> int:
        return sum(len(n.tokens) for n in self.notes)

    def answer(self, entity: str, hops: int) -> str | None:
        """Deterministic graph traversal over notes (the Experiment B reader)."""
        cur = entity
        for _ in range(hops):
            n = self.lookup(cur, "links")
            if n is None or n.target is None:
                return None
            cur = n.target
        n = self.lookup(cur, "value")
        return n.target if n is not None else None


def value_note(entity: str, value: str, source_ref: str | None = None, confidence: float = 1.0) -> SemanticNote:
    return SemanticNote(entity, f"value {value}", source_ref, confidence)


def link_note(entity: str, target: str, source_ref: str | None = None, confidence: float = 1.0) -> SemanticNote:
    return SemanticNote(entity, f"links {target}", source_ref, confidence)
