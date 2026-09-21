"""Source store and selective fallback (plan section 9)."""

from __future__ import annotations

from dataclasses import dataclass, field

from .data import Lecture


def chunk_ref(i: int) -> str:
    return f"chunk{i}"


@dataclass
class SourceStore:
    """Keeps the original source available so notes stay auditable (Rule 7)."""

    chunks: dict[str, str] = field(default_factory=dict)
    fetches: int = 0
    fetched_tokens: int = 0

    @classmethod
    def from_lecture(cls, lecture: Lecture) -> "SourceStore":
        return cls({chunk_ref(i): c.text for i, c in enumerate(lecture.chunks)})

    def fetch(self, ref: str) -> str:
        text = self.chunks[ref]
        self.fetches += 1
        self.fetched_tokens += len(text.split())
        return text
