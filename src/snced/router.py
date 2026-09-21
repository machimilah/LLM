"""Memory router (plan sections 8, 9; calibrated in P5).

Two policies, both rule-based:

  lecture   fall back when ANY note in the document is uncertain
  question  fall back only when a note the current question actually needs is
            uncertain (the notes reached by walking the anchor's chain)

Calibration on the synthetic benchmark (results/tables/router_calibration.md)
showed the lecture policy pays enormously for very little: at 1528-token
contexts it used 996 memory slots to gain 1.4 points over notes-only, while the
question policy gained 0.6 points for 79 slots. Question-level routing at a
threshold of 0.5-0.95 is therefore the default.

Learned routing with an access cost is still future work (plan section 10).
"""

from __future__ import annotations

from .notes import SemanticNote

DEFAULT_POLICY = "question"
DEFAULT_THRESHOLD = 0.5


def note_sufficient(note: SemanticNote | None, threshold: float = DEFAULT_THRESHOLD) -> bool:
    """A note is usable when it exists, carries micro-context and is confident."""
    return note is not None and bool(note.micro_context.strip()) and note.confidence >= threshold


def route(notes: list[SemanticNote | None], threshold: float = DEFAULT_THRESHOLD) -> str:
    """Return "snm" when every relevant note is sufficient, else "fallback"."""
    if notes and all(note_sufficient(n, threshold) for n in notes):
        return "snm"
    return "fallback"


def required_note_confidence(preds: list[dict], entity: str, hops: int) -> float:
    """Lowest confidence among the notes a question actually walks through.

    `preds` is one dict per source chunk, as produced by the compiler:
    {"kind": "value"|"link"|"filler", "anchor": str, "target": str, "confidence": float}.
    Returns 0.0 when the chain cannot be followed, which forces fallback.
    """
    facts = [p for p in preds if p.get("kind") != "filler"]
    cur, used = entity, []
    for _ in range(hops):
        link = next((p for p in facts if p["anchor"] == cur and p["kind"] == "link"), None)
        if link is None:
            return 0.0
        used.append(link)
        cur = link["target"]
    value = next((p for p in facts if p["anchor"] == cur and p["kind"] == "value"), None)
    if value is None:
        return 0.0
    used.append(value)
    return min(p["confidence"] for p in used)


def route_question(preds: list[dict], entity: str, hops: int,
                   threshold: float = DEFAULT_THRESHOLD) -> str:
    """Per-question routing: only the notes this question needs decide the path."""
    return "snm" if required_note_confidence(preds, entity, hops) >= threshold else "fallback"
