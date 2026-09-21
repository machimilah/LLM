from snced.router import required_note_confidence, route_question

PREDS = [
    {"kind": "value", "anchor": "e1", "target": "v2", "confidence": 0.99},
    {"kind": "link", "anchor": "e1", "target": "e2", "confidence": 0.95},
    {"kind": "value", "anchor": "e2", "target": "v7", "confidence": 0.40},
    {"kind": "filler", "anchor": None, "target": None, "confidence": 0.10},
]


def test_confidence_uses_only_the_notes_the_question_needs():
    assert required_note_confidence(PREDS, "e1", 0) == 0.99          # value note only
    assert required_note_confidence(PREDS, "e1", 1) == 0.40          # link + the weak value note
    assert required_note_confidence(PREDS, "e2", 0) == 0.40


def test_missing_chain_forces_fallback():
    assert required_note_confidence(PREDS, "e9", 0) == 0.0
    assert required_note_confidence(PREDS, "e2", 1) == 0.0           # e2 has no link note
    assert route_question(PREDS, "e9", 0) == "fallback"


def test_routing_threshold():
    assert route_question(PREDS, "e1", 0, threshold=0.5) == "snm"
    assert route_question(PREDS, "e1", 1, threshold=0.5) == "fallback"
    assert route_question(PREDS, "e1", 1, threshold=0.3) == "snm"
