from snced.notes import SemanticNote, value_note
from snced.router import note_sufficient, route


def test_note_sufficiency():
    assert note_sufficient(value_note("e1", "v2"))
    assert not note_sufficient(None)
    assert not note_sufficient(SemanticNote("e1", ""))
    assert not note_sufficient(value_note("e1", "v2", confidence=0.1))


def test_route():
    good = value_note("e1", "v2")
    assert route([good, good]) == "snm"
    assert route([good, SemanticNote("e1", "", confidence=0.0)]) == "fallback"
    assert route([]) == "fallback"
