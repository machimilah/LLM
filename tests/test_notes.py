from snced.notes import Notebook, SemanticNote, link_note, value_note


def test_note_text_and_fields():
    n = value_note("e3", "v4", "chunk7", 0.9)
    assert n.text == "e3 value v4"
    assert n.tokens == ["e3", "value", "v4"]
    assert (n.relation, n.target) == ("value", "v4")
    assert SemanticNote.from_dict(n.to_dict()) == n


def test_notebook_traversal():
    nb = Notebook([link_note("e1", "e2"), link_note("e2", "e3"), value_note("e3", "v5"), value_note("e1", "v0")])
    assert nb.answer("e1", 0) == "v0"
    assert nb.answer("e1", 2) == "v5"
    assert nb.answer("e3", 1) is None  # missing link note
    assert nb.slots == 4 and nb.token_count == 12
