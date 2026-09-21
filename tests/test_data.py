import random

from snced.data import Vocab, make_lecture, sample_questions


def test_lecture_structure():
    lec = make_lecture(random.Random(0))
    assert len(lec.chunks) == 42
    kinds = [c.kind for c in lec.chunks]
    assert kinds.count("value") == 12 and kinds.count("link") == 12 and kinds.count("filler") == 18
    assert all(lec.links[e] != e for e in lec.entities)


def test_required_chunks_support_answer():
    rng = random.Random(1)
    lec = make_lecture(rng)
    for q in sample_questions(rng, lec, per_type=4):
        req = [lec.chunks[i] for i in lec.required_chunks(q.entity, q.hops)]
        assert len(req) == q.hops + 1
        assert req[-1].kind == "value" and req[-1].target == q.answer


def test_vocab_covers_everything():
    vocab = Vocab()
    rng = random.Random(2)
    for _ in range(20):
        lec = make_lecture(rng)
        assert 1 not in vocab.encode(lec.tokens())  # no UNK
        for q in sample_questions(rng, lec, 2):
            assert 1 not in vocab.encode(q.text.split())
