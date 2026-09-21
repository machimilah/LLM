import random

from snced.adversarial import VARIANTS, make_variant, sample_variant_questions
from snced.data import VALUES, Vocab


def test_variants_stay_in_vocabulary():
    """Adversarial text must use known words, or results would measure UNK handling."""
    vocab = Vocab()
    rng = random.Random(0)
    for variant in VARIANTS:
        lec = make_variant(rng, variant)
        assert 1 not in vocab.encode(lec.tokens())  # 1 == UNK


def test_temporal_update_answer_is_the_later_value():
    rng = random.Random(1)
    lec = make_variant(rng, "temporal_update")
    for e in lec.entities:
        value_chunks = [c for c in lec.chunks if c.kind == "value" and c.anchor == e]
        # The lecture's answer must match the LAST statement about the entity.
        assert lec.values[e] == value_chunks[-1].target
    assert any(len([c for c in lec.chunks if c.kind == "value" and c.anchor == e]) > 1 for e in lec.entities)


def test_exact_value_needs_two_tokens():
    rng = random.Random(2)
    lec = make_variant(rng, "exact_value")
    for e in lec.entities:
        assert len(lec.values[e].split()) == 2
        assert all(tok in VALUES for tok in lec.values[e].split())
    q = sample_variant_questions(rng, lec, 1)[0]
    assert q.answer == lec.answer(q.entity, q.hops)
