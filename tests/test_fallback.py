import random

from snced.data import make_lecture, sample_questions
from snced.fallback import SourceStore, chunk_ref
from snced.memory import build_memory


def test_source_store_counts_fetches():
    lec = make_lecture(random.Random(0))
    store = SourceStore.from_lecture(lec)
    text = store.fetch(chunk_ref(3))
    assert text == lec.chunks[3].text
    assert store.fetches == 1 and store.fetched_tokens == len(text.split())


def test_fallback_restores_answer_only_when_corrupted():
    rng = random.Random(5)
    lec = make_lecture(rng)
    q = sample_questions(rng, lec, 1)[2]
    for seed in range(50):
        noisy = build_memory(lec, q, "snm_noisy", random.Random(seed), noise_p=0.5)
        fixed = build_memory(lec, q, "snm_fallback", random.Random(seed), noise_p=0.5)
        assert noisy.corrupted == fixed.corrupted == fixed.fallback
        assert q.answer in fixed.tokens
        assert (q.answer in noisy.tokens) != noisy.corrupted


def test_word_only_hides_answer():
    rng = random.Random(6)
    lec = make_lecture(rng)
    for q in sample_questions(rng, lec, 3):
        assert q.answer not in build_memory(lec, q, "word_only", rng).tokens
