import random

import torch

from snced.compiler import NoteCompiler, chunk_batch, compile_notebook, compiler_loss
from snced.data import Vocab, make_lecture


def test_compiler_shapes_and_loss():
    vocab = Vocab()
    lecs = [make_lecture(random.Random(i)) for i in range(2)]
    b = chunk_batch(vocab, lecs)
    model = NoteCompiler(len(vocab))
    out = model(b["ids"], b["lengths"])
    assert out["kind"].shape == (84, 3)
    loss = compiler_loss(out, b)
    assert torch.isfinite(loss)
    loss.backward()


def test_compile_notebook_is_query_independent():
    vocab = Vocab()
    lec = make_lecture(random.Random(0))
    nb, preds = compile_notebook(NoteCompiler(len(vocab)), vocab, lec)
    assert len(preds) == len(lec.chunks)
    assert all(n.source_ref.startswith("chunk") for n in nb.notes)
