import torch

from snced.data import PAD, SEP, Vocab
from snced.models.ced import CEDBaseline


def _model(**kw):
    torch.manual_seed(0)
    return CEDBaseline(len(Vocab()), conv_kernel=6, **kw).eval()


def test_encoder_is_causal():
    m = _model()
    ctx = torch.randint(4, 50, (1, 20))
    ctx2 = ctx.clone()
    ctx2[0, 15:] = torch.randint(4, 50, (5,))
    with torch.no_grad():
        a, b = m.encode(ctx), m.encode(ctx2)
    assert torch.allclose(a[0, :15], b[0, :15], atol=1e-5)
    assert not torch.allclose(a[0, 15:], b[0, 15:])


def test_right_padding_does_not_change_outputs():
    m = _model()
    ctx = torch.randint(4, 50, (1, 20))
    q = torch.randint(4, 50, (1, 6))
    padded_ctx = torch.cat([ctx, torch.full((1, 7), PAD)], dim=1)
    padded_q = torch.cat([q, torch.full((1, 3), PAD)], dim=1)
    with torch.no_grad():
        assert torch.allclose(m(ctx, q), m(padded_ctx, padded_q), atol=1e-5)


def test_generative_decoding_shapes():
    m = _model(generative=True)
    vocab_size = len(Vocab())
    ctx = torch.randint(4, 50, (3, 20))
    prompt = torch.cat([torch.randint(4, 50, (3, 6)), torch.full((3, 1), SEP)], dim=1)
    with torch.no_grad():
        mem = m.encode(ctx)
        assert m.decode_all(mem, ctx == PAD, prompt).shape == (3, 7, vocab_size)
        assert m.generate(mem, ctx == PAD, prompt, 3).shape == (3, 3)
