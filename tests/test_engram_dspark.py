import torch

from snced.data import SEP, Vocab
from snced.models.dspark import DraftModule, speculative_decode, verify_identical_to_greedy
from snced.models.engram import EngramMemory
from snced.models.snced_general import SNCEDGeneral


def _model():
    torch.manual_seed(0)
    vocab = Vocab()
    return SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."]), vocab


# ---------------------------------------------------------------- Engram


def test_engram_starts_closed_and_preserves_the_hidden_state():
    """A component that helps nothing must not hurt: gates start near zero."""
    torch.manual_seed(0)
    eng = EngramMemory(d_model=32, table_size=4096)
    ids = torch.randint(1, 500, (2, 16))
    hidden = torch.randn(2, 16, 32)
    out = eng(ids, hidden)
    assert out.shape == hidden.shape
    assert (out - hidden).abs().mean() < 0.2         # barely perturbed at init
    usage = eng.gate_usage(ids, hidden)
    assert all(v < 0.1 for v in usage.values())      # all three gates nearly shut


def test_engram_hashes_are_deterministic_and_context_sensitive():
    eng = EngramMemory(d_model=16, table_size=1024)
    a = torch.tensor([[5, 6, 7, 8]])
    b = torch.tensor([[5, 6, 9, 8]])                 # same unigram at the end, different trigram
    h3_a = eng.hash_ngrams(a, 3, 2_000_003)
    h3_b = eng.hash_ngrams(b, 3, 2_000_003)
    assert torch.equal(h3_a, eng.hash_ngrams(a, 3, 2_000_003))   # deterministic
    assert h3_a[0, -1] != h3_b[0, -1]                            # context changes the key
    assert int(h3_a.max()) < 1024                                # inside the table


def test_engram_reports_its_true_cost():
    eng = EngramMemory(d_model=64, orders=(2, 3, 4), table_size=2 ** 14)
    assert eng.parameter_count > 3 * 2 ** 14 * 64                # the storage footprint is real
    # Active compute per token stays tiny: that is the trade being made.
    assert eng.active_flops_per_token(64) < 1000


def test_engram_gradients_reach_the_tables():
    eng = EngramMemory(d_model=16, table_size=512)
    ids = torch.randint(1, 100, (2, 8))
    hidden = torch.randn(2, 8, 16, requires_grad=True)
    eng(ids, hidden).sum().backward()
    assert any(t.weight.grad is not None and t.weight.grad.abs().sum() > 0 for t in eng.tables)


# ---------------------------------------------------------------- DSpark


def test_draft_proposes_a_block_with_confidences():
    model, vocab = _model()
    draft = DraftModule(model.backbone.d_model, len(vocab), block=5)
    state = torch.randn(3, model.backbone.d_model)
    tokens, conf = draft(state)
    assert tokens.shape == (3, 5) and conf.shape == (3, 5)
    assert ((0 <= conf) & (conf <= 1)).all()


def test_speculative_decoding_is_bit_identical_to_greedy():
    """The correctness criterion: throughput must not change what is generated."""
    model, vocab = _model()
    model.eval()
    draft = DraftModule(model.backbone.d_model, len(vocab), block=4)
    ctx = torch.randint(4, len(vocab), (2, 40))
    ctx[:, 39] = vocab.stoi["."]
    states, detailed = model.encode(ctx)
    nb = model.compile(ctx, states)
    prompt = torch.cat([torch.randint(4, len(vocab), (2, 5)), torch.full((2, 1), SEP)], dim=1)

    assert verify_identical_to_greedy(model, nb.slots, nb.mask, prompt, 6, draft)


def test_speculative_stats_are_consistent():
    model, vocab = _model()
    model.eval()
    draft = DraftModule(model.backbone.d_model, len(vocab), block=3)
    ctx = torch.randint(4, len(vocab), (1, 40))
    ctx[:, 39] = vocab.stoi["."]
    states, _ = model.encode(ctx)
    nb = model.compile(ctx, states)
    prompt = torch.cat([torch.randint(4, len(vocab), (1, 4)), torch.full((1, 1), SEP)], dim=1)

    out, stats = speculative_decode(model, nb.slots, nb.mask, prompt, 6, draft)
    assert out.shape[1] == 6
    assert stats.generated >= 6
    assert stats.accepted <= stats.drafted
    assert stats.verification_steps <= 6          # never worse than one pass per token
    assert 0 <= stats.acceptance_rate <= 1


def test_draft_training_loss_runs():
    model, vocab = _model()
    draft = DraftModule(model.backbone.d_model, len(vocab), block=3)
    state = torch.randn(4, model.backbone.d_model)
    targets = torch.randint(4, len(vocab), (4, 3))
    loss = draft.loss(state, targets)
    loss.backward()
    assert torch.isfinite(loss)
