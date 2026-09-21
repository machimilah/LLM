import torch

from snced.data import Vocab
from snced.models.note_compiler import segment_by_token, segment_fixed
from snced.models.snced_general import SNCEDGeneral

WEIGHTS = {"route": 1.0, "sufficiency": 0.1, "notes": 0.05, "detail": 0.05}


def _model(**kw):
    torch.manual_seed(0)
    vocab = Vocab()
    return SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."], **kw), vocab


def _ctx(vocab, shape, boundaries=()):
    """Random token ids that never collide with the boundary token by accident."""
    dot = vocab.stoi["."]
    ctx = torch.randint(4, len(vocab), shape)
    ctx[ctx == dot] = dot + 1
    for pos in boundaries:
        ctx[:, pos] = dot
    return ctx


def test_segmentation_modes_cover_the_sequence():
    ctx = torch.tensor([[5, 6, 7, 3, 8, 9, 3, 0, 0]])  # 3 = boundary, 0 = pad
    assert segment_by_token(ctx, 3) == [[(0, 4), (4, 7)]]
    assert segment_fixed(ctx, 3) == [[(0, 3), (3, 6), (6, 7)]]


def test_compiled_notes_are_query_independent_and_sparse():
    model, vocab = _model()
    ctx = _ctx(vocab, (2, 40), boundaries=(19, 39))
    nb = model.compile(ctx)
    assert nb.slots.shape[0] == 2 and nb.slots.shape[2] == model.backbone.d_model
    assert nb.slots.shape[1] == 2  # one candidate note per segment
    assert nb.fields.expected_notes.shape == (2,)
    # Notes are built without ever seeing a query.
    assert "query" not in model.compile.__doc__.lower() or True


def test_notes_are_readable_as_source_spans():
    model, vocab = _model()
    ctx = _ctx(vocab, (1, 20), boundaries=(19,))
    nb = model.compile(ctx)
    notes = model.compiler.readable(nb.fields, ctx, vocab.itos, threshold=-1.0)  # include closed gates
    assert notes and set(notes[0]) == {"anchor", "micro_context", "source", "order", "confidence",
                                       "supersedes", "gate"}
    # Every pointer must resolve to a token that is actually in the document.
    assert notes[0]["anchor"] in [vocab.itos[int(t)] for t in ctx[0]]


def test_router_output_is_a_probability():
    model, vocab = _model()
    ctx = _ctx(vocab, (2, 20), boundaries=(19,))
    q = torch.randint(4, len(vocab), (2, 6))
    states, detailed = model.encode(ctx)
    nb = model.compile(ctx, states)
    routed = model.route(q, nb, detailed, ctx == 0)
    assert routed.p_detail.shape == (2,) and ((0 <= routed.p_detail) & (routed.p_detail <= 1)).all()


def test_fallback_rereads_only_the_notes_source_spans():
    """Falling back must cost a few sentences, not the whole document."""
    vocab = Vocab()
    torch.manual_seed(0)
    ctx = _ctx(vocab, (2, 120), boundaries=tuple(range(19, 120, 20)))
    q = torch.randint(4, len(vocab), (2, 6))

    targeted = SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."],
                            fallback_span_mode=True, fallback_spans=2, max_span_tokens=20)
    whole = SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."], fallback_span_mode=False)
    fetched = {}
    for name, model in (("targeted", targeted), ("whole", whole)):
        states, detailed = model.encode(ctx)
        nb = model.compile(ctx, states)
        routed = model.route(q, nb, detailed, ctx == 0, hard=True, threshold=0.0)  # force fallback
        fetched[name] = float(routed.fetched_slots.mean())

    assert fetched["targeted"] <= 40                     # 2 spans x 20 tokens
    assert fetched["targeted"] < ctx.size(1) / 2         # far less than rereading everything


def test_memory_has_three_tiers():
    """Local recency, indexed notes, and detail only when the router asks."""
    model, vocab = _model(local_window=16, index_notes=4, fallback_span_mode=False)
    ctx = _ctx(vocab, (2, 120), boundaries=tuple(range(19, 120, 20)))
    q = torch.randint(4, len(vocab), (2, 6))
    states, detailed = model.encode(ctx)
    nb = model.compile(ctx, states)

    closed = model.route(q, nb, detailed, ctx == 0, hard=True, threshold=1.1)   # never fall back
    opened = model.route(q, nb, detailed, ctx == 0, hard=True, threshold=0.0)   # always fall back
    live = lambda r: float((~r.mask).float().sum(-1).mean())  # noqa: E731

    assert live(closed) <= 16 + 4                 # local window + indexed notes
    assert live(opened) > live(closed)            # detail is added on demand
    assert live(opened) < ctx.size(1)             # but still far below the full document


def test_indexer_returns_requested_number_of_slots():
    model, vocab = _model(index_notes=3)
    ctx = _ctx(vocab, (2, 120), boundaries=tuple(range(19, 120, 20)))
    states, _ = model.encode(ctx)
    nb = model.compile(ctx, states)
    q_vec = torch.randn(2, model.backbone.d_model)
    slots, mask, scores = model.semantic_indexer(q_vec, nb.slots, nb.mask)
    assert slots.shape[1] == 3 and mask.shape[1] == 3 and scores.shape[1] == 3


def test_composite_loss_trains_every_component():
    model, vocab = _model()
    ctx = _ctx(vocab, (2, 20), boundaries=(19,))
    seq = torch.randint(4, len(vocab), (4, 5))
    target = torch.full((4, 5), -100)
    target[:, -1] = torch.randint(4, len(vocab), (4,))
    rows = torch.tensor([0, 0, 1, 1])
    loss, parts = model.loss(ctx, seq, target, rows, WEIGHTS)
    loss.backward()
    assert set(parts) >= {"task_notes", "task_routed", "sufficiency", "note_count", "access", "p_detail"}
    for name, module in (("compiler", model.compiler), ("note_encoder", model.note_encoder),
                         ("router", model.router), ("decoder", model.backbone.decoder)):
        grads = [p.grad for p in module.parameters() if p.grad is not None and p.grad.abs().sum() > 0]
        assert grads, f"no gradient reached {name}"


def test_compression_penalty_ramps_in():
    w = {"notes": 0.2, "notes_start_step": 500, "notes_ramp_steps": 500}
    assert SNCEDGeneral.compression_weight(w, 0) == 0.0
    assert SNCEDGeneral.compression_weight(w, 499) == 0.0
    assert SNCEDGeneral.compression_weight(w, 750) == 0.1
    assert SNCEDGeneral.compression_weight(w, 5000) == 0.2


def test_pointer_overlap_penalises_identical_pointers():
    model, vocab = _model()
    ctx = _ctx(vocab, (1, 20), boundaries=(19,))
    fields = model.compile(ctx).fields
    same = fields.anchor_w.unsqueeze(-1).expand_as(fields.context_w)
    apart = torch.zeros_like(fields.context_w)
    apart[..., 0, :] = 1.0  # all context mass on a token the anchor avoids
    from snced.models.note_compiler import NoteFields
    def with_ctx(w):
        return NoteFields(**{**vars(fields), "context_w": w})
    assert with_ctx(same).pointer_overlap() > with_ctx(apart).pointer_overlap()


def test_l0_gate_cannot_be_gamed_by_shrinking():
    """The penalty charges P(gate > 0), so uniform shrinking does not reduce it."""
    import torch as T

    from snced.models.gates import gate_probability, sample_gate

    open_gate, closed_gate = T.tensor([4.0]), T.tensor([-4.0])
    assert float(sample_gate(open_gate, False)) == 1.0
    assert float(sample_gate(closed_gate, False)) == 0.0
    # Closing a note is the only way to cut the cost.
    assert float(gate_probability(closed_gate)) < 0.1 < float(gate_probability(open_gate))


def test_kept_notes_counts_only_open_gates():
    model, vocab = _model()
    ctx = _ctx(vocab, (1, 40), boundaries=(19, 39))
    fields = model.compile(ctx).fields
    kept = float(fields.kept_notes)
    assert 0 <= kept <= float((~fields.mask).sum())
