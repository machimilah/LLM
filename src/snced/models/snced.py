"""SN-CED: a CED baseline plus a native Semantic Note Compiler (plan sections 4, 6, 14).

The compiler reads the final causal-encoder states of every sentence (chunk)
before any question is known, and predicts chunk kind, anchor entity and
target. Predicted notes ("e5 value v3", "e5 links e9") become the decoder's
memory in place of the full lecture. The base CED (encoder, decoder,
projection) is shared, so the detailed-memory path is still available for
fallback.

Note memory modes:
  "note_slots"      a dedicated note encoder maps each (anchor, relation, target)
                    triple to ONE memory slot (default).
  "shared_encoder"  re-encode the packed note tokens with the frozen lecture
                    encoder (attempt 2; the decoder could not read it).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from ..compiler import KINDS
from ..data import ENTITIES, N_ENTITY_POOL, N_VALUES, PAD, SEP, VALUES, Lecture, Vocab
from .ced import CEDBaseline


@dataclass
class CompiledNotes:
    """Batched notebook: token ids for the note memory plus per-lecture confidence."""

    note_ids: torch.Tensor  # (B, N) padded note tokens
    note_count: list[int]  # notes per lecture
    min_confidence: torch.Tensor  # (B,) lowest confidence among a lecture's notes
    preds: list[list[dict]]  # per lecture, per chunk: kind / anchor / target / confidence


class SNCED(nn.Module):
    def __init__(self, base: CEDBaseline, vocab: Vocab, d_hidden: int = 128, n_queries: int = 4,
                 max_chunk_len: int = 32, note_mode: str = "note_slots") -> None:
        super().__init__()
        self.base, self.vocab, self.note_mode = base, vocab, note_mode
        self.period = vocab.stoi["."]
        d = base.d_model
        # Attention pooling: learned queries read the chunk's encoder states
        # (plus within-chunk position), so the compiler can pick out the anchor
        # and target tokens instead of averaging them away.
        self.proj = nn.Linear(d, d_hidden)
        self.pos = nn.Embedding(max_chunk_len, d_hidden)
        self.queries = nn.Parameter(torch.randn(n_queries, d_hidden) * 0.02)
        self.pool = nn.MultiheadAttention(d_hidden, 4, batch_first=True)
        self.compiler = nn.Sequential(nn.Linear(n_queries * d_hidden, d_hidden), nn.GELU())
        self.kind_head = nn.Linear(d_hidden, len(KINDS))
        self.anchor_head = nn.Linear(d_hidden, N_ENTITY_POOL)
        self.value_head = nn.Linear(d_hidden, N_VALUES)
        self.link_head = nn.Linear(d_hidden, N_ENTITY_POOL)
        # One memory slot per note, in the decoder's memory space.
        self.note_encoder = nn.Sequential(
            nn.Linear(3 * d, d_hidden), nn.GELU(), nn.Linear(d_hidden, d_hidden), nn.GELU(), nn.Linear(d_hidden, d)
        )

    def compiler_modules(self):
        return (self.proj, self.pos, self.pool, self.compiler, self.kind_head, self.anchor_head,
                self.value_head, self.link_head)

    def compiler_parameters(self):
        yield self.queries
        for m in self.compiler_modules():
            yield from m.parameters()

    def chunk_spans(self, ctx: torch.Tensor, states: torch.Tensor):
        """Split encoder states into sentences. Returns padded spans (N, L, d), pad mask, owner lecture."""
        spans, owner = [], []
        for b in range(ctx.size(0)):
            start = 0
            for end in (ctx[b] == self.period).nonzero().squeeze(1).tolist():
                spans.append(states[b, start : end + 1])
                owner.append(b)
                start = end + 1
        L = max(len(x) for x in spans)
        padded = torch.zeros(len(spans), L, states.size(-1))
        mask = torch.ones(len(spans), L, dtype=torch.bool)
        for i, x in enumerate(spans):
            padded[i, : len(x)], mask[i, : len(x)] = x, False
        return padded, mask, torch.tensor(owner)

    def head_logits(self, spans: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.proj(spans) + self.pos(torch.arange(spans.size(1)))
        q = self.queries.expand(spans.size(0), -1, -1)
        pooled, _ = self.pool(q, h, h, key_padding_mask=mask, need_weights=False)
        z = self.compiler(pooled.flatten(1))
        return {"kind": self.kind_head(z), "anchor": self.anchor_head(z),
                "value": self.value_head(z), "link": self.link_head(z)}

    def chunk_logits(self, ctx: torch.Tensor, states: torch.Tensor | None = None):
        """Head logits for every chunk. Returns (logits dict, lecture index per chunk)."""
        if states is None:
            states = self.base.encode_states(ctx)
        spans, mask, owner = self.chunk_spans(ctx, states)
        return self.head_logits(spans, mask), owner

    @torch.no_grad()
    def compile(self, ctx: torch.Tensor, states: torch.Tensor | None = None) -> CompiledNotes:
        logits, owner = self.chunk_logits(ctx, states)
        probs = {k: v.softmax(-1) for k, v in logits.items()}
        # Move every head to Python lists in one go. Indexing the tensors chunk
        # by chunk cost ~0.9 ms each in dispatch overhead and dominated the
        # measured compile time (results/tables/timing.md).
        best = {k: (v.max(-1).values.tolist(), v.argmax(-1).tolist()) for k, v in probs.items()}
        kind_conf, kind_idx = best["kind"]
        anchor_conf, anchor_idx = best["anchor"]
        value_conf, value_idx = best["value"]
        link_conf, link_idx = best["link"]
        owners = owner.tolist()

        B = ctx.size(0)
        notes: list[list[int]] = [[] for _ in range(B)]
        preds: list[list[dict]] = [[] for _ in range(B)]
        min_conf = [1.0] * B
        encode = self.vocab.stoi
        for i, b in enumerate(owners):
            kind = KINDS[kind_idx[i]]
            conf = kind_conf[i]
            pred = {"kind": kind, "anchor": None, "target": None, "confidence": conf}
            if kind != "filler":
                anchor = ENTITIES[anchor_idx[i]]
                if kind == "value":
                    target, p_t, rel = VALUES[value_idx[i]], value_conf[i], "value"
                else:
                    target, p_t, rel = ENTITIES[link_idx[i]], link_conf[i], "links"
                conf *= anchor_conf[i] * p_t
                notes[b] += (encode[anchor], encode[rel], encode[target])
                pred.update(anchor=anchor, target=target, confidence=conf)
                if conf < min_conf[b]:
                    min_conf[b] = conf
            preds[b].append(pred)
        return self._pack(notes, preds, torch.tensor(min_conf))

    def gold_notes(self, lectures: list[Lecture]) -> CompiledNotes:
        notes = []
        for lec in lectures:
            toks: list[str] = []
            for c in lec.chunks:
                if c.kind != "filler":
                    toks += [c.anchor, "value" if c.kind == "value" else "links", c.target]
            notes.append(self.vocab.encode(toks))
        return self._pack(notes, [[] for _ in lectures], torch.ones(len(lectures)))

    @staticmethod
    def _pack(notes: list[list[int]], preds, min_conf) -> CompiledNotes:
        notes = [n if n else [SEP] * 3 for n in notes]  # an empty notebook still needs one key
        ids = torch.full((len(notes), max(map(len, notes))), PAD, dtype=torch.long)
        for i, n in enumerate(notes):
            ids[i, : len(n)] = torch.tensor(n)
        return CompiledNotes(ids, [len(n) // 3 for n in notes], min_conf, preds)

    def note_memory(self, notes: CompiledNotes) -> tuple[torch.Tensor, torch.Tensor]:
        """Decoder memory built from notes only, plus its padding mask."""
        if self.note_mode == "shared_encoder":
            return self.base.encode(notes.note_ids), notes.note_ids == PAD
        B, n = notes.note_ids.shape
        triples = notes.note_ids.view(B, n // 3, 3)
        emb = self.base.tok(triples).flatten(2)  # (B, notes, 3d)
        # A slot exists when any of its three tokens is present, so content
        # ablations (anchor-only, relation+target-only) keep their slots.
        return self.note_encoder(emb), (triples == PAD).all(-1)


def chunk_labels(lectures: list[Lecture]) -> dict[str, torch.Tensor]:
    """Gold compiler labels in chunk order (-100 = not applicable)."""
    kind, anchor, value, link = [], [], [], []
    for lec in lectures:
        for c in lec.chunks:
            kind.append(KINDS.index(c.kind))
            anchor.append(ENTITIES.index(c.anchor) if c.anchor else -100)
            value.append(VALUES.index(c.target) if c.kind == "value" else -100)
            link.append(ENTITIES.index(c.target) if c.kind == "link" else -100)
    return {k: torch.tensor(v) for k, v in (("kind", kind), ("anchor", anchor), ("value", value), ("link", link))}
