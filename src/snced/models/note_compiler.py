"""Open-vocabulary Semantic Note Compiler (plan section 6).

Reads the final causal-encoder states once, before any question exists, and
emits notes. A note is

    (anchor, micro-context, source pointer, confidence, order)

where anchor and micro-context are *pointers into the source*, not labels from
a closed set. Pointers are soft during training (attention over the segment's
token states), so the whole compiler is differentiable and can be trained from
the task and sufficiency losses without gold note labels; taking the argmax of
the same distributions recovers the note as text for inspection.

Importance gating decides which segments deserve a note at all, which is what
makes the notebook sparse; the expected number of notes is returned so a
compression penalty can be applied to it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .gates import gate_probability, sample_gate


@dataclass
class NoteFields:
    """Per-note content, before the note encoder turns it into memory slots."""

    anchor: torch.Tensor  # (B, N, d) soft-pointer readout of the anchor
    context: torch.Tensor  # (B, N, k*d) micro-context readout (k pointer heads)
    importance: torch.Tensor  # (B, N) L0 gate value; 0 means the note is dropped
    keep_prob: torch.Tensor  # (B, N) P(gate > 0), what the compression penalty charges for
    confidence: torch.Tensor  # (B, N) note confidence, used by the router and fallback
    supersedes: torch.Tensor  # (B, N) probability this note updates an earlier one
    order: torch.Tensor  # (B, N) note index in document order (temporal ordering)
    source: torch.Tensor  # (B, N) first token of the segment the note came from
    source_end: torch.Tensor  # (B, N) one past the last token of that segment
    mask: torch.Tensor  # (B, N) True where a note slot is padding
    anchor_pos: torch.Tensor  # (B, N) argmax anchor position, for readable notes
    context_pos: torch.Tensor  # (B, N, k) argmax micro-context positions
    anchor_w: torch.Tensor  # (B, N, L) anchor pointer distribution
    context_w: torch.Tensor  # (B, N, L, k) micro-context pointer distributions

    def pointer_overlap(self) -> torch.Tensor:
        """How much the anchor and micro-context point at the same tokens.

        Without pressure to separate, both pointers happily select the same
        word and the note degenerates to "v5 -> v5 v5": the fact without the
        thing it belongs to.
        """
        overlap = torch.einsum("bnl,bnlk->bnk", self.anchor_w, self.context_w).mean(-1)
        keep = (~self.mask).float()
        return (overlap * keep).sum() / keep.sum().clamp(min=1)

    @property
    def expected_notes(self) -> torch.Tensor:
        """Differentiable note count: sum of P(gate > 0), not of gate magnitudes.

        Charging magnitudes let the model shrink every gate uniformly and rescale
        the slots to compensate, which reduced the penalty without dropping a
        single note (see models/gates.py).
        """
        return (self.keep_prob * (~self.mask)).sum(-1)

    @property
    def kept_notes(self) -> torch.Tensor:
        """Notes actually surviving the gate, countable at evaluation."""
        return ((self.importance > 0) & (~self.mask)).float().sum(-1)


def segment_by_token(ctx: torch.Tensor, boundary_id: int, pad_id: int = 0) -> list[list[tuple[int, int]]]:
    """Split each sequence at a boundary token (e.g. sentence end)."""
    out = []
    for row in ctx:
        bounds, start = [], 0
        for pos in (row == boundary_id).nonzero().squeeze(1).tolist():
            bounds.append((start, pos + 1))
            start = pos + 1
        end = int((row != pad_id).sum())
        if start < end:
            bounds.append((start, end))
        out.append(bounds or [(0, end)])
    return out


def segment_fixed(ctx: torch.Tensor, window: int, pad_id: int = 0) -> list[list[tuple[int, int]]]:
    """Fixed-width segmentation, for text with no usable boundary token."""
    out = []
    for row in ctx:
        end = int((row != pad_id).sum())
        out.append([(s, min(s + window, end)) for s in range(0, max(end, 1), window)])
    return out


class OpenVocabNoteCompiler(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 128, context_heads: int = 2,
                 max_notes: int | None = None, gate_init: float = 2.0) -> None:
        super().__init__()
        self.context_heads, self.max_notes = context_heads, max_notes
        self.seg_query = nn.Parameter(torch.randn(d_hidden) * 0.02)
        self.proj = nn.Linear(d_model, d_hidden)
        # Pointer scorers: one for the anchor, `context_heads` for the micro-context.
        self.anchor_ptr = nn.Linear(d_hidden, 1)
        self.context_ptr = nn.Linear(d_hidden, context_heads)
        self.summary = nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU())
        self.importance = nn.Linear(d_hidden, 1)
        # Open the gates at initialisation. Closed gates hand the notes path an
        # empty memory, and that loss damages the shared decoder the moment it
        # unfreezes - the detailed reference path fell from 100% to ~45%.
        nn.init.constant_(self.importance.bias, gate_init)
        self.confidence = nn.Linear(d_hidden, 1)
        self.supersedes = nn.Linear(d_hidden, 1)

    @staticmethod
    def gather_segments(states: torch.Tensor, segments: list[list[tuple[int, int]]]):
        """Pack variable-length segments into (B, N, L, d) with a padding mask."""
        B, _, d = states.shape
        n_max = max(len(s) for s in segments)
        l_max = max((b - a) for segs in segments for a, b in segs)
        packed = states.new_zeros(B, n_max, l_max, d)
        tok_mask = states.new_ones(B, n_max, l_max, dtype=torch.bool)  # True = pad
        note_mask = states.new_ones(B, n_max, dtype=torch.bool)
        offsets = states.new_zeros(B, n_max, dtype=torch.long)
        ends = states.new_zeros(B, n_max, dtype=torch.long)
        for b, segs in enumerate(segments):
            for i, (a, e) in enumerate(segs):
                packed[b, i, : e - a] = states[b, a:e]
                tok_mask[b, i, : e - a] = False
                note_mask[b, i] = False
                offsets[b, i], ends[b, i] = a, e
        return packed, tok_mask, note_mask, offsets, ends

    def forward(self, states: torch.Tensor, segments: list[list[tuple[int, int]]]) -> NoteFields:
        packed, tok_mask, note_mask, offsets, ends = self.gather_segments(states, segments)
        B, N, L, _ = packed.shape
        h = self.proj(packed)  # (B, N, L, d_hidden)

        neg = torch.finfo(h.dtype).min
        anchor_logits = self.anchor_ptr(h).squeeze(-1).masked_fill(tok_mask, neg)
        anchor_w = anchor_logits.softmax(-1)
        anchor = torch.einsum("bnl,bnld->bnd", anchor_w, packed)

        ctx_logits = self.context_ptr(h).masked_fill(tok_mask.unsqueeze(-1), neg)  # (B,N,L,k)
        ctx_w = ctx_logits.softmax(-2)
        context = torch.einsum("bnlk,bnld->bnkd", ctx_w, packed).flatten(2)

        pooled = self.summary((h * (~tok_mask).unsqueeze(-1)).sum(2) / (~tok_mask).sum(2, keepdim=True).clamp(min=1))
        log_alpha = self.importance(pooled).squeeze(-1)
        gate = sample_gate(log_alpha, self.training)
        order = torch.arange(N, device=states.device).expand(B, N).float()
        return NoteFields(
            anchor=anchor,
            context=context,
            importance=gate.masked_fill(note_mask, 0.0),
            keep_prob=gate_probability(log_alpha).masked_fill(note_mask, 0.0),
            confidence=self.confidence(pooled).squeeze(-1).sigmoid().masked_fill(note_mask, 0.0),
            supersedes=self.supersedes(pooled).squeeze(-1).sigmoid().masked_fill(note_mask, 0.0),
            order=order,
            source=offsets.float(),
            source_end=ends.float(),
            mask=note_mask,
            anchor_pos=anchor_logits.argmax(-1) + offsets,
            context_pos=ctx_logits.argmax(-2) + offsets.unsqueeze(-1),
            anchor_w=anchor_w,
            context_w=ctx_w,
        )

    @staticmethod
    def readable(fields: NoteFields, ctx: torch.Tensor, itos: list[str], b: int = 0,
                 threshold: float = 0.5) -> list[dict]:
        """The notebook as text: what the pointers actually selected."""
        out = []
        fields = NoteFields(**{k: (v.detach() if torch.is_tensor(v) else v) for k, v in vars(fields).items()})
        for i in range(fields.mask.size(1)):
            if fields.mask[b, i] or fields.importance[b, i] <= threshold:
                continue
            anchor = itos[int(ctx[b, fields.anchor_pos[b, i]])]
            context = [itos[int(ctx[b, p])] for p in fields.context_pos[b, i].tolist()]
            out.append({"anchor": anchor, "micro_context": " ".join(context),
                        "source": int(fields.source[b, i]), "order": int(fields.order[b, i]),
                        "confidence": float(fields.confidence[b, i]),
                        "gate": float(fields.importance[b, i]),
                        "supersedes": float(fields.supersedes[b, i])})
        return out
