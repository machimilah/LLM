"""Note encoder: note fields -> decoder memory slots (plan section 4).

Kept as its own module because re-encoding notes with the lecture encoder does
not work: the encoder is trained on prose and gives packed note text unusable
states (32% vs 99.4%, results/logs/FAILED_RUNS.md).

Each note becomes one memory slot built from its anchor, its micro-context, and
its metadata - source pointer, order in the document, confidence, and whether it
supersedes an earlier note about the same anchor. Order and supersession enter
the slot itself, so temporal updates are representable rather than collapsed.
"""

from __future__ import annotations

import torch
from torch import nn

from .note_compiler import NoteFields


class NoteEncoder(nn.Module):
    def __init__(self, d_model: int, context_heads: int = 2, d_hidden: int = 128,
                 max_order: int = 512) -> None:
        super().__init__()
        self.order_emb = nn.Embedding(max_order, d_model)
        self.meta = nn.Linear(3, d_model)  # confidence, supersedes, normalised source position
        fields = d_model * (3 + context_heads)  # anchor + k x context + order + meta
        self.mlp = nn.Sequential(
            nn.Linear(fields, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_model),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, fields: NoteFields, context_length: int | None = None):
        """Returns (memory slots (B, N, d), padding mask (B, N))."""
        order = fields.order.long().clamp(max=self.order_emb.num_embeddings - 1)
        denom = max(context_length or 1, 1)
        meta = torch.stack([fields.confidence, fields.supersedes, fields.source / denom], dim=-1)
        x = torch.cat([fields.anchor, fields.context, self.order_emb(order), self.meta(meta)], dim=-1)
        slots = self.norm(self.mlp(x))
        # Importance gates the slot: unimportant segments contribute nothing, and
        # the gate is differentiable so the compression penalty can shrink it.
        slots = slots * fields.importance.unsqueeze(-1)
        return slots, fields.mask
