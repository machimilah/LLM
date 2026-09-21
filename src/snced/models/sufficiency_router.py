"""Learned sufficiency router (plan sections 8 and 10).

Semantic Note Memory is the default path. The router looks at the query and at
the notebook it would reason from, and predicts whether the notes are
sufficient; only when they are not does the decoder additionally get the
detailed memory. Detailed access carries an explicit cost in the loss, so the
router has to justify every access by the task loss it saves.

This replaces the earlier rule: walk the question's chain through the notes and
compare the minimum confidence with a hand-set threshold. That worked only
because the synthetic task had a known chain structure.
"""

from __future__ import annotations

import torch
from torch import nn


class SufficiencyRouter(nn.Module):
    def __init__(self, d_model: int, d_hidden: int = 64) -> None:
        super().__init__()
        self.query_pool = nn.Linear(d_model, d_model)
        self.note_pool = nn.Linear(d_model, d_model)
        self.score = nn.Sequential(
            nn.Linear(2 * d_model + 2, d_hidden), nn.GELU(), nn.Linear(d_hidden, 1),
        )
        # Which notes to reread, when rereading is needed at all.
        self.relevance = nn.Bilinear(d_model, d_model, 1)

    def note_relevance(self, query_states: torch.Tensor, query_mask: torch.Tensor,
                       note_slots: torch.Tensor, note_mask: torch.Tensor) -> torch.Tensor:
        """Score every note against the query, for targeted source retrieval."""
        q = (self.query_pool(query_states) * query_mask.unsqueeze(-1)).sum(1) / query_mask.sum(1, keepdim=True).clamp(min=1)
        q = q.unsqueeze(1).expand_as(note_slots)
        scores = self.relevance(q.contiguous(), note_slots.contiguous()).squeeze(-1)
        return scores.masked_fill(note_mask, torch.finfo(scores.dtype).min)

    def forward(self, query_states: torch.Tensor, query_mask: torch.Tensor,
                note_slots: torch.Tensor, note_mask: torch.Tensor,
                confidence: torch.Tensor) -> torch.Tensor:
        """Probability that the DETAILED memory is needed for this query."""
        q = (self.query_pool(query_states) * query_mask.unsqueeze(-1)).sum(1) / query_mask.sum(1, keepdim=True).clamp(min=1)
        keep = (~note_mask).float().unsqueeze(-1)
        n = (self.note_pool(note_slots) * keep).sum(1) / keep.sum(1).clamp(min=1)
        # Two scalar summaries of the notebook: how confident and how large it is.
        conf = (confidence * (~note_mask)).sum(1, keepdim=True) / (~note_mask).sum(1, keepdim=True).clamp(min=1)
        size = (~note_mask).float().sum(1, keepdim=True) / note_mask.size(1)
        return self.score(torch.cat([q, n, conf, size], dim=-1)).squeeze(-1).sigmoid()


def access_cost(p_detail: torch.Tensor, fetched_slots: torch.Tensor, note_slots: float) -> torch.Tensor:
    """Expected extra memory paid for, in slots, relative to notes-only.

    Charging actual slots fetched keeps the penalty honest: rereading three
    sentences must cost far less than rereading a 2,700-token document, which
    is the whole point of keeping a source pointer on every note.
    """
    if not torch.is_tensor(fetched_slots):
        fetched_slots = torch.as_tensor(float(fetched_slots), device=p_detail.device)
    return (p_detail * fetched_slots).mean() / max(note_slots, 1.0)
