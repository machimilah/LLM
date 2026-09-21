"""Memory tiers and indexers (plan sections 4, 7, 8).

Three tiers, cheapest first:

  local     the most recent tokens, always available, no retrieval needed
  notes     Semantic Note Memory, compiled once per document
  detailed  token-level memory, compressed, opened only when asked for

Indexers are what make this scale. Attending over every note is fine for 24
notes; a long-lived agent holds thousands, and then the decoder must *retrieve*
rather than attend to everything. Both indexers score memory against the query
and return the top-k, so the decoder's cost stops growing with the notebook.
"""

from __future__ import annotations

import torch
from torch import nn


class LocalMemory(nn.Module):
    """The last `window` tokens of the context, kept verbatim.

    Recency is the one thing notes are bad at: a fact from two sentences ago has
    not been consolidated yet. This tier costs nothing to build - it is a slice.
    """

    def __init__(self, window: int = 64) -> None:
        super().__init__()
        self.window = window

    def forward(self, detailed: torch.Tensor, pad_mask: torch.Tensor):
        lengths = (~pad_mask).sum(1)
        B, T, d = detailed.shape
        w = min(self.window, T)
        out = detailed.new_zeros(B, w, d)
        mask = torch.ones(B, w, dtype=torch.bool, device=detailed.device)
        for b in range(B):
            end = int(lengths[b])
            start = max(0, end - w)
            n = end - start
            if n > 0:
                out[b, :n] = detailed[b, start:end]
                mask[b, :n] = False
        return out, mask


class CompressedDetailMemory(nn.Module):
    """Token-level memory at reduced resolution (plan: "compressed detailed memory").

    Mean-pools every `stride` tokens and projects, so the detailed tier costs
    1/stride of the slots. Exact quotation still needs the raw source, which is
    what the note source pointers are for.
    """

    def __init__(self, d_model: int, stride: int = 4) -> None:
        super().__init__()
        self.stride = stride
        self.proj = nn.Sequential(nn.Linear(d_model, d_model), nn.GELU(), nn.LayerNorm(d_model))

    def forward(self, detailed: torch.Tensor, pad_mask: torch.Tensor):
        B, T, d = detailed.shape
        pad = (-T) % self.stride
        if pad:
            detailed = torch.cat([detailed, detailed.new_zeros(B, pad, d)], 1)
            pad_mask = torch.cat([pad_mask, pad_mask.new_ones(B, pad)], 1)
        keep = (~pad_mask).float().unsqueeze(-1)
        chunks = (detailed * keep).view(B, -1, self.stride, d).sum(2)
        counts = keep.view(B, -1, self.stride, 1).sum(2).clamp(min=1)
        pooled = self.proj(chunks / counts)
        mask = pad_mask.view(B, -1, self.stride).all(-1)
        return pooled, mask


class Indexer(nn.Module):
    """Retrieve the top-k most relevant memory slots for a query.

    Scores every slot against the query and keeps the best k, so decoder cost
    depends on k rather than on how much has been remembered. Returns the
    selection scores too, so the router can see how good the best matches were.
    """

    def __init__(self, d_model: int, top_k: int = 16) -> None:
        super().__init__()
        self.top_k = top_k
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def score(self, query_vec: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        s = torch.einsum("bd,bnd->bn", self.query(query_vec), self.key(memory)) * self.scale
        return s.masked_fill(mask, torch.finfo(s.dtype).min)

    def forward(self, query_vec: torch.Tensor, memory: torch.Tensor, mask: torch.Tensor,
                top_k: int | None = None):
        k = min(top_k or self.top_k, memory.size(1))
        scores = self.score(query_vec, memory, mask)
        idx = scores.topk(k, dim=1).indices
        gathered = memory.gather(1, idx.unsqueeze(-1).expand(-1, -1, memory.size(-1)))
        gathered_mask = mask.gather(1, idx)
        # Keep the retrieved slots differentiable w.r.t. their own scores.
        weights = scores.gather(1, idx).softmax(-1).unsqueeze(-1)
        return gathered * (1 + weights - weights.detach()), gathered_mask, scores.gather(1, idx)
