"""Scale components: MoE feed-forward, sliding-window and sparse attention.

These do not change what the memory does; they are what a frontier-scale model
needs to make the same computation affordable. They are implemented small and
switchable so their cost and effect can be measured here rather than assumed.

  MoE              top-k routed experts, so parameters grow without activating
                   them all per token (DeepSeek-style)
  sliding window   each token attends to the last `window` tokens
  sparse (top-k)   each token attends to its k highest-scoring keys
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of experts, with the usual load-balancing loss."""

    def __init__(self, d_model: int, d_ff: int, n_experts: int = 4, top_k: int = 1) -> None:
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.w_in = nn.Parameter(torch.empty(n_experts, d_model, d_ff))
        self.w_out = nn.Parameter(torch.empty(n_experts, d_ff, d_model))
        for w in (self.w_in, self.w_out):
            nn.init.normal_(w, std=d_model ** -0.5)
        self.last_balance_loss = torch.tensor(0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, d = x.shape
        flat = x.reshape(-1, d)
        probs = self.gate(flat).softmax(-1)
        weight, idx = probs.topk(self.top_k, dim=-1)
        out = torch.zeros_like(flat)
        for slot in range(self.top_k):
            experts = idx[:, slot]
            for e in range(self.n_experts):
                sel = experts == e
                if not sel.any():
                    continue
                h = F.gelu(flat[sel] @ self.w_in[e]) @ self.w_out[e]
                out[sel] += weight[:, slot : slot + 1][sel] * h
        # Load balancing: tokens per expert should match the routing mass.
        frac = torch.zeros(self.n_experts, device=x.device)
        frac.scatter_add_(0, idx[:, 0], torch.ones_like(idx[:, 0], dtype=frac.dtype))
        frac = frac / max(flat.size(0), 1)
        self.last_balance_loss = (frac * probs.mean(0)).sum() * self.n_experts
        return out.view(B, T, d)

    @property
    def active_fraction(self) -> float:
        """Share of expert parameters touched per token."""
        return self.top_k / self.n_experts


def sliding_window_mask(n: int, window: int, device: torch.device) -> torch.Tensor:
    """Causal mask that also forgets anything older than `window` tokens."""
    i = torch.arange(n, device=device)
    far = (i[:, None] - i[None, :]) >= window
    future = torch.ones(n, n, dtype=torch.bool, device=device).triu(1)
    return far | future


def topk_sparse_mask(scores: torch.Tensor, k: int, causal: bool = True) -> torch.Tensor:
    """Keep each query's k best keys, so attention cost is k rather than T.

    `scores` is (B, heads, T, S); returns a boolean mask where True = blocked.
    """
    B, H, T, S = scores.shape
    if causal:
        scores = scores.masked_fill(torch.ones(T, S, dtype=torch.bool, device=scores.device).triu(1),
                                    torch.finfo(scores.dtype).min)
    k = min(k, S)
    keep_idx = scores.topk(k, dim=-1).indices
    mask = torch.ones(B, H, T, S, dtype=torch.bool, device=scores.device)
    mask.scatter_(-1, keep_idx, False)
    if causal:
        mask |= torch.ones(T, S, dtype=torch.bool, device=scores.device).triu(1)
    return mask
