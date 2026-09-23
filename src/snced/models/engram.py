"""Engram: a hashed n-gram associative memory gated into the residual stream.

Frequently reusable lexical and local-pattern knowledge does not need to be
recomputed by attention every time it appears. Engram stores it parametrically:
each position hashes its trailing 2-, 3- and 4-grams into large embedding
tables, and the retrieved vectors are gated into the hidden state.

It is a *performance* component, not part of the SN-CED hypothesis. It trades a
very large parameter and storage footprint for cheaper active computation, so
the only honest way to evaluate it is a matched ablation: same model, same
training budget, with and without, comparing quality per ACTIVE FLOP while
reporting the parameter cost separately. `active_flops_per_token` and
`parameter_count` exist for exactly that.

The gate is learned and initialised near zero, so a model that gains nothing
from Engram can switch it off rather than being harmed by it.
"""

from __future__ import annotations

import torch
from torch import nn

# Odd multipliers, one per n-gram order, for a cheap rolling polynomial hash.
HASH_MULTIPLIERS = (1_000_003, 2_000_003, 3_000_017)


class EngramMemory(nn.Module):
    def __init__(self, d_model: int, orders: tuple[int, ...] = (2, 3, 4),
                 table_size: int = 2 ** 16, gate_init: float = -3.0) -> None:
        super().__init__()
        self.orders, self.table_size = orders, table_size
        self.tables = nn.ModuleList([nn.Embedding(table_size, d_model) for _ in orders])
        for t in self.tables:
            nn.init.normal_(t.weight, std=0.02)
        # One gate per order, from the hidden state: the model decides per token
        # how much stored pattern knowledge to admit. Starts closed.
        self.gate = nn.Linear(d_model, len(orders))
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, gate_init)
        # Normalise each table's output BEFORE gating. LayerNorm is scale
        # invariant, so normalising the gated sum would cancel the gate: the
        # contribution stayed unit-sized however shut the gate was, and no
        # gradient about magnitude reached the tables.
        self.norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in orders])

    def hash_ngrams(self, ids: torch.Tensor, n: int, mult: int) -> torch.Tensor:
        """Hash the trailing n-gram ending at each position, into [0, table_size)."""
        B, T = ids.shape
        acc = torch.zeros_like(ids)
        for offset in range(n):
            shifted = torch.zeros_like(ids)
            if offset == 0:
                shifted = ids
            elif offset < T:
                shifted[:, offset:] = ids[:, :-offset]
            acc = acc * mult + shifted + 1
        return acc.abs() % self.table_size

    def forward(self, ids: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Add gated n-gram memory to the hidden state."""
        gates = self.gate(hidden).sigmoid()  # (B, T, orders)
        out = torch.zeros_like(hidden)
        for i, (n, mult) in enumerate(zip(self.orders, HASH_MULTIPLIERS)):
            keys = self.hash_ngrams(ids, n, mult)
            out = out + gates[..., i : i + 1] * self.norms[i](self.tables[i](keys))
        return hidden + out

    @property
    def parameter_count(self) -> int:
        """Storage footprint: the honest cost of this component."""
        return sum(p.numel() for p in self.parameters())

    def active_flops_per_token(self, d_model: int) -> float:
        """Compute actually spent per token: gating plus one lookup per order.

        Table lookups are memory traffic, not arithmetic, which is the whole
        point - the parameters are large but the active compute is tiny.
        """
        return 2 * d_model * len(self.orders) + len(self.orders) * d_model

    @torch.no_grad()
    def gate_usage(self, ids: torch.Tensor, hidden: torch.Tensor) -> dict[str, float]:
        """How open the gates actually are, per n-gram order."""
        g = self.gate(hidden).sigmoid().mean(dim=(0, 1))
        return {f"{n}gram_gate": float(g[i]) for i, n in enumerate(self.orders)}
