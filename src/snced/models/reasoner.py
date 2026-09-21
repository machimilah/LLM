"""Shared neural reasoner for the representation ablation (plan section 11.1).

The same architecture and hyperparameters are trained on every memory
condition; only the memory tokens it reads change.
"""

from __future__ import annotations

import torch
from torch import nn

from ..data import N_VALUES, PAD


class Reasoner(nn.Module):
    def __init__(self, vocab_size: int, d_emb: int = 32, d_hidden: int = 56) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=PAD)
        self.gru = nn.GRU(d_emb, d_hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(2 * d_hidden, d_hidden), nn.ReLU(), nn.Linear(d_hidden, N_VALUES))

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        return self.head(torch.cat([h[-2], h[-1]], dim=-1))
