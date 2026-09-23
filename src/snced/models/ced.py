"""Small Causal Encoder-Decoder baseline (plan sections 14 and P2).

Causal encoder over the full lecture -> decoder global memory projected from
the final encoder hidden states -> decoder over the question (causal
self-attention + cross-attention to memory) -> answer class at the last
question position.

Self-attention uses rotary position embeddings, and encoder blocks can add a
short causal depthwise convolution (conv_kernel > 0). Attention-only variants
sat on the "answer any value in context" plateau for thousands of steps; the
convolution lets a value token see its entity from step one
(results/logs/FAILED_RUNS.md).
Sequences are right-padded, so causal masking alone keeps real positions from
seeing padding; cross-attention masks padded memory explicitly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from ..data import N_VALUES, PAD
from .engram import EngramMemory


def rope_tables(n: int, head_dim: int, device: torch.device, base: float = 10000.0):
    inv = 1.0 / base ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim)
    ang = torch.outer(torch.arange(n, device=device).float(), inv)
    return ang.cos(), ang.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = x[..., ::2], x[..., 1::2]
    out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
    return out.flatten(-2)


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, rope: bool) -> None:
        super().__init__()
        self.h, self.dh, self.rope = n_heads, d_model // n_heads, rope
        self.q = nn.Linear(d_model, d_model)
        self.kv = nn.Linear(d_model, 2 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, src, causal: bool, key_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, D = x.shape
        q = self.q(x).view(B, T, self.h, self.dh).transpose(1, 2)
        k, v = self.kv(src).view(B, src.size(1), 2, self.h, self.dh).permute(2, 0, 3, 1, 4)
        if self.rope:
            cos, sin = rope_tables(T, self.dh, x.device)
            q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        mask = key_mask[:, None, None, :] if key_mask is not None else None
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=causal)
        return self.out(y.transpose(1, 2).reshape(B, T, D))


class CausalConv(nn.Module):
    """Short causal depthwise convolution: each position mixes the previous k-1 tokens."""

    def __init__(self, d_model: int, kernel: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.conv = nn.Conv1d(d_model, d_model, kernel, groups=d_model, padding=kernel - 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(self.norm(x).transpose(1, 2))[..., : x.size(1)]
        return x + y.transpose(1, 2)


class Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, cross: bool, conv_kernel: int = 0) -> None:
        super().__init__()
        self.conv = CausalConv(d_model, conv_kernel) if conv_kernel else None
        self.n1 = nn.LayerNorm(d_model)
        self.self_attn = Attention(d_model, n_heads, rope=True)
        self.cross = cross
        if cross:
            self.n2 = nn.LayerNorm(d_model)
            self.cross_attn = Attention(d_model, n_heads, rope=False)
        self.n3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, memory=None, memory_mask=None):
        if self.conv is not None:
            x = self.conv(x)
        h = self.n1(x)
        x = x + self.self_attn(h, h, causal=True)
        if self.cross:
            x = x + self.cross_attn(self.n2(x), memory, causal=False, key_mask=memory_mask)
        return x + self.ff(self.n3(x))


class CEDBaseline(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_enc: int = 2,
        n_dec: int = 3,
        d_ff: int = 128,
        conv_kernel: int = 0,
        generative: bool = False,
        engram: bool = False,
        engram_table_size: int = 2 ** 16,
    ) -> None:
        super().__init__()
        self.d_model, self.n_dec, self.generative = d_model, n_dec, generative
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=PAD)
        self.encoder = nn.ModuleList(
            Block(d_model, n_heads, d_ff, cross=False, conv_kernel=conv_kernel) for _ in range(n_enc)
        )
        # Optional parametric n-gram memory, gated into the residual stream.
        # Off by default: it is a performance component, and its value has to be
        # shown by a matched ablation (experiments/synthetic/engram_ablation.py).
        self.engram = EngramMemory(d_model, table_size=engram_table_size) if engram else None
        self.enc_norm = nn.LayerNorm(d_model)
        self.mem_proj = nn.Linear(d_model, d_model)
        self.decoder = nn.ModuleList(Block(d_model, n_heads, d_ff, cross=True) for _ in range(n_dec))
        self.dec_norm = nn.LayerNorm(d_model)
        # Classification: answer value at the last question token.
        # Generative: next-token logits over the vocabulary at every position.
        self.head = nn.Linear(d_model, vocab_size if generative else N_VALUES)

    def encode_states(self, ctx: torch.Tensor) -> torch.Tensor:
        """Final encoder hidden states (what the note compiler reads)."""
        h = self.tok(ctx)
        if self.engram is not None:
            h = self.engram(ctx, h)
        for blk in self.encoder:
            h = blk(h)
        return self.enc_norm(h)

    def encode(self, ctx: torch.Tensor) -> torch.Tensor:
        """Decoder memory projected from the final encoder states."""
        return self.mem_proj(self.encode_states(ctx))

    def decode_states(self, memory: torch.Tensor, ctx_pad: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        """Final decoder hidden states, before the output head (used by DSpark)."""
        x = self.tok(seq)
        for blk in self.decoder:
            x = blk(x, memory, ~ctx_pad)
        return self.dec_norm(x)

    def decode_all(self, memory: torch.Tensor, ctx_pad: torch.Tensor, seq: torch.Tensor) -> torch.Tensor:
        return self.head(self.decode_states(memory, ctx_pad, seq))

    def decode(self, memory: torch.Tensor, ctx_pad: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        logits = self.decode_all(memory, ctx_pad, q)
        last = (q != PAD).sum(1) - 1
        return logits[torch.arange(q.size(0)), last]

    @torch.no_grad()
    def generate(self, memory: torch.Tensor, ctx_pad: torch.Tensor, prompt: torch.Tensor, n_steps: int) -> torch.Tensor:
        """Greedy decoding. Prompts in one call must share a length (no padding)."""
        seq = prompt
        for _ in range(n_steps):
            nxt = self.decode_all(memory, ctx_pad, seq)[:, -1].argmax(-1, keepdim=True)
            seq = torch.cat([seq, nxt], dim=1)
        return seq[:, prompt.size(1) :]

    def forward(self, ctx: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(ctx), ctx == PAD, q)

    def memory_bytes(self, ctx_len: int, bytes_per_el: int = 4) -> int:
        """Detailed-memory footprint: cross-attention K and V for every decoder layer."""
        return 2 * self.n_dec * ctx_len * self.d_model * bytes_per_el
