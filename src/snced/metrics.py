"""Aggregation helpers for multi-seed reporting (plan section 19)."""

from __future__ import annotations

import statistics


def mean_std(xs: list[float]) -> tuple[float, float]:
    if not xs:
        return float("nan"), float("nan")
    return statistics.fmean(xs), (statistics.stdev(xs) if len(xs) > 1 else 0.0)


def encoder_flops(T: float, d: int, d_ff: int, n_layers: int, conv_kernel: int = 0) -> float:
    """Approximate forward FLOPs of the causal encoder over T tokens (matmuls + conv)."""
    per_token = 2 * (4 * d * d + 2 * d * d_ff) + 2 * d * conv_kernel + 2 * T * d  # causal attn ~ T/2 keys
    return n_layers * T * per_token


def decoder_memory_flops(S: float, d: int, n_layers: int) -> float:
    """Projecting S memory positions into cross-attention K and V, once per lecture."""
    return n_layers * S * 2 * (2 * d * d)


def decoder_query_flops(Lq: float, S: float, d: int, d_ff: int, n_layers: int) -> float:
    """One question: self-attention, cross-attention over S memory slots, feed-forward."""
    self_attn = Lq * 2 * 4 * d * d + 2 * Lq * Lq * d
    cross = Lq * 2 * 2 * d * d + 2 * 2 * Lq * S * d
    ff = Lq * 2 * 2 * d * d_ff
    return n_layers * (self_attn + cross + ff)


def fmt(xs: list[float], pct: bool = False, digits: int = 2) -> str:
    m, s = mean_std(xs)
    k = 100.0 if pct else 1.0
    unit = "%" if pct else ""
    return f"{m * k:.{digits}f}{unit} ± {s * k:.{digits}f}"
