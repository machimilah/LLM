"""Hard-concrete (L0) gates for note selection.

The first compression penalty multiplied each note slot by a sigmoid gate and
penalised the sum of those gates. The model learned to game it: every gate
settled at ~0.44 (spread 0.007) while the note encoder scaled its outputs up to
compensate, so the penalty fell without a single note being dropped.

An L0 gate cannot be gamed that way. The penalty is the probability that a gate
is non-zero, not its magnitude, so shrinking a gate towards 0.5 saves nothing -
only actually closing it does. At evaluation the gate is deterministic and
genuinely sparse: closed notes contribute nothing and can be dropped from the
memory entirely.

Louizos, Welling & Kingma (2018), "Learning Sparse Neural Networks through L0
Regularization".
"""

from __future__ import annotations

import math

import torch

BETA = 2.0 / 3.0
GAMMA = -0.1
ZETA = 1.1
EPS = 1e-6
# Gates saturate open during the penalty warm-up and then cannot be closed: at
# log_alpha >> 0 the sigmoid gradient vanishes, so no compression weight moves
# them (measured: every gate exactly 1.0, no pruning at lambda=16). Clamping the
# logit keeps a usable gradient in both directions.
LOG_ALPHA_LIMIT = 4.0


def clamp_logit(log_alpha: torch.Tensor) -> torch.Tensor:
    """Keep the gate logit inside a range where its gradient survives."""
    return log_alpha.clamp(-LOG_ALPHA_LIMIT, LOG_ALPHA_LIMIT)


def sample_gate(log_alpha: torch.Tensor, training: bool) -> torch.Tensor:
    """Stretched-concrete sample in training, deterministic gate at evaluation."""
    log_alpha = clamp_logit(log_alpha)
    if training:
        u = torch.rand_like(log_alpha).clamp(EPS, 1 - EPS)
        s = torch.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / BETA)
    else:
        s = torch.sigmoid(log_alpha)
    return (s * (ZETA - GAMMA) + GAMMA).clamp(0.0, 1.0)


def gate_probability(log_alpha: torch.Tensor) -> torch.Tensor:
    """P(gate > 0): the differentiable stand-in for "this note is kept"."""
    return torch.sigmoid(clamp_logit(log_alpha) - BETA * math.log(-GAMMA / ZETA))
