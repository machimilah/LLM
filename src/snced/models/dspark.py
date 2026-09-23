"""DSpark: a speculative-decoding draft module.

A small draft head proposes a block of future tokens from the decoder's current
state; the main decoder then verifies the whole block in one pass and accepts
the longest prefix it agrees with. Rejected tokens cost nothing but the draft,
and the token the main model wanted is used instead, so **greedy output is
bit-identical to decoding without DSpark**. That identity is the correctness
criterion, and `verify_identical_to_greedy` tests it.

This is a throughput component, not a reasoning one. The evidence that matters
is systems evidence:

  accepted tokens per verification step   how often the draft is right
  tokens per second                        end-to-end speedup
  quality                                  must be unchanged, not merely close

Speedup only appears when the main decoder pass is expensive relative to the
draft, and when generations are long enough to amortise it. On short outputs
(this project's chains are 1-3 tokens) there is nothing to gain; the machinery
is here for the scale where there is.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class SpeculativeStats:
    generated: int = 0
    verification_steps: int = 0
    drafted: int = 0
    accepted: int = 0

    @property
    def accepted_per_step(self) -> float:
        return self.accepted / max(self.verification_steps, 1)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / max(self.drafted, 1)

    def as_dict(self) -> dict:
        return {"generated": self.generated, "verification_steps": self.verification_steps,
                "drafted": self.drafted, "accepted": self.accepted,
                "accepted_per_step": self.accepted_per_step,
                "acceptance_rate": self.acceptance_rate}


class DraftModule(nn.Module):
    """Proposes `block` future tokens from one decoder state.

    Deliberately cheap: a couple of small stages, not a second transformer. If
    drafting costs as much as verifying, speculative decoding cannot win.
    """

    def __init__(self, d_model: int, vocab_size: int, block: int = 5, d_hidden: int = 128,
                 stages: int = 3) -> None:
        super().__init__()
        self.block = block
        self.trunk = nn.Sequential(nn.Linear(d_model, d_hidden), nn.GELU())
        self.stages = nn.ModuleList([
            nn.Sequential(nn.Linear(d_hidden, d_hidden), nn.GELU()) for _ in range(stages)
        ])
        self.heads = nn.ModuleList([nn.Linear(d_hidden, vocab_size) for _ in range(block)])
        self.confidence = nn.Linear(d_hidden, block)

    def forward(self, state: torch.Tensor):
        """Returns (draft token ids (B, block), per-token confidence (B, block))."""
        h = self.trunk(state)
        for stage in self.stages:
            h = h + stage(h)
        tokens = torch.stack([head(h).argmax(-1) for head in self.heads], dim=1)
        return tokens, self.confidence(h).sigmoid()

    def loss(self, state: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Train the draft to predict the next `block` tokens. -100 = ignore."""
        h = self.trunk(state)
        for stage in self.stages:
            h = h + stage(h)
        total = state.new_zeros(())
        for i, head in enumerate(self.heads):
            if i < targets.size(1):
                total = total + nn.functional.cross_entropy(
                    head(h), targets[:, i], ignore_index=-100)
        return total


@torch.no_grad()
def speculative_decode(model, memory: torch.Tensor, mask: torch.Tensor, prompt: torch.Tensor,
                       steps: int, draft: DraftModule, min_confidence: float = 0.0):
    """Greedy decoding with block drafting and exact verification.

    `model` must expose `decode_all`, `decode_states` and `backbone.generate`
    (both CEDBaseline-backed models here do).

    Output is identical to plain greedy decoding: every accepted token is one
    the main model would have produced anyway, and the first disagreement is
    replaced by the main model's own choice.
    """
    seq = prompt
    stats = SpeculativeStats()
    while stats.generated < steps:
        states = model.decode_states(memory, mask, seq)
        last = model.head_of(states[:, -1]) if hasattr(model, "head_of") else model.decode_all(memory, mask, seq)[:, -1]
        tokens, conf = draft(states[:, -1])
        k = min(draft.block, steps - stats.generated)
        proposal = tokens[:, :k]
        if min_confidence > 0:
            keep = (conf[:, :k] >= min_confidence).cumprod(dim=1).bool()
            proposal = torch.where(keep, proposal, torch.full_like(proposal, -1))
        stats.drafted += int((proposal >= 0).sum())

        # Verify the whole block in one pass: feed the draft and compare with
        # what the main model would have chosen at each position.
        candidate = torch.cat([seq, proposal.clamp(min=0)], dim=1)
        verify = model.decode_all(memory, mask, candidate)
        wanted = torch.cat([last.argmax(-1, keepdim=True),
                            verify[:, seq.size(1):-1].argmax(-1)], dim=1)
        agree = (proposal == wanted) & (proposal >= 0)
        n_accept = int(agree.cumprod(dim=1).sum(dim=1).min())  # common prefix across the batch

        accepted = wanted[:, : n_accept + 1]  # accepted draft plus the model's own next token
        seq = torch.cat([seq, accepted], dim=1)
        stats.accepted += n_accept
        stats.verification_steps += 1
        stats.generated += accepted.size(1)
    return seq[:, prompt.size(1) : prompt.size(1) + steps], stats


@torch.no_grad()
def verify_identical_to_greedy(model, memory, mask, prompt, steps, draft) -> bool:
    """Correctness criterion: speculative output must equal greedy output."""
    plain = model.backbone.generate(memory, mask, prompt, steps)
    spec, _ = speculative_decode(model, memory, mask, prompt, steps, draft)
    return bool(torch.equal(plain, spec))
