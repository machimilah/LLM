"""Learned query-independent Semantic Note Compiler (plan sections 6 and 11.2).

A small causal GRU reads every chunk of a lecture before any question exists.
Heads predict chunk type, anchor entity, value target and link target; the
predictions are turned into a Notebook of anchor + micro-context notes.
"""

from __future__ import annotations

import torch
from torch import nn

from .data import ENTITIES, N_ENTITY_POOL, N_VALUES, PAD, VALUES, Lecture, Vocab
from .fallback import chunk_ref
from .notes import Notebook, link_note, value_note

KINDS = ("filler", "value", "link")


class NoteCompiler(nn.Module):
    def __init__(self, vocab_size: int, d_emb: int = 24, d_hidden: int = 60) -> None:
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_emb, padding_idx=PAD)
        self.gru = nn.GRU(d_emb, d_hidden, batch_first=True)  # unidirectional = causal
        self.kind_head = nn.Linear(d_hidden, len(KINDS))
        self.anchor_head = nn.Linear(d_hidden, N_ENTITY_POOL)
        self.value_head = nn.Linear(d_hidden, N_VALUES)
        self.link_head = nn.Linear(d_hidden, N_ENTITY_POOL)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
        packed = nn.utils.rnn.pack_padded_sequence(
            self.emb(ids), lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, h = self.gru(packed)
        h = h[-1]
        return {
            "kind": self.kind_head(h),
            "anchor": self.anchor_head(h),
            "value": self.value_head(h),
            "link": self.link_head(h),
        }


def chunk_batch(vocab: Vocab, lectures: list[Lecture]) -> dict[str, torch.Tensor]:
    """Flatten lectures into per-chunk tensors with gold labels (-100 = ignore)."""
    seqs, kind, anchor, value, link = [], [], [], [], []
    for lec in lectures:
        for c in lec.chunks:
            seqs.append(vocab.encode(c.tokens))
            kind.append(KINDS.index(c.kind))
            anchor.append(ENTITIES.index(c.anchor) if c.anchor else -100)
            value.append(VALUES.index(c.target) if c.kind == "value" else -100)
            link.append(ENTITIES.index(c.target) if c.kind == "link" else -100)
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return {
        "ids": ids,
        "lengths": lengths,
        "kind": torch.tensor(kind),
        "anchor": torch.tensor(anchor),
        "value": torch.tensor(value),
        "link": torch.tensor(link),
    }


def compiler_loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    ce = nn.functional.cross_entropy
    loss = ce(out["kind"], batch["kind"])
    for head in ("anchor", "value", "link"):
        if (batch[head] != -100).any():
            loss = loss + ce(out[head], batch[head], ignore_index=-100)
    return loss


@torch.no_grad()
def compile_notebook(model: NoteCompiler, vocab: Vocab, lecture: Lecture) -> tuple[Notebook, list[dict]]:
    """Compile one lecture into notes. Returns the notebook and per-chunk predictions."""
    model.eval()
    b = chunk_batch(vocab, [lecture])
    out = model(b["ids"], b["lengths"])
    probs = {k: v.softmax(-1) for k, v in out.items()}
    nb = Notebook()
    preds = []
    for i in range(len(lecture.chunks)):
        kind = KINDS[int(probs["kind"][i].argmax())]
        anchor = ENTITIES[int(probs["anchor"][i].argmax())]
        conf = float(probs["kind"][i].max() * probs["anchor"][i].max())
        pred = {"kind": kind, "anchor": None, "target": None}
        if kind == "value":
            target = VALUES[int(probs["value"][i].argmax())]
            conf *= float(probs["value"][i].max())
            nb.add(value_note(anchor, target, chunk_ref(i), conf))
            pred.update(anchor=anchor, target=target)
        elif kind == "link":
            target = ENTITIES[int(probs["link"][i].argmax())]
            conf *= float(probs["link"][i].max())
            nb.add(link_note(anchor, target, chunk_ref(i), conf))
            pred.update(anchor=anchor, target=target)
        preds.append(pred)
    return nb, preds
