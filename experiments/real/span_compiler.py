"""Open-vocabulary span-based Semantic Note Compiler (plan sections 6, 20, P7).

Token-level BIO tagging over a frozen MiniLM: every sentence of a document is
tagged for ANCHOR / RELATION / TARGET spans, plus sentence-level "is this a
fact" and "is it negated". Notes are the extracted strings, so the compiler is
not tied to any entity list. It runs before any question is known and keeps
document order and a source pointer per note.

Trained on bAbI stories whose names, places and objects are randomly
substituted, inside WikiText prose. Evaluated on real BABILong documents (book
prose) both with the original cast and with entities the compiler never saw.

Usage: python experiments/real/span_compiler.py [--train-docs 3000]
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402

import torch  # noqa: E402
from notes_data import split_sentences  # noqa: E402
from span_data import (  # noqa: E402
    build_span_document, canonical_relation, find_span_facts, load_noise, mined_vocab, rename_document,
)
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from snced.runlog import RunRecord, count_params, seed_everything  # noqa: E402

EXPERIMENT = "span_compiler"
# Backbone for the span tagger. MiniLM-L6 (22M, 384-dim) plateaued at 82% retention on real
# documents with training loss 0.0000 - a generalisation limit, not a compute one
# (results/tables/stage0_gpu.md), so the encoder is selectable.
ENCODER_ID = os.environ.get("SNCED_ENCODER", "sentence-transformers/all-MiniLM-L6-v2")
TAGS = ("O", "B-ANCHOR", "I-ANCHOR", "B-REL", "I-REL", "B-TARGET", "I-TARGET")
TAG_ID = {t: i for i, t in enumerate(TAGS)}
TASKS = ("qa1", "qa2", "qa9")


@dataclass(frozen=True)
class SpanNote:
    anchor: str
    relation: str
    target: str
    negated: bool
    source: int

    @property
    def semantic(self) -> str:
        return canonical_relation(self.relation, self.negated)

    @property
    def text(self) -> str:
        return f"{self.anchor} {'not ' if self.negated else ''}{self.relation} {self.target}".replace("  ", " ")

    @property
    def tokens(self) -> list[str]:
        return self.text.split()


class TokenEncoder:
    def __init__(self, threads: int) -> None:
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(ENCODER_ID)
        self.model = AutoModel.from_pretrained(ENCODER_ID).eval()
        self.dim = self.model.config.hidden_size

    def batch(self, sentences: list[str]):
        enc = self.tok(sentences, padding=True, truncation=True, max_length=96,
                       return_offsets_mapping=True, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {k: (v.to(device) if hasattr(v, "to") else v) for k, v in enc.items()}

    def unfreeze(self) -> None:
        """Allow gradients into the encoder (fine-tuning mode)."""
        self.frozen = False
        self.model.train()
        for p in self.model.parameters():
            p.requires_grad_(True)

    def encode(self, sentences: list[str]):
        enc = self.batch(sentences)
        ctx = torch.no_grad() if getattr(self, "frozen", True) else torch.enable_grad()
        with ctx:
            out = self.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        return out.last_hidden_state, enc


class SpanHeads(nn.Module):
    """Small trained part: per-token tagger plus two sentence-level heads."""

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.tagger = nn.Linear(hidden, len(TAGS))
        self.is_fact = nn.Linear(hidden, 2)
        self.negated = nn.Linear(hidden, 2)

    def forward(self, states: torch.Tensor, mask: torch.Tensor):
        h = self.trunk(states)
        pooled = (h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return {"tags": self.tagger(h), "is_fact": self.is_fact(pooled), "negated": self.negated(pooled)}


def char_spans_to_tags(offsets, spans: list[tuple[int, int]], kind: str, tags: list[int]) -> None:
    for start, end in spans:
        first = True
        for i, (a, b) in enumerate(offsets):
            if a == b:  # special token
                continue
            if a >= start and b <= end:
                tags[i] = TAG_ID[("B-" if first else "I-") + kind]
                first = False


def make_labels(enc, sentences: list[str], labels: list[list]):
    """Per-token BIO tags and per-sentence fact/negation labels."""
    n, L = len(sentences), enc["input_ids"].size(1)
    tags = torch.zeros(n, L, dtype=torch.long)
    is_fact = torch.zeros(n, dtype=torch.long)
    negated = torch.zeros(n, dtype=torch.long)
    for i, facts in enumerate(labels):
        if not facts:
            continue
        f = facts[0]
        is_fact[i] = 1
        negated[i] = int(f.negated)
        offs = enc["offset_mapping"][i].tolist()
        row = tags[i].tolist()
        char_spans_to_tags(offs, [f.anchor], "ANCHOR", row)
        char_spans_to_tags(offs, [f.relation], "REL", row)
        char_spans_to_tags(offs, [f.target], "TARGET", row)
        tags[i] = torch.tensor(row)
    tags[enc["attention_mask"] == 0] = -100
    return {"tags": tags, "is_fact": is_fact, "negated": negated}


def decode_spans(tag_ids: list[int], offsets, sentence: str) -> dict[str, str]:
    """BIO -> the first span of each kind, as text."""
    out: dict[str, str] = {}
    cur_kind, cur = None, None
    for i, t in enumerate(tag_ids):
        a, b = offsets[i]
        if a == b:
            continue
        name = TAGS[t]
        if name == "O":
            if cur_kind and cur_kind not in out:
                out[cur_kind] = sentence[cur[0]:cur[1]]
            cur_kind, cur = None, None
            continue
        pos, kind = name.split("-")
        if pos == "B" or cur_kind != kind:
            if cur_kind and cur_kind not in out:
                out[cur_kind] = sentence[cur[0]:cur[1]]
            cur_kind, cur = kind, [a, b]
        else:
            cur[1] = b
    if cur_kind and cur_kind not in out:
        out[cur_kind] = sentence[cur[0]:cur[1]]
    return out


@torch.no_grad()
def compile_notes(enc: TokenEncoder, heads: SpanHeads, sentences: list[str], batch: int = 64) -> list[SpanNote]:
    """Query-independent compilation of a whole document."""
    heads.eval()
    notes: list[SpanNote] = []
    for s in range(0, len(sentences), batch):
        chunk = sentences[s : s + batch]
        states, e = enc.encode(chunk)
        out = heads(states, e["attention_mask"])
        keep = out["is_fact"].argmax(-1) == 1
        tags = out["tags"].argmax(-1)
        for j in keep.nonzero().squeeze(1).tolist():
            spans = decode_spans(tags[j].tolist(), e["offset_mapping"][j].tolist(), chunk[j])
            if {"ANCHOR", "REL", "TARGET"} <= spans.keys():
                notes.append(SpanNote(spans["ANCHOR"].lower(), spans["REL"].lower(), spans["TARGET"].lower(),
                                      bool(out["negated"][j].argmax()), s + j))
    return notes


def answer_from_span_notes(notes: list[SpanNote], question: str) -> str | None:
    """Replay open-vocabulary notes in order; entity names are whatever was extracted."""
    import re

    location: dict[str, str] = {}
    holder: dict[str, str] = {}
    dropped_at: dict[str, str] = {}
    not_in: dict[str, set[str]] = {}
    for n in notes:
        rel, a, t = n.semantic, n.anchor, n.target
        if rel == "move" or rel == "state_in":
            location[a] = t
            not_in.setdefault(a, set()).discard(t)
        elif rel == "state_not_in":
            not_in.setdefault(a, set()).add(t)
            if location.get(a) == t:
                location.pop(a)
        elif rel == "take":
            holder[t] = a
            dropped_at.pop(t, None)
        elif rel == "drop":
            holder.pop(t, None)
            if a in location:
                dropped_at[t] = location[a]
    q = question.strip().rstrip("?").strip().lower()
    m = re.match(r"where is the (\w+)", q)
    if m:
        obj = m.group(1)
        return location.get(holder[obj]) if obj in holder else dropped_at.get(obj)
    m = re.match(r"where is (\w+)", q)
    if m:
        return location.get(m.group(1))
    m = re.match(r"is (\w+) in the (\w+)", q)
    if m:
        person, place = m.group(1), m.group(2)
        if place in not_in.get(person, set()) and location.get(person) != place:
            return "no"
        if person in location:
            return "yes" if location[person] == place else "no"
    return None


def gold_span_notes(sentences: list[str]) -> list[SpanNote]:
    notes = []
    for i, s in enumerate(sentences):
        for f in find_span_facts(s):
            a, r, t = f.read(s)
            notes.append(SpanNote(a.lower(), r.lower(), t.lower(), f.negated, i))
    return notes


@torch.no_grad()
def evaluate(enc: TokenEncoder, heads: SpanHeads, limit: int, rename_seed: int | None,
             vocab: dict | None = None) -> dict:
    from datasets import load_dataset

    res = {}
    for task in TASKS:
        ds = load_dataset("RMT-team/babilong", "1k", split=task)
        rng = random.Random(rename_seed or 0)
        hit = gold_hit = tp = fp = fn = 0
        note_toks = doc_toks = 0
        n = min(limit, len(ds))
        for i in range(n):
            row = ds[i]
            doc, question, answer = row["input"], row["question"], row["target"].strip().lower()
            if rename_seed is not None:
                doc, question, answer = rename_document(
                    rng, doc, question, answer,
                    vocab["eval_names"] if vocab else None, vocab["eval_nouns"] if vocab else None)
                answer = answer.lower()
            sents = split_sentences(doc)
            gold = gold_span_notes(sents)
            pred = compile_notes(enc, heads, sents)
            hit += answer_from_span_notes(pred, question) == answer
            gold_hit += answer_from_span_notes(gold, question) == answer
            g = {(x.anchor, x.semantic, x.target, x.source) for x in gold}
            p = {(x.anchor, x.semantic, x.target, x.source) for x in pred}
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
            note_toks += sum(len(x.tokens) for x in pred)
            doc_toks += len(doc.split())
        res[task] = {"retention": hit / n, "gold_retention": gold_hit / n,
                     "precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn),
                     "note_tokens": note_toks / n, "doc_tokens": doc_toks / n,
                     "compression": note_toks / doc_toks}
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-docs", type=int, default=3000)
    ap.add_argument("--target-words", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--finetune-encoder", action="store_true",
                    help="also train the sentence encoder (frozen features limit span generalisation)")
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    global EXPERIMENT
    if args.tag:
        EXPERIMENT = f"{EXPERIMENT}_{args.tag}"

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    enc = TokenEncoder(args.threads)
    heads = SpanHeads(enc.dim)
    groups = [{"params": list(heads.parameters()), "lr": args.lr}]
    if args.finetune_encoder:
        enc.unfreeze()
        groups.append({"params": list(enc.model.parameters()), "lr": args.encoder_lr})
    opt = torch.optim.AdamW(groups, lr=args.lr, weight_decay=0.01)
    ce = nn.functional.cross_entropy

    print("building span-labelled documents (substituted entities, WikiText noise)...", flush=True)
    from notes_data import load_stories

    stories = load_stories((1, 2, 9))
    noise = load_noise()
    vocab = mined_vocab(noise)
    print(f"  entity vocabulary mined from prose: {len(vocab['train_names'])} train names / "
          f"{len(vocab['train_nouns'])} train nouns, held-out halves for evaluation", flush=True)
    sentences, labels = [], []
    for _ in range(args.train_docs):
        s, l, _ = build_span_document(rng, rng.choice(stories), noise, args.target_words,
                                      names=vocab["train_names"], nouns=vocab["train_nouns"])
        sentences += s
        labels += l
    print(f"  {len(sentences)} sentences, {sum(1 for x in labels if x)} facts", flush=True)

    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        heads.train()
        order = list(range(0, len(sentences), args.batch))
        rng.shuffle(order)
        for k, s in enumerate(order):
            chunk, lab = sentences[s : s + args.batch], labels[s : s + args.batch]
            states, e = enc.encode(chunk)
            y = make_labels(e, chunk, lab)
            out = heads(states, e["attention_mask"])
            loss = ce(out["tags"].flatten(0, 1), y["tags"].flatten(), ignore_index=-100)
            loss = loss + ce(out["is_fact"], y["is_fact"])
            m = y["is_fact"] == 1
            if m.any():
                loss = loss + ce(out["negated"][m], y["negated"][m])
            opt.zero_grad()
            loss.backward()
            opt.step()
        print(f"  epoch {epoch+1}: loss {float(loss):.4f} ({time.perf_counter() - t0:.0f}s)", flush=True)
    train_s = time.perf_counter() - t0

    suffix = f"{'_' + args.tag if args.tag else ''}_seed{args.seed}"
    ckpt = ROOT / "results" / "checkpoints" / f"span_compiler{suffix}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), ckpt)
    if args.finetune_encoder:
        enc.model.eval()
        torch.save(enc.model.state_dict(), ROOT / "results" / "checkpoints" / f"span_encoder{suffix}.pt")

    for label, rename in (("original_entities", None), ("heldout_entities", 1234)):
        print(f"evaluating on real BABILong ({label})...", flush=True)
        res = evaluate(enc, heads, args.limit, rename, vocab)
        for task, r in res.items():
            RunRecord(
                model="minilm_frozen+span_heads",
                condition=f"span_compiler_{label}",
                dataset=f"babilong_1k_{task}",
                seed=args.seed,
                context_length=int(r["doc_tokens"]),
                parameters=count_params(heads),
                note_tokens=r["note_tokens"],
                accuracy=r["retention"],
                task_success=r["retention"],
                train_wall_s=train_s,
                extra={"task": task, "entities": label, "finetuned_encoder": args.finetune_encoder, "gold_retention": r["gold_retention"],
                       "fact_precision": r["precision"], "fact_recall": r["recall"],
                       "compression": r["compression"], "open_vocabulary": True,
                       "trained_parameters": count_params(heads)},
            ).save(EXPERIMENT)
            print(f"  {task}: retention {r['retention']*100:.1f}% (gold {r['gold_retention']*100:.1f}%) "
                  f"| P {r['precision']*100:.1f} R {r['recall']*100:.1f} "
                  f"| notes {r['note_tokens']:.0f}/{r['doc_tokens']:.0f} tok ({r['compression']*100:.1f}%)",
                  flush=True)


if __name__ == "__main__":
    main()
