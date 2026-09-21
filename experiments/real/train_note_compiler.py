"""Train a small query-independent Semantic Note Compiler on real text.

The compiler reads a document sentence by sentence, before any question
exists, and emits notes (anchor, relation, target, source index) in document
order. It is a frozen MiniLM sentence encoder (22M, not fine-tuned) plus small
trained heads, so the trained part is tiny.

Training documents use WikiText noise; evaluation uses the real BABILong
documents, whose noise is book text from a different corpus. Some training
sentences glue a fact onto neighbouring prose, as happens in BABILong.

Retention is measured with a deterministic reader over the predicted notes
(no LLM): it replays the notes in order and answers the question. Gold notes
score 100% on all three tasks, so any loss here is the compiler's.

Usage: python experiments/real/train_note_compiler.py [--train-docs 3000]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402

import torch  # noqa: E402
from notes_data import (  # noqa: E402
    PERSONS, RELATIONS, TARGETS, Note, answer_from_notes, build_document, find_facts,
    load_noise_sentences, load_stories, notes_from_sentences, split_sentences,
)
from torch import nn  # noqa: E402
from transformers import AutoModel, AutoTokenizer  # noqa: E402

from snced.runlog import RunRecord, count_params, seed_everything  # noqa: E402

EXPERIMENT = "note_compiler_real"
ENCODER_ID = "sentence-transformers/all-MiniLM-L6-v2"
TASKS = {"qa1": 1, "qa2": 2, "qa9": 9}


class SentenceEncoder:
    """Frozen sentence encoder; the compiler never fine-tunes it."""

    def __init__(self, threads: int) -> None:
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(ENCODER_ID)
        self.model = AutoModel.from_pretrained(ENCODER_ID).eval()
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def encode(self, sentences: list[str], batch: int = 128) -> torch.Tensor:
        out = []
        for i in range(0, len(sentences), batch):
            b = self.tok(sentences[i : i + batch], padding=True, truncation=True, max_length=64,
                         return_tensors="pt")
            h = self.model(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1)
            out.append((h * m).sum(1) / m.sum(1).clamp(min=1))
        return torch.cat(out)


class NoteCompilerHeads(nn.Module):
    """Small trained heads: is this sentence a fact, and which (anchor, relation, target)?"""

    def __init__(self, dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU())
        self.is_fact = nn.Linear(hidden, 2)
        self.relation = nn.Linear(hidden, len(RELATIONS))
        self.anchor = nn.Linear(hidden, len(PERSONS))
        self.target = nn.Linear(hidden, len(TARGETS))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.trunk(x)
        return {"is_fact": self.is_fact(h), "relation": self.relation(h), "anchor": self.anchor(h),
                "target": self.target(h)}


def glue(rng: random.Random, sentences: list[str], labels: list[list], p: float = 0.3):
    """Occasionally merge a fact sentence into its neighbouring prose, as BABILong does."""
    out_s, out_l = [], []
    i = 0
    while i < len(sentences):
        if labels[i] and i + 1 < len(sentences) and not labels[i + 1] and rng.random() < p:
            out_s.append(sentences[i] + " " + sentences[i + 1])
            out_l.append(labels[i])
            i += 2
        else:
            out_s.append(sentences[i])
            out_l.append(labels[i])
            i += 1
    return out_s, out_l


def make_training_set(rng: random.Random, n_docs: int, target_words: int):
    stories, noise = load_stories(tuple(TASKS.values())), load_noise_sentences()
    sentences, labels = [], []
    for _ in range(n_docs):
        doc = build_document(rng, rng.choice(stories), noise, target_words)
        s, l = glue(rng, doc.sentences, doc.labels)
        sentences += s
        labels += l
    return sentences, labels


def label_tensors(labels: list[list]) -> dict[str, torch.Tensor]:
    is_fact, rel, anc, tgt = [], [], [], []
    for fs in labels:
        if fs:
            a, r, t = fs[0]
            is_fact.append(1); rel.append(RELATIONS.index(r))
            anc.append(PERSONS.index(a)); tgt.append(TARGETS.index(t))
        else:
            is_fact.append(0); rel.append(-100); anc.append(-100); tgt.append(-100)
    return {k: torch.tensor(v) for k, v in
            (("is_fact", is_fact), ("relation", rel), ("anchor", anc), ("target", tgt))}


@torch.no_grad()
def compile_notes(enc: SentenceEncoder, heads: NoteCompilerHeads, sentences: list[str]) -> list[Note]:
    """Query-independent: this runs before any question is seen."""
    heads.eval()
    out = heads(enc.encode(sentences))
    keep = out["is_fact"].argmax(-1) == 1
    notes = []
    for i in keep.nonzero().squeeze(1).tolist():
        notes.append(Note(PERSONS[int(out["anchor"][i].argmax())],
                          RELATIONS[int(out["relation"][i].argmax())],
                          TARGETS[int(out["target"][i].argmax())], i))
    return notes


@torch.no_grad()
def evaluate_real(enc: SentenceEncoder, heads: NoteCompilerHeads, limit: int) -> dict:
    from datasets import load_dataset

    res = {}
    for task in TASKS:
        ds = load_dataset("RMT-team/babilong", "1k", split=task)
        hit = gold_hit = 0
        tp = fp = fn = 0
        note_toks = doc_toks = 0
        for i in range(min(limit, len(ds))):
            row = ds[i]
            sents = split_sentences(row["input"])
            gold = notes_from_sentences(sents)
            pred = compile_notes(enc, heads, sents)
            target = row["target"].strip().lower()
            hit += answer_from_notes(pred, row["question"]) == target
            gold_hit += answer_from_notes(gold, row["question"]) == target
            g = {(n.anchor, n.relation, n.target, n.source) for n in gold}
            p = {(n.anchor, n.relation, n.target, n.source) for n in pred}
            tp += len(g & p); fp += len(p - g); fn += len(g - p)
            note_toks += sum(len(n.tokens) for n in pred)
            doc_toks += len(row["input"].split())
        n = min(limit, len(ds))
        res[task] = {
            "retention": hit / n,
            "gold_retention": gold_hit / n,
            "fact_precision": tp / max(1, tp + fp),
            "fact_recall": tp / max(1, tp + fn),
            "note_tokens": note_toks / n,
            "doc_tokens": doc_toks / n,
            "compression": note_toks / doc_toks,
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-docs", type=int, default=3000)
    ap.add_argument("--val-docs", type=int, default=200)
    ap.add_argument("--target-words", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--limit", type=int, default=100, help="BABILong documents per task at eval")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    seed_everything(args.seed)
    rng = random.Random(args.seed)
    enc = SentenceEncoder(args.threads)
    print("building training documents (WikiText noise)...", flush=True)
    tr_s, tr_l = make_training_set(rng, args.train_docs, args.target_words)
    va_s, va_l = make_training_set(rng, args.val_docs, args.target_words)
    print(f"  {len(tr_s)} train sentences ({sum(1 for x in tr_l if x)} facts), {len(va_s)} val", flush=True)

    t0 = time.perf_counter()
    X, Xv = enc.encode(tr_s), enc.encode(va_s)
    print(f"  encoded in {time.perf_counter() - t0:.0f}s", flush=True)
    Y, Yv = label_tensors(tr_l), label_tensors(va_l)

    heads = NoteCompilerHeads(enc.dim)
    opt = torch.optim.AdamW(heads.parameters(), lr=args.lr, weight_decay=0.01)
    ce = nn.functional.cross_entropy
    t0 = time.perf_counter()
    for epoch in range(args.epochs):
        heads.train()
        perm = torch.randperm(len(X))
        for i in range(0, len(perm), args.batch):
            idx = perm[i : i + args.batch]
            out = heads(X[idx])
            loss = ce(out["is_fact"], Y["is_fact"][idx])
            for head in ("relation", "anchor", "target"):
                if (Y[head][idx] != -100).any():
                    loss = loss + ce(out[head], Y[head][idx], ignore_index=-100)
            opt.zero_grad()
            loss.backward()
            opt.step()
        heads.eval()
        with torch.no_grad():
            ov = heads(Xv)
            fact_acc = float((ov["is_fact"].argmax(-1) == Yv["is_fact"]).float().mean())
            m = Yv["is_fact"] == 1
            tuple_acc = float(((ov["relation"].argmax(-1) == Yv["relation"]) &
                               (ov["anchor"].argmax(-1) == Yv["anchor"]) &
                               (ov["target"].argmax(-1) == Yv["target"]))[m].float().mean())
        print(f"  epoch {epoch+1}: loss {float(loss):.4f} val fact_acc {fact_acc:.4f} tuple_acc {tuple_acc:.4f}",
              flush=True)
    train_s = time.perf_counter() - t0

    print("evaluating on real BABILong documents (book-text noise, unseen corpus)...", flush=True)
    res = evaluate_real(enc, heads, args.limit)
    ckpt = ROOT / "results" / "checkpoints" / f"note_compiler_real_seed{args.seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save(heads.state_dict(), ckpt)

    for task, r in res.items():
        rec = RunRecord(
            model="minilm_frozen+heads",
            condition="trained_note_compiler",
            dataset=f"babilong_1k_{task}",
            seed=args.seed,
            context_length=int(r["doc_tokens"]),
            parameters=count_params(heads),
            note_tokens=r["note_tokens"],
            accuracy=r["retention"],
            task_success=r["retention"],
            train_wall_s=train_s,
            extra={
                "task": task,
                "gold_note_retention": r["gold_retention"],
                "fact_precision": r["fact_precision"],
                "fact_recall": r["fact_recall"],
                "compression": r["compression"],
                "trained_parameters": count_params(heads),
                "frozen_encoder": ENCODER_ID,
                "train_docs": args.train_docs,
                "train_noise": "wikitext-103",
                "eval_noise": "babilong (pg19 books)",
                "val_fact_acc": fact_acc,
                "val_tuple_acc": tuple_acc,
            },
        )
        rec.save(EXPERIMENT)
        print(f"  {task}: retention {r['retention']*100:.1f}% (gold {r['gold_retention']*100:.1f}%) "
              f"| fact P {r['fact_precision']*100:.1f} R {r['fact_recall']*100:.1f} "
              f"| notes {r['note_tokens']:.0f} tok vs {r['doc_tokens']:.0f} ({r['compression']*100:.1f}%)",
              flush=True)


if __name__ == "__main__":
    main()
