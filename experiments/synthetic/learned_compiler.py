"""Experiment B: learned query-independent causal note compiler (plan section 11.2).

The compiler reads every lecture chunk and freezes a notebook before any
question is sampled. Recovery is measured with deterministic graph traversal
over the predicted notes, NOT an LLM decoder (see plan 11.2 limitation).

Usage: python experiments/synthetic/learned_compiler.py [--seeds 11 22 33]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import load_config  # noqa: E402

import torch  # noqa: E402

from snced.compiler import NoteCompiler, chunk_batch, compile_notebook, compiler_loss  # noqa: E402
from snced.data import QTYPE_NAMES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.runlog import RunRecord, count_params, peak_rss_mb, seed_everything  # noqa: E402

EXPERIMENT = "learned_compiler"


def run(seed: int, cfg: dict, vocab: Vocab) -> RunRecord:
    c, d = cfg["compiler"], cfg["data"]
    seed_everything(seed)
    rng = random.Random(seed)
    train_docs = [make_lecture(rng, d["n_entities"], d["n_filler"]) for _ in range(c["n_train_docs"])]
    test_docs = [make_lecture(rng, d["n_entities"], d["n_filler"]) for _ in range(c["n_test_docs"])]

    model = NoteCompiler(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=c["lr"])
    tr = chunk_batch(vocab, train_docs)
    n = len(tr["kind"])
    t0 = time.perf_counter()
    train_tokens = 0
    for _ in range(c["epochs"]):
        model.train()
        perm = torch.randperm(n)
        for s in range(0, n, c["batch_size"]):
            idx = perm[s : s + c["batch_size"]]
            batch = {k: v[idx] for k, v in tr.items()}
            loss = compiler_loss(model(batch["ids"], batch["lengths"]), batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_tokens += int(batch["lengths"].sum())
    wall = time.perf_counter() - t0

    # Compile every test lecture first (query-independent), then ask questions.
    q_rng = random.Random(seed + 1)
    raw_tokens, note_tokens, raw_slots, note_slots = [], [], [], []
    exact, total_chunks, compile_ms = 0, 0, 0.0
    hits = {k: 0 for k in QTYPE_NAMES}
    asked = {k: 0 for k in QTYPE_NAMES}
    for lec in test_docs:
        t = time.perf_counter()
        nb, preds = compile_notebook(model, vocab, lec)
        compile_ms += (time.perf_counter() - t) * 1000
        raw_tokens.append(len(lec.tokens()))
        note_tokens.append(nb.token_count)
        raw_slots.append(len(lec.chunks))
        note_slots.append(nb.slots)
        for ch, p in zip(lec.chunks, preds):
            total_chunks += 1
            exact += p["kind"] == ch.kind and p["anchor"] == ch.anchor and p["target"] == ch.target
        for q in sample_questions(q_rng, lec, per_type=c["queries_per_type"]):
            asked[q.hops] += 1
            hits[q.hops] += nb.answer(q.entity, q.hops) == q.answer

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    recovery = {QTYPE_NAMES[k]: hits[k] / asked[k] for k in QTYPE_NAMES}
    overall = sum(hits.values()) / sum(asked.values())
    rec = RunRecord(
        model="gru_note_compiler",
        condition="learned_compiler",
        dataset="synthetic_lecture_v1",
        seed=seed,
        context_length=int(mean(raw_tokens)),
        parameters=count_params(model),
        training_tokens=train_tokens,
        note_tokens=mean(note_tokens),
        note_count=mean(note_slots),
        accuracy=overall,
        task_success=overall,
        peak_host_memory_mb=peak_rss_mb(),
        note_compile_ms=compile_ms / len(test_docs),
        train_wall_s=wall,
        extra={
            "raw_tokens": mean(raw_tokens),
            "raw_slots": mean(raw_slots),
            "text_token_reduction": 1 - mean(note_tokens) / mean(raw_tokens),
            "slot_reduction": 1 - mean(note_slots) / mean(raw_slots),
            "exact_chunk_extraction": exact / total_chunks,
            "recovery_by_type": recovery,
            "reader": "deterministic_graph_traversal",
            "query_independent": True,
        },
    )
    rec.save(EXPERIMENT)
    return rec


def main() -> None:
    cfg = load_config("synthetic_small.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["compiler"]["seeds"])
    args = ap.parse_args()
    torch.set_num_threads(4)
    vocab = Vocab()
    for seed in args.seeds:
        r = run(seed, cfg, vocab)
        e = r.extra
        print(
            f"seed={seed} params={r.parameters} raw={e['raw_tokens']:.2f} notes={r.note_tokens:.2f} "
            f"reduction={e['text_token_reduction']:.4f} slots={r.note_count:.2f} "
            f"exact={e['exact_chunk_extraction']:.4f} recovery={r.accuracy:.4f} {e['recovery_by_type']}",
            flush=True,
        )


if __name__ == "__main__":
    main()
