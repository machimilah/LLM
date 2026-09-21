"""Scaling benchmark: how cost grows with document length and with reuse.

The commercial claim for this architecture is not "higher accuracy". It is that
the memory a model carries stops growing with the length of what it has read,
so cost per question flattens while a full-context model's keeps climbing.

This measures exactly that, on a trained model, at several context lengths:

  quality        accuracy from notes vs from full context
  active memory  decoder memory slots, and the KV bytes they imply
  compute        analytic FLOPs for a workload of N questions per document
  wall clock     measured, one condition at a time on an idle machine

Every number here comes from a 445k-parameter model on synthetic documents.
That is a mechanism demonstration, not a product benchmark, and the report says
so on the same page as the numbers.

Usage: python experiments/synthetic/scaling_benchmark.py [--fillers 18 54 108 216]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_config  # noqa: E402

import torch  # noqa: E402
from end_to_end import make_batch  # noqa: E402
from snced_compare import predict  # noqa: E402

from snced.data import PAD, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.metrics import decoder_memory_flops, decoder_query_flops, encoder_flops, mean_std  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.models.snced import SNCED  # noqa: E402
from snced.runlog import RunRecord, seed_everything  # noqa: E402

EXPERIMENT = "scaling_benchmark"


@torch.no_grad()
def measure(model: SNCED, vocab: Vocab, items, use_notes: bool, queries: int, mcfg: dict) -> dict:
    base = model.base
    build_s = answer_s = 0.0
    correct = n = 0
    slots = 0.0
    for i in range(0, len(items), 48):
        part = items[i : i + 48]
        b = make_batch(vocab, part)
        ctx_pad = b["ctx"] == PAD

        t0 = time.perf_counter()
        states = base.encode_states(b["ctx"])
        detail = base.mem_proj(states)
        if use_notes:
            notes = model.compile(b["ctx"], states)
            mem, mask = model.note_memory(notes)
        else:
            mem, mask = detail, ctx_pad
        build_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        pred = predict(base, mem, mask, b, vocab)
        answer_s += time.perf_counter() - t0
        correct += int((pred == b["y"]).sum())
        n += len(part)
        slots += float((~mask).float().sum())

    mean_slots = slots / n
    T = sum(len(lec.tokens()) for lec, _ in items) / len(items)
    lq = sum(len(q.text.split()) + 1 + q.hops for _, q in items) / len(items)
    d, dff, ne, nd, k = mcfg["d_model"], mcfg["d_ff"], mcfg["n_enc"], mcfg["n_dec"], mcfg["conv_kernel"]
    build_flops = encoder_flops(T, d, dff, ne, k)
    if use_notes:
        build_flops += mean_slots * 2 * (3 * d * 128 + 128 * 128 + 128 * d)
    flops = (build_flops + decoder_memory_flops(mean_slots, d, nd)
             + queries * decoder_query_flops(lq, mean_slots, d, dff, nd))
    return {
        "accuracy": correct / n,
        "slots": mean_slots,
        "kv_bytes": int(2 * nd * mean_slots * d * 4),
        "context_tokens": T,
        "build_ms": build_s * 1000 / n,
        "answer_ms": answer_s * 1000 / n,
        "task_ms": build_s * 1000 / n + queries * answer_s * 1000 / n,
        "gflops": flops / 1e9,
    }


def main() -> None:
    cfg = load_config("snced.yaml")
    base_cfg = load_config(cfg["base_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2, 3])
    ap.add_argument("--fillers", type=int, nargs="+", default=[18, 54, 108, 216])
    ap.add_argument("--n-docs", type=int, default=40)
    ap.add_argument("--queries", type=int, default=36)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    vocab = Vocab()
    rows = []

    for seed in args.seeds:
        seed_everything(seed)
        ref = CEDBaseline(len(vocab), **base_cfg["model"])
        ref.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"ced_baseline_seed{seed}.pt",
                                       map_location="cpu"))
        model = SNCED(copy.deepcopy(ref), vocab, note_mode=cfg["note_mode"])
        model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"snced_mixed_seed{seed}.pt",
                                         map_location="cpu"))
        model.eval()
        for n_filler in args.fillers:
            rng = random.Random(seed * 7919 + n_filler)
            items = []
            for _ in range(args.n_docs):
                lec = make_lecture(rng, base_cfg["data"]["n_entities"], n_filler)
                items += [(lec, q) for q in sample_questions(rng, lec, 2)]
            for use_notes in (False, True):
                r = measure(model, vocab, items, use_notes, args.queries, base_cfg["model"])
                r.update(seed=seed, n_filler=n_filler, condition="notes" if use_notes else "full_context")
                rows.append(r)
                RunRecord(model="snced", condition=r["condition"],
                          dataset=f"synthetic_lecture_v1_filler{n_filler}", seed=seed,
                          context_length=int(r["context_tokens"]), accuracy=r["accuracy"],
                          task_success=r["accuracy"], note_count=r["slots"],
                          kv_cache_bytes=r["kv_bytes"], prefill_ms=r["build_ms"],
                          decode_ms=r["answer_ms"], total_task_ms=r["task_ms"],
                          estimated_flops=int(r["gflops"] * 1e9),
                          extra={"n_filler": n_filler, "queries_per_task": args.queries,
                                 "memory_slots": r["slots"]}).save(EXPERIMENT)
                print(f"seed={seed} ctx={r['context_tokens']:.0f} {r['condition']:13s} "
                      f"acc={r['accuracy']:.3f} slots={r['slots']:.0f} kv={r['kv_bytes']/1024:.0f}KB "
                      f"gflops={r['gflops']:.2f} task_ms={r['task_ms']:.0f}", flush=True)

    # Aggregate into the table the report uses.
    lines = ["# Scaling benchmark", "",
             f"Seeds {args.seeds}, {args.n_docs} documents per length, {args.queries} questions per document.",
             "", "| Context | Full context | Notes | Memory | KV | GFLOPs | Time per task |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for n_filler in args.fillers:
        f = [r for r in rows if r["n_filler"] == n_filler and r["condition"] == "full_context"]
        s = [r for r in rows if r["n_filler"] == n_filler and r["condition"] == "notes"]
        if not f or not s:
            continue
        g = lambda rs, k: mean_std([r[k] for r in rs])[0]  # noqa: E731
        lines.append(
            f"| {g(f,'context_tokens'):.0f} tok | {g(f,'accuracy')*100:.1f}% | {g(s,'accuracy')*100:.1f}% "
            f"| {g(s,'slots'):.0f} vs {g(f,'slots'):.0f} ({g(s,'slots')/g(f,'slots')*100:.1f}%) "
            f"| {g(s,'kv_bytes')/1024:.0f}KB vs {g(f,'kv_bytes')/1024:.0f}KB "
            f"| {g(s,'gflops'):.2f} vs {g(f,'gflops'):.2f} "
            f"| {g(s,'task_ms'):.0f}ms vs {g(f,'task_ms'):.0f}ms |")
    path = ROOT / "results" / "tables" / "scaling_benchmark.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
