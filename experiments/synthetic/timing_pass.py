"""Clean timing and cost-per-successful-task pass (plan section 17).

Earlier wall-clock numbers were taken while other runs shared the CPU and are
not comparable. This measures one condition at a time, after a warm-up, with
nothing else running, and converts the result into the plan's preferred
commercial metric: cost per SUCCESSFUL task.

Cost is reported in two currencies:
  ms per successful task     measured wall clock / accuracy
  GFLOPs per successful task analytic compute / accuracy

A "task" is one document plus `--queries` questions about it, so building the
memory is paid once and amortised, as it would be for an agent re-querying a
document.

Usage: python experiments/synthetic/timing_pass.py [--seeds 1 2 3]
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
from snced.router import required_note_confidence  # noqa: E402
from snced.runlog import RunRecord, gpu_memory_mb, pick_device, seed_everything  # noqa: E402

EXPERIMENT = "timing"
CONDITIONS = ("full_context", "snm_notes", "snm_fallback")


@torch.no_grad()
def measure(model: SNCED, vocab: Vocab, items, condition: str, threshold: float, queries: int) -> dict:
    base = model.base
    build_s = answer_s = 0.0
    correct = n = 0
    slots_total = 0.0
    for i in range(0, len(items), 48):
        part = items[i : i + 48]
        b = make_batch(vocab, part)
        ctx_pad = b["ctx"] == PAD

        t0 = time.perf_counter()
        states = base.encode_states(b["ctx"])
        detail = base.mem_proj(states)
        if condition != "full_context":
            notes = model.compile(b["ctx"], states)
            note_mem, note_pad = model.note_memory(notes)
        build_s += time.perf_counter() - t0

        if condition == "full_context":
            mem, mem_pad = detail, ctx_pad
        elif condition == "snm_notes":
            mem, mem_pad = note_mem, note_pad
        else:
            conf = torch.tensor([required_note_confidence(notes.preds[j], q.entity, q.hops)
                                 for j, (_, q) in enumerate(part)])
            use_fb = conf < threshold
            width = max(detail.size(1), note_mem.size(1))
            mem = torch.zeros(len(part), width, base.d_model)
            mem_pad = torch.ones(len(part), width, dtype=torch.bool)
            for j in range(len(part)):
                src, sp = (detail[j], ctx_pad[j]) if use_fb[j] else (note_mem[j], note_pad[j])
                mem[j, : src.size(0)], mem_pad[j, : src.size(0)] = src, sp

        t0 = time.perf_counter()
        pred = predict(base, mem, mem_pad, b, vocab)
        answer_s += time.perf_counter() - t0
        correct += int((pred == b["y"]).sum())
        n += len(part)
        slots_total += float((~mem_pad).float().sum())

    accuracy = correct / n
    # Every evaluation row carries its own copy of the lecture, so the batch
    # encodes n documents, not n/6. Dividing by anything else inflates the
    # build cost for every condition equally and makes task_ms meaningless.
    build_ms = build_s * 1000 / n
    answer_ms = answer_s * 1000 / n
    task_ms = build_ms + queries * answer_ms
    return {"accuracy": accuracy, "mean_slots": slots_total / n, "build_ms_per_doc": build_ms,
            "answer_ms_per_question": answer_ms, "task_ms": task_ms,
            "ms_per_successful_task": task_ms / max(accuracy, 1e-9)}


def analytic_flops(condition: str, T: float, slots: float, mcfg: dict, queries: int, lq: float,
                   note_layers) -> float:
    d, dff, ne, nd, k = mcfg["d_model"], mcfg["d_ff"], mcfg["n_enc"], mcfg["n_dec"], mcfg["conv_kernel"]
    build = encoder_flops(T, d, dff, ne, k)
    if condition != "full_context":
        build += slots * sum(2 * lay.in_features * lay.out_features for lay in note_layers)
    return (build + decoder_memory_flops(slots, d, nd)
            + queries * decoder_query_flops(lq, slots, d, dff, nd))


def main() -> None:
    cfg = load_config("snced.yaml")
    base_cfg = load_config(cfg["base_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--fillers", type=int, nargs="+", default=[18, 108])
    ap.add_argument("--n-docs", type=int, default=60)
    ap.add_argument("--queries", type=int, default=36, help="questions per document in the workload")
    ap.add_argument("--checkpoint-tag", default="mixed")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    vocab = Vocab()
    rows = []

    for seed in args.seeds:
        seed_everything(seed)
        ref = CEDBaseline(len(vocab), **base_cfg["model"])
        ref.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"ced_baseline_seed{seed}.pt",
                                       map_location="cpu"))
        model = SNCED(copy.deepcopy(ref).to(device), vocab, note_mode=cfg["note_mode"]).to(device)
        tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
        model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"snced{tag}_seed{seed}.pt",
                                         map_location="cpu"))
        model.eval()
        note_layers = [lay for lay in model.note_encoder if isinstance(lay, torch.nn.Linear)]

        for n_filler in args.fillers:
            rng = random.Random(seed * 7919 + n_filler)
            items = []
            for _ in range(args.n_docs):
                lec = make_lecture(rng, base_cfg["data"]["n_entities"], n_filler)
                items += [(lec, q) for q in sample_questions(rng, lec, 2)]
            T = sum(len(lec.tokens()) for lec, _ in items) / len(items)
            lq = sum(len(q.text.split()) + 1 + q.hops for _, q in items) / len(items)
            measure(model, vocab, items[:48], "full_context", 0.5, args.queries)  # warm-up
            for cond in CONDITIONS:
                r = measure(model, vocab, items, cond, cfg["router"]["fallback_threshold"], args.queries)
                flops = analytic_flops(cond, T, r["mean_slots"], base_cfg["model"], args.queries, lq,
                                       note_layers)
                r.update(seed=seed, n_filler=n_filler, condition=cond, gflops_per_task=flops / 1e9,
                         gflops_per_successful_task=flops / 1e9 / max(r["accuracy"], 1e-9))
                rows.append(r)
                RunRecord(model="snced", condition=cond, dataset=f"synthetic_lecture_v1_filler{n_filler}",
                          seed=seed, context_length=int(T), accuracy=r["accuracy"],
                          task_success=r["accuracy"], note_count=r["mean_slots"],
                          kv_cache_bytes=int(ref.memory_bytes(1) * r["mean_slots"]),
                          prefill_ms=r["build_ms_per_doc"], decode_ms=r["answer_ms_per_question"],
                          total_task_ms=r["task_ms"], estimated_flops=int(flops),
                          peak_gpu_memory_mb=gpu_memory_mb(),
                          extra={"n_filler": n_filler, "queries_per_task": args.queries,
                                 "ms_per_successful_task": r["ms_per_successful_task"],
                                 "gflops_per_successful_task": r["gflops_per_successful_task"],
                                 "device": str(device), "exclusive_cpu": True}).save(EXPERIMENT)
                print(f"seed={seed} filler={n_filler} {cond:14s} acc={r['accuracy']:.4f} "
                      f"slots={r['mean_slots']:.1f} build={r['build_ms_per_doc']:.1f}ms "
                      f"answer={r['answer_ms_per_question']:.2f}ms task={r['task_ms']:.0f}ms", flush=True)

    lines = ["# Timing and cost per successful task", "",
             f"Seeds: {args.seeds}. One condition at a time on an otherwise idle machine, after warm-up.",
             f"A task = one document + {args.queries} questions, so memory building is amortised.",
             f"Device: {device}.", ""]
    for n_filler in args.fillers:
        lines += [f"## {n_filler} filler chunks", "",
                  "| Condition | Accuracy | Slots | Build ms/doc | Answer ms/q | Task ms | ms / successful task | GFLOPs / successful task |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for cond in CONDITIONS:
            rs = [r for r in rows if r["n_filler"] == n_filler and r["condition"] == cond]
            g = lambda k: mean_std([r[k] for r in rs])  # noqa: E731
            lines.append(
                f"| {cond} | {g('accuracy')[0]*100:.2f}% | {g('mean_slots')[0]:.1f} "
                f"| {g('build_ms_per_doc')[0]:.1f} | {g('answer_ms_per_question')[0]:.2f} "
                f"| {g('task_ms')[0]:.0f} | {g('ms_per_successful_task')[0]:.0f} "
                f"| {g('gflops_per_successful_task')[0]:.3f} |")
        lines.append("")
    (ROOT / "results" / "tables" / "timing.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
