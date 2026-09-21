"""Calibrate the memory router (plan sections 8, 9, P5; Plot E).

The rule-based router used so far is lecture-level: if ANY note in the lecture
falls below a confidence threshold, the whole lecture is answered from detailed
memory. That fires far more often than notes are actually wrong, which spends
most of the memory saving for almost no accuracy.

This sweeps two routing policies over thresholds:

  lecture   fall back when the lecture's lowest-confidence note is below t
  question  fall back only when a note the question actually needs is below t
            (the notes reached by replaying the anchor's chain)

and reports the resulting quality / memory / fallback-rate curve, so a
threshold can be chosen instead of guessed.

Usage: python experiments/synthetic/router_calibration.py [--seeds 1 2 3]
"""

from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_config  # noqa: E402

import torch  # noqa: E402
from end_to_end import make_batch  # noqa: E402
from snced_compare import predict  # noqa: E402

from snced.data import PAD, QTYPE_NAMES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.metrics import mean_std  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.models.snced import SNCED  # noqa: E402
from snced.router import required_note_confidence  # noqa: E402
from snced.runlog import RunRecord, seed_everything  # noqa: E402

EXPERIMENT = "router_calibration"
THRESHOLDS = (0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.01)


@torch.no_grad()
def sweep(model: SNCED, vocab: Vocab, items, policy: str) -> dict[float, dict]:
    base = model.base
    out = {t: {"correct": 0, "n": 0, "slots": 0.0, "fallback": 0} for t in THRESHOLDS}
    for i in range(0, len(items), 48):
        part = items[i : i + 48]
        b = make_batch(vocab, part)
        ctx_pad = b["ctx"] == PAD
        states = base.encode_states(b["ctx"])
        detail = base.mem_proj(states)
        notes = model.compile(b["ctx"], states)
        note_mem, note_pad = model.note_memory(notes)
        pred_notes = predict(base, note_mem, note_pad, b, vocab)
        pred_detail = predict(base, detail, ctx_pad, b, vocab)
        n_note_slots = (~note_pad).float().sum(1)
        n_detail_slots = (~ctx_pad).float().sum(1)
        if policy == "lecture":
            conf = notes.min_confidence
        else:
            conf = torch.tensor([required_note_confidence(notes.preds[j], q.entity, q.hops)
                                 for j, (_, q) in enumerate(part)])
        for t in THRESHOLDS:
            use_fb = conf < t
            pred = torch.where(use_fb, pred_detail, pred_notes)
            slots = torch.where(use_fb, n_detail_slots, n_note_slots)
            s = out[t]
            s["correct"] += int((pred == b["y"]).sum())
            s["n"] += len(part)
            s["slots"] += float(slots.sum())
            s["fallback"] += int(use_fb.sum())
    return {t: {"accuracy": s["correct"] / s["n"], "mean_slots": s["slots"] / s["n"],
                "fallback_rate": s["fallback"] / s["n"]} for t, s in out.items()}


def main() -> None:
    cfg = load_config("snced.yaml")
    base_cfg = load_config(cfg["base_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--fillers", type=int, nargs="+", default=[18, 108])
    ap.add_argument("--n-docs", type=int, default=100)
    ap.add_argument("--checkpoint-tag", default="mixed")
    ap.add_argument("--threads", type=int, default=6)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    vocab = Vocab()

    rows = []
    for seed in args.seeds:
        seed_everything(seed)
        ref = CEDBaseline(len(vocab), **base_cfg["model"])
        ref.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"ced_baseline_seed{seed}.pt"))
        model = SNCED(copy.deepcopy(ref), vocab, note_mode=cfg["note_mode"])
        tag = f"_{args.checkpoint_tag}" if args.checkpoint_tag else ""
        model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"snced{tag}_seed{seed}.pt"))
        model.eval()
        for n_filler in args.fillers:
            rng = random.Random(seed * 7919 + n_filler)
            items = []
            for _ in range(args.n_docs):
                lec = make_lecture(rng, base_cfg["data"]["n_entities"], n_filler)
                items += [(lec, q) for q in sample_questions(rng, lec, 2)]
            for policy in ("lecture", "question"):
                res = sweep(model, vocab, items, policy)
                for t, r in res.items():
                    rows.append({"seed": seed, "n_filler": n_filler, "policy": policy, "threshold": t, **r})
                    print(f"seed={seed} filler={n_filler} {policy:8s} t={t:<6} acc={r['accuracy']:.4f} "
                          f"slots={r['mean_slots']:.1f} fallback={r['fallback_rate']:.3f}", flush=True)

    # Aggregate and pick the cheapest threshold whose accuracy is within 0.5pp of the best.
    lines = ["# Router calibration (Plot E)", "",
             f"Seeds: {args.seeds}. {args.n_docs} documents per length, 6 questions each.",
             "`lecture` = fall back if any note in the lecture is uncertain (the policy used so far).",
             "`question` = fall back only if a note the question actually needs is uncertain.", ""]
    best_choice = {}
    for n_filler in args.fillers:
        lines += [f"## {n_filler} filler chunks", "",
                  "| Policy | Threshold | Accuracy | Mean slots | Fallback rate |", "|---|---:|---:|---:|---:|"]
        for policy in ("lecture", "question"):
            sel = [r for r in rows if r["n_filler"] == n_filler and r["policy"] == policy]
            for t in THRESHOLDS:
                rs = [r for r in sel if r["threshold"] == t]
                acc = mean_std([r["accuracy"] for r in rs])
                slots = mean_std([r["mean_slots"] for r in rs])
                fb = mean_std([r["fallback_rate"] for r in rs])
                lines.append(f"| {policy} | {t} | {acc[0]*100:.2f} ± {acc[1]*100:.2f}% | {slots[0]:.1f} "
                             f"| {fb[0]*100:.1f} ± {fb[1]*100:.1f}% |")
            best = max(mean_std([r["accuracy"] for r in sel if r["threshold"] == t])[0] for t in THRESHOLDS)
            ok = [t for t in THRESHOLDS
                  if mean_std([r["accuracy"] for r in sel if r["threshold"] == t])[0] >= best - 0.005]
            cheapest = min(ok, key=lambda t: mean_std([r["mean_slots"] for r in sel if r["threshold"] == t])[0])
            best_choice[(n_filler, policy)] = cheapest
        lines.append("")
    lines += ["## Calibrated thresholds (cheapest within 0.5pp of the best accuracy)", "",
              "| Filler | Policy | Threshold |", "|---|---|---:|"]
    for (n_filler, policy), t in best_choice.items():
        lines.append(f"| {n_filler} | {policy} | {t} |")
    path = ROOT / "results" / "tables" / "router_calibration.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-8:]))

    for r in rows:
        RunRecord(model="snced", condition=f"router_{r['policy']}_t{r['threshold']}",
                  dataset=f"synthetic_lecture_v1_filler{r['n_filler']}", seed=r["seed"],
                  accuracy=r["accuracy"], task_success=r["accuracy"], fallback_rate=r["fallback_rate"],
                  note_count=r["mean_slots"],
                  extra={"policy": r["policy"], "threshold": r["threshold"], "n_filler": r["n_filler"],
                         "mean_slots": r["mean_slots"]}).save(EXPERIMENT)


if __name__ == "__main__":
    main()
