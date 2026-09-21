"""Train the general SN-CED end to end (plan sections 6, 8, 10).

Everything is learned together from one composite loss: the encoder, the
open-vocabulary note compiler, the note encoder, the sufficiency router and the
decoder. There are no gold note labels - the notebook is shaped only by the
task loss, a sufficiency term that asks notes to behave like full context, a
compression penalty on the number of notes, and a price on detailed-memory
access.

Reported per evaluation:
  notes      answer from Semantic Note Memory alone
  routed     answer from the path the router actually chose (hard threshold)
  detailed   answer from full context (the reference path)
plus the notebook size, the router's firing rate, and a sample of the notes as
text, so the memory stays inspectable.

Usage: python experiments/synthetic/general_train.py [--seeds 1] [--device auto]
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import deque
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_config  # noqa: E402

import torch  # noqa: E402
from end_to_end import make_batch, make_train_batch  # noqa: E402

from snced.data import PAD, QTYPE_NAMES, SEP, VALUES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.models.snced_general import Notebook, SNCEDGeneral, _index_fields  # noqa: E402
from snced.runlog import (  # noqa: E402
    RunRecord, count_params, gpu_memory_mb, peak_rss_mb, pick_device, seed_everything, to_device,
)

EXPERIMENT = "snced_general"
PATHS = ("notes", "routed", "detailed")


@torch.no_grad()
def answer(model: SNCEDGeneral, b: dict, vocab: Vocab, path: str, threshold: float):
    """Greedy chain decoding over the memory produced by `path`."""
    model.eval()
    ctx, q = b["ctx"], b["q"]
    states, detailed = model.encode(ctx)
    detail_mask = ctx == PAD
    nb = model.compile(ctx, states)
    fired = torch.zeros(len(q))
    if path == "detailed":
        mem, mask = detailed, detail_mask
    elif path == "notes":
        mem, mask = nb.slots, nb.mask
    else:
        routed = model.route(q, nb, detailed, detail_mask, hard=True, threshold=threshold)
        mem, mask, fired = routed.memory, routed.mask, (routed.p_detail > threshold).float()
    pred = torch.full_like(b["y"], -1)
    for k in QTYPE_NAMES:
        idx = (b["qtype"] == k).nonzero().squeeze(1)
        if len(idx) == 0:
            continue
        qk = q[idx]
        qk = qk[:, : int((qk[0] != PAD).sum())]
        prompt = torch.cat([qk, torch.full((len(idx), 1), SEP, device=qk.device)], dim=1)
        out = model.backbone.generate(mem[idx], mask[idx], prompt, k + 1)[:, -1].tolist()
        pred[idx] = torch.tensor([VALUES.index(vocab.itos[t]) if vocab.itos[t] in VALUES else -1
                                  for t in out], device=pred.device)
    slots = float((~mask).float().sum(-1).mean())
    return pred, slots, float(fired.mean())


@torch.no_grad()
def evaluate(model: SNCEDGeneral, batches, vocab: Vocab, threshold: float) -> dict:
    out = {}
    for path in PATHS:
        correct = total = 0
        by_type = {k: [0, 0] for k in QTYPE_NAMES}
        slots = fired = 0.0
        for b in batches:
            pred, s, f = answer(model, b, vocab, path, threshold)
            correct += int((pred == b["y"]).sum())
            total += len(pred)
            slots += s
            fired += f
            for k in QTYPE_NAMES:
                m = b["qtype"] == k
                by_type[k][0] += int((pred[m] == b["y"][m]).sum())
                by_type[k][1] += int(m.sum())
        out[path] = {"accuracy": correct / total, "slots": slots / len(batches),
                     "router_fired": fired / len(batches),
                     "by_type": {QTYPE_NAMES[k]: c / n for k, (c, n) in by_type.items()}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="snced_general.yaml")
    known, _ = ap.parse_known_args()
    cfg = load_config(known.config)
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["train"]["seeds"])
    ap.add_argument("--max-steps", type=int, default=cfg["train"]["max_steps"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--freeze-encoder", action="store_true",
                    help="keep a warm-started encoder fixed (it already solves the task)")
    ap.add_argument("--decoder-frozen-steps", type=int, default=0,
                    help="hold the decoder still until the compiler writes usable notes; "
                         "an untrained compiler otherwise destroys a warm-started decoder in a few steps")
    ap.add_argument("--init-backbone", default="",
                    help="warm-start encoder+decoder from a trained CED checkpoint "
                         "(plan Phase 2: train the base model first, then attach the note branch)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    vocab = Vocab()
    t, d, w = cfg["train"], cfg["data"], cfg["loss"]

    for seed in args.seeds:
        seed_everything(seed)
        model = SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."], **cfg["model"]).to(device)
        if args.init_backbone:
            path = ROOT / "results" / "checkpoints" / args.init_backbone.format(seed=seed)
            missing = model.backbone.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
            model.backbone.to(device)
            print(f"warm-started backbone from {path.name} "
                  f"(missing {len(missing.missing_keys)}, unexpected {len(missing.unexpected_keys)})",
                  flush=True)
        if args.freeze_encoder:
            for module in (model.backbone.tok, model.backbone.encoder, model.backbone.enc_norm,
                           model.backbone.mem_proj):
                for prm in module.parameters():
                    prm.requires_grad_(False)
        decoder_modules = (*model.backbone.decoder, model.backbone.dec_norm, model.backbone.head)
        if args.decoder_frozen_steps:
            for module in decoder_modules:
                for prm in module.parameters():
                    prm.requires_grad_(False)
        opt = torch.optim.AdamW([prm for prm in model.parameters() if prm.requires_grad],
                                lr=t["lr"], weight_decay=t["weight_decay"])
        rng = random.Random(seed)

        eval_rng = random.Random(seed + 10_000)
        eval_items = []
        for _ in range(cfg["eval"]["n_docs"]):
            lec = make_lecture(eval_rng, d["n_entities"], d["n_filler"])
            eval_items += [(lec, q) for q in sample_questions(eval_rng, lec, cfg["eval"]["queries_per_type"])]
        eval_batches = [to_device(make_batch(vocab, eval_items[i : i + 48]), device)
                        for i in range(0, len(eval_items), 48)]

        log_path = ROOT / "results" / "logs" / f"{EXPERIMENT}_seed{seed}.jsonl"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("w")
        print(f"parameters: {count_params(model):,} "
              f"(compiler {count_params(model.compiler):,}, notes {count_params(model.note_encoder):,}, "
              f"router {count_params(model.router):,})", flush=True)

        stages, stage, final_start = t["curriculum"], 0, None
        recent = deque(maxlen=t["advance_window"])
        best, t0, step, max_steps = None, time.perf_counter(), 0, args.max_steps
        while step < max_steps:
            step += 1
            n_fill, qtypes = stages[stage]
            lr = t["lr"] * min(1.0, step / t["warmup"])
            if final_start is not None:
                lr *= 0.5 * (1 + math.cos(math.pi * (step - final_start) / t["final_stage_steps"]))
            for g in opt.param_groups:
                g["lr"] = lr

            if args.decoder_frozen_steps and step == args.decoder_frozen_steps + 1:
                for module in decoder_modules:
                    for prm in module.parameters():
                        prm.requires_grad_(True)
                opt.add_param_group({"params": [prm for module in decoder_modules
                                                for prm in module.parameters()], "lr": t["lr"]})
                print(f"step={step} decoder unfrozen", flush=True)

            model.train()
            b = to_device(make_train_batch(vocab, rng, t["lectures_per_batch"], d["n_entities"],
                                           n_fill, qtypes, True), device)
            loss, parts = model.loss(b["ctx"], b["q"], b["tgt"], b["rows"], w, step)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
            opt.step()
            recent.append(math.exp(-parts["task_notes"]))  # rough per-token accuracy proxy

            if final_start is None and len(recent) == recent.maxlen and sum(recent) / len(recent) > t["advance_at"]:
                if stage + 1 < len(stages):
                    stage += 1
                    recent.clear()
                    print(f"step={step} advance to stage {stage}: {stages[stage]}", flush=True)
                if stage == len(stages) - 1:
                    final_start = step
                    max_steps = min(max_steps, step + t["final_stage_steps"])

            if step % t["eval_every"] == 0 or step == max_steps:
                ev = evaluate(model, eval_batches, vocab, cfg["eval"]["router_threshold"])
                row = {"step": step, "stage": stage, "lr": lr, **parts,
                       **{p: ev[p]["accuracy"] for p in PATHS},
                       "note_slots": ev["notes"]["slots"], "router_fired": ev["routed"]["router_fired"],
                       "routed_slots": ev["routed"]["slots"],
                       "wall_s": time.perf_counter() - t0}
                log.write(json.dumps(row) + "\n")
                log.flush()
                print(f"step={step} stage={stage} notes={ev['notes']['accuracy']:.3f} "
                      f"routed={ev['routed']['accuracy']:.3f} detailed={ev['detailed']['accuracy']:.3f} "
                      f"| slots={ev['notes']['slots']:.1f} routed_slots={ev['routed']['slots']:.1f} "
                      f"fired={ev['routed']['router_fired']:.2f} "
                      f"| L_notes={parts['task_notes']:.3f} suff={parts['sufficiency']:.3f} "
                      f"n_notes={parts['note_count']:.1f} overlap={parts['separation']:.3f}", flush=True)
                if best is None or ev["routed"]["accuracy"] >= best[0]["routed"]["accuracy"]:
                    best = (ev, step, {k: v.detach().clone() for k, v in model.state_dict().items()})
        log.close()

        ev, best_step, state = best
        ckpt = ROOT / "results" / "checkpoints" / f"{EXPERIMENT}_seed{seed}.pt"
        torch.save(state, ckpt)
        model.load_state_dict(state)
        sample = model.compiler.readable(model.compile(eval_batches[0]["ctx"][:1]).fields,
                                         eval_batches[0]["ctx"][:1], vocab.itos, threshold=0.0)
        print("sample notes:", json.dumps(sample[:6], indent=None), flush=True)

        for path in PATHS:
            RunRecord(
                model="snced_general", condition=path, dataset="synthetic_lecture_v1",
                seed=seed, context_length=int(eval_batches[0]["ctx"].size(1)),
                parameters=count_params(model), accuracy=ev[path]["accuracy"],
                task_success=ev[path]["accuracy"], note_count=ev["notes"]["slots"],
                fallback_rate=ev["routed"]["router_fired"], peak_host_memory_mb=peak_rss_mb(),
                peak_gpu_memory_mb=gpu_memory_mb(), train_wall_s=time.perf_counter() - t0,
                extra={"accuracy_by_type": ev[path]["by_type"], "memory_slots": ev[path]["slots"],
                       "best_step": best_step, "loss_weights": w, "note_labels": "none (learned)",
                       "router": "learned sufficiency", "compiler": "open-vocabulary pointers",
                       "sample_notes": sample[:6]},
            ).save(EXPERIMENT)
        print(f"seed={seed} FINAL notes={ev['notes']['accuracy']:.4f} routed={ev['routed']['accuracy']:.4f} "
              f"detailed={ev['detailed']['accuracy']:.4f} slots={ev['notes']['slots']:.1f}", flush=True)


if __name__ == "__main__":
    main()
