"""Decisive controlled comparison: CED baseline vs CED + Semantic Note Memory.

Starting from a healthy CED baseline checkpoint (same seed):
  Stage A  encoder frozen; train the query-independent note compiler.
  Stage B  encoder + compiler frozen; fine-tune the shared decoder on both
           detailed memory and predicted-note memory.
Then every condition is evaluated on the same held-out lectures at several
context lengths (only the first matches training):

  ced_baseline      original checkpoint, full detailed memory
  snced_detailed    SN-CED model, full detailed memory (compute-matched control)
  snced_notes       SN-CED model, predicted notes only
  snced_gold_notes  SN-CED model, gold notes (upper bound; isolates compiler errors)
  snced_fallback    predicted notes; lectures with a low-confidence note use detailed memory

Usage: python experiments/synthetic/snced_compare.py [--seeds 1]
"""

from __future__ import annotations

import argparse
import copy
import math
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT, load_config  # noqa: E402

import torch  # noqa: E402
from end_to_end import make_batch, make_train_batch, pad  # noqa: E402
from torch import nn  # noqa: E402

from snced.data import PAD, QTYPE_NAMES, SEP, VALUES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.metrics import decoder_memory_flops, decoder_query_flops, encoder_flops  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.models.snced import SNCED, chunk_labels  # noqa: E402
from snced.router import required_note_confidence  # noqa: E402
from snced.runlog import (  # noqa: E402
    RunRecord, count_params, gpu_memory_mb, peak_rss_mb, pick_device, seed_everything, to_device,
)

EXPERIMENT = "snced_compare"
CKPT_TAG = ""  # set by --tag; keeps checkpoints of different variants apart
CONDITIONS = ("ced_baseline", "snced_detailed", "snced_notes", "snced_gold_notes", "snced_fallback")


def cosine_lr(step: int, total: int, base: float, warmup: int) -> float:
    return base * min(1.0, step / warmup) * 0.5 * (1 + math.cos(math.pi * step / total))


@torch.no_grad()
def cache_spans(model: SNCED, vocab: Vocab, lecs) -> tuple[torch.Tensor, torch.Tensor, dict]:
    """Encode lectures once with the frozen encoder; keep per-sentence states and gold labels."""
    spans, masks = [], []
    for i in range(0, len(lecs), 32):
        ctx = pad([vocab.encode(lec.tokens()) for lec in lecs[i : i + 32]])
        s, m, _ = model.chunk_spans(ctx, model.base.encode_states(ctx))
        spans.append(s)
        masks.append(m)
    L = max(s.size(1) for s in spans)
    spans = torch.cat([nn.functional.pad(s, (0, 0, 0, L - s.size(1))) for s in spans])
    masks = torch.cat([nn.functional.pad(m, (0, L - m.size(1)), value=True) for m in masks])
    return spans, masks, chunk_labels(lecs)


@torch.no_grad()
def extraction_rate(model: SNCED, spans, masks, y) -> float:
    lo = model.head_logits(spans, masks)
    kind = lo["kind"].argmax(-1)
    ok = kind == y["kind"]
    fact = y["kind"] != 0  # KINDS[0] == "filler"
    ok &= ~fact | (lo["anchor"].argmax(-1) == y["anchor"])
    ok &= (y["value"] == -100) | (lo["value"].argmax(-1) == y["value"])
    ok &= (y["link"] == -100) | (lo["link"].argmax(-1) == y["link"])
    return float(ok.float().mean())


def train_compiler(model: SNCED, vocab: Vocab, cfg: dict, rng: random.Random, n_entities: int) -> dict:
    """Stage A on cached encoder states; stops once held-out extraction passes the gate."""
    t = cfg["train"]
    pool = t.get("compiler_train_fillers", [t["n_filler"]])
    lecs = [make_lecture(rng, n_entities, rng.choice(pool)) for _ in range(t["compiler_pool_lectures"])]
    val = [make_lecture(rng, n_entities, rng.choice(pool)) for _ in range(t["compiler_val_lectures"])]
    spans, masks, y = cache_spans(model, vocab, lecs)
    vspans, vmasks, vy = cache_spans(model, vocab, val)
    opt = torch.optim.AdamW(model.compiler_parameters(), lr=t["compiler_lr"])
    ce = nn.functional.cross_entropy
    history, step, rate = [], 0, 0.0
    while step < t["compiler_max_steps"]:
        step += 1
        model.train()
        idx = torch.randint(0, len(spans), (t["compiler_batch_chunks"],))
        lo = model.head_logits(spans[idx], masks[idx])
        loss = ce(lo["kind"], y["kind"][idx])
        for head in ("anchor", "value", "link"):
            if (y[head][idx] != -100).any():
                loss = loss + ce(lo[head], y[head][idx], ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 200 == 0:
            model.eval()
            rate = extraction_rate(model, vspans, vmasks, vy)
            history.append({"step": step, "loss": float(loss), "val_extraction": rate})
            print(f"  compiler step={step} loss={float(loss):.4f} val_extraction={rate:.4f}", flush=True)
            if rate >= t["compiler_gate"]:
                break
    return {"steps": step, "val_extraction": rate, "passed_gate": rate >= t["compiler_gate"], "history": history}


@torch.no_grad()
def val_accuracy(model: SNCED, vocab: Vocab, b: dict, memory: str) -> float:
    model.eval()
    base = model.base
    if memory == "gold_notes":
        mem, pad = model.note_memory(model.gold_notes(b["lectures"]))
    else:
        mem, pad = base.encode(b["ctx"]), b["ctx"] == PAD
    return float((predict(base, mem, pad, b, vocab) == b["y"]).float().mean())


def train_decoder(model: SNCED, vocab: Vocab, cfg: dict, rng: random.Random, n_entities: int) -> dict:
    """Stage B: decoder (+ note encoder) on detailed and predicted-note memory.

    Gate: continue until gold-note validation accuracy reaches `decoder_gate`
    (and the detailed path stays >= 0.99), between min and max steps.
    """
    t = cfg["train"]
    base = model.base
    modules = [*base.decoder, base.dec_norm, base.head, model.note_encoder]
    params = [p for m in modules for p in m.parameters()]
    opt = torch.optim.AdamW(params, lr=t["decoder_lr"], weight_decay=0.01)
    ce = nn.functional.cross_entropy
    vrng = random.Random(rng.random())
    val_items = []
    for _ in range(t["decoder_val_lectures"]):
        lec = make_lecture(vrng, n_entities, t["n_filler"])
        val_items += [(lec, q) for q in sample_questions(vrng, lec, 2)]
    val = to_device(make_batch(vocab, val_items), next(base.parameters()).device)
    val["lectures"] = [lec for lec, _ in val_items]
    train_tokens, recent, history, step = 0, [], [], 0
    gold_acc = detail_acc = 0.0
    while step < t["decoder_max_steps"]:
        step += 1
        model.train()
        for g in opt.param_groups:
            g["lr"] = t["decoder_lr"] * min(1.0, step / t["warmup"])
        b = to_device(make_train_batch(vocab, rng, t["lectures_per_batch"], n_entities, t["n_filler"],
                                       [0, 1, 2], True), next(base.parameters()).device)
        with torch.no_grad():
            detail = base.encode(b["ctx"])
            notes = model.compile(b["ctx"])
        note_mem, note_pad = model.note_memory(notes)  # note encoder is trainable
        r = b["rows"]
        loss = 0.0
        for mem, mem_pad in ((detail, b["ctx"] == PAD), (note_mem, note_pad)):
            logits = base.decode_all(mem[r], mem_pad[r], b["q"])
            loss = loss + ce(logits.flatten(0, 1), b["tgt"].flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        train_tokens += int((b["ctx"] != PAD).sum() + (notes.note_ids != PAD).sum())
        recent.append(float(loss))
        if step % 250 == 0:
            gold_acc, detail_acc = val_accuracy(model, vocab, val, "gold_notes"), val_accuracy(model, vocab, val, "detailed")
            history.append({"step": step, "loss": sum(recent[-250:]) / 250, "val_gold_notes": gold_acc,
                            "val_detailed": detail_acc})
            print(f"  decoder step={step} loss={history[-1]['loss']:.4f} val_gold_notes={gold_acc:.4f} "
                  f"val_detailed={detail_acc:.4f}", flush=True)
            if step >= t["decoder_min_steps"] and gold_acc >= t["decoder_gate"] and detail_acc >= 0.99:
                break
    return {"tokens": train_tokens, "steps": step, "val_gold_notes": gold_acc, "val_detailed": detail_acc,
            "passed_gate": gold_acc >= t["decoder_gate"], "history": history}


@torch.no_grad()
def predict(base: CEDBaseline, mem: torch.Tensor, mem_pad: torch.Tensor, b: dict, vocab: Vocab) -> torch.Tensor:
    """Greedy chain decoding per question type; returns the predicted answer index (-1 = not a value)."""
    pred = torch.full_like(b["y"], -1)
    for k in QTYPE_NAMES:
        idx = (b["qtype"] == k).nonzero().squeeze(1)
        if len(idx) == 0:
            continue
        qk = b["q"][idx]
        qk = qk[:, : int((qk[0] != PAD).sum())]
        prompt = torch.cat([qk, torch.full((len(idx), 1), SEP, device=qk.device)], dim=1)
        out = base.generate(mem[idx], mem_pad[idx], prompt, k + 1)[:, -1].tolist()
        pred[idx] = torch.tensor([VALUES.index(vocab.itos[t]) if vocab.itos[t] in VALUES else -1 for t in out])
    return pred


@torch.no_grad()
def evaluate(ref: CEDBaseline, model: SNCED, vocab: Vocab, items, threshold: float,
             policy: str = "question") -> dict:
    """Run every condition on the same questions. Timings are per lecture / per question."""
    ref.eval()
    model.eval()
    base = model.base
    stats = {c: {"correct": {k: 0 for k in QTYPE_NAMES}, "total": {k: 0 for k in QTYPE_NAMES},
                 "mem_slots": 0.0, "mem_ms": 0.0, "decode_ms": 0.0, "fallback": 0, "lectures": 0}
             for c in CONDITIONS}
    exact = total_chunks = 0
    # Each question references its own lecture row; group items by lecture to time memory building.
    for i in range(0, len(items), 48):
        chunk_items = items[i : i + 48]
        b = to_device(make_batch(vocab, chunk_items), next(base.parameters()).device)
        lecs = [lec for lec, _ in chunk_items]
        n = len(chunk_items)
        ctx_pad = b["ctx"] == PAD

        t0 = time.perf_counter()
        ref_mem = ref.encode(b["ctx"])
        t_ref = time.perf_counter() - t0

        t0 = time.perf_counter()
        states = base.encode_states(b["ctx"])
        detail = base.mem_proj(states)
        t_detail = time.perf_counter() - t0
        t0 = time.perf_counter()
        notes = model.compile(b["ctx"], states)
        note_mem, note_pad = model.note_memory(notes)
        t_notes = time.perf_counter() - t0 + t_detail  # notes need the full encoder pass first
        gold = model.gold_notes(lecs)
        gold_mem, gold_pad = model.note_memory(gold)

        # Calibrated default: route per question, so only the notes a question
        # needs can trigger fallback (results/tables/router_calibration.md).
        if policy == "question":
            conf = torch.tensor([required_note_confidence(notes.preds[j], q.entity, q.hops)
                                 for j, (_, q) in enumerate(chunk_items)])
        else:
            conf = notes.min_confidence
        use_fb = conf < threshold
        fb_mem = torch.zeros(n, max(detail.size(1), note_mem.size(1)), base.d_model, device=detail.device)
        fb_pad = torch.ones(n, fb_mem.size(1), dtype=torch.bool, device=detail.device)
        for j in range(n):
            src, src_pad = (detail[j], ctx_pad[j]) if use_fb[j] else (note_mem[j], note_pad[j])
            fb_mem[j, : src.size(0)], fb_pad[j, : src.size(0)] = src, src_pad

        for lec, preds in zip(lecs, notes.preds):
            for c, p in zip(lec.chunks, preds):
                total_chunks += 1
                exact += p["kind"] == c.kind and p["anchor"] == c.anchor and p["target"] == c.target

        runs = {
            "ced_baseline": (ref, ref_mem, ctx_pad, t_ref, (~ctx_pad).sum(1)),
            "snced_detailed": (base, detail, ctx_pad, t_detail, (~ctx_pad).sum(1)),
            "snced_notes": (base, note_mem, note_pad, t_notes, (~note_pad).sum(1)),
            "snced_gold_notes": (base, gold_mem, gold_pad, t_notes, (~gold_pad).sum(1)),
            "snced_fallback": (base, fb_mem, fb_pad, t_notes, (~fb_pad).sum(1)),
        }
        for cond, (dec, mem, pad, t_mem, slots) in runs.items():
            t0 = time.perf_counter()
            pred = predict(dec, mem, pad, b, vocab)
            s = stats[cond]
            s["decode_ms"] += (time.perf_counter() - t0) * 1000
            s["mem_ms"] += t_mem * 1000
            s["mem_slots"] += float(slots.float().sum())
            s["lectures"] += n
            if cond == "snced_fallback":
                s["fallback"] += int(use_fb.sum())
            for k in QTYPE_NAMES:
                m = b["qtype"] == k
                s["correct"][k] += int((pred[m] == b["y"][m]).sum())
                s["total"][k] += int(m.sum())

    out = {}
    for cond, s in stats.items():
        nq = sum(s["total"].values())
        out[cond] = {
            "accuracy": sum(s["correct"].values()) / nq,
            "by_type": {QTYPE_NAMES[k]: s["correct"][k] / s["total"][k] for k in QTYPE_NAMES},
            "mem_slots": s["mem_slots"] / s["lectures"],
            "memory_build_ms_per_lecture": s["mem_ms"] / s["lectures"],
            "decode_ms_per_question": s["decode_ms"] / nq,
            "fallback_rate": s["fallback"] / s["lectures"],
        }
    out["_compiler_exact_chunk_extraction"] = exact / total_chunks
    return out


def flops_per_lecture(cond: str, T: float, S: float, note_flops: float, mcfg: dict, q: int,
                      lq: float) -> float:
    d, dff, ne, nd, k = mcfg["d_model"], mcfg["d_ff"], mcfg["n_enc"], mcfg["n_dec"], mcfg["conv_kernel"]
    total = encoder_flops(T, d, dff, ne, k)  # every condition reads the full lecture once
    if cond in ("snced_notes", "snced_gold_notes", "snced_fallback"):
        total += note_flops  # build note memory from the notebook (Rule 4)
    total += decoder_memory_flops(S, d, nd) + q * decoder_query_flops(lq, S, d, dff, nd)
    return total


def run(seed: int, cfg: dict, base_cfg: dict, vocab: Vocab, device: torch.device | None = None) -> None:
    seed_everything(seed)
    device = device or torch.device("cpu")
    ckpt = ROOT / "results" / "checkpoints" / f"ced_baseline_seed{seed}.pt"
    ref = CEDBaseline(len(vocab), **base_cfg["model"])
    ref.load_state_dict(torch.load(ckpt, map_location="cpu"))
    ref = ref.to(device)
    model = SNCED(copy.deepcopy(ref), vocab, note_mode=cfg["note_mode"]).to(device)
    for p in model.base.parameters():
        p.requires_grad_(False)
    n_ent = base_cfg["data"]["n_entities"]
    rng = random.Random(seed + 500)

    t0 = time.perf_counter()
    print(f"seed={seed} stage A: note compiler", flush=True)
    comp = train_compiler(model, vocab, cfg, rng, n_ent)
    for m in (*model.base.decoder, model.base.dec_norm, model.base.head):
        for p in m.parameters():
            p.requires_grad_(True)
    for p in model.compiler_parameters():
        p.requires_grad_(False)
    print(f"seed={seed} stage B: decoder on detailed + note memory", flush=True)
    dec = train_decoder(model, vocab, cfg, rng, n_ent)
    dec_tokens = dec["tokens"]
    wall = time.perf_counter() - t0
    torch.save(model.state_dict(), ROOT / "results" / "checkpoints" / f"snced{CKPT_TAG}_seed{seed}.pt")

    compiler_params = sum(p.numel() for p in model.compiler_parameters())
    note_params = sum(p.numel() for p in model.note_encoder.parameters()) if cfg["note_mode"] == "note_slots" else 0
    for n_filler in cfg["eval"]["fillers"]:
        erng = random.Random(seed * 7919 + n_filler)
        items = []
        for _ in range(cfg["eval"]["n_docs"]):
            lec = make_lecture(erng, n_ent, n_filler)
            items += [(lec, q) for q in sample_questions(erng, lec, cfg["eval"]["queries_per_type"])]
        res = evaluate(ref, model, vocab, items, cfg["router"]["fallback_threshold"],
                       cfg["router"].get("policy", "question"))
        T = sum(len(lec.tokens()) for lec, _ in items) / len(items)
        m = base_cfg["model"]
        if cfg["note_mode"] == "note_slots":
            n_notes = res["snced_notes"]["mem_slots"]
            layers = [lay for lay in model.note_encoder if isinstance(lay, nn.Linear)]
            note_flops = n_notes * sum(2 * lay.in_features * lay.out_features for lay in layers)
        else:
            n_notes = res["snced_notes"]["mem_slots"] / 3
            note_flops = encoder_flops(3 * n_notes, m["d_model"], m["d_ff"], m["n_enc"], m["conv_kernel"])
        lq = sum(len(q.text.split()) + 1 + q.hops for _, q in items) / len(items)
        print(f"seed={seed} filler={n_filler} T={T:.0f} extraction={res['_compiler_exact_chunk_extraction']:.4f}")
        for cond in CONDITIONS:
            r = res[cond]
            S = r["mem_slots"]
            flops = flops_per_lecture(cond, T, S, note_flops, base_cfg["model"],
                                      cfg["eval"]["workload_queries_per_lecture"], lq)
            is_snced = cond != "ced_baseline"
            rec = RunRecord(
                model="snced" if is_snced else "ced_baseline",
                condition=cond,
                dataset=f"synthetic_lecture_v1_filler{n_filler}",
                seed=seed,
                context_length=int(T),
                parameters=count_params(ref) + (compiler_params + note_params if is_snced else 0),
                training_tokens=dec_tokens if is_snced else 0,
                note_tokens=3 * n_notes if "notes" in cond or cond == "snced_fallback" else 0.0,
                note_count=n_notes if "notes" in cond or cond == "snced_fallback" else 0.0,
                fallback_rate=r["fallback_rate"],
                accuracy=r["accuracy"],
                task_success=r["accuracy"],
                peak_host_memory_mb=peak_rss_mb(),
                peak_gpu_memory_mb=gpu_memory_mb(),
                kv_cache_bytes=int(ref.memory_bytes(1) * S),
                prefill_ms=r["memory_build_ms_per_lecture"],
                decode_ms=r["decode_ms_per_question"],
                note_compile_ms=r["memory_build_ms_per_lecture"] if "notes" in cond or cond == "snced_fallback" else 0.0,
                total_task_ms=r["memory_build_ms_per_lecture"] + 36 * r["decode_ms_per_question"],
                train_wall_s=wall if is_snced else 0.0,
                estimated_flops=int(flops),
                extra={
                    "n_filler": n_filler,
                    "accuracy_by_type": r["by_type"],
                    "memory_slots": S,
                    "compiler_exact_chunk_extraction": res["_compiler_exact_chunk_extraction"],
                    "compiler_training": {k: v for k, v in comp.items() if k != "history"},
                    "decoder_training": {k: v for k, v in dec.items() if k != "history"},
                    "note_mode": cfg["note_mode"],
                    "note_encoder_parameters": note_params,
                    "compiler_parameters": compiler_params,
                    "fallback_threshold": cfg["router"]["fallback_threshold"],
                    "router_policy": cfg["router"].get("policy", "question"),
                    "flops_workload_queries": cfg["eval"]["workload_queries_per_lecture"],
                    "trained_filler": cfg["train"]["n_filler"],
                    "compiler_train_fillers": cfg["train"].get("compiler_train_fillers"),
                    "note_supervision": "gold chunk labels (extra supervision the baseline does not get)",
                },
            )
            rec.save(EXPERIMENT)
            print(
                f"  {cond:17s} acc={r['accuracy']:.4f} {r['by_type']} slots={S:.1f} "
                f"fallback={r['fallback_rate']:.3f} GFLOPs/lecture={flops / 1e9:.3f} "
                f"decode_ms/q={r['decode_ms_per_question']:.3f}",
                flush=True,
            )


def main() -> None:
    cfg = load_config("snced.yaml")
    base_cfg = load_config(cfg["base_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1])
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run; records go to snced_compare_smoke")
    ap.add_argument("--tag", default="", help="suffix for the results dir and checkpoints")
    args = ap.parse_args()
    global EXPERIMENT, CKPT_TAG
    if args.tag:
        EXPERIMENT = f"{EXPERIMENT}_{args.tag}"
        CKPT_TAG = f"_{args.tag}"
    if args.quick:
        EXPERIMENT = "snced_compare_smoke" + (f"_{args.tag}" if args.tag else "")
        cfg["train"].update(compiler_max_steps=200, compiler_pool_lectures=40, compiler_val_lectures=8,
                            decoder_max_steps=250, decoder_min_steps=250, decoder_val_lectures=4)
        cfg["eval"].update(fillers=[18], n_docs=8)
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    vocab = Vocab()
    for seed in args.seeds:
        run(seed, cfg, base_cfg, vocab, device)


if __name__ == "__main__":
    main()
