"""P2: end-to-end training of the small CED baseline on synthetic lectures.

Training data is generated on the fly as multi-query batches under an adaptive
curriculum (question type -> distractor length); evaluation always uses a fixed
held-out set at the full task size. The run is marked healthy only when every
question type reaches the target accuracy (Rule 2).

Usage: python experiments/synthetic/end_to_end.py [--config ced_baseline.yaml] [--seeds 1] [--max-steps N]
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
from _common import ROOT, load_config  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from snced.data import PAD, QTYPE_NAMES, SEP, VALUES, Question, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.runlog import (  # noqa: E402
    RunRecord, count_params, gpu_memory_mb, peak_rss_mb, pick_device, seed_everything, to_device,
)

EXPERIMENT = "ced_baseline"


def pad(seqs: list[list[int]]) -> torch.Tensor:
    out = torch.full((len(seqs), max(map(len, seqs))), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = torch.tensor(s)
    return out


def make_batch(vocab: Vocab, items) -> dict[str, torch.Tensor]:
    return {
        "ctx": pad([vocab.encode(lec.tokens()) for lec, _ in items]),
        "q": pad([vocab.encode(q.text.split()) for _, q in items]),
        "y": torch.tensor([VALUES.index(q.answer) for _, q in items]),
        "qtype": torch.tensor([q.hops for _, q in items]),
    }


def chain_tokens(lec, entity: str, hops: int) -> list[str]:
    """Generative target: each intermediate entity, then the answer value."""
    return lec.chain_entities(entity, hops)[1:] + [lec.answer(entity, hops)]


def make_train_batch(vocab: Vocab, rng: random.Random, n_lectures: int, n_entities: int, n_filler: int, qtypes,
                     generative: bool = False):
    """Multi-query batch: each lecture is asked about every entity for every active type.

    `rows` maps each question to the lecture (context row) it is about. In
    generative mode `q` is question + [SEP] + chain (teacher forcing) and `tgt`
    holds next-token targets, with -100 everywhere except the chain.
    """
    lecs = [make_lecture(rng, n_entities, n_filler) for _ in range(n_lectures)]
    rows, qs, ys, tgts = [], [], [], []
    for i, lec in enumerate(lecs):
        for hops in qtypes:
            for e in lec.entities:
                rows.append(i)
                q = vocab.encode(Question(e, hops, "").text.split())
                ys.append(VALUES.index(lec.answer(e, hops)))
                if generative:
                    chain = vocab.encode(chain_tokens(lec, e, hops))
                    seq = q + [SEP] + chain
                    qs.append(seq[:-1])
                    tgts.append([-100] * len(q) + chain)
                else:
                    qs.append(q)
    batch = {
        "ctx": pad([vocab.encode(lec.tokens()) for lec in lecs]),
        "rows": torch.tensor(rows),
        "q": pad(qs),
        "y": torch.tensor(ys),
    }
    if generative:
        tgt = torch.full(batch["q"].shape, -100, dtype=torch.long)
        for j, t in enumerate(tgts):
            tgt[j, : len(t)] = torch.tensor(t)
        batch["tgt"] = tgt
    return batch


@torch.no_grad()
def evaluate(model: CEDBaseline, batches: list[dict], vocab_itos: list[str]) -> dict:
    model.eval()
    correct = {k: 0 for k in QTYPE_NAMES}
    total = {k: 0 for k in QTYPE_NAMES}
    enc_ms = dec_ms = 0.0
    for b in batches:
        t0 = time.perf_counter()
        mem = model.encode(b["ctx"])
        t1 = time.perf_counter()
        if model.generative:
            # Greedy-decode the chain per question type (same prompt length within a type)
            # and score the final generated token against the answer value.
            pred = torch.full_like(b["y"], -1)
            ctx_pad = b["ctx"] == PAD
            for k in QTYPE_NAMES:
                idx = (b["qtype"] == k).nonzero().squeeze(1)
                if len(idx) == 0:
                    continue
                qk = b["q"][idx]
                qk = qk[:, : int((qk[0] != PAD).sum())]
                prompt = torch.cat([qk, torch.full((len(idx), 1), SEP, device=qk.device)], dim=1)
                out = model.generate(mem[idx], ctx_pad[idx], prompt, k + 1)[:, -1]
                pred[idx] = torch.tensor([VALUES.index(vocab_itos[t]) if vocab_itos[t] in VALUES else -1 for t in out.tolist()])
        else:
            pred = model.decode(mem, b["ctx"] == PAD, b["q"]).argmax(-1)
        t2 = time.perf_counter()
        enc_ms += (t1 - t0) * 1000
        dec_ms += (t2 - t1) * 1000
        for k in QTYPE_NAMES:
            m = b["qtype"] == k
            correct[k] += int((pred[m] == b["y"][m]).sum())
            total[k] += int(m.sum())
    n = sum(total.values())
    by_type = {QTYPE_NAMES[k]: correct[k] / total[k] for k in QTYPE_NAMES}
    return {"accuracy": sum(correct.values()) / n, "by_type": by_type, "prefill_ms": enc_ms / n, "decode_ms": dec_ms / n}


def run(seed: int, cfg: dict, vocab: Vocab, max_steps: int, device: torch.device | None = None) -> RunRecord:
    t, d = cfg["train"], cfg["data"]
    device = device or torch.device("cpu")
    seed_everything(seed)
    eval_rng = random.Random(seed + 10_000)
    eval_items = []
    for _ in range(cfg["eval"]["n_docs"]):
        lec = make_lecture(eval_rng, d["n_entities"], d["n_filler"])
        eval_items += [(lec, q) for q in sample_questions(eval_rng, lec, cfg["eval"]["queries_per_type"])]
    eval_batches = [to_device(make_batch(vocab, eval_items[i : i + 64]), device)
                    for i in range(0, len(eval_items), 64)]

    model = CEDBaseline(len(vocab), **cfg["model"]).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    rng = random.Random(seed)
    log_path = ROOT / "results" / "logs" / f"{EXPERIMENT}_seed{seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w")

    stages = t["curriculum"]
    stage, final_start, stage_log = 0, None, []
    recent = deque(maxlen=t["advance_window"])
    train_tokens, best, t0, step = 0, None, time.perf_counter(), 0
    while step < max_steps:
        step += 1
        n_fill, qtypes = stages[stage]
        # LR: linear warmup, constant during the curriculum, cosine to zero in the final stage.
        lr = t["lr"] * min(1.0, step / t["warmup"])
        if final_start is not None:
            lr *= 0.5 * (1 + math.cos(math.pi * (step - final_start) / t["final_stage_steps"]))
        for g in opt.param_groups:
            g["lr"] = lr

        model.train()
        b = to_device(make_train_batch(vocab, rng, t["lectures_per_batch"], d["n_entities"], n_fill,
                                       qtypes, model.generative), device)
        memory = model.encode(b["ctx"])
        mem, ctx_pad = memory[b["rows"]], (b["ctx"] == PAD)[b["rows"]]
        if model.generative:
            logits = model.decode_all(mem, ctx_pad, b["q"])
            loss = nn.functional.cross_entropy(logits.flatten(0, 1), b["tgt"].flatten(), ignore_index=-100)
            scored = b["tgt"] != -100
            batch_acc = float((logits.argmax(-1) == b["tgt"])[scored].float().mean())
        else:
            logits = model.decode(mem, ctx_pad, b["q"])
            loss = nn.functional.cross_entropy(logits, b["y"])
            batch_acc = float((logits.argmax(-1) == b["y"]).float().mean())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), t["grad_clip"])
        opt.step()
        train_tokens += int((b["ctx"] != PAD).sum() + (b["q"] != PAD).sum())
        recent.append(batch_acc)

        if final_start is None and len(recent) == recent.maxlen and sum(recent) / len(recent) > t["advance_at"]:
            if stage + 1 < len(stages):
                stage += 1
                recent.clear()
                stage_log.append({"step": step, "stage": stage, "config": stages[stage]})
                print(f"step={step} advance to stage {stage}: {stages[stage]}", flush=True)
            if stage == len(stages) - 1:
                final_start = step
                max_steps = min(max_steps, step + t["final_stage_steps"])

        if step % t["eval_every"] == 0 or step == max_steps:
            ev = evaluate(model, eval_batches, vocab.itos)
            row = {"step": step, "loss": float(loss), "stage": stage, "lr": lr,
                   "train_acc": sum(recent) / max(1, len(recent)), "wall_s": time.perf_counter() - t0, **ev}
            log.write(json.dumps(row) + "\n")
            log.flush()
            print(f"step={step} stage={stage} loss={float(loss):.4f} acc={ev['accuracy']:.4f} {ev['by_type']}", flush=True)
            if best is None or ev["accuracy"] >= best[0]["accuracy"]:
                best = (ev, step, {k: v.clone() for k, v in model.state_dict().items()})
    log.close()
    wall = time.perf_counter() - t0

    ev, best_step, state = best
    ckpt = ROOT / "results" / "checkpoints" / f"{EXPERIMENT}_seed{seed}.pt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    # A smoke run must never overwrite a trained checkpoint: a 2-step --device
    # test destroyed ced_baseline_seed1.pt once already (see FAILED_RUNS.md).
    if step < 1000 and ckpt.exists():
        print(f"not saving: {step} steps is a smoke run and {ckpt.name} already exists", flush=True)
    else:
        torch.save(state, ckpt)
    ctx_len = int(sum((b["ctx"] != PAD).sum() for b in eval_batches) / len(eval_items))
    target = cfg["eval"]["target_accuracy"]
    rec = RunRecord(
        model="ced_baseline",
        condition="detailed_memory_only",
        dataset="synthetic_lecture_v1",
        seed=seed,
        context_length=ctx_len,
        parameters=count_params(model),
        training_tokens=train_tokens,
        accuracy=ev["accuracy"],
        task_success=ev["accuracy"],
        peak_host_memory_mb=peak_rss_mb(),
        peak_gpu_memory_mb=gpu_memory_mb(),
        kv_cache_bytes=model.memory_bytes(ctx_len),
        prefill_ms=ev["prefill_ms"],
        decode_ms=ev["decode_ms"],
        total_task_ms=ev["prefill_ms"] + ev["decode_ms"],
        train_wall_s=wall,
        extra={
            "accuracy_by_type": ev["by_type"],
            "best_step": best_step,
            "steps": step,
            "reached_final_stage": final_start is not None,
            "stage_log": stage_log,
            "healthy": all(a >= target for a in ev["by_type"].values()),
            "target_accuracy": target,
            "model_config": cfg["model"],
            "checkpoint": str(ckpt.relative_to(ROOT)),
        },
    )
    rec.save(EXPERIMENT)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="ced_baseline.yaml")
    known, _ = ap.parse_known_args()
    cfg = load_config(known.config)
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["train"]["seeds"])
    ap.add_argument("--max-steps", type=int, default=cfg["train"]["max_steps"])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    device = pick_device(args.device)
    print(f"device: {device}", flush=True)
    vocab = Vocab()
    for seed in args.seeds:
        r = run(seed, cfg, vocab, args.max_steps, device)
        print(f"seed={seed} FINAL acc={r.accuracy:.4f} {r.extra['accuracy_by_type']} healthy={r.extra['healthy']}")


if __name__ == "__main__":
    main()
