"""Memory-representation ablations (plan section 15, Phase 4).

Every condition is evaluated with the same encoder, decoder and harness, on the
SN-CED checkpoints from snced_compare.py:

  full_context        all encoded lecture tokens (the CED baseline memory)
  sliding_window      the last B tokens of the lecture
  rag_topk            sentences ranked by overlap with the question, up to B tokens
  iterative_rag       multi-round lexical expansion from the question's entity
  extractive_facts    sentences the compiler marks as facts, kept as prose
  extractive_facts_b  the same, truncated to B tokens
  snm_notes           anchor + micro-context notes (one slot each)
  snm_keyword_only    notes stripped to the anchor
  snm_phrase_only     notes stripped to relation + target
  snm_fallback        notes, with low-confidence lectures using full context

B is the note-token budget (24 notes x 3 tokens = 72). RAG conditions are
query-conditioned; SNM and the summaries are query-independent.

Usage: python experiments/synthetic/ablations.py [--seeds 1 2 3]
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
from end_to_end import make_batch, make_train_batch, pad  # noqa: E402
from snced_compare import predict  # noqa: E402
from torch import nn  # noqa: E402

from snced.data import ENTITIES, PAD, QTYPE_NAMES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.metrics import decoder_memory_flops, decoder_query_flops, encoder_flops  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.models.snced import SNCED  # noqa: E402
from snced.runlog import RunRecord, count_params, peak_rss_mb, seed_everything  # noqa: E402

EXPERIMENT = "ablations"
TEXT_CONDITIONS = ("full_context", "sliding_window", "rag_topk", "iterative_rag", "extractive_facts",
                   "extractive_facts_b")
NOTE_CONDITIONS = ("snm_notes", "snm_keyword_only", "snm_phrase_only", "snm_fallback")
CONDITIONS = TEXT_CONDITIONS + NOTE_CONDITIONS


def entities_in(tokens: list[str]) -> set[str]:
    return {t for t in tokens if t in ENTITIES}


def select_tokens(lecture, question, condition: str, cfg: dict, fact_chunks: list[int]) -> list[str]:
    """Token-level memory for the text conditions."""
    budget, chunks = cfg["budget"]["tokens"], lecture.chunks
    if condition == "full_context":
        return lecture.tokens()
    if condition == "sliding_window":
        return lecture.tokens()[-budget:]
    if condition == "rag_topk":
        q_ents = entities_in(question.text.split())
        ranked = sorted(range(len(chunks)), key=lambda i: -len(entities_in(chunks[i].tokens) & q_ents))
        picked = [i for i in ranked[: cfg["budget"]["rag_top_k"]] if entities_in(chunks[i].tokens) & q_ents]
        return take_budget(chunks, sorted(picked), budget)
    if condition == "iterative_rag":
        # Round 1: sentences mentioning the question entity. Each later round adds
        # sentences mentioning any entity seen so far (pure lexical expansion).
        seen_ents, picked = entities_in(question.text.split()), []
        for _ in range(cfg["budget"]["iterative_rag_rounds"]):
            new = [i for i in range(len(chunks)) if i not in picked and entities_in(chunks[i].tokens) & seen_ents]
            picked += new
            for i in new:
                seen_ents |= entities_in(chunks[i].tokens)
        return [t for i in sorted(picked) for t in chunks[i].tokens]
    if condition == "extractive_facts":
        return [t for i in fact_chunks for t in chunks[i].tokens]
    if condition == "extractive_facts_b":
        return take_budget(chunks, fact_chunks, budget)
    raise ValueError(condition)


def take_budget(chunks, idx: list[int], budget: int) -> list[str]:
    out: list[str] = []
    for i in idx:
        if len(out) + len(chunks[i].tokens) > budget:
            break
        out += chunks[i].tokens
    return out


def note_ids_for(model: SNCED, notes, condition: str) -> torch.Tensor:
    """Strip note content for the keyword-only / phrase-only ablations."""
    ids = notes.note_ids.clone().view(notes.note_ids.size(0), -1, 3)
    if condition == "snm_keyword_only":
        ids[..., 1:] = PAD
    elif condition == "snm_phrase_only":
        ids[..., 0] = PAD
    return ids.view(notes.note_ids.size(0), -1)


def finetune_content_ablation(model: SNCED, vocab: Vocab, cfg: dict, condition: str, rng: random.Random,
                              n_entities: int, n_filler: int) -> SNCED:
    """Short fine-tune of decoder + note encoder on a stripped note format."""
    m = copy.deepcopy(model)
    c = cfg["content_ablation"]
    params = [p for mod in (*m.base.decoder, m.base.dec_norm, m.base.head, m.note_encoder) for p in mod.parameters()]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=c["lr"], weight_decay=0.01)
    for step in range(1, c["steps"] + 1):
        m.train()
        b = make_train_batch(vocab, rng, c["lectures_per_batch"], n_entities, n_filler, [0, 1, 2], True)
        with torch.no_grad():
            notes = m.compile(b["ctx"])
        stripped = copy.copy(notes)
        stripped.note_ids = note_ids_for(m, notes, condition)
        mem, pad_mask = m.note_memory(stripped)
        r = b["rows"]
        logits = m.base.decode_all(mem[r], pad_mask[r], b["q"])
        loss = nn.functional.cross_entropy(logits.flatten(0, 1), b["tgt"].flatten(), ignore_index=-100)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 200 == 0:
            print(f"    {condition} step={step} loss={float(loss):.4f}", flush=True)
    return m


@torch.no_grad()
def evaluate(models: dict, ref: CEDBaseline, vocab: Vocab, items, cfg: dict, threshold: float) -> dict:
    base_model: SNCED = models["snm_notes"]
    stats = {c: {"correct": {k: 0 for k in QTYPE_NAMES}, "total": {k: 0 for k in QTYPE_NAMES},
                 "slots": 0.0, "tokens": 0.0, "fallback": 0, "n": 0} for c in CONDITIONS}
    for i in range(0, len(items), 48):
        part = items[i : i + 48]
        b = make_batch(vocab, part)
        lecs = [lec for lec, _ in part]
        notes = base_model.compile(b["ctx"])
        fact_chunks = [[j for j, p in enumerate(pr) if p["kind"] != "filler"] for pr in notes.preds]

        for cond in CONDITIONS:
            model = models.get(cond, base_model)
            if cond in TEXT_CONDITIONS:
                toks = [select_tokens(lec, q, cond, cfg, fc) for (lec, q), fc in zip(part, fact_chunks)]
                ctx = pad([vocab.encode(t) if t else [PAD] for t in toks])
                mem, mem_pad = model.base.encode(ctx), ctx == PAD
                n_tokens = (~mem_pad).float().sum(1)
                slots = n_tokens
            else:
                nb = copy.copy(notes)
                nb.note_ids = note_ids_for(model, notes, cond)
                mem, mem_pad = model.note_memory(nb)
                slots = (~mem_pad).float().sum(1)
                n_tokens = (notes.note_ids != PAD).float().sum(1)
                if cond == "snm_fallback":
                    use_fb = notes.min_confidence < threshold
                    full = model.base.encode(b["ctx"])
                    wide = torch.zeros(len(part), max(full.size(1), mem.size(1)), model.base.d_model)
                    wide_pad = torch.ones(len(part), wide.size(1), dtype=torch.bool)
                    for j in range(len(part)):
                        src, sp = (full[j], (b["ctx"] == PAD)[j]) if use_fb[j] else (mem[j], mem_pad[j])
                        wide[j, : src.size(0)], wide_pad[j, : src.size(0)] = src, sp
                    mem, mem_pad = wide, wide_pad
                    slots = (~mem_pad).float().sum(1)
                    n_tokens = torch.where(use_fb, (b["ctx"] != PAD).float().sum(1), n_tokens)
                    stats[cond]["fallback"] += int(use_fb.sum())
            pred = predict(model.base, mem, mem_pad, b, vocab)
            s = stats[cond]
            s["slots"] += float(slots.sum())
            s["tokens"] += float(n_tokens.sum())
            s["n"] += len(part)
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
            "slots": s["slots"] / s["n"],
            "tokens": s["tokens"] / s["n"],
            "fallback_rate": s["fallback"] / s["n"],
        }
    return out


def run(seed: int, cfg: dict, base_cfg: dict, sn_cfg: dict, vocab: Vocab) -> None:
    seed_everything(seed)
    ref = CEDBaseline(len(vocab), **base_cfg["model"])
    ref.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"ced_baseline_seed{seed}.pt"))
    model = SNCED(copy.deepcopy(ref), vocab, note_mode=sn_cfg["note_mode"])
    model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"snced_seed{seed}.pt"))
    model.eval()
    n_ent = base_cfg["data"]["n_entities"]
    rng = random.Random(seed + 900)

    models = {c: model for c in CONDITIONS}
    for cond in ("snm_keyword_only", "snm_phrase_only"):
        print(f"seed={seed} fine-tuning {cond}", flush=True)
        models[cond] = finetune_content_ablation(model, vocab, cfg, cond, rng, n_ent, sn_cfg["train"]["n_filler"])
        models[cond].eval()

    note_params = sum(p.numel() for p in model.note_encoder.parameters())
    comp_params = sum(p.numel() for p in model.compiler_parameters())
    m = base_cfg["model"]
    for n_filler in cfg["eval"]["fillers"]:
        erng = random.Random(seed * 7919 + n_filler)
        items = []
        for _ in range(cfg["eval"]["n_docs"]):
            lec = make_lecture(erng, n_ent, n_filler)
            items += [(lec, q) for q in sample_questions(erng, lec, cfg["eval"]["queries_per_type"])]
        t0 = time.perf_counter()
        res = evaluate(models, ref, vocab, items, cfg, sn_cfg["router"]["fallback_threshold"])
        T = sum(len(lec.tokens()) for lec, _ in items) / len(items)
        lq = sum(len(q.text.split()) + 1 + q.hops for _, q in items) / len(items)
        print(f"seed={seed} filler={n_filler} T={T:.0f} ({time.perf_counter() - t0:.0f}s)", flush=True)
        for cond in CONDITIONS:
            r = res[cond]
            note_layers = [lay for lay in model.note_encoder if isinstance(lay, nn.Linear)]
            if cond.startswith("snm"):
                build = r["slots"] * sum(2 * lay.in_features * lay.out_features for lay in note_layers)
                build += encoder_flops(T, m["d_model"], m["d_ff"], m["n_enc"], m["conv_kernel"])  # compile reads all
            else:
                build = encoder_flops(r["tokens"], m["d_model"], m["d_ff"], m["n_enc"], m["conv_kernel"])
                if cond in ("extractive_facts", "extractive_facts_b"):
                    build += encoder_flops(T, m["d_model"], m["d_ff"], m["n_enc"], m["conv_kernel"])  # selector reads all
            q = cfg["eval"]["workload_queries_per_lecture"]
            # Query-conditioned memories (RAG) are rebuilt for every question;
            # query-independent memories are built once per lecture and reused.
            rebuilds = q if cond in ("rag_topk", "iterative_rag") else 1
            flops = (rebuilds * (build + decoder_memory_flops(r["slots"], m["d_model"], m["n_dec"]))
                     + q * decoder_query_flops(lq, r["slots"], m["d_model"], m["d_ff"], m["n_dec"]))
            rec = RunRecord(
                model="snced" if cond.startswith("snm") else "ced_memory_variant",
                condition=cond,
                dataset=f"synthetic_lecture_v1_filler{n_filler}",
                seed=seed,
                context_length=int(T),
                parameters=count_params(ref) + (comp_params + note_params if cond.startswith("snm") else 0),
                note_tokens=r["tokens"] if cond.startswith("snm") else 0.0,
                note_count=r["slots"] if cond.startswith("snm") else 0.0,
                fallback_rate=r["fallback_rate"],
                accuracy=r["accuracy"],
                task_success=r["accuracy"],
                peak_host_memory_mb=peak_rss_mb(),
                kv_cache_bytes=int(ref.memory_bytes(1) * r["slots"]),
                estimated_flops=int(flops),
                extra={
                    "n_filler": n_filler,
                    "accuracy_by_type": r["by_type"],
                    "memory_slots": r["slots"],
                    "retained_tokens": r["tokens"],
                    "token_budget": cfg["budget"]["tokens"],
                    "query_conditioned": cond in ("rag_topk", "iterative_rag"),
                    "mean_query_len": lq,
                    "finetuned_steps": cfg["content_ablation"]["steps"] if cond in
                    ("snm_keyword_only", "snm_phrase_only") else 0,
                },
            )
            rec.save(EXPERIMENT)
            print(f"  {cond:19s} acc={r['accuracy']:.4f} {r['by_type']} slots={r['slots']:.1f} "
                  f"tokens={r['tokens']:.1f} GFLOPs={flops / 1e9:.3f} fallback={r['fallback_rate']:.3f}", flush=True)


def main() -> None:
    cfg = load_config("ablations.yaml")
    base_cfg = load_config(cfg["base_config"])
    sn_cfg = load_config(cfg["snced_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        global EXPERIMENT
        EXPERIMENT = "ablations_smoke"
        cfg["eval"].update(fillers=[18], n_docs=8)
        cfg["content_ablation"].update(steps=2)
    torch.set_num_threads(args.threads)
    vocab = Vocab()
    for seed in args.seeds:
        run(seed, cfg, base_cfg, sn_cfg, vocab)


if __name__ == "__main__":
    main()
