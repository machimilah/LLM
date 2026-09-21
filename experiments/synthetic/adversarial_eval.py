"""Adversarial evaluation of SN-CED memory (plan section 20).

Runs the trained SN-CED models zero-shot on lecture variants that stress lossy
semantic memory: a later statement overriding an earlier one, and values that
need more tokens than a note can hold. Full context, predicted notes and gold
notes are compared on the same questions, so a drop can be attributed to the
compiler (predicted vs gold) or to the note schema itself (gold vs full).

Usage: python experiments/synthetic/adversarial_eval.py [--seeds 1 2 3]
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
from end_to_end import pad  # noqa: E402

from snced.adversarial import VARIANTS, make_variant, sample_variant_questions  # noqa: E402
from snced.data import PAD, QTYPE_NAMES, SEP, VALUES, Vocab  # noqa: E402
from snced.metrics import mean_std  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.models.snced import SNCED  # noqa: E402
from snced.runlog import RunRecord, seed_everything  # noqa: E402

EXPERIMENT = "adversarial"
CONDITIONS = ("full_context", "snm_notes", "snm_gold_notes")


def make_eval_batch(vocab: Vocab, items):
    return {
        "ctx": pad([vocab.encode(lec.tokens()) for lec, _ in items]),
        "q": pad([vocab.encode(q.text.split()) for _, q in items]),
        "qtype": torch.tensor([q.hops for _, q in items]),
        "answers": [q.answer for _, q in items],
    }


@torch.no_grad()
def generate_answers(base: CEDBaseline, mem, mem_pad, b, vocab: Vocab, max_tokens: int) -> list[str]:
    """Greedy chain decoding; the answer is whatever follows the final hop."""
    out = [""] * len(b["answers"])
    for k in QTYPE_NAMES:
        idx = (b["qtype"] == k).nonzero().squeeze(1)
        if len(idx) == 0:
            continue
        qk = b["q"][idx]
        qk = qk[:, : int((qk[0] != PAD).sum())]
        prompt = torch.cat([qk, torch.full((len(idx), 1), SEP)], dim=1)
        gen = base.generate(mem[idx], mem_pad[idx], prompt, k + max_tokens)
        for row, j in enumerate(idx.tolist()):
            toks = [vocab.itos[t] for t in gen[row].tolist()]
            # Drop the intermediate hop entities; keep the trailing value tokens.
            out[j] = " ".join([t for t in toks if t in VALUES][:max_tokens])
    return out


def score(pred: str, answer: str) -> bool:
    return pred.split()[: len(answer.split())] == answer.split()


@torch.no_grad()
def evaluate(model: SNCED, vocab: Vocab, items, variant: str) -> dict:
    base = model.base
    max_tokens = 2 if variant == "exact_value" else 1
    stats = {c: {"correct": 0, "n": 0, "slots": 0.0} for c in CONDITIONS}
    for i in range(0, len(items), 48):
        part = items[i : i + 48]
        b = make_eval_batch(vocab, part)
        ctx_pad = b["ctx"] == PAD
        states = base.encode_states(b["ctx"])
        detail = base.mem_proj(states)
        notes = model.compile(b["ctx"], states)
        note_mem, note_pad = model.note_memory(notes)
        gold = model.gold_notes([lec for lec, _ in part])
        gold_mem, gold_pad = model.note_memory(gold)
        for cond, (mem, mp) in (("full_context", (detail, ctx_pad)), ("snm_notes", (note_mem, note_pad)),
                                ("snm_gold_notes", (gold_mem, gold_pad))):
            preds = generate_answers(base, mem, mp, b, vocab, max_tokens)
            s = stats[cond]
            s["correct"] += sum(score(p, a) for p, a in zip(preds, b["answers"]))
            s["n"] += len(part)
            s["slots"] += float((~mp).float().sum())
    return {c: {"accuracy": s["correct"] / s["n"], "mean_slots": s["slots"] / s["n"]} for c, s in stats.items()}


def main() -> None:
    cfg = load_config("snced.yaml")
    base_cfg = load_config(cfg["base_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--n-docs", type=int, default=80)
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
        for variant in VARIANTS:
            rng = random.Random(seed * 104729 + hash(variant) % 1000)
            items = []
            for _ in range(args.n_docs):
                lec = make_variant(rng, variant, base_cfg["data"]["n_entities"], base_cfg["data"]["n_filler"])
                items += [(lec, q) for q in sample_variant_questions(rng, lec, 2)]
            res = evaluate(model, vocab, items, variant)
            for cond, r in res.items():
                rows.append({"seed": seed, "variant": variant, "condition": cond, **r})
                RunRecord(model="snced", condition=cond, dataset=f"adversarial_{variant}", seed=seed,
                          accuracy=r["accuracy"], task_success=r["accuracy"], note_count=r["mean_slots"],
                          extra={"variant": variant, "mean_slots": r["mean_slots"],
                                 "n_questions": len(items)}).save(EXPERIMENT)
            print(f"seed={seed} {variant:16s} " + " ".join(
                f"{c}={res[c]['accuracy']*100:.1f}%" for c in CONDITIONS), flush=True)

    lines = ["# Adversarial evaluation (plan section 20)", "",
             f"Seeds: {args.seeds}. {args.n_docs} documents per variant, 6 questions each, zero-shot:",
             "the models were trained only on the standard lectures.", "",
             "| Variant | Full context | Predicted notes | Gold notes | Note slots |",
             "|---|---:|---:|---:|---:|"]
    for variant in VARIANTS:
        cells = []
        for cond in CONDITIONS:
            rs = [r["accuracy"] for r in rows if r["variant"] == variant and r["condition"] == cond]
            m, s = mean_std(rs)
            cells.append(f"{m*100:.1f} ± {s*100:.1f}%")
        slots = mean_std([r["mean_slots"] for r in rows if r["variant"] == variant
                          and r["condition"] == "snm_notes"])[0]
        lines.append(f"| {variant} | {cells[0]} | {cells[1]} | {cells[2]} | {slots:.1f} |")
    lines += ["", "Reading: a gap between gold notes and full context is a limit of the note schema;",
              "a gap between predicted and gold notes is compiler error."]
    (ROOT / "results" / "tables" / "adversarial.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
