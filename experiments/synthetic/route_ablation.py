"""Why does `routed` score far below `notes`?

The routed path concatenates three memories (local tokens + indexed notes +
optionally reread spans) where the notes path hands the decoder note slots
alone. This ablates one component at a time on an ALREADY TRAINED checkpoint -
no retraining - to find which addition costs the accuracy.

Arms, all evaluated on the same eval set as training used:
  notes            note slots only (the 0.99 reference)
  routed           exactly what the router builds today
  routed_no_local  same, minus the local-token tier
  routed_no_index  same, but all notes instead of the indexer's top-k
  routed_no_detail  same, with the detail gate forced shut
  routed_notes_only  indexed notes alone, nothing concatenated
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "synthetic"))

import torch  # noqa: E402
import yaml  # noqa: E402
from end_to_end import make_batch  # noqa: E402

from snced.data import PAD, QTYPE_NAMES, SEP, VALUES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.models.snced_general import SNCEDGeneral  # noqa: E402
from snced.runlog import seed_everything  # noqa: E402

cfg = yaml.safe_load((ROOT / "configs" / "snced_general_quick.yaml").read_text())


@torch.no_grad()
def build_memory(model, b, arm, threshold=0.5):
    ctx, q = b["ctx"], b["q"]
    states, detailed = model.encode(ctx)
    detail_mask = ctx == PAD
    nb = model.compile(ctx, states)
    if arm == "notes":
        return nb.slots, nb.mask

    q_states = model.backbone.tok(q)
    q_mask = q != model.backbone.tok.padding_idx
    q_vec = (q_states * q_mask.unsqueeze(-1)).sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)
    p_detail = model.router(q_states, q_mask, nb.slots, nb.mask, nb.fields.confidence)
    gate = (p_detail > threshold).float()
    if arm == "routed_no_detail":
        gate = torch.zeros_like(gate)

    if arm == "routed_no_index":
        notes, notes_mask = nb.slots, nb.mask
    else:
        notes, notes_mask, _ = model.semantic_indexer(q_vec, nb.slots, nb.mask)
    if arm == "routed_notes_only":
        return notes, notes_mask

    parts, masks = [], []
    if arm != "routed_no_local":
        local, local_mask = model.local_memory(detailed, detail_mask)
        parts.append(local)
        masks.append(local_mask)
    parts.append(notes)
    masks.append(notes_mask)

    extra, extra_mask, _ = model._source_spans(q_states, q_mask, nb, detailed)
    parts.append(extra * gate.view(-1, 1, 1))
    masks.append(extra_mask | (gate < 1e-6).view(-1, 1))
    return torch.cat(parts, 1), torch.cat(masks, 1)


@torch.no_grad()
def score(model, batches, vocab, arm):
    model.eval()
    correct = total = 0
    slots = 0.0
    for b in batches:
        mem, mask = build_memory(model, b, arm)
        pred = torch.full_like(b["y"], -1)
        for k in QTYPE_NAMES:
            idx = (b["qtype"] == k).nonzero().squeeze(1)
            if len(idx) == 0:
                continue
            qk = b["q"][idx]
            qk = qk[:, : int((qk[0] != PAD).sum())]
            prompt = torch.cat([qk, torch.full((len(idx), 1), SEP)], dim=1)
            out = model.backbone.generate(mem[idx], mask[idx], prompt, k + 1)[:, -1].tolist()
            pred[idx] = torch.tensor([VALUES.index(vocab.itos[t]) if vocab.itos[t] in VALUES else -1
                                      for t in out])
        correct += int((pred == b["y"]).sum())
        total += len(pred)
        slots += float((~mask).float().sum(-1).mean())
    return correct / total, slots / len(batches)


ARMS = ["notes", "routed", "routed_no_local", "routed_no_index",
        "routed_no_detail", "routed_notes_only"]

vocab = Vocab()
torch.set_num_threads(12)
rows = {}
for seed in (1, 3):
    seed_everything(seed)
    model = SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."], **cfg["model"])
    model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"snced_general_seed{seed}.pt",
                                     map_location="cpu"))
    model.eval()

    d = cfg["data"]
    eval_rng = random.Random(seed + 10_000)
    items = []
    for _ in range(cfg["eval"]["n_docs"]):
        lec = make_lecture(eval_rng, d["n_entities"], d["n_filler"])
        items += [(lec, q) for q in sample_questions(eval_rng, lec, cfg["eval"]["queries_per_type"])]
    batches = [make_batch(vocab, items[i : i + 48]) for i in range(0, len(items), 48)]

    for arm in ARMS:
        acc, slots = score(model, batches, vocab, arm)
        rows[(seed, arm)] = (acc, slots)
        print(f"seed={seed} {arm:20s} acc={acc:.4f} slots={slots:.1f}", flush=True)

print("\n| Arm | seed 1 | seed 3 | slots |")
print("|---|---:|---:|---:|")
for arm in ARMS:
    a1, s1 = rows[(1, arm)]
    a3, _ = rows[(3, arm)]
    print(f"| {arm} | {a1*100:.1f}% | {a3*100:.1f}% | {s1:.0f} |")
