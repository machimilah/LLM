"""Is the indexer ranking badly, or just keeping too few notes?

Sweeps top_k, and swaps the indexer's own scorer for the router's
note_relevance head - which IS trained, via the span-reread path - to separate
"wrong ranking" from "not enough slots".
"""
import random, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src")); sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(ROOT / "experiments" / "synthetic"))
import torch, yaml
from end_to_end import make_batch
from snced.data import PAD, QTYPE_NAMES, SEP, VALUES, Vocab, make_lecture, sample_questions
from snced.models.snced_general import SNCEDGeneral
from snced.runlog import seed_everything

cfg = yaml.safe_load((ROOT / "configs" / "snced_general_quick.yaml").read_text())

@torch.no_grad()
def score(model, batches, vocab, k, scorer):
    model.eval(); correct = total = 0
    for b in batches:
        ctx, q = b["ctx"], b["q"]
        states, detailed = model.encode(ctx)
        nb = model.compile(ctx, states)
        q_states = model.backbone.tok(q)
        q_mask = q != model.backbone.tok.padding_idx
        if scorer == "indexer":
            q_vec = (q_states * q_mask.unsqueeze(-1)).sum(1) / q_mask.sum(1, keepdim=True).clamp(min=1)
            s = model.semantic_indexer.score(q_vec, nb.slots, nb.mask)
        else:
            s = model.router.note_relevance(q_states, q_mask, nb.slots, nb.mask)
        kk = min(k, nb.slots.size(1))
        idx = s.topk(kk, dim=1).indices
        mem = nb.slots.gather(1, idx.unsqueeze(-1).expand(-1, -1, nb.slots.size(-1)))
        mask = nb.mask.gather(1, idx)
        pred = torch.full_like(b["y"], -1)
        for t in QTYPE_NAMES:
            i = (b["qtype"] == t).nonzero().squeeze(1)
            if len(i) == 0: continue
            qk = q[i]; qk = qk[:, : int((qk[0] != PAD).sum())]
            prompt = torch.cat([qk, torch.full((len(i), 1), SEP)], dim=1)
            out = model.backbone.generate(mem[i], mask[i], prompt, t + 1)[:, -1].tolist()
            pred[i] = torch.tensor([VALUES.index(vocab.itos[x]) if vocab.itos[x] in VALUES else -1 for x in out])
        correct += int((pred == b["y"]).sum()); total += len(pred)
    return correct / total

vocab = Vocab(); torch.set_num_threads(12)
for seed in (1, 3):
    seed_everything(seed)
    model = SNCEDGeneral(len(vocab), boundary_token=vocab.stoi["."], **cfg["model"])
    model.load_state_dict(torch.load(ROOT/"results"/"checkpoints"/f"snced_general_seed{seed}.pt", map_location="cpu"))
    model.eval()
    d = cfg["data"]; r = random.Random(seed + 10_000); items = []
    for _ in range(cfg["eval"]["n_docs"]):
        lec = make_lecture(r, d["n_entities"], d["n_filler"])
        items += [(lec, qq) for qq in sample_questions(r, lec, cfg["eval"]["queries_per_type"])]
    batches = [make_batch(vocab, items[i:i+48]) for i in range(0, len(items), 48)]
    for scorer in ("indexer", "router"):
        line = f"seed={seed} scorer={scorer:8s} " + "  ".join(
            f"k={k}:{score(model,batches,vocab,k,scorer)*100:5.1f}%" for k in (8, 16, 24, 30))
        print(line, flush=True)
