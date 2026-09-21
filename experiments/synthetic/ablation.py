"""Experiment A: representation and fallback ablation (plan section 11.1).

Usage: python experiments/synthetic/ablation.py [--seeds 123 456 789]
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import load_config  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from snced.data import PAD, SEP, VALUES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.memory import CONDITIONS, build_memory  # noqa: E402
from snced.models.reasoner import Reasoner  # noqa: E402
from snced.runlog import RunRecord, count_params, peak_rss_mb, seed_everything  # noqa: E402

EXPERIMENT = "ablation"


def make_split(rng: random.Random, n: int, data_cfg: dict):
    """Balanced direct/one-hop/two-hop questions, one of each per lecture."""
    items = []
    while len(items) < n:
        lec = make_lecture(rng, data_cfg["n_entities"], data_cfg["n_filler"])
        items += [(lec, q) for q in sample_questions(rng, lec, per_type=1)]
    return items[:n]


def encode_split(vocab: Vocab, items, condition: str, seed: int, split_id: int, noise_p: float):
    seqs, labels, qtypes, active, fallback, lecture_tokens = [], [], [], [], [], []
    for i, (lec, q) in enumerate(items):
        # Per-item rng: noisy and fallback conditions see identical corruptions.
        rng = random.Random(seed * 1_000_003 + split_id * 100_000 + i)
        mem = build_memory(lec, q, condition, rng, noise_p)
        q_tokens = q.text.split()
        seqs.append(vocab.encode(q_tokens) + [SEP] + vocab.encode(mem.tokens))
        labels.append(VALUES.index(q.answer))
        qtypes.append(q.hops)
        active.append(len(q_tokens) + len(mem.tokens))
        fallback.append(mem.fallback)
        lecture_tokens.append(len(lec.tokens()))
    lengths = torch.tensor([len(s) for s in seqs])
    ids = torch.full((len(seqs), int(lengths.max())), PAD, dtype=torch.long)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s)
    return {
        "ids": ids,
        "lengths": lengths,
        "y": torch.tensor(labels),
        "qtype": torch.tensor(qtypes),
        "active": active,
        "fallback": fallback,
        "lecture_tokens": lecture_tokens,
    }


@torch.no_grad()
def predict(model: nn.Module, d: dict) -> torch.Tensor:
    model.eval()
    return model(d["ids"], d["lengths"]).argmax(-1)


def run(seed: int, condition: str, cfg: dict, vocab: Vocab) -> RunRecord:
    a = cfg["ablation"]
    seed_everything(seed)
    rng = random.Random(seed)
    splits = [make_split(rng, a[k], cfg["data"]) for k in ("n_train", "n_val", "n_test")]
    tr, va, te = (encode_split(vocab, s, condition, seed, j, a["noise_p"]) for j, s in enumerate(splits))

    seed_everything(seed)  # identical init for every condition
    model = Reasoner(len(vocab))
    opt = torch.optim.Adam(model.parameters(), lr=a["lr"])
    best_val, best_state, train_tokens = -1.0, None, 0
    t0 = time.perf_counter()
    for _ in range(a["epochs"]):
        model.train()
        perm = torch.randperm(len(tr["y"]))
        for s in range(0, len(perm), a["batch_size"]):
            idx = perm[s : s + a["batch_size"]]
            loss = nn.functional.cross_entropy(model(tr["ids"][idx], tr["lengths"][idx]), tr["y"][idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_tokens += int(tr["lengths"][idx].sum())
        val_acc = float((predict(model, va) == va["y"]).float().mean())
        if val_acc > best_val:
            best_val, best_state = val_acc, {k: v.clone() for k, v in model.state_dict().items()}
    wall = time.perf_counter() - t0
    model.load_state_dict(best_state)

    t1 = time.perf_counter()
    pred = predict(model, te)
    infer_ms = (time.perf_counter() - t1) * 1000 / len(te["y"])
    correct = pred == te["y"]
    by_type = {name: float(correct[te["qtype"] == k].float().mean()) for k, name in enumerate(("direct", "one_hop", "two_hop"))}
    mean_active = sum(te["active"]) / len(te["active"])
    mean_lecture = sum(te["lecture_tokens"]) / len(te["lecture_tokens"])
    rec = RunRecord(
        model="gru_reasoner",
        condition=condition,
        dataset="synthetic_lecture_v1",
        seed=seed,
        context_length=int(mean_lecture),
        parameters=count_params(model),
        training_tokens=train_tokens,
        fallback_rate=sum(te["fallback"]) / len(te["fallback"]),
        accuracy=float(correct.float().mean()),
        task_success=float(correct.float().mean()),
        peak_host_memory_mb=peak_rss_mb(),
        total_task_ms=infer_ms,
        train_wall_s=wall,
        extra={
            "accuracy_by_type": by_type,
            "best_val_accuracy": best_val,
            "mean_active_tokens": mean_active,
            "mean_lecture_tokens": mean_lecture,
            # One read of the lecture to build the index/notes + N answered queries.
            "token_exposure_100q": mean_lecture + a["exposure_questions"] * mean_active,
            "query_conditioned_notes": True,
        },
    )
    rec.save(EXPERIMENT)
    return rec


def main() -> None:
    cfg = load_config("synthetic_small.yaml")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=cfg["ablation"]["seeds"])
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS))
    args = ap.parse_args()
    torch.set_num_threads(4)
    vocab = Vocab()
    for seed in args.seeds:
        for cond in args.conditions:
            r = run(seed, cond, cfg, vocab)
            print(
                f"seed={seed} {cond:14s} acc={r.accuracy:.4f} active={r.extra['mean_active_tokens']:.2f} "
                f"fallback={r.fallback_rate:.3f} params={r.parameters} wall={r.train_wall_s:.1f}s",
                flush=True,
            )


if __name__ == "__main__":
    main()
