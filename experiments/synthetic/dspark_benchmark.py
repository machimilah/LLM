"""DSpark benchmark: throughput without changing a single token of output.

Speculative decoding is a systems claim, so it is measured as one:

  correctness   speculative output must be IDENTICAL to greedy, not merely close
  acceptance    accepted draft tokens per verification step
  throughput    tokens per second, and verification passes saved

The draft module is trained briefly against the main model's own greedy
continuations - it only has to predict what the big model would do next.

Honest expectation on this benchmark: SN-CED answers with 1-3 token chains, so
there is almost nothing to speculate about. The value of this harness is that
it proves the machinery is exact and measures acceptance, which is what
transfers to a setting with long generations.

Usage: python experiments/synthetic/dspark_benchmark.py [--seeds 2]
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

from snced.data import PAD, SEP, QTYPE_NAMES, Vocab, make_lecture, sample_questions  # noqa: E402
from snced.models.ced import CEDBaseline  # noqa: E402
from snced.models.dspark import DraftModule, speculative_decode  # noqa: E402
from snced.models.snced import SNCED  # noqa: E402
from snced.runlog import RunRecord, seed_everything  # noqa: E402

EXPERIMENT = "dspark"


class Adapter:
    """Gives the v1 SNCED the interface speculative_decode expects."""

    def __init__(self, model: SNCED) -> None:
        self.model, self.backbone = model, model.base

    def decode_all(self, memory, mask, seq):
        return self.backbone.decode_all(memory, mask, seq)

    def decode_states(self, memory, mask, seq):
        return self.backbone.decode_states(memory, mask, seq)


def train_draft(adapter: Adapter, vocab: Vocab, items, draft: DraftModule, steps: int, block: int):
    """Teach the draft to predict the main model's own next tokens."""
    opt = torch.optim.AdamW(draft.parameters(), lr=1e-3)
    rng = random.Random(0)
    for step in range(steps):
        part = [items[rng.randrange(len(items))] for _ in range(16)]
        b = make_batch(vocab, part)
        with torch.no_grad():
            mem, mask = adapter.model.note_memory(adapter.model.compile(b["ctx"]))
            prompt = torch.cat([b["q"], torch.full((len(part), 1), SEP)], dim=1)
            target = adapter.backbone.generate(mem, mask, prompt, block)
            state = adapter.decode_states(mem, mask, prompt)[:, -1]
        loss = draft.loss(state, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return float(loss)


@torch.no_grad()
def benchmark(adapter: Adapter, vocab: Vocab, items, draft: DraftModule, steps: int):
    identical = True
    plain_s = spec_s = 0.0
    agg = {"drafted": 0, "accepted": 0, "verification_steps": 0, "generated": 0}
    for i in range(0, len(items), 48):
        part = items[i : i + 48]
        b = make_batch(vocab, part)
        mem, mask = adapter.model.note_memory(adapter.model.compile(b["ctx"]))
        prompt = torch.cat([b["q"], torch.full((len(part), 1), SEP)], dim=1)

        t0 = time.perf_counter()
        plain = adapter.backbone.generate(mem, mask, prompt, steps)
        plain_s += time.perf_counter() - t0

        t0 = time.perf_counter()
        spec, stats = speculative_decode(adapter, mem, mask, prompt, steps, draft)
        spec_s += time.perf_counter() - t0

        identical &= bool(torch.equal(plain, spec))
        for k in agg:
            agg[k] += getattr(stats, k)
    n_tokens = agg["generated"]
    return {
        "identical_to_greedy": identical,
        "accepted_per_step": agg["accepted"] / max(agg["verification_steps"], 1),
        "acceptance_rate": agg["accepted"] / max(agg["drafted"], 1),
        "plain_tokens_per_s": n_tokens / max(plain_s, 1e-9),
        "spec_tokens_per_s": n_tokens / max(spec_s, 1e-9),
        "speedup": plain_s / max(spec_s, 1e-9),
        **agg,
    }


def main() -> None:
    cfg = load_config("snced.yaml")
    base_cfg = load_config(cfg["base_config"])
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[2])
    ap.add_argument("--block", type=int, default=3)
    ap.add_argument("--gen-steps", type=int, default=3)
    ap.add_argument("--draft-steps", type=int, default=300)
    ap.add_argument("--n-docs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=12)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    vocab = Vocab()
    lines = ["# DSpark benchmark", "",
             f"Seeds {args.seeds}, draft block {args.block}, {args.gen_steps} generated tokens.", "",
             "| Seed | Output identical to greedy | Accepted/step | Acceptance rate | Tokens/s plain | Tokens/s speculative | Speedup |",
             "|---|:--:|---:|---:|---:|---:|---:|"]

    for seed in args.seeds:
        seed_everything(seed)
        ref = CEDBaseline(len(vocab), **base_cfg["model"])
        ref.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"ced_baseline_seed{seed}.pt",
                                       map_location="cpu"))
        model = SNCED(copy.deepcopy(ref), vocab, note_mode=cfg["note_mode"])
        model.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / f"snced_mixed_seed{seed}.pt",
                                         map_location="cpu"))
        model.eval()
        adapter = Adapter(model)

        rng = random.Random(seed)
        items = []
        for _ in range(args.n_docs):
            lec = make_lecture(rng, base_cfg["data"]["n_entities"], base_cfg["data"]["n_filler"])
            items += [(lec, q) for q in sample_questions(rng, lec, 2)]

        draft = DraftModule(ref.d_model, len(vocab), block=args.block)
        loss = train_draft(adapter, vocab, items, draft, args.draft_steps, args.block)
        print(f"seed={seed} draft trained, final loss {loss:.4f}", flush=True)

        r = benchmark(adapter, vocab, items, draft, args.gen_steps)
        lines.append(f"| {seed} | {'yes' if r['identical_to_greedy'] else 'NO'} "
                     f"| {r['accepted_per_step']:.2f} | {r['acceptance_rate']*100:.1f}% "
                     f"| {r['plain_tokens_per_s']:.0f} | {r['spec_tokens_per_s']:.0f} "
                     f"| {r['speedup']:.2f}x |")
        print(lines[-1], flush=True)
        RunRecord(model="snced+dspark", condition="speculative", dataset="synthetic_lecture_v1",
                  seed=seed, accuracy=1.0 if r["identical_to_greedy"] else 0.0,
                  task_success=1.0 if r["identical_to_greedy"] else 0.0,
                  extra={"draft_block": args.block, "draft_params": sum(p.numel() for p in draft.parameters()),
                         **r}).save(EXPERIMENT)

    lines += ["", "**Correctness is the primary result:** identical output means throughput work",
              "cannot change what the model says.", "",
              "This benchmark generates 1-3 token chains, so there is little to speculate about and",
              "no speedup should be expected here - the draft costs a pass that the short generation",
              "cannot amortise. Acceptance rate is the number that transfers to long generations."]
    (ROOT / "results" / "tables" / "dspark.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[-6:]))


if __name__ == "__main__":
    main()
