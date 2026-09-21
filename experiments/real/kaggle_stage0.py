"""Stage 0 + 0b on a free GPU (Kaggle / Colab): the two gates that decide everything.

Stage 0   train the open-vocabulary note compiler on real text until it keeps
          >= 95% of required facts on documents with unseen entities
Stage 0b  with a competent 7B model in 4-bit, compare answering from notes
          against full context, RAG and summaries on BABILong

Both fit a single 16 GB T4. Every long step checkpoints, so a killed session
resumes instead of restarting (`--resume`), and the run stops itself before the
platform's time limit (`--session-hours`).

    python experiments/real/kaggle_stage0.py --stage compiler --session-hours 8
    python experiments/real/kaggle_stage0.py --stage qa --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import ROOT  # noqa: E402

import torch  # noqa: E402
from torch import nn  # noqa: E402

from snced.runlog import RunRecord, pick_device, seed_everything  # noqa: E402
from snced.training import SessionBudget, TrainState, load_resumable, save_resumable, write_manifest  # noqa: E402

CKPT_DIR = ROOT / "results" / "checkpoints"
GATE_RETENTION = 0.95
GATE_BASELINE_QA = 0.70


# ---------------------------------------------------------------- stage 0
def run_compiler_stage(args) -> None:
    """Train the span compiler with the encoder unfrozen, checkpointing as it goes."""
    from span_compiler import SpanHeads, TokenEncoder, evaluate, make_labels  # noqa: E402
    from span_data import build_span_document, load_noise, mined_vocab  # noqa: E402

    from notes_data import load_stories  # noqa: E402

    device = pick_device(args.device)
    seed_everything(args.seed)
    rng = random.Random(args.seed)
    budget = SessionBudget(args.session_hours)

    enc = TokenEncoder(threads=args.threads)
    enc.model.to(device)
    if args.finetune_encoder:
        enc.unfreeze()
    heads = SpanHeads(enc.dim).to(device)
    params = list(heads.parameters()) + (list(enc.model.parameters()) if args.finetune_encoder else [])
    opt = torch.optim.AdamW([
        {"params": list(heads.parameters()), "lr": args.lr},
        *([{"params": list(enc.model.parameters()), "lr": args.encoder_lr}] if args.finetune_encoder else []),
    ])

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"kaggle_compiler_seed{args.seed}.pt"
    state = TrainState()
    if args.resume and ckpt.exists():
        state = load_resumable(ckpt, heads, opt, rng)
        print(f"resumed at step {state.step} (best retention {state.best_metric:.4f})", flush=True)

    print("building training documents...", flush=True)
    stories, noise = load_stories((1, 2, 9)), load_noise()
    vocab = mined_vocab(noise)
    sentences, labels = [], []
    for _ in range(args.train_docs):
        s, l, _ = build_span_document(rng, rng.choice(stories), noise, args.target_words,
                                      names=vocab["train_names"], nouns=vocab["train_nouns"])
        sentences += s
        labels += l
    print(f"  {len(sentences)} sentences, {sum(1 for x in labels if x)} facts", flush=True)

    ce = nn.functional.cross_entropy
    order = list(range(0, len(sentences), args.batch))
    while state.step < args.max_steps and not budget.expired:
        rng.shuffle(order)
        for start in order:
            if state.step >= args.max_steps or budget.expired:
                break
            state.step += 1
            heads.train()
            chunk, lab = sentences[start : start + args.batch], labels[start : start + args.batch]
            states, e = enc.encode(chunk)
            y = make_labels(e, chunk, lab)
            out = heads(states.to(device), e["attention_mask"].to(device))
            loss = ce(out["tags"].flatten(0, 1), y["tags"].flatten().to(device), ignore_index=-100)
            loss = loss + ce(out["is_fact"], y["is_fact"].to(device))
            m = y["is_fact"] == 1
            if m.any():
                loss = loss + ce(out["negated"][m.to(device)], y["negated"][m].to(device))
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

            if state.step % args.eval_every == 0:
                res = evaluate(enc, heads, args.eval_limit, rename_seed=1234, vocab=vocab)
                worst = min(r["retention"] for r in res.values())
                state.history.append({"step": state.step, "loss": float(loss),
                                      **{t: r["retention"] for t, r in res.items()}})
                print(f"step={state.step} loss={float(loss):.4f} "
                      + " ".join(f"{t}={r['retention']:.3f}" for t, r in res.items())
                      + f" | worst={worst:.3f} | {budget.remaining_min:.0f} min left", flush=True)
                if worst > state.best_metric:
                    state.best_metric = worst
                save_resumable(ckpt, heads, opt, state, rng)
                if worst >= GATE_RETENTION:
                    print(f"GATE PASSED: retention {worst:.3f} >= {GATE_RETENTION}", flush=True)
                    break

    save_resumable(ckpt, heads, opt, state, rng)
    write_manifest(ckpt.with_suffix(".json"), stage="compiler", step=state.step,
                   best_retention=state.best_metric, gate=GATE_RETENTION,
                   passed=state.best_metric >= GATE_RETENTION, history=state.history)
    print(f"stopped at step {state.step}; best retention {state.best_metric:.4f} "
          f"({'PASS' if state.best_metric >= GATE_RETENTION else 'below gate'})", flush=True)


# ---------------------------------------------------------------- stage 0b
def run_qa_stage(args) -> None:
    """Baseline gate first, then the five-arm comparison, on a quantised 7B."""
    from babilong_pilot import ANSWER_PROMPT, CONDITIONS, Runner, build_memory, score  # noqa: E402
    from datasets import load_dataset  # noqa: E402

    from snced.training import load_quantized_causal_lm  # noqa: E402

    budget = SessionBudget(args.session_hours)
    seed_everything(args.seed)
    model, tok = load_quantized_causal_lm(args.model, bits=args.bits)
    runner = Runner.__new__(Runner)  # reuse the pilot's helpers with a quantised model
    runner.tok, runner.model = tok, model
    cfg = model.config
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    runner.kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 2

    out_path = ROOT / "results" / "logs" / f"kaggle_qa_{args.length}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)  # a fresh checkout has no results/ tree
    done = set()
    if args.resume and out_path.exists():
        for line in out_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done.add((r["task"], r["i"], r["condition"]))
        print(f"resuming: {len(done)} answers already recorded", flush=True)
    log = out_path.open("a", encoding="utf-8")

    conditions = ["full_context"] if args.gate_only else list(args.conditions or CONDITIONS)
    for task in args.tasks:
        ds = load_dataset("RMT-team/babilong", args.length, split=task)
        stats = {c: [0, 0] for c in conditions}
        for i in range(min(args.limit, len(ds))):
            if budget.expired:
                print("session budget reached; rerun with --resume", flush=True)
                break
            row = ds[i]
            doc, question, target = row["input"], row["question"].strip(), row["target"].strip()
            for cond in conditions:
                if (task, i, cond) in done:
                    stats[cond][1] += 1
                    continue
                t0 = time.perf_counter()
                mem, build_s, built = build_memory(runner, cond, doc, question, args.budget, args.max_new)
                pred, dt, n_prompt = runner.generate(ANSWER_PROMPT.format(memory=mem, question=question), 8)
                ok = score(pred, target)
                stats[cond][0] += ok
                stats[cond][1] += 1
                log.write(json.dumps({"task": task, "i": i, "condition": cond, "correct": bool(ok),
                                      "pred": pred[:60], "target": target, "prompt_tokens": n_prompt,
                                      "build_s": round(build_s, 2), "answer_s": round(dt, 2),
                                      "total_s": round(time.perf_counter() - t0, 2)}) + "\n")
                log.flush()
            if (i + 1) % 10 == 0:
                print(f"{task} {i+1}: " + " ".join(f"{c}={stats[c][0]}/{stats[c][1]}" for c in conditions)
                      + f" | {budget.remaining_min:.0f} min left", flush=True)

        for cond in conditions:
            correct, n = stats[cond]
            if not n:
                continue
            RunRecord(model=args.model, condition=cond, dataset=f"babilong_{args.length}_{task}",
                      seed=args.seed, accuracy=correct / n, task_success=correct / n,
                      extra={"task": task, "n": n, "bits": args.bits,
                             "gate": GATE_BASELINE_QA if cond == "full_context" else None,
                             "passed": (correct / n) >= GATE_BASELINE_QA if cond == "full_context" else None},
                      ).save("kaggle_qa")
            print(f"{task} {cond}: {correct}/{n} = {correct / n:.3f}", flush=True)
            if cond == "full_context" and correct / n < GATE_BASELINE_QA:
                print(f"BASELINE GATE FAILED ({correct / n:.3f} < {GATE_BASELINE_QA}): "
                      "a memory comparison under this model would be meaningless (Rule 2).", flush=True)
    log.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["compiler", "qa"], required=True)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--session-hours", type=float, default=8.5)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    # stage 0
    ap.add_argument("--train-docs", type=int, default=8000)
    ap.add_argument("--target-words", type=int, default=400)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--encoder-lr", type=float, default=2e-5)
    ap.add_argument("--finetune-encoder", action="store_true", default=True)
    ap.add_argument("--max-steps", type=int, default=20000)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--eval-limit", type=int, default=100)
    # stage 0b
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--bits", type=int, default=4, choices=[4, 8])
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2", "qa9"])
    ap.add_argument("--length", default="1k")
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--budget", type=int, default=256)
    ap.add_argument("--max-new", type=int, default=128)
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--gate-only", action="store_true",
                    help="run only full_context, to check the baseline gate before spending time")
    args = ap.parse_args()
    (run_compiler_stage if args.stage == "compiler" else run_qa_stage)(args)


if __name__ == "__main__":
    main()
