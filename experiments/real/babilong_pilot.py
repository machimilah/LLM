"""Real-model pilot: SNM as an external runtime on BABILong (plan sections 16, 29, P7).

A pretrained decoder-only LLM has no encoder to hang note slots on, so SNM is
applied as a runtime: the model reads each document ONCE, writes notes before
seeing the question, and later answers from notes instead of the full text.

Conditions, all answered by the same model with the same answer prompt:
  full_context   the whole document in the prompt
  rag_topk       sentences ranked by overlap with the question, to the budget
  summary        an LLM summary of the document, query-independent
  snm_notes      LLM-written "anchor -> micro-context" notes, query-independent
  snm_trained    notes from the small trained compiler (train_note_compiler.py)

Both memory-writing prompts were given the same task-aware hint (that the text
hides short statements about people and places); an earlier generic prompt made
the 0.5B model summarise the literary filler instead. That tuning is applied
equally to the summary and notes arms and is a deviation from zero-shot use.

RAG is query-conditioned; summary and notes are written before the question and
are reusable. Memory arms are capped at the same token budget.

Usage: python experiments/real/babilong_pilot.py --tasks qa1 qa2 --length 1k --limit 25
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT  # noqa: E402

import torch  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from snced.runlog import RunRecord, seed_everything  # noqa: E402

EXPERIMENT = "babilong_pilot"
CONDITIONS = ("full_context", "rag_topk", "summary", "snm_notes", "snm_trained")
STOP = set("what where who is are was were the a an of to in on at and or do does did how many "
           "before after then than that this it he she they them his her".split())

NOTES_PROMPT = (
    "The text below mixes a story with short simple statements about people moving between "
    "places (for example: 'Mary went to the kitchen.'). Find ONLY those short statements about "
    "people and places. Write each one as a note `person -> place`, in the order they appear, "
    "keeping the LAST place for each person. Ignore everything else. Notes only.\n\n{text}"
)
SUMMARY_PROMPT = (
    "The text below mixes a story with short simple statements about people moving between "
    "places (for example: 'Mary went to the kitchen.'). Write a short summary that preserves "
    "those statements - who went where, and where each person ends up - and ignores the rest. "
    "At most 120 words, nothing else.\n\n{text}"
)

ANSWER_PROMPT = "{memory}\n\nQuestion: {question}\nAnswer with a single word."


def sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]


def keywords(q: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", q.lower()) if w not in STOP}


class Runner:
    def __init__(self, model_id: str, threads: int, dtype: str = "float32") -> None:
        torch.set_num_threads(threads)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).eval()
        cfg = self.model.config
        head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        # fp32 K and V for every layer: the memory a cached prompt would occupy.
        self.kv_bytes_per_token = 2 * cfg.num_hidden_layers * cfg.num_key_value_heads * head_dim * 4

    def n_tokens(self, text: str) -> int:
        return len(self.tok(text)["input_ids"])

    def truncate(self, text: str, budget: int) -> str:
        ids = self.tok(text)["input_ids"]
        return text if len(ids) <= budget else self.tok.decode(ids[:budget], skip_special_tokens=True)

    @torch.no_grad()
    def generate(self, prompt: str, max_new_tokens: int) -> tuple[str, float, int]:
        msg = [{"role": "user", "content": prompt}]
        enc = self.tok.apply_chat_template(msg, add_generation_prompt=True, return_tensors="pt")
        ids = enc if isinstance(enc, torch.Tensor) else enc["input_ids"]
        t0 = time.perf_counter()
        out = self.model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new_tokens,
                                  do_sample=False, pad_token_id=self.tok.eos_token_id)
        dt = time.perf_counter() - t0
        text = self.tok.decode(out[0, ids.size(1):], skip_special_tokens=True).strip()
        return text, dt, int(ids.size(1))


def build_memory(runner: Runner, cond: str, doc: str, question: str, budget: int, max_new: int):
    """Returns (memory text, build seconds, tokens generated while building)."""
    if cond == "full_context":
        return doc, 0.0, 0
    if cond == "rag_topk":
        kw = keywords(question)
        ranked = sorted(sentences(doc), key=lambda s: -len(keywords(s) & kw))
        picked, used = [], 0
        for s in ranked:
            n = runner.n_tokens(s)
            if used + n > budget:
                break
            picked.append(s)
            used += n
        return " ".join(picked), 0.0, 0
    if cond == "snm_trained":
        t0 = time.perf_counter()
        notes = compile_with_trained(doc)
        text = "\n".join(n.text for n in notes)
        return runner.truncate(text, budget), time.perf_counter() - t0, runner.n_tokens(text)
    prompt = (NOTES_PROMPT if cond == "snm_notes" else SUMMARY_PROMPT).format(text=doc)
    text, dt, _ = runner.generate(prompt, max_new)
    return runner.truncate(text, budget), dt, runner.n_tokens(text)


_COMPILER = None


def compile_with_trained(doc: str):
    """Lazy-load the trained note compiler (frozen MiniLM + small heads)."""
    global _COMPILER
    if _COMPILER is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from train_note_compiler import NoteCompilerHeads, SentenceEncoder

        enc = SentenceEncoder(threads=torch.get_num_threads())
        heads = NoteCompilerHeads(enc.dim)
        heads.load_state_dict(torch.load(ROOT / "results" / "checkpoints" / "note_compiler_real_seed0.pt"))
        heads.eval()
        _COMPILER = (enc, heads)
    from notes_data import split_sentences
    from train_note_compiler import compile_notes

    enc, heads = _COMPILER
    return compile_notes(enc, heads, split_sentences(doc))


def score(pred: str, target: str) -> bool:
    return target.lower() in re.findall(r"[a-z]+", pred.lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--tasks", nargs="+", default=["qa1", "qa2"])
    ap.add_argument("--length", default="1k")
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--budget", type=int, default=256, help="token cap for the memory arms")
    ap.add_argument("--max-new", type=int, default=128, help="generation cap when writing notes/summary")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=list(CONDITIONS),
                    help="subset of conditions, e.g. --conditions full_context for a baseline gate")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    args = ap.parse_args()

    seed_everything(args.seed)
    runner = Runner(args.model, args.threads, args.dtype)
    log_path = ROOT / "results" / "logs" / f"{EXPERIMENT}_{args.length}_{args.seed}.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("a")

    for task in args.tasks:
        ds = load_dataset("RMT-team/babilong", args.length, split=task)
        stats = {c: {"correct": 0, "n": 0, "prompt_tokens": 0, "answer_s": 0.0, "build_s": 0.0,
                     "built_tokens": 0} for c in args.conditions}
        for i in range(min(args.limit, len(ds))):
            row = ds[i]
            doc, question, target = row["input"], row["question"].strip(), row["target"].strip()
            for cond in args.conditions:
                mem, build_s, built = build_memory(runner, cond, doc, question, args.budget, args.max_new)
                prompt = ANSWER_PROMPT.format(memory=mem, question=question)
                pred, dt, n_prompt = runner.generate(prompt, 8)
                ok = score(pred, target)
                s = stats[cond]
                s["correct"] += ok
                s["n"] += 1
                s["prompt_tokens"] += n_prompt
                s["answer_s"] += dt
                s["build_s"] += build_s
                s["built_tokens"] += built
                log.write(json.dumps({"task": task, "i": i, "condition": cond, "correct": bool(ok),
                                      "pred": pred[:80], "target": target, "prompt_tokens": n_prompt,
                                      "build_s": round(build_s, 2), "answer_s": round(dt, 2)}) + "\n")
                log.flush()
            print(f"{task} {i+1}/{args.limit} " + " ".join(
                f"{c}={stats[c]['correct']}/{stats[c]['n']}" for c in args.conditions), flush=True)

        for cond in args.conditions:
            s = stats[cond]
            rec = RunRecord(
                model=args.model,
                condition=cond,
                dataset=f"babilong_{args.length}_{task}",
                seed=args.seed,
                context_length=s["prompt_tokens"] // s["n"],
                parameters=sum(p.numel() for p in runner.model.parameters()),
                accuracy=s["correct"] / s["n"],
                task_success=s["correct"] / s["n"],
                kv_cache_bytes=runner.kv_bytes_per_token * (s["prompt_tokens"] // s["n"]),
                note_tokens=s["built_tokens"] / s["n"],
                note_compile_ms=s["build_s"] / s["n"] * 1000,
                decode_ms=s["answer_s"] / s["n"] * 1000,
                total_task_ms=(s["build_s"] + s["answer_s"]) / s["n"] * 1000,
                extra={
                    "task": task,
                    "length": args.length,
                    "n": s["n"],
                    "budget_tokens": args.budget,
                    "query_conditioned": cond == "rag_topk",
                    "reusable_memory": cond in ("summary", "snm_notes"),
                    "mean_prompt_tokens": s["prompt_tokens"] / s["n"],
                },
            )
            rec.save(EXPERIMENT)
            print(f"  {task} {cond:13s} acc={rec.accuracy:.3f} prompt_tokens={rec.extra['mean_prompt_tokens']:.0f} "
                  f"build_ms={rec.note_compile_ms:.0f} answer_ms={rec.decode_ms:.0f}", flush=True)
    log.close()


if __name__ == "__main__":
    main()
