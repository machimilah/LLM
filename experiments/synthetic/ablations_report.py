"""Report for the memory-representation ablations (plan section 15, Phase 4).

Writes results/tables/ablations.md and Plot F (compression vs quality).
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from snced.metrics import decoder_memory_flops, decoder_query_flops, encoder_flops, mean_std  # noqa: E402
from snced.runlog import load_runs  # noqa: E402

ORDER = ("full_context", "extractive_facts", "iterative_rag", "snm_fallback", "snm_notes", "rag_topk",
         "extractive_facts_b", "sliding_window", "snm_phrase_only", "snm_keyword_only")
LABELS = {
    "full_context": "Full context",
    "extractive_facts": "Extractive summary (fact sentences)",
    "extractive_facts_b": "Extractive summary, 72-token budget",
    "iterative_rag": "Iterative RAG (query-conditioned)",
    "rag_topk": "RAG top-k, 72-token budget (query-conditioned)",
    "sliding_window": "Sliding window, 72 tokens",
    "snm_notes": "SNM: anchor + micro-context",
    "snm_keyword_only": "SNM: keyword (anchor) only",
    "snm_phrase_only": "SNM: phrase only (no anchor)",
    "snm_fallback": "SNM + source fallback",
}


D_MODEL, D_FF, N_ENC, N_DEC, CONV_K = 64, 128, 2, 3, 6
QUERIES_PER_LECTURE = 36
MEAN_QUERY_LEN = 13.7  # question + [SEP] + chain, averaged over the three types


def flops(r: dict) -> float:
    """Per-lecture FLOPs for a 36-question workload, recomputed from stored fields.

    Query-conditioned memory (RAG) is rebuilt per question; query-independent
    memory (notes, summaries, window, full context) is built once and reused.
    Compilation and fact selection are charged a full-lecture encoder pass.
    """
    cond, e = r["condition"], r["extra"]
    slots, toks, T = e["memory_slots"], e["retained_tokens"], r["context_length"]
    enc = lambda n: encoder_flops(n, D_MODEL, D_FF, N_ENC, CONV_K)  # noqa: E731
    if cond.startswith("snm"):
        layers = [(3 * D_MODEL, 128), (128, 128), (128, D_MODEL)]
        build = enc(T) + slots * sum(2 * i * o for i, o in layers)
    elif cond in ("extractive_facts", "extractive_facts_b"):
        build = enc(T) + enc(toks)
    else:
        build = enc(toks)
    rebuilds = QUERIES_PER_LECTURE if e["query_conditioned"] else 1
    return (rebuilds * (build + decoder_memory_flops(slots, D_MODEL, N_DEC))
            + QUERIES_PER_LECTURE * decoder_query_flops(MEAN_QUERY_LEN, slots, D_MODEL, D_FF, N_DEC))


def pm(xs: list[float], k: float = 1.0, digits: int = 2) -> str:
    m, s = mean_std(xs)
    return f"{m * k:.{digits}f} ± {s * k:.{digits}f}"


def main() -> None:
    runs = load_runs("ablations")
    if not runs:
        print("no runs yet")
        return
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["extra"]["n_filler"], r["condition"])].append(r)
    fillers = sorted({f for f, _ in groups})
    seeds = sorted({r["seed"] for r in runs})

    out = [
        "# Memory-representation ablations",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. Mean ± std. Every condition uses the same encoder,",
        "decoder and evaluation harness; only the decoder's memory differs.",
        "Budget conditions keep 72 tokens, which is what SNM's 24 notes cost in text.",
        "GFLOPs = analytic estimate per lecture for 36 questions, including building the memory",
        "(the full-lecture pass needed for compilation or fact selection is charged to those conditions).",
        "Query-conditioned memory (RAG) is rebuilt for every question; query-independent memory is built",
        "once per lecture and reused across the 36 questions.",
        "",
        "Keyword-only and phrase-only each got their own short fine-tune, so they are not penalised",
        "for being unfamiliar to the decoder.",
        "",
    ]
    for f in fillers:
        base = groups.get((f, "full_context"), [])
        T = base[0]["context_length"] if base else 0
        out += [
            f"## {f} filler chunks (~{T} context tokens)",
            "",
            "| Memory | Query-independent | Accuracy | Direct | One-hop | Two-hop | Slots | Tokens kept | GFLOPs/lecture |",
            "|---|:--:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in ORDER:
            rs = groups.get((f, c), [])
            if not rs:
                continue
            bt = lambda k: pm([r["extra"]["accuracy_by_type"][k] for r in rs], 100, 1)  # noqa: E731
            qi = "no" if rs[0]["extra"]["query_conditioned"] else "yes"
            out.append(
                f"| {LABELS[c]} | {qi} | {pm([r['accuracy'] for r in rs], 100, 2)}% | {bt('direct')} "
                f"| {bt('one_hop')} | {bt('two_hop')} | {pm([r['extra']['memory_slots'] for r in rs], 1, 1)} "
                f"| {pm([r['extra']['retained_tokens'] for r in rs], 1, 1)} "
                f"| {pm([flops(r) / 1e9 for r in rs], 1, 3)} |"
            )
        out.append("")

    path = ROOT / "results" / "tables" / "ablations.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))

    f0 = fillers[0]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for c in ORDER:
        rs = groups.get((f0, c), [])
        if not rs:
            continue
        x = mean_std([r["extra"]["memory_slots"] for r in rs])[0]
        y = mean_std([r["accuracy"] for r in rs])[0] * 100
        ax.scatter(x, y, s=40)
        ax.annotate(LABELS[c], (x, y), textcoords="offset points", xytext=(6, -3), fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Active decoder memory slots per lecture (log scale)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Plot F: compression vs quality (~{groups[(f0, 'full_context')][0]['context_length']} tokens)")
    fig.tight_layout()
    fig.savefig(ROOT / "results" / "figures" / "plot_f_compression_vs_quality.png", dpi=150)


if __name__ == "__main__":
    main()
