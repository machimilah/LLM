"""Report for the CED vs SN-CED comparison (plan sections 13, 14, 18).

Reads results/raw/snced_compare, aggregates across seeds and writes
results/tables/snced_vs_ced.md plus Plot A (quality vs active memory) and
Plot C (quality vs context length) to results/figures/.
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

from snced.metrics import mean_std  # noqa: E402
from snced.runlog import load_runs  # noqa: E402

CONDITIONS = ("ced_baseline", "snced_detailed", "snced_notes", "snced_gold_notes", "snced_fallback")
LABELS = {
    "ced_baseline": "CED baseline (full memory)",
    "snced_detailed": "SN-CED, full memory",
    "snced_notes": "SN-CED, notes only",
    "snced_gold_notes": "SN-CED, gold notes (upper bound)",
    "snced_fallback": "SN-CED, notes + fallback",
}


def pm(xs: list[float], k: float = 1.0, digits: int = 2) -> str:
    m, s = mean_std(xs)
    return f"{m * k:.{digits}f} ± {s * k:.{digits}f}"


def main() -> None:
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="snced_compare")
    ap.add_argument("--out", default="snced_vs_ced")
    args = ap.parse_args()
    runs = load_runs(args.experiment)
    if not runs:
        print("no runs yet")
        return
    groups: dict[tuple[int, str], list[dict]] = defaultdict(list)
    for r in runs:
        groups[(r["extra"]["n_filler"], r["condition"])].append(r)
    fillers = sorted({f for f, _ in groups})
    seeds = sorted({r["seed"] for r in runs})

    out = [
        "# CED vs SN-CED: controlled comparison (synthetic lectures)",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. Values are mean ± std across seeds.",
        "Trained at 18 filler chunks; 54 and 108 are longer, unseen contexts.",
        "Active memory = decoder cross-attention slots per lecture. KV = fp32 K and V for all decoder layers.",
        "GFLOPs = analytic estimate per lecture for 36 questions, including the full-lecture encoder pass",
        "for every condition and note re-encoding for note conditions (Rules 4 and 5).",
        "",
        "Caveats:",
        "- The note compiler is trained with gold chunk labels, supervision the baseline does not get.",
        "- Wall-clock timings were measured on a shared CPU with other runs in flight and include one",
        "  outlier of ~23.7 s/question (SN-CED full memory, 939 tokens, seed 1). Use the FLOPs column;",
        "  treat decode-ms means, especially their standard deviations, as unreliable.",
        "- Notes-only degrades at unseen context lengths while gold notes do not: the compiler, not the",
        "  note representation, is what fails to generalize.",
        "",
    ]
    for f in fillers:
        base = groups.get((f, "ced_baseline"), [])
        base_slots = mean_std([r["extra"]["memory_slots"] for r in base])[0] if base else float("nan")
        T = base[0]["context_length"] if base else 0
        out += [
            f"## {f} filler chunks (~{T} context tokens)",
            "",
            "| Condition | Accuracy | Direct | One-hop | Two-hop | Active memory | vs baseline | KV bytes | GFLOPs/lecture | Decode ms/q | Fallback |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for c in CONDITIONS:
            rs = groups.get((f, c), [])
            if not rs:
                continue
            slots = [r["extra"]["memory_slots"] for r in rs]
            bt = lambda k: pm([r["extra"]["accuracy_by_type"][k] for r in rs], 100, 1)  # noqa: E731
            out.append(
                f"| {LABELS[c]} | {pm([r['accuracy'] for r in rs], 100, 2)}% | {bt('direct')} | {bt('one_hop')} "
                f"| {bt('two_hop')} | {pm(slots, 1, 1)} | {mean_std(slots)[0] / base_slots * 100:.1f}% "
                f"| {mean_std([r['kv_cache_bytes'] for r in rs])[0]:,.0f} "
                f"| {pm([r['estimated_flops'] / 1e9 for r in rs], 1, 3)} "
                f"| {pm([r['decode_ms'] for r in rs], 1, 2)} | {pm([r['fallback_rate'] for r in rs], 100, 1)}% |"
            )
        ex = [r["extra"]["compiler_exact_chunk_extraction"] for r in groups.get((f, "snced_notes"), [])]
        if ex:
            out += ["", f"Compiler exact chunk extraction: {pm(ex, 100, 2)}%."]
        out.append("")

    tables = ROOT / "results" / "tables"
    tables.mkdir(parents=True, exist_ok=True)
    (tables / f"{args.out}.md").write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))

    figs = ROOT / "results" / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    shown = ("ced_baseline", "snced_notes", "snced_fallback")

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in shown:
        xs = [mean_std([r["extra"]["memory_slots"] for r in groups[(f, c)]])[0] for f in fillers if (f, c) in groups]
        ys = [mean_std([r["accuracy"] for r in groups[(f, c)]])[0] * 100 for f in fillers if (f, c) in groups]
        ax.plot(xs, ys, marker="o", label=LABELS[c])
        for x, y, f in zip(xs, ys, fillers):
            ax.annotate(f"{f}", (x, y), textcoords="offset points", xytext=(4, 4), fontsize=7)
    ax.set_xlabel("Active decoder memory (slots per lecture)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Plot A: quality vs active memory (labels = filler chunks)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / f"plot_a_quality_vs_memory_{args.out}.png", dpi=150)

    fig, ax = plt.subplots(figsize=(6, 4))
    for c in CONDITIONS:
        pts = [(groups[(f, c)][0]["context_length"], mean_std([r["accuracy"] for r in groups[(f, c)]])[0] * 100)
               for f in fillers if (f, c) in groups]
        if pts:
            ax.plot(*zip(*pts), marker="o", label=LABELS[c])
    ax.set_xlabel("Context length (tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Plot C: quality vs context length")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(figs / f"plot_c_quality_vs_context_{args.out}.png", dpi=150)


if __name__ == "__main__":
    main()
