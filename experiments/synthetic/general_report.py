"""Report for the general SN-CED architecture (v2).

Reads results/raw/snced_general and writes results/tables/snced_general.md:
the three access paths side by side, the notebook size the L0 gate settled on,
and how often the router asked to reread.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT  # noqa: E402

from snced.metrics import mean_std  # noqa: E402
from snced.runlog import load_runs  # noqa: E402

PATHS = ("detailed", "notes", "routed")
LABEL = {"detailed": "Full context (reference)", "notes": "Semantic Note Memory only",
         "routed": "Router's choice"}


def main() -> None:
    runs = load_runs("snced_general")
    if not runs:
        print("no runs yet")
        return
    by_path: dict[str, list[dict]] = defaultdict(list)
    for r in runs:
        by_path[r["condition"]].append(r)
    seeds = sorted({r["seed"] for r in runs})
    ctx = mean_std([r["context_length"] for r in runs])[0]

    out = [
        "# SN-CED general architecture (v2): results",
        "",
        f"Seeds: {', '.join(map(str, seeds))}. Context ~{ctx:.0f} tokens.",
        "",
        "The compiler writes open-vocabulary notes by pointing into the source, the router is",
        "learned, and **no gold note labels are used anywhere**: the notebook is shaped only by the",
        "task loss, a sufficiency term against the full-context path, an L0 penalty on the number of",
        "notes, and a price on rereading.",
        "",
        "| Path | Accuracy | Memory slots | Notes kept | Router fired |",
        "|---|---:|---:|---:|---:|",
    ]
    for path in PATHS:
        rs = by_path.get(path, [])
        if not rs:
            continue
        acc = mean_std([r["accuracy"] for r in rs])
        slots = mean_std([r["extra"]["memory_slots"] for r in rs])
        kept = mean_std([r["note_count"] for r in rs])
        fired = mean_std([r["fallback_rate"] for r in rs])
        out.append(f"| {LABEL[path]} | {acc[0]*100:.2f} ± {acc[1]*100:.2f}% | {slots[0]:.1f} "
                   f"| {kept[0]:.1f} | {fired[0]*100:.1f}% |")

    notes = by_path.get("notes", [])
    if notes:
        ratio = mean_std([r["extra"]["memory_slots"] for r in notes])[0] / max(ctx, 1)
        out += ["", f"Note memory is {ratio*100:.1f}% of the context length."]
        sample = notes[0]["extra"].get("sample_notes") or []
        if sample:
            out += ["", "Sample of the notebook, as text (the pointers' argmax):", "",
                    "| anchor | micro-context | source | order | gate |", "|---|---|---:|---:|---:|"]
            for n in sample[:6]:
                out.append(f"| {n['anchor']} | {n['micro_context']} | {n['source']} | {n['order']} "
                           f"| {n.get('gate', float('nan')):.2f} |")

    out += ["", "## Caveats", "",
            "- Warm-started from a trained CED backbone with the encoder frozen (plan Phase 2).",
            "  Training all of it from scratch at once left both paths near 30%.",
            "- The router never had to fire: notes were sufficient, so its value is untested here.",
            "- Synthetic benchmark only; the real-text compiler is a separate, weaker result",
            "  (results/tables/span_compiler.md)."]
    path = ROOT / "results" / "tables" / "snced_general.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
