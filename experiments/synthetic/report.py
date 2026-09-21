"""Reproducibility report for the historical experiments (plan P1).

Aggregates every raw run in results/raw/{ablation,learned_compiler}, compares
against the numbers recorded in the plan, and checks the Phase 0 acceptance
criteria. Writes results/tables/reproduction.md.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import ROOT  # noqa: E402

from snced.memory import CONDITIONS  # noqa: E402
from snced.metrics import fmt, mean_std  # noqa: E402
from snced.runlog import load_runs  # noqa: E402

# Three-seed means recorded in the plan (sections 11.1, 11.2, Appendix A).
HIST_A = {
    "iterative_rag": (1.0000, 43.48),
    "snm": (1.0000, 24.41),
    "snm_noisy": (0.8222, 23.99),
    "snm_fallback": (1.0000, 27.70),
    "word_only": (0.1215, 16.36),
}
HIST_B = {
    "raw_tokens": 573.44,
    "note_tokens": 71.98,
    "raw_slots": 42.00,
    "note_slots": 23.99,
    "text_token_reduction": 0.8745,
    "slot_reduction": 0.4287,
    "direct": 1.0000,
    "one_hop": 0.9979,
    "two_hop": 1.0000,
    "overall": 0.9993,
}


def ablation_section(runs: list[dict]) -> tuple[list[str], dict]:
    lines = [
        "## Experiment A: representation and fallback ablation",
        "",
        "| Condition | Seeds | Accuracy (new) | Accuracy (plan) | Active tokens/q (new) | Active tokens/q (plan) | Fallback rate | Exposure, 100 q |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    means = {}
    for cond in CONDITIONS:
        rs = [r for r in runs if r["condition"] == cond]
        if not rs:
            continue
        acc = [r["accuracy"] for r in rs]
        act = [r["extra"]["mean_active_tokens"] for r in rs]
        exp = [r["extra"]["token_exposure_100q"] for r in rs]
        fb = [r["fallback_rate"] for r in rs]
        means[cond] = (mean_std(acc)[0], mean_std(act)[0])
        h_acc, h_act = HIST_A[cond]
        lines.append(
            f"| {cond} | {', '.join(str(r['seed']) for r in rs)} | {fmt(acc, pct=True)} | {h_acc * 100:.2f}% "
            f"| {fmt(act)} | {h_act:.2f} | {fmt(fb, pct=True)} | {fmt(exp, digits=0)} |"
        )
    if "iterative_rag" in means and "snm" in means:
        red = 1 - means["snm"][1] / means["iterative_rag"][1]
        lines += ["", f"SNM uses {red * 100:.1f}% fewer active tokens than iterative RAG (plan: 43.9%)."]
    return lines, means


def compiler_section(runs: list[dict]) -> tuple[list[str], dict]:
    get = {
        "raw_tokens": lambda r: r["extra"]["raw_tokens"],
        "note_tokens": lambda r: r["note_tokens"],
        "raw_slots": lambda r: r["extra"]["raw_slots"],
        "note_slots": lambda r: r["note_count"],
        "text_token_reduction": lambda r: r["extra"]["text_token_reduction"],
        "slot_reduction": lambda r: r["extra"]["slot_reduction"],
        "direct": lambda r: r["extra"]["recovery_by_type"]["direct"],
        "one_hop": lambda r: r["extra"]["recovery_by_type"]["one_hop"],
        "two_hop": lambda r: r["extra"]["recovery_by_type"]["two_hop"],
        "overall": lambda r: r["accuracy"],
    }
    pct = {"text_token_reduction", "slot_reduction", "direct", "one_hop", "two_hop", "overall"}
    lines = [
        "## Experiment B: learned query-independent note compiler",
        "",
        f"Seeds: {', '.join(str(r['seed']) for r in runs)}. "
        f"Compiler parameters: {runs[0]['parameters']:,} (plan: 22,923). "
        "Reader: deterministic graph traversal over predicted notes, not an LLM decoder.",
        "",
        "| Metric | New (mean ± std) | Plan |",
        "|---|---:|---:|",
    ]
    means = {}
    for key, f in get.items():
        xs = [f(r) for r in runs]
        means[key] = mean_std(xs)[0]
        p = key in pct
        lines.append(f"| {key} | {fmt(xs, pct=p)} | {HIST_B[key] * (100 if p else 1):.2f}{'%' if p else ''} |")
    exact = ", ".join(f"seed {r['seed']}: {r['extra']['exact_chunk_extraction'] * 100:.2f}%" for r in runs)
    lines += ["", f"Exact chunk extraction: {exact}."]
    return lines, means


def main() -> None:
    a_runs, b_runs = load_runs("ablation"), load_runs("learned_compiler")
    out = [
        "# Reproduction report: historical SNM experiments",
        "",
        "The original prototype scripts were not available, so both experiments were",
        "reimplemented from the plan's description (src/snced/data.py documents the",
        "reconstruction). Exact token counts depend on template wording and are not",
        "expected to match; the acceptance criteria concern the qualitative pattern.",
        "",
        "Caveat: the reconstruction appears slightly easier than the original. The",
        "compiler reaches 100% extraction on every seed (original seed 33: 99.94%),",
        "and notes/questions are shorter, so the SNM-vs-RAG active-token gap is larger",
        "than recorded in the plan. Do not read the larger gap as an improvement.",
        "",
    ]
    a_lines, a = ablation_section(a_runs) if a_runs else ([], {})
    b_lines, b = compiler_section(b_runs) if b_runs else ([], {})
    out += a_lines + [""] + b_lines + [""]

    checks = [
        ("A: SNM accuracy near 100%", a.get("snm", (0,))[0] >= 0.98),
        ("A: word-only accuracy near chance (12.5%)", abs(a.get("word_only", (1,))[0] - 0.125) <= 0.04),
        ("B: textual compression near 87%", abs(b.get("text_token_reduction", 0) - 0.8745) <= 0.03),
        ("B: information recovery near 99.9%", b.get("overall", 0) >= 0.995),
    ]
    out += ["## Phase 0 acceptance criteria", "", "| Criterion | Result |", "|---|---|"]
    out += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks]
    path = ROOT / "results" / "tables" / "reproduction.md"
    path.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))


if __name__ == "__main__":
    main()
