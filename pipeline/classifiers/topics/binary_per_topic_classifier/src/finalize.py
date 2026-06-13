"""
Finalize: produce results/appendix7_reproduction.md (the table) +
results/FINAL_REPORT.md.

Run: python -m src.finalize
"""

from __future__ import annotations

import json
from pathlib import Path

from .registry import all_entries

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TARGETS_PATH = PROJECT_ROOT / "experiments" / "appendix7_targets.json"

FAMILIES = ["catboost", "svc", "complement_nb", "nb", "logreg"]
FAMILY_HUMAN = {"catboost": "CatBoost", "svc": "SVC",
                "complement_nb": "Compl. NB", "nb": "NB", "logreg": "Лог. регрессия"}


def _best_for_family_topic(entries):
    """family -> topic -> {precision, recall, f1, exp_id}"""
    out = {}
    for e in entries:
        if e.get("flagged"):
            continue
        fam = e.get("family")
        if not fam or fam == "mixture":
            continue
        for topic, m in (e.get("test_per_topic") or {}).items():
            cur = out.setdefault(fam, {}).get(topic)
            if cur is None or m["f1"] > cur["f1"]:
                out.setdefault(fam, {})[topic] = {**m, "exp_id": e["exp_id"]}
    return out


def _wavg_for_family(entries, family):
    best = max(
        (e for e in entries
         if not e.get("flagged") and e.get("family") == family
         and e.get("test_weighted_avg_f1") is not None),
        key=lambda e: e["test_weighted_avg_f1"],
        default=None,
    )
    if not best:
        return None
    return {
        "f1": best["test_weighted_avg_f1"],
        "precision": best.get("test_weighted_avg_precision"),
        "recall": best.get("test_weighted_avg_recall"),
        "exp_id": best["exp_id"],
    }


def main():
    entries = all_entries()
    targets = json.loads(TARGETS_PATH.read_text())
    topics = targets["topics_order"]
    targets_pt = targets["per_topic"]
    targets_wa = targets["weighted_avg"]

    bests = _best_for_family_topic(entries)

    # ---- Build appendix7_reproduction.md ----
    lines = []
    lines.append("# Appendix 7 reproduction\n")
    lines.append("Per-topic binary classifier metrics. Cell format: **our P/R/F1** | authors' P/R/F1.\n")

    header = "| Тема |"
    for f in FAMILIES:
        header += f" {FAMILY_HUMAN[f]} P/R/F1 (ours \\| theirs) |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(FAMILIES)))

    for topic in topics:
        row = f"| **{topic}** |"
        for f in FAMILIES:
            our = bests.get(f, {}).get(topic)
            their = targets_pt.get(topic, {}).get(f)
            if our is None and their is None:
                row += " — |"
            elif our is None:
                row += f" — \\| {their['precision']:.2f}/{their['recall']:.2f}/{their['f1']:.2f} |"
            elif their is None:
                row += f" {our['precision']:.2f}/{our['recall']:.2f}/**{our['f1']:.2f}** \\| — |"
            else:
                row += (f" {our['precision']:.2f}/{our['recall']:.2f}/**{our['f1']:.2f}** "
                        f"\\| {their['precision']:.2f}/{their['recall']:.2f}/{their['f1']:.2f} |")
        lines.append(row)

    # Weighted avg row
    wavg_row = "| **Weighted avg** |"
    for f in FAMILIES:
        our = _wavg_for_family(entries, f)
        their = targets_wa.get(f)
        if our is None and their is None:
            wavg_row += " — |"
        elif our is None:
            wavg_row += f" — \\| {their['precision']:.2f}/{their['recall']:.2f}/{their['f1']:.2f} |"
        elif their is None:
            wavg_row += f" {our['precision']:.2f}/{our['recall']:.2f}/**{our['f1']:.2f}** \\| — |"
        else:
            wavg_row += (f" {our['precision']:.2f}/{our['recall']:.2f}/**{our['f1']:.2f}** "
                         f"\\| {their['precision']:.2f}/{their['recall']:.2f}/{their['f1']:.2f} |")
    lines.append(wavg_row)

    lines.append("")
    lines.append("## Δ vs authors (F1 only)")
    lines.append("")
    delta_header = "| Тема |"
    for f in FAMILIES:
        delta_header += f" {FAMILY_HUMAN[f]} ΔF1 |"
    lines.append(delta_header)
    lines.append("|" + "---|" * (1 + len(FAMILIES)))

    for topic in topics:
        row = f"| {topic} |"
        for f in FAMILIES:
            our = bests.get(f, {}).get(topic)
            their = targets_pt.get(topic, {}).get(f)
            if our is None or their is None:
                row += " — |"
            else:
                d = our["f1"] - their["f1"]
                arrow = "↑" if d >= 0 else "↓"
                row += f" {arrow}{abs(d):.3f} |"
        lines.append(row)

    (RESULTS_DIR / "appendix7_reproduction.md").write_text("\n".join(lines))

    # ---- FINAL_REPORT.md ----
    final_lines = [
        "# Final report — Appendix 7 reproduction",
        "",
        "## Goal achievement (weighted avg, test)",
        "",
        "| Family | Authors' F1 | Our F1 | Δ | Our exp_id |",
        "|---|---|---|---|---|",
    ]
    for f in FAMILIES:
        our = _wavg_for_family(entries, f)
        their_f1 = targets_wa.get(f, {}).get("f1")
        if our is None:
            final_lines.append(f"| {FAMILY_HUMAN[f]} | {their_f1:.2f} | — | — | — |")
        else:
            d = our["f1"] - their_f1
            final_lines.append(f"| {FAMILY_HUMAN[f]} | {their_f1:.2f} | {our['f1']:.4f} | {d:+.4f} | {our['exp_id']} |")

    # mixture
    mix = next((e for e in entries if e.get("family") == "mixture" and not e.get("flagged")
                and e.get("test_weighted_avg_f1") is not None), None)
    if mix:
        best_authors = max(t["f1"] for t in targets_wa.values())
        final_lines.extend([
            "",
            "## Mixture-of-experts (best per-topic)",
            "",
            f"- Mixture wavg F1: **{mix['test_weighted_avg_f1']:.4f}**",
            f"- Vs authors' best single (SVC=0.76): {mix['test_weighted_avg_f1'] - best_authors:+.4f}",
            f"- Family per topic: {mix.get('family_per_topic', {})}",
        ])

    final_lines.append("")
    final_lines.append("See `appendix7_reproduction.md` for the full table.")
    (RESULTS_DIR / "FINAL_REPORT.md").write_text("\n".join(final_lines))

    print(f"Wrote {RESULTS_DIR}/appendix7_reproduction.md")
    print(f"Wrote {RESULTS_DIR}/FINAL_REPORT.md")


if __name__ == "__main__":
    main()
