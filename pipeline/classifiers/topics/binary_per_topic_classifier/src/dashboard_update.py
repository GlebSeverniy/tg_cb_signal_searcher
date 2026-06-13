"""
Dashboard. Two views in one HTML:
1. Experiment timeline (rows = experiments, columns = wavg metrics)
2. Appendix 7 reproduction matrix: rows = topics, cols = model families,
   cells = best F1 found so far per (family, topic) with our value next to authors'.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "experiment_registry.json"
TARGETS_PATH = PROJECT_ROOT / "experiments" / "appendix7_targets.json"
OUT_PATH = PROJECT_ROOT / "dashboard" / "index.html"

FAMILIES = ["catboost", "svc", "complement_nb", "nb", "logreg"]


def _load_json(p):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return None


def _best_per_family_topic(entries):
    """For each (family, topic), find best test_f1 across non-flagged experiments."""
    out = {}  # family -> topic -> {f1, exp_id}
    for e in entries:
        if e.get("flagged"):
            continue
        family = e.get("family")
        if not family or family == "mixture":
            continue
        for topic, m in (e.get("test_per_topic") or {}).items():
            best = out.setdefault(family, {}).get(topic)
            if best is None or m["f1"] > best["f1"]:
                out.setdefault(family, {})[topic] = {"f1": m["f1"], "exp_id": e["exp_id"]}
    return out


def render() -> str:
    entries = _load_json(REGISTRY_PATH) or []
    targets = _load_json(TARGETS_PATH) or {}
    targets_per_topic = (targets or {}).get("per_topic", {})
    targets_wavg = (targets or {}).get("weighted_avg", {})

    # ------------------ Timeline table ------------------
    rows = []
    for e in sorted(entries, key=lambda x: x.get("ts", ""), reverse=True):
        cv = e.get("cv_weighted_avg_f1")
        test = e.get("test_weighted_avg_f1")
        cls = "best" if e.get("is_best_for_family_overall") else ("flagged" if e.get("flagged") else "")
        cv_cell = f"{cv:.4f}" if cv is not None else "—"
        test_cell = f"{test:.4f}" if test is not None else "—"
        family = e.get("family", "")
        wins = ", ".join(e.get("best_for_topics", []))
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{html.escape(e.get('exp_id', ''))}</td>"
            f"<td>{html.escape(family)}</td>"
            f"<td>{html.escape(e.get('ts', ''))}</td>"
            f"<td>{cv_cell}</td>"
            f"<td>{test_cell}</td>"
            f"<td>{html.escape(', '.join(e.get('feature_groups', [])))}</td>"
            f"<td>{html.escape(wins) or ''}</td>"
            f"<td>{html.escape(e.get('notes', ''))}</td>"
            f"</tr>"
        )

    # ------------------ Appendix 7 matrix ------------------
    bests = _best_per_family_topic(entries)
    topic_order = (targets or {}).get("topics_order", []) or sorted({
        t for fam in bests.values() for t in fam.keys()
    })

    matrix_rows = []
    for topic in topic_order:
        cells = [f"<td><b>{html.escape(topic)}</b></td>"]
        for fam in FAMILIES:
            our = bests.get(fam, {}).get(topic)
            their = targets_per_topic.get(topic, {}).get(fam, {})
            their_f1 = their.get("f1")
            our_f1 = our.get("f1") if our else None
            if our_f1 is None:
                cell = f"— / {their_f1:.2f}" if their_f1 is not None else "— / —"
                color = "#999"
            else:
                d = our_f1 - (their_f1 or 0)
                color = "#388e3c" if d >= -0.02 else ("#f57c00" if d >= -0.05 else "#d32f2f")
                cell = f"{our_f1:.3f} / {their_f1:.2f}" if their_f1 is not None else f"{our_f1:.3f}"
            cells.append(f"<td style='color:{color}'>{cell}</td>")
        matrix_rows.append(f"<tr>{''.join(cells)}</tr>")

    # Weighted avg row
    wavg_cells = ["<td><b>Weighted avg</b></td>"]
    for fam in FAMILIES:
        # ours: take max wavg over non-flagged entries of this family
        ours_w = None
        for e in entries:
            if e.get("flagged") or e.get("family") != fam:
                continue
            t = e.get("test_weighted_avg_f1")
            if t is not None and (ours_w is None or t > ours_w):
                ours_w = t
        their_w = targets_wavg.get(fam, {}).get("f1")
        if ours_w is None:
            wavg_cells.append(f"<td>— / {their_w:.2f}</td>" if their_w else "<td>—</td>")
        else:
            d = ours_w - (their_w or 0)
            color = "#388e3c" if d >= -0.02 else ("#f57c00" if d >= -0.05 else "#d32f2f")
            cell = f"{ours_w:.3f} / {their_w:.2f}" if their_w else f"{ours_w:.3f}"
            wavg_cells.append(f"<td style='color:{color};font-weight:600'>{cell}</td>")
    matrix_rows.append(f"<tr style='background:#f5f5f5'>{''.join(wavg_cells)}</tr>")

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Binary per-topic classifier</title>
<style>
body {{ font-family: -apple-system, sans-serif; padding: 24px; max-width: 1300px; margin: 0 auto; }}
h2 {{ margin-top: 32px; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; margin-top: 12px; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #f5f5f5; }}
tr.best td {{ background: #e8f5e9; font-weight: 600; }}
tr.flagged td {{ background: #ffebee; color: #888; text-decoration: line-through; }}
.legend {{ font-size: 12px; color: #555; margin: 8px 0; }}
</style></head>
<body>
<h1>Binary per-topic classifier — experiments</h1>

<h2>Appendix 7 reproduction (ours / authors')</h2>
<p class="legend">
  Cell shows our best F1 (left) vs authors' F1 (right) per (topic, family).
  <span style="color:#388e3c">green</span> = within ±0.02 of authors,
  <span style="color:#f57c00">orange</span> = ±0.05,
  <span style="color:#d32f2f">red</span> = worse.
</p>
<table>
<thead><tr><th>Topic</th>{''.join(f'<th>{f}</th>' for f in FAMILIES)}</tr></thead>
<tbody>
{''.join(matrix_rows) or '<tr><td colspan="6">No experiments yet.</td></tr>'}
</tbody>
</table>

<h2>Experiment timeline</h2>
<table>
<thead><tr><th>exp_id</th><th>family</th><th>ts</th><th>cv wavg F1</th><th>test wavg F1</th><th>features</th><th>best for topics</th><th>notes</th></tr></thead>
<tbody>
{''.join(rows) or '<tr><td colspan="8">No experiments yet.</td></tr>'}
</tbody>
</table>

<p style="color:#888;font-size:12px;margin-top:24px">Generated by src/dashboard_update.py</p>
</body></html>
"""


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render())
    print(f"Wrote {OUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
