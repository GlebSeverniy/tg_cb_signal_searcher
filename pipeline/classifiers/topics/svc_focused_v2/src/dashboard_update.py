"""Simple HTML dashboard. v2: shows CV-selected best per cell."""

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


def _best_per_fam_topic_cv(entries):
    out = {}
    for e in entries:
        if e.get("flagged") or e.get("family") == "mixture":
            continue
        fam = e.get("family")
        for topic, m in (e.get("cv_per_topic") or {}).items():
            if m.get("f1_mean") is None:
                continue
            cur = out.setdefault(fam, {}).get(topic)
            if cur is None or m["f1_mean"] > cur["f1_mean"]:
                out.setdefault(fam, {})[topic] = {"f1": m["f1_mean"], "exp_id": e["exp_id"]}
    return out


def render():
    entries = _load_json(REGISTRY_PATH) or []
    targets = _load_json(TARGETS_PATH) or {}
    targets_pt = targets.get("per_topic", {})
    topics_order = targets.get("topics_order", [])
    priority = set(targets.get("priority_topics", {}).get("high", []))

    bests = _best_per_fam_topic_cv(entries)

    matrix_rows = []
    for topic in topics_order:
        cells = [f"<td><b>{html.escape(topic)}</b>{' ★' if topic in priority else ''}</td>"]
        for fam in FAMILIES:
            our = bests.get(fam, {}).get(topic)
            their = targets_pt.get(topic, {}).get(fam, {})
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

    rows = []
    for e in sorted(entries, key=lambda x: x.get("ts", ""), reverse=True):
        cv = e.get("cv_macro_f1")
        test = e.get("test_macro_f1")
        cls = "mix" if e.get("family") == "mixture" else ("flagged" if e.get("flagged") else "")
        cv_cell = f"{cv:.4f}" if cv is not None else "—"
        test_cell = f"{test:.4f}" if test is not None else "—"
        wins = ", ".join(e.get("best_for_topics_cv", []))
        rows.append(
            f"<tr class='{cls}'>"
            f"<td>{html.escape(e.get('exp_id', ''))}</td>"
            f"<td>{html.escape(e.get('family', ''))}</td>"
            f"<td>{cv_cell}</td>"
            f"<td>{test_cell}</td>"
            f"<td>{html.escape(', '.join(e.get('feature_groups', [])))}</td>"
            f"<td>{html.escape(wins)}</td>"
            f"<td>{html.escape(e.get('notes', ''))}</td>"
            f"</tr>"
        )

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>SVC focused v2</title>
<style>
body {{ font-family: -apple-system, sans-serif; padding: 24px; max-width: 1300px; margin: 0 auto; }}
table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
th, td {{ padding: 6px 10px; border-bottom: 1px solid #eee; text-align: left; }}
th {{ background: #f5f5f5; }}
tr.mix td {{ background: #e3f2fd; font-weight: 600; }}
tr.flagged td {{ background: #ffebee; color: #888; text-decoration: line-through; }}
.legend {{ font-size: 12px; color: #555; margin: 8px 0; }}
</style></head>
<body>
<h1>SVC focused v2 — 8 topics, CV-honest selection</h1>
<p class="legend">Cell: our CV F1 / authors F1. ★ = priority topic.
  <span style="color:#388e3c">green</span> ≥ −0.02; <span style="color:#f57c00">orange</span> ≥ −0.05;
  <span style="color:#d32f2f">red</span> worse.</p>
<h2>Best per cell (by CV)</h2>
<table>
<thead><tr><th>Topic</th>{''.join(f'<th>{f}</th>' for f in FAMILIES)}</tr></thead>
<tbody>{''.join(matrix_rows) or '<tr><td colspan="6">No experiments yet.</td></tr>'}</tbody>
</table>
<h2>Experiments</h2>
<table>
<thead><tr><th>exp_id</th><th>family</th><th>cv macro</th><th>test macro</th><th>features</th><th>cv-best for</th><th>notes</th></tr></thead>
<tbody>{''.join(rows) or '<tr><td colspan="7">No experiments.</td></tr>'}</tbody>
</table>
</body></html>
"""


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(render())
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
