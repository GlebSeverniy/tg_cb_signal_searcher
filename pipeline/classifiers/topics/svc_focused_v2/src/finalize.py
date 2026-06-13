"""
Finalize: repro check + appendix7_reproduction_v2 + v1_vs_v2_comparison + FINAL_REPORT.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib

from .data_loader import load_test, load_test_binary
from .registry import all_entries
from . import ALLOWED_TOPICS, PRIORITY_TOPICS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
TARGETS_PATH = PROJECT_ROOT / "experiments" / "appendix7_targets.json"

FAMILIES = ["catboost", "svc", "complement_nb", "nb", "logreg"]
FAMILY_HUMAN = {"catboost": "CatBoost", "svc": "SVC",
                "complement_nb": "Compl. NB", "nb": "NB", "logreg": "Лог. регрессия"}

V1_RESULTS_PATH = Path.home() / "Downloads" / "binary_per_topic_classifier" / "results" / "svc_per_topic_metrics.json"
V1_FINAL_PATH = Path.home() / "Downloads" / "binary_per_topic_classifier" / "results" / "FINAL_REPORT.md"


def _best_for_family_topic_cv(entries):
    """Best per (family, topic) by CV F1."""
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
                out.setdefault(fam, {})[topic] = {**m, "exp_id": e["exp_id"]}
    return out


def _best_test_per_family_topic(entries):
    """Best per (family, topic) by TEST F1 — для отчёта (не для отбора)."""
    out = {}
    for e in entries:
        if e.get("flagged") or e.get("family") == "mixture":
            continue
        fam = e.get("family")
        for topic, m in (e.get("test_per_topic") or {}).items():
            if m.get("f1") is None:
                continue
            cur = out.setdefault(fam, {}).get(topic)
            if cur is None or m["f1"] > cur["f1"]:
                out.setdefault(fam, {})[topic] = {**m, "exp_id": e["exp_id"]}
    return out


def _repro_check(exp_id: str = "final_mixture_v2_cv") -> dict:
    """
    Перезагрузить mixture из joblib, прогнать на test, сверить побитово.

    v2 canonical: для каждой темы зовёт load_test_binary(topic) → 1093 строки
    с binary 0/1 меткой из данных (не multiclass-derived). Должно побитово совпасть
    с результатами из final_metrics.json, которые тоже посчитаны через load_test_binary.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score
    pkg = joblib.load(RESULTS_DIR / "final_mixture.joblib")
    repro = {}
    for t, pipe in pkg.items():
        # v2: binary column loading — все 1093 тестовых строки с бинарной меткой
        X_test, y_bin = load_test_binary(t)
        import pandas as pd
        X_test = pd.Series(X_test).reset_index(drop=True)
        y_bin = pd.Series(y_bin).reset_index(drop=True)
        y_pred = pipe.predict(X_test)
        repro[t] = {
            "precision": float(precision_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "n_test": int(len(X_test)),
            "n_positives": int(y_bin.sum()),
        }
    # Compare with stored
    stored = json.loads((RESULTS_DIR / "final_metrics.json").read_text())["per_topic"]
    diffs = {}
    for t in repro:
        if t in stored:
            d = abs(repro[t]["f1"] - stored[t]["f1"])
            diffs[t] = round(d, 6)
    return {"repro": repro, "stored": stored, "max_diff": max(diffs.values()) if diffs else 0,
            "diffs": diffs}


def main():
    entries = all_entries()
    targets = json.loads(TARGETS_PATH.read_text())
    topics = targets["topics_order"]
    targets_pt = targets["per_topic"]

    bests_cv = _best_for_family_topic_cv(entries)
    bests_test = _best_test_per_family_topic(entries)

    # ---- Appendix 7 reproduction v2 ----
    lines = ["# Appendix 7 reproduction v2 (8 topics, CV-selected best per cell)\n"]
    lines.append(f"_Selection: per (family, topic) cell — entry с максимальным cv_per_topic[topic]['f1_mean']._")
    lines.append(f"_Test F1 в скобках — для отчёта, не используется для отбора._\n")

    header = "| Тема |"
    for f in FAMILIES:
        header += f" {FAMILY_HUMAN[f]} CV F1 / Test F1 / Authors |"
    lines.append(header)
    lines.append("|" + "---|" * (1 + len(FAMILIES)))

    for topic in topics:
        row = f"| **{topic}**{' ★' if topic in PRIORITY_TOPICS else ''} |"
        for f in FAMILIES:
            cv = bests_cv.get(f, {}).get(topic)
            test = bests_test.get(f, {}).get(topic)
            their = targets_pt.get(topic, {}).get(f)
            cv_f1 = cv["f1_mean"] if cv else None
            test_f1 = test["f1"] if test else None
            their_f1 = their["f1"] if their else None
            cell = ""
            if cv_f1 is not None:
                cell += f"{cv_f1:.3f}"
            else:
                cell += "—"
            if test_f1 is not None:
                cell += f" / ({test_f1:.3f})"
            else:
                cell += " / —"
            if their_f1 is not None:
                cell += f" / {their_f1:.2f}"
            else:
                cell += " / —"
            row += f" {cell} |"
        lines.append(row)

    (RESULTS_DIR / "appendix7_reproduction_v2.md").write_text("\n".join(lines))
    print(f"Wrote {RESULTS_DIR / 'appendix7_reproduction_v2.md'}")

    # ---- v1 vs v2 comparison ----
    v1_metrics = None
    v1_macro = None
    if V1_RESULTS_PATH.exists():
        try:
            v1_full = json.loads(V1_RESULTS_PATH.read_text())
            # v1 format: {"per_topic": {topic: {...}}, "macro_avg_f1": ...}
            v1_metrics = v1_full.get("per_topic", v1_full)
            v1_macro = v1_full.get("macro_avg_f1")
        except Exception as e:
            print(f"Could not load v1 metrics: {e}")

    final_v2 = json.loads((RESULTS_DIR / "final_metrics.json").read_text())
    v2_per_topic = final_v2.get("per_topic", {})

    cmp_lines = ["# v1 vs v2 comparison\n"]
    cmp_lines.append("> **Important caveat**: v1 numbers are on **11-topic** test (Ковид/Другое/НПС included; partly inflated by test-fishing per audit). v2 numbers are on **8-topic** test using **binary loading** (1093 rows per topic, not 973 multiclass). Direct per-topic F1 comparison is meaningful only on common topics; macro is NOT comparable (different denominator).")
    cmp_lines.append("")
    if v1_macro is not None:
        cmp_lines.append(f"- v1 macro F1 on 11 topics (test, fishing-inflated): **{v1_macro:.4f}** (auditor-recomputed CV-fair = 0.6385)")
    cmp_lines.append(f"- v2 macro F1 on 8 topics (test, CV-selected mixture): **{final_v2.get('macro_avg_f1', 0):.4f}**")
    cmp_lines.append(f"- v2 selection method: {final_v2.get('selection_method')}")
    cmp_lines.append(f"v2 selection_method: {final_v2.get('selection_method')}")
    cmp_lines.append("")
    cmp_lines.append("| Topic | v1 F1 (test-fish) | v2 F1 (CV-fair) | Δ v2-v1 | Authors | Δ v2-Authors |")
    cmp_lines.append("|---|---|---|---|---|---|")

    for topic in topics:
        v1_f = "—"
        if v1_metrics and topic in v1_metrics:
            v1_f = f"{v1_metrics[topic].get('f1', 0):.4f}"
        v2_f = "—"
        v2_val = None
        if topic in v2_per_topic:
            v2_val = v2_per_topic[topic].get("f1")
            if v2_val is not None:
                v2_f = f"{v2_val:.4f}"
        their = targets_pt.get(topic, {})
        their_max = max([m.get("f1", 0) for m in their.values()]) if their else 0
        delta_v1 = (v2_val - v1_metrics[topic]["f1"]) if (v2_val is not None and v1_metrics and topic in v1_metrics) else None
        delta_auth = (v2_val - their_max) if v2_val is not None else None
        delta_v1_str = f"{delta_v1:+.4f}" if delta_v1 is not None else "—"
        delta_auth_str = f"{delta_auth:+.4f}" if delta_auth is not None else "—"
        cmp_lines.append(f"| {topic}{' ★' if topic in PRIORITY_TOPICS else ''} | {v1_f} | {v2_f} | "
                          f"{delta_v1_str} | "
                          f"{their_max:.2f} | {delta_auth_str} |")

    (RESULTS_DIR / "v1_vs_v2_comparison.md").write_text("\n".join(cmp_lines))
    print(f"Wrote {RESULTS_DIR / 'v1_vs_v2_comparison.md'}")

    # ---- Repro check ----
    print("\nRunning repro check...")
    try:
        repro = _repro_check()
        print(f"  max diff: {repro['max_diff']}")
    except Exception as e:
        print(f"  repro failed: {e}")
        repro = {"max_diff": "FAILED", "error": str(e)}

    # ---- FINAL_REPORT ----
    boot_path = RESULTS_DIR / "bootstrap_ci.json"
    boot = json.loads(boot_path.read_text()) if boot_path.exists() else None

    fin = ["# Final report v2", ""]
    fin.append(f"## Goal achievement")
    fin.append(f"- Macro F1 on test (CV-selected): **{final_v2.get('macro_avg_f1', 0):.4f}**")
    if boot:
        fin.append(f"- Bootstrap 95% CI: [{boot['macro_f1_ci95'][0]:.4f}, {boot['macro_f1_ci95'][1]:.4f}]")
    fin.append(f"- Authors macro on 8 (SVC): 0.7775")
    fin.append(f"- Selection method: {final_v2.get('selection_method')} ✅ (NOT test-fishing)")
    fin.append("")
    fin.append(f"## Priority topics gap")
    for t in PRIORITY_TOPICS:
        if t in v2_per_topic:
            our = v2_per_topic[t].get("f1", 0)
            their = targets_pt.get(t, {})
            their_max = max([m.get("f1", 0) for m in their.values()]) if their else 0
            fin.append(f"- {t}: ours={our:.4f}, authors_best={their_max:.2f}, Δ={our-their_max:+.4f}")
    fin.append("")
    fin.append(f"## Repro check")
    fin.append(f"- max F1 diff (joblib reload vs stored): {repro['max_diff']}")
    fin.append("")
    fin.append(f"See: appendix7_reproduction_v2.md, v1_vs_v2_comparison.md, bootstrap_ci.json")
    (RESULTS_DIR / "FINAL_REPORT.md").write_text("\n".join(fin))
    print(f"Wrote {RESULTS_DIR / 'FINAL_REPORT.md'}")


if __name__ == "__main__":
    main()
