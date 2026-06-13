"""
Registry. v2: добавлено поле topics_used (должны быть 8), флаг cv_based_selection
для записей mixture.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REGISTRY_PATH = PROJECT_ROOT / "experiments" / "experiment_registry.json"


def _load() -> List[dict]:
    if not REGISTRY_PATH.exists():
        return []
    with open(REGISTRY_PATH) as f:
        try:
            data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []


def _save(entries: List[dict]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = REGISTRY_PATH.with_suffix(".json.tmp")
    with open(tmp, "w") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False, default=str)
    os.replace(tmp, REGISTRY_PATH)


def publish_per_topic_experiment(
    exp_id: str, family: str, cv_results: Dict,
    feature_groups: List[str], model_desc: str,
    notes: str = "", cv_seconds: Optional[float] = None,
) -> None:
    entries = _load()
    if any(e.get("exp_id") == exp_id for e in entries):
        entries = [e for e in entries if e.get("exp_id") != exp_id]

    cv_per_topic_flat = {}
    for topic, m in cv_results.get("per_topic", {}).items():
        cv_per_topic_flat[topic] = {
            "f1_mean": m.get("f1_mean"),
            "f1_std": m.get("f1_std"),
            "precision_mean": m.get("precision_mean"),
            "recall_mean": m.get("recall_mean"),
        }

    entries.append({
        "exp_id": exp_id,
        "family": family,
        "ts": datetime.utcnow().isoformat(timespec="seconds"),
        "feature_groups": feature_groups,
        "model_desc": model_desc,
        "notes": notes,
        "topics_used": cv_results.get("topics_used", []),
        "cv_per_topic": cv_per_topic_flat,
        "cv_macro_f1": cv_results.get("macro_avg_f1_mean"),
        "cv_macro_precision": cv_results.get("macro_avg_precision_mean"),
        "cv_macro_recall": cv_results.get("macro_avg_recall_mean"),
        "topic_support": cv_results.get("topic_support", {}),
        "cv_seconds": cv_seconds,
        "test_per_topic": None,
        "test_macro_f1": None,
        "test_macro_precision": None,
        "test_macro_recall": None,
        "is_best_for_family_overall_cv": False,
        "best_for_topics_cv": [],   # по CV — главное!
        "is_best_for_family_overall_test": False,  # для отчёта
        "best_for_topics_test": [],
        "flagged": False,
        "cv_based_selection": True,  # invariant: для не-mixture entries
    })

    _recompute_bests(entries)
    _save(entries)


def update_per_topic_with_test(exp_id: str, test_results: Dict) -> None:
    entries = _load()
    found = False
    for e in entries:
        if e.get("exp_id") == exp_id:
            e["test_per_topic"] = test_results.get("per_topic")
            e["test_macro_f1"] = test_results.get("macro_avg_f1")
            e["test_macro_precision"] = test_results.get("macro_avg_precision")
            e["test_macro_recall"] = test_results.get("macro_avg_recall")
            if "family_per_topic" in test_results:
                e["family_per_topic"] = test_results["family_per_topic"]
            found = True
    if not found:
        raise KeyError(f"exp_id {exp_id} not found")

    _recompute_bests(entries)
    _save(entries)


def _recompute_bests(entries: List[dict]) -> None:
    """Recompute best flags. CV-based — главные (для отбора в mixture). Test-based — только для отчёта."""
    for e in entries:
        e["is_best_for_family_overall_cv"] = False
        e["best_for_topics_cv"] = []
        e["is_best_for_family_overall_test"] = False
        e["best_for_topics_test"] = []

    by_family: Dict[str, List[dict]] = {}
    for e in entries:
        if e.get("flagged"):
            continue
        if e.get("family") == "mixture":
            continue
        if e.get("cv_macro_f1") is not None:
            by_family.setdefault(e["family"], []).append(e)

    # CV-based best per family
    for family, fam_entries in by_family.items():
        if not fam_entries:
            continue
        best = max(fam_entries, key=lambda x: x["cv_macro_f1"])
        best["is_best_for_family_overall_cv"] = True

    # CV-based best per topic (across families)
    all_topics = set()
    for e in entries:
        if e.get("cv_per_topic") and not e.get("flagged"):
            all_topics.update(e["cv_per_topic"].keys())

    for topic in all_topics:
        best_f1 = -1.0
        best_exp = None
        for e in entries:
            if e.get("flagged") or e.get("family") == "mixture":
                continue
            tp = (e.get("cv_per_topic") or {}).get(topic)
            if tp and tp.get("f1_mean") is not None and tp["f1_mean"] > best_f1:
                best_f1 = tp["f1_mean"]
                best_exp = e
        if best_exp:
            best_exp["best_for_topics_cv"].append(topic)

    # Test-based — только для отчёта (НЕ используется для mixture)
    by_family_test: Dict[str, List[dict]] = {}
    for e in entries:
        if e.get("flagged") or e.get("family") == "mixture":
            continue
        if e.get("test_macro_f1") is not None:
            by_family_test.setdefault(e["family"], []).append(e)
    for family, fam_entries in by_family_test.items():
        if not fam_entries:
            continue
        best = max(fam_entries, key=lambda x: x["test_macro_f1"])
        best["is_best_for_family_overall_test"] = True

    for topic in all_topics:
        best_f1 = -1.0
        best_exp = None
        for e in entries:
            if e.get("flagged") or e.get("family") == "mixture":
                continue
            tp = (e.get("test_per_topic") or {}).get(topic)
            if tp and tp.get("f1") is not None and tp["f1"] > best_f1:
                best_f1 = tp["f1"]
                best_exp = e
        if best_exp:
            best_exp["best_for_topics_test"].append(topic)


def all_entries() -> List[dict]:
    return _load()


def best_for_family_cv(family: str) -> Optional[dict]:
    """v2: главный метод отбора best — по CV."""
    for e in _load():
        if e.get("family") == family and e.get("is_best_for_family_overall_cv") and not e.get("flagged"):
            return e
    return None


def best_for_topic_cv(topic: str) -> Optional[dict]:
    """Какой эксперимент имеет лучший CV F1 для данной темы."""
    best = None
    for e in _load():
        if e.get("flagged") or e.get("family") == "mixture":
            continue
        tp = (e.get("cv_per_topic") or {}).get(topic)
        if tp and tp.get("f1_mean") is not None:
            if best is None or tp["f1_mean"] > best["cv_per_topic"][topic]["f1_mean"]:
                best = e
    return best


def flag_experiment(exp_id: str, reason: str) -> None:
    entries = _load()
    for e in entries:
        if e.get("exp_id") == exp_id:
            e["flagged"] = True
            e["flag_reason"] = reason
    _recompute_bests(entries)
    _save(entries)
