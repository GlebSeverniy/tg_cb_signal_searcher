"""
Per-topic experiment registry. JSON file shared by all agents.

Schema (one entry per experiment, i.e. per (model_family, hyperparams) pair
evaluated on all topics):
{
  "exp_id": "exp_002_svc_baseline",
  "family": "svc" | "logreg" | "nb" | "complement_nb" | "catboost" | "mixture",
  "ts": "...",
  "feature_groups": [...],
  "model_desc": "LinearSVC({...})",
  "notes": "...",

  "cv_per_topic": {topic: {f1_mean, f1_std, precision_mean, recall_mean, ...}},
  "cv_weighted_avg_f1": float,
  "cv_weighted_avg_precision": float,
  "cv_weighted_avg_recall": float,
  "cv_macro_avg_f1": float,          # simple mean across topics — matches authors' Appendix 7 summary
  "cv_macro_avg_precision": float,
  "cv_macro_avg_recall": float,
  "cv_macro_std_f1": float,          # std-dev of per-topic f1_mean
  "topic_support": {topic: int},
  "cv_seconds": float,

  "test_per_topic": {topic: {precision, recall, f1, support}},
  "test_weighted_avg_f1": float | null,
  "test_weighted_avg_precision": float | null,
  "test_weighted_avg_recall": float | null,
  "test_macro_avg_f1": float | null,  # simple mean across topics — use to compare with authors
  "test_macro_avg_precision": float | null,
  "test_macro_avg_recall": float | null,
  "family_per_topic": {topic: family} | null,        # only for mixture entries

  "is_best_for_family_overall": false,                # best macro F1 within its family
  "best_for_topics": ["topic1", "topic2", ...],       # topics where this exp is current best
  "flagged": false
}
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
        json.dump(entries, f, indent=2, default=str)
    os.replace(tmp, REGISTRY_PATH)


def publish_per_topic_experiment(
    exp_id: str,
    family: str,
    cv_results: Dict,
    feature_groups: List[str],
    model_desc: str,
    notes: str = "",
    cv_seconds: Optional[float] = None,
) -> None:
    entries = _load()
    if any(e.get("exp_id") == exp_id for e in entries):
        entries = [e for e in entries if e.get("exp_id") != exp_id]

    # Flatten per-topic
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
        "cv_per_topic": cv_per_topic_flat,
        "cv_weighted_avg_f1": cv_results.get("weighted_avg_f1_mean"),
        "cv_weighted_avg_precision": cv_results.get("weighted_avg_precision_mean"),
        "cv_weighted_avg_recall": cv_results.get("weighted_avg_recall_mean"),
        "cv_macro_avg_f1": cv_results.get("macro_avg_f1"),
        "cv_macro_avg_precision": cv_results.get("macro_avg_precision"),
        "cv_macro_avg_recall": cv_results.get("macro_avg_recall"),
        "cv_macro_std_f1": cv_results.get("macro_std_f1"),
        "topic_support": cv_results.get("topic_support", {}),
        "cv_seconds": cv_seconds,
        "test_per_topic": None,
        "test_weighted_avg_f1": None,
        "test_weighted_avg_precision": None,
        "test_weighted_avg_recall": None,
        "test_macro_avg_f1": None,
        "test_macro_avg_precision": None,
        "test_macro_avg_recall": None,
        "family_per_topic": None,
        "is_best_for_family_overall": False,
        "best_for_topics": [],
        "flagged": False,
    })

    _recompute_bests(entries)
    _save(entries)


def update_per_topic_with_test(
    exp_id: str,
    test_results: Dict,
) -> None:
    entries = _load()
    found = False
    for e in entries:
        if e.get("exp_id") == exp_id:
            e["test_per_topic"] = test_results.get("per_topic")
            e["test_weighted_avg_f1"] = test_results.get("weighted_avg_f1")
            e["test_weighted_avg_precision"] = test_results.get("weighted_avg_precision")
            e["test_weighted_avg_recall"] = test_results.get("weighted_avg_recall")
            e["test_macro_avg_f1"] = test_results.get("macro_avg_f1")
            e["test_macro_avg_precision"] = test_results.get("macro_avg_precision")
            e["test_macro_avg_recall"] = test_results.get("macro_avg_recall")
            if "family_per_topic" in test_results:
                e["family_per_topic"] = test_results["family_per_topic"]
            found = True
    if not found:
        raise KeyError(f"exp_id {exp_id} not found — call publish_per_topic_experiment first")

    _recompute_bests(entries)
    _save(entries)


def _recompute_bests(entries: List[dict]) -> None:
    """Recompute is_best_for_family_overall and best_for_topics flags."""
    for e in entries:
        e["is_best_for_family_overall"] = False
        e["best_for_topics"] = []

    by_family: Dict[str, List[dict]] = {}
    for e in entries:
        if e.get("flagged"):
            continue
        # Require at least test results to rank
        if e.get("test_macro_avg_f1") is None and e.get("test_weighted_avg_f1") is None:
            continue
        by_family.setdefault(e["family"], []).append(e)

    def _family_sort_key(x: dict) -> float:
        """Rank by test_macro_avg_f1 first (matches authors); fall back to cv_macro_avg_f1."""
        if x.get("test_macro_avg_f1") is not None:
            return x["test_macro_avg_f1"]
        if x.get("cv_macro_avg_f1") is not None:
            return x["cv_macro_avg_f1"]
        return x.get("test_weighted_avg_f1") or -1.0

    for family, fam_entries in by_family.items():
        best = max(fam_entries, key=_family_sort_key)
        best["is_best_for_family_overall"] = True

    # Best per topic across all families
    all_topics = set()
    for e in entries:
        if e.get("test_per_topic") and not e.get("flagged"):
            all_topics.update(e["test_per_topic"].keys())

    for topic in all_topics:
        best_f1 = -1.0
        best_exp = None
        for e in entries:
            if e.get("flagged"):
                continue
            tp = (e.get("test_per_topic") or {}).get(topic)
            if tp and tp["f1"] > best_f1:
                best_f1 = tp["f1"]
                best_exp = e
        if best_exp:
            best_exp["best_for_topics"].append(topic)


def all_entries() -> List[dict]:
    return _load()


def best_for_family(family: str) -> Optional[dict]:
    for e in _load():
        if e.get("family") == family and e.get("is_best_for_family_overall") and not e.get("flagged"):
            return e
    return None


def flag_experiment(exp_id: str, reason: str) -> None:
    entries = _load()
    for e in entries:
        if e.get("exp_id") == exp_id:
            e["flagged"] = True
            e["flag_reason"] = reason
    _recompute_bests(entries)
    _save(entries)
