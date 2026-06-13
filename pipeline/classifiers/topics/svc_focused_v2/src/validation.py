"""
Validation — single source of truth.

v2: macro-avg по 8 темам (а не 11). Threshold tuning только на OOF.

v2 canonical API:
  cv_evaluate_binary_per_topic_loader  — для каждой темы вызывает loader_fn(topic)
                                          → (X, y_bin) с 4371 строками.
  evaluate_on_test_per_topic_loader    — аналогично для теста (1093 строки).

Legacy multiclass API (backward compat):
  cv_evaluate_binary_per_topic         — принимает уже загруженные X, y_multiclass.
  evaluate_on_test_per_topic           — принимает уже загруженные fitted_pipelines, X_test, y_test.
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold

from . import RANDOM_STATE, ALLOWED_TOPICS, COL_TOPIC_MAP


def _run_cv_for_topic(
    model_factory: Callable,
    X: pd.Series,
    y_bin: pd.Series,
    topic: str,
    n_splits: int,
    verbose: bool,
    return_oof: bool,
) -> Dict:
    """Shared CV logic for a single binary topic. Used by both loader and legacy APIs."""
    n_pos = int(y_bin.sum())
    if n_pos < n_splits:
        print(f"  WARNING topic '{topic}' has only {n_pos} positives — skipping", flush=True)
        return {
            "precision_mean": float("nan"),
            "recall_mean": float("nan"),
            "f1_mean": float("nan"),
            "f1_std": float("nan"),
            "fold_scores": [],
            "support_mean": float(n_pos),
            "fit_seconds_mean": 0.0,
            "skipped": True,
        }

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    ps, rs, fs, fits = [], [], [], []
    oof = np.zeros(len(X), dtype=int) if return_oof else None

    for fi, (tr_idx, va_idx) in enumerate(skf.split(X, y_bin)):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y_bin.iloc[tr_idx], y_bin.iloc[va_idx]

        model = model_factory()
        t0 = time.time()
        model.fit(X_tr, y_tr)
        fits.append(time.time() - t0)

        y_pred = model.predict(X_va)
        ps.append(precision_score(y_va, y_pred, pos_label=1, zero_division=0))
        rs.append(recall_score(y_va, y_pred, pos_label=1, zero_division=0))
        fs.append(f1_score(y_va, y_pred, pos_label=1, zero_division=0))
        if return_oof:
            oof[va_idx] = y_pred

    result = {
        "precision_mean": float(np.mean(ps)),
        "precision_std": float(np.std(ps)),
        "recall_mean": float(np.mean(rs)),
        "recall_std": float(np.std(rs)),
        "f1_mean": float(np.mean(fs)),
        "f1_std": float(np.std(fs)),
        "fold_scores": [float(x) for x in fs],
        "support_mean": float(n_pos / n_splits),
        "fit_seconds_mean": float(np.mean(fits)),
        "n_total": int(len(X)),
        "n_positives": n_pos,
    }
    if return_oof:
        result["oof_predictions"] = oof.tolist()

    if verbose:
        print(
            f"  topic '{topic}': F1={result['f1_mean']:.4f} ± {result['f1_std']:.4f}, "
            f"P={result['precision_mean']:.3f}, R={result['recall_mean']:.3f}, "
            f"n_pos={n_pos}, n_total={len(X)}",
            flush=True,
        )
    return result


def _aggregate_cv_results(per_topic: Dict, topics: List, n_train: int, n_splits: int,
                           verbose: bool) -> Dict:
    """Aggregate per-topic CV results into macro metrics."""
    valid = [v for v in per_topic.values() if not v.get("skipped")]
    macro_p = float(np.mean([v["precision_mean"] for v in valid])) if valid else 0
    macro_r = float(np.mean([v["recall_mean"] for v in valid])) if valid else 0
    macro_f = float(np.mean([v["f1_mean"] for v in valid])) if valid else 0
    topic_support = {t: per_topic[t].get("n_positives", 0) for t in per_topic}

    if verbose:
        print(
            f"\nMacro CV (8 topics): P={macro_p:.4f}, R={macro_r:.4f}, F1={macro_f:.4f}",
            flush=True,
        )
    return {
        "per_topic": per_topic,
        "macro_avg_f1_mean": macro_f,
        "macro_avg_precision_mean": macro_p,
        "macro_avg_recall_mean": macro_r,
        "topic_support": topic_support,
        "topics_used": [str(t) for t in topics],
        "n_train": n_train,
        "n_splits": n_splits,
    }


# ============================================================
# v2 CANONICAL — loader-based API (binary columns, all rows)
# ============================================================

def cv_evaluate_binary_per_topic_loader(
    model_factory: Callable,
    loader_fn: Callable,
    topics: Optional[List] = None,
    n_splits: int = 5,
    verbose: bool = True,
    return_oof: bool = False,
) -> Dict:
    """
    v2 CANONICAL. Per-topic binary CV с загрузкой через binary loader.

    Для каждой темы вызывает loader_fn(topic) → (X, y_bin) с 4371 строками
    (все строки датасета, включая multi-label rows). Это позволяет использовать
    ВСЕ позитивные примеры для каждого топика (например, НБП: 160 vs 112 в multiclass).

    Args:
        model_factory: callable → Pipeline (fitted inside CV per fold).
        loader_fn:     callable(topic: str) → (X: pd.Series, y: pd.Series),
                       где y — binary 0/1 column из данных.
                       Например: from src.data_loader import load_train_binary
        topics:        список canonical ALLOWED_TOPICS (None = все 8).
        n_splits:      число fold-ов StratifiedKFold (default=5).
        verbose:       печатать per-topic результаты.
        return_oof:    добавить oof_predictions в результаты.

    Returns:
        {
          'per_topic': {topic: {f1_mean, f1_std, precision_mean, recall_mean,
                                n_total, n_positives, fold_scores, ...}},
          'macro_avg_f1_mean': float,
          'topic_support': {topic: n_positives},
          'topics_used': [...],
          'n_train': int,   # 4371 (per topic, same for all)
          'n_splits': int,
        }
    """
    if topics is None:
        topics = list(ALLOWED_TOPICS)

    # INVARIANT: все темы должны быть из COL_TOPIC_MAP (canonical ALLOWED_TOPICS)
    assert all(t in COL_TOPIC_MAP for t in topics), (
        f"topics must be canonical ALLOWED_TOPICS. "
        f"Invalid: {[t for t in topics if t not in COL_TOPIC_MAP]}"
    )

    per_topic: Dict[str, Dict] = {}
    n_train_ref = 0

    for topic in topics:
        X, y_bin = loader_fn(topic)
        y_bin = pd.Series(y_bin).reset_index(drop=True)
        X = pd.Series(X).reset_index(drop=True)
        n_train_ref = len(X)

        result = _run_cv_for_topic(model_factory, X, y_bin, topic, n_splits, verbose, return_oof)
        per_topic[str(topic)] = result

    return _aggregate_cv_results(per_topic, topics, n_train_ref, n_splits, verbose)


def evaluate_on_test_per_topic_loader(
    fitted_pipelines: Dict[str, object],
    loader_fn: Callable,
) -> Dict:
    """
    v2 CANONICAL. Per-topic binary test eval с загрузкой через binary loader.

    Для каждой темы вызывает loader_fn(topic) → (X_test, y_bin_test).
    Использует 1093 строки для каждого топика (vs ~973 в multiclass подходе).

    ОДНОКРАТНО на финальной mixture. НЕ использовать для отбора mixture.

    Args:
        fitted_pipelines: {topic: fitted_pipeline}
        loader_fn:        callable(topic: str) → (X: pd.Series, y: pd.Series)
                          Например: from src.data_loader import load_test_binary
    """
    per_topic = {}

    for topic, pipe in fitted_pipelines.items():
        X_test, y_bin = loader_fn(topic)
        X_test = pd.Series(X_test).reset_index(drop=True)
        y_bin = pd.Series(y_bin).reset_index(drop=True)
        n_pos = int(y_bin.sum())

        y_pred = pipe.predict(X_test)
        per_topic[str(topic)] = {
            "precision": float(precision_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "support": n_pos,
            "n_test": int(len(X_test)),
            "y_pred": y_pred.tolist(),  # для bootstrap CI
        }

    valid_topics = list(per_topic.keys())
    macro_p = float(np.mean([per_topic[t]["precision"] for t in valid_topics])) if valid_topics else 0
    macro_r = float(np.mean([per_topic[t]["recall"] for t in valid_topics])) if valid_topics else 0
    macro_f = float(np.mean([per_topic[t]["f1"] for t in valid_topics])) if valid_topics else 0
    n_test = per_topic[valid_topics[0]]["n_test"] if valid_topics else 0

    return {
        "per_topic": per_topic,
        "macro_avg_precision": macro_p,
        "macro_avg_recall": macro_r,
        "macro_avg_f1": macro_f,
        "n_test": n_test,
    }


# ============================================================
# v2 CANONICAL WITH OOF THRESHOLD TUNING
# ============================================================

def cv_evaluate_binary_per_topic_loader_with_threshold(
    model_factory: Callable,
    loader_fn: Callable,
    topics: Optional[List] = None,
    n_splits: int = 5,
    n_thresholds: int = 200,
    verbose: bool = True,
) -> Dict:
    """
    v2 CANONICAL with OOF threshold tuning. Same as cv_evaluate_binary_per_topic_loader
    but tunes decision threshold τ on OOF per topic.

    Per topic:
      1. CV with StratifiedKFold(5) — accumulate OOF scores (decision_function or predict_proba).
      2. After all folds — sweep τ over [oof_scores.min, oof_scores.max] (n_thresholds=200).
         Pick τ* maximising OOF F1.
      3. Compute per-fold metrics re-thresholded at τ* for f1_mean/std.
      4. Return per_topic[topic] with extra fields: 'best_threshold', 'oof_f1_at_best_threshold'.

    INVARIANT: τ is chosen ONLY on OOF scores (val fold scores, never train fold, NEVER test).
    NEVER call this with test data inside model_factory or loader_fn.

    Args:
        model_factory: callable → Pipeline (fitted inside CV per fold).
        loader_fn:     callable(topic: str) → (X: pd.Series, y: pd.Series), binary 0/1.
                       Например: from src.data_loader import load_train_binary
        topics:        список canonical ALLOWED_TOPICS (None = все 8).
        n_splits:      число fold-ов StratifiedKFold (default=5).
        n_thresholds:  число точек sweep'а threshold (default=200).
        verbose:       печатать per-topic результаты.

    Returns:
        Same schema as cv_evaluate_binary_per_topic_loader plus per_topic[*]['best_threshold'].
    """
    if topics is None:
        topics = list(ALLOWED_TOPICS)

    assert all(t in COL_TOPIC_MAP for t in topics), (
        f"topics must be canonical ALLOWED_TOPICS. "
        f"Invalid: {[t for t in topics if t not in COL_TOPIC_MAP]}"
    )

    per_topic: Dict[str, Dict] = {}
    n_train_ref = 0

    for topic in topics:
        X, y_bin = loader_fn(topic)
        y_bin = pd.Series(y_bin).reset_index(drop=True)
        X = pd.Series(X).reset_index(drop=True)
        n_train_ref = len(X)
        n_pos = int(y_bin.sum())

        if n_pos < n_splits:
            print(f"  WARNING topic '{topic}' has only {n_pos} positives — skipping", flush=True)
            per_topic[str(topic)] = {
                "precision_mean": float("nan"), "recall_mean": float("nan"),
                "f1_mean": float("nan"), "f1_std": float("nan"),
                "fold_scores": [], "support_mean": float(n_pos),
                "fit_seconds_mean": 0.0, "skipped": True,
                "best_threshold": 0.0, "oof_f1_at_best_threshold": float("nan"),
            }
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

        # Step 1: Accumulate OOF scores across all folds
        oof_scores = np.zeros(len(X), dtype=float)
        oof_y = np.zeros(len(X), dtype=int)
        fold_times = []
        fold_val_indices = []

        for fi, (tr_idx, va_idx) in enumerate(skf.split(X, y_bin)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr = y_bin.iloc[tr_idx]

            model = model_factory()
            t0 = time.time()
            model.fit(X_tr, y_tr)
            fold_times.append(time.time() - t0)

            # Get scores on validation fold — ONLY OOF, NOT test
            if hasattr(model, "decision_function"):
                scores_va = model.decision_function(X_va)
            elif hasattr(model, "predict_proba"):
                scores_va = model.predict_proba(X_va)[:, 1]
            else:
                scores_va = model.predict(X_va).astype(float)

            oof_scores[va_idx] = scores_va
            oof_y[va_idx] = y_bin.iloc[va_idx].values
            fold_val_indices.append(va_idx)

        # Step 2: Sweep τ on OOF scores ONLY (no test data involved here)
        # ASSERTION: oof_scores and oof_y are derived solely from train folds (OOF val scores)
        score_min, score_max = oof_scores.min(), oof_scores.max()
        if score_min == score_max:
            best_threshold = score_min
        else:
            thresholds = np.linspace(score_min, score_max, n_thresholds)
            best_f1_oof = -1.0
            best_threshold = 0.0
            for tau in thresholds:
                y_pred_tau = (oof_scores >= tau).astype(int)
                f1_tau = f1_score(oof_y, y_pred_tau, pos_label=1, zero_division=0)
                if f1_tau > best_f1_oof:
                    best_f1_oof = f1_tau
                    best_threshold = float(tau)

        # OOF F1 at best τ (aggregate, not per-fold)
        oof_pred_best = (oof_scores >= best_threshold).astype(int)
        oof_f1_at_best = float(f1_score(oof_y, oof_pred_best, pos_label=1, zero_division=0))
        oof_p_at_best = float(precision_score(oof_y, oof_pred_best, pos_label=1, zero_division=0))
        oof_r_at_best = float(recall_score(oof_y, oof_pred_best, pos_label=1, zero_division=0))

        # Step 3: Per-fold metrics re-thresholded at τ* for f1_mean/std reporting
        ps, rs, fs = [], [], []
        for va_idx in fold_val_indices:
            y_true_fold = oof_y[va_idx]
            y_pred_fold = (oof_scores[va_idx] >= best_threshold).astype(int)
            ps.append(precision_score(y_true_fold, y_pred_fold, pos_label=1, zero_division=0))
            rs.append(recall_score(y_true_fold, y_pred_fold, pos_label=1, zero_division=0))
            fs.append(f1_score(y_true_fold, y_pred_fold, pos_label=1, zero_division=0))

        result = {
            "precision_mean": float(np.mean(ps)),
            "precision_std": float(np.std(ps)),
            "recall_mean": float(np.mean(rs)),
            "recall_std": float(np.std(rs)),
            "f1_mean": float(np.mean(fs)),
            "f1_std": float(np.std(fs)),
            "fold_scores": [float(x) for x in fs],
            "support_mean": float(n_pos / n_splits),
            "fit_seconds_mean": float(np.mean(fold_times)),
            "n_total": int(len(X)),
            "n_positives": n_pos,
            # OOF threshold tuning results
            "best_threshold": float(best_threshold),
            "oof_f1_at_best_threshold": oof_f1_at_best,
            "oof_precision_at_best_threshold": oof_p_at_best,
            "oof_recall_at_best_threshold": oof_r_at_best,
        }
        per_topic[str(topic)] = result

        if verbose:
            print(
                f"  topic '{topic}': F1={result['f1_mean']:.4f} ± {result['f1_std']:.4f}, "
                f"P={result['precision_mean']:.3f}, R={result['recall_mean']:.3f}, "
                f"τ={best_threshold:.4f}, oof_F1@τ={oof_f1_at_best:.4f}, "
                f"n_pos={n_pos}, n_total={len(X)}",
                flush=True,
            )

    return _aggregate_cv_results(per_topic, topics, n_train_ref, n_splits, verbose)


# ============================================================
# LEGACY — multiclass API (backward compat, kept for old exps)
# ============================================================

def cv_evaluate_binary_per_topic(
    model_factory: Callable,
    X: pd.Series,
    y: pd.Series,
    topics: Optional[List] = None,
    n_splits: int = 5,
    verbose: bool = True,
    return_oof: bool = False,
) -> Dict:
    """
    DEPRECATED: multiclass-derived binary CV. Kept for backward compatibility.

    v1 approach: derives y_bin = (y_multiclass == topic). Loses ~30% positives
    for topics present in multi-label rows (e.g. НБП: 160 binary → 112 multiclass).

    For new experiments, use cv_evaluate_binary_per_topic_loader instead:
        cv_evaluate_binary_per_topic_loader(factory, load_train_binary, topics)

    Returns same schema as cv_evaluate_binary_per_topic_loader.
    """
    if topics is None:
        topics = list(ALLOWED_TOPICS)

    per_topic: Dict[str, Dict] = {}

    for topic in topics:
        y_bin = (y == topic).astype(int)
        result = _run_cv_for_topic(model_factory, X, y_bin, topic, n_splits, verbose, return_oof)
        per_topic[str(topic)] = result

    return _aggregate_cv_results(per_topic, topics, len(X), n_splits, verbose)


def evaluate_on_test_per_topic(
    fitted_pipelines: Dict[str, object],
    X_test: pd.Series,
    y_test: pd.Series,
) -> Dict:
    """
    DEPRECATED: multiclass-derived test eval. Kept for backward compatibility.

    v1 approach: y_bin = (y_test == topic) — uses only ~973 rows (excludes rows
    with excluded primary topic). For new code, use evaluate_on_test_per_topic_loader.

    Per-topic binary test eval. ОДНОКРАТНО на финальной mixture.
    """
    per_topic = {}

    for topic, pipe in fitted_pipelines.items():
        y_bin = (y_test == topic).astype(int)
        n_pos = int(y_bin.sum())
        y_pred = pipe.predict(X_test)
        per_topic[str(topic)] = {
            "precision": float(precision_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "support": n_pos,
            "n_test": int(len(X_test)),
            "y_pred": y_pred.tolist(),  # для bootstrap CI
        }

    valid_topics = list(per_topic.keys())
    macro_p = float(np.mean([per_topic[t]["precision"] for t in valid_topics])) if valid_topics else 0
    macro_r = float(np.mean([per_topic[t]["recall"] for t in valid_topics])) if valid_topics else 0
    macro_f = float(np.mean([per_topic[t]["f1"] for t in valid_topics])) if valid_topics else 0

    return {
        "per_topic": per_topic,
        "macro_avg_precision": macro_p,
        "macro_avg_recall": macro_r,
        "macro_avg_f1": macro_f,
        "n_test": int(len(X_test)),
    }
