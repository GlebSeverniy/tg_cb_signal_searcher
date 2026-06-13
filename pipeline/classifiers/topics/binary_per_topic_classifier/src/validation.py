"""
Validation — single source of truth.

Per-topic CV: for each topic, build binary y (topic vs rest) and run StratifiedKFold(5).
"""

from __future__ import annotations

import time
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from . import RANDOM_STATE


def cv_evaluate_binary_per_topic(
    model_factory: Callable[[], Pipeline],
    X: pd.Series,
    y: pd.DataFrame,
    topics: Optional[List] = None,
    n_splits: int = 5,
    verbose: bool = True,
) -> Dict:
    """
    For each topic, evaluate binary (topic vs rest) classification via StratifiedKFold.

    y: pd.DataFrame with one column per canonical topic name, values 0/1.
       (produced by data_loader.load_train_binary())

    Returns:
      {
        'per_topic': {
            topic: {
                'precision_mean': ..., 'precision_std': ...,
                'recall_mean': ..., 'recall_std': ...,
                'f1_mean': ..., 'f1_std': ...,
                'fold_scores': [...],
                'support_mean': avg positives per fold,
                'fit_seconds_mean': ...,
            },
            ...
        },
        'weighted_avg_precision_mean': float,
        'weighted_avg_recall_mean': float,
        'weighted_avg_f1_mean': float,
        'topic_support': {topic: total positives in train},
        'n_train': int,
        'n_splits': int,
      }
    """
    if topics is None:
        topics = list(y.columns)

    per_topic: Dict[str, Dict] = {}
    topic_support: Dict[str, int] = {}

    for topic in topics:
        y_bin = y[topic].astype(int)
        n_pos = int(y_bin.sum())
        topic_support[str(topic)] = n_pos
        if n_pos < n_splits:
            print(f"  WARNING topic '{topic}' has only {n_pos} positives — skipping CV", flush=True)
            per_topic[str(topic)] = {
                "precision_mean": float("nan"),
                "recall_mean": float("nan"),
                "f1_mean": float("nan"),
                "f1_std": float("nan"),
                "fold_scores": [],
                "support_mean": float(n_pos),
                "fit_seconds_mean": 0.0,
                "skipped": True,
            }
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        ps, rs, fs, fits = [], [], [], []

        for fi, (tr_idx, va_idx) in enumerate(skf.split(X, y_bin)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y_bin.iloc[tr_idx], y_bin.iloc[va_idx]

            model = model_factory()  # fresh
            t0 = time.time()
            model.fit(X_tr, y_tr)
            fit_t = time.time() - t0
            fits.append(fit_t)

            y_pred = model.predict(X_va)
            ps.append(precision_score(y_va, y_pred, pos_label=1, zero_division=0))
            rs.append(recall_score(y_va, y_pred, pos_label=1, zero_division=0))
            fs.append(f1_score(y_va, y_pred, pos_label=1, zero_division=0))

        per_topic[str(topic)] = {
            "precision_mean": float(np.mean(ps)),
            "precision_std": float(np.std(ps)),
            "recall_mean": float(np.mean(rs)),
            "recall_std": float(np.std(rs)),
            "f1_mean": float(np.mean(fs)),
            "f1_std": float(np.std(fs)),
            "fold_scores": [float(x) for x in fs],
            "support_mean": float(n_pos / n_splits),
            "fit_seconds_mean": float(np.mean(fits)),
        }

        if verbose:
            print(
                f"  topic '{topic}': F1={per_topic[str(topic)]['f1_mean']:.4f} ± {per_topic[str(topic)]['f1_std']:.4f}, "
                f"P={per_topic[str(topic)]['precision_mean']:.3f}, "
                f"R={per_topic[str(topic)]['recall_mean']:.3f}, "
                f"n_pos={n_pos}",
                flush=True,
            )

    # Weighted avg by topic frequency on train
    total = sum(topic_support.values()) or 1
    w_p = sum(topic_support[t] * per_topic[t]["precision_mean"]
              for t in per_topic if not per_topic[t].get("skipped")) / total
    w_r = sum(topic_support[t] * per_topic[t]["recall_mean"]
              for t in per_topic if not per_topic[t].get("skipped")) / total
    w_f = sum(topic_support[t] * per_topic[t]["f1_mean"]
              for t in per_topic if not per_topic[t].get("skipped")) / total

    # Macro avg (simple mean across topics) — matches authors' Appendix 7 summary row
    non_skipped = [t for t in per_topic if not per_topic[t].get("skipped")]
    n_valid = len(non_skipped) or 1
    m_p = sum(per_topic[t]["precision_mean"] for t in non_skipped) / n_valid
    m_r = sum(per_topic[t]["recall_mean"] for t in non_skipped) / n_valid
    m_f = sum(per_topic[t]["f1_mean"] for t in non_skipped) / n_valid
    f1_vals = [per_topic[t]["f1_mean"] for t in non_skipped]
    m_f_std = float(np.std(f1_vals)) if f1_vals else float("nan")

    res = {
        "per_topic": per_topic,
        "weighted_avg_precision_mean": float(w_p),
        "weighted_avg_recall_mean": float(w_r),
        "weighted_avg_f1_mean": float(w_f),
        "macro_avg_precision": float(m_p),
        "macro_avg_recall": float(m_r),
        "macro_avg_f1": float(m_f),
        "macro_std_f1": float(m_f_std),
        "topic_support": topic_support,
        "n_train": int(len(X)),
        "n_splits": n_splits,
    }
    if verbose:
        print(
            f"\nWeighted-avg CV: P={w_p:.4f}, R={w_r:.4f}, F1={w_f:.4f}",
            flush=True,
        )
        print(
            f"Macro-avg CV: P={m_p:.4f}, R={m_r:.4f}, F1={m_f:.4f} ± {m_f_std:.4f} "
            f"(this matches authors' Appendix 7 summary)",
            flush=True,
        )
    return res


def cv_evaluate_binary_per_topic_with_threshold(
    model_factory: Callable[[], Pipeline],
    X: pd.Series,
    y: pd.DataFrame,
    topics: Optional[List] = None,
    n_splits: int = 5,
    score_method: str = "decision_function",
    threshold_grid: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Dict:
    """
    Per-topic binary CV with OOF threshold tuning.

    For each topic:
      1. Collect OOF decision scores across all folds (no data leak — scored on val fold only).
      2. Find τ* maximising F1 on OOF predictions.
      3. Report both default-threshold metrics and tuned metrics.

    score_method: "decision_function" (LinearSVC, SVC) or "predict_proba" (LogReg, CatBoost).
    threshold_grid: explicit grid of τ values to search; if None, uses 200 evenly-spaced points
                    between OOF score min and max.

    Returns dict with keys:
      per_topic: {
          topic: {
              precision_mean, recall_mean, f1_mean, f1_std, fold_scores,  # default τ=0
              f1_oof_tuned, precision_oof_tuned, recall_oof_tuned,         # OOF at τ*
              tuned_threshold,
              support_mean, fit_seconds_mean,
          }
      }
      macro_avg_f1, macro_avg_precision, macro_avg_recall,                 # default
      macro_avg_f1_tuned, macro_avg_precision_tuned, macro_avg_recall_tuned,
      weighted_avg_f1_mean, weighted_avg_precision_mean, weighted_avg_recall_mean,  # default
      weighted_avg_f1_tuned,
      thresholds: {topic: τ*},
      topic_support, n_train, n_splits,
    """
    if topics is None:
        topics = list(y.columns)

    per_topic: Dict[str, Dict] = {}
    topic_support: Dict[str, int] = {}

    default_threshold = 0.0 if score_method == "decision_function" else 0.5

    for topic in topics:
        y_bin = y[topic].astype(int)
        n_pos = int(y_bin.sum())
        topic_support[str(topic)] = n_pos
        if n_pos < n_splits:
            print(f"  WARNING topic '{topic}' has only {n_pos} positives — skipping CV", flush=True)
            per_topic[str(topic)] = {
                "precision_mean": float("nan"),
                "recall_mean": float("nan"),
                "f1_mean": float("nan"),
                "f1_std": float("nan"),
                "fold_scores": [],
                "f1_oof_tuned": float("nan"),
                "precision_oof_tuned": float("nan"),
                "recall_oof_tuned": float("nan"),
                "tuned_threshold": default_threshold,
                "support_mean": float(n_pos),
                "fit_seconds_mean": 0.0,
                "skipped": True,
            }
            continue

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        ps_def, rs_def, fs_def, fits = [], [], [], []

        # OOF accumulators
        oof_scores = np.empty(len(y_bin), dtype=float)
        oof_y = np.empty(len(y_bin), dtype=int)
        idx_arr = np.arange(len(y_bin))

        for fi, (tr_idx, va_idx) in enumerate(skf.split(X, y_bin)):
            X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
            y_tr, y_va = y_bin.iloc[tr_idx], y_bin.iloc[va_idx]

            model = model_factory()  # fresh
            t0 = time.time()
            model.fit(X_tr, y_tr)
            fit_t = time.time() - t0
            fits.append(fit_t)

            # Collect scores for OOF threshold search
            if score_method == "decision_function":
                scores_va = model.decision_function(X_va)
            else:
                scores_va = model.predict_proba(X_va)[:, 1]

            oof_scores[va_idx] = scores_va
            oof_y[va_idx] = y_va.values

            # Default-threshold fold metrics
            y_pred_def = (scores_va >= default_threshold).astype(int)
            ps_def.append(precision_score(y_va, y_pred_def, pos_label=1, zero_division=0))
            rs_def.append(recall_score(y_va, y_pred_def, pos_label=1, zero_division=0))
            fs_def.append(f1_score(y_va, y_pred_def, pos_label=1, zero_division=0))

        # Find optimal threshold on full OOF
        if threshold_grid is not None:
            grid = threshold_grid
        else:
            s_min, s_max = oof_scores.min(), oof_scores.max()
            grid = np.linspace(s_min, s_max, 200)

        best_tau = default_threshold
        best_f1_tuned = -1.0
        for tau in grid:
            y_pred_tau = (oof_scores >= tau).astype(int)
            f1_tau = f1_score(oof_y, y_pred_tau, pos_label=1, zero_division=0)
            if f1_tau > best_f1_tuned:
                best_f1_tuned = f1_tau
                best_tau = float(tau)

        y_pred_tuned = (oof_scores >= best_tau).astype(int)
        p_tuned = float(precision_score(oof_y, y_pred_tuned, pos_label=1, zero_division=0))
        r_tuned = float(recall_score(oof_y, y_pred_tuned, pos_label=1, zero_division=0))
        f1_tuned = float(f1_score(oof_y, y_pred_tuned, pos_label=1, zero_division=0))

        per_topic[str(topic)] = {
            "precision_mean": float(np.mean(ps_def)),
            "precision_std": float(np.std(ps_def)),
            "recall_mean": float(np.mean(rs_def)),
            "recall_std": float(np.std(rs_def)),
            "f1_mean": float(np.mean(fs_def)),
            "f1_std": float(np.std(fs_def)),
            "fold_scores": [float(x) for x in fs_def],
            "f1_oof_tuned": f1_tuned,
            "precision_oof_tuned": p_tuned,
            "recall_oof_tuned": r_tuned,
            "tuned_threshold": best_tau,
            "support_mean": float(n_pos / n_splits),
            "fit_seconds_mean": float(np.mean(fits)),
        }

        if verbose:
            f1_def_val = float(np.mean(fs_def))
            print(
                f"  topic '{topic}': f1_default={f1_def_val:.4f}, "
                f"f1_tuned={f1_tuned:.4f} (τ={best_tau:.4f}), n_pos={n_pos}",
                flush=True,
            )

    # Aggregate metrics — default threshold
    total = sum(topic_support.values()) or 1
    non_skipped = [t for t in per_topic if not per_topic[t].get("skipped")]
    n_valid = len(non_skipped) or 1

    w_p = sum(topic_support[t] * per_topic[t]["precision_mean"]
              for t in non_skipped) / total
    w_r = sum(topic_support[t] * per_topic[t]["recall_mean"]
              for t in non_skipped) / total
    w_f = sum(topic_support[t] * per_topic[t]["f1_mean"]
              for t in non_skipped) / total

    m_p = sum(per_topic[t]["precision_mean"] for t in non_skipped) / n_valid
    m_r = sum(per_topic[t]["recall_mean"] for t in non_skipped) / n_valid
    m_f = sum(per_topic[t]["f1_mean"] for t in non_skipped) / n_valid
    f1_vals = [per_topic[t]["f1_mean"] for t in non_skipped]
    m_f_std = float(np.std(f1_vals)) if f1_vals else float("nan")

    # Aggregate metrics — tuned threshold
    w_f_tuned = sum(topic_support[t] * per_topic[t]["f1_oof_tuned"]
                    for t in non_skipped) / total
    m_p_tuned = sum(per_topic[t]["precision_oof_tuned"] for t in non_skipped) / n_valid
    m_r_tuned = sum(per_topic[t]["recall_oof_tuned"] for t in non_skipped) / n_valid
    m_f_tuned = sum(per_topic[t]["f1_oof_tuned"] for t in non_skipped) / n_valid

    thresholds = {t: per_topic[t]["tuned_threshold"] for t in non_skipped}

    res = {
        "per_topic": per_topic,
        # default-threshold aggregates
        "weighted_avg_precision_mean": float(w_p),
        "weighted_avg_recall_mean": float(w_r),
        "weighted_avg_f1_mean": float(w_f),
        "macro_avg_precision": float(m_p),
        "macro_avg_recall": float(m_r),
        "macro_avg_f1": float(m_f),
        "macro_std_f1": float(m_f_std),
        # tuned-threshold aggregates
        "weighted_avg_f1_tuned": float(w_f_tuned),
        "macro_avg_precision_tuned": float(m_p_tuned),
        "macro_avg_recall_tuned": float(m_r_tuned),
        "macro_avg_f1_tuned": float(m_f_tuned),
        # per-topic thresholds
        "thresholds": thresholds,
        "topic_support": topic_support,
        "n_train": int(len(X)),
        "n_splits": n_splits,
    }
    if verbose:
        print(
            f"\nWeighted-avg CV (default τ): P={w_p:.4f}, R={w_r:.4f}, F1={w_f:.4f}",
            flush=True,
        )
        print(
            f"Macro-avg CV (default τ):    P={m_p:.4f}, R={m_r:.4f}, F1={m_f:.4f} ± {m_f_std:.4f}",
            flush=True,
        )
        print(
            f"Macro-avg CV (tuned τ):      P={m_p_tuned:.4f}, R={m_r_tuned:.4f}, F1={m_f_tuned:.4f}",
            flush=True,
        )
    return res


def evaluate_on_test_per_topic(
    fitted_pipelines: Dict[str, Pipeline],
    X_test: pd.Series,
    y_test: pd.DataFrame,
) -> Dict:
    """
    Evaluate per-topic binary classifiers on the held-out test set.

    fitted_pipelines: dict {topic: fitted Pipeline (binary)}
    y_test: pd.DataFrame with one column per canonical topic name, values 0/1.
    """
    per_topic = {}
    support = {}

    for topic, pipe in fitted_pipelines.items():
        y_bin = y_test[topic].astype(int)
        n_pos = int(y_bin.sum())
        support[str(topic)] = n_pos
        y_pred = pipe.predict(X_test)
        per_topic[str(topic)] = {
            "precision": float(precision_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "recall": float(recall_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "f1": float(f1_score(y_bin, y_pred, pos_label=1, zero_division=0)),
            "support": n_pos,
        }

    total = sum(support.values()) or 1
    w_p = sum(support[t] * per_topic[t]["precision"] for t in per_topic) / total
    w_r = sum(support[t] * per_topic[t]["recall"] for t in per_topic) / total
    w_f = sum(support[t] * per_topic[t]["f1"] for t in per_topic) / total

    # Macro avg (simple mean) — matches authors' Appendix 7 summary row
    non_skipped = [t for t in per_topic if not per_topic[t].get("skipped")]
    n_valid = len(non_skipped) or 1
    m_p = sum(per_topic[t]["precision"] for t in non_skipped) / n_valid
    m_r = sum(per_topic[t]["recall"] for t in non_skipped) / n_valid
    m_f = sum(per_topic[t]["f1"] for t in non_skipped) / n_valid
    f1_vals = [per_topic[t]["f1"] for t in non_skipped]
    m_f_std = float(np.std(f1_vals)) if f1_vals else float("nan")

    return {
        "per_topic": per_topic,
        "weighted_avg_precision": float(w_p),
        "weighted_avg_recall": float(w_r),
        "weighted_avg_f1": float(w_f),
        "macro_avg_precision": float(m_p),
        "macro_avg_recall": float(m_r),
        "macro_avg_f1": float(m_f),
        "macro_std_f1": float(m_f_std),
        "n_test": int(len(X_test)),
    }
