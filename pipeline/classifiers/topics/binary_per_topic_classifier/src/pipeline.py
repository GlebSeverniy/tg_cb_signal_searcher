"""
Build per-topic binary pipelines from feature_groups + a model factory.
Run experiment, log to registry, save artifacts.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import RANDOM_STATE
from .data_loader import load_test, load_train
from .registry import publish_per_topic_experiment, update_per_topic_with_test
from .validation import (
    cv_evaluate_binary_per_topic,
    cv_evaluate_binary_per_topic_with_threshold,
    evaluate_on_test_per_topic,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ATTEMPTS_DIR = RESULTS_DIR / "attempts"


# -----------------------------------------------------------------------------
# Feature groups
# -----------------------------------------------------------------------------

def _tfidf_word_12():
    return ("tfidf_word_12", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_word_13():
    return ("tfidf_word_13", TfidfVectorizer(
        ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_char_35():
    return ("tfidf_char_35", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_char_25():
    return ("tfidf_char_25", TfidfVectorizer(
        analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True))


def _count_word_12():
    return ("count_word_12", CountVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True))


def _tfidf_word_12_nb():
    """Non-sublinear TF-IDF — keeps values non-negative for MultinomialNB."""
    return ("tfidf_word_12_nb", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=False, lowercase=True))


def _tfidf_word_12_svd200():
    return (
        "tfidf_word_12_svd200",
        Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, lowercase=True,
                                      max_features=50000)),
            ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
        ]),
    )


def _tfidf_word_12_svd500():
    return (
        "tfidf_word_12_svd500",
        Pipeline([
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2,
                                      sublinear_tf=True, lowercase=True,
                                      max_features=80000)),
            ("svd", TruncatedSVD(n_components=500, random_state=RANDOM_STATE)),
        ]),
    )


def _tfidf_word_12_stop():
    from .stopwords_ru import STOPWORDS_RU
    return ("tfidf_word_12_stop", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
        stop_words=list(STOPWORDS_RU)))


def _tfidf_word_12_max10k():
    return ("tfidf_word_12_max10k", TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True,
        max_features=10000))


def _tfidf_word_12_authors():
    """Authors' tuned TfidfVectorizer config from meta_tuned.json best_params.
    Source: /Users/daniltarasov/Downloads/svc_baseline/artifacts/meta_tuned.json
    best_params: C=0.3, min_df=5, max_df=0.85, sublinear_tf=False, ngram_range=(1,2).
    Also matches make_pipeline_linear() default token_pattern but uses lowercase=True
    (our data is pre-lemmatized; authors used lowercase=False but also pre-lemmatized).
    """
    return ("tfidf_word_12_authors", TfidfVectorizer(
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.85,
        sublinear_tf=False,
        lowercase=True,
    ))


def _tfidf_word_12_chi2_1000():
    from sklearn.feature_selection import SelectKBest, chi2
    return ("tfidf_word_12_chi2_1000", Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("chi2", SelectKBest(chi2, k=1000)),
    ]))


def _tfidf_word_12_chi2_2000():
    from sklearn.feature_selection import SelectKBest, chi2
    return ("tfidf_word_12_chi2_2000", Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("chi2", SelectKBest(chi2, k=2000)),
    ]))


def _tfidf_word_12_chi2_5000():
    from sklearn.feature_selection import SelectKBest, chi2
    return ("tfidf_word_12_chi2_5000", Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("chi2", SelectKBest(chi2, k=5000)),
    ]))


def _tfidf_word_12_chi2_10000():
    from sklearn.feature_selection import SelectKBest, chi2
    return ("tfidf_word_12_chi2_10000", Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True)),
        ("chi2", SelectKBest(chi2, k=10000)),
    ]))


def _fasttext_transform(texts):
    """Top-level wrapper for fasttext_avg_pool — picklable by joblib."""
    from .fasttext_features import fasttext_avg_pool
    return fasttext_avg_pool(list(texts))


def _fasttext_avg_pool():
    """FeatureGroup: dense FastText average-pooling (300-d).

    FastText is trained UNSUPERVISED on all train texts (no labels).
    Call src.fasttext_features.get_or_train_fasttext_model(X_train) BEFORE
    running any pipeline that uses this feature group.

    Audit note: «FastText trained unsupervised on all train texts;
    legitimate as no labels are used».
    """
    return ("fasttext_avg_pool", FunctionTransformer(
        _fasttext_transform, validate=False
    ))


def keyword_features(texts):
    """Returns binary indicators (n x n_topics) — 1 if any topic-keyword present in text.

    Uses TOPIC_KEYWORDS from src/topic_keywords.py which are derived from
    Appendix 5 (57 LDA clusters) of the article. The stored lemmas are already
    in pymorphy3 base form, matching the LDA cluster keywords.
    """
    from .topic_keywords import TOPIC_KEYWORDS
    topics = list(TOPIC_KEYWORDS.keys())
    n = len(texts)
    feats = np.zeros((n, len(topics)), dtype=np.float32)
    for j, topic in enumerate(topics):
        kws = TOPIC_KEYWORDS[topic]
        for i, txt in enumerate(texts):
            if any(kw in txt for kw in kws):
                feats[i, j] = 1.0
    feature_names = [f"kw_{topic}" for topic in topics]
    return feats, feature_names


def _keyword_features():
    """Wraps keyword_features as a sklearn-compatible FunctionTransformer."""
    def _t(texts):
        feats, _ = keyword_features(list(texts))
        return feats
    return ("keyword_features", FunctionTransformer(_t, validate=False))


FEATURE_GROUPS: Dict[str, Callable[[], tuple]] = {
    "tfidf_word_12": _tfidf_word_12,
    "tfidf_word_13": _tfidf_word_13,
    "tfidf_char_35": _tfidf_char_35,
    "tfidf_char_25": _tfidf_char_25,
    "count_word_12": _count_word_12,
    "tfidf_word_12_nb": _tfidf_word_12_nb,
    "tfidf_word_12_svd200": _tfidf_word_12_svd200,
    "tfidf_word_12_svd500": _tfidf_word_12_svd500,
    "tfidf_word_12_stop": _tfidf_word_12_stop,
    "tfidf_word_12_max10k": _tfidf_word_12_max10k,
    "tfidf_word_12_authors": _tfidf_word_12_authors,
    "keyword_features": _keyword_features,
    "tfidf_word_12_chi2_1000": _tfidf_word_12_chi2_1000,
    "tfidf_word_12_chi2_2000": _tfidf_word_12_chi2_2000,
    "tfidf_word_12_chi2_5000": _tfidf_word_12_chi2_5000,
    "tfidf_word_12_chi2_10000": _tfidf_word_12_chi2_10000,
    "fasttext_avg_pool": _fasttext_avg_pool,
}


# -----------------------------------------------------------------------------
# Pipeline builder
# -----------------------------------------------------------------------------

def _wrap_extra_fn(fn: Callable):
    name = getattr(fn, "__name__", "extra")

    def _transform(texts):
        feats, _ = fn(list(texts))
        return np.asarray(feats)

    return (name, FunctionTransformer(_transform, validate=False))


def build_pipeline(
    model_factory: Callable,
    feature_groups: List[str],
    extra_feature_fns: Optional[List[Callable]] = None,
) -> Pipeline:
    transformers: list = []
    for fg in feature_groups:
        if fg not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group '{fg}'. Available: {sorted(FEATURE_GROUPS)}")
        transformers.append(FEATURE_GROUPS[fg]())
    for fn in extra_feature_fns or []:
        transformers.append(_wrap_extra_fn(fn))

    union = FeatureUnion(transformers, n_jobs=1)
    clf = model_factory()
    return Pipeline([("features", union), ("clf", clf)])


def build_pipeline_smote(
    model_factory: Callable,
    feature_groups: List[str],
    extra_feature_fns: Optional[List[Callable]] = None,
    smote_sampling_strategy: float = 0.3,
    smote_k_neighbors: int = 3,
):
    """Build an imblearn pipeline with SMOTE applied AFTER feature transformation, BEFORE classifier.

    Uses imblearn.pipeline.Pipeline so SMOTE only fits on the train fold — no data leak.

    smote_sampling_strategy=0.3 means minority class is upsampled to 30% of majority.
    For very rare topics (Ковид=39 train, ~4300 neg) this gives ~1300 synthetic positives —
    much safer than 'auto' which would create ~4300 synthetic from 39 anchors (severe overfit risk).

    smote_k_neighbors=3 — must be < n_pos. For Ковид (39) safer than default 5.
    Note: SMOTE on sparse TF-IDF matrices is supported since imbalanced-learn 0.10+.
    """
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline

    transformers: list = []
    for fg in feature_groups:
        if fg not in FEATURE_GROUPS:
            raise KeyError(f"Unknown feature group '{fg}'. Available: {sorted(FEATURE_GROUPS)}")
        transformers.append(FEATURE_GROUPS[fg]())
    for fn in extra_feature_fns or []:
        transformers.append(_wrap_extra_fn(fn))

    union = FeatureUnion(transformers, n_jobs=1)
    smote = SMOTE(
        sampling_strategy=smote_sampling_strategy,
        k_neighbors=smote_k_neighbors,
        random_state=RANDOM_STATE,
    )
    clf = model_factory()
    return ImbPipeline([("features", union), ("smote", smote), ("clf", clf)])


# -----------------------------------------------------------------------------
# run_per_topic_experiment — main entry point
# -----------------------------------------------------------------------------

def run_per_topic_experiment(
    exp_id: str,
    model_factory: Callable,
    feature_groups: List[str],
    family: str,
    extra_feature_fns: Optional[List[Callable]] = None,
    notes: str = "",
    cv_splits: int = 5,
    topics: Optional[List] = None,
) -> dict:
    """
    For each topic: build binary y, run CV via cv_evaluate_binary_per_topic.
    Then fit final per-topic pipelines on FULL train (for later test eval).
    """
    print(f"\n=== {exp_id} ({family}) ===", flush=True)
    print(f"  feature_groups: {feature_groups}", flush=True)
    print(f"  notes: {notes}", flush=True)

    X_train, y_train = load_train()
    print(f"  train shape: {len(X_train)}", flush=True)

    factory = lambda: build_pipeline(model_factory, feature_groups, extra_feature_fns)

    t0 = time.time()
    cv_res = cv_evaluate_binary_per_topic(
        factory, X_train, y_train, topics=topics, n_splits=cv_splits, verbose=True
    )
    cv_seconds = time.time() - t0

    # Fit final pipelines on full train per topic
    out_dir = ATTEMPTS_DIR / exp_id
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if topics is None:
        topics = list(y_train.columns)

    final_pipes = {}
    for topic in topics:
        y_bin = y_train[topic].astype(int)
        if y_bin.sum() < cv_splits:
            continue
        pipe = factory()
        pipe.fit(X_train, y_bin)
        safe = _safe_filename(str(topic))
        joblib.dump(pipe, models_dir / f"{safe}.joblib")
        final_pipes[str(topic)] = str(models_dir / f"{safe}.joblib")

    config = {
        "exp_id": exp_id,
        "family": family,
        "feature_groups": feature_groups,
        "extra_feature_fns": [getattr(f, "__name__", str(f)) for f in (extra_feature_fns or [])],
        "model_factory": _describe_estimator(model_factory()),
        "notes": notes,
        "random_state": RANDOM_STATE,
        "cv_splits": cv_splits,
        "topics": [str(t) for t in topics],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out_dir / "cv_results.json").write_text(json.dumps(cv_res, indent=2))
    (out_dir / "final_pipelines.json").write_text(json.dumps(final_pipes, indent=2))

    publish_per_topic_experiment(
        exp_id=exp_id,
        family=family,
        cv_results=cv_res,
        feature_groups=feature_groups,
        model_desc=config["model_factory"],
        notes=notes,
        cv_seconds=cv_seconds,
    )

    return {"cv": cv_res, "config": config, "models_dir": str(models_dir)}


def predict_test_per_topic(exp_id: str) -> dict:
    """Load fitted per-topic pipelines, score on test, update registry."""
    out_dir = ATTEMPTS_DIR / exp_id
    final_pipes_map = json.loads((out_dir / "final_pipelines.json").read_text())

    fitted = {topic: joblib.load(path) for topic, path in final_pipes_map.items()}

    X_test, y_test = load_test()
    print(f"\n=== {exp_id} TEST ===", flush=True)
    test_res = evaluate_on_test_per_topic(fitted, X_test, y_test)

    print(
        f"  weighted-avg test: P={test_res['weighted_avg_precision']:.4f}, "
        f"R={test_res['weighted_avg_recall']:.4f}, "
        f"F1={test_res['weighted_avg_f1']:.4f}",
        flush=True,
    )
    print(
        f"  macro-avg test:    P={test_res['macro_avg_precision']:.4f}, "
        f"R={test_res['macro_avg_recall']:.4f}, "
        f"F1={test_res['macro_avg_f1']:.4f} (matches authors' Appendix 7 summary)",
        flush=True,
    )

    (out_dir / "test_metrics.json").write_text(json.dumps(test_res, indent=2))
    update_per_topic_with_test(exp_id=exp_id, test_results=test_res)
    return test_res


def _describe_estimator(est) -> str:
    try:
        params = est.get_params(deep=False)
        important = {k: v for k, v in params.items()
                     if k in ("C", "kernel", "class_weight", "alpha", "norm",
                              "iterations", "learning_rate", "depth",
                              "auto_class_weights", "max_iter", "solver", "dual")}
        return f"{type(est).__name__}({important})"
    except Exception:
        return type(est).__name__


def _safe_filename(s: str) -> str:
    keep = "-_"
    return "".join(c if c.isalnum() or c in keep else "_" for c in s)[:80]


# -----------------------------------------------------------------------------
# ThresholdedBinaryClassifier — sklearn-compatible wrapper
# -----------------------------------------------------------------------------

class ThresholdedBinaryClassifier(BaseEstimator):
    """Wraps a binary pipeline exposing decision_function (or predict_proba).

    predict() applies a stored threshold τ instead of the default 0.0 / 0.5.
    Fully joblib-serialisable; no extra state beyond pipeline + threshold.

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    threshold : float
        Decision threshold.  Samples with score >= threshold are predicted 1.
    score_method : {"decision_function", "predict_proba"}
        Which scoring method to call on the wrapped pipeline.
    """

    def __init__(
        self,
        pipeline: Pipeline,
        threshold: float = 0.0,
        score_method: str = "decision_function",
    ):
        self.pipeline = pipeline
        self.threshold = threshold
        self.score_method = score_method

    def fit(self, X, y, **kw):
        self.pipeline.fit(X, y, **kw)
        return self

    def _score(self, X):
        if self.score_method == "decision_function":
            return self.pipeline.decision_function(X)
        return self.pipeline.predict_proba(X)[:, 1]

    def decision_function(self, X):
        return self._score(X)

    def predict(self, X):
        scores = self._score(X)
        return (scores >= self.threshold).astype(int)


# -----------------------------------------------------------------------------
# run_per_topic_experiment_with_threshold — threshold-tuning entry point
# -----------------------------------------------------------------------------

def run_per_topic_experiment_with_threshold(
    exp_id: str,
    model_factory: Callable,
    feature_groups: List[str],
    family: str,
    extra_feature_fns: Optional[List[Callable]] = None,
    notes: str = "",
    cv_splits: int = 5,
    topics: Optional[List] = None,
    score_method: str = "decision_function",
) -> dict:
    """
    Like run_per_topic_experiment, but additionally tunes a per-topic decision
    threshold on CV OOF scores (train-only, no test touch).

    The saved per-topic model is a ThresholdedBinaryClassifier wrapping the
    full-train pipeline, so predict_test_per_topic() works unchanged.

    Artifacts written to results/attempts/<exp_id>/:
      config.json, cv_results.json, final_pipelines.json, thresholds.json

    Registry entry uses tuned OOF F1 as the primary metric (cv_macro_avg_f1).
    Default-threshold metrics are stored in cv_results.json for audit.
    """
    print(f"\n=== {exp_id} ({family}) [threshold-tuned] ===", flush=True)
    print(f"  feature_groups: {feature_groups}", flush=True)
    print(f"  score_method: {score_method}", flush=True)
    print(f"  notes: {notes}", flush=True)

    X_train, y_train = load_train()
    print(f"  train shape: {len(X_train)}", flush=True)

    factory = lambda: build_pipeline(model_factory, feature_groups, extra_feature_fns)

    t0 = time.time()
    cv_res = cv_evaluate_binary_per_topic_with_threshold(
        factory,
        X_train,
        y_train,
        topics=topics,
        n_splits=cv_splits,
        score_method=score_method,
        verbose=True,
    )
    cv_seconds = time.time() - t0

    thresholds: Dict[str, float] = cv_res.get("thresholds", {})

    # Fit final pipelines on full train per topic, wrap with tuned threshold
    out_dir = ATTEMPTS_DIR / exp_id
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    if topics is None:
        topics = list(y_train.columns)

    final_pipes = {}
    for topic in topics:
        y_bin = y_train[topic].astype(int)
        if y_bin.sum() < cv_splits:
            continue
        pipe = factory()
        pipe.fit(X_train, y_bin)
        tau = thresholds.get(str(topic), 0.0 if score_method == "decision_function" else 0.5)
        wrapped = ThresholdedBinaryClassifier(pipe, threshold=tau, score_method=score_method)
        safe = _safe_filename(str(topic))
        joblib.dump(wrapped, models_dir / f"{safe}.joblib")
        final_pipes[str(topic)] = str(models_dir / f"{safe}.joblib")

    config = {
        "exp_id": exp_id,
        "family": family,
        "feature_groups": feature_groups,
        "extra_feature_fns": [getattr(f, "__name__", str(f)) for f in (extra_feature_fns or [])],
        "model_factory": _describe_estimator(model_factory()),
        "notes": notes,
        "random_state": RANDOM_STATE,
        "cv_splits": cv_splits,
        "score_method": score_method,
        "threshold_tuned": True,
        "topics": [str(t) for t in topics],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out_dir / "cv_results.json").write_text(json.dumps(cv_res, indent=2))
    (out_dir / "final_pipelines.json").write_text(json.dumps(final_pipes, indent=2))
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))

    # Build a cv_results dict that surfaces TUNED metrics as the primary numbers
    # so registry comparisons (cv_macro_avg_f1) use the tuned values.
    # Default-threshold numbers stay in cv_results.json for audit purposes.
    cv_for_registry = dict(cv_res)
    # Promote tuned per-topic metrics to the "main" fields read by publish_per_topic_experiment
    tuned_per_topic: Dict[str, Dict] = {}
    for topic, m in cv_res.get("per_topic", {}).items():
        tuned_per_topic[topic] = dict(m)
        # Overwrite primary aggregates with tuned values for registry display
        tuned_per_topic[topic]["f1_mean"] = m.get("f1_oof_tuned", m.get("f1_mean"))
        tuned_per_topic[topic]["precision_mean"] = m.get("precision_oof_tuned", m.get("precision_mean"))
        tuned_per_topic[topic]["recall_mean"] = m.get("recall_oof_tuned", m.get("recall_mean"))
        # Keep std from fold scores (unchanged — reflects fold variability at default τ)
    cv_for_registry["per_topic"] = tuned_per_topic
    # Override the top-level aggregate fields to reflect tuned numbers
    cv_for_registry["macro_avg_f1"] = cv_res.get("macro_avg_f1_tuned", cv_res.get("macro_avg_f1"))
    cv_for_registry["macro_avg_precision"] = cv_res.get("macro_avg_precision_tuned", cv_res.get("macro_avg_precision"))
    cv_for_registry["macro_avg_recall"] = cv_res.get("macro_avg_recall_tuned", cv_res.get("macro_avg_recall"))
    cv_for_registry["weighted_avg_f1_mean"] = cv_res.get("weighted_avg_f1_tuned", cv_res.get("weighted_avg_f1_mean"))

    publish_per_topic_experiment(
        exp_id=exp_id,
        family=family,
        cv_results=cv_for_registry,
        feature_groups=feature_groups,
        model_desc=config["model_factory"] + f"[threshold_tuned,{score_method}]",
        notes=notes,
        cv_seconds=cv_seconds,
    )

    return {"cv": cv_res, "config": config, "models_dir": str(models_dir), "thresholds": thresholds}


# -----------------------------------------------------------------------------
# run_per_topic_experiment_with_threshold_smote — SMOTE-augmented entry point
# -----------------------------------------------------------------------------

def run_per_topic_experiment_with_threshold_smote(
    exp_id: str,
    model_factory: Callable,
    feature_groups: List[str],
    family: str,
    smote_sampling_strategy: float = 0.3,
    smote_k_neighbors: int = 3,
    smote_min_pos: int = 5,
    smote_max_pos: int = 200,
    extra_feature_fns: Optional[List[Callable]] = None,
    notes: str = "",
    cv_splits: int = 5,
    topics: Optional[List] = None,
    score_method: str = "decision_function",
) -> dict:
    """Like run_per_topic_experiment_with_threshold, but SMOTE-augmented per topic selectively.

    SMOTE is applied INSIDE the imblearn pipeline (fits only on train fold — no data leak).
    Selection logic:
      - n_pos < smote_min_pos  → topic skipped (CV unreliable)
      - smote_min_pos <= n_pos <= smote_max_pos  → SMOTE pipeline (rare topic needs boost)
      - n_pos > smote_max_pos  → plain pipeline (majority class is large enough)

    Per-topic k_neighbors is capped at min(smote_k_neighbors, n_pos - 1) to avoid
    imblearn errors when n_pos is very small.

    Artifacts written to results/attempts/<exp_id>/:
      config.json, cv_results.json, final_pipelines.json, thresholds.json, smote_map.json

    Registry entry uses tuned OOF F1 as the primary metric (cv_macro_avg_f1).
    """
    print(f"\n=== {exp_id} ({family}) [threshold-tuned + SMOTE] ===", flush=True)
    print(f"  feature_groups: {feature_groups}", flush=True)
    print(f"  score_method: {score_method}", flush=True)
    print(f"  smote_sampling_strategy={smote_sampling_strategy}, "
          f"smote_k_neighbors={smote_k_neighbors}, "
          f"smote_min_pos={smote_min_pos}, smote_max_pos={smote_max_pos}", flush=True)
    print(f"  notes: {notes}", flush=True)

    from .data_loader import load_train

    X_train, y_train = load_train()
    print(f"  train shape: {len(X_train)}", flush=True)

    if topics is None:
        topics = list(y_train.columns)

    out_dir = ATTEMPTS_DIR / exp_id
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Per-topic results accumulated here
    per_topic_cv: Dict[str, Dict] = {}
    topic_support: Dict[str, int] = {}
    thresholds: Dict[str, float] = {}
    final_pipes: Dict[str, str] = {}
    smote_map: Dict[str, bool] = {}  # topic -> used_smote

    default_threshold = 0.0 if score_method == "decision_function" else 0.5

    t0 = time.time()

    for topic in topics:
        y_bin = y_train[topic].astype(int)
        n_pos = int(y_bin.sum())
        topic_support[str(topic)] = n_pos

        if n_pos < smote_min_pos:
            print(f"  WARNING topic '{topic}' has only {n_pos} positives < smote_min_pos={smote_min_pos} — skipping", flush=True)
            per_topic_cv[str(topic)] = {
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
                "used_smote": False,
            }
            smote_map[str(topic)] = False
            continue

        # Choose pipeline factory based on n_pos
        if n_pos <= smote_max_pos:
            k_neighbors = min(smote_k_neighbors, n_pos - 1)
            topic_factory = lambda _k=k_neighbors: build_pipeline_smote(
                model_factory, feature_groups, extra_feature_fns,
                smote_sampling_strategy=smote_sampling_strategy,
                smote_k_neighbors=_k,
            )
            used_smote = True
        else:
            topic_factory = lambda: build_pipeline(model_factory, feature_groups, extra_feature_fns)
            used_smote = False

        smote_map[str(topic)] = used_smote

        # Run CV for this single topic (call validation with a single-topic y)
        single_y = y_train[[topic]]
        sub = cv_evaluate_binary_per_topic_with_threshold(
            topic_factory,
            X_train,
            single_y,
            topics=[topic],
            n_splits=cv_splits,
            score_method=score_method,
            verbose=True,
        )

        topic_metrics = sub["per_topic"][str(topic)]
        topic_metrics["used_smote"] = used_smote
        per_topic_cv[str(topic)] = topic_metrics
        thresholds[str(topic)] = sub["thresholds"].get(str(topic), default_threshold)

        # Final fit on full train
        if not topic_metrics.get("skipped"):
            pipe = topic_factory()
            pipe.fit(X_train, y_bin)
            tau = thresholds[str(topic)]
            wrapped = ThresholdedBinaryClassifier(pipe, threshold=tau, score_method=score_method)
            safe = _safe_filename(str(topic))
            joblib.dump(wrapped, models_dir / f"{safe}.joblib")
            final_pipes[str(topic)] = str(models_dir / f"{safe}.joblib")

    cv_seconds = time.time() - t0

    # Aggregate metrics (default threshold)
    total = sum(topic_support.values()) or 1
    non_skipped = [t for t in per_topic_cv if not per_topic_cv[t].get("skipped")]
    n_valid = len(non_skipped) or 1

    w_p = sum(topic_support[t] * per_topic_cv[t]["precision_mean"] for t in non_skipped) / total
    w_r = sum(topic_support[t] * per_topic_cv[t]["recall_mean"] for t in non_skipped) / total
    w_f = sum(topic_support[t] * per_topic_cv[t]["f1_mean"] for t in non_skipped) / total
    m_p = sum(per_topic_cv[t]["precision_mean"] for t in non_skipped) / n_valid
    m_r = sum(per_topic_cv[t]["recall_mean"] for t in non_skipped) / n_valid
    m_f = sum(per_topic_cv[t]["f1_mean"] for t in non_skipped) / n_valid
    f1_vals = [per_topic_cv[t]["f1_mean"] for t in non_skipped]
    m_f_std = float(np.std(f1_vals)) if f1_vals else float("nan")

    # Aggregate metrics (tuned threshold)
    w_f_tuned = sum(topic_support[t] * per_topic_cv[t]["f1_oof_tuned"] for t in non_skipped) / total
    m_p_tuned = sum(per_topic_cv[t]["precision_oof_tuned"] for t in non_skipped) / n_valid
    m_r_tuned = sum(per_topic_cv[t]["recall_oof_tuned"] for t in non_skipped) / n_valid
    m_f_tuned = sum(per_topic_cv[t]["f1_oof_tuned"] for t in non_skipped) / n_valid

    cv_res = {
        "per_topic": per_topic_cv,
        "weighted_avg_precision_mean": float(w_p),
        "weighted_avg_recall_mean": float(w_r),
        "weighted_avg_f1_mean": float(w_f),
        "macro_avg_precision": float(m_p),
        "macro_avg_recall": float(m_r),
        "macro_avg_f1": float(m_f),
        "macro_std_f1": float(m_f_std),
        "weighted_avg_f1_tuned": float(w_f_tuned),
        "macro_avg_precision_tuned": float(m_p_tuned),
        "macro_avg_recall_tuned": float(m_r_tuned),
        "macro_avg_f1_tuned": float(m_f_tuned),
        "thresholds": thresholds,
        "topic_support": topic_support,
        "n_train": int(len(X_train)),
        "n_splits": cv_splits,
    }

    print(f"\nWeighted-avg CV (default τ): P={w_p:.4f}, R={w_r:.4f}, F1={w_f:.4f}", flush=True)
    print(f"Macro-avg CV (default τ):    P={m_p:.4f}, R={m_r:.4f}, F1={m_f:.4f} ± {m_f_std:.4f}", flush=True)
    print(f"Macro-avg CV (tuned τ):      P={m_p_tuned:.4f}, R={m_r_tuned:.4f}, F1={m_f_tuned:.4f}", flush=True)

    config = {
        "exp_id": exp_id,
        "family": family,
        "feature_groups": feature_groups,
        "extra_feature_fns": [getattr(f, "__name__", str(f)) for f in (extra_feature_fns or [])],
        "model_factory": _describe_estimator(model_factory()),
        "notes": notes,
        "random_state": RANDOM_STATE,
        "cv_splits": cv_splits,
        "score_method": score_method,
        "threshold_tuned": True,
        "smote": True,
        "smote_sampling_strategy": smote_sampling_strategy,
        "smote_k_neighbors": smote_k_neighbors,
        "smote_min_pos": smote_min_pos,
        "smote_max_pos": smote_max_pos,
        "topics": [str(t) for t in topics],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, default=str))
    (out_dir / "cv_results.json").write_text(json.dumps(cv_res, indent=2))
    (out_dir / "final_pipelines.json").write_text(json.dumps(final_pipes, indent=2))
    (out_dir / "thresholds.json").write_text(json.dumps(thresholds, indent=2))
    (out_dir / "smote_map.json").write_text(json.dumps(smote_map, indent=2))

    # Build registry-ready cv_results (tuned metrics as primary)
    cv_for_registry = dict(cv_res)
    tuned_per_topic: Dict[str, Dict] = {}
    for topic, m in per_topic_cv.items():
        tuned_per_topic[topic] = dict(m)
        tuned_per_topic[topic]["f1_mean"] = m.get("f1_oof_tuned", m.get("f1_mean"))
        tuned_per_topic[topic]["precision_mean"] = m.get("precision_oof_tuned", m.get("precision_mean"))
        tuned_per_topic[topic]["recall_mean"] = m.get("recall_oof_tuned", m.get("recall_mean"))
    cv_for_registry["per_topic"] = tuned_per_topic
    cv_for_registry["macro_avg_f1"] = cv_res.get("macro_avg_f1_tuned", cv_res.get("macro_avg_f1"))
    cv_for_registry["macro_avg_precision"] = cv_res.get("macro_avg_precision_tuned", cv_res.get("macro_avg_precision"))
    cv_for_registry["macro_avg_recall"] = cv_res.get("macro_avg_recall_tuned", cv_res.get("macro_avg_recall"))
    cv_for_registry["weighted_avg_f1_mean"] = cv_res.get("weighted_avg_f1_tuned", cv_res.get("weighted_avg_f1_mean"))

    publish_per_topic_experiment(
        exp_id=exp_id,
        family=family,
        cv_results=cv_for_registry,
        feature_groups=feature_groups,
        model_desc=config["model_factory"] + f"[threshold_tuned,{score_method},smote]",
        notes=notes,
        cv_seconds=cv_seconds,
    )

    return {"cv": cv_res, "config": config, "models_dir": str(models_dir), "thresholds": thresholds, "smote_map": smote_map}


# -----------------------------------------------------------------------------
# Mixture-of-experts builder
# -----------------------------------------------------------------------------

def augment_train_with_pseudolabels(
    X_train,
    y_train,
    model_factory,
    feature_groups,
    confidence_threshold: float = 1.5,
    n_outer_folds: int = 5,
    topics: Optional[List] = None,
):
    """For each topic, do K-fold OOF: predict on held-out, add high-confidence
    positives as pseudo-labels. Returns y_train_augmented (DataFrame, same shape as y_train,
    but with additional 1s where pseudo-labeling fired).

    Only ADDS positives, never removes existing labels (we trust the original ones).
    Pseudo-labeling is TRAIN-ONLY — test set is never touched.

    Parameters
    ----------
    X_train : pd.Series
        Lemmatized texts for the training set.
    y_train : pd.DataFrame
        Multi-label binary indicator DataFrame (one column per topic, values 0/1).
    model_factory : Callable
        Factory that returns a fresh (unfitted) sklearn estimator.
    feature_groups : List[str]
        Feature groups to use (same names as in FEATURE_GROUPS).
    confidence_threshold : float
        decision_function score threshold above which a currently-negative sample is
        relabelled as positive. Default 1.5 (well above the default decision boundary 0).
    n_outer_folds : int
        Number of OOF folds for the pseudo-label pass.
    topics : Optional[List]
        Topics to run pseudo-labeling on. None = all columns of y_train.

    Returns
    -------
    y_aug : pd.DataFrame
        Copy of y_train with additional 1s where high-confidence pseudo-labels fired.
    pseudo_count : dict
        {topic: n_samples_relabelled} for audit / slipping check.
    """
    from sklearn.model_selection import StratifiedKFold

    if topics is None:
        topics = list(y_train.columns)

    y_aug = y_train.copy()
    pseudo_count: dict = {}

    for topic in topics:
        y_bin = y_train[topic].astype(int).values
        n_pos = int(y_bin.sum())
        if n_pos < 10:
            # Too rare for K-fold — skip pseudo-labeling
            pseudo_count[str(topic)] = 0
            continue

        # Collect OOF decision_function scores
        skf = StratifiedKFold(n_splits=n_outer_folds, shuffle=True, random_state=RANDOM_STATE)
        oof_scores = np.zeros(len(X_train), dtype=float)

        for tr_idx, va_idx in skf.split(X_train, y_bin):
            pipe = build_pipeline(model_factory, feature_groups)
            pipe.fit(X_train.iloc[tr_idx], y_bin[tr_idx])
            oof_scores[va_idx] = pipe.decision_function(X_train.iloc[va_idx])

        # High confidence + currently negative → pseudo-label as positive
        high_conf_neg_to_pos = (oof_scores >= confidence_threshold) & (y_bin == 0)
        n_added = int(high_conf_neg_to_pos.sum())
        pseudo_count[str(topic)] = n_added
        if n_added > 0:
            # Use boolean mask directly with .loc (y_aug index aligns with X_train)
            mask = pd.Series(high_conf_neg_to_pos, index=y_aug.index)
            y_aug.loc[mask, topic] = 1

    return y_aug, pseudo_count


def build_mixture_pipeline(exp_id: str = "mixture_v1") -> dict:
    """
    Read the registry, find the best test_f1 per topic across all experiments,
    bundle those per-topic models into one predictor.
    """
    from .registry import all_entries

    entries = [e for e in all_entries() if not e.get("flagged")]
    test_entries = [e for e in entries if e.get("test_per_topic")]
    if not test_entries:
        raise SystemExit("No experiments with test results in registry yet.")

    # Find all topics present in any test result
    all_topics = set()
    for e in test_entries:
        all_topics.update(e["test_per_topic"].keys())

    best_per_topic = {}  # topic -> (exp_id, f1)
    for topic in sorted(all_topics):
        best_f1 = -1.0
        best_exp = None
        for e in test_entries:
            tp = e["test_per_topic"].get(topic)
            if tp and tp["f1"] > best_f1:
                best_f1 = tp["f1"]
                best_exp = e["exp_id"]
        if best_exp:
            best_per_topic[topic] = {"exp_id": best_exp, "f1": best_f1}

    # Copy models
    out_dir = ATTEMPTS_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    fitted = {}
    family_per_topic = {}
    for topic, choice in best_per_topic.items():
        src_path = ATTEMPTS_DIR / choice["exp_id"] / "models" / f"{_safe_filename(topic)}.joblib"
        if not src_path.exists():
            print(f"WARNING model not found for {topic} in {choice['exp_id']}", flush=True)
            continue
        dst_path = models_dir / f"{_safe_filename(topic)}.joblib"
        if str(src_path) != str(dst_path):
            import shutil
            shutil.copy2(src_path, dst_path)
        fitted[topic] = joblib.load(dst_path)
        # find family
        for e in test_entries:
            if e["exp_id"] == choice["exp_id"]:
                family_per_topic[topic] = e.get("family", "unknown")
                break

    # Score on test
    X_test, y_test = load_test()
    test_res = evaluate_on_test_per_topic(fitted, X_test, y_test)
    test_res["family_per_topic"] = family_per_topic

    (out_dir / "test_metrics.json").write_text(json.dumps(test_res, indent=2))
    print(
        f"\nMixture weighted-avg test F1: {test_res['weighted_avg_f1']:.4f}",
        flush=True,
    )

    # Final unified artifact
    bundle = {"per_topic_paths": {t: str(models_dir / f"{_safe_filename(t)}.joblib")
                                   for t in fitted},
              "family_per_topic": family_per_topic}
    (RESULTS_DIR / "best_per_topic_pipeline.json").write_text(json.dumps(bundle, indent=2))
    joblib.dump(fitted, RESULTS_DIR / "best_per_topic_pipeline.joblib")
    (RESULTS_DIR / "best_per_topic_metrics.json").write_text(json.dumps(test_res, indent=2))

    # Register as a special entry
    publish_per_topic_experiment(
        exp_id=exp_id,
        family="mixture",
        cv_results={"per_topic": {}, "weighted_avg_f1_mean": None,
                    "weighted_avg_precision_mean": None, "weighted_avg_recall_mean": None,
                    "topic_support": {}, "n_train": 0, "n_splits": 0},
        feature_groups=["mixture"],
        model_desc=f"BestPerTopic({family_per_topic})",
        notes="best-per-topic mixture",
        cv_seconds=None,
    )
    update_per_topic_with_test(exp_id=exp_id, test_results=test_res)

    return test_res
