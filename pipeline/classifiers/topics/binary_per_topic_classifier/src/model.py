"""
Model factories. Five families allowed (see CLAUDE.md):
  LinearSVC / SVC(linear), LogisticRegression, MultinomialNB, ComplementNB, CatBoostClassifier.

Each factory returns an UNFITTED estimator ready for a Pipeline.
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC

from . import RANDOM_STATE


# ---------- SVC family ----------

def make_linear_svc(C: float = 1.0, class_weight=None, max_iter: int = 5000):
    return LinearSVC(
        C=C,
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
        dual="auto",
    )


def make_svc_linear(C: float = 1.0, class_weight=None):
    return SVC(
        kernel="linear",
        C=C,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
    )


def make_svc_rbf(C: float = 1.0, gamma="scale", class_weight=None):
    """SVC with RBF kernel — nonlinear boundary.

    Extension of SVC family (see CLAUDE.md: 'SVC family includes SVC(linear)').
    Experiments using this factory are marked as family='svc' and noted as
    'RBF extension of SVC family — nonlinear boundary experiment'.

    gamma='scale' (default sklearn): gamma = 1 / (n_features * X.var()).
    For sparse TF-IDF matrices this is computed from the dense variance estimate
    (sklearn handles sparse X.var() via mean_squared). May be slow (O(n^2)) —
    recommended only after smoke test confirms ≤ 1 min/topic on train size.
    """
    return SVC(
        kernel="rbf",
        C=C,
        gamma=gamma,
        class_weight=class_weight,
        random_state=RANDOM_STATE,
        decision_function_shape="ovr",
    )


# ---------- LogReg ----------

def make_logreg(C: float = 1.0, class_weight=None, solver: str = "liblinear", max_iter: int = 2000):
    return LogisticRegression(
        C=C,
        class_weight=class_weight,
        solver=solver,
        max_iter=max_iter,
        random_state=RANDOM_STATE,
    )


# ---------- Naive Bayes ----------

def make_multinomial_nb(alpha: float = 1.0):
    """MultinomialNB requires non-negative features (counts or non-sublinear TF-IDF)."""
    return MultinomialNB(alpha=alpha)


def make_complement_nb(alpha: float = 1.0, norm: bool = False):
    """ComplementNB — designed for imbalanced text classification."""
    return ComplementNB(alpha=alpha, norm=norm)


# ---------- CatBoost ----------

def make_catboost(
    iterations: int = 300,
    learning_rate: float = 0.1,
    depth: int = 6,
    auto_class_weights: str = "Balanced",
    verbose: int = 0,
):
    """
    CatBoostClassifier for binary tasks. Expects DENSE numeric input
    (use a feature_group with TruncatedSVD, e.g. 'tfidf_word_12_svd200').

    For native text features, see make_catboost_text below.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError(
            "catboost is not installed. Run: pip install catboost --break-system-packages"
        ) from e

    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        auto_class_weights=auto_class_weights,
        random_seed=RANDOM_STATE,
        verbose=verbose,
        allow_writing_files=False,
    )


def make_catboost_text(
    iterations: int = 500,
    learning_rate: float = 0.05,
    auto_class_weights: str = "Balanced",
    verbose: int = 0,
):
    """
    CatBoost using native text features. Requires text input as a 2D array
    of shape (n, 1) with the column declared as text via Pool.
    Use this through src.pipeline.run_per_topic_experiment with
    feature_groups=['catboost_text_passthrough'] which routes text directly.
    """
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError(
            "catboost is not installed. Run: pip install catboost --break-system-packages"
        ) from e

    return CatBoostClassifier(
        iterations=iterations,
        learning_rate=learning_rate,
        auto_class_weights=auto_class_weights,
        random_seed=RANDOM_STATE,
        verbose=verbose,
        allow_writing_files=False,
        text_features=[0],  # the only column
    )
