"""Model factories. 5 семейств (как в v1)."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC, SVC

from . import RANDOM_STATE


class ThresholdedBinaryClassifier:
    """Wraps any binary classifier (with decision_function or predict_proba) + fixed threshold τ.

    predict(X) returns (score(X) >= τ).astype(int).

    τ is chosen once on OOF CV scores (NOT on test predictions).
    After construction, self.threshold is frozen — predict() never re-tunes it.

    Compatible with sklearn Pipeline: fit/predict interface.
    Note: ThresholdedBinaryClassifier is used as the FINAL step inside a Pipeline,
    i.e. Pipeline([("features", FeatureUnion(...)), ("clf", ThresholdedBinaryClassifier(...))]).
    """

    def __init__(self, base_estimator, threshold: float = 0.0):
        self.base_estimator = base_estimator
        self.threshold = threshold

    def fit(self, X, y):
        self.base_estimator.fit(X, y)
        return self

    def _scores(self, X):
        if hasattr(self.base_estimator, "decision_function"):
            return self.base_estimator.decision_function(X)
        elif hasattr(self.base_estimator, "predict_proba"):
            return self.base_estimator.predict_proba(X)[:, 1]
        else:
            return self.base_estimator.predict(X).astype(float)

    def predict(self, X):
        # ASSERTION: threshold is fixed — never re-tuned here (OOF-only tuning)
        assert hasattr(self, "threshold"), "threshold must be set before predict()"
        s = self._scores(X)
        return (s >= self.threshold).astype(int)

    def get_params(self, deep=True):
        params = {"base_estimator": self.base_estimator, "threshold": self.threshold}
        if deep and hasattr(self.base_estimator, "get_params"):
            for k, v in self.base_estimator.get_params(deep=True).items():
                params[f"base_estimator__{k}"] = v
        return params

    def set_params(self, **params):
        if "threshold" in params:
            self.threshold = params.pop("threshold")
        if "base_estimator" in params:
            self.base_estimator = params.pop("base_estimator")
        if params and hasattr(self.base_estimator, "set_params"):
            # strip "base_estimator__" prefix
            sub = {k[len("base_estimator__"):]: v for k, v in params.items()
                   if k.startswith("base_estimator__")}
            if sub:
                self.base_estimator.set_params(**sub)
        return self

    def __getattr__(self, name):
        # Delegate unknown attributes to base_estimator (e.g. classes_, coef_)
        # Avoid infinite recursion: only called when attribute not found normally
        if name in ("base_estimator", "threshold"):
            raise AttributeError(name)
        return getattr(self.base_estimator, name)


def make_linear_svc(C: float = 1.0, class_weight=None, max_iter: int = 5000):
    return LinearSVC(C=C, class_weight=class_weight, max_iter=max_iter,
                     random_state=RANDOM_STATE, dual="auto")


def make_svc_linear(C: float = 1.0, class_weight=None):
    return SVC(kernel="linear", C=C, class_weight=class_weight, random_state=RANDOM_STATE)


def make_svc_rbf(C: float = 1.0, gamma="scale", class_weight=None):
    return SVC(kernel="rbf", C=C, gamma=gamma, class_weight=class_weight,
               random_state=RANDOM_STATE)


def make_logreg(C: float = 1.0, class_weight=None, solver: str = "liblinear", max_iter: int = 2000):
    return LogisticRegression(C=C, class_weight=class_weight, solver=solver,
                              max_iter=max_iter, random_state=RANDOM_STATE)


def make_multinomial_nb(alpha: float = 1.0):
    return MultinomialNB(alpha=alpha)


def make_complement_nb(alpha: float = 1.0, norm: bool = False):
    return ComplementNB(alpha=alpha, norm=norm)


def make_catboost(iterations: int = 300, learning_rate: float = 0.1, depth: int = 6,
                  auto_class_weights: str = "Balanced", verbose: int = 0):
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError("catboost not installed: pip install catboost --break-system-packages") from e
    return CatBoostClassifier(iterations=iterations, learning_rate=learning_rate, depth=depth,
                              auto_class_weights=auto_class_weights, random_seed=RANDOM_STATE,
                              verbose=verbose, allow_writing_files=False)


def make_catboost_text(iterations: int = 500, learning_rate: float = 0.05,
                       auto_class_weights: str = "Balanced", verbose: int = 0):
    try:
        from catboost import CatBoostClassifier
    except ImportError as e:
        raise ImportError("catboost not installed") from e
    return CatBoostClassifier(iterations=iterations, learning_rate=learning_rate,
                              auto_class_weights=auto_class_weights, random_seed=RANDOM_STATE,
                              verbose=verbose, allow_writing_files=False, text_features=[0])
