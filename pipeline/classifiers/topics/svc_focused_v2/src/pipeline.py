"""
Pipeline: build pipelines, run experiments, **build mixture by CV (not test)**.

v2 critical fix: build_mixture_pipeline_by_cv (default) — отбор по cv_per_topic.
build_mixture_pipeline_by_test (legacy, для сравнения и репро v1) явно помечен.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

from . import RANDOM_STATE, ALLOWED_TOPICS
from .data_loader import load_test, load_train, load_train_binary, load_test_binary
from .registry import publish_per_topic_experiment, update_per_topic_with_test, all_entries
from .validation import (
    cv_evaluate_binary_per_topic_loader,
    cv_evaluate_binary_per_topic_loader_with_threshold,
    evaluate_on_test_per_topic_loader,
    cv_evaluate_binary_per_topic,
    evaluate_on_test_per_topic,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
ATTEMPTS_DIR = RESULTS_DIR / "attempts"


# ---- Feature groups (наследуются из v1) ----

def _tfidf_word_12():
    return ("tfidf_word_12", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_word_13():
    return ("tfidf_word_13", TfidfVectorizer(ngram_range=(1, 3), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_char_35():
    return ("tfidf_char_35", TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_char_25():
    return ("tfidf_char_25", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True, lowercase=True))


def _tfidf_char_24():
    return ("tfidf_char_24", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2, sublinear_tf=True, lowercase=True))


def _count_word_12():
    return ("count_word_12", CountVectorizer(ngram_range=(1, 2), min_df=2, lowercase=True))


def _tfidf_word_12_nb():
    return ("tfidf_word_12_nb", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=False, lowercase=True))


def _tfidf_word_12_svd200():
    return ("tfidf_word_12_svd200",
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, max_features=50000)),
                ("svd", TruncatedSVD(n_components=200, random_state=RANDOM_STATE)),
            ]))


def _tfidf_word_12_svd500():
    return ("tfidf_word_12_svd500",
            Pipeline([
                ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, lowercase=True, max_features=80000)),
                ("svd", TruncatedSVD(n_components=500, random_state=RANDOM_STATE)),
            ]))


# ---- Keyword-per-topic feature group (App.5 LDA clusters) ----
# 8 разрешённых тем × ~45 keywords each (union of all clusters per topic).
# Feature i = sum of keyword occurrences for topic i in the text, normalised by token count.
# Keywords are matched as substrings in lowercase lemmatised text (str.count).
# NO test labels are used — pure text features, no leakage.

_KW_TOPICS_ORDER = [
    "ДКП в мире",
    "ДКП",
    "Защита прав потребителей",
    "Финансовый рынок",
    "НБП",
    "Финтех",
    "Геополитика",
    "НДО",
]

_TOPIC_KEYWORDS: Dict[str, list] = {
    # Clusters {0, 12, 29}
    "ДКП в мире": [
        "дкп", "ецб", "япония", "европа", "ужесточение", "англия", "спикер", "макро",
        "британия", "австралия", "отрицательный", "повышение", "канада", "инфляция",
        "стимулирование",
        "кризис", "европейский", "мировой", "экономика", "ецб", "центральный",
        "евро", "германия", "еврозона", "глобальный", "рецессия", "мировой экономика",
        "сша", "фрс", "повышение ставка", "ужесточение", "индекс",
        "ставка фрс", "дкп сша", "американский", "пауэлл", "рост", "акция",
    ],
    # Clusters {1, 3, 5, 14, 21, 25, 30, 43, 51, 52, 56}
    "ДКП": [
        "прогноз", "ввп", "экономика", "рост", "инфляция", "рф", "сценарий", "оценка",
        "рост ввп", "ожидать", "уровень", "темп", "банк россия", "прогнозировать",
        "российский экономика",
        "набиуллин", "глава", "эльвира", "набиуллина", "эльвира набиуллин",
        "председатель", "центробанк", "глава эльвира", "прессконференция",
        "валюта", "курс", "доллар", "евро", "покупка", "валютный", "минфин",
        "интервенция", "покупка валюта", "официальный",
        "ключевой", "ключевой ставка", "повышать", "годовой", "повышение",
        "снижать", "решение", "процентный", "сохранять",
        "денежнокредитный", "денежнокредитный политика", "условие", "мера",
        "приводить", "спрос", "высокий", "экономический рост",
        "цена", "рост цена", "потребительский", "показатель", "товар",
        "вырастать", "потребительский цена", "росстат", "данные",
        "ожидание", "инфляционный", "риск", "инфляционный ожидание",
        "давление", "динамика", "текущий",
        "снижение", "снижение ставка", "заседание", "снижать ставка",
        "дальнейший", "сигнал", "близкий",
        "департамент", "директор", "тремасов", "финансовый", "аналитика",
        "отдел", "управление", "экономист", "начальник", "материал",
        "совет", "совет директор", "репо", "офз", "аукцион", "участие",
        "резерв", "золото", "юань", "доля", "звр", "международный",
        "золотовалютный", "тонна", "объем", "золотовалютный резерв",
    ],
    # Clusters {4, 9, 11, 22, 35}
    "Защита прав потребителей": [
        "клиент", "продукт", "комиссия", "продажа", "банка", "ассоциация",
        "условие", "покупка", "вводить", "договор", "товар", "фас", "потребитель",
        "отказываться", "письмо",
        "мошенник", "счет", "сотрудник", "карта", "схема", "средство",
        "мошенничество", "номер", "банковский", "звонок", "переводить", "жертва",
        "данные", "центробанк",
        "суд", "бывший", "владелец", "иск", "уголовный", "компания", "банкстер",
        "актив", "группа", "арест", "акционер", "асв", "ананьев", "rub",
        "операция", "закон", "документ", "нарушение", "организация", "орган",
        "информация", "деятельность", "проверка", "право", "денежный", "требование",
        "штраф",
        "список", "черный", "зампред", "надзор", "руководство", "черный список",
        "попадать", "поздышев", "должность", "руководитель", "уходить", "зачистка",
    ],
    # Clusters {2, 10, 13, 17, 23, 24, 27, 32, 33, 34, 38, 39, 42, 50, 54, 55}
    "Финансовый рынок": [
        "финансовый", "центральный", "центральный банк", "финансовый рынок",
        "риск", "система", "стабильность", "банк россия", "финансовый система",
        "финансовый стабильность", "инструмент", "развитие", "финансовый организация",
        "организация", "финансовый сектор",
        "кредит", "кредитный", "заемщик", "заем", "бизнес", "выдавать", "выдача",
        "мера", "поддержка", "кредитование", "долг", "малый", "каникулы",
        "фонд", "пенсионный", "пенсия", "нпф", "накопление", "выплата",
        "пенсионный фонд", "иис", "пенсионер", "реформа", "гражданин", "швецов",
        "ипотека", "ипотечный", "льготный", "жилье", "программа",
        "недвижимость", "льготный ипотека", "ипотечный кредит", "застройщик",
        "квартира", "ставка ипотека", "спрос",
        "облигация", "доходность", "инвестор", "акция", "выпуск",
        "бумага", "портфель", "корпоративный", "бонд", "зарабатывать",
        "рейтинг", "капитал", "агентство", "актив", "норматив",
        "страховой", "страхование", "страховщик", "осаго", "пирамида",
        "страховой компания", "асв", "финансовый пирамида", "полис", "тариф",
        "страховка", "жалоба",
        "акция", "компания", "дивиденд", "сделка", "продажа", "сбербанк",
        "продавать", "пакет", "прибыль", "цена", "газпром",
        "банковский", "сектор", "ликвидность", "банковский сектор",
        "банковский система", "прибыль", "банка россия", "банк рф", "убыток",
        "биржа", "ценный", "ценный бумага", "торги", "брокер", "иностранный",
        "мосбиржа", "московский", "московский биржа", "фондовый",
        "открытие", "втб", "санация", "траст", "бинбанк", "фк",
        "промсвязьбанк", "фк открытие", "банк открытие", "санировать",
        "задорнов", "костин",
        "вклад", "депозит", "рублевый", "население", "максимальный",
        "ставка депозит", "ставка вклад", "отток", "сбережение", "снижаться",
        "россиянин",
        "счет", "восточный", "аветисян", "открывать", "валютный счет",
        "остаток", "тинькофф", "счет банк", "обслуживание", "банк восточный",
        "лицензия", "отзывать", "кредитный организация", "отзывать лицензия",
        "отзыв", "администрация", "временный", "лицензия банк",
        "временный администрация",
        "офз", "нерезидент", "аукцион", "длинный", "рынок офз", "размещение",
        "госдолг", "летний", "размещать", "кривой", "доля нерезидент",
        "доход", "кредитование", "объем", "данные", "потребительский",
        "расти", "сравнение", "реальный",
    ],
    # Clusters {8, 18, 19}
    "НБП": [
        "бюджет", "минфин", "расход", "фнб", "долг", "бюджетный", "средство",
        "дефицит", "экономика", "доход", "счет", "правительство", "денежный",
        "финансирование", "объем",
        "налог", "регион", "налоговый", "доход", "региональный", "бюджет",
        "федеральный", "ндс", "платить", "семья", "край", "сумма", "ндфл",
        "фнс", "местный",
        "правительство", "развитие", "проект", "министр", "силуанов",
        "экономика", "экономический", "государство", "мишустин", "финансы",
        "кудрин", "сектор", "отрасль",
    ],
    # Clusters {6, 20, 40}
    "Финтех": [
        "цифровой", "криптовалюта", "законопроект", "регулирование", "госдума",
        "валюта", "запрет", "крипто", "минфин", "запрещать", "цифровой валюта",
        "центробанк", "банк россия", "использование", "рф",
        "перевод", "платеж", "сбербанк", "система", "греф", "быстрый", "сбп",
        "система быстрый", "быстрый платеж", "клиент", "сервис", "сбер",
        "оплата", "комиссия", "карта",
        "информация", "данные", "доступ", "компания", "экосистема", "база",
        "информационный", "платформа", "оператор", "данный", "единый",
        "пользователь",
    ],
    # Clusters {41, 44, 45, 49}
    "Геополитика": [
        "санкция", "сша", "вводить", "актив", "рф", "ограничение", "попадать",
        "замораживать", "санкционный", "введение", "отношение", "вводить санкция",
        "западный", "санкция россия", "запад",
        "украина", "трамп", "нацбезопасность", "президент", "байден",
        "переговоры", "американский", "китай", "нато", "советник", "северный",
        "угроза", "президент сша",
        "украинский", "военный", "всу", "днр", "нацбатальон", "минобороны",
        "территория", "житель", "мариуполь", "мирный", "киев", "боевик", "лнр",
        "путин", "владимир", "владимир путин", "власть", "выборы",
        "правительство", "называть", "кремль", "белоруссия", "дмитрий",
        "партия", "депутат",
    ],
    # Cluster {28}
    "НДО": [
        "наличный", "наличный валюта", "обращение", "снимать", "снятие",
        "монета", "купюра", "банкомат", "банк россия", "выпускать", "центробанк",
        "касса", "ограничение", "югра", "номинал",
    ],
}


class _KeywordPerTopicTransformer:
    """sklearn-compatible transformer: 8 numeric features (one per allowed topic).

    feature_i = count of topic_i keywords found in text / max(1, n_tokens).
    Matching: substring count in lowercase text (str.count per keyword).
    No fit needed (stateless). No test labels used — pure text feature, no leakage.

    ASSERTION: len(output) == 8 for every input row.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        result = []
        for text in X:
            t = str(text).lower()
            n_tokens = max(1, len(t.split()))
            row = []
            for topic in _KW_TOPICS_ORDER:
                kws = _TOPIC_KEYWORDS[topic]
                cnt = sum(t.count(kw) for kw in kws)
                row.append(cnt / n_tokens)
            assert len(row) == 8, f"Expected 8 keyword features, got {len(row)}"
            result.append(row)
        return np.array(result, dtype=float)

    def fit_transform(self, X, y=None):
        return self.transform(X)

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


def _keyword_per_topic_count():
    return ("keyword_per_topic_count", _KeywordPerTopicTransformer())


FEATURE_GROUPS: Dict[str, Callable] = {
    "tfidf_word_12": _tfidf_word_12,
    "tfidf_word_13": _tfidf_word_13,
    "tfidf_char_35": _tfidf_char_35,
    "tfidf_char_25": _tfidf_char_25,
    "tfidf_char_24": _tfidf_char_24,
    "count_word_12": _count_word_12,
    "tfidf_word_12_nb": _tfidf_word_12_nb,
    "tfidf_word_12_svd200": _tfidf_word_12_svd200,
    "tfidf_word_12_svd500": _tfidf_word_12_svd500,
    "keyword_per_topic_count": _keyword_per_topic_count,
}


def _wrap_extra_fn(fn: Callable):
    name = getattr(fn, "__name__", "extra")

    def _transform(texts):
        feats, _ = fn(list(texts))
        return np.asarray(feats)
    return (name, FunctionTransformer(_transform, validate=False))


def build_pipeline(model_factory: Callable, feature_groups: List[str],
                    extra_feature_fns: Optional[List[Callable]] = None) -> Pipeline:
    transformers: list = []
    for fg in feature_groups:
        if fg not in FEATURE_GROUPS:
            raise KeyError(f"Unknown FG '{fg}'. Available: {sorted(FEATURE_GROUPS)}")
        transformers.append(FEATURE_GROUPS[fg]())
    for fn in extra_feature_fns or []:
        transformers.append(_wrap_extra_fn(fn))
    union = FeatureUnion(transformers, n_jobs=1)
    return Pipeline([("features", union), ("clf", model_factory())])


# ---- Run experiment ----

def run_per_topic_experiment(exp_id: str, model_factory: Callable, feature_groups: List[str],
                              family: str, extra_feature_fns: Optional[List[Callable]] = None,
                              notes: str = "", cv_splits: int = 5,
                              topics: Optional[List] = None) -> dict:
    """
    v2 CANONICAL: использует load_train_binary(topic) для каждой темы.
    Все 4371 строк train используются для каждого binary classifier.
    CV и финальный fit — на binary 0/1 column из данных (не multiclass-derived).
    """
    print(f"\n=== {exp_id} ({family}) ===", flush=True)
    print(f"  feature_groups: {feature_groups}", flush=True)
    if topics is None:
        topics = list(ALLOWED_TOPICS)
    print(f"  topics: {len(topics)} ({list(topics)[:3]}...)", flush=True)
    print(f"  Loading data via load_train_binary (binary columns, all 4371 rows)", flush=True)

    factory = lambda: build_pipeline(model_factory, feature_groups, extra_feature_fns)

    t0 = time.time()
    # v2 canonical: loader-based CV — каждая тема получает все 4371 строк
    cv_res = cv_evaluate_binary_per_topic_loader(
        factory, load_train_binary, topics=topics, n_splits=cv_splits, verbose=True
    )
    cv_seconds = time.time() - t0

    out_dir = ATTEMPTS_DIR / exp_id
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Финальный fit на полном train (все 4371 строк per topic)
    final_pipes = {}
    for topic in topics:
        X_full, y_bin = load_train_binary(topic)
        if y_bin.sum() < cv_splits:
            print(f"  WARNING: {topic} has < {cv_splits} positives — skipping final fit", flush=True)
            continue
        pipe = factory()
        pipe.fit(X_full, y_bin)
        safe = _safe_filename(str(topic))
        joblib.dump(pipe, models_dir / f"{safe}.joblib")
        final_pipes[str(topic)] = str(models_dir / f"{safe}.joblib")

    config = {
        "exp_id": exp_id, "family": family,
        "feature_groups": feature_groups,
        "extra_feature_fns": [getattr(f, "__name__", str(f)) for f in (extra_feature_fns or [])],
        "model_factory": _describe_estimator(model_factory()),
        "notes": notes, "random_state": RANDOM_STATE, "cv_splits": cv_splits,
        "topics": [str(t) for t in topics],
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))
    (out_dir / "cv_results.json").write_text(json.dumps(cv_res, indent=2, ensure_ascii=False))
    (out_dir / "final_pipelines.json").write_text(json.dumps(final_pipes, indent=2, ensure_ascii=False))

    publish_per_topic_experiment(
        exp_id=exp_id, family=family, cv_results=cv_res,
        feature_groups=feature_groups, model_desc=config["model_factory"],
        notes=notes, cv_seconds=cv_seconds,
    )
    return {"cv": cv_res, "config": config, "models_dir": str(models_dir)}


def predict_test_per_topic(exp_id: str) -> dict:
    """
    Test eval. ВАЖНО: используется для отчёта/диагностики и для финальной mixture,
    но НЕ для отбора best-per-topic. Mixture отбирается через build_mixture_pipeline_by_cv.

    v2 canonical: для каждой темы зовёт load_test_binary(topic) → 1093 строки с бинарной меткой.
    """
    out_dir = ATTEMPTS_DIR / exp_id
    final_pipes_map = json.loads((out_dir / "final_pipelines.json").read_text())
    fitted = {topic: joblib.load(path) for topic, path in final_pipes_map.items()}

    print(f"\n=== {exp_id} TEST (DIAGNOSTIC, не для отбора mixture) ===", flush=True)
    print(f"  Loading test data via load_test_binary (binary columns, all 1093 rows)", flush=True)
    # v2 canonical: loader-based test eval — 1093 строки per topic с бинарной меткой
    test_res = evaluate_on_test_per_topic_loader(fitted, load_test_binary)
    print(
        f"  macro test: P={test_res['macro_avg_precision']:.4f}, "
        f"R={test_res['macro_avg_recall']:.4f}, "
        f"F1={test_res['macro_avg_f1']:.4f}", flush=True,
    )

    # Strip y_pred before saving (large)
    test_to_save = {**test_res}
    test_to_save["per_topic"] = {t: {k: v for k, v in m.items() if k != "y_pred"}
                                 for t, m in test_res["per_topic"].items()}
    (out_dir / "test_metrics.json").write_text(json.dumps(test_to_save, indent=2, ensure_ascii=False))

    # Save y_pred separately (for bootstrap CI later)
    (out_dir / "test_predictions.json").write_text(json.dumps(
        {t: m["y_pred"] for t, m in test_res["per_topic"].items()},
        indent=2,
    ))

    update_per_topic_with_test(exp_id=exp_id, test_results=test_to_save)
    return test_res


def run_per_topic_experiment_with_threshold(
    exp_id: str, model_factory: Callable, feature_groups: List[str],
    family: str, extra_feature_fns: Optional[List[Callable]] = None,
    notes: str = "", cv_splits: int = 5, n_thresholds: int = 200,
    topics: Optional[List] = None,
) -> dict:
    """
    Same as run_per_topic_experiment but with OOF threshold tuning per topic.

    OOF threshold τ* is chosen on OOF decision_function/predict_proba scores ONLY.
    τ is NEVER tuned on test predictions.

    For each topic:
      1. CV: accumulate OOF scores, sweep τ over 200 points, pick τ* maximising OOF F1.
      2. Final fit: Pipeline(...) on full train → ThresholdedBinaryClassifier(pipe, τ*).
      3. Save ThresholdedBinaryClassifier to joblib — predict_test_per_topic applies τ* automatically.
      4. thresholds_per_topic saved in config.json.

    ASSERTION: τ comes from cv_evaluate_binary_per_topic_loader_with_threshold which uses
    only OOF val-fold scores. Test data is NOT accessible during CV.
    """
    from .model import ThresholdedBinaryClassifier

    print(f"\n=== {exp_id} ({family}) [with OOF threshold tuning] ===", flush=True)
    print(f"  feature_groups: {feature_groups}", flush=True)
    if topics is None:
        topics = list(ALLOWED_TOPICS)
    print(f"  topics: {len(topics)} ({list(topics)[:3]}...)", flush=True)
    print(f"  Loading data via load_train_binary (binary columns, all rows)", flush=True)

    factory = lambda: build_pipeline(model_factory, feature_groups, extra_feature_fns)

    t0 = time.time()
    # OOF threshold tuning — τ chosen ONLY on OOF val-fold scores, NOT on test
    cv_res = cv_evaluate_binary_per_topic_loader_with_threshold(
        factory, load_train_binary, topics=topics, n_splits=cv_splits,
        n_thresholds=n_thresholds, verbose=True,
    )
    cv_seconds = time.time() - t0

    # Extract per-topic thresholds from CV results (OOF-derived, not test-derived)
    thresholds_per_topic = {
        topic: cv_res["per_topic"].get(str(topic), {}).get("best_threshold", 0.0)
        for topic in topics
    }
    print(f"\n  OOF thresholds per topic:", flush=True)
    for topic, tau in thresholds_per_topic.items():
        print(f"    {topic}: τ={tau:.4f}", flush=True)

    out_dir = ATTEMPTS_DIR / exp_id
    models_dir = out_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # Final fit per topic: Pipeline fitted on full train, wrapped in ThresholdedBinaryClassifier(τ*)
    final_pipes = {}
    for topic in topics:
        topic_cv = cv_res["per_topic"].get(str(topic), {})
        if topic_cv.get("skipped"):
            print(f"  WARNING: {topic} skipped during CV — skipping final fit", flush=True)
            continue

        X_full, y_bin = load_train_binary(topic)
        if y_bin.sum() < cv_splits:
            print(f"  WARNING: {topic} has < {cv_splits} positives — skipping final fit", flush=True)
            continue

        # Fit base pipeline on full training data
        base_pipe = factory()
        base_pipe.fit(X_full, y_bin)

        # Wrap with OOF-tuned threshold τ* — NEVER re-tune on test
        tau = thresholds_per_topic[topic]
        thresholded = ThresholdedBinaryClassifier(base_pipe, threshold=tau)

        safe = _safe_filename(str(topic))
        joblib.dump(thresholded, models_dir / f"{safe}.joblib")
        final_pipes[str(topic)] = str(models_dir / f"{safe}.joblib")

    config = {
        "exp_id": exp_id, "family": family,
        "feature_groups": feature_groups,
        "extra_feature_fns": [getattr(f, "__name__", str(f)) for f in (extra_feature_fns or [])],
        "model_factory": _describe_estimator(model_factory()),
        "notes": notes, "random_state": RANDOM_STATE, "cv_splits": cv_splits,
        "n_thresholds": n_thresholds,
        "topics": [str(t) for t in topics],
        "thresholds_per_topic": thresholds_per_topic,
        "threshold_method": "OOF_CV_only",  # invariant: τ from OOF, NOT from test
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False, default=str))
    (out_dir / "cv_results.json").write_text(json.dumps(cv_res, indent=2, ensure_ascii=False))
    (out_dir / "final_pipelines.json").write_text(json.dumps(final_pipes, indent=2, ensure_ascii=False))

    publish_per_topic_experiment(
        exp_id=exp_id, family=family, cv_results=cv_res,
        feature_groups=feature_groups, model_desc=config["model_factory"],
        notes=notes + " [OOF threshold tuning]", cv_seconds=cv_seconds,
    )
    return {"cv": cv_res, "config": config, "models_dir": str(models_dir)}


# ---- Mixture builders ----

def build_mixture_pipeline_by_cv(exp_id: str = "final_mixture_v2_cv") -> dict:
    """
    ★ КАНОНИЧЕСКАЯ mixture v2. Для каждой темы выбирается экспериментальная entry
    с максимальным cv_per_topic[topic]['f1_mean'] (НЕ test_per_topic).

    Это ЕДИНСТВЕННЫЙ корректный способ отбора (см. inputs/INDEPENDENT_AUDIT.md).
    """
    entries = [e for e in all_entries() if not e.get("flagged") and e.get("family") != "mixture"]

    # Topics: 8 разрешённых
    topics = list(ALLOWED_TOPICS)

    best_per_topic = {}
    family_per_topic = {}

    for topic in topics:
        best_f1 = -1.0
        best_exp = None
        for e in entries:
            cv_pt = (e.get("cv_per_topic") or {}).get(topic)
            if cv_pt and cv_pt.get("f1_mean") is not None and cv_pt["f1_mean"] > best_f1:
                best_f1 = cv_pt["f1_mean"]
                best_exp = e
        if best_exp:
            # ★ INVARIANT для аудитора:
            assert "cv_per_topic" in best_exp, "mixture must use CV F1, not test"
            best_per_topic[topic] = {"exp_id": best_exp["exp_id"], "cv_f1": best_f1}
            family_per_topic[topic] = best_exp["family"]

    print(f"\n=== {exp_id} (CV-honest mixture) ===", flush=True)
    for topic in topics:
        bp = best_per_topic.get(topic)
        if bp:
            print(f"  {topic}: {bp['exp_id']} ({family_per_topic.get(topic)}, cv_f1={bp['cv_f1']:.4f})",
                  flush=True)

    # Copy models
    out_dir = ATTEMPTS_DIR / exp_id
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = out_dir / "models"
    models_dir.mkdir(exist_ok=True)

    fitted = {}
    for topic, choice in best_per_topic.items():
        src_path = ATTEMPTS_DIR / choice["exp_id"] / "models" / f"{_safe_filename(topic)}.joblib"
        if not src_path.exists():
            print(f"WARNING: model missing for {topic} in {choice['exp_id']}", flush=True)
            continue
        dst_path = models_dir / f"{_safe_filename(topic)}.joblib"
        if str(src_path) != str(dst_path):
            import shutil
            shutil.copy2(src_path, dst_path)
        fitted[topic] = joblib.load(dst_path)

    # Score on test (ОДИН РАЗ — финал)
    # v2 canonical: loader-based eval — каждая тема получает все 1093 строки с binary меткой
    print(f"  Scoring final mixture on test via load_test_binary (binary columns)", flush=True)
    test_res = evaluate_on_test_per_topic_loader(fitted, load_test_binary)
    test_res["family_per_topic"] = family_per_topic
    test_res["selection_method"] = "cv_macro_f1"  # invariant marker

    # Save artifacts
    test_to_save = {
        "selection_method": "cv_macro_f1",
        "family_per_topic": family_per_topic,
        "best_per_topic_cv": {t: c for t, c in best_per_topic.items()},
        "macro_avg_f1": test_res["macro_avg_f1"],
        "macro_avg_precision": test_res["macro_avg_precision"],
        "macro_avg_recall": test_res["macro_avg_recall"],
        "n_test": test_res["n_test"],
        "per_topic": {t: {k: v for k, v in m.items() if k != "y_pred"}
                      for t, m in test_res["per_topic"].items()},
    }
    (out_dir / "test_metrics.json").write_text(json.dumps(test_to_save, indent=2, ensure_ascii=False))
    (out_dir / "test_predictions.json").write_text(json.dumps(
        {t: m["y_pred"] for t, m in test_res["per_topic"].items()}, indent=2,
    ))

    # Final artifacts
    bundle = {
        "selection_method": "cv_macro_f1",
        "per_topic_paths": {t: str(models_dir / f"{_safe_filename(t)}.joblib") for t in fitted},
        "family_per_topic": family_per_topic,
    }
    (RESULTS_DIR / "final_mixture.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False))
    joblib.dump(fitted, RESULTS_DIR / "final_mixture.joblib")
    (RESULTS_DIR / "final_metrics.json").write_text(json.dumps(test_to_save, indent=2, ensure_ascii=False))

    # Register
    publish_per_topic_experiment(
        exp_id=exp_id, family="mixture",
        cv_results={"per_topic": {}, "macro_avg_f1_mean": None,
                    "macro_avg_precision_mean": None, "macro_avg_recall_mean": None,
                    "topic_support": {}, "topics_used": list(topics),
                    "n_train": 0, "n_splits": 0},
        feature_groups=["mixture_cv"],
        model_desc=f"BestPerTopicByCV({family_per_topic})",
        notes="CV-honest mixture. Selection by max cv_per_topic[*]['f1_mean']. NO test fishing.",
        cv_seconds=None,
    )
    update_per_topic_with_test(exp_id=exp_id, test_results=test_to_save)

    print(f"\n  test macro F1 (final mixture): {test_res['macro_avg_f1']:.4f}", flush=True)
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
