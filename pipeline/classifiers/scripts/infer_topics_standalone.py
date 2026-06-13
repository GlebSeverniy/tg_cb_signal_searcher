"""STANDALONE демо: 7 тем (6 binary-SVC + НБП CatBoost) на сырых текстах.

Пути — ОТНОСИТЕЛЬНО этого комплекта (../topics/...), машина-независимо.
Окружение: conda env topics311 (sklearn 1.6.1, pyarrow 24, catboost 1.2.9, pymorphy3, nltk).

Запуск:
    # демо на встроенных примерах:
    python infer_topics_standalone.py
    # на своём паркете с колонкой text -> сохранит result + колонку lemmas:
    python infer_topics_standalone.py --parquet /path/in.parquet --out /path/out_topics.parquet

Выход (на каждый текст): topic_<Тема> ∈ {0,1} + score_<Тема> (raw decision_function).
Порог τ вшит в каждую модель: topic = (score >= τ). lemmas тоже сохраняются —
их потом скармливают econom-скрипту (он работает по тем же леммам).

КРИТИЧНО — порядок загрузки. binary_per_topic и svc_focused_v2 оба содержат пакет
с именем `src`. Грузим binary ПЕРВЫМ (его src кэшируется), потом svc/НБП. Не менять.
Лемматизатор грузим по ПУТИ ФАЙЛА (importlib) под именем != 'src', чтобы не занять `src`.
"""
from __future__ import annotations
import argparse, importlib.util, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd

warnings.simplefilter("ignore")

BUNDLE = Path(__file__).resolve().parents[1]          # .../classifiers
TOPICS = BUNDLE / "topics"
BIN_ROOT = TOPICS / "binary_per_topic_classifier"
SVC_ROOT = TOPICS / "svc_focused_v2"
LEMMA_PY = TOPICS / "lemmatizer.py"
BIN_MIX = BIN_ROOT / "results" / "svc_per_topic_pipeline.joblib"
NBP_STACKER = SVC_ROOT / "results" / "nbp_v3" / "nbp_stacker.joblib"

# финальный порядок тем (как в продакшн-сборке)
TOPICS_ORDER = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "НБП", "Финтех", "Геополитика", "НДО"]
BINARY_TOPICS = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "Финтех", "Геополитика", "НДО"]
TOO_SHORT = 20

DEMO = [
    "ЦБ повысил ключевую ставку до 16% и просигналил о её сохранении надолго",
    "Минфин разместил ОФЗ на 50 млрд рублей под рекордную доходность",
    "Новая льготная ипотека сворачивается, банки ужесточили условия по вкладам",
    "Сегодня солнечная погода, кот спит на подоконнике",
    "Криптобиржа запустила новый сервис цифрового рубля и финтех-кошелёк",
]


def load_lemmatizer():
    spec = importlib.util.spec_from_file_location("lemmatizer_bundle", str(LEMMA_PY))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="вход с колонкой text; иначе встроенное демо")
    ap.add_argument("--out", help="куда сохранить результат (parquet)")
    args = ap.parse_args()

    # 1) тексты
    if args.parquet:
        df = pd.read_parquet(args.parquet)
        texts = df["text"].tolist()
    else:
        df = pd.DataFrame({"text": DEMO})
        texts = DEMO

    # 2) лемматизация (та же, что при обучении)
    lem = load_lemmatizer()
    morph, stop = lem.get_morph(), lem.get_stopwords()
    lemmas, too_short = [], []
    for t in texts:
        if t is None or len(t) < TOO_SHORT:
            lemmas.append(""); too_short.append(True)
        else:
            lemmas.append(lem.lemmatize_text(t, morph=morph, stopwords=stop)); too_short.append(False)

    # 3) модели — СНАЧАЛА binary (кэширует src), ПОТОМ svc/НБП
    import joblib
    sys.path.insert(0, str(BIN_ROOT)); sys.path.insert(0, str(BIN_ROOT / "src"))
    mix = joblib.load(BIN_MIX)
    sys.path.insert(0, str(SVC_ROOT))
    sys.path.insert(0, str(SVC_ROOT / "results" / "nbp_v3"))
    sys.path.insert(0, str(SVC_ROOT / "src"))
    nbp = joblib.load(NBP_STACKER)

    X = pd.Series([l or "" for l in lemmas])
    out = {"text_too_short": too_short}
    for t in BINARY_TOPICS:
        m = mix[t]; thr = getattr(m, "threshold", 0.0)
        s = np.asarray(m.decision_function(X), dtype="float32")
        out["topic_" + t.replace(" ", "_")] = (s >= thr).astype("int8")
        out["score_" + t.replace(" ", "_")] = s
    s = np.asarray(nbp.decision_function(X), dtype="float32")
    out["topic_НБП"] = (s >= nbp.threshold).astype("int8")
    out["score_НБП"] = s

    res = df.copy().reset_index(drop=True)
    res["lemmas"] = lemmas
    for k, v in out.items():
        res[k] = v
    # too_short -> topic=0, score=NaN
    short = np.array(too_short)
    for t in TOPICS_ORDER:
        res.loc[short, "topic_" + t.replace(" ", "_")] = 0
        res.loc[short, "score_" + t.replace(" ", "_")] = np.nan

    if args.out:
        res.to_parquet(args.out, index=False, compression="zstd")
        print(f"saved: {args.out}  rows={len(res)}")
    else:
        for i, t in enumerate(texts):
            active = [t2 for t2 in TOPICS_ORDER if res.iloc[i]["topic_" + t2.replace(" ", "_")] == 1]
            print(f"\n[{i}] {t[:80]}")
            print(f"    темы: {active or '— нет —'}")


if __name__ == "__main__":
    main()
