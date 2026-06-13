"""STANDALONE демо: econom-классификатор (TF-IDF + XGBoost, 6 классов) на ЛЕММАХ.

Пути — ОТНОСИТЕЛЬНО комплекта (../econom/...), машина-независимо.
Окружение: conda env econom_extractor (py3.9, sklearn 1.0.2, xgboost 1.7.6, scipy 1.10.1, numpy 1.23.5).
            ВАЖНО: pickles обучены на sklearn 1.0.2 — в новом sklearn распаковка может падать.

ВХОД — колонка `lemmas` (результат lemmatizer.py / шага topics). Модель применялась
в проде ИМЕННО к этим леммам, поэтому для воспроизводимости подавать те же леммы,
а не сырой текст. econom делает поверх лемм свою регэксп-очистку.

Запуск:
    # демо на встроенных леммах:
    python infer_econom_standalone.py
    # на паркете с колонкой lemmas (напр. выход infer_topics_standalone.py):
    python infer_econom_standalone.py --parquet /path/with_lemmas.parquet --out /path/out_econom.parquet

Выход: econom_class ∈ {0..5}, is_econom = (class==0), p_econom = P(class0), p_c0..p_c5.
class 0 = ЭКОНОМИЧЕСКИЙ.
"""
from __future__ import annotations
import argparse, pickle, re, warnings
from pathlib import Path
import numpy as np, pandas as pd
import xgboost as xgb_lib

warnings.simplefilter("ignore")

BUNDLE = Path(__file__).resolve().parents[1]            # .../classifiers
CM = BUNDLE / "econom" / "models" / "class_model"
N_CLASS = 6

DEMO_LEMMAS = [  # уже лемматизированные строки (как из lemmatizer.py)
    "цб повысить ключевой ставка процент сигнал сохранение",
    "минфин разместить офз рубль доходность аукцион",
    "кот спать подоконник солнечный погода",
    "криптобиржа запустить сервис цифровой рубль кошелёк",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", help="вход с колонкой lemmas; иначе встроенное демо")
    ap.add_argument("--out", help="куда сохранить результат (parquet)")
    args = ap.parse_args()

    tfidf = pickle.load(open(CM / "tfidf.pkl", "rb"))
    xgb_grid = pickle.load(open(CM / "xgboost_6classes.pkl", "rb"))
    stop_re = pickle.load(open(CM / "stop_words.pkl", "rb"))
    trash_re = pickle.load(open(CM / "trash_phrases.pkl", "rb"))
    booster = xgb_grid.best_estimator_.get_booster()
    print(f"tfidf vocab={len(tfidf.vocabulary_)}  features={booster.num_features()}")

    stop_pat = re.compile(stop_re)
    trash_pat = re.compile(trash_re)
    non_text_pat = re.compile(r"[^A-Za-zА-Яа-яё\s]")
    ws_pat = re.compile(r"\s+")

    def preproc(text):
        if not isinstance(text, str):
            return ""
        text = stop_pat.sub(" ", text)
        text = trash_pat.sub(" ", text)
        text = non_text_pat.sub("", text)
        return ws_pat.sub(" ", text).strip()

    if args.parquet:
        df = pd.read_parquet(args.parquet)
        lemmas = df["lemmas"].tolist()
    else:
        df = pd.DataFrame({"lemmas": DEMO_LEMMAS})
        lemmas = DEMO_LEMMAS

    X = tfidf.transform([preproc(l) for l in lemmas])
    proba = booster.predict(xgb_lib.DMatrix(X))
    pred = proba.argmax(axis=1).astype("int8")

    res = df.copy().reset_index(drop=True)
    res["econom_class"] = pred
    res["is_econom"] = (pred == 0).astype("int8")
    res["p_econom"] = proba[:, 0].astype("float32")
    for i in range(N_CLASS):
        res[f"p_c{i}"] = proba[:, i].astype("float32")

    if args.out:
        res.to_parquet(args.out, index=False, compression="zstd")
        print(f"saved: {args.out}  rows={len(res)}")
    else:
        for i, l in enumerate(lemmas):
            print(f"\n[{i}] {l[:70]}")
            print(f"    econom_class={int(pred[i])}  is_econom={int(pred[i]==0)}  p_econom={proba[i,0]:.3f}")


if __name__ == "__main__":
    main()
