"""Шаг 3: econom-классификатор (TF-IDF + XGBoost 6-class) на unique_lemmas.

Env: econom_extractor (py3.9, sklearn 1.0.2, xgboost 1.7.6).
Вход:  run_100chan/unique_lemmas.parquet [text_hash, lemmas]
Выход: run_100chan/unique_pred_econom.parquet
       [text_hash, econom_class(int8), is_econom(int8), p_econom(f32), p_c0..p_c5]

class 0 = экономический. Препроцесс — как в run_econom_classifier.py.
"""
from __future__ import annotations

import pickle
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost as xgb_lib

warnings.simplefilter("ignore")

RUN = Path("/Users/daniltarasov/Downloads/topic_classifier_production/run_100chan")
REPO = Path("/Users/daniltarasov/Downloads/russian_news_database")
INPUT = RUN / "unique_lemmas.parquet"
OUTPUT = RUN / "unique_pred_econom.parquet"
BATCH = 50_000


def main() -> None:
    t0 = time.monotonic()
    print("loading econom models...", flush=True)
    tfidf = pickle.load(open(REPO / "models/class_model/tfidf.pkl", "rb"))
    xgb_grid = pickle.load(open(REPO / "models/class_model/xgboost_6classes.pkl", "rb"))
    stop_re = pickle.load(open(REPO / "models/class_model/stop_words.pkl", "rb"))
    trash_re = pickle.load(open(REPO / "models/class_model/trash_phrases.pkl", "rb"))
    booster = xgb_grid.best_estimator_.get_booster()
    n_class = 6
    print(f"  tfidf vocab={len(tfidf.vocabulary_)}, features={booster.num_features()}", flush=True)

    stop_pat = re.compile(stop_re)
    trash_pat = re.compile(trash_re)
    non_text_pat = re.compile(r'[^A-Za-zА-Яа-яё\s]')
    ws_pat = re.compile(r'\s+')

    def preproc(text):
        if not isinstance(text, str):
            return ""
        text = stop_pat.sub(" ", text)
        text = trash_pat.sub(" ", text)
        text = non_text_pat.sub("", text)
        return ws_pat.sub(" ", text).strip()

    fields = [
        pa.field("text_hash", pa.string()),
        pa.field("econom_class", pa.int8()),
        pa.field("is_econom", pa.int8()),
        pa.field("p_econom", pa.float32()),
    ]
    for i in range(n_class):
        fields.append(pa.field(f"p_c{i}", pa.float32()))
    out_schema = pa.schema(fields)

    pf = pq.ParquetFile(INPUT)
    n_total = pf.metadata.num_rows
    print(f"INPUT rows={n_total:,}", flush=True)

    done = 0
    t_inf = time.monotonic()
    with pq.ParquetWriter(OUTPUT, out_schema, compression="zstd") as writer:
        for bi, batch in enumerate(pf.iter_batches(batch_size=BATCH, columns=["text_hash", "lemmas"])):
            d = batch.to_pydict()
            n = len(d["lemmas"])
            texts = [preproc(l) for l in d["lemmas"]]
            X = tfidf.transform(texts)
            proba = booster.predict(xgb_lib.DMatrix(X))
            pred = proba.argmax(axis=1).astype("int8")
            cols = {
                "text_hash": d["text_hash"],
                "econom_class": pred,
                "is_econom": (pred == 0).astype("int8"),
                "p_econom": proba[:, 0].astype("float32"),
            }
            for i in range(n_class):
                cols[f"p_c{i}"] = proba[:, i].astype("float32")
            writer.write_table(pa.table(cols, schema=out_schema))
            done += n
            el = time.monotonic() - t_inf
            rate = done / el if el > 0 else 0
            eta = (n_total - done) / rate / 60 if rate > 0 else 0
            print(f"  batch {bi:>3}: {done:>8,}/{n_total:,} ({100*done/n_total:5.1f}%) "
                  f"| {rate:>5,.0f} t/s | eta {eta:4.1f}m", flush=True)

    print(f"\nsaved: {OUTPUT.name} ({OUTPUT.stat().st_size/1e6:.1f} MB)")
    print(f"DONE in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
