"""eval: econom XGBoost (env econom_extractor).
usage: eval_econom.py <lemmas.parquet> <out_econom.parquet>
"""
from __future__ import annotations
import sys, re, pickle, warnings
from pathlib import Path
import numpy as np, pandas as pd, pyarrow.parquet as pq
import xgboost as xgb_lib
warnings.simplefilter("ignore")
REPO = Path("/Users/daniltarasov/Downloads/russian_news_database")


def main():
    lem = sys.argv[1]; out = sys.argv[2]
    tfidf = pickle.load(open(REPO/"models/class_model/tfidf.pkl","rb"))
    xgb_grid = pickle.load(open(REPO/"models/class_model/xgboost_6classes.pkl","rb"))
    stop_re = pickle.load(open(REPO/"models/class_model/stop_words.pkl","rb"))
    trash_re = pickle.load(open(REPO/"models/class_model/trash_phrases.pkl","rb"))
    booster = xgb_grid.best_estimator_.get_booster()
    stop_pat=re.compile(stop_re); trash_pat=re.compile(trash_re)
    non_text=re.compile(r'[^A-Za-zА-Яа-яё\s]'); ws=re.compile(r'\s+')
    def pp(t):
        if not isinstance(t,str): return ""
        t=stop_pat.sub(" ",t); t=trash_pat.sub(" ",t); t=non_text.sub("",t)
        return ws.sub(" ",t).strip()

    df = pq.read_table(lem).to_pandas()
    texts = [pp(l) for l in df["lemmas"].tolist()]
    X = tfidf.transform(texts)
    proba = booster.predict(xgb_lib.DMatrix(X))
    pred = proba.argmax(1).astype("int8")
    out_df = pd.DataFrame({
        "row_idx": df["row_idx"].values,
        "econom_class": pred,
        "is_econom": (pred==0).astype("int8"),
        "p_econom": proba[:,0].astype("float32"),
    })
    # too_short rows → не классифицируем (is_econom=0, class=-1)
    short = df["text_too_short"].values.astype(bool)
    out_df.loc[short, "is_econom"] = 0
    out_df.loc[short, "econom_class"] = -1
    out_df.loc[short, "p_econom"] = np.nan
    out_df.to_parquet(out, index=False, compression="zstd")
    print(f"econom done: {len(out_df):,} rows, is_econom={int(out_df['is_econom'].sum()):,} -> {out}")


if __name__ == "__main__":
    main()
