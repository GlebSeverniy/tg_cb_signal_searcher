"""eval: лемматизация + 7 тем v3 на размеченном файле (env topics311).

usage: eval_topics.py <input.parquet> <out_prefix>
выход:
  <out_prefix>_lemmas.parquet  [row_idx, lemmas, text_too_short]
  <out_prefix>_topics.parquet  [row_idx, 7×topic_*, 7×score_*]
"""
from __future__ import annotations
import sys, os, time, warnings
from pathlib import Path
import numpy as np, pandas as pd, pyarrow as pa, pyarrow.parquet as pq

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.simplefilter("ignore")
PROD = Path("/Users/daniltarasov/Downloads/topic_classifier_production")

# Грузим lemmatizer ПО ПУТИ ФАЙЛА под именем НЕ 'src', чтобы не занять пакет `src`
# (его потом должен занять binary_per_topic.src.pipeline). lemmatizer.py самодостаточен.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location("lemmatizer_prod", str(PROD / "src" / "lemmatizer.py"))
_lem = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_lem)

TOPICS = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "НБП", "Финтех", "Геополитика", "НДО"]
BINARY = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "Финтех", "Геополитика", "НДО"]
BIN_MIX = Path("/Users/daniltarasov/Downloads/binary_per_topic_classifier/results/svc_per_topic_pipeline.joblib")
NBP_STACKER = Path("/Users/daniltarasov/Downloads/svc_focused_v2/results/nbp_v3/nbp_stacker.joblib")
TOO_SHORT = 20


def main():
    inp = Path(sys.argv[1]); pref = sys.argv[2]
    t0 = time.monotonic()

    # --- lemmatize all rows (single process, files small) ---
    morph = _lem.get_morph()
    stop = _lem.get_stopwords()
    texts = pq.read_table(inp, columns=["text"]).column("text").to_pylist()
    n = len(texts)
    print(f"{inp.name}: {n:,} rows", flush=True)
    lemmas = [""] * n
    too_short = [False] * n
    for i, t in enumerate(texts):
        if t is None or len(t) < TOO_SHORT:
            too_short[i] = True
        else:
            lemmas[i] = _lem.lemmatize_text(t, morph=morph, stopwords=stop)
    pq.write_table(pa.table({"row_idx": list(range(n)), "lemmas": lemmas,
                             "text_too_short": too_short}),
                   f"{pref}_lemmas.parquet", compression="zstd")
    print(f"  lemmatized in {time.monotonic()-t0:.1f}s, too_short={sum(too_short)}", flush=True)

    # --- load models (binary FIRST, then svc — порядок важен из-за коллизии пакета src) ---
    import joblib
    sys.path.insert(0, "/Users/daniltarasov/Downloads/binary_per_topic_classifier")
    sys.path.insert(0, "/Users/daniltarasov/Downloads/binary_per_topic_classifier/src")
    mix = joblib.load(BIN_MIX)
    sys.path.insert(0, "/Users/daniltarasov/Downloads/svc_focused_v2")
    sys.path.insert(0, "/Users/daniltarasov/Downloads/svc_focused_v2/results/nbp_v3")
    sys.path.insert(0, "/Users/daniltarasov/Downloads/svc_focused_v2/src")
    nbp = joblib.load(NBP_STACKER)

    X = pd.Series([l or "" for l in lemmas])
    cols = {"row_idx": list(range(n))}
    for t in BINARY:
        m = mix[t]; thr = getattr(m, "threshold", 0.0)
        s = np.asarray(m.decision_function(X), dtype="float32")
        cols["topic_" + t.replace(" ", "_")] = (s >= thr).astype("int8")
        cols["score_" + t.replace(" ", "_")] = s
    s = np.asarray(nbp.decision_function(X), dtype="float32")
    cols["topic_НБП"] = (s >= nbp.threshold).astype("int8")
    cols["score_НБП"] = s
    # too_short → topic=0, score=NaN
    short = np.array(too_short)
    for t in TOPICS:
        tc = "topic_" + t.replace(" ", "_"); sc = "score_" + t.replace(" ", "_")
        cols[tc][short] = 0
        cols[sc][short] = np.nan
    pd.DataFrame(cols).to_parquet(f"{pref}_topics.parquet", index=False, compression="zstd")
    print(f"  topics done. saved {pref}_topics.parquet  ({time.monotonic()-t0:.1f}s total)", flush=True)


if __name__ == "__main__":
    main()
