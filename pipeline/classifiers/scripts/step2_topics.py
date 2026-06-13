"""Шаг 2: 7 тем v3 на unique_lemmas (env topics311).

6 тем из binary_per_topic + НБП из CatBoost-стэкера.
Вход:  run_100chan/unique_lemmas.parquet [text_hash, lemmas]
Выход: run_100chan/unique_pred_topics.parquet [text_hash + 7 topic_* + 7 score_*]
"""
from __future__ import annotations

import os
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

os.environ.setdefault("OMP_NUM_THREADS", "4")
warnings.simplefilter("ignore")

RUN = Path("/Users/daniltarasov/Downloads/topic_classifier_production/run_100chan")
INPUT = RUN / "unique_lemmas.parquet"
OUTPUT = RUN / "unique_pred_topics.parquet"

BIN_MIX = Path("/Users/daniltarasov/Downloads/binary_per_topic_classifier/results/svc_per_topic_pipeline.joblib")
NBP_STACKER = Path("/Users/daniltarasov/Downloads/svc_focused_v2/results/nbp_v3/nbp_stacker.joblib")

# финальный порядок тем как в assemble_final_v3.py
TOPICS = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "НБП", "Финтех", "Геополитика", "НДО"]
BINARY_TOPICS = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "Финтех", "Геополитика", "НДО"]
BATCH = 20_000


def main() -> None:
    t0 = time.monotonic()
    import joblib
    import catboost
    print(f"catboost: {catboost.__version__}", flush=True)

    # ВАЖНО: binary_per_topic и svc_focused_v2 оба имеют пакет `src` с одинаковыми
    # именами классов. Грузим mix ПЕРВЫМ при binary-only пути — тогда sys.modules['src']
    # кэшируется на binary-версию (в ней есть ThresholdedBinaryClassifier). Потом добавляем
    # пути svc и грузим NBP (nbp_predictor находится по своему пути, а нужные классы берутся
    # из уже закэшированного src). Точно как в проверенном смоук-тесте.
    sys.path.insert(0, "/Users/daniltarasov/Downloads/binary_per_topic_classifier")
    sys.path.insert(0, "/Users/daniltarasov/Downloads/binary_per_topic_classifier/src")
    print(f"loading binary mix...", flush=True)
    mix = joblib.load(BIN_MIX)

    sys.path.insert(0, "/Users/daniltarasov/Downloads/svc_focused_v2")
    sys.path.insert(0, "/Users/daniltarasov/Downloads/svc_focused_v2/results/nbp_v3")
    sys.path.insert(0, "/Users/daniltarasov/Downloads/svc_focused_v2/src")
    print(f"loading NBP stacker...", flush=True)
    nbp = joblib.load(NBP_STACKER)
    print("thresholds:", flush=True)
    for t in BINARY_TOPICS:
        print(f"  {t:<28s} τ={getattr(mix[t],'threshold',0.0):+.4f}")
    print(f"  {'НБП':<28s} τ={nbp.threshold:+.4f}")

    pf = pq.ParquetFile(INPUT)
    n_total = pf.metadata.num_rows
    print(f"\nINPUT rows={n_total:,}", flush=True)

    fields = [pa.field("text_hash", pa.string())]
    for t in TOPICS:
        fields.append(pa.field("topic_" + t.replace(" ", "_"), pa.int8()))
    for t in TOPICS:
        fields.append(pa.field("score_" + t.replace(" ", "_"), pa.float32()))
    out_schema = pa.schema(fields)

    done = 0
    t_inf = time.monotonic()
    with pq.ParquetWriter(OUTPUT, out_schema, compression="zstd") as writer:
        for bi, batch in enumerate(pf.iter_batches(batch_size=BATCH, columns=["text_hash", "lemmas"])):
            d = batch.to_pydict()
            n = len(d["lemmas"])
            X = pd.Series([l or "" for l in d["lemmas"]])
            cols = {"text_hash": d["text_hash"]}
            # 6 бинарных
            for t in BINARY_TOPICS:
                m = mix[t]
                thr = getattr(m, "threshold", 0.0)
                s = np.asarray(m.decision_function(X), dtype="float32")
                cols["topic_" + t.replace(" ", "_")] = (s >= thr).astype("int8")
                cols["score_" + t.replace(" ", "_")] = s
            # НБП
            s = np.asarray(nbp.decision_function(X), dtype="float32")
            cols["topic_НБП"] = (s >= nbp.threshold).astype("int8")
            cols["score_НБП"] = s

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
