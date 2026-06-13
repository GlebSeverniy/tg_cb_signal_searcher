"""Шаг 4: финальная сборка.

Берём ВСЕ исходные колонки нового паркета (478,458 строк) и приклеиваем разметку:
  - text_hash, text_too_short                       (служебные, из row_keys)
  - 7×topic_* (int8) + 7×score_* (f32)              (по text_hash)
  - econom_class(int8), is_econom(int8), p_econom, p_c0..p_c5   (по text_hash)

too_short-строки (hash="__TOO_SHORT__"): topic=0, score=NaN, is_econom=0,
econom_class=-1 (служебный «не классифицирован»), p_*=NaN.

Выход: /Users/daniltarasov/Downloads/final_100chan_labeled.parquet
Env: topics311 (pandas + pyarrow24).
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

PROD = Path("/Users/daniltarasov/Downloads/topic_classifier_production")
RUN = PROD / "run_100chan"
SRC = Path("/Users/daniltarasov/Downloads/final parquet 100 chan")
ROW_KEYS = RUN / "row_keys.parquet"
PRED_TOPICS = RUN / "unique_pred_topics.parquet"
PRED_ECONOM = RUN / "unique_pred_econom.parquet"
OUT = Path("/Users/daniltarasov/Downloads/final_100chan_labeled.parquet")

TOPICS = ["Защита прав потребителей", "ДКП", "Финансовый рынок", "НБП", "Финтех", "Геополитика", "НДО"]
TCOLS = ["topic_" + t.replace(" ", "_") for t in TOPICS]
SCOLS = ["score_" + t.replace(" ", "_") for t in TOPICS]
PCOLS = [f"p_c{i}" for i in range(6)]


def main() -> None:
    t0 = time.monotonic()

    print("loading source (all 40 cols)...", flush=True)
    df = pd.read_parquet(SRC)
    n = len(df)
    print(f"  rows={n:,}, cols={df.shape[1]}", flush=True)

    print("attaching row_keys (text_hash, text_too_short)...", flush=True)
    rk = pd.read_parquet(ROW_KEYS)
    assert len(rk) == n, f"row_keys mismatch {len(rk)} vs {n}"
    # row_keys в исходном порядке (row_idx 0..n-1); приклеиваем по позиции
    rk = rk.sort_values("row_idx").reset_index(drop=True)
    df = df.reset_index(drop=True)
    df["text_hash"] = rk["text_hash"].values
    df["text_too_short"] = rk["text_too_short"].values

    print("loading predictions...", flush=True)
    pt = pd.read_parquet(PRED_TOPICS)
    pe = pd.read_parquet(PRED_ECONOM)
    print(f"  topics preds: {len(pt):,}  econom preds: {len(pe):,}", flush=True)

    print("merging by text_hash...", flush=True)
    df = df.merge(pt, on="text_hash", how="left")
    df = df.merge(pe, on="text_hash", how="left")

    # too_short / любые непросчитанные → дефолты
    for c in TCOLS:
        df[c] = df[c].fillna(0).astype("int8")
    for c in SCOLS:
        df[c] = df[c].astype("float32")  # NaN остаётся
    df["is_econom"] = df["is_econom"].fillna(0).astype("int8")
    df["econom_class"] = df["econom_class"].fillna(-1).astype("int8")
    df["p_econom"] = df["p_econom"].astype("float32")
    for c in PCOLS:
        df[c] = df[c].astype("float32")

    # --- save ---
    print(f"\nwriting {OUT.name} ...", flush=True)
    df.to_parquet(OUT, index=False, compression="zstd")
    print(f"  saved: {OUT} ({OUT.stat().st_size/1e6:.0f} MB)  rows={len(df):,} cols={df.shape[1]}", flush=True)

    # ── статистика ──
    print("\n" + "=" * 78)
    print(f"ВСЕГО строк: {n:,}")
    short_n = int(df["text_too_short"].sum())
    print(f"text_too_short: {short_n:,} ({100*short_n/n:.2f}%)  — без разметки")
    classified = n - short_n
    print(f"классифицировано (не короткие): {classified:,}")

    print(f"\nЭКОНОМИЧЕСКИЕ:")
    eco = int((df["is_econom"] == 1).sum())
    print(f"  is_econom=1: {eco:,} ({100*eco/n:.2f}% от всех, {100*eco/classified:.2f}% от классиф.)")

    print(f"\nТЕМЫ (positives, % от всех {n:,}):")
    for t, c in zip(TOPICS, TCOLS):
        pos = int((df[c] == 1).sum())
        print(f"  {t:<28s} {pos:>8,}  {100*pos/n:5.2f}%")

    active = pd.DataFrame({t: (df[c] == 1).astype("int8") for t, c in zip(TOPICS, TCOLS)})
    nper = active.sum(axis=1)
    any_topic = int((nper >= 1).sum())
    print(f"\nхотя бы 1 тема: {any_topic:,} ({100*any_topic/n:.2f}%)")
    print("multi-label:")
    for k in [0, 1, 2, 3]:
        cc = int((nper == k).sum())
        print(f"  {k} тем: {cc:>8,} ({100*cc/n:5.2f}%)")
    c4 = int((nper >= 4).sum())
    print(f"  4+ тем: {c4:>8,} ({100*c4/n:5.2f}%)")

    # пересечение тема ∪ эконом (это и есть «дешёвый префильтр»)
    topic_or_econom = int(((nper >= 1) | (df["is_econom"] == 1)).sum())
    topic_and_econom = int(((nper >= 1) & (df["is_econom"] == 1)).sum())
    print(f"\nтема ∪ эконом: {topic_or_econom:,} ({100*topic_or_econom/n:.2f}%)")
    print(f"тема ∩ эконом: {topic_and_econom:,} ({100*topic_and_econom/n:.2f}%)")

    print(f"\nDONE in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
