"""Применение переразметки тем t1/t4 к BERT-корпусу.

Вход:
  bert_relevance_corpus.parquet            — текущий фул (66,180)
  revise_t1t4/results/annotated_revised.parquet — пересмотр t1/t4 для 5,257 релевантных постов

Что делаем:
  по (channel_id, message_id) подменяем t1_relevant и t4_relevant на пересмотренные
  (только снятие флагов: t1 −813, t4 −2053, добавлений нет). t2/t3/t5 не трогаем.
  relevant = OR(t1..t5) пересчитываем. Посты, потерявшие последний флаг, становятся
  жёсткими негативами (annotated=True, relevant=0) — остаются в корпусе.
  Остальные колонки (в т.ч. reasoning) без изменений.

Выход: перезапись bert_relevance_corpus.parquet (новый канонический фул).
Пред-ревизионный снапшот сохранён в папке bert_corpus_v2_5src/.
"""
from __future__ import annotations
import pandas as pd, numpy as np, os
from pathlib import Path

DL = Path("/Users/daniltarasov/Downloads")
CORP = DL / "bert_relevance_corpus.parquet"
REV = DL / "revise_t1t4/results/annotated_revised.parquet"
T = [f"t{i}_relevant" for i in range(1, 6)]


def main():
    corp = pd.read_parquet(CORP)
    rev = pd.read_parquet(REV)
    n0 = len(corp)
    rel0 = int((corp["relevant"] == 1).sum())

    rv = rev[["channel_id", "message_id"]].copy()
    rv["t1_new"] = rev["t1_relevant"].fillna(False).astype(bool).values
    rv["t4_new"] = rev["t4_relevant"].fillna(False).astype(bool).values
    assert rv.duplicated(["channel_id", "message_id"]).sum() == 0

    corp = corp.merge(rv, on=["channel_id", "message_id"], how="left", validate="one_to_one")
    matched = corp["t1_new"].notna()
    assert int(matched.sum()) == len(rv), "не все пересмотренные посты нашлись"

    # снимки до
    t1_before = int(corp["t1_relevant"].sum())
    t4_before = int(corp["t4_relevant"].sum())

    # применяем только на совпавших; вне ревизии — оставляем как есть
    corp.loc[matched, "t1_relevant"] = corp.loc[matched, "t1_new"].astype(bool)
    corp.loc[matched, "t4_relevant"] = corp.loc[matched, "t4_new"].astype(bool)
    corp.drop(columns=["t1_new", "t4_new"], inplace=True)
    for t in T:
        corp[t] = corp[t].astype(bool)

    corp["relevant"] = corp[T].any(axis=1).astype("int8")

    rel1 = int((corp["relevant"] == 1).sum())
    dropped = rel0 - rel1

    # сохраняем порядок колонок как был
    corp.to_parquet(CORP, index=False, compression="zstd")

    print(f"rows: {n0:,} (без изменений: {len(corp)==n0})")
    print(f"relevant: {rel0:,} -> {rel1:,}  (стало негативами: {dropped:,})")
    print(f"t1: {t1_before:,} -> {int(corp['t1_relevant'].sum()):,}")
    print(f"t4: {t4_before:,} -> {int(corp['t4_relevant'].sum()):,}")
    print("\nтемы после ревизии (t1-t5):")
    for t in T:
        print(f"  {t}: {int(corp[t].sum()):,}")
    # негативы по типу
    hard = int(((corp['relevant']==0) & (corp['annotated'])).sum())
    filt = int(((corp['relevant']==0) & (~corp['annotated'])).sum())
    print(f"\nжёсткие негативы (annotated, rel=0): {hard:,}")
    print(f"отфильтрованные негативы (annotated=False): {filt:,}")
    print(f"annotated=True всего: {int(corp['annotated'].sum()):,}")
    print("\nintegrity:")
    print("  dup keys:", int(corp.duplicated(['channel_id','message_id']).sum()))
    print("  null text:", int(corp['text'].isna().sum()))
    print(f"  saved: {CORP.name} ({os.path.getsize(CORP)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
