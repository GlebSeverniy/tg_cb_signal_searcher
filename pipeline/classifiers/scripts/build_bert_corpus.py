"""Сборка единого корпуса для обучения BERT (релевантность).

Источник постов (и негативов) — final_100chan_labeled.parquet (полные подневные вырезки).
LLM-разметка вшивается по (channel_id, message_id).

5 источников, даты не пересекаются:
  7days  : 7 дат, union → полная вырезка 19,851
  7days2 : 7 дат, union → полная вырезка 17,202 (5,454 размечено)
  5days  : 5 дат, union → полная вырезка 11,534
  dec15  : 15–19.12     → полная вырезка 10,357
  apr26  : 25–27.04     → уже полная   7,236 (все annotated)

Единая разметка t1–t5:
  dec15/7days/7days2/5days — t1–t5 как есть (одни и те же 5 тем).
  apr26 — СНАЧАЛА полный обмен t3↔t4, ПОТОМ удаляем t6 → t1–t5 совпадают с остальными.

Негативы (посты вне разметки на тех днях): t1–t5=False, relevant=0, annotated=False.
"""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

DL = Path("/Users/daniltarasov/Downloads")
LAB = DL / "final_100chan_labeled.parquet"
OUT = DL / "bert_relevance_corpus.parquet"

SRC = {
 "7days": (DL/"cbr_dec15_perception_haiku_relonly_recall_7days/results/annotated_7days.parquet",
           ["2023-10-15","2023-10-27","2024-01-16","2024-02-16","2024-03-01","2024-03-22","2024-04-03"]),
 "7days2": (DL/"cbr_dec15_perception_haiku_relonly_recall_7days_2/results/annotated_7days2.parquet",
           ["2023-11-17","2023-11-30","2024-01-28","2024-03-13","2024-03-27","2024-03-28","2024-04-15"]),
 "5days": (DL/"cbr_dec15_perception_haiku_relonly_recall_5days/results/annotated_5days.parquet",
           ["2023-11-11","2024-01-22","2024-02-22","2024-03-07","2024-04-09"]),
 "dec15": (DL/"annotated-sample-15-12-2023.parquet",
           ["2023-12-15","2023-12-16","2023-12-17","2023-12-18","2023-12-19"]),
 "apr26": (Path("/Users/daniltarasov/Library/Containers/ru.keepcoder.Telegram/Data/tmp/annotated_26-04-24.parquet"),
           ["2024-04-25","2024-04-26","2024-04-27"]),
}

T = [f"t{i}_relevant" for i in range(1,6)]   # унифицированные 5 тем
KEEP_ORIG = ["channel_id","channel_name","channel_type","message_id","message_type",
             "post_date","post_date_unixtime","edited","edited_unixtime","from_id",
             "text","text_entities","reactions_total","has_reactions","reactions",
             "has_media","poll","inline_bot_buttons","source_file","text_hash"]


def get_markup(name, path):
    """Вернуть [channel_id, message_id, t1..t5, reasoning, annotated] из размеченного файла."""
    df = pd.read_parquet(path)
    if name == "apr26":
        # полный обмен t3<->t4, затем удалить t6
        old_t3 = df["t3_relevant"].copy()
        df["t3_relevant"] = df["t4_relevant"]
        df["t4_relevant"] = old_t3
        # t6 удаляется тем, что просто не берём его в T
        # проверка наличия t6
        assert "t6_relevant" in df.columns
    for t in T:
        df[t] = df[t].fillna(False).astype(bool)
    if "annotated" not in df.columns:
        df["annotated"] = True
    rs = df["reasoning"] if "reasoning" in df.columns else pd.Series([None]*len(df))
    out = df[["channel_id","message_id"]].copy()
    for t in T:
        out[t] = df[t].values
    out["reasoning"] = rs.values
    out["annotated_llm"] = df["annotated"].fillna(False).astype(bool).values
    return out


def main():
    lab = pd.read_parquet(LAB, columns=KEEP_ORIG)
    lab["_d"] = lab["post_date"].astype(str).str[:10]
    print(f"100chan: {len(lab):,} rows")

    parts = []
    print(f"\n{'src':<7}{'полн.вырезка':>13}{'размечено':>11}{'relevant(t1-5)':>15}")
    for name,(path,dates) in SRC.items():
        full = lab[lab["_d"].isin(dates)].drop(columns="_d").copy()
        mk = get_markup(name, path)
        # left join разметки
        m = full.merge(mk, on=["channel_id","message_id"], how="left", validate="one_to_one")
        # негативы: незаматченные → t=False, annotated=False
        matched = m["annotated_llm"].notna()
        for t in T:
            m[t] = m[t].fillna(False).astype(bool)
        m["annotated"] = m["annotated_llm"].fillna(False).astype(bool)
        m.drop(columns=["annotated_llm"], inplace=True)
        m["relevant"] = m[T].any(axis=1).astype("int8")
        m["source"] = name
        parts.append(m)
        print(f"{name:<7}{len(full):>13,}{int(matched.sum()):>11,}{int(m['relevant'].sum()):>15,}")

    corpus = pd.concat(parts, ignore_index=True)

    # порядок колонок
    cols = KEEP_ORIG + ["source","annotated"] + T + ["relevant","reasoning"]
    corpus = corpus[cols]
    corpus.to_parquet(OUT, index=False, compression="zstd")

    import os
    print(f"\nSAVED: {OUT}  ({os.path.getsize(OUT)/1e6:.1f} MB)")
    print(f"rows: {len(corpus):,}  cols: {corpus.shape[1]}")
    print(f"relevant=1: {int((corpus['relevant']==1).sum()):,}  relevant=0: {int((corpus['relevant']==0).sum()):,}")
    print(f"annotated=True (LLM видел): {int(corpus['annotated'].sum()):,}  | annotated=False (негатив из фильтра): {int((~corpus['annotated']).sum()):,}")
    # негативы по типу
    hard_neg = int(((corpus['relevant']==0) & (corpus['annotated'])).sum())
    filt_neg = int(((corpus['relevant']==0) & (~corpus['annotated'])).sum())
    print(f"  из негативов: LLM-проверенные (annotated, не релев)={hard_neg:,} | отфильтрованные (не видел LLM)={filt_neg:,}")
    print("\nпо темам (t1-t5):")
    for t in T:
        print(f"  {t}: {int(corpus[t].sum()):,}")
    print("\nпо источникам:")
    print(corpus.groupby("source").agg(n=("relevant","size"), rel=("relevant","sum")).to_string())
    # integrity
    print("\nintegrity:")
    print("  dup (channel_id,message_id):", int(corpus.duplicated(["channel_id","message_id"]).sum()))
    print("  null text:", int(corpus["text"].isna().sum()))


if __name__ == "__main__":
    main()
