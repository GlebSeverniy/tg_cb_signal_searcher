"""Стратифицированный train/test сплит BERT-корпуса.

Тест = 20%, стратификация по комбинации меток (t1..t5).
Негативы (00000) — отдельный страт. Это сохраняет одновременно:
  - долю релевантных/нерелевантных,
  - маргинальное распределение каждой из 5 тем,
  - совместную встречаемость тем.
Из каждой комбинации в тест берётся round(20%) постов (seed=42), остальное в train.
"""
from __future__ import annotations
import pandas as pd, numpy as np
from pathlib import Path

DL = Path("/Users/daniltarasov/Downloads")
SRC = DL / "bert_relevance_corpus.parquet"
TRAIN = DL / "bert_relevance_corpus_train.parquet"
TEST = DL / "bert_relevance_corpus_test.parquet"
T = [f"t{i}_relevant" for i in range(1, 6)]
SEED = 42
TEST_FRAC = 0.20


def main():
    df = pd.read_parquet(SRC).reset_index(drop=True)
    n = len(df)
    combo = df[T].astype(int).astype(str).agg("".join, axis=1)

    test_idx = []
    for c, grp in df.groupby(combo):
        k = int(round(TEST_FRAC * len(grp)))
        if k > 0:
            test_idx.extend(grp.sample(n=k, random_state=SEED).index.tolist())
    test_idx = set(test_idx)

    df["split"] = np.where(df.index.isin(test_idx), "test", "train")
    train = df[df["split"] == "train"].drop(columns="split").reset_index(drop=True)
    test = df[df["split"] == "test"].drop(columns="split").reset_index(drop=True)
    train.to_parquet(TRAIN, index=False, compression="zstd")
    test.to_parquet(TEST, index=False, compression="zstd")

    # ── verification ──
    def stats(d):
        rel = int((d["relevant"] == 1).sum())
        return rel, len(d), 100*rel/len(d)

    print(f"{'set':<8}{'постов':>9}{'релев.':>9}{'%релев':>9}")
    for nm, d in [("FULL", df), ("train", train), ("test", test)]:
        rel, tot, pct = stats(d)
        print(f"{nm:<8}{tot:>9,}{rel:>9,}{pct:>8.2f}%")

    print(f"\ntest доля от полного: {100*len(test)/n:.2f}%  (релевантных в тесте: {100*int((test['relevant']==1).sum())/int((df['relevant']==1).sum()):.1f}% от всех релевантных)")

    # баланс по 5 темам: доля темы среди ВСЕХ постов (должна совпадать full/train/test)
    print(f"\n{'тема':<14}{'full %':>9}{'train %':>9}{'test %':>9}{'test n':>8}")
    for t in T:
        f_ = 100*df[t].mean(); tr = 100*train[t].mean(); te = 100*test[t].mean()
        print(f"{t:<14}{f_:>8.3f}%{tr:>8.3f}%{te:>8.3f}%{int(test[t].sum()):>8}")

    # баланс по source (не стратифицировали — проверим, что не уехало)
    print(f"\n{'source':<8}{'full %':>9}{'train %':>9}{'test %':>9}")
    for s in sorted(df["source"].unique()):
        f_=100*(df["source"]==s).mean(); tr=100*(train["source"]==s).mean(); te=100*(test["source"]==s).mean()
        print(f"{s:<8}{f_:>8.2f}%{tr:>8.2f}%{te:>8.2f}%")

    # integrity
    import os
    print("\nintegrity:")
    print(f"  train+test = {len(train)+len(test):,} (== {n:,}: {len(train)+len(test)==n})")
    overlap = set(zip(train.channel_id,train.message_id)) & set(zip(test.channel_id,test.message_id))
    print(f"  пересечение train/test по ключу: {len(overlap)}")
    print(f"  saved: {TRAIN.name} ({os.path.getsize(TRAIN)/1e6:.1f} MB), {TEST.name} ({os.path.getsize(TEST)/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
