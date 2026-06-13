"""Шаг 1: prep + лемматизация нового 100-канального паркета.

Вход:  /Users/daniltarasov/Downloads/final parquet 100 chan  (478,458 строк)
Выход:
  run_100chan/row_keys.parquet      [row_idx, text_hash, text_too_short]  (по всем строкам, в исходном порядке)
  run_100chan/unique_lemmas.parquet [text_hash, lemmas]                   (только уникальные не-короткие тексты)

Логика хэша/коротких — 1-в-1 как в dedupe_and_collect.py:
  text_hash = blake2b(text.utf-8, digest_size=16).hexdigest()
  text is None или len(text) < 20  → text_too_short=True, hash="__TOO_SHORT__", НЕ лемматизируем.

Лемматизация — Pool(6), src.lemmatizer.lemmatize_text (та же, что в обучении).
Запускать в env topics311 (pymorphy3 + pyarrow).
"""
from __future__ import annotations

import hashlib
import multiprocessing as mp
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROD = Path("/Users/daniltarasov/Downloads/topic_classifier_production")
sys.path.insert(0, str(PROD))  # для src.lemmatizer

RUN = PROD / "run_100chan"
INPUT = Path("/Users/daniltarasov/Downloads/final parquet 100 chan")
ROW_KEYS = RUN / "row_keys.parquet"
UNIQUE_LEMMAS = RUN / "unique_lemmas.parquet"

TOO_SHORT_THRESHOLD = 20
N_WORKERS = 6
CHUNKSIZE = 300


def text_hash(t: str) -> str:
    return hashlib.blake2b(t.encode("utf-8", "replace"), digest_size=16).hexdigest()


def _lemmatize(args):
    idx, text = args
    from src.lemmatizer import lemmatize_text  # lazy per-process
    return idx, lemmatize_text(text or "")


def _init_worker():
    from src.lemmatizer import get_morph
    get_morph()


def main() -> None:
    t0 = time.monotonic()
    print(f"INPUT: {INPUT.name}", flush=True)

    # ── 1) один проход: hashes + too_short + сбор уникальных текстов ──
    tbl = pq.read_table(INPUT, columns=["text"])
    texts = tbl.column("text").to_pylist()
    n = len(texts)
    print(f"rows: {n:,}", flush=True)

    hashes = [""] * n
    too_short = [False] * n
    seen: dict[str, str] = {}  # hash → text (только не-короткие)
    too_short_n = 0
    for i in range(n):
        t = texts[i]
        if t is None or len(t) < TOO_SHORT_THRESHOLD:
            hashes[i] = "__TOO_SHORT__"
            too_short[i] = True
            too_short_n += 1
        else:
            h = text_hash(t)
            hashes[i] = h
            if h not in seen:
                seen[h] = t

    uniq = len(seen)
    print(f"too short/NULL: {too_short_n:,} ({100*too_short_n/n:.2f}%)", flush=True)
    print(f"unique non-short texts: {uniq:,}  (dedup {(n-too_short_n)/max(1,uniq):.2f}x)", flush=True)

    # row_keys.parquet
    rk_schema = pa.schema([
        pa.field("row_idx", pa.int64()),
        pa.field("text_hash", pa.string()),
        pa.field("text_too_short", pa.bool_()),
    ])
    pq.write_table(pa.table({
        "row_idx": list(range(n)),
        "text_hash": hashes,
        "text_too_short": too_short,
    }, schema=rk_schema), ROW_KEYS, compression="zstd")
    print(f"saved row_keys: {ROW_KEYS.name}", flush=True)

    # ── 2) лемматизация уникальных ──
    uniq_hashes = list(seen.keys())
    uniq_texts = list(seen.values())
    lemmas = [""] * uniq
    print(f"\nlemmatizing {uniq:,} unique texts, workers={N_WORKERS}...", flush=True)
    args = ((i, uniq_texts[i]) for i in range(uniq))
    done = 0
    t_lem = time.monotonic()
    last = t_lem
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        for idx, lemma in pool.imap_unordered(_lemmatize, args, chunksize=CHUNKSIZE):
            lemmas[idx] = lemma
            done += 1
            now = time.monotonic()
            if now - last > 10:
                el = now - t_lem
                rate = done / el if el > 0 else 0
                eta = (uniq - done) / rate / 60 if rate > 0 else 0
                print(f"  {done:>8,}/{uniq:,} ({100*done/uniq:5.1f}%)  "
                      f"{rate:>5,.0f} t/s  eta {eta:4.1f}m  elapsed {el/60:4.1f}m", flush=True)
                last = now

    out_schema = pa.schema([pa.field("text_hash", pa.string()), pa.field("lemmas", pa.string())])
    pq.write_table(pa.table({"text_hash": uniq_hashes, "lemmas": lemmas}, schema=out_schema),
                   UNIQUE_LEMMAS, compression="zstd")
    non_empty = sum(1 for l in lemmas if l)
    print(f"\nsaved unique_lemmas: {UNIQUE_LEMMAS.name} ({UNIQUE_LEMMAS.stat().st_size/1e6:.1f} MB)")
    print(f"non-empty lemmas: {non_empty:,}/{uniq:,} ({100*non_empty/uniq:.2f}%)")
    print(f"DONE in {(time.monotonic()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
