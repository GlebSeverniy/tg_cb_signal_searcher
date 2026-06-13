"""
Data loading + sanity checks.

v2: ИСКЛЮЧЕНЫ темы Ковид, Другое, НПС. В binary per-topic подходе это означает,
что их бинарные столбцы никогда не используются как y. Строки НЕ удаляются —
все 4371 train / 1093 test строки используются как негативные примеры для
каждого из 8 разрешённых топиков.

Data structure: multi-label binary columns (e.g. df["НБП"] = 0/1).
Topics column (e.g. "ДКП; Финрынки и банки") is a convenience string only.

Same train/test files as v1 (frozen split).
Prefers train_lemmas.parquet / test_lemmas.parquet for speed + lemma support.

Column alias map (data → display name):
  ЗПП и ФГН        → Защита прав потребителей
  Финрынки и банки → Финансовый рынок
  (others same)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple, List, Optional

import pandas as pd

from . import EXCLUDED_TOPICS, ALLOWED_TOPICS, TOPIC_COL_MAP, COL_TOPIC_MAP

DEFAULT_DATA_DIR = Path("/Users/daniltarasov/Downloads/svc_baseline")

# Preferred train/test file names (in order of preference)
_TRAIN_PREFERENCE = ["train_lemmas.parquet", "train.parquet", "train.csv", "train.tsv"]
_TEST_PREFERENCE = ["test_lemmas.parquet", "test.parquet", "test.csv", "test.tsv"]


def get_data_dir() -> Path:
    return Path(os.environ.get("SVC_DATA_DIR", str(DEFAULT_DATA_DIR)))


def _find_preferred(data_dir: Path, preference_list: List[str]) -> Path:
    """Return first file from preference_list that exists in data_dir."""
    for name in preference_list:
        p = data_dir / name
        if p.exists():
            return p
    # Fallback: list what's there
    existing = sorted(p.name for p in data_dir.iterdir() if p.is_file())
    raise FileNotFoundError(
        f"None of {preference_list} found in {data_dir}. "
        f"Existing files: {existing}"
    )


def _read_any(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".csv":
        return pd.read_csv(path)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t")
    if s == ".xlsx":
        return pd.read_excel(path)
    if s == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file format: {path}")


def _get_text_col(df: pd.DataFrame) -> str:
    """Prefer 'lemmas' (pre-lemmatized) over 'text' (raw)."""
    for col in ("lemmas", "text", "message", "content", "post", "body"):
        if col in df.columns:
            return col
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if not obj_cols:
        raise RuntimeError(f"No text column found. Columns: {df.columns.tolist()}")
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


def _validate_binary_cols(df: pd.DataFrame) -> None:
    """Assert that all expected binary topic columns are present."""
    missing = [col for col in TOPIC_COL_MAP if col not in df.columns]
    if missing:
        raise RuntimeError(
            f"Expected binary topic columns missing: {missing}. "
            f"Available: {df.columns.tolist()}"
        )


def load_raw(split: str = "train") -> pd.DataFrame:
    """
    Load raw dataframe (no filtering). split='train' or 'test'.
    Returns full dataframe with all binary topic columns + text/lemmas.
    """
    data_dir = get_data_dir()
    if split == "train":
        path = _find_preferred(data_dir, _TRAIN_PREFERENCE)
    elif split == "test":
        path = _find_preferred(data_dir, _TEST_PREFERENCE)
    else:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    df = _read_any(path)
    _validate_binary_cols(df)
    print(f"  loaded {split} from {path.name}: {len(df)} rows", flush=True)
    return df


def load_train_binary(topic: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load train data for a single binary topic classifier.

    Args:
        topic: canonical topic name from ALLOWED_TOPICS
               (e.g. 'Защита прав потребителей', 'НБП', 'Финансовый рынок')

    Returns:
        (X, y) where:
          X — text/lemmas Series (all 4371 rows, reset_index)
          y — binary 0/1 Series (1 = positive for this topic)
    """
    if topic not in COL_TOPIC_MAP:
        raise ValueError(
            f"Topic '{topic}' not in ALLOWED_TOPICS. "
            f"Allowed: {ALLOWED_TOPICS}"
        )
    df = load_raw("train")
    tc = _get_text_col(df)
    col = COL_TOPIC_MAP[topic]  # data column name
    X = df[tc].astype(str).fillna("").reset_index(drop=True)
    y = df[col].astype(int).reset_index(drop=True)
    n_pos = int(y.sum())
    print(f"  train_binary({topic}): {len(X)} rows, {n_pos} positives", flush=True)
    return X, y


def load_test_binary(topic: str) -> Tuple[pd.Series, pd.Series]:
    """
    Load test data for a single binary topic classifier.

    Args:
        topic: canonical topic name from ALLOWED_TOPICS

    Returns:
        (X, y) where X is text/lemmas and y is 0/1 binary label.
    """
    if topic not in COL_TOPIC_MAP:
        raise ValueError(
            f"Topic '{topic}' not in ALLOWED_TOPICS. "
            f"Allowed: {ALLOWED_TOPICS}"
        )
    df = load_raw("test")
    tc = _get_text_col(df)
    col = COL_TOPIC_MAP[topic]
    X = df[tc].astype(str).fillna("").reset_index(drop=True)
    y = df[col].astype(int).reset_index(drop=True)
    n_pos = int(y.sum())
    print(f"  test_binary({topic}): {len(X)} rows, {n_pos} positives", flush=True)
    return X, y


def load_train() -> Tuple[pd.Series, pd.Series]:
    """
    Load train X and y_multiclass for compatibility.

    y_multiclass uses the primary (first) topic from the 'topics' column
    for single-label rows. Multi-label rows get only their first topic.

    IMPORTANT: For binary per-topic classification, prefer load_train_binary(topic).
    This function is kept for pipeline compatibility but binary loading is canonical.

    Excluded topics (Ковид, Другое, НПС): rows where primary topic is excluded
    are dropped. Multi-label rows where any allowed topic = 1 are kept.
    """
    df = load_raw("train")
    tc = _get_text_col(df)

    # Build primary label from binary columns (first allowed topic column = 1)
    # This preserves the correct binary structure for ALL rows including multi-label
    # For single-label use: take the 'topics' column primary topic, mapped to display name
    # Filter out rows where primary topic is excluded
    alias_inv = {v: k for k, v in {
        "ЗПП и ФГН": "Защита прав потребителей",
        "Финрынки и банки": "Финансовый рынок",
        "ДКП в мире": "ДКП в мире",
        "ДКП": "ДКП",
        "НБП": "НБП",
        "Финтех": "Финтех",
        "Геополитика": "Геополитика",
        "НДО": "НДО",
    }.items()}

    # Get primary topic from 'topics' column (first semicolon-split part)
    primary = df["topics"].astype(str).str.split(";").str[0].str.strip()

    # Map data alias → display name; unmapped = excluded or unknown
    excluded_cols = {"Ковид", "Другое", "НПС"}
    data_to_display = {
        "ЗПП и ФГН": "Защита прав потребителей",
        "Финрынки и банки": "Финансовый рынок",
        "ДКП в мире": "ДКП в мире",
        "ДКП": "ДКП",
        "НБП": "НБП",
        "Финтех": "Финтех",
        "Геополитика": "Геополитика",
        "НДО": "НДО",
    }
    label = primary.map(data_to_display)

    # Keep only rows with a valid (non-excluded, non-NA) allowed topic as primary
    mask = label.notna()
    df_filtered = df[mask].copy().reset_index(drop=True)
    label_filtered = label[mask].reset_index(drop=True)

    before = len(df)
    after = len(df_filtered)
    print(
        f"  load_train (multiclass): {before} → {after} rows "
        f"({before - after} excluded-primary removed)",
        flush=True,
    )

    # Sanity: only allowed topics remain
    remaining = set(label_filtered.unique())
    unexpected = remaining - set(ALLOWED_TOPICS)
    if unexpected:
        raise RuntimeError(f"Unexpected labels after filter: {unexpected}")

    X = df_filtered[tc].astype(str).fillna("").reset_index(drop=True)
    y = label_filtered.reset_index(drop=True)
    return X, y


def load_test() -> Tuple[pd.Series, pd.Series]:
    """
    Load test X and y_multiclass. See load_train() docstring.
    For binary per-topic classification, prefer load_test_binary(topic).
    """
    df = load_raw("test")
    tc = _get_text_col(df)

    data_to_display = {
        "ЗПП и ФГН": "Защита прав потребителей",
        "Финрынки и банки": "Финансовый рынок",
        "ДКП в мире": "ДКП в мире",
        "ДКП": "ДКП",
        "НБП": "НБП",
        "Финтех": "Финтех",
        "Геополитика": "Геополитика",
        "НДО": "НДО",
    }
    primary = df["topics"].astype(str).str.split(";").str[0].str.strip()
    label = primary.map(data_to_display)
    mask = label.notna()
    df_filtered = df[mask].copy().reset_index(drop=True)
    label_filtered = label[mask].reset_index(drop=True)

    before = len(df)
    after = len(df_filtered)
    print(
        f"  load_test (multiclass): {before} → {after} rows "
        f"({before - after} excluded-primary removed)",
        flush=True,
    )

    remaining = set(label_filtered.unique())
    unexpected = remaining - set(ALLOWED_TOPICS)
    if unexpected:
        raise RuntimeError(f"Unexpected labels after filter: {unexpected}")

    X = df_filtered[tc].astype(str).fillna("").reset_index(drop=True)
    y = label_filtered.reset_index(drop=True)
    return X, y


def get_topics() -> list:
    """Список 8 разрешённых тем (в порядке из appendix7_targets.json)."""
    return list(ALLOWED_TOPICS)


def describe() -> None:
    data_dir = get_data_dir()
    print(f"=== data_dir = {data_dir} ===", flush=True)
    print(f"Files: {sorted(p.name for p in data_dir.iterdir() if p.is_file())}", flush=True)

    print("\n--- Binary per-topic stats ---", flush=True)
    df_tr = load_raw("train")
    df_te = load_raw("test")
    tc_tr = _get_text_col(df_tr)
    tc_te = _get_text_col(df_te)
    print(f"Text column used (train): {tc_tr}", flush=True)
    print(f"Text column used (test):  {tc_te}", flush=True)
    print(f"\n{'Topic':<35} {'DataCol':<20} {'Train+':>8} {'Test+':>7}", flush=True)
    for topic in ALLOWED_TOPICS:
        col = COL_TOPIC_MAP[topic]
        tr_pos = int(df_tr[col].sum())
        te_pos = int(df_te[col].sum())
        print(f"  {topic:<33} {col:<20} {tr_pos:>8} {te_pos:>7}", flush=True)

    print(f"\n--- Multiclass (primary topic) stats ---", flush=True)
    X_tr, y_tr = load_train()
    X_te, y_te = load_test()
    print(f"\nTrain shape (after filter): {len(X_tr)}", flush=True)
    print(f"Test shape (after filter):  {len(X_te)}", flush=True)
    print(f"\nTrain label distribution:", flush=True)
    print(y_tr.value_counts().to_string(), flush=True)
    print(f"\nTest label distribution:", flush=True)
    print(y_te.value_counts().to_string(), flush=True)

    lens_tr = X_tr.str.len()
    print(
        f"\nText length on train: mean={lens_tr.mean():.1f}, "
        f"median={lens_tr.median():.0f}, p95={lens_tr.quantile(0.95):.0f}",
        flush=True,
    )

    # Disjoint check via pk
    if "pk" in df_tr.columns and "pk" in df_te.columns:
        overlap = set(df_tr["pk"]) & set(df_te["pk"])
        print(f"\nPK overlap between train/test: {len(overlap)}", flush=True)


if __name__ == "__main__":
    describe()
