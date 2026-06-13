"""
Data loading + sanity checks. Same data as the multiclass project.

DATA_DIR is /Users/daniltarasov/Downloads/svc_baseline/ unless overridden via env var SVC_DATA_DIR.
The split is FROZEN — no code re-splits.

Decision (Шаг 1): use *_lemmas.parquet files.
  - train_lemmas.parquet contains: lemmatized text in 'lemmas' col + 11 binary indicator columns + 'text' raw.
  - test_lemmas.parquet  contains the same structure.
  - y_binary is a DataFrame of 11 indicator columns (0/1), renamed to canonical author names.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Tuple, List

import pandas as pd


DEFAULT_DATA_DIR = Path("/Users/daniltarasov/Downloads/svc_baseline")

# Canonical topics order from appendix7_targets.json::topics_order
CANONICAL_TOPICS_ORDER = [
    "Защита прав потребителей",
    "ДКП в мире",
    "ДКП",
    "Финансовый рынок",
    "НБП",
    "Другое",
    "Финтех",
    "Геополитика",
    "Ковид",
    "НПС",
    "НДО",
]

# Map: our parquet column name → canonical author name
# Source: experiments/topic_alias_map.json (our_label_to_authors_canonical)
OUR_COL_TO_CANONICAL = {
    "ЗПП и ФГН":        "Защита прав потребителей",
    "ДКП в мире":       "ДКП в мире",
    "ДКП":              "ДКП",
    "Финрынки и банки": "Финансовый рынок",
    "НБП":              "НБП",
    "Другое":           "Другое",
    "Финтех":           "Финтех",
    "Геополитика":      "Геополитика",
    "Ковид":            "Ковид",
    "НПС":              "НПС",
    "НДО":              "НДО",
}

# Expected train-set positives per canonical topic (from discovery_report).
# Used for sanity check — raises RuntimeError on mismatch.
EXPECTED_TRAIN_POSITIVES = {
    "Защита прав потребителей": 255,
    "ДКП в мире":               821,
    "ДКП":                      1156,
    "Финансовый рынок":         1601,
    "НБП":                      160,
    "Другое":                   458,
    "Финтех":                   198,
    "Геополитика":              231,
    "Ковид":                    39,
    "НПС":                      62,
    "НДО":                      63,
}

EXPECTED_TEST_POSITIVES = {
    "Защита прав потребителей": 64,
    "ДКП в мире":               205,
    "ДКП":                      289,
    "Финансовый рынок":         400,
    "НБП":                      40,
    "Другое":                   115,
    "Финтех":                   49,
    "Геополитика":              58,
    "Ковид":                    10,
    "НПС":                      15,
    "НДО":                      16,
}

# Priority-ordered file preferences per split key.
# First match that exists wins. We prefer *_lemmas.parquet > plain .parquet > .csv.
_FILE_PRIORITY = {
    "train": ["train_lemmas.parquet", "train.parquet", "train.csv"],
    "test":  ["test_lemmas.parquet",  "test.parquet",  "test.csv"],
}


def get_data_dir() -> Path:
    return Path(os.environ.get("SVC_DATA_DIR", str(DEFAULT_DATA_DIR)))


def _find_file(data_dir: Path, split: str) -> Path:
    """Return path for 'train' or 'test', preferring lemmatized parquet."""
    priority = _FILE_PRIORITY.get(split)
    if priority is None:
        raise ValueError(f"split must be 'train' or 'test', got '{split}'")
    for name in priority:
        p = data_dir / name
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No data file found for split='{split}' in {data_dir}. "
        f"Looked for: {priority}. "
        f"Available files: {sorted(p.name for p in data_dir.iterdir() if p.is_file())}"
    )


def _read_parquet_or_csv(path: Path) -> pd.DataFrame:
    s = path.suffix.lower()
    if s == ".parquet":
        return pd.read_parquet(path)
    if s == ".csv":
        return pd.read_csv(path)
    if s == ".tsv":
        return pd.read_csv(path, sep="\t")
    if s == ".xlsx":
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _extract_binary_y(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract the 11 binary indicator columns, rename to canonical names,
    and order according to CANONICAL_TOPICS_ORDER.
    Returns a DataFrame with int8 values, columns = CANONICAL_TOPICS_ORDER.
    """
    our_cols = list(OUR_COL_TO_CANONICAL.keys())
    missing = [c for c in our_cols if c not in df.columns]
    if missing:
        raise RuntimeError(
            f"Binary indicator columns missing from data: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )
    y = df[our_cols].rename(columns=OUR_COL_TO_CANONICAL)
    # Reorder to canonical order
    y = y[CANONICAL_TOPICS_ORDER].astype(int)
    return y.reset_index(drop=True)


def _extract_text(df: pd.DataFrame, text_col: str = "lemmas") -> pd.Series:
    """Return the requested column, falling back to 'lemmas' then 'text'."""
    if text_col in df.columns:
        return df[text_col].astype(str).fillna("").reset_index(drop=True)
    if "lemmas" in df.columns:
        return df["lemmas"].astype(str).fillna("").reset_index(drop=True)
    if "text" in df.columns:
        return df["text"].astype(str).fillna("").reset_index(drop=True)
    raise RuntimeError(
        f"Column '{text_col}' (and fallbacks 'lemmas'/'text') not found. "
        f"Columns: {df.columns.tolist()}"
    )


_MYSTEM_DATA_DIR = Path("/Users/daniltarasov/Downloads/binary_per_topic_classifier/data_mystem")
_ACRONYM_DATA_DIR = Path("/Users/daniltarasov/Downloads/binary_per_topic_classifier/data_acronym")


def _sanity_check_y(y_binary: pd.DataFrame, expected: dict, split: str) -> None:
    """Raise RuntimeError if per-topic positive counts deviate from expected."""
    actual = y_binary.sum().to_dict()
    diffs = {
        topic: (actual.get(topic, 0), exp)
        for topic, exp in expected.items()
        if actual.get(topic, 0) != exp
    }
    if diffs:
        lines = [f"  {t}: got {a}, expected {e}" for t, (a, e) in sorted(diffs.items())]
        raise RuntimeError(
            f"Sanity check FAILED for {split} set — per-topic positive counts differ:\n"
            + "\n".join(lines)
        )


# ---------------------------------------------------------------------------
# Public API (new binary-aware interface)
# ---------------------------------------------------------------------------

def load_train_binary(lemmatizer: str = "pymorphy3") -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns (X, y_binary).

    X: pd.Series of lemmatized texts.
    y_binary: pd.DataFrame with one column per canonical topic name,
              ordered per appendix7_targets.json::topics_order, values 0/1.

    lemmatizer:
        "pymorphy3" (default) — reads from train_lemmas.parquet, column 'lemmas'.
        "mystem"              — reads from data_mystem/train_mystem.parquet, column 'lemmas_mystem'.
                                Binary indicator columns are still read from the same parquet
                                (they are included in train_mystem.parquet as well).
    """
    if lemmatizer == "pymorphy3":
        path = _find_file(get_data_dir(), "train")
        df = _read_parquet_or_csv(path)
        X = _extract_text(df, text_col="lemmas")
    elif lemmatizer == "mystem":
        path = _MYSTEM_DATA_DIR / "train_mystem.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"MyStem train parquet not found: {path}. "
                "Run experiments/build_mystem_lemmas.py first."
            )
        df = pd.read_parquet(path)
        X = _extract_text(df, text_col="lemmas_mystem")
    elif lemmatizer == "pymorphy3_acronym":
        path = _ACRONYM_DATA_DIR / "train_acronym.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Acronym-augmented train parquet not found: {path}. "
                "Run experiments/build_acronym_lemmas.py first."
            )
        df = pd.read_parquet(path)
        X = _extract_text(df, text_col="lemmas_acronym")
    else:
        raise ValueError(
            f"Unknown lemmatizer: '{lemmatizer}'. Choose 'pymorphy3', 'mystem', or 'pymorphy3_acronym'."
        )
    y = _extract_binary_y(df)
    _sanity_check_y(y, EXPECTED_TRAIN_POSITIVES, "train")
    return X, y


def load_test_binary(lemmatizer: str = "pymorphy3") -> Tuple[pd.Series, pd.DataFrame]:
    """
    Returns (X, y_binary).

    X: pd.Series of lemmatized texts.
    y_binary: pd.DataFrame with one column per canonical topic name,
              ordered per appendix7_targets.json::topics_order, values 0/1.

    lemmatizer:
        "pymorphy3" (default) — reads from test_lemmas.parquet, column 'lemmas'.
        "mystem"              — reads from data_mystem/test_mystem.parquet, column 'lemmas_mystem'.
    """
    if lemmatizer == "pymorphy3":
        path = _find_file(get_data_dir(), "test")
        df = _read_parquet_or_csv(path)
        X = _extract_text(df, text_col="lemmas")
    elif lemmatizer == "mystem":
        path = _MYSTEM_DATA_DIR / "test_mystem.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"MyStem test parquet not found: {path}. "
                "Run experiments/build_mystem_lemmas.py first."
            )
        df = pd.read_parquet(path)
        X = _extract_text(df, text_col="lemmas_mystem")
    elif lemmatizer == "pymorphy3_acronym":
        path = _ACRONYM_DATA_DIR / "test_acronym.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"Acronym-augmented test parquet not found: {path}. "
                "Run experiments/build_acronym_lemmas.py first."
            )
        df = pd.read_parquet(path)
        X = _extract_text(df, text_col="lemmas_acronym")
    else:
        raise ValueError(
            f"Unknown lemmatizer: '{lemmatizer}'. Choose 'pymorphy3', 'mystem', or 'pymorphy3_acronym'."
        )
    y = _extract_binary_y(df)
    _sanity_check_y(y, EXPECTED_TEST_POSITIVES, "test")
    return X, y


# ---------------------------------------------------------------------------
# load_train / load_test — backward-compatible wrappers.
# Now return (X: pd.Series, y: pd.DataFrame) instead of (X, y: pd.Series).
# All callers in validation.py / pipeline.py use y[topic] which works for DataFrame.
# ---------------------------------------------------------------------------

def load_train() -> Tuple[pd.Series, pd.DataFrame]:
    """Returns (X_lemmas, y_binary_df). y is a DataFrame of 11 binary columns."""
    return load_train_binary()


def load_test() -> Tuple[pd.Series, pd.DataFrame]:
    """Returns (X_lemmas, y_binary_df). y is a DataFrame of 11 binary columns."""
    return load_test_binary()


def get_topics() -> List[str]:
    """Canonical topic names in appendix7_targets.json order."""
    return list(CANONICAL_TOPICS_ORDER)


def describe() -> None:
    print_ = __import__("functools").partial(print, flush=True)
    data_dir = get_data_dir()
    print_(f"=== data_dir = {data_dir} ===")
    print_(f"Files: {sorted(p.name for p in data_dir.iterdir() if p.is_file())}")

    X_tr, y_tr = load_train()
    X_te, y_te = load_test()

    print_(f"\nTrain: {len(X_tr)} rows, {y_tr.shape[1]} topic columns")
    print_(f"Test:  {len(X_te)} rows, {y_te.shape[1]} topic columns")
    print_(f"\nCanonical topic columns: {y_tr.columns.tolist()}")
    print_(f"\nTrain positives per topic:\n{y_tr.sum().to_string()}")
    print_(f"\nTest positives per topic:\n{y_te.sum().to_string()}")

    lens_tr = X_tr.str.len()
    print_(f"\nText (lemmas) length on train: mean={lens_tr.mean():.1f}, "
           f"median={lens_tr.median():.0f}, p95={lens_tr.quantile(0.95):.0f}")
    print_(f"Empty texts on train: {(lens_tr == 0).sum()}")
    print_(f"Empty texts on test:  {(X_te.str.len() == 0).sum()}")

    # Check for train/test pk overlap
    path_tr = _find_file(data_dir, "train")
    path_te = _find_file(data_dir, "test")
    df_tr = _read_parquet_or_csv(path_tr)
    df_te = _read_parquet_or_csv(path_te)
    if "pk" in df_tr.columns and "pk" in df_te.columns:
        overlap = set(df_tr["pk"]) & set(df_te["pk"])
        print_(f"\nTrain/test pk overlap: {len(overlap)} (should be 0)")


if __name__ == "__main__":
    describe()
