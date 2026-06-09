from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANNOTATED_DIR = PROJECT_ROOT / "data" / "annotated_data"
APRIL_INPUT = ANNOTATED_DIR / "annotated_26-04-24.parquet"
DECEMBER_INPUT = ANNOTATED_DIR / "annotated-sample-15-12-2023.parquet"
DEFAULT_OUTPUT = ANNOTATED_DIR / "annotated_merged_15-12-2023_26-04-2024.parquet"

TOPIC_COLUMNS = [
    "t1_relevant",
    "t2_relevant",
    "t3_relevant",
    "t4_relevant",
    "t5_relevant",
    "t6_relevant",
]

DECEMBER_TO_CANONICAL_TOPICS = {
    "t1_relevant": "t1_relevant",
    "t2_relevant": "t2_relevant",
    "t3_relevant": "t4_relevant",
    "t4_relevant": "t3_relevant",
    "t5_relevant": "t5_relevant",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge two annotated parquet files and normalize topic columns to the "
            "26-04-2024 topic scheme."
        )
    )
    parser.add_argument(
        "--april-input",
        type=Path,
        default=APRIL_INPUT,
        help="Parquet file with 26-04-2024 annotations.",
    )
    parser.add_argument(
        "--december-input",
        type=Path,
        default=DECEMBER_INPUT,
        help="Parquet file with 15-12-2023 annotations.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output merged parquet file.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, columns: list[str], path: Path) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise ValueError(f"{path} does not contain required columns: {missing}")


def normalize_topic_values(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column in TOPIC_COLUMNS:
        df[column] = df[column].fillna(False).astype(bool)
    return df


def normalize_december_topics(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    require_columns(df, list(DECEMBER_TO_CANONICAL_TOPICS), path)

    normalized = df.copy()
    original_topics = normalized[list(DECEMBER_TO_CANONICAL_TOPICS)].copy()

    for source_column, target_column in DECEMBER_TO_CANONICAL_TOPICS.items():
        normalized[target_column] = original_topics[source_column]

    normalized["t6_relevant"] = False
    normalized["annotation_source"] = "15-12-2023"
    normalized["topic_scheme"] = "26-04-2024"
    return normalize_topic_values(normalized)


def normalize_april_topics(df: pd.DataFrame, path: Path) -> pd.DataFrame:
    require_columns(df, TOPIC_COLUMNS, path)

    normalized = df.copy()
    normalized["annotation_source"] = "26-04-2024"
    normalized["topic_scheme"] = "26-04-2024"
    return normalize_topic_values(normalized)


def align_columns(left: pd.DataFrame, right: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = list(left.columns)
    columns.extend(column for column in right.columns if column not in left.columns)
    return left.reindex(columns=columns), right.reindex(columns=columns)


def merge_annotations(april_path: Path, december_path: Path) -> pd.DataFrame:
    april = pd.read_parquet(april_path)
    december = pd.read_parquet(december_path)

    april = normalize_april_topics(april, april_path)
    december = normalize_december_topics(december, december_path)
    december, april = align_columns(december, april)

    merged = pd.concat([december, april], ignore_index=True)
    if "post_date" in merged.columns:
        merged["post_date"] = pd.to_datetime(merged["post_date"], errors="coerce")
        merged = merged.sort_values(["post_date", "channel_id", "message_id"], na_position="last")
    return merged.reset_index(drop=True)


def main() -> None:
    args = parse_args()
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)

    merged = merge_annotations(args.april_input, args.december_input)
    merged.to_parquet(output, index=False)

    print(f"Saved: {output}")
    print(f"Rows: {len(merged)}")
    print("Topic counts:")
    for column in TOPIC_COLUMNS:
        print(f"  {column}: {int(merged[column].sum())}")


if __name__ == "__main__":
    main()
