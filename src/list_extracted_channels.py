from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "clean" / "extracted_messages.parquet"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="List Telegram channels found in extracted_messages.parquet."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input parquet file.")
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional CSV file for the channel list. If omitted, only prints to stdout.",
    )
    parser.add_argument(
        "--no-at",
        action="store_true",
        help="Do not add @ to channel usernames restored from source_file.",
    )
    return parser.parse_args()


def username_from_source_file(value: object, add_at: bool) -> str:
    if pd.isna(value):
        return ""

    stem = Path(str(value).strip()).stem.strip()
    if not stem:
        return ""
    if not add_at or stem.startswith("@"):
        return stem
    return f"@{stem}"


def load_channels(input_path: Path, add_at: bool) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input parquet file not found: {input_path}")

    columns = set(pq.read_schema(input_path).names)
    read_columns = [column for column in ("source_file", "channel_id", "channel_name") if column in columns]

    if "source_file" not in read_columns and "channel_name" not in read_columns:
        raise ValueError("Expected at least one of 'source_file' or 'channel_name' in parquet columns.")

    df = pd.read_parquet(input_path, columns=read_columns, engine="pyarrow")

    if "source_file" in df.columns:
        df["channel_username"] = df["source_file"].map(lambda value: username_from_source_file(value, add_at))
    else:
        df["channel_username"] = ""

    group_columns = [column for column in ("channel_username", "source_file", "channel_id", "channel_name") if column in df.columns]
    channels = (
        df.groupby(group_columns, dropna=False)
        .size()
        .reset_index(name="messages_count")
        .sort_values(["channel_username", "channel_name", "channel_id"], na_position="last")
        .reset_index(drop=True)
    )
    return channels


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    args = parse_args()
    channels = load_channels(args.input, add_at=not args.no_at)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        channels.to_csv(args.output, index=False, encoding="utf-8-sig")
        print(f"Saved channels: {args.output}")

    print(f"Channels found: {len(channels)}")
    print(channels.to_string(index=False))


if __name__ == "__main__":
    main()
