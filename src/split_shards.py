from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "data" / "clean" / "extracted_messages.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean" / "shards"
DEFAULT_N_SHARDS = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split extracted Telegram posts into CSV shards for TGStat collection."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input parquet file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for shard_XXX.csv files.",
    )
    parser.add_argument("--n-shards", type=int, default=DEFAULT_N_SHARDS, help="Number of CSV shards.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed used for shuffling.")
    parser.add_argument("--limit", type=int, help="Optional row limit for quick test runs.")
    parser.add_argument(
        "--include-service",
        action="store_true",
        help="Include service Telegram messages. By default only message_type == 'message' is used.",
    )
    return parser.parse_args()


def username_from_source_file(value: object) -> str:
    if pd.isna(value):
        return ""

    stem = Path(str(value).strip()).stem.strip()
    if not stem:
        return ""

    return stem if stem.startswith("@") else f"@{stem}"


def load_posts(input_path: Path, include_service: bool, limit: int | None = None) -> pd.DataFrame:
    columns = set(pq.read_schema(input_path).names)

    if "channel_username" in columns:
        channel_col = "channel_username"
        read_columns = [channel_col, "message_id"]
    elif "source_file" in columns:
        channel_col = "source_file"
        read_columns = [channel_col, "message_id"]
    else:
        raise ValueError("Expected either 'channel_username' or 'source_file' in parquet columns.")

    if "message_id" not in columns:
        raise ValueError("Expected 'message_id' in parquet columns.")

    if not include_service and "message_type" in columns:
        read_columns.append("message_type")

    df = pd.read_parquet(input_path, columns=read_columns, engine="pyarrow")

    if not include_service and "message_type" in df.columns:
        df = df[df["message_type"] == "message"]

    if channel_col == "source_file":
        channel_username = df[channel_col].map(username_from_source_file)
    else:
        channel_username = df[channel_col].fillna("").astype(str).str.strip()
        channel_username = channel_username.where(
            (channel_username == "") | channel_username.str.startswith("@"),
            "@" + channel_username,
        )

    posts = pd.DataFrame(
        {
            "channel_username": channel_username,
            "post_id": df["message_id"],
        }
    )

    posts["post_id"] = posts["post_id"].astype("string").str.strip()
    posts = posts[(posts["channel_username"] != "") & (posts["post_id"] != "")]
    posts = posts.drop_duplicates().reset_index(drop=True)
    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        posts = posts.head(limit)
    return posts


def write_shards(posts: pd.DataFrame, output_dir: Path, n_shards: int, seed: int) -> None:
    if n_shards <= 0:
        raise ValueError("--n-shards must be greater than zero.")

    output_dir.mkdir(parents=True, exist_ok=True)

    posts = posts.sample(frac=1, random_state=seed).reset_index(drop=True)
    shard_size = math.ceil(len(posts) / n_shards) if len(posts) else 0

    for i in range(n_shards):
        start = i * shard_size
        stop = (i + 1) * shard_size
        part = posts.iloc[start:stop] if shard_size else posts.iloc[0:0]
        out = output_dir / f"shard_{i:03d}.csv"
        part.to_csv(out, index=False)
        print(f"{out} {len(part)}")


def main() -> None:
    args = parse_args()
    posts = load_posts(args.input, args.include_service, args.limit)
    print(f"Loaded posts: {len(posts)}")
    write_shards(posts, args.output_dir, args.n_shards, args.seed)


if __name__ == "__main__":
    main()
