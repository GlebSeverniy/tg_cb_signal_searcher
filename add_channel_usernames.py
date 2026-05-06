r"""
add_channel_usernames.py

Добавляет в таблицу с извлеченными Telegram-постами отдельный столбец
channel_username в формате "@username".

Логика:
    1. Просматривает JSON-файлы из data/raw/telegram_json.
    2. Берет channel_id из верхнего поля "id" внутри JSON.
    3. Берет username из имени файла: example_channel.json -> @example_channel.
    4. По channel_id проставляет username всем постам в extracted_messages.parquet.

По умолчанию обновляет:
    data/clean/extracted_messages.parquet

Перед перезаписью по умолчанию создает backup:
    data/clean/extracted_messages.before_channel_username.parquet

Примеры:
    python add_channel_usernames.py

    python add_channel_usernames.py --dry-run

    python add_channel_usernames.py ^
      --json-dir "C:\Users\elist\ВУзик\Курсасч\Классификатор ЦБ\SignalSearcher\telegram_signal_searcher\data\raw\telegram_json" ^
      --input "C:\Users\elist\ВУзик\Курсасч\Классификатор ЦБ\SignalSearcher\telegram_signal_searcher\data\clean\extracted_messages.parquet" ^
      --output "C:\Users\elist\ВУзик\Курсасч\Классификатор ЦБ\SignalSearcher\telegram_signal_searcher\data\clean\extracted_messages.parquet"
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


DEFAULT_JSON_DIR = Path("data/raw/telegram_json")
DEFAULT_INPUT = Path("data/clean/extracted_messages.parquet")
DEFAULT_OUTPUT = Path("data/clean/extracted_messages.parquet")
DEFAULT_BACKUP = Path("data/clean/extracted_messages.before_channel_username.parquet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add @channel usernames to extracted Telegram posts using raw JSON filenames."
    )
    parser.add_argument(
        "--json-dir",
        type=Path,
        default=DEFAULT_JSON_DIR,
        help="Directory with raw Telegram JSON files.",
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Input extracted messages parquet file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output parquet file. By default overwrites --input after backup.",
    )
    parser.add_argument(
        "--channel-id-col",
        default="channel_id",
        help="Column with Telegram channel ID in the parquet table.",
    )
    parser.add_argument(
        "--username-col",
        default="channel_username",
        help="Column to create/update with @username.",
    )
    parser.add_argument(
        "--backup",
        type=Path,
        default=DEFAULT_BACKUP,
        help="Backup path used when output overwrites input.",
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not create backup before overwriting input.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only show mapping and coverage statistics, do not write parquet.",
    )
    return parser.parse_args()


def extract_top_level_value(raw_text: str, key: str) -> Any:
    """Read a top-level JSON value without decoding the whole messages array."""
    decoder = json.JSONDecoder()
    marker = f'"{key}"'
    key_pos = raw_text.find(marker)
    messages_pos = raw_text.find('"messages"')

    if key_pos == -1 or (messages_pos != -1 and key_pos > messages_pos):
        return None

    colon_pos = raw_text.find(":", key_pos + len(marker))
    if colon_pos == -1:
        return None

    value_pos = colon_pos + 1
    while value_pos < len(raw_text) and raw_text[value_pos] in " \r\n\t":
        value_pos += 1

    try:
        value, _ = decoder.raw_decode(raw_text, value_pos)
    except json.JSONDecodeError:
        return None

    return value


def normalize_channel_id(value: Any) -> str:
    if pd.isna(value):
        return ""

    if isinstance(value, float) and value.is_integer():
        return str(int(value))

    text = str(value).strip()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]

    if text.startswith("-100") and text[4:].isdigit():
        text = text[4:]

    return text


def username_from_json_path(path: Path) -> str:
    stem = path.stem.strip()
    if not stem:
        raise ValueError(f"Empty JSON filename stem: {path}")
    return stem if stem.startswith("@") else f"@{stem}"


def build_channel_id_to_username(json_dir: Path) -> dict[str, str]:
    if not json_dir.exists():
        raise FileNotFoundError(f"JSON directory does not exist: {json_dir}")

    mapping: dict[str, str] = {}
    duplicate_ids: dict[str, list[str]] = {}
    skipped: list[str] = []

    json_files = sorted(json_dir.glob("*.json"))
    logging.info("Found JSON files: %s", len(json_files))

    for json_path in json_files:
        try:
            raw_text = json_path.read_text(encoding="utf-8", errors="replace")
            channel_id = normalize_channel_id(extract_top_level_value(raw_text, "id"))
            username = username_from_json_path(json_path)
        except Exception as exc:
            skipped.append(f"{json_path.name}: {exc}")
            continue

        if not channel_id:
            skipped.append(f"{json_path.name}: top-level id not found")
            continue

        if channel_id in mapping and mapping[channel_id] != username:
            duplicate_ids.setdefault(channel_id, [mapping[channel_id]]).append(username)
            continue

        mapping[channel_id] = username

    for channel_id, usernames in duplicate_ids.items():
        logging.warning(
            "Duplicate channel_id=%s in JSON filenames: %s. Keeping first: %s",
            channel_id,
            ", ".join(usernames),
            usernames[0],
        )

    if skipped:
        logging.warning("Skipped JSON files: %s", len(skipped))
        for item in skipped[:20]:
            logging.warning("Skipped: %s", item)
        if len(skipped) > 20:
            logging.warning("Skipped list truncated: %s more", len(skipped) - 20)

    logging.info("Built channel_id -> username mapping: %s", len(mapping))
    return mapping


def add_channel_usernames(
    df: pd.DataFrame,
    mapping: dict[str, str],
    channel_id_col: str,
    username_col: str,
) -> pd.DataFrame:
    if channel_id_col not in df.columns:
        available = ", ".join(map(str, df.columns))
        raise ValueError(f"Column {channel_id_col!r} not found. Available columns: {available}")

    result = df.copy()
    normalized_ids = result[channel_id_col].map(normalize_channel_id)
    mapped_usernames = normalized_ids.map(mapping)

    if username_col in result.columns:
        old_filled = result[username_col].notna() & (result[username_col].astype(str).str.strip() != "")
        logging.info("Existing non-empty %s values: %s", username_col, int(old_filled.sum()))

    result[username_col] = mapped_usernames.fillna("")
    return result


def log_coverage(df: pd.DataFrame, channel_id_col: str, username_col: str) -> None:
    total_rows = len(df)
    filled_rows = int((df[username_col].astype(str).str.strip() != "").sum())
    total_channels = df[channel_id_col].nunique(dropna=True)
    filled_channels = df.loc[df[username_col].astype(str).str.strip() != "", channel_id_col].nunique(dropna=True)

    logging.info("Rows total: %s", total_rows)
    logging.info("Rows with %s: %s", username_col, filled_rows)
    logging.info("Rows without %s: %s", username_col, total_rows - filled_rows)
    logging.info("Channels total in table: %s", total_channels)
    logging.info("Channels with %s: %s", username_col, filled_channels)
    logging.info("Channels without %s: %s", username_col, total_channels - filled_channels)

    missing_examples = (
        df.loc[df[username_col].astype(str).str.strip() == "", channel_id_col]
        .drop_duplicates()
        .head(20)
        .tolist()
    )
    if missing_examples:
        logging.warning("Example channel_id values without username: %s", missing_examples)


def save_parquet_with_optional_backup(df: pd.DataFrame, input_path: Path, output_path: Path, backup_path: Path, no_backup: bool) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    same_file = input_path.resolve() == output_path.resolve()
    if same_file and not no_backup:
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, backup_path)
        logging.info("Backup saved: %s", backup_path)

    df.to_parquet(output_path, index=False)
    logging.info("Updated parquet saved: %s", output_path)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    args = parse_args()

    mapping = build_channel_id_to_username(args.json_dir)
    if not mapping:
        raise SystemExit("No channel_id -> username mapping was built. Nothing to write.")

    logging.info("Reading parquet: %s", args.input)
    df = pd.read_parquet(args.input)
    logging.info("Loaded table shape: %s rows, %s columns", len(df), len(df.columns))

    updated_df = add_channel_usernames(
        df=df,
        mapping=mapping,
        channel_id_col=args.channel_id_col,
        username_col=args.username_col,
    )
    log_coverage(updated_df, args.channel_id_col, args.username_col)

    preview_columns = [args.channel_id_col, args.username_col]
    if "channel_name" in updated_df.columns:
        preview_columns.append("channel_name")
    logging.info(
        "Preview:\n%s",
        updated_df[preview_columns].drop_duplicates().head(20).to_string(index=False),
    )

    if args.dry_run:
        logging.info("Dry run enabled: parquet was not written.")
        return

    save_parquet_with_optional_backup(
        df=updated_df,
        input_path=args.input,
        output_path=args.output,
        backup_path=args.backup,
        no_backup=args.no_backup,
    )


if __name__ == "__main__":
    main()
