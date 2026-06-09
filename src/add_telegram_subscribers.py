from __future__ import annotations

import argparse
import asyncio
import os
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    from telethon import TelegramClient, errors, functions
except ImportError as exc:
    raise SystemExit("Telethon is not installed. Run: pip install telethon") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = PROJECT_ROOT / "outputs" / "extracted_channels.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "outputs" / "extracted_channels_with_subscribers.csv"
DEFAULT_SESSION = PROJECT_ROOT / "configs" / "telegram_downloader"
COMPAT_SESSION_SUFFIX = "_telethon142"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Add Telegram subscribers count to a channel CSV via Telegram API."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Input CSV with channels.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output CSV with subscribers count.")
    parser.add_argument(
        "--channel-col",
        default="channel_username",
        help="Column with Telegram channel username/link. Default: channel_username.",
    )
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION, help="Telethon session file path.")
    parser.add_argument("--limit", type=int, help="Optional channel limit for test runs.")
    parser.add_argument(
        "--delay",
        type=float,
        default=0.3,
        help="Delay between API calls in seconds. Default: 0.3.",
    )
    return parser.parse_args()


def configure_stdout() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def normalize_channel(value: object) -> str:
    if pd.isna(value):
        return ""

    text = str(value).strip()
    if not text:
        return ""

    for prefix in ("https://t.me/", "http://t.me/", "t.me/"):
        if text.lower().startswith(prefix):
            text = text[len(prefix) :]
            break

    text = text.strip().strip("/")
    if not text:
        return ""

    return text if text.startswith("@") else f"@{text}"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def load_credentials() -> tuple[int, str]:
    api_id_text = get_required_env("TG_API_ID").strip()
    if not api_id_text.isdigit():
        raise ValueError("TG_API_ID must be an integer.")

    api_hash = get_required_env("TG_API_HASH").strip()
    if not api_hash:
        raise ValueError("TG_API_HASH must not be empty.")

    return int(api_id_text), api_hash


def session_file_from_base(session_base: Path) -> Path:
    if session_base.suffix == ".session":
        return session_base
    return session_base.with_suffix(".session")


def session_base_from_file(session_file: Path) -> Path:
    if session_file.suffix == ".session":
        return session_file.with_suffix("")
    return session_file


def table_columns(cursor: sqlite3.Cursor, table_name: str) -> list[str]:
    return [row[1] for row in cursor.execute(f"pragma table_info({table_name})").fetchall()]


def create_compatible_session(source_file: Path, compat_file: Path) -> None:
    compat_file.parent.mkdir(parents=True, exist_ok=True)
    if compat_file.exists():
        compat_file.unlink()

    source = sqlite3.connect(source_file)
    target = sqlite3.connect(compat_file)
    try:
        source_cur = source.cursor()
        target_cur = target.cursor()

        target_cur.execute("create table version (version integer primary key)")
        target_cur.execute(
            """
            create table sessions (
                dc_id integer primary key,
                server_address text,
                port integer,
                auth_key blob,
                takeout_id integer
            )
            """
        )
        target_cur.execute(
            """
            create table entities (
                id integer primary key,
                hash integer not null,
                username text,
                phone integer,
                name text,
                date integer
            )
            """
        )
        target_cur.execute(
            """
            create table sent_files (
                md5_digest blob,
                file_size integer,
                type integer,
                id integer,
                hash integer,
                primary key(md5_digest, file_size, type)
            )
            """
        )
        target_cur.execute(
            """
            create table update_state (
                id integer primary key,
                pts integer,
                qts integer,
                date integer,
                seq integer
            )
            """
        )

        target_cur.executemany(
            "insert into version values (?)",
            source_cur.execute("select version from version").fetchall(),
        )
        target_cur.executemany(
            "insert into sessions (dc_id, server_address, port, auth_key, takeout_id) values (?, ?, ?, ?, ?)",
            source_cur.execute("select dc_id, server_address, port, auth_key, takeout_id from sessions").fetchall(),
        )

        for table_name in ("entities", "sent_files", "update_state"):
            source_columns = table_columns(source_cur, table_name)
            target_columns = table_columns(target_cur, table_name)
            shared_columns = [column for column in target_columns if column in source_columns]
            if not shared_columns:
                continue

            columns_sql = ", ".join(shared_columns)
            placeholders = ", ".join("?" for _ in shared_columns)
            rows = source_cur.execute(f"select {columns_sql} from {table_name}").fetchall()
            target_cur.executemany(
                f"insert into {table_name} ({columns_sql}) values ({placeholders})",
                rows,
            )

        target.commit()
    finally:
        source.close()
        target.close()


def ensure_compatible_session(session_base: Path) -> Path:
    session_file = session_file_from_base(session_base)
    if not session_file.exists():
        return session_base

    with sqlite3.connect(session_file) as connection:
        columns = table_columns(connection.cursor(), "sessions")

    if columns == ["dc_id", "server_address", "port", "auth_key", "takeout_id"]:
        return session_base

    compat_file = session_file.with_name(f"{session_file.stem}{COMPAT_SESSION_SUFFIX}.session")
    create_compatible_session(session_file, compat_file)
    print(f"Using compatible Telethon session copy: {compat_file}")
    return session_base_from_file(compat_file)


def load_channels(input_path: Path, channel_col: str, limit: int | None) -> pd.DataFrame:
    if not input_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_path}")

    df = pd.read_csv(input_path)
    if channel_col not in df.columns:
        raise ValueError(f"Column not found in input CSV: {channel_col}")

    df[channel_col] = df[channel_col].map(normalize_channel)
    df = df[df[channel_col] != ""].drop_duplicates(subset=[channel_col]).reset_index(drop=True)

    if limit is not None:
        if limit <= 0:
            raise ValueError("--limit must be greater than zero.")
        df = df.head(limit).copy()

    return df


async def get_subscribers_count(client: TelegramClient, channel: str) -> tuple[int | None, str]:
    try:
        entity = await client.get_entity(channel)
        full = await client(functions.channels.GetFullChannelRequest(entity))
        return getattr(full.full_chat, "participants_count", None), ""
    except errors.FloodWaitError as exc:
        wait_seconds = int(exc.seconds)
        print(f"Flood wait for {wait_seconds}s on {channel}; sleeping...")
        await asyncio.sleep(wait_seconds + 1)
        return await get_subscribers_count(client, channel)
    except Exception as exc:
        return None, f"{exc.__class__.__name__}: {exc}"


async def add_subscribers(args: argparse.Namespace) -> None:
    api_id, api_hash = load_credentials()
    channels = load_channels(args.input, args.channel_col, args.limit)

    if channels.empty:
        raise ValueError(f"No channels found in {args.input}")

    args.session.parent.mkdir(parents=True, exist_ok=True)
    session = ensure_compatible_session(args.session)

    subscribers_counts: list[int | None] = []
    errors_list: list[str] = []
    checked_at = utc_now()

    async with TelegramClient(str(session), api_id, api_hash) as client:
        total = len(channels)
        for index, channel in enumerate(channels[args.channel_col], start=1):
            count, error = await get_subscribers_count(client, channel)
            subscribers_counts.append(count)
            errors_list.append(error)

            status = count if error == "" else error
            print(f"[{index}/{total}] {channel}: {status}")

            if args.delay > 0 and index < total:
                await asyncio.sleep(args.delay)

    channels["subscribers_count"] = subscribers_counts
    channels["subscribers_checked_at"] = checked_at
    channels["subscribers_error"] = errors_list

    args.output.parent.mkdir(parents=True, exist_ok=True)
    channels.to_csv(args.output, index=False, encoding="utf-8-sig")
    print(f"Saved: {args.output}")


def main() -> None:
    configure_stdout()
    args = parse_args()
    asyncio.run(add_subscribers(args))


if __name__ == "__main__":
    main()
