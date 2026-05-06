import argparse
import asyncio
import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path

try:
    from telethon import TelegramClient
except ImportError as exc:
    raise SystemExit(
        "Telethon is not installed. Run: pip install telethon"
    ) from exc


DEFAULT_CHANNELS_FILE = Path("configs/channels.txt")
DEFAULT_OUTPUT_DIR = Path("data/raw/telegram_json")
SESSION_FILE = Path("configs/telegram_downloader")
LOG_FILE = Path("logs/download_log.csv")
DATE_FROM = datetime(2023, 10, 1, tzinfo=timezone.utc)
DATE_TO = datetime(2024, 5, 1, tzinfo=timezone.utc)

LOG_COLUMNS = [
    "started_at",
    "finished_at",
    "status",
    "channel",
    "channel_id",
    "channel_name",
    "channel_type",
    "messages_count",
    "output_file",
    "error",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def append_log(row: dict):
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow({column: row.get(column, "") for column in LOG_COLUMNS})


def load_channels(channels_file: Path) -> list[str]:
    if not channels_file.exists():
        raise FileNotFoundError(f"Channels file not found: {channels_file}")

    channels = []
    for line in channels_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            channels.append(line)

    if not channels:
        raise ValueError(f"No channels found in {channels_file}")

    return channels


def get_required_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Environment variable {name} is not set")
    return value


def safe_filename(channel: str) -> str:
    value = channel.strip().replace("https://t.me/", "").replace("http://t.me/", "")
    value = value.strip("@/").replace("/", "_")
    return value or "channel"


def format_datetime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return value.isoformat()


def format_unixtime(value):
    if value is None:
        return None

    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)

    return str(int(value.timestamp()))


def format_reactions(message):
    if not message.reactions or not message.reactions.results:
        return []

    reactions = []
    for item in message.reactions.results:
        reaction = item.reaction
        emoticon = getattr(reaction, "emoticon", None)
        document_id = getattr(reaction, "document_id", None)

        reactions.append(
            {
                "reaction": emoticon or str(document_id or reaction),
                "count": item.count,
            }
        )

    return reactions


def format_message(message):
    sender_id = str(message.sender_id) if message.sender_id else None

    return {
        "id": message.id,
        "type": "message",
        "date": format_datetime(message.date),
        "date_unixtime": format_unixtime(message.date),
        "edited": format_datetime(message.edit_date),
        "edited_unixtime": format_unixtime(message.edit_date),
        "from": None,
        "from_id": sender_id,
        "text": message.message or "",
        "reactions": format_reactions(message),
    }


def normalize_message_date(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)

    return value.astimezone(timezone.utc)


async def download_channel(client, channel: str, output_dir: Path, limit: int | None):
    started_at = utc_now()
    entity = await client.get_entity(channel)
    channel_name = getattr(entity, "title", None) or getattr(entity, "username", channel)
    channel_id = getattr(entity, "id", None)
    channel_type = entity.__class__.__name__

    print(f"\nDownloading: {channel_name} ({channel})")

    messages = []
    async for message in client.iter_messages(entity, limit=limit):
        message_date = normalize_message_date(message.date)
        if message_date < DATE_FROM:
            break
        if message_date >= DATE_TO:
            continue
        if message.message:
            messages.append(format_message(message))

    messages.reverse()

    output = {
        "id": channel_id,
        "name": channel_name,
        "type": channel_type,
        "messages": messages,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{safe_filename(channel)}.json"
    output_file.write_text(
        json.dumps(output, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    append_log(
        {
            "started_at": started_at,
            "finished_at": utc_now(),
            "status": "success",
            "channel": channel,
            "channel_id": channel_id,
            "channel_name": channel_name,
            "channel_type": channel_type,
            "messages_count": len(messages),
            "output_file": str(output_file),
        }
    )

    print(f"Saved {len(messages)} messages to {output_file}")


async def main_async(args):
    api_id = int(get_required_env("TG_API_ID"))
    api_hash = get_required_env("TG_API_HASH")

    channels = load_channels(args.channels_file)

    async with TelegramClient(str(SESSION_FILE), api_id, api_hash) as client:
        for channel in channels:
            started_at = utc_now()
            try:
                await download_channel(client, channel, args.output_dir, args.limit)
            except Exception as exc:
                append_log(
                    {
                        "started_at": started_at,
                        "finished_at": utc_now(),
                        "status": "failed",
                        "channel": channel,
                        "error": str(exc),
                    }
                )
                print(f"Failed to download {channel}: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Download Telegram channel history to JSON files."
    )
    parser.add_argument(
        "--channels-file",
        type=Path,
        default=DEFAULT_CHANNELS_FILE,
        help=f"Path to channels list. Default: {DEFAULT_CHANNELS_FILE}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for JSON files. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max messages per channel. By default downloads all available messages.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
