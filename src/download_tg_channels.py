import argparse
import asyncio
import csv
import importlib.util
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlparse

try:
    from telethon import TelegramClient
    from telethon.network.connection import (
        ConnectionHttp,
        ConnectionTcpAbridged,
        ConnectionTcpFull,
        ConnectionTcpIntermediate,
        ConnectionTcpObfuscated,
    )
except ImportError as exc:
    raise SystemExit(
        "Telethon is not installed. Run: pip install telethon"
    ) from exc


DEFAULT_CHANNELS_FILE = Path("configs/channels.txt")
DEFAULT_OUTPUT_DIR = Path("data/raw/telegram_json/new")
SESSION_FILE = Path("configs/telegram_downloader")
LOG_FILE = Path("logs/download_log.csv")
DATE_FROM = datetime(2024, 6, 1, tzinfo=timezone.utc)
DATE_TO = datetime(2024, 9, 1, tzinfo=timezone.utc)

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

CONNECTION_TYPES = {
    "tcp-full": ConnectionTcpFull,
    "tcp-abridged": ConnectionTcpAbridged,
    "tcp-intermediate": ConnectionTcpIntermediate,
    "tcp-obfuscated": ConnectionTcpObfuscated,
    "http": ConnectionHttp,
}


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


def parse_proxy_url(value: str | None):
    if not value:
        return None

    parsed = urlparse(value)
    scheme = parsed.scheme.lower()
    if scheme not in {"socks5", "socks4", "http"}:
        raise ValueError("Proxy must use socks5://, socks4:// or http:// scheme")

    if not parsed.hostname or not parsed.port:
        raise ValueError("Proxy URL must include host and port, e.g. socks5://127.0.0.1:1080")

    if not (
        importlib.util.find_spec("python_socks")
        or importlib.util.find_spec("socks")
    ):
        raise RuntimeError(
            "Proxy support requires PySocks or python-socks. "
            "Run: .\\.venv\\Scripts\\python.exe -m pip install PySocks"
        )

    username = unquote(parsed.username) if parsed.username else None
    password = unquote(parsed.password) if parsed.password else None

    return (scheme, parsed.hostname, parsed.port, True, username, password)


def build_client_kwargs(args) -> dict:
    proxy_url = args.proxy or os.getenv("TG_PROXY")
    proxy = parse_proxy_url(proxy_url)

    kwargs = {}

    if args.connection:
        kwargs["connection"] = CONNECTION_TYPES[args.connection]
        print(f"Using Telegram connection: {args.connection}")

    if args.timeout is not None:
        kwargs["timeout"] = args.timeout

    if args.connection_retries is not None:
        kwargs["connection_retries"] = args.connection_retries

    if args.retry_delay is not None:
        kwargs["retry_delay"] = args.retry_delay

    if proxy:
        kwargs["proxy"] = proxy
        print(f"Using Telegram proxy: {proxy_url}")

    if kwargs:
        print("Telegram client options:", sorted(kwargs))
    else:
        print("Telegram client options: legacy defaults")

    return kwargs


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


def format_peer_id(value):
    if value is None:
        return None

    text = str(value)
    return text or None


def get_reply_to_message_id(message):
    reply_to = getattr(message, "reply_to", None)
    return getattr(reply_to, "reply_to_msg_id", None)


def get_reply_to_peer_id(message):
    reply_to = getattr(message, "reply_to", None)
    return format_peer_id(getattr(reply_to, "reply_to_peer_id", None))


def get_replies_count(message):
    replies = getattr(message, "replies", None)
    return getattr(replies, "replies", None)


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
        # Telethon exposes channel post views via message.views. Telegram mostly
        # returns it for channel posts; for other messages it can be None/null.
        "views": getattr(message, "views", None),
        "forwards": getattr(message, "forwards", None),
        "reply_to_message_id": get_reply_to_message_id(message),
        "reply_to_peer_id": get_reply_to_peer_id(message),
        "replies": get_replies_count(message),
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


async def check_connection(client):
    await client.connect()
    try:
        session = client.session
        print("Connected to Telegram API.")
        print("Session DC:", getattr(session, "dc_id", None))
        print("Session server:", getattr(session, "server_address", None))
        print("Session port:", getattr(session, "port", None))
        print("Authorized:", await client.is_user_authorized())
    finally:
        await client.disconnect()


async def main_async(args):
    api_id = int(get_required_env("TG_API_ID"))
    api_hash = get_required_env("TG_API_HASH")

    channels = load_channels(args.channels_file)
    client_kwargs = build_client_kwargs(args)
    client = TelegramClient(str(args.session_file), api_id, api_hash, **client_kwargs)

    if args.check_connection:
        await check_connection(client)
        return

    async with client:
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
        description=(
            "Download Telegram channel history to JSON files. "
            "Quick check: run with --limit 20, open the JSON, and inspect "
            "message fields views/forwards/replies/reply_to_message_id."
        )
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
        "--session-file",
        type=Path,
        default=SESSION_FILE,
        help=f"Telethon session file without .session suffix. Default: {SESSION_FILE}",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Max messages per channel. By default downloads all available messages.",
    )
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Only connect to Telegram API, print session/DC info, and exit.",
    )
    parser.add_argument(
        "--connection",
        choices=sorted(CONNECTION_TYPES),
        default=None,
        help=(
            "Telethon transport. If direct MTProto times out, try "
            "tcp-obfuscated or http. Default: Telethon legacy default."
        ),
    )
    parser.add_argument(
        "--proxy",
        default=None,
        help=(
            "Optional proxy URL for Telegram API, e.g. socks5://127.0.0.1:1080 "
            "or http://127.0.0.1:8080. Can also be set via TG_PROXY."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="Connection timeout in seconds. Default: Telethon legacy default.",
    )
    parser.add_argument(
        "--connection-retries",
        type=int,
        default=None,
        help="Telegram connection retries. Default: Telethon legacy default.",
    )
    parser.add_argument(
        "--retry-delay",
        type=int,
        default=None,
        help="Delay between connection retries in seconds. Default: Telethon legacy default.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
