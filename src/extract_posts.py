import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


RAW_DIR = Path("data/raw/telegram_json/new")
OUTPUT_DIR = Path("data/clean")
OUTPUT_FILE = OUTPUT_DIR / "extracted_messages.parquet"
LOG_FILE = Path("logs/extract_log.csv")

LOG_COLUMNS = [
    "started_at",
    "finished_at",
    "status",
    "source_file",
    "file_size_bytes",
    "channel_id",
    "channel_name",
    "channel_type",
    "messages_total",
    "rows_extracted",
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


def normalize_text(text_value) -> str:
    if text_value is None:
        return ""

    if isinstance(text_value, str):
        return text_value

    if isinstance(text_value, list):
        parts = []
        for item in text_value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(normalize_text(item.get("text")))
        return "".join(parts)

    return str(text_value)


def to_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def count_reactions(reactions) -> int:
    if isinstance(reactions, dict):
        reactions = reactions.get("results", [])

    if not isinstance(reactions, list):
        return 0

    total = 0
    for reaction in reactions:
        if isinstance(reaction, dict):
            total += to_int(reaction.get("count"))

    return total


def json_or_empty(value) -> str:
    if value in (None, [], {}):
        return ""

    return json.dumps(value, ensure_ascii=False)


def extract_top_level_value(raw_text: str, key: str):
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


def extract_messages_array(raw_text: str):
    decoder = json.JSONDecoder()
    marker_pos = raw_text.find('"messages"')
    if marker_pos == -1:
        return [], "messages key not found"

    array_start = raw_text.find("[", marker_pos)
    if array_start == -1:
        return [], "messages array start not found"

    messages = []
    index = array_start + 1
    total_length = len(raw_text)

    while index < total_length:
        while index < total_length and raw_text[index] in " \r\n\t,":
            index += 1

        if index >= total_length:
            return messages, "file ended before messages array was closed"

        if raw_text[index] == "]":
            return messages, ""

        if raw_text[index] != "{":
            return messages, f"unexpected character in messages array at offset {index}"

        try:
            message, index = decoder.raw_decode(raw_text, index)
        except json.JSONDecodeError as exc:
            return messages, f"stopped at malformed message near offset {index}: {exc}"

        messages.append(message)

    return messages, "file ended before messages array was closed"


def load_telegram_json(file_path: Path):
    raw_text = file_path.read_text(encoding="utf-8")

    try:
        data = json.loads(raw_text)
        if isinstance(data, dict):
            messages = data.get("messages", [])
            if not isinstance(messages, list):
                messages = []
            return {
                "id": data.get("id"),
                "name": data.get("name"),
                "type": data.get("type"),
                "messages": messages,
                "status": "success",
                "error": "",
            }

        if isinstance(data, list):
            return {
                "id": None,
                "name": file_path.stem,
                "type": "list",
                "messages": data,
                "status": "success",
                "error": "",
            }

        raise ValueError(f"Unsupported JSON root type: {type(data).__name__}")

    except json.JSONDecodeError as exc:
        messages, salvage_error = extract_messages_array(raw_text)
        if not messages:
            raise ValueError(f"JSON is malformed and no messages were recovered: {exc}")

        return {
            "id": extract_top_level_value(raw_text, "id"),
            "name": extract_top_level_value(raw_text, "name") or file_path.stem,
            "type": extract_top_level_value(raw_text, "type"),
            "messages": messages,
            "status": "partial_success",
            "error": salvage_error or str(exc),
        }


def get_message_value(message: dict, key: str):
    if not isinstance(message, dict):
        return None

    return message.get(key)


def build_row(message: dict, channel_info: dict, file_path: Path):
    reactions = get_message_value(message, "reactions")
    reactions_total = count_reactions(reactions)
    text = normalize_text(get_message_value(message, "text"))

    return {
        "channel_id": channel_info["channel_id"],
        "channel_name": channel_info["channel_name"],
        "channel_type": channel_info["channel_type"],
        "message_id": get_message_value(message, "id"),
        "message_type": get_message_value(message, "type"),
        "post_date": get_message_value(message, "date"),
        "post_date_unixtime": get_message_value(message, "date_unixtime"),
        "edited": get_message_value(message, "edited"),
        "edited_unixtime": get_message_value(message, "edited_unixtime"),
        "from_name": get_message_value(message, "from"),
        "from_id": get_message_value(message, "from_id"),
        "actor": get_message_value(message, "actor"),
        "actor_id": get_message_value(message, "actor_id"),
        "action": get_message_value(message, "action"),
        "title": get_message_value(message, "title"),
        "author": get_message_value(message, "author"),
        "forwarded_from": get_message_value(message, "forwarded_from"),
        "forwarded_from_id": get_message_value(message, "forwarded_from_id"),
        "reply_to_message_id": get_message_value(message, "reply_to_message_id"),
        "reply_to_peer_id": get_message_value(message, "reply_to_peer_id"),
        "text": text,
        "text_entities": json_or_empty(get_message_value(message, "text_entities")),
        "reactions_total": reactions_total,
        "has_reactions": int(reactions_total > 0),
        "reactions": json_or_empty(reactions),
        "views": get_message_value(message, "views"),
        "forwards": get_message_value(message, "forwards"),
        "replies": get_message_value(message, "replies"),
        "has_media": int(
            any(
                get_message_value(message, key)
                for key in ("photo", "file", "media_type", "mime_type", "poll")
            )
        ),
        "media_type": get_message_value(message, "media_type"),
        "mime_type": get_message_value(message, "mime_type"),
        "file_name": get_message_value(message, "file_name"),
        "file_size": get_message_value(message, "file_size"),
        "photo": get_message_value(message, "photo"),
        "photo_file_size": get_message_value(message, "photo_file_size"),
        "width": get_message_value(message, "width"),
        "height": get_message_value(message, "height"),
        "duration_seconds": get_message_value(message, "duration_seconds"),
        "poll": json_or_empty(get_message_value(message, "poll")),
        "inline_bot_buttons": json_or_empty(get_message_value(message, "inline_bot_buttons")),
        "source_file": file_path.name,
    }


def extract_messages_from_file(file_path: Path):
    data = load_telegram_json(file_path)

    channel_info = {
        "channel_id": data["id"],
        "channel_name": data["name"],
        "channel_type": data["type"],
        "messages_total": len(data["messages"]),
        "status": data["status"],
        "error": data["error"],
    }

    rows = []
    skipped_items = 0

    for message in data["messages"]:
        if not isinstance(message, dict):
            skipped_items += 1
            continue
        rows.append(build_row(message, channel_info, file_path))

    if skipped_items:
        warning = f"skipped non-object messages: {skipped_items}"
        channel_info["error"] = "; ".join(
            part for part in (channel_info["error"], warning) if part
        )

    return rows, channel_info


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    json_files = sorted(RAW_DIR.glob("*.json"))

    if not json_files:
        print(f"No JSON files found in {RAW_DIR}.")
        return

    all_rows = []
    failed_files = []
    partial_files = []

    for file_path in json_files:
        started_at = utc_now()
        print(f"\nProcessing file: {file_path.name}")

        try:
            rows, channel_info = extract_messages_from_file(file_path)
            all_rows.extend(rows)

            append_log(
                {
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "status": channel_info["status"],
                    "source_file": file_path.name,
                    "file_size_bytes": file_path.stat().st_size,
                    "channel_id": channel_info["channel_id"],
                    "channel_name": channel_info["channel_name"],
                    "channel_type": channel_info["channel_type"],
                    "messages_total": channel_info["messages_total"],
                    "rows_extracted": len(rows),
                    "error": channel_info["error"],
                }
            )

            print(f"{channel_info['status']}: {file_path.name} (rows: {len(rows)})")
            if channel_info["status"] == "partial_success":
                partial_files.append(file_path.name)

        except Exception as exc:
            append_log(
                {
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "status": "failed",
                    "source_file": file_path.name,
                    "file_size_bytes": file_path.stat().st_size,
                    "error": str(exc),
                }
            )
            print(f"Error in file {file_path.name}: {exc}")
            failed_files.append(file_path.name)
            continue

    if not all_rows:
        print("No data to save.")
        return

    df = pd.DataFrame(all_rows)

    print(f"\nTotal extracted rows: {len(df)}")
    print("Saving to Parquet...")

    df.to_parquet(OUTPUT_FILE, index=False)

    print(f"Done: {OUTPUT_FILE}")
    print(f"Parse log: {LOG_FILE}")

    if partial_files:
        print("\nPartially recovered files:")
        for file_name in partial_files:
            print(f" - {file_name}")

    if failed_files:
        print("\nProblem files were skipped:")
        for file_name in failed_files:
            print(f" - {file_name}")


if __name__ == "__main__":
    main()
