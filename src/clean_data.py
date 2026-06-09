from pathlib import Path

import pandas as pd


INPUT_FILE = Path("data/clean/extracted_messages.parquet")
OUTPUT_DIR = Path("data/processed")
OUTPUT_FILE = OUTPUT_DIR / "posts_cleaned.parquet"


def main():
    if not INPUT_FILE.exists():
        print(f"❌ Файл не найден: {INPUT_FILE}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"📖 Читаю файл: {INPUT_FILE}")
    df = pd.read_parquet(INPUT_FILE)

    print(f"📊 Всего строк до очистки: {len(df)}")

    # 1. Оставляем только обычные сообщения
    df = df[df["message_type"] == "message"].copy()
    print(f"✅ После фильтра message_type == 'message': {len(df)}")

    # 2. Заполняем пропуски в тексте и убираем пробелы по краям
    df["text"] = df["text"].fillna("").astype(str).str.strip()

    # 3. Убираем сообщения с пустым текстом
    df = df[df["text"] != ""].copy()
    print(f"✅ После удаления пустого текста: {len(df)}")

    # 4. Приводим даты к datetime
    df["post_date"] = pd.to_datetime(df["post_date"], errors="coerce")
    df["edited"] = pd.to_datetime(df["edited"], errors="coerce")

    # 5. Создаём признак редактирования
    df["is_edited"] = df["edited"].notna().astype(int)

    # 6. Создаём уникальный идентификатор поста
    df["post_uid"] = (
        df["channel_id"].astype(str).fillna("")
        + "_"
        + df["message_id"].astype(str).fillna("")
    )

    # 7. Добавляем длину текста
    df["text_length"] = df["text"].str.len()

    # 8. Упорядочим колонки
    columns_order = [
        "post_uid",
        "channel_id",
        "channel_name",
        "channel_type",
        "message_id",
        "message_type",
        "post_date",
        "post_date_unixtime",
        "edited",
        "edited_unixtime",
        "is_edited",
        "from_name",
        "from_id",
        "reply_to_message_id",
        "reply_to_peer_id",
        "text",
        "text_length",
        "views",
        "forwards",
        "replies",
        "reactions_total",
        "has_reactions",
        "source_file",
    ]

    existing_columns = [col for col in columns_order if col in df.columns]
    df = df[existing_columns]

    # 9. Сортируем по дате
    df = df.sort_values(by="post_date", ascending=True)

    print(f"💾 Сохраняю очищенные данные: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)

    print("✅ Готово.")
    print(f"📦 Итоговое количество строк: {len(df)}")
    print(f"📁 Файл сохранён: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
