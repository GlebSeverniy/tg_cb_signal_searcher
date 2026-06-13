# Данные

Папка содержит сырые Telegram-выгрузки, очищенные таблицы, разметку, TGStat-кэши и временные файлы. Большая часть содержимого является данными или generated artifacts и может отсутствовать в git.

## Основные подпапки

| Путь | Назначение |
|---|---|
| `raw/telegram_json/` | сырые JSON по каналам |
| `raw/telegram_json/new/` | новые выгрузки из `src/download_tg_channels.py` |
| `clean/` | объединённые parquet/csv после парсинга и TGStat |
| `clean/shards/` | CSV-shard-ы `channel_username,post_id` для TGStat |
| `processed/` | очищенные посты для анализа |
| `annotated_data/` | размеченные parquet и служебные материалы корпуса |
| `posts_to_check/` | небольшие CSV для ручной/TGStat-проверки |
| `cache/`, `debug/`, `playwright_profile/` | кэш TGStat, debug HTML и профиль браузера |
| `predictions/` | локальные предсказания/эксперименты |

## Входы

- `raw/telegram_json/new/*.json` - вход для `src/extract_posts.py`.
- `clean/extracted_messages.parquet` - вход для `src/clean_data.py`, `src/split_shards.py`, `src/analyze_post_times.py`.
- `posts_to_check/posts.csv` - вход для TGStat collector; нужны колонки `channel_username`, `post_id`.

## Выходы

- `clean/extracted_messages.parquet` - результат извлечения постов.
- `processed/posts_cleaned.parquet` - очищенная таблица обычных постов.
- `clean/shards/shard_*.csv` - shard-ы для TGStat.
- `clean/tgstat_views.csv` - динамика просмотров TGStat.

## Как использовать

```powershell
.\.venv\Scripts\python.exe src\extract_posts.py
.\.venv\Scripts\python.exe src\clean_data.py
.\.venv\Scripts\python.exe src\split_shards.py --n-shards 12
```

## Примечания

Важно: не удаляйте parquet/json/cache-файлы без проверки. Многие notebook-и и результаты ссылаются на конкретные имена файлов.

