# Скрипты `src`

Здесь лежат основные локальные Python-скрипты для сбора, очистки и подготовки Telegram/TGStat-данных.

## Основные файлы

| Файл | Назначение |
|---|---|
| `download_tg_channels.py` | скачивает посты каналов через Telethon |
| `extract_posts.py` | собирает JSON Telegram в `extracted_messages.parquet` |
| `clean_data.py` | фильтрует обычные посты, чистит текст и сохраняет `posts_cleaned.parquet` |
| `split_shards.py` | делает CSV-shard-ы для TGStat |
| `tgstat_views_collector.py` | собирает динамику просмотров постов с TGStat |
| `tgstat_subscribers_collector.py` | собирает подписчиков каналов с TGStat |
| `add_telegram_subscribers.py` | добавляет число подписчиков через Telegram API |
| `list_extracted_channels.py` | выводит список каналов из `extracted_messages.parquet` |
| `analyze_post_times.py` | строит CSV-отчёты по времени публикаций |
| `merge_annotated_data.py` | объединяет размеченные декабрь/апрель parquet и нормализует темы |
| `build_dataset.py`, `train_model.py`, `predict.py` | пустые заготовки, не рабочие entry point |

## Входы

См. `docs/RUNNING.md`. Часть скриптов имеет CLI через `argparse`, часть использует константы путей внутри файла.

## Выходы

Основные результаты пишутся в `data/clean/`, `data/processed/`, `outputs/` и `logs/`.

## Как запускать

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py --help
.\.venv\Scripts\python.exe src\split_shards.py --help
.\.venv\Scripts\python.exe src\tgstat_views_collector.py --help
```

Скрипты `extract_posts.py` и `clean_data.py` сейчас рассчитаны на запуск после ручной проверки путей внутри файла.

## Примечания

Важно: не путайте эти скрипты с BERT training/inference. Обучение моделей находится в notebooks и `pipeline/`.

