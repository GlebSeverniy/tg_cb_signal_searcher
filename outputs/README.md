# Выходные отчёты

Папка содержит CSV/табличные результаты, созданные локальными скриптами из `src/`.

## Основные файлы

| Путь | Назначение |
|---|---|
| `post_time_stats/` | отчёты `src/analyze_post_times.py` |
| `extracted_channels.csv` | список каналов, извлечённый из parquet |
| `extracted_channels_with_subscribers*.csv` | каналы с числом подписчиков |
| `tgstat_subscribers_test*.csv` | тестовые результаты TGStat subscribers collector |

## Входы

Обычно входы лежат в `data/clean/` или `data/posts_to_check/`.

## Выходы

Скрипты пишут сюда небольшие CSV-отчёты. Большие prediction-файлы BERT лежат в `pipeline/pred/`.

## Как использовать

```powershell
.\.venv\Scripts\python.exe src\analyze_post_times.py `
  --input data\clean\extracted_messages.parquet `
  --output-dir outputs\post_time_stats
```

## Примечания

Файлы в этой папке являются generated artifacts. Их можно пересоздать, если сохранены исходные данные и параметры запуска.

