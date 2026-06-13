# Конфиги и локальные credentials

Здесь лежат входные списки каналов и локальные Telethon session-файлы.

## Основные файлы

| Файл | Назначение |
|---|---|
| `channels.txt` | список Telegram-каналов для `src/download_tg_channels.py` |
| `*.session`, `*.session-journal` | локальные Telethon-сессии, создаются автоматически |

## Входы

`channels.txt` должен содержать по одному каналу на строку. Можно использовать `@username`, `t.me/...` или другие формы, которые понимает Telethon.

## Выходы

Скрипты Telegram API могут создавать `.session`-файлы рядом с указанным `--session-file`.

## Как использовать

```powershell
$env:TG_API_ID="123456"
$env:TG_API_HASH="your_api_hash"
.\.venv\Scripts\python.exe src\download_tg_channels.py --channels-file configs\channels.txt
```

## Примечания

Важно: не сохраняйте API keys и session-файлы в репозиторий. `.gitignore` уже исключает `configs/*.session`.

