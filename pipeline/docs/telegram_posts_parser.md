# Парсер Telegram-постов: устройство и схема данных

## Где лежит код

Основной код первичного сбора и парсинга находится в соседнем проекте:

```text
../SignalSearcher/telegram_signal_searcher/
```

Ключевые файлы:

| Файл | Назначение |
|---|---|
| `src/download_tg_channels.py` | скачивание истории каналов через Telethon в JSON |
| `src/extract_posts.py` | извлечение сообщений из JSON в единую parquet-таблицу |
| `add_channel_usernames.py` | исторический helper для добавления `channel_username`; в текущем корне файла нет |
| `src/clean_data.py` | очистка постов для анализа |
| `src/analyze_post_times.py` | отчеты по времени публикаций |

Важно: `add_channel_usernames.py` числится удалённым в текущей рабочей папке. Для TGStat сейчас можно использовать `src/split_shards.py`: если в parquet нет `channel_username`, он восстанавливает username из `source_file`.

В BERT-архиве уже лежат downstream-таблицы, собранные из таких Telegram-постов:

| Файл | Что содержит |
|---|---|
| `corpus_full.parquet` | BERT-корпус с постами и LLM/BERT-разметкой |
| `data/RELEVANCE.parquet` | посты с префильтром, BERT-topic предсказаниями и служебными полями |
| `data/prefilter/work/input_prepared.parquet` | подготовленный вход для префильтра |

## Общая архитектура

Пайплайн состоит из четырех практических шагов:

```text
channels.txt
  -> download_tg_channels.py
  -> raw JSON per channel
  -> extract_posts.py
  -> extracted_messages.parquet
  -> split_shards.py / clean_data.py
  -> parquet для анализа, префильтра и BERT
```

Разделение сделано правильно: скачивание Telegram API отделено от парсинга JSON. Если один канал скачался плохо или JSON поврежден, можно перепарсить только файлы, не трогая Telegram API.

## Шаг 1. Скачивание каналов через Telethon

Файл: `src/download_tg_channels.py`.

Вход:

- `configs/channels.txt` - список каналов, по одному на строку;
- переменные окружения `TG_API_ID` и `TG_API_HASH`;
- опционально `TG_PROXY` или `--proxy`;
- локальная session-файл база `configs/telegram_downloader`.

Скрипт создает `TelegramClient` и для каждого канала вызывает:

```python
client.get_entity(channel)
client.iter_messages(entity, limit=limit)
```

Для каждого сообщения формируется компактный JSON-объект:

| Поле | Смысл |
|---|---|
| `id` | ID сообщения внутри канала |
| `type` | тип, обычно `message` |
| `date`, `date_unixtime` | дата публикации |
| `edited`, `edited_unixtime` | дата редактирования |
| `from`, `from_id` | отправитель |
| `text` | текст сообщения |
| `reactions` | реакции с counts |
| `views` | просмотры Telegram |
| `forwards` | пересылки |
| `reply_to_message_id` | ID сообщения, на которое отвечали |
| `reply_to_peer_id` | peer ответа |
| `replies` | число ответов |

Результат одного канала сохраняется в файл:

```text
data/raw/telegram_json/new/<channel>.json
```

Структура JSON:

```json
{
  "id": 123456789,
  "name": "Channel Name",
  "type": "Channel",
  "messages": [
    {
      "id": 1,
      "type": "message",
      "date": "2024-06-01T12:00:00+00:00",
      "text": "..."
    }
  ]
}
```

Лог скачивания пишется в:

```text
logs/download_log.csv
```

В логе есть статус, канал, ID канала, имя, тип, количество сообщений, выходной файл и ошибка.

## Шаг 2. Извлечение постов из JSON

Файл: `src/extract_posts.py`.

Скрипт читает все JSON-файлы из:

```text
data/raw/telegram_json/new/*.json
```

и собирает общую таблицу:

```text
data/clean/extracted_messages.parquet
```

### Почему парсер не просто делает `json.load`

Telegram JSON-экспорты и большие скачанные файлы могут быть повреждены или обрезаны. Поэтому `extract_posts.py` поддерживает два режима:

1. Обычный режим: `json.loads(raw_text)`.
2. Fallback: если весь JSON не парсится, скрипт вручную ищет массив `"messages"` и читает объекты из него через `JSONDecoder.raw_decode`.

Это позволяет получить `partial_success`: файл формально битый, но часть сообщений из массива удалось восстановить.

## Нормализация текста

Telegram export иногда хранит `text` не строкой, а списком: куски текста и объекты форматирования. Поэтому есть функция `normalize_text`.

Она делает так:

- `None` -> пустая строка;
- `str` -> как есть;
- `list` -> склеивает строковые элементы и `item["text"]` из dict-элементов;
- все остальное -> `str(value)`.

Это важно для BERT: входная колонка `text` должна быть обычной строкой.

## Нормализация реакций

Функция `count_reactions` считает суммарные реакции.

Поддерживаются форматы:

- `{"results": [...]}`;
- просто список реакций.

Каждый элемент должен иметь `count`. Сумма записывается в:

```text
reactions_total
```

Дополнительно создается:

```text
has_reactions = int(reactions_total > 0)
```

Сырые реакции сериализуются обратно в JSON-строку через `json_or_empty`.

## Поля, которые собирает `extract_posts.py`

На уровне канала:

| Поле | Смысл |
|---|---|
| `channel_id` | ID канала |
| `channel_name` | имя канала |
| `channel_type` | тип сущности Telegram |
| `source_file` | имя исходного JSON |

На уровне сообщения:

| Поле | Смысл |
|---|---|
| `message_id` | ID поста внутри канала |
| `message_type` | тип сообщения |
| `post_date` | дата публикации |
| `post_date_unixtime` | unix-время публикации |
| `edited` | дата редактирования |
| `edited_unixtime` | unix-время редактирования |
| `from_name`, `from_id` | отправитель |
| `actor`, `actor_id`, `action` | service/action-поля, если есть |
| `title`, `author` | дополнительные поля Telegram export |
| `forwarded_from`, `forwarded_from_id` | источник пересылки |
| `reply_to_message_id`, `reply_to_peer_id` | reply-связи |
| `text` | нормализованный текст |
| `text_entities` | форматирование/ссылки в JSON-строке |
| `reactions_total`, `has_reactions`, `reactions` | реакции |
| `views`, `forwards`, `replies` | метрики Telegram |
| `has_media` | есть ли медиа |
| `media_type`, `mime_type`, `file_name`, `file_size` | файл/медиа |
| `photo`, `photo_file_size`, `width`, `height` | фото/размеры |
| `duration_seconds` | длительность видео/аудио |
| `poll` | опрос в JSON-строке |
| `inline_bot_buttons` | inline-кнопки в JSON-строке |

`has_media` вычисляется по наличию хотя бы одного из признаков:

```text
photo, file, media_type, mime_type, poll
```

## Логирование парсинга

`extract_posts.py` пишет `logs/extract_log.csv`.

Колонки лога:

| Поле | Смысл |
|---|---|
| `started_at`, `finished_at` | время обработки файла |
| `status` | `success`, `partial_success`, `failed` |
| `source_file` | имя JSON |
| `file_size_bytes` | размер файла |
| `channel_id`, `channel_name`, `channel_type` | канал |
| `messages_total` | сообщений в JSON |
| `rows_extracted` | сколько строк извлечено |
| `error` | ошибка или предупреждение |

Если часть сообщений не является объектами, они пропускаются, а в лог добавляется предупреждение.

## Шаг 3. Подготовка username канала

Для TGStat и некоторых downstream-задач нужен `channel_username`, но Telegram API не всегда хранит его в сообщении.

Исторический файл `add_channel_usernames.py` решал это так:

1. Берет имена JSON-файлов, например `a_articles.json`.
2. Строит mapping `channel_id -> @username`.
3. Мержит этот mapping в `extracted_messages.parquet`.
4. Перед перезаписью делает backup:

```text
data/clean/extracted_messages.before_channel_username.parquet
```

Это особенно важно для TGStat, где URL строится через username.

В текущей версии корня проекта этого helper-файла нет. Для подготовки TGStat-входов используйте `src/split_shards.py`: он читает `channel_username`, а если такой колонки нет, восстанавливает `@username` из `source_file` и пишет CSV с колонками `channel_username`, `post_id`.

## Шаг 4. Очистка для анализа

Файл: `src/clean_data.py`.

Он делает компактную аналитическую таблицу:

```text
data/processed/posts_cleaned.parquet
```

Основные операции:

1. Оставляет только `message_type == "message"`.
2. Удаляет строки с пустым `text`.
3. Приводит `post_date` и `edited` к datetime.
4. Создает `is_edited`.
5. Создает `post_uid = channel_id + "_" + message_id`.
6. Создает `text_length`.
7. Сортирует по `post_date`.

## Как это связано с BERT-корпусом

BERT-корпус использует уже подготовленные посты, но сохраняет основные парсерные поля:

```text
channel_id
channel_name
channel_type
message_id
message_type
post_date
post_date_unixtime
edited
edited_unixtime
from_id
text
text_entities
reactions_total
has_reactions
reactions
has_media
poll
inline_bot_buttons
source_file
text_hash
```

Ключ строки:

```text
(channel_id, message_id)
```

Для BERT это важно по двум причинам:

1. По этому ключу LLM-разметка мержилась обратно к полным подневным вырезкам.
2. По этому ключу проверялось, что train/test не пересекаются.

`text_hash` используется для дедупликации одинаковых текстов и ускорения префильтра. В `classifiers/scripts/run_prefilter_pipeline.py` он строится как `blake2b` digest size 16 от текста, если готового `text_hash` нет.

## Схема шире, чем минимальный parser script

В `data/RELEVANCE.parquet` есть дополнительные поля:

```text
grouped_id
via_bot_id
pinned
silent
post
noforwards
ttl_period
restriction_reason
forwarded_from_post
forwarded_date
media_class
webpage
webpage_url
```

Это значит, что финальный parquet в BERT-ветке был собран из более широкой версии Telegram-выгрузки/нормализации, чем минимальный `extract_posts.py` в текущем виде. Но логика та же:

- канал и сообщение нормализуются в табличную строку;
- текст приводится к строке;
- вложенные Telegram-структуры сериализуются;
- пост получает стабильный ключ `(channel_id, message_id)`;
- downstream-скрипты фильтруют, размечают и дедуплицируют эти строки.

## Сильные стороны парсера

- Умеет восстанавливать часть данных из поврежденного JSON.
- Не падает на странном формате `text`.
- Сохраняет исходные вложенные поля как JSON-строки, а не теряет их.
- Логирует каждый файл отдельно.
- Отделяет скачивание от извлечения.
- Дает стабильные ключи для дальнейшего merge.

## Ограничения

- `channel_name` не уникален. Нужны `channel_id` и `message_id`.
- `channel_username` выводится из имени JSON-файла, если Telegram не дал его явно.
- Медиа не скачиваются как файлы, сохраняются только метаданные.
- Поврежденный JSON восстанавливается best-effort: можно получить `partial_success`, но не гарантию полного восстановления.
- Для BERT главным входом остается `text`; картинки, видео и inline-кнопки не анализируются моделью напрямую.
