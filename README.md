# telegram_signal_searcher

Проект для сбора, очистки и анализа Telegram-постов. Сейчас основной пайплайн работает с экспортами Telegram в JSON, собирает их в Parquet-таблицы, добавляет `@username` каналов, анализирует время публикаций и отдельно умеет собирать динамику просмотров постов со страниц TGStat.

## Быстрый Старт

Работать удобнее из корня проекта:

```powershell
cd "C:\Users\elist\ВУзик\Курсасч\Классификатор ЦБ\SignalSearcher\telegram_signal_searcher"
```

Использовать локальное окружение:

```powershell
.\.venv\Scripts\python.exe <script.py>
```

Если окружение нужно создать заново:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install h2 playwright
.\.venv\Scripts\python.exe -m playwright install chromium
```

Для скачивания Telegram через Telethon нужны переменные окружения:

```powershell
$env:TG_API_ID="123456"
$env:TG_API_HASH="your_api_hash"
```

## Структура Проекта

```text
configs/
  channels.txt                     # список Telegram-каналов для скачивания

data/
  raw/telegram_json/               # сырые JSON-экспорты каналов
  clean/                           # основные очищенные/обогащенные таблицы
  processed/                       # подготовленные данные для анализа/ML
  posts_to_check/posts.csv         # список постов для проверки в TGStat
  cache/                           # отладочные HTML-кэши TGStat

logs/
  download_log.csv                 # лог скачивания каналов
  extract_log.csv                  # лог извлечения постов из JSON

outputs/
  post_time_stats/                 # CSV-отчеты по времени публикаций

models/                            # место под ML-модели
notebooks/                         # исследовательские ноутбуки
src/                               # основные Python-скрипты
```

## Основные Данные

- `data/raw/telegram_json/*.json` - сырые Telegram JSON по каналам. Имя файла обычно соответствует username канала, например `a_articles.json`.
- `data/clean/extracted_messages.parquet` - главная таблица извлеченных сообщений.
- `data/clean/extracted_messages.before_channel_username.parquet` - backup перед добавлением `channel_username`.
- `data/processed/posts_cleaned.parquet` - очищенная таблица постов после `clean_data.py`.
- `data/posts_to_check/posts.csv` - CSV с колонками `channel_username`, `post_id` для сбора статистики TGStat.
- `data/clean/tgstat_views.csv` - long-format CSV с динамикой просмотров TGStat.

## Рекомендуемый Пайплайн

1. Скачать каналы из Telegram в JSON:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py
```

2. Извлечь посты из JSON в Parquet:

```powershell
.\.venv\Scripts\python.exe src\extract_posts.py
```

3. Добавить `channel_username` из имен JSON-файлов:

```powershell
.\.venv\Scripts\python.exe add_channel_usernames.py
```

4. Очистить таблицу постов для анализа:

```powershell
.\.venv\Scripts\python.exe src\clean_data.py
```

5. Построить отчеты по времени публикаций:

```powershell
.\.venv\Scripts\python.exe src\analyze_post_times.py
```

6. Собрать динамику просмотров TGStat для выбранных постов:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\clean\tgstat_views.csv `
  --html-cache-dir data\cache\tgstat_html `
  --use-playwright `
  --concurrency 8 `
  --min-concurrency 3 `
  --max-retries 2 `
  --retry-base-delay 5 `
  --throttle-cooldown 60 `
  --recovery-window 20 `
  --resume
```

## Скрипты

### `src/download_tg_channels.py`

Скачивает историю каналов через Telethon и сохраняет каждый канал в отдельный JSON.

Вход:
- `configs/channels.txt` - список каналов, по одному на строку.
- переменные окружения `TG_API_ID` и `TG_API_HASH`.

Выход:
- `data/raw/telegram_json/<channel>.json`
- `logs/download_log.csv`

Особенность: в коде задан период скачивания с `2023-10-01` до `2024-05-01` UTC.

Запуск по умолчанию:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py
```

Запуск с ограничением количества сообщений:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py `
  --channels-file configs\channels.txt `
  --output-dir data\raw\telegram_json `
  --limit 1000
```

### `src/extract_posts.py`

Читает все JSON-файлы из `data/raw/telegram_json`, извлекает сообщения и собирает единую таблицу.

Вход:
- `data/raw/telegram_json/*.json`

Выход:
- `data/clean/extracted_messages.parquet`
- `logs/extract_log.csv`

Скрипт умеет частично восстанавливать данные из поврежденных JSON: если весь JSON не парсится, он пытается отдельно извлечь массив `messages`.

Запуск:

```powershell
.\.venv\Scripts\python.exe src\extract_posts.py
```

### `add_channel_usernames.py`

Добавляет в `extracted_messages.parquet` колонку `channel_username` в формате `@username`. Username берется из имени JSON-файла, а сопоставление делается по `channel_id`.

Вход:
- `data/raw/telegram_json/*.json`
- `data/clean/extracted_messages.parquet`

Выход:
- обновленный `data/clean/extracted_messages.parquet`
- backup `data/clean/extracted_messages.before_channel_username.parquet`

Проверка без записи:

```powershell
.\.venv\Scripts\python.exe add_channel_usernames.py --dry-run
```

Основной запуск:

```powershell
.\.venv\Scripts\python.exe add_channel_usernames.py
```

Запуск с явными путями:

```powershell
.\.venv\Scripts\python.exe add_channel_usernames.py `
  --json-dir data\raw\telegram_json `
  --input data\clean\extracted_messages.parquet `
  --output data\clean\extracted_messages.parquet
```

### `src/clean_data.py`

Очищает `extracted_messages.parquet` для дальнейшего анализа:
- оставляет только `message_type == "message"`;
- удаляет строки с пустым текстом;
- приводит даты к datetime;
- добавляет `is_edited`, `post_uid`, `text_length`;
- оставляет компактный набор колонок.

Вход:
- `data/clean/extracted_messages.parquet`

Выход:
- `data/processed/posts_cleaned.parquet`

Запуск:

```powershell
.\.venv\Scripts\python.exe src\clean_data.py
```

### `src/analyze_post_times.py`

Строит CSV-отчеты о том, в какие часы и дни чаще всего публикуются посты. Использует DuckDB поверх Parquet, поэтому не загружает всю таблицу в pandas.

Вход по умолчанию:
- `data/clean/extracted_messages.parquet`

Выход:
- `outputs/post_time_stats/posts_by_hour.csv`
- `outputs/post_time_stats/posts_by_weekday_hour.csv`
- `outputs/post_time_stats/channel_peak_hour.csv`
- `outputs/post_time_stats/post_time_summary.csv`

Запуск по умолчанию, московское время UTC+3:

```powershell
.\.venv\Scripts\python.exe src\analyze_post_times.py
```

Запуск с явными параметрами:

```powershell
.\.venv\Scripts\python.exe src\analyze_post_times.py `
  --input data\clean\extracted_messages.parquet `
  --output-dir outputs\post_time_stats `
  --utc-offset-hours 3
```

Если нужно включить сервисные сообщения:

```powershell
.\.venv\Scripts\python.exe src\analyze_post_times.py --include-service-posts
```

### `src/tgstat_views_collector.py`

Собирает динамику просмотров Telegram-постов со страниц TGStat:

```text
https://tgstat.ru/en/channel/@<channel_username>/<post_id>/stat
```

Из HTML извлекаются JS-переменные:
- `chartData10min`
- `chartDataHour`
- `chartDataDay`

Выходной CSV имеет long format:
- `channel_username`
- `post_id`
- `tgstat_url`
- `group_by`
- `x`
- `views_delta`
- `views_cumulative`
- `collected_at`
- `status`
- `error`

Вход по умолчанию:
- `data/clean/extracted_messages.parquet`

Для тестов удобнее использовать:
- `data/posts_to_check/posts.csv`

#### Порядок Запуска TGStat-Парсера

Рекомендуемый порядок такой:

1. Подготовить `data/posts_to_check/posts.csv` с колонками `channel_username`, `post_id`.
2. Первый раз запустить Playwright в видимом режиме с persistent profile.
3. Если TGStat/Cloudflare покажет проверку, пройти ее вручную в открывшемся окне браузера.
4. Остановить или дождаться окончания тестового запуска.
5. Дальше запускать сбор с тем же `--user-data-dir`: cookies/session сохранятся между запусками.
6. Всегда использовать `--resume`, чтобы не перекачивать посты, которые уже успешно собраны (`status=ok`).

#### Первый Тестовый Запуск

Запуск на 3 постах, чтобы открыть браузер, создать профиль и проверить, что TGStat отдает `chartData`:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\posts_to_check\tgstat_views_test.csv `
  --html-cache-dir data\cache\tgstat_html `
  --debug-html-dir data\debug\tgstat_html `
  --use-playwright `
  --user-data-dir data\playwright_profile `
  --headless false `
  --concurrency 3 `
  --min-concurrency 2 `
  --limit 3 `
  --resume
```

#### Рабочий Запуск На Небольшой Пачке

Эта команда обрабатывает первые 30 постов из `posts.csv`, использует persistent profile Playwright и сохраняет debug HTML для проблемных страниц:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\clean\tgstat_views.csv `
  --html-cache-dir data\cache\tgstat_html `
  --debug-html-dir data\debug\tgstat_html `
  --use-playwright `
  --user-data-dir data\playwright_profile `
  --headless false `
  --concurrency 4 `
  --min-concurrency 1 `
  --limit 30 `
  --timeout 90 `
  --rate-limit-delay 300 `
  --throttle-cooldown 300 `
  --resume
```

#### Что Означают Параметры В Команде

- `--input data\posts_to_check\posts.csv` - входной список постов. Нужны колонки `channel_username` и `post_id`.
- `--output data\clean\tgstat_views.csv` - итоговый CSV в long format. При `--resume` уже успешные посты из этого файла пропускаются.
- `--html-cache-dir data\cache\tgstat_html` - папка обычного HTML-кэша. Валидным кэшем считается только HTML, где есть `chartData*`.
- `--debug-html-dir data\debug\tgstat_html` - папка для debug HTML. Туда сохраняются страницы `blocked/captcha/rate-limit` и `no_chart_data` вместе с JSON metadata.
- `--use-playwright` - использовать Playwright как основной загрузчик страниц TGStat.
- `--user-data-dir data\playwright_profile` - persistent browser profile. В нем сохраняются cookies/session между запусками. Это нужно, чтобы один раз пройти Cloudflare-проверку вручную и потом продолжать с тем же профилем.
- `--headless false` - открыть видимое окно браузера. Для первого запуска и ручной проверки Cloudflare это важно.
- `--concurrency 4` - максимум 4 параллельные страницы Playwright.
- `--min-concurrency 1` - нижняя граница adaptive concurrency. При rate-limit скрипт может снижать параллельность до 1.
- `--limit 30` - обработать только первые 30 строк входного файла. Уберите этот флаг для полного запуска.
- `--timeout 90` - ждать загрузку страницы до 90 секунд.
- `--rate-limit-delay 300` - при обнаружении blocked/captcha/rate-limit включить глобальную паузу новых задач примерно на 300 секунд.
- `--throttle-cooldown 300` - не снижать concurrency чаще, чем раз в 300 секунд.
- `--resume` - продолжить с места остановки: пропускаются только посты, которые уже есть в output CSV со `status=ok`.

#### Полный Запуск Без Ограничения `--limit`

Когда тест на 30 постах прошел нормально, можно убрать `--limit`:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\clean\tgstat_views.csv `
  --html-cache-dir data\cache\tgstat_html `
  --debug-html-dir data\debug\tgstat_html `
  --use-playwright `
  --user-data-dir data\playwright_profile `
  --headless false `
  --concurrency 4 `
  --min-concurrency 1 `
  --timeout 90 `
  --rate-limit-delay 300 `
  --throttle-cooldown 300 `
  --resume
```

Скрипт не должен агрессивно обходить защиту сайта: при признаках rate-limit/block он делает retry, backoff и снижает параллельность.

### Заготовки Под ML

Эти файлы сейчас существуют, но пустые:

- `src/build_dataset.py`
- `src/train_model.py`
- `src/predict.py`

Они зарезервированы под будущий ML-пайплайн: сбор датасета, обучение модели и предсказания.

## Подготовка `posts.csv` Для TGStat

Файл `data/posts_to_check/posts.csv` должен иметь такой формат:

```csv
channel_username,post_id
@a_articles,14065
@a_articles,14066
```

Сейчас в проекте есть файл на 100 записей, созданный из `data/clean/extracted_messages.parquet`.

Если нужно вручную добавить строку:

```csv
@channel_username,12345
```

Если нужно сгенерировать новый список, удобнее брать колонки `channel_username` и `message_id` из `extracted_messages.parquet`, переименовав `message_id` в `post_id`.

## Логи И Результаты

- `logs/download_log.csv` - результат скачивания каналов.
- `logs/extract_log.csv` - результат извлечения сообщений из JSON.
- `outputs/post_time_stats/*.csv` - аналитика по времени публикаций.
- `data/clean/tgstat_views.csv` - динамика просмотров TGStat.

## Частые Проблемы

### TGStat возвращает 403 или страницу `Just a moment`

Используйте Playwright:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\clean\tgstat_views.csv `
  --use-playwright `
  --resume
```

### Playwright не находит Google Chrome

Это не критично: скрипт пытается перейти на bundled Chromium. Если Chromium не установлен:

```powershell
.\.venv\Scripts\python.exe -m playwright install chromium
```

### Нужно продолжить после падения

Используйте тот же output и `--resume`. Скрипт пропустит только посты со `status=ok`; временные ошибки можно будет попробовать собрать снова.

### README или старые скрипты показывают битую кириллицу

Часть старых файлов была создана/прочитана с проблемной кодировкой консоли Windows. Новый README сохранен в UTF-8. Для вывода русских символов в PowerShell можно использовать:

```powershell
$env:PYTHONIOENCODING="utf-8"
```

## Текущее Состояние

Рабочие скрипты:
- `src/download_tg_channels.py`
- `src/extract_posts.py`
- `add_channel_usernames.py`
- `src/clean_data.py`
- `src/analyze_post_times.py`
- `src/tgstat_views_collector.py`

Пустые заготовки:
- `src/build_dataset.py`
- `src/train_model.py`
- `src/predict.py`

Основной актуальный датасет:
- `data/clean/extracted_messages.parquet`
