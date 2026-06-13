# TGStat: идея, попытки реализации и сложности

## Зачем вообще был нужен TGStat

Основной Telegram-парсер дает текст поста, дату, канал, реакции, просмотры и другие поля из Telegram. Но для задачи поиска "сигналов" этого мало: хотелось понимать не только содержание поста, но и его фактическую видимость.

Идея TGStat-ветки:

1. Взять уже найденные Telegram-посты.
2. Для каждого поста собрать динамику просмотров с TGStat.
3. Получить не просто итоговые `views`, а кривую распространения: сколько просмотров набиралось по 10 минутам, часам и дням.
4. Дополнительно собрать число подписчиков канала, чтобы нормировать охват.
5. Использовать это как признаки для анализа влияния поста: сильный ли был сигнал, насколько быстро он распространялся, какой относительный охват получил.

Эта ветка лежит в соседнем проекте:

```text
../SignalSearcher/telegram_signal_searcher/
```

Ключевые файлы:

| Файл | Назначение |
|---|---|
| `src/tgstat_views_collector.py` | сбор динамики просмотров постов с TGStat |
| `src/tgstat_subscribers_collector.py` | сбор подписчиков каналов со страниц TGStat |
| `src/add_telegram_subscribers.py` | альтернативный сбор подписчиков через Telegram API |
| `data/posts_to_check/posts.csv` | входной список постов для TGStat |
| `data/clean/tgstat_views.csv` | собранный результат по просмотрам |

## Как пытались собирать просмотры постов

Скрипт `src/tgstat_views_collector.py` принимает таблицу с колонками:

```text
channel_username, post_id
```

Для каждой строки строится URL:

```text
https://tgstat.ru/en/channel/@<channel_username>/<post_id>/stat
```

Дальше скрипт загружает HTML страницы и ищет внутри JS-переменные:

```text
chartData10min
chartDataHour
chartDataDay
```

TGStat хранит точки графика как JSON внутри `JSON.parse(...)`. Скрипт извлекает эти массивы, декодирует JS-строку, парсит JSON и разворачивает результат в long-format CSV.

Итоговая схема `tgstat_views.csv`:

| Колонка | Смысл |
|---|---|
| `channel_username` | username канала |
| `post_id` | ID поста внутри канала |
| `tgstat_url` | URL страницы TGStat |
| `group_by` | гранулярность: `10min`, `hour`, `day` |
| `x` | точка времени или подпись оси X |
| `views_delta` | прирост просмотров в этой точке |
| `views_cumulative` | накопленные просмотры внутри выбранной гранулярности |
| `collected_at` | время сбора |
| `status` | статус обработки |
| `error` | ошибка, если была |

В текущем результате:

| Файл | Строк | Статусы | Каналы |
|---|---:|---|---|
| `data/clean/tgstat_views.csv` | 3 967 | все `ok` | `a_articles` |
| `data/posts_to_check/tgstat_views_test.csv` | 114 | все `ok` | `a_articles` |

По `data/clean/tgstat_views.csv` распределение гранулярностей такое:

| `group_by` | Строк |
|---|---:|
| `hour` | 2 400 |
| `day` | 967 |
| `10min` | 600 |

То есть технически сбор был доведен до рабочего результата на ограниченной пачке постов.

## Как пытались собирать подписчиков каналов

Было два подхода.

### TGStat-подход

Файл: `src/tgstat_subscribers_collector.py`.

Скрипт открывает страницу канала:

```text
https://tgstat.ru/en/channel/<username>
```

Затем пытается достать число подписчиков несколькими способами:

1. из JSON-подобных фрагментов HTML;
2. из обычного текста рядом со словами `subscribers` / `подписчиков`;
3. из DOM-кандидатов рядом с этими словами;
4. из meta-тегов и title.

Числа нормализуются из компактных форматов вроде `12.5K`, `1,2 млн`, `120 000`.

### Telegram API-подход

Файл: `src/add_telegram_subscribers.py`.

Этот вариант использует Telethon и метод:

```text
functions.channels.GetFullChannelRequest(entity)
```

Оттуда берется `participants_count`. Это надежнее, когда доступ есть через Telegram API, но требует авторизованной сессии и может упираться в `FloodWait`.

## Почему реализация была сложной

### 1. TGStat защищается от массового сбора

Скрипт явно обрабатывает:

- HTTP 403;
- HTTP 429;
- Cloudflare / `Just a moment`;
- captcha;
- `access denied`;
- `too many requests`;
- страницы без `chartData`.

Для этого добавлены:

- случайный `User-Agent`;
- задержки и retry;
- adaptive concurrency;
- глобальная пауза при rate-limit;
- снижение параллельности;
- debug HTML для проблемных страниц.

### 2. Простого `httpx` часто недостаточно

В скрипте есть два режима:

| Режим | Для чего |
|---|---|
| `httpx` | быстрый сбор, если TGStat отдает обычный HTML |
| `Playwright` | загрузка через браузер, если нужна JS/Cloudflare/cookies |

Практический рекомендуемый режим в README - Playwright с persistent profile:

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

Смысл: открыть видимый браузер, один раз пройти проверку, сохранить cookies/session в `user-data-dir`, потом продолжать с тем же профилем.

### 3. Нужен стабильный `channel_username`

Telegram-экспорт и Telethon-дамп не всегда дают username канала как отдельное поле. Поэтому в пайплайне появился шаг добавления `channel_username` из имени JSON-файла.

Если username нет или он изменился, TGStat URL не построится. Если пост удален, канал переименован или TGStat не индексирует канал, будет `404` или `no_chart_data`.

### 4. `message_id` должен совпасть с TGStat `post_id`

Обычно Telegram `message_id` канального поста соответствует ID в URL TGStat. Но это чувствительная связка:

- пост мог быть удален;
- канал мог поменять username;
- в исходном списке могли быть не канальные сообщения;
- TGStat мог не иметь страницы статистики для конкретного поста.

Поэтому входной `posts.csv` собирался отдельно и тестировался маленькими пачками.

### 5. Парсинг HTML хрупкий

TGStat не отдавал удобный стабильный API в этой реализации. Данные вытаскивались из HTML и JS:

```text
var chartDataHour = JSON.parse('...')
```

Если TGStat поменяет имя переменной, формат страницы, экранирование строки или язык страницы, парсер может сломаться. Поэтому в коде есть:

- fast regex для типичного случая;
- fallback-чтение JS string literal;
- декодирование escape-последовательностей;
- отдельные статусы `parse_error`, `no_chart_data`, `forbidden`.

### 6. Объем данных быстро растет

Один пост превращается в много строк:

- точки `10min`;
- точки `hour`;
- точки `day`.

Поэтому результат пишется инкрементально в CSV, а не держится целиком до конца. Для возобновления есть `--resume`: уже успешные посты со `status=ok` пропускаются.

## Почему TGStat не стал ядром BERT-корпуса

TGStat решает другую задачу. BERT-корпус строился вокруг семантической разметки:

```text
text -> t1..t5 -> direction
```

TGStat же дает признаки распространения:

```text
post -> views over time, subscribers
```

Эти признаки потенциально полезны для следующего слоя анализа, например:

- ранжировать релевантные посты по охвату;
- оценивать "силу" сигнала;
- нормировать просмотры на подписчиков;
- искать посты, которые быстро набрали внимание;
- строить индекс восприятия не только по тональности/направлению, но и по видимости.

Но для обучения topic/direction BERT они не обязательны. Более того, TGStat-сбор оказался инфраструктурно нестабильным: защиты сайта, rate-limit, необходимость браузера, ручные проверки и хрупкий HTML-парсинг. Поэтому ядро BERT-пайплайна осталось на Telegram-дампах и LLM/BERT-разметке, а TGStat - как отдельная ветка обогащения.

## Текущее состояние

Рабочие части:

- `tgstat_views_collector.py` умеет собирать post-view dynamics и уже дал `tgstat_views.csv`.
- `tgstat_subscribers_collector.py` умеет пробовать подписчиков через TGStat.
- `add_telegram_subscribers.py` умеет брать подписчиков через Telegram API.
- Поддержаны `resume`, debug HTML, Playwright, cache, backoff и adaptive concurrency.

Ограничения:

- результат собран только для ограниченной тестовой пачки;
- в BERT-пайплайн эти признаки не встроены;
- надежность зависит от TGStat и защиты сайта;
- для масштабного запуска нужен аккуратный throttling и контроль статусов.
