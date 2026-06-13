# telegram_signal_searcher

Проект для курсовой работы о том, как сигналы Банка России распространяются и интерпретируются в Telegram. В репозитории есть два связанных слоя:

- сбор и подготовка Telegram/TGStat-данных (`src/`, `data/`, `outputs/`);
- исследовательский ML-пайплайн: prefilter-классификаторы, BERT/ruBERT-модель по пяти темам, модель направления сигнала и оценка результатов (`pipeline/`, `notebooks/`).

Рекомендуемая точка входа для нового человека - этот README, затем `docs/RUNNING.md`, `docs/DATA_SCHEMA.md` и `docs/MODELS.md`.

## Структура проекта

| Путь | Назначение |
|---|---|
| `configs/` | списки Telegram-каналов и локальные Telethon session-файлы; ключи не хранить в git |
| `data/` | сырые JSON, очищенные parquet/csv, разметка, TGStat-кэши и временные файлы |
| `src/` | основные CLI-скрипты для Telegram, TGStat, очистки и подготовки shard-ов |
| `notebooks/` | исследовательские и обучающие ноутбуки BERT/ruBERT |
| `pipeline/` | корпус, prefilter-комплект, сохранённые BERT-модели, inference-ноутбуки и метрики |
| `pipeline/classifiers/` | самодостаточный комплект дешёвых prefilter-классификаторов |
| `pipeline/model_topic/` | сохранённая multi-label topic model по пяти темам |
| `pipeline/model_direction/` | сохранённая binary direction model для пар `пост x тема` |
| `pipeline/pred/` | итоговые предсказания и evaluation-таблицы |
| `outputs/` | отчёты и CSV, созданные скриптами из `src/` |
| `logs/` | логи скачивания и парсинга Telegram |
| `docs/` | сжатая документация по запуску, данным, моделям и обучению |

Папки `data/`, `models/`, `outputs/`, `logs/`, большие parquet/json/model-файлы и локальные `.session` обычно не должны попадать в git. В текущей рабочей папке они присутствуют как исследовательские артефакты.

## Основной пайплайн

1. Сбор Telegram-постов через Telethon в JSON.
2. Извлечение сообщений из JSON в `data/clean/extracted_messages.parquet`.
3. Очистка и нормализация постов в `data/processed/posts_cleaned.parquet`.
4. Подготовка списков постов/shard-ов для TGStat и сбор просмотров/подписчиков.
5. Prefilter: дешёвые тематические и экономические классификаторы отбирают кандидатов.
6. Формирование BERT-корпуса с пятью темами сигналов Банка России.
7. Обучение multi-label ruBERT topic model.
8. Инференс topic model и подбор/применение threshold-ов.
9. Direction model определяет `+` или `-` для каждой пары `пост x тема`.
10. Метрики, error analysis и агрегация результатов для курсовой.

## Данные

Ключевые файлы:

| Файл | Роль |
|---|---|
| `configs/channels.txt` | входной список Telegram-каналов |
| `data/raw/telegram_json/new/*.json` | сырые выгрузки Telegram API за заданный период |
| `data/clean/extracted_messages.parquet` | объединённая таблица сообщений; 2 271 292 строки, 40 колонок в текущей папке |
| `data/processed/posts_cleaned.parquet` | очищенные посты; 452 142 строки, 18 колонок |
| `pipeline/corpus_full.parquet` | полный BERT-корпус; 66 180 строк |
| `pipeline/corpus_train.parquet` | train split; 52 946 строк |
| `pipeline/corpus_test.parquet` | test split; 13 234 строки |
| `pipeline/direction_train.parquet` / `pipeline/direction_test.parquet` | данные для модели направления |
| `pipeline/pred/bert_candidates_topic_predictions_optimum_with_direction_predictions.parquet` | итоговая таблица topic + direction prediction |
| `pipeline/pred/direction_eval/metrics_summary.json` | сводка оценки полной связки topic + direction |

Подробная схема колонок: `docs/DATA_SCHEMA.md`.

## Модели

В проекте есть три типа моделей:

- Prefilter-классификаторы в `pipeline/classifiers/`: 7 тематических SVC/CatBoost-голов и экономический XGBoost-классификатор. Они используются для дешёвого отбора кандидатов до BERT.
- Topic model в `pipeline/model_topic/`: `ai-forever/ruBert-base`, multi-label classification по `t1_relevant`...`t5_relevant`, sigmoid + отдельный threshold на каждую тему.
- Direction model в `pipeline/model_direction/`: `ai-forever/ruBert-base`, binary classification `+/-` для входа вида `[TOPIC=t_i] text`.

Важно: `relevant` не является отдельной целевой переменной финальной topic model. Это служебный OR-индикатор: `relevant = OR(t1_relevant..t5_relevant)`.

Подробности по артефактам и CPU/GPU: `docs/MODELS.md`.

## Как запустить

Создать окружение и установить общие зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

Для Telegram API нужны переменные окружения:

```powershell
$env:TG_API_ID="123456"
$env:TG_API_HASH="your_api_hash"
```

Проверить соединение с Telegram:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py --check-connection
```

Скачать каналы в JSON:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py `
  --channels-file configs\channels.txt `
  --output-dir data\raw\telegram_json\new `
  --limit 1000
```

Извлечь посты и очистить таблицу:

```powershell
.\.venv\Scripts\python.exe src\extract_posts.py
.\.venv\Scripts\python.exe src\clean_data.py
```

Сделать CSV-shard-ы для TGStat:

```powershell
.\.venv\Scripts\python.exe src\split_shards.py `
  --input data\clean\extracted_messages.parquet `
  --output-dir data\clean\shards `
  --n-shards 12
```

Собрать динамику просмотров TGStat:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\clean\tgstat_views.csv `
  --use-playwright `
  --user-data-dir data\playwright_profile `
  --resume
```

Обучение BERT обычно выполнялось в Kaggle/Colab notebooks, а не из пустых файлов `src/train_model.py` и `src/predict.py`. Основной notebook topic model: `notebooks/notebookca8db8e44a (7).ipynb`. Inference notebooks лежат в `pipeline/inferences_notebooks/`.

Больше сценариев: `docs/RUNNING.md`.

## Важные caveats

- `relevant` - только служебный OR-индикатор, target для BERT topic model состоит из пяти колонок `t1_relevant`...`t5_relevant`.
- `annotated=True, relevant=0` - жёсткий негатив, проверенный разметкой.
- `annotated=False, relevant=0` - слабый негатив, отсеянный prefilter-ом; среди таких строк возможны пропущенные релевантные посты.
- `test_clean`/validation должны строиться только по `annotated=True`; `test_full` годится скорее для диагностики.
- `t3_relevant` и `t4_relevant` оказались сложнее по recall; thresholds можно корректировать после обучения.
- Prefilter и BERT решают разные задачи: 7 тем prefilter-а не равны 5 темам сигналов ЦБ.
- Обучение ruBERT-large требует GPU. Инференс на нескольких тысячах постов возможен на CPU, но GPU быстрее.
- `pipeline/classifiers/` использует два несовместимых окружения для старых pickle/joblib-артефактов; точные файлы зависимостей лежат рядом.

## Воспроизводимость

В коде и ноутбуках чаще всего используется `random_state=42` / `seed=42`. Для BERT-пайплайна ключевые настройки сохранены в:

- `pipeline/model_topic/topic_model_config (1).json`;
- `pipeline/model_direction/direction_model_config.json`;
- `pipeline/docs/bert_training_history.md`.

Корневой `requirements.txt` описывает общее окружение для Telegram/TGStat и BERT-ноутбуков. Для prefilter-классификаторов используйте отдельные файлы:

- `pipeline/classifiers/requirements-topics311.txt`;
- `pipeline/classifiers/requirements-econom_extractor.txt`.

## Быстрый старт для чтения проекта

1. Откройте `docs/DATA_SCHEMA.md`, чтобы понять колонки и splits.
2. Откройте `docs/MODELS.md`, чтобы отличать prefilter, topic model и direction model.
3. Для запуска сбора данных используйте `docs/RUNNING.md`.
4. Для истории обучения и методологических решений используйте `docs/TRAINING_PROCESS.md` и `pipeline/docs/bert_training_history.md`.
