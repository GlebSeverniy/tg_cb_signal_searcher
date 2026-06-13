# Как запускать проект

Команды ниже рассчитаны на запуск из корня репозитория.

## 1. Подготовка окружения

Проверенный локальный стек в рабочей папке использовал Python 3.12 для Telegram/TGStat-скриптов. BERT-ноутбуки запускались в Kaggle/Colab GPU-окружениях, где версии PyTorch/CUDA задавались платформой.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -U pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m playwright install chromium
```

Важно: `pipeline/classifiers/` содержит старые pickle/joblib-модели. Для них лучше использовать отдельные pinned-файлы:

```text
pipeline/classifiers/requirements-topics311.txt
pipeline/classifiers/requirements-econom_extractor.txt
```

## 2. Сбор Telegram-постов

Вход: `configs/channels.txt`, по одному каналу на строку.

Нужны credentials Telegram API:

```powershell
$env:TG_API_ID="123456"
$env:TG_API_HASH="your_api_hash"
```

Ключи и `.session`-файлы не надо сохранять в репозиторий.

Проверить соединение:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py --check-connection
```

Скачать посты:

```powershell
.\.venv\Scripts\python.exe src\download_tg_channels.py `
  --channels-file configs\channels.txt `
  --output-dir data\raw\telegram_json\new `
  --session-file configs\telegram_downloader `
  --limit 1000
```

Без `--limit` скрипт скачивает все доступные сообщения в периоде, заданном в коде `DATE_FROM` / `DATE_TO`. В текущей версии это `2024-06-01` - `2024-09-01` UTC.

## 3. Извлечение и очистка данных

`src/extract_posts.py` не принимает CLI-аргументы. Он читает:

```text
data/raw/telegram_json/new/*.json
```

и создаёт:

```text
data/clean/extracted_messages.parquet
logs/extract_log.csv
```

Запуск:

```powershell
.\.venv\Scripts\python.exe src\extract_posts.py
```

`src/clean_data.py` тоже рассчитан на фиксированные пути внутри файла:

```text
input:  data/clean/extracted_messages.parquet
output: data/processed/posts_cleaned.parquet
```

Запуск:

```powershell
.\.venv\Scripts\python.exe src\clean_data.py
```

## 4. Разбиение на shard-ы для TGStat

`src/split_shards.py` создаёт CSV-файлы `shard_000.csv`, `shard_001.csv` и т.д. с колонками `channel_username`, `post_id`.

```powershell
.\.venv\Scripts\python.exe src\split_shards.py `
  --input data\clean\extracted_messages.parquet `
  --output-dir data\clean\shards `
  --n-shards 12 `
  --seed 42
```

Для быстрой проверки:

```powershell
.\.venv\Scripts\python.exe src\split_shards.py --limit 1000
```

## 5. TGStat: просмотры и подписчики

Динамика просмотров постов:

```powershell
.\.venv\Scripts\python.exe src\tgstat_views_collector.py `
  --input data\posts_to_check\posts.csv `
  --output data\clean\tgstat_views.csv `
  --html-cache-dir data\cache\tgstat_html `
  --debug-html-dir data\debug\tgstat_html `
  --use-playwright `
  --user-data-dir data\playwright_profile `
  --headless false `
  --resume
```

Вход должен содержать `channel_username` и `post_id`. При `--resume` строки с уже успешным `status=ok` пропускаются.

Подписчики через TGStat:

```powershell
.\.venv\Scripts\python.exe src\tgstat_subscribers_collector.py `
  --input outputs\extracted_channels.csv `
  --output outputs\extracted_channels_with_subscribers.csv `
  --use-playwright `
  --resume
```

Подписчики через Telegram API:

```powershell
.\.venv\Scripts\python.exe src\add_telegram_subscribers.py `
  --input outputs\extracted_channels.csv `
  --output outputs\extracted_channels_with_subscribers.csv `
  --channel-col channel_username `
  --limit 100
```

## 6. Prefilter перед BERT

Главный wrapper:

```powershell
python pipeline\classifiers\scripts\run_prefilter_pipeline.py `
  --input pipeline\corpus_test.parquet `
  --output pipeline\data\prefilter\bert_candidates.parquet `
  --report pipeline\data\prefilter\prefilter_report.json `
  --work-dir pipeline\data\prefilter\work `
  --overwrite
```

Важно: для воспроизводимости prefilter-а нужны отдельные окружения из `pipeline/classifiers/requirements-*.txt`. Демо-скрипты `infer_topics_standalone.py` и `infer_econom_standalone.py` используют относительные пути, но econom-модель требует старый `scikit-learn`.

## 7. Обучение BERT topic model

Основной notebook: `notebooks/notebookca8db8e44a (7).ipynb`.

Входы:

```text
pipeline/corpus_train.parquet
pipeline/corpus_test.parquet
```

Ключевые параметры:

| Параметр | Смысл |
|---|---|
| `MODEL_SIZE` | `base` или `large`; сохранённая модель в `pipeline/model_topic/` - base |
| `DATA_MODE` | режим работы с `annotated=True/False`; финальный конфиг использует `weak_negatives` |
| `WEAK_NEGATIVE_WEIGHT` | вес слабых негативов; в сохранённом конфиге `0.2` |
| `POS_WEIGHT_MODE` | компенсация дисбаланса; в сохранённом конфиге `sqrt` |
| `MAX_LENGTH` | максимальная длина токенизации, обычно `512` |

Обучение лучше запускать на GPU. Локальные файлы `src/build_dataset.py`, `src/train_model.py`, `src/predict.py` сейчас пустые заготовки и не являются entry point.

## 8. Инференс BERT topic model

Notebook: `pipeline/inferences_notebooks/kaggle_model_topic_inference.ipynb`.

Нужны файлы модели из `pipeline/model_topic/`:

```text
config (1).json
model (1).safetensors
tokenizer (1).json
tokenizer_config (1).json
topic_model_config (1).json
```

Notebook нормализует имена в рабочей папке и читает thresholds из `topic_model_config.json`. На выходе добавляются:

```text
prob_t1_relevant ... prob_t5_relevant
pred_t1_relevant ... pred_t5_relevant
prob_t1 ... prob_t5
pred_t1 ... pred_t5
bert_topic_any
bert_topic_count
bert_topic_labels
```

Чтобы вручную ослабить пороги для сложных тем, правьте `thresholds` в `topic_model_config.json`, например:

```json
{
  "t3_relevant": 0.05,
  "t4_relevant": 0.05
}
```

## 9. Direction inference

Notebook: `pipeline/inferences_notebooks/kaggle_model_direction_inference.ipynb`.

Он берёт topic-предсказания (`pred_t*` или `bert_topic_labels`), строит пары:

```text
[TOPIC=t_i] text
```

и применяет `pipeline/model_direction/`. На выходе появляются `*_direction_pred`, `*_prob_minus`, `*_prob_plus`, `direction_pair_count`, `direction_plus_topics`, `direction_minus_topics`.

## 10. Метрики и проверка результатов

Основная сводка:

```text
pipeline/pred/direction_eval/metrics_summary.json
```

Рядом лежат CSV/parquet:

```text
classification_report_*.csv
per_topic_metrics_*.csv
confusion_matrix_*.csv
all_mismatches.csv
mismatches_wrong_direction.csv
mismatches_missing_predictions_for_gold_pairs.csv
```

Для интерпретации важно различать `covered_pairs` и `strict_gold_pairs`: в strict-режиме пропущенная topic-пара считается ошибкой direction-пайплайна целиком.

