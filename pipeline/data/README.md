# Данные pipeline

Папка содержит входы и промежуточные результаты для pipeline-инференса.

## Основные файлы

| Путь | Назначение |
|---|---|
| `RELEVANCE.parquet` | вход/архив для inference-ноутбуков |
| `bert_candidates_topic_predictions_optimum.parquet` | topic predictions до direction-stage |
| `prefilter/` | результаты prefilter-запусков и рабочие файлы |

## Входы

Parquet с Telegram-постами и колонкой `text`.

## Выходы

Candidate/parquet-файлы с prefilter или topic-prediction колонками.

## Примечания

Эта папка дополняет `pipeline/corpus_*.parquet`, которые лежат на верхнем уровне `pipeline/`.

