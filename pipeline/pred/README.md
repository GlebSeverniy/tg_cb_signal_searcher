# Предсказания и оценка

Папка хранит итоговые prediction-файлы и метрики полной связки topic + direction.

## Основные файлы

| Путь | Назначение |
|---|---|
| `bert_candidates_topic_predictions_optimum_with_direction_predictions.parquet` | post-level таблица с topic и direction prediction |
| `annotated_direct_revised (2).parquet` | gold-разметка направления для оценки |
| `direction_eval/metrics_summary.json` | сводная оценка |
| `direction_eval/*.csv` | отчёты, confusion matrices и списки ошибок |

## Входы

Topic predictions и gold-разметка направления.

## Выходы

CSV/parquet с метриками, mismatches, missing topic-pairs, extra predictions и confusion matrices.

## Как использовать

Начинайте с:

```text
pipeline/pred/direction_eval/metrics_summary.json
```

Затем смотрите `all_mismatches.csv`, `mismatches_wrong_direction.csv` и `mismatches_missing_predictions_for_gold_pairs.csv`.

## Примечания

Важно различать covered-pairs metrics и strict metrics. Strict-режим считает пропущенные topic-пары ошибкой полного пайплайна.

