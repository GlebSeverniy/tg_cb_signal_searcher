# Inference notebooks

Ноутбуки для применения сохранённых BERT-моделей к новым данным.

## Основные файлы

| Файл | Назначение |
|---|---|
| `kaggle_model_topic_inference.ipynb` | применяет topic model и thresholds |
| `kaggle_model_direction_inference.ipynb` | применяет direction model к предсказанным темам |

## Входы

- Для topic inference: parquet/csv с колонкой `text` и папка модели `pipeline/model_topic/`.
- Для direction inference: topic predictions (`pred_t*` или `bert_topic_labels`) и папка `pipeline/model_direction/`.

## Выходы

Prediction parquet/csv с вероятностями, бинарными метками и агрегатами.

## Как использовать

Ноутбуки рассчитаны на Kaggle. Проверьте пути к `/kaggle/input` или локальным fallback-путям в первых ячейках.

## Примечания

Topic inference использует sigmoid + per-label thresholds. Direction inference использует softmax по классам `-` и `+`.

