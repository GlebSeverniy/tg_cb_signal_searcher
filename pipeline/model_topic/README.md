# Topic model

Сохранённая ruBERT-модель для multi-label классификации Telegram-постов по пяти темам сигналов Банка России.

## Основные файлы

| Файл | Назначение |
|---|---|
| `config (1).json` | Transformers config |
| `model (1).safetensors` | веса модели |
| `tokenizer (1).json` | tokenizer |
| `tokenizer_config (1).json` | tokenizer config |
| `topic_model_config (1).json` | labels, thresholds и параметры обучения |
| `training_args (1).bin` | сохранённые TrainingArguments |

## Входы

Parquet/csv с колонкой `text`. Для оценки нужны `t1_relevant`...`t5_relevant`.

## Выходы

Inference добавляет `prob_t*`, `pred_t*`, `bert_topic_any`, `bert_topic_count`, `bert_topic_labels`.

## Как использовать

Основной notebook:

```text
pipeline/inferences_notebooks/kaggle_model_topic_inference.ipynb
```

## Примечания

Важно: файлы имеют суффикс ` (1)`; inference notebook нормализует имена в рабочей папке. Thresholds лежат в `topic_model_config (1).json`.

