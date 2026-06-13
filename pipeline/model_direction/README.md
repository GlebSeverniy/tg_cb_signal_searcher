# Direction model

Сохранённая ruBERT-модель направления сигнала. Она классифицирует пары `пост x тема` в `+` или `-`.

## Основные файлы

| Файл | Назначение |
|---|---|
| `config.json` | Transformers config |
| `model.safetensors` | веса модели |
| `tokenizer.json` | tokenizer |
| `tokenizer_config.json` | tokenizer config |
| `direction_model_config.json` | labels, формат входа и параметры обучения |
| `training_args.bin` | сохранённые TrainingArguments |
| `tokenizer.zip` | архивная копия tokenizer-а |

## Входы

Long-таблица с колонками `model_text`, `topic`, `text`. `model_text` строится как:

```text
[TOPIC=t_i] text
```

## Выходы

`direction_pred`, `prob_minus`, `prob_plus`, а после сборки на post-level - `t1_direction_pred`...`t5_direction_pred`.

## Как использовать

Основной notebook:

```text
pipeline/inferences_notebooks/kaggle_model_direction_inference.ipynb
```

## Примечания

Direction model не исправляет пропущенные темы: если topic model не создала пару `пост x тема`, direction inference для неё не запускается.

