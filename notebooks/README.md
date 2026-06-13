# Ноутбуки

Папка содержит исследовательские и обучающие notebooks. Многие из них рассчитаны на Kaggle/Colab GPU, а не на локальный CPU.

## Основные файлы

| Файл | Назначение |
|---|---|
| `bert_fine_tuning.ipynb` | исторический бинарный baseline `text -> relevant` |
| `notebookca8db8e44a (4).ipynb` | экспериментальная multi-label topic model, ruBERT-large |
| `notebookca8db8e44a (7).ipynb` | основной notebook topic model по `t1..t5` |
| `direction_notation_colab.ipynb` | Colab-версия обучения модели направления |
| `train_direction_rubert_large.ipynb` | Kaggle/large-настройки для direction model |

## Входы

Обычно используются:

```text
pipeline/corpus_train.parquet
pipeline/corpus_test.parquet
pipeline/direction_train.parquet
pipeline/direction_test.parquet
```

## Выходы

Notebook-и сохраняют модели, metrics json/csv, threshold tables и prediction parquet/csv в рабочие папки Kaggle/Colab. Сохранённые финальные артефакты перенесены в `pipeline/model_topic/`, `pipeline/model_direction/`, `pipeline/pred/`.

## Как использовать

Откройте notebook в Kaggle/Colab, проверьте пути к входным parquet и параметры `MODEL_SIZE`, `DATA_MODE`, `POS_WEIGHT_MODE`, `MAX_LENGTH`.

## Примечания

Важно: финальная topic model - multi-label по пяти темам. `relevant` в этих данных является OR-индикатором, а не отдельной BERT-головой.

