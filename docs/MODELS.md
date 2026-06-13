# Модели

## Карта моделей

| Блок | Где лежит | Роль |
|---|---|---|
| Prefilter | `pipeline/classifiers/` | дешёвый отбор кандидатов до BERT |
| Topic BERT | `pipeline/model_topic/` | multi-label классификация по пяти темам ЦБ |
| Direction BERT | `pipeline/model_direction/` | направление `+/-` для каждой пары `пост x тема` |

## Prefilter-классификаторы

Prefilter нужен, чтобы не прогонять через дорогую разметку/инференс весь массив Telegram-постов. Он использует объединение:

```text
topic_any OR is_econom
```

В `pipeline/classifiers/` лежат:

- 7 тематических prefilter-флагов: SVC/CatBoost/joblib-артефакты;
- экономический XGBoost-классификатор;
- `topics/lemmatizer.py`, который обязательно должен использоваться перед темами и экономикой;
- `scripts/run_prefilter_pipeline.py`, `infer_topics_standalone.py`, `infer_econom_standalone.py`;
- отдельные requirements для несовместимых окружений.

Важно: это не финальная BERT topic model. Prefilter-темы - рубрикатор для отбора кандидатов, а пять `t1..t5` - исследовательские сигналы Банка России.

## Topic BERT

Артефакты:

```text
pipeline/model_topic/config (1).json
pipeline/model_topic/model (1).safetensors
pipeline/model_topic/tokenizer (1).json
pipeline/model_topic/tokenizer_config (1).json
pipeline/model_topic/topic_model_config (1).json
pipeline/model_topic/training_args (1).bin
```

Конфиг topic model:

| Параметр | Значение |
|---|---|
| Базовая модель | `ai-forever/ruBert-base` |
| Задача | `multi_label_topic_classification` |
| Labels | `t1_relevant`...`t5_relevant` |
| Text col | `text` |
| Max length | `512` |
| Loss | `BCEWithLogitsLoss` |
| Data mode | `weak_negatives` |
| Weak negative weight | `0.2` |
| Pos weight mode | `sqrt` |
| Best checkpoint metric | `macro_pr_auc` |

Thresholds из сохранённого конфига:

| Label | Threshold |
|---|---:|
| `t1_relevant` | 0.80 |
| `t2_relevant` | 0.35 |
| `t3_relevant` | 0.75 |
| `t4_relevant` | 0.80 |
| `t5_relevant` | 0.95 |

Threshold применяется после sigmoid независимо для каждой темы.

## Base vs large

В ноутбуках есть переключатель `MODEL_SIZE = "base" | "large"`. `large` полезен для экспериментов на GPU, но в текущем `pipeline/model_topic/` сохранён base-артефакт `ai-forever/ruBert-base`.

Для обучения:

- `base` проще запустить и быстрее обучить;
- `large` требует больше VRAM, обычно нужен Kaggle/Colab GPU и аккуратные batch size / gradient accumulation;
- для обоих вариантов `MAX_LENGTH=512` выбран, чтобы не терять длинные Telegram-посты.

## Direction BERT

Артефакты:

```text
pipeline/model_direction/config.json
pipeline/model_direction/model.safetensors
pipeline/model_direction/tokenizer.json
pipeline/model_direction/tokenizer_config.json
pipeline/model_direction/direction_model_config.json
pipeline/model_direction/training_args.bin
```

Конфиг direction model:

| Параметр | Значение |
|---|---|
| Базовая модель | `ai-forever/ruBert-base` |
| Задача | `binary_topic_direction_classification` |
| Input format | `[TOPIC=t_i] text` |
| Labels | `-`, `+` |
| Text col | `model_text` |
| Source text col | `text` |
| Topic col | `topic` |
| Max length | `512` |
| Class weight mode | `balanced` |
| Best checkpoint metric | `macro_f1` |

Direction model запускается только для тем, которые уже предсказала topic model.

## Inference

Topic inference:

```text
pipeline/inferences_notebooks/kaggle_model_topic_inference.ipynb
```

Выход:

```text
prob_t*
pred_t*
bert_topic_any
bert_topic_count
bert_topic_labels
```

Direction inference:

```text
pipeline/inferences_notebooks/kaggle_model_direction_inference.ipynb
```

Выход:

```text
t*_direction_pred
t*_prob_minus
t*_prob_plus
direction_pair_count
direction_plus_topics
direction_minus_topics
```

## Метрики

Основная оценка полной связки:

```text
pipeline/pred/direction_eval/metrics_summary.json
```

Ключевые значения текущего результата:

| Метрика | Значение |
|---|---:|
| covered gold pairs | 538 |
| correct covered pairs | 455 |
| coverage rate gold pairs | 0.632 |
| accuracy на covered pairs | 0.846 |
| macro F1 на covered pairs | 0.822 |
| strict accuracy, missing topic = wrong | 0.535 |
| strict macro F1 | 0.624 |

Итог: direction model на покрытых парах работает заметно лучше, чем полный strict-пайплайн. Основное ограничение полной связки - неполный recall topic-stage.

## CPU/GPU

- Обучение BERT желательно выполнять на GPU.
- ruBERT-large практически требует GPU.
- Инференс на примерно 4000 постов возможен на CPU, но будет медленнее; GPU заметно ускоряет batch inference.
- Prefilter легче BERT, но старые pickle/joblib-модели требуют аккуратного окружения.

