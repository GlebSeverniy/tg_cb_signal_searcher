# Схема данных

Документ описывает основные таблицы проекта и ключевые колонки. Точные размеры ниже соответствуют текущей рабочей папке.

## Основные файлы

| Файл | Строк | Колонок | Назначение |
|---|---:|---:|---|
| `data/clean/extracted_messages.parquet` | 2 271 292 | 40 | сообщения, извлечённые из Telegram JSON |
| `data/processed/posts_cleaned.parquet` | 452 142 | 18 | очищенные обычные посты с текстом |
| `pipeline/corpus_full.parquet` | 66 180 | 29 | полный корпус BERT по пяти темам |
| `pipeline/corpus_train.parquet` | 52 946 | 29 | train split для topic model |
| `pipeline/corpus_test.parquet` | 13 234 | 29 | test split для topic model |
| `pipeline/direction_train.parquet` | 2 750 | 35 | train данные направления |
| `pipeline/direction_test.parquet` | 683 | 35 | test данные направления |
| `pipeline/pred/bert_candidates_topic_predictions_optimum_with_direction_predictions.parquet` | 663 | 87 | post-level результат topic + direction |

## Telegram-поля

| Колонка | Где встречается | Смысл |
|---|---|---|
| `channel_id` | почти все parquet | числовой ID Telegram-канала |
| `channel_name` | почти все parquet | название канала |
| `channel_type` | Telegram/raw/corpus | тип сущности из Telegram |
| `message_id` | почти все parquet | ID поста внутри канала |
| `message_type` | Telegram/raw/corpus | `message` или сервисная запись |
| `post_date` | почти все parquet | дата публикации |
| `post_date_unixtime` | Telegram/raw/corpus | Unix-время публикации |
| `edited`, `edited_unixtime` | Telegram/raw/corpus | время редактирования, если есть |
| `from_id` | Telegram/raw/corpus | автор/peer ID, если доступен |
| `text` | почти все parquet | основной вход для BERT |
| `text_entities` | Telegram/raw/corpus | сериализованные entities Telegram |
| `reactions_total` | Telegram/raw/corpus | сумма реакций |
| `has_reactions` | Telegram/raw/corpus | индикатор наличия реакций |
| `views`, `forwards`, `replies` | raw Telegram | метрики, если отданы Telegram API |
| `source_file` | почти все parquet | исходный JSON-файл канала |
| `text_hash` | corpus/pipeline | хэш текста для дедупликации/связки |

## Разметка topic model

Финальная topic model решает multi-label задачу по пяти темам:

| Колонка | Смысл |
|---|---|
| `t1_relevant` | высокая ставка надолго |
| `t2_relevant` | таргет инфляции 4% не пересматривается |
| `t3_relevant` | сберегать сейчас выгодно |
| `t4_relevant` | инфляция связана с перегретым спросом |
| `t5_relevant` | конец массовой льготной ипотеки |

Важно: в BERT target входят только `t1_relevant`...`t5_relevant`.

`relevant` - служебная колонка:

```text
relevant = OR(t1_relevant, t2_relevant, t3_relevant, t4_relevant, t5_relevant)
```

Она удобна для фильтрации и подсчётов, но не является отдельной целевой переменной финальной multi-label модели.

## `annotated` и типы негативов

| Условие | Название | Интерпретация |
|---|---|---|
| `annotated=True`, `relevant=0` | жёсткий негатив | пост был проверен разметкой/LLM и признан нерелевантным |
| `annotated=False`, `relevant=0` | слабый негатив | пост не прошёл prefilter и получил 0 по построению |

Важно: `annotated=False` негативы потенциально шумные. Среди них могут быть реально релевантные посты, которые не прошли prefilter. Поэтому validation/test_clean лучше строить только на `annotated=True`, а слабые негативы использовать в train с пониженным весом.

## Direction-разметка

В direction-таблицах есть колонки:

```text
t1_direction
t2_direction
t3_direction
t4_direction
t5_direction
direction_reasoning
```

Direction model работает не на post-level строках, а на long-представлении `пост x тема`. Входной текст модели строится как:

```text
[TOPIC=t_i] text
```

Целевые классы: `+` и `-`.

## Предсказания topic model

Inference notebook добавляет вероятности и бинарные флаги:

| Колонки | Смысл |
|---|---|
| `prob_t1_relevant`...`prob_t5_relevant` | вероятности после sigmoid |
| `pred_t1_relevant`...`pred_t5_relevant` | бинарные предсказания после thresholds |
| `prob_t1`...`prob_t5` | короткие алиасы вероятностей |
| `pred_t1`...`pred_t5` | короткие алиасы предсказаний |
| `bert_relevance_score` | агрегированная BERT-оценка релевантности |
| `bert_topic_any` | есть хотя бы одна предсказанная тема |
| `bert_topic_count` | количество предсказанных тем |
| `bert_topic_labels` | список предсказанных тем строкой |

Threshold применяется отдельно для каждой темы после sigmoid. Значения сохранены в `pipeline/model_topic/topic_model_config (1).json`.

## Предсказания direction model

Итоговый файл topic + direction содержит:

```text
t1_direction_pred ... t5_direction_pred
t1_prob_minus ... t5_prob_minus
t1_prob_plus ... t5_prob_plus
t1_direction_margin_abs ... t5_direction_margin_abs
direction_pair_count
direction_plus_topics
direction_minus_topics
```

Если topic model не предсказала тему, direction model для этой пары не запускается. Поэтому strict-метрики полного пайплайна зависят от recall topic-stage.

## Prefilter-поля

В candidate-файлах встречаются:

| Колонка | Смысл |
|---|---|
| `prefilter_topic_any` | сработал хотя бы один из 7 тематических prefilter-флагов |
| `prefilter_nbp` | сработала отдельная НБП-голова |
| `prefilter_is_econom` | экономический классификатор определил пост как экономический |
| `prefilter_union` | итоговый отбор: topic_any OR is_econom |
| `prefilter_source` | источник прохождения prefilter-а |

Не путайте 7 тем prefilter-а с пятью BERT-темами `t1..t5`: это разные уровни разметки.

