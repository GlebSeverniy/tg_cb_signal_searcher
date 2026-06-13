# BERT: история обучения, версии, метрики и результаты

## Кратко

В проекте сложился двухэтапный BERT-пайплайн:

1. `topic model`: определяет, к каким из пяти посылов ЦБ относится Telegram-пост.
2. `direction model`: для каждой найденной пары `пост x тема` определяет направление сигнала, `+` или `-`.

До финальной постановки также был отдельный бинарный вариант `relevant / not relevant`. Он описан в ноутбуке, но финальным рабочим артефактом в текущем архиве является именно multi-label topic model.

Основные артефакты:

| Что | Где лежит |
|---|---|
| Корпус BERT | `corpus_full.parquet`, `corpus_train.parquet`, `corpus_test.parquet` |
| Ноутбук обучения topic model | `notebookca8db8e44a (7).ipynb` |
| Финальная topic model | `model_topic/` |
| Ноутбуки topic/direction inference | `notebooks/kaggle_model_topic_inference.ipynb`, `notebooks/kaggle_model_direction_inference.ipynb` |
| Данные и метрики direction model | `model_direction/direction_training/` |
| Финальная direction model | `final_direction_model/` |
| Полная оценка связки topic + direction | `pred/direction_eval/metrics_summary.json` |
| Исторические ноутбуки из соседнего проекта | `../SignalSearcher/telegram_signal_searcher/notebooks/` |

## Данные для BERT

Корпус собран из Telegram-постов вокруг заседаний и дат-зондов ЦБ. Источник постов - полный 100-канальный дамп, затем на него накладывалась LLM-разметка релевантности.

Канонический корпус после ревизии t1/t4:

| Split | Строк | `relevant=1` | `relevant=0` | `annotated=True` |
|---|---:|---:|---:|---:|
| full | 66 180 | 3 433 | 62 747 | 28 051 |
| train | 52 946 | 2 748 | 50 198 | 22 367 |
| test | 13 234 | 685 | 12 549 | 5 684 |

Разбиение `train/test` делалось в `classifiers/scripts/split_train_test.py`: 20% в test, стратификация по комбинации пяти меток `t1..t5`, seed 42. Это сохраняет долю релевантных постов и распределение тем.

Темы:

| Тема | Смысл | Позитивов в full |
|---|---|---:|
| `t1_relevant` | Ставка высокая надолго | 1 219 |
| `t2_relevant` | Таргет инфляции 4% не пересматривается | 506 |
| `t3_relevant` | Сберегать сейчас выгодно | 1 302 |
| `t4_relevant` | Инфляция от перегретого спроса | 648 |
| `t5_relevant` | Конец массовой льготной ипотеки | 738 |

Важный нюанс: `annotated=False` - это посты, которые были отсечены дешевым префильтром и получили `relevant=0` конструктивно. Они полезны как массовые негативы, но могут содержать шум. Поэтому в обучении topic model использовался режим слабых негативов.

## Версия 0: бинарная relevance-модель

Первая постановка была проще:

```text
text -> relevant
```

То есть модель отвечала только на вопрос: относится ли пост хотя бы к одному из пяти посылов ЦБ. Эта версия описана в `../SignalSearcher/telegram_signal_searcher/notebooks/bert_fine_tuning.ipynb`.

Ключевые настройки:

| Параметр | Значение |
|---|---|
| Базовая модель | `DeepPavlov/rubert-base-cased` |
| Цель | бинарный `relevant` |
| Loss | `BCEWithLogitsLoss` с `pos_weight` |
| Основной режим данных | `clean`: только `annotated=True` |
| Дополнительные режимы | `hybrid`, `full` |
| Max length | 512 |
| Epochs | 5 |
| Learning rate | `2e-5` |
| Batch | train 4, eval 8, gradient accumulation 4 |
| Best checkpoint | по `pr_auc` |
| Порог | подбирался по validation на F1 |

Эта версия полезна как исторический baseline, но у нее есть методологический предел: бинарный `relevant` не говорит, какая именно тема сработала. Для дальнейшей модели направления нужна тема, поэтому финальная архитектура перешла к multi-label классификации `t1..t5`.

## Версия 1: multi-label topic model

Финальная topic model решает задачу:

```text
text -> t1_relevant, t2_relevant, t3_relevant, t4_relevant, t5_relevant
```

Один пост может относиться к нескольким темам, поэтому используется sigmoid по каждой теме, а не softmax.

Артефакт: `model_topic/`.

Конфиг из `model_topic/topic_model_config (1).json`:

| Параметр | Значение |
|---|---|
| Базовая модель | `ai-forever/ruBert-base` |
| Архитектура | `BertForSequenceClassification` |
| Hidden size / layers / heads | 768 / 12 / 12 |
| Task | `multi_label_topic_classification` |
| Text col | `text` |
| Max length | 512 |
| Loss | `BCEWithLogitsLoss` |
| Слабые негативы | `annotated=False` в train с весом 0.2 |
| `pos_weight` | `sqrt`, cap 15 |
| Validation size | 0.15 от train |
| Epochs | 4 |
| Learning rate | `2e-5` |
| Weight decay | 0.01 |
| Best checkpoint | `macro_pr_auc` |

Финальные пороги, подобранные на validation:

| Label | Threshold |
|---|---:|
| `t1_relevant` | 0.80 |
| `t2_relevant` | 0.35 |
| `t3_relevant` | 0.75 |
| `t4_relevant` | 0.80 |
| `t5_relevant` | 0.95 |

Почему так:

- `macro_pr_auc` выбран для checkpoint, потому что он не зависит от фиксированного порога 0.5.
- Пороги разные по темам из-за разного баланса классов и разной уверенности модели.
- `weak_negatives` снижает вред от шумных `annotated=False`, но позволяет дать модели больше отрицательного контекста.

## Метрики topic-stage на сохраненном candidate-файле

Полного `topic_metrics.json` от training run в текущей папке нет. Поэтому ниже приведена воспроизводимая оценка по сохраненному файлу `data/bert_candidates_topic_predictions_optimum.parquet`, где есть и gold labels, и BERT-предсказания.

Контекст файла:

| Показатель | Значение |
|---|---:|
| Строк | 4 280 |
| Gold relevant posts | 663 |
| Gold topic-pairs | 851 |
| Predicted topic-pairs | 729 |
| Correct topic-pairs | 538 |

Метрики по topic-pairs:

| Average | Precision | Recall | F1 |
|---|---:|---:|---:|
| micro | 0.738 | 0.632 | 0.681 |
| macro | 0.714 | 0.617 | 0.656 |
| weighted | 0.719 | 0.632 | 0.666 |

Дополнительно:

| Метрика | Значение |
|---|---:|
| Exact match по 5 меткам | 0.897 |
| Macro PR-AUC | 0.685 |
| Micro PR-AUC | 0.721 |
| Any-topic precision | 0.806 |
| Any-topic recall | 0.735 |
| Any-topic F1 | 0.769 |

Per-topic:

| Тема | Support | Pred | Correct | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|
| t1 | 239 | 261 | 211 | 0.808 | 0.883 | 0.844 | 0.890 |
| t2 | 96 | 69 | 44 | 0.638 | 0.458 | 0.533 | 0.567 |
| t3 | 251 | 165 | 103 | 0.624 | 0.410 | 0.495 | 0.530 |
| t4 | 123 | 97 | 61 | 0.629 | 0.496 | 0.555 | 0.585 |
| t5 | 142 | 137 | 119 | 0.869 | 0.838 | 0.853 | 0.853 |

Главный вывод: `t1` и `t5` модель ловит хорошо, а `t2`, `t3`, `t4` заметно проседают по recall. Именно этот recall потом ограничивает полный pipeline с direction model: если topic-stage не создал пару `пост x тема`, direction-stage уже не сможет ее исправить.

## Префильтр перед BERT

Перед BERT использовался дешевый префильтр `topic_any OR is_econom`, чтобы не прогонять все посты через дорогую разметку/инференс.

На `corpus_test.parquet`:

| Показатель | Значение |
|---|---:|
| Всего постов | 13 234 |
| После фильтра `message` и непустого текста | 13 234 |
| Уникальных текстов | 13 137 |
| Выбрано темами | 3 417 |
| Из них НБП | 90 |
| Выбрано эконом-классификатором | 2 264 |
| Union candidates | 4 280 |
| Доля выбранных | 32.3% |
| Gold relevant в test | 685 |
| Gold relevant выбранных | 663 |
| Missed relevant | 22 |
| Recall префильтра на relevant | 96.79% |

Префильтр значительно сокращает объем, но он не является частью BERT-модели. Это отдельный upstream-фильтр с собственным recall.

## Версия 2: direction model

Direction model нужна после topic model. Она отвечает не "о чем пост", а "как он влияет на восприятие темы":

```text
[TOPIC=t_i] text -> "+" / "-"
```

Почему вход именно такой: один и тот же пост может относиться к нескольким темам, и направление может различаться по темам. Поэтому обучающая единица - это пара `пост x тема`.

Артефакты:

| Что | Где |
|---|---|
| Подготовленные long-датасеты | `model_direction/direction_training/direction_dataset/` |
| Метрики | `model_direction/direction_training/direction_metrics.json` |
| Финальная модель | `final_direction_model/` |

Данные direction model:

| Split | Pair rows | `+` | `-` |
|---|---:|---:|---:|
| train | 3 022 | 1 817 | 1 205 |
| valid | 526 | 297 | 229 |
| test | 865 | 513 | 352 |

По темам в test:

| Тема | Pair rows |
|---|---:|
| t1 | 240 |
| t2 | 98 |
| t3 | 256 |
| t4 | 128 |
| t5 | 143 |

Настройки финальной base-модели:

| Параметр | Значение |
|---|---|
| Базовая модель | `ai-forever/ruBert-base` |
| Архитектура | `BertForSequenceClassification` |
| Hidden size / layers / heads | 768 / 12 / 12 |
| Labels | `-`: 0, `+`: 1 |
| Loss | `CrossEntropyLoss` |
| Class weights | `-`: 1.254, `+`: 0.832 |
| Max length | 512 |
| Epochs | 4 |
| Learning rate | `2e-5` |
| Weight decay | 0.01 |
| Best checkpoint | `macro_f1` |

В архиве есть следы `ruBert-large`: `model_direction/config.json` и `model_direction/direction_model_config.json` указывают на large-конфигурацию. Но сохраненные checkpoints в `model_direction/direction_training/rubert_base_direction_binary/` и `final_direction_model/` - это base-модель. Метрики ниже относятся к сохраненному base-training run.

Validation history:

| Epoch | Eval macro-F1 | Accuracy | F1 `-` | F1 `+` |
|---:|---:|---:|---:|---:|
| 1 | 0.676 | 0.696 | 0.596 | 0.756 |
| 2 | 0.739 | 0.745 | 0.698 | 0.780 |
| 3 | 0.688 | 0.703 | 0.618 | 0.758 |
| 4 | 0.701 | 0.715 | 0.636 | 0.766 |

Лучший checkpoint: epoch 2, step 378, `eval_macro_f1 = 0.7389`.

Test metrics direction model:

| Метрика | Значение |
|---|---:|
| Accuracy | 0.762 |
| Macro precision | 0.754 |
| Macro recall | 0.760 |
| Macro F1 | 0.756 |
| Weighted F1 | 0.763 |
| F1 `-` | 0.719 |
| F1 `+` | 0.794 |
| ROC-AUC | 0.828 |
| PR-AUC plus | 0.875 |

Confusion matrix:

| | pred `-` | pred `+` |
|---|---:|---:|
| true `-` | 263 | 89 |
| true `+` | 117 | 396 |

Per-topic test metrics:

| Topic | n | Accuracy | Macro F1 | F1 `-` | F1 `+` |
|---|---:|---:|---:|---:|---:|
| t1 | 240 | 0.788 | 0.744 | 0.638 | 0.850 |
| t2 | 98 | 0.735 | 0.723 | 0.667 | 0.780 |
| t3 | 256 | 0.746 | 0.741 | 0.706 | 0.777 |
| t4 | 128 | 0.844 | 0.829 | 0.880 | 0.778 |
| t5 | 143 | 0.692 | 0.688 | 0.651 | 0.725 |

## Оценка полного pipeline: topic + direction

Файл: `pred/direction_eval/metrics_summary.json`.

На полной связке сначала topic model выбирает релевантные темы, затем direction model ставит `+/-`. Поэтому в метриках появляются два режима:

- covered pairs: оцениваем только пары, где topic-stage дал предсказание;
- strict gold pairs: пропущенная topic-пара считается ошибкой.

Сводка:

| Показатель | Значение |
|---|---:|
| Gold posts raw | 5 257 |
| Prediction posts raw | 663 |
| Overlap posts for metrics | 663 |
| Gold direction pairs on overlap | 851 |
| Predicted direction pairs | 605 |
| Covered gold pairs | 538 |
| Correct covered pairs | 455 |
| Wrong direction pairs | 83 |
| Missing predictions for gold pairs | 313 |
| Extra predictions without gold pair | 67 |
| Coverage rate gold pairs | 0.632 |

Covered-pairs metrics:

| Метрика | Значение |
|---|---:|
| Accuracy | 0.846 |
| Macro F1 | 0.822 |
| Weighted F1 | 0.844 |

Strict metrics, где missing topic-pair считается wrong:

| Метрика | Значение |
|---|---:|
| Accuracy | 0.535 |
| Micro F1 | 0.655 |
| Macro F1 | 0.624 |
| Weighted F1 | 0.644 |

Главный вывод: сама direction model на покрытых парах работает достаточно сильно, но полный pipeline проседает из-за неполного покрытия topic-stage.

## Итоговая логика пайплайна

1. Telegram-посты собираются и нормализуются.
2. Префильтр `topic_any OR is_econom` отбирает кандидатов.
3. Topic BERT ставит вероятности и бинарные метки `t1..t5`.
4. Для каждой предсказанной темы строится строка `[TOPIC=t_i] text`.
5. Direction BERT ставит `+` или `-`.
6. Итоговая таблица хранит post-level результат: topic probabilities, topic labels, direction predictions and probabilities.

## Что важно помнить при защите/описании

- BERT topic model - multi-label, не multi-class.
- Direction model - single-label binary classification на long-таблице `post x topic`.
- `annotated=False` негативы не равны ручной/LLM-проверенной истине; это слабые негативы.
- Лучшие direction-метрики нельзя напрямую сравнивать со strict pipeline-метриками: strict включает ошибки topic-stage.
- В текущем архиве topic training report не сохранен как отдельный `topic_metrics.json`, поэтому часть topic-метрик пересчитана по сохраненному prediction-файлу.
