# Комплект классификаторов-префильтров — инструкция

Этот каталог (`classifiers/`) — **самодостаточный комплект двух дешёвых классификаторов**,
которыми отбирались посты-кандидаты ДО дорогой LLM-разметки релевантности. Их объединение
(«**тема ∪ эконом**») — это тот самый префильтр, по которому строились union-вырезки корпуса
в родительской папке (`corpus_full.parquet` и т.д.).

Всё, что нужно для запуска (модели + исходный код для распаковки + скрипты + демо), лежит здесь
и работает **по относительным путям**. Демо-скрипты протестированы из этого каталога.

---

## 0. TL;DR — как быстро убедиться, что всё живо

```bash
cd classifiers/scripts

# 7 тем на сырых текстах (env topics311):
/opt/anaconda3/envs/topics311/bin/python infer_topics_standalone.py

# econom-классификатор на леммах (env econom_extractor):
/opt/anaconda3/envs/econom_extractor/bin/python infer_econom_standalone.py
```
Оба печатают предсказания на встроенных примерах. На свой паркет — флаги `--parquet` / `--out`
(см. шапку скриптов).

---

## 1. Что это за два классификатора

### A. Тематический классификатор — 7 тем (v3)
Мультилейбл, 7 независимых бинарных голов. Вход — **лемматизированный** текст. Выход на тему:
`topic_<Тема> ∈ {0,1}` и `score_<Тема>` (сырое `decision_function`); `topic = (score >= τ)`,
порог τ зашит в модель (OOF-калиброванный, не пере-тюнить).

Состоит из двух артефактов:
- **6 бинарных SVC-голов** (TF-IDF + LinearSVC, обёрнуты порогом) — файл
  `topics/binary_per_topic_classifier/results/svc_per_topic_pipeline.joblib` (33 МБ).
  Темы: `Защита прав потребителей`, `ДКП`, `Финансовый рынок`, `Финтех`, `Геополитика`, `НДО`.
- **НБП — отдельный CatBoost-стэкер** (база A/F + keyword-фичи + мета-модель), потому что НБП плохо
  ловился линейной моделью — файл `topics/svc_focused_v2/results/nbp_v3/nbp_stacker.joblib` (9 МБ).

Порядок тем в финальной разметке (как в корпусе):
`Защита прав потребителей, ДКП, Финансовый рынок, НБП, Финтех, Геополитика, НДО`.

**Пороги τ** (topic=1, если score ≥ τ):

| Тема | τ |
|---|---:|
| Защита прав потребителей | −0.4762 |
| ДКП | −0.0584 |
| Финансовый рынок | −0.0540 |
| Финтех | −0.5504 |
| Геополитика | −0.5246 |
| НДО | −0.3363 |
| НБП (CatBoost-стэкер) | **+2.8560** |

**Качество (test, v3):** macro-F1 = **0.6975** (bootstrap CI95 [0.660, 0.730]).
Per-theme F1: Защита 0.571 · ДКП 0.748 · Фин.рынок 0.788 · НБП 0.615 · Финтех 0.594 · Геополитика 0.580 · НДО 0.800.
(НБП-стэкер v3 поднял F1 НБП 0.548 → 0.615. Подробности: `topics/svc_focused_v2/results/nbp_v3/NBP_IMPROVEMENT_REPORT.md`, `v3_report.json`.)

> Историческая заметка: была ещё 8-я тема «ДКП в мире» — в v3-сетапе **не используется**.
> Здесь ровно 7 тем.

### B. Econom-классификатор — экономическая ли новость
TF-IDF + XGBoost, **6 классов**, `class 0 = экономическая`. Вход — **те же леммы**, поверх которых
модель делает свою регэксп-очистку (стопслова/мусорные фразы/не-буквы). Выход:
`econom_class ∈ {0..5}`, `is_econom = (class==0)`, `p_econom = P(class0)`, `p_c0..p_c5`.
Артефакты: `econom/models/class_model/{tfidf.pkl (vocab 11151), xgboost_6classes.pkl, stop_words.pkl, trash_phrases.pkl}`.
Происхождение — внешний репозиторий `russian_news_database` (см. `econom/econom_repo_README.md`).

### Префильтр = объединение
`тема ∪ эконом` = пост прошёл, если у него **≥1 темы ИЛИ is_econom=1**. Именно по этому правилу
из полного дампа отбирались посты на LLM-разметку. На полном 100-канальном дампе доля прошедших
≈ 30%. Recall префильтра против LLM-эталона ≈ 83–90% (см. `scripts/eval_analyze.py`).

---

## 2. Карта каталога

```
classifiers/
├── INSTRUCTIONS.md                ← этот файл
├── topics/                        ← тематический классификатор (7 тем)
│   ├── lemmatizer.py              ← ЛЕММАТИЗАТОР (pymorphy3). Обязателен ДО любой темы/эконома
│   ├── binary_per_topic_classifier/
│   │   ├── results/svc_per_topic_pipeline.joblib   ← 6 бинарных SVC-голов
│   │   └── src/                   ← код пакета `src` (нужен joblib'у для распаковки!)
│   └── svc_focused_v2/
│       ├── results/nbp_v3/
│       │   ├── nbp_stacker.joblib ← НБП CatBoost-стэкер
│       │   ├── nbp_predictor.py   ← классы NBPStackedClassifier/KeywordCountTransformer (для распаковки)
│       │   ├── NBP_IMPROVEMENT_REPORT.md, v3_report.json
│       └── src/                   ← второй пакет `src` (нужен для распаковки НБП)
├── econom/                        ← экономический классификатор
│   ├── models/class_model/{tfidf,xgboost_6classes,stop_words,trash_phrases}.pkl
│   └── econom_repo_README.md
└── scripts/
    ├── infer_topics_standalone.py ← ДЕМО (относит. пути): сырой текст → 7 тем
    ├── infer_econom_standalone.py ← ДЕМО (относит. пути): леммы → эконом
    ├── step1_prep_lemmatize.py    ← прод-пайплайн: дедуп + лемматизация
    ├── step2_topics.py            ← прод-пайплайн: 7 тем на уникальных леммах
    ├── step3_econom.py            ← прод-пайплайн: эконом на уникальных леммах
    ├── step4_assemble.py          ← прод-пайплайн: сборка финального паркета
    ├── eval_topics.py, eval_econom.py, eval_analyze.py  ← замер recall префильтра на LLM-эталоне
    ├── build_bert_corpus.py       ← сборка корпуса BERT из размеченных вырезок
    ├── split_train_test.py        ← стратифицированный train/test
    └── apply_t1t4_revision.py     ← применение ревизии тем t1/t4 (см. корневой README)
```

---

## 3. Окружения (conda) — строго по версиям

Два РАЗНЫХ окружения: пиклы обучены на несовместимых версиях sklearn/xgboost.

| Env | Python | Ключевые пакеты | Для чего |
|---|---|---|---|
| **topics311** | 3.11 | sklearn **1.6.1**, pyarrow 24, pandas 3.0, numpy 1.26.4, **catboost 1.2.9**, pymorphy3, nltk | лемматизация + 7 тем (step1, step2, infer_topics) |
| **econom_extractor** | 3.9 | sklearn **1.0.2**, **xgboost 1.7.6**, scipy 1.10.1, numpy 1.23.5 | эконом (step3, infer_econom) |

Запуск интерпретатором env напрямую:
`/opt/anaconda3/envs/topics311/bin/python ...`, `/opt/anaconda3/envs/econom_extractor/bin/python ...`.

> econom-пиклы сделаны на sklearn 1.0.2 — в свежем sklearn распаковка ломается. XGBoost при загрузке
> печатает WARNING про «старый сериализованный бустер» — **это безобидно**, предсказания корректны.

---

## 4. ⚠️ Две ловушки, без которых ничего не заведётся

### Ловушка 1 — коллизия пакета `src`
`binary_per_topic_classifier` и `svc_focused_v2` **оба содержат пакет с именем `src`** с похожими
классами. joblib при распаковке импортирует модуль буквально как `src.pipeline` / `src.model`.
Поэтому:
- грузить **binary ПЕРВЫМ** (с его путями в `sys.path`) — тогда `sys.modules['src']` кэшируется на
  binary-версию (там лежит `ThresholdedBinaryClassifier`);
- **потом** добавлять пути `svc_focused_v2` и грузить НБП.
- **Порядок не менять.** Точный рецепт — в `scripts/step2_topics.py` и `scripts/infer_topics_standalone.py`.
- Лемматизатор грузить **по пути файла** (`importlib`, под именем НЕ `src`), чтобы не занять имя `src`
  раньше времени — см. `infer_topics_standalone.py` / `eval_topics.py`.

Из-за этого тематический инференс держат **в одном процессе** строго в нужном порядке, а эконом —
**в отдельном процессе** (другой env). Не пытаться грузить обе темы-модели «в любом порядке».

### Ловушка 2 — лемматизация ОБЯЗАТЕЛЬНА и должна быть ТОЙ ЖЕ
Обе модели обучены на леммах из `topics/lemmatizer.py` (pymorphy3, конкретная очистка + стопслова +
`ё→е`). Любой новый текст **обязан** пройти через тот же `lemmatize_text`, иначе TF-IDF словарь не
совпадёт и качество рухнет. **Эконом тоже работает по этим леммам** (а не по сырому тексту) — поэтому
лемматизацию делают ОДИН раз в topics311, а эконом потом ест готовые `lemmas`. Короткие/пустые тексты
(`len(text) < 20`) не лемматизируются: `text_too_short=True`, тема=0, score=NaN, is_econom=0, econom_class=−1.

---

## 5. Как прогнать на новых данных

### Вариант A — быстрый, для понимания (демо-скрипты, относительные пути)
```bash
cd classifiers/scripts
# темы: parquet с колонкой text → out с lemmas + topic_*/score_*
/opt/anaconda3/envs/topics311/bin/python infer_topics_standalone.py --parquet in.parquet --out tmp_topics.parquet
# эконом: берёт колонку lemmas из предыдущего выхода
/opt/anaconda3/envs/econom_extractor/bin/python infer_econom_standalone.py --parquet tmp_topics.parquet --out tmp_econom.parquet
```
(Демо-скрипты — для небольших файлов и понимания механики.)

### Вариант B — продакшн-пайплайн (как считался весь корпус, с дедупом)
Скрипты `step1..step4` рассчитаны на большой паркет и **дедуп по хэшу текста** (считаем темы/эконом
только на уникальных текстах, потом размазываем обратно). Они написаны под **абсолютные пути**
оригинального дерева (см. §6) — на этой машине запускаются как есть:
```bash
# 1) дедуп + лемматизация → row_keys.parquet (по всем) + unique_lemmas.parquet (уникальные)
/opt/anaconda3/envs/topics311/bin/python step1_prep_lemmatize.py
# 2) 7 тем на unique_lemmas → unique_pred_topics.parquet
/opt/anaconda3/envs/topics311/bin/python step2_topics.py
# 3) эконом на unique_lemmas → unique_pred_econom.parquet
/opt/anaconda3/envs/econom_extractor/bin/python step3_econom.py
# 4) сборка: исходные колонки + темы + эконом по text_hash → final_*_labeled.parquet
/opt/anaconda3/envs/topics311/bin/python step4_assemble.py
```
Чтобы прогнать на ДРУГОМ входе — поменять константы путей вверху каждого step-скрипта
(`INPUT`, `RUN`, `OUT`) и/или указать на копии моделей в этом комплекте.

---

## 6. Схемы вход/выход

**Вход:** колонка `text` (сырой текст поста). Ключ строки — `(channel_id, message_id)`.

**Выход тем (на текст):** `text_hash`, `text_too_short`,
`topic_Защита_прав_потребителей … topic_НДО` (int8, 7 шт), `score_…` (float32, 7 шт).
Пробелы в названиях тем → подчёркивания.

**Выход эконома:** `text_hash`, `econom_class` (int8), `is_econom` (int8), `p_econom` (f32), `p_c0..p_c5` (f32).

**Финальный паркет (`final_100chan_labeled.parquet`)** = все исходные колонки + 7 `topic_*` + 7 `score_*`
+ `econom_class/is_econom/p_econom/p_c0..p_c5` + `text_hash/text_too_short`. На нём считается префильтр:
`union = (≥1 topic) | (is_econom==1)`.

---

## 7. Связь с корпусом BERT (родительская папка)
`corpus_full.parquet` собран так: бралась **полная подневная вырезка** дней-зондов, посты, прошедшие
`тема ∪ эконом`, уходили в LLM-разметку (релевантность 5 тем восприятия ЦБ), не прошедшие — оставались
негативами. То есть **этот префильтр — первый слой всей разметки**. Поэтому recall префильтра важен:
он задаёт потолок полноты позитивов в корпусе (caveat про чистоту негативов — в корневом README).

> Не путать **7 тем тематического классификатора** (рубрикатор: ДКП, Финрынок, НБП, …) и
> **5 тем релевантности `t1..t5`** в корпусе (посылы заседания ЦБ). Это разные разметки;
> тематический классификатор + эконом — лишь дешёвый отбор кандидатов.

---

## 8. Откуда копии (оригиналы на этой машине)
Комплект — самодостаточная копия. Оригиналы (на них же ссылаются абсолютные пути в step-скриптах):

| Что | Оригинал |
|---|---|
| binary SVC mix | `~/Downloads/binary_per_topic_classifier/results/svc_per_topic_pipeline.joblib` |
| binary `src` | `~/Downloads/binary_per_topic_classifier/src/` |
| НБП стэкер | `~/Downloads/svc_focused_v2/results/nbp_v3/nbp_stacker.joblib` |
| svc `src` + nbp_predictor | `~/Downloads/svc_focused_v2/src/`, `~/Downloads/svc_focused_v2/results/nbp_v3/nbp_predictor.py` |
| лемматизатор | `~/Downloads/topic_classifier_production/src/lemmatizer.py` |
| econom-модели | `~/Downloads/russian_news_database/models/class_model/` |
| step/eval скрипты | `~/Downloads/topic_classifier_production/run_100chan/`, `…/run_eval/` |

Демо-скрипты (`infer_*_standalone.py`) используют **относительные** пути и работают из комплекта
без оригиналов. Прод-скрипты (`step*`) используют **абсолютные** пути — правьте константы под себя.

---

## 9. Траблшутинг
- `ModuleNotFoundError: No module named 'src.pipeline'` → нарушен порядок/пути загрузки (Ловушка 1).
  Грузить binary первым, пути в `sys.path` — как в `infer_topics_standalone.py`.
- `AttributeError … ThresholdedBinaryClassifier` при загрузке НБП → НБП загружен раньше binary, либо
  не добавлены пути `svc_focused_v2`. Соблюдать порядок.
- econom-пикл не распаковывается → не тот env (нужен sklearn **1.0.2**, env `econom_extractor`).
- XGBoost WARNING про старый бустер → игнорировать, это норма.
- Темы все нулевые / мусорные скоры → текст не лемматизирован тем самым `lemmatizer.py` (Ловушка 2).
- pymorphy3 при первом запуске может тянуть словари / NLTK-стопслова — нужен один раз интернет.
```
