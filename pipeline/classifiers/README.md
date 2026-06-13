# Prefilter-классификаторы

Папка содержит самодостаточный комплект дешёвых классификаторов, которыми отбирались посты-кандидаты перед BERT/LLM-разметкой.

## Основные файлы

| Путь | Назначение |
|---|---|
| `INSTRUCTIONS.md` | подробная инструкция по комплекту |
| `requirements-topics311.txt` | окружение для лемматизации и тематического prefilter-а |
| `requirements-econom_extractor.txt` | окружение для экономического XGBoost-классификатора |
| `scripts/run_prefilter_pipeline.py` | wrapper полного prefilter-пайплайна |
| `scripts/infer_topics_standalone.py` | демо тематического prefilter-а |
| `scripts/infer_econom_standalone.py` | демо econom-классификатора |
| `topics/` | модели и код тематического классификатора |
| `econom/` | модели экономического классификатора |

## Входы

Parquet с колонкой `text`. Для полного пайплайна также полезны `channel_id`, `message_id`, `message_type`.

## Выходы

Parquet с prefilter-колонками: `topic_*`, `score_*`, `econom_class`, `is_econom`, `p_econom`, `prefilter_union`.

## Как запускать

```powershell
python pipeline\classifiers\scripts\run_prefilter_pipeline.py `
  --input pipeline\corpus_test.parquet `
  --output pipeline\data\prefilter\bert_candidates.parquet `
  --report pipeline\data\prefilter\prefilter_report.json `
  --overwrite
```

## Примечания

Важно: старые joblib/pickle-артефакты чувствительны к версиям `scikit-learn`, `xgboost` и `catboost`. Для точного воспроизведения используйте отдельные requirements-файлы в этой папке.

