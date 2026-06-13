"""
Лемматизатор русских текстов — точная копия пайплайна из исследования.

Источник: svc_baseline/preprocess.py из оригинального исследовательского репозитория.
Дублирован сюда специально, чтобы production-пакет был самодостаточным.

Что делает:
    raw text → очистка → токенизация → стопслова → pymorphy3 lemma → итоговая строка

Это та же самая обработка, через которую прошли train/test данные при обучении модели.
**Любой новый текст ОБЯЗАТЕЛЬНО должен пройти через эту же лемматизацию,**
иначе TF-IDF features будут не совпадать со словарём векторайзера и качество резко упадёт.

Скорость: ~500-1000 постов/секунду на современном CPU (зависит от длины текста).
RAM: ~500 МБ (pymorphy3 загружает словари OpenCorpora один раз).
"""

from __future__ import annotations

import re
import time
from typing import Iterable, List

import pymorphy3

# ============================================================================
# Глобальный анализатор pymorphy3.
# Создаётся лениво — он тяжёлый, инициализация занимает 2-3 секунды.
# Не пересоздавать на каждом тексте!
# ============================================================================
_MORPH: pymorphy3.MorphAnalyzer | None = None


def get_morph() -> pymorphy3.MorphAnalyzer:
    """Возвращает singleton pymorphy3-анализатора. Lazy init."""
    global _MORPH
    if _MORPH is None:
        _MORPH = pymorphy3.MorphAnalyzer()
    return _MORPH


# ============================================================================
# Стопслова: NLTK Russian + кастомный список Telegram/новостного шума.
# Это те же стопслова, что использовались при обучении. НЕ МЕНЯТЬ список.
# ============================================================================
def get_stopwords() -> set[str]:
    """Базовые стопслова: NLTK ru + домен-специфичные шумовые слова.

    NLTK ('russian') — стандартный список из ~150 русских предлогов/местоимений
    + наша добавка из ~20 слов про мессенджеры и новости.
    """
    try:
        from nltk.corpus import stopwords
        sw = set(stopwords.words("russian"))
    except LookupError:
        # Если стоп-слов нет — скачиваем (требует сетевого доступа на первом запуске).
        import nltk
        nltk.download("stopwords", quiet=True)
        from nltk.corpus import stopwords
        sw = set(stopwords.words("russian"))

    # Доп. стопслова для Telegram/новостей.
    # Эти слова часто встречаются и не несут темы.
    sw |= {
        "это", "весь", "также", "тот", "такой", "свой", "наш", "мочь",
        "год", "день", "месяц", "сегодня", "вчера", "завтра", "неделя",
        "сообщить", "сообщать", "заявить", "заявлять", "пишет", "сообщение",
        "telegram", "тг", "канал", "пост", "фото", "видео",
    }
    return sw


# ============================================================================
# Регулярки для очистки текста до лемматизации.
# Точно такие же, как в оригинале (svc_baseline/preprocess.py).
# ============================================================================

# URL-ы (https://..., www.... и t.me/...)
_RE_URL = re.compile(r"https?://\S+|www\.\S+|t\.me/\S+", re.IGNORECASE)
# @username
_RE_MENTION = re.compile(r"@[A-Za-z0-9_]+")
# #hashtag (любые буквы Unicode)
_RE_HASHTAG = re.compile(r"#\w+", re.UNICODE)
# HTML-теги <...>
_RE_HTML = re.compile(r"<[^>]+>")
# Всё, кроме кириллицы / латиницы / цифр / пробелов / дефиса
_RE_NONLETTER = re.compile(r"[^а-яёa-z0-9\s\-]+", re.IGNORECASE)
# Множественные пробелы
_RE_SPACES = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Минимальная нормализация перед лемматизацией.

    Шаги:
        1. lowercase
        2. удалить URL, @mention, #hashtag, HTML-теги
        3. ё → е (важно: pymorphy3 не нормализует ё→е сам)
        4. удалить все символы кроме букв/цифр/пробелов/дефиса
        5. сжать множественные пробелы
    """
    if not isinstance(text, str):
        return ""
    t = text.lower()
    t = _RE_URL.sub(" ", t)
    t = _RE_MENTION.sub(" ", t)
    t = _RE_HASHTAG.sub(" ", t)
    t = _RE_HTML.sub(" ", t)
    t = t.replace("ё", "е")
    t = _RE_NONLETTER.sub(" ", t)
    t = _RE_SPACES.sub(" ", t).strip()
    return t


def lemmatize_text(
    text: str,
    morph: pymorphy3.MorphAnalyzer | None = None,
    stopwords: set[str] | None = None,
) -> str:
    """Очистка + лемматизация одного текста. Возвращает строку из лемм через пробел.

    Если morph и stopwords не переданы — возьмём singletons (но это медленно при
    bulk-обработке: каждый раз будет проверка `if _MORPH is None`).
    Для bulk предпочтительно передавать morph/stopwords напрямую.

    Логика фильтрации токенов:
        - длина < 2 → drop
        - токен — чистая цифра → drop
        - токен в стопсловах → drop
        - после лемматизации: лемма в стопсловах ИЛИ длина < 2 → drop

    Двойная проверка стопслов (до и после лемматизации) важна потому, что
    pymorphy может привести слово к стоп-форме (например, «делал» → «делать»).
    """
    if morph is None:
        morph = get_morph()
    if stopwords is None:
        stopwords = get_stopwords()

    cleaned = clean_text(text)
    if not cleaned:
        return ""

    out: List[str] = []
    for tok in cleaned.split():
        if len(tok) < 2:
            continue
        if tok.isdigit():
            continue
        if tok in stopwords:
            continue
        # parse() возвращает список разборов; берём наиболее вероятный (индекс 0).
        # Это та же эвристика, что и в оригинале — без выбора по части речи.
        lemma = morph.parse(tok)[0].normal_form
        if lemma in stopwords or len(lemma) < 2:
            continue
        out.append(lemma)
    return " ".join(out)


def lemmatize_batch(
    texts: Iterable[str],
    verbose: bool = True,
    progress_every: int = 500,
) -> List[str]:
    """Bulk-лемматизация. Возвращает список строк той же длины, что и вход.

    Args:
        texts: iterable строк (raw text).
        verbose: печатать прогресс каждые progress_every постов.
        progress_every: размер шага прогресс-бара.

    Returns:
        list[str] — список лемматизированных строк.

    Example:
        >>> texts = ["Минфин разместил облигации на 500 млрд",
        ...          "ЦБ повысил ключевую ставку до 16%"]
        >>> lemmatize_batch(texts)
        ['минфин разместить облигация млрд', 'цб повысить ключевой ставка']
    """
    morph = get_morph()
    stopwords = get_stopwords()
    texts_list = list(texts)
    n = len(texts_list)

    out: List[str] = []
    t0 = time.time()
    for i, txt in enumerate(texts_list):
        out.append(lemmatize_text(txt, morph, stopwords))
        if verbose and ((i + 1) % progress_every == 0 or (i + 1) == n):
            elapsed = time.time() - t0
            rate = (i + 1) / max(elapsed, 0.001)
            print(
                f"  [{i+1}/{n}] elapsed={elapsed:.1f}s, rate={rate:.0f} posts/sec",
                flush=True,
            )
    return out


# ============================================================================
# CLI: можно запустить как `python -m src.lemmatizer some_file.csv`
# для лемматизации файла с колонкой 'text'.
# ============================================================================
if __name__ == "__main__":
    import sys
    import pandas as pd

    if len(sys.argv) < 2:
        print(
            "Usage: python -m src.lemmatizer <input.csv|input.parquet> "
            "[text_col=text] [output_path=stdout]"
        )
        sys.exit(1)

    inp = sys.argv[1]
    text_col = sys.argv[2] if len(sys.argv) > 2 else "text"
    out_path = sys.argv[3] if len(sys.argv) > 3 else None

    if inp.endswith(".parquet"):
        df = pd.read_parquet(inp)
    else:
        df = pd.read_csv(inp)
    print(f"Loaded {len(df)} rows from {inp}, lemmatizing column '{text_col}'...")

    df["lemmas"] = lemmatize_batch(df[text_col].astype(str).tolist())

    if out_path:
        if out_path.endswith(".parquet"):
            df.to_parquet(out_path, index=False)
        else:
            df.to_csv(out_path, index=False)
        print(f"Saved to {out_path}")
    else:
        print(df[[text_col, "lemmas"]].head(5))
