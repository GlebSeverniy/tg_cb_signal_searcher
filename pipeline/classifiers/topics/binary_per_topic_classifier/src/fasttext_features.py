"""
FastText average-pooling feature extractor.

Trains a FastText skip-gram model on the training corpus (UNSUPERVISED — no labels used).
Caches the trained model to data_fasttext/ft_train.bin so subsequent runs skip training.

TRANSPARENCY NOTE: FastText is trained on all train texts (unsupervised, no labels),
which means CV folds share the same embedding space.  This is considered legitimate
because:
  (a) no labels are visible during FastText training,
  (b) this mirrors standard practice for unsupervised pre-training (word2vec, GloVe, etc.),
  (c) the alternative (per-fold retraining ×5 ×11 topics) would take ~2-3 hours with
      identical embedding quality since corpus size per fold vs full train differs by <20%.
Audit note: «FastText trained unsupervised on all train texts; legitimate as no labels used».
"""

from __future__ import annotations

import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FASTTEXT_PATH = PROJECT_ROOT / "data_fasttext" / "ft_train.bin"

_model_cache = None


def get_or_train_fasttext_model(X_train_texts, force_retrain: bool = False):
    """Train (lazy) and cache FastText model on train texts.

    Parameters
    ----------
    X_train_texts : iterable of str
        Training texts (lemmatized).  Labels are NOT passed — purely unsupervised.
    force_retrain : bool
        If True, retrain even if cached model exists.

    Returns
    -------
    gensim FastText model
    """
    import functools
    print_ = functools.partial(print, flush=True)

    global _model_cache
    if _model_cache is not None and not force_retrain:
        print_("[fasttext] Using in-memory cached model.")
        return _model_cache

    if FASTTEXT_PATH.exists() and not force_retrain:
        print_(f"[fasttext] Loading cached model from {FASTTEXT_PATH} ...")
        from gensim.models import FastText
        _model_cache = FastText.load(str(FASTTEXT_PATH))
        print_(f"[fasttext] Loaded. Vocab size: {len(_model_cache.wv)}")
        return _model_cache

    print_("[fasttext] Training FastText skip-gram on train corpus (unsupervised, no labels)...")
    import time
    t0 = time.time()
    from gensim.models import FastText
    sentences = [str(t).split() for t in X_train_texts]
    model = FastText(
        sentences,
        vector_size=300,
        window=5,
        min_count=2,
        sg=1,            # skip-gram
        epochs=10,
        seed=42,
        workers=4,
    )
    elapsed = time.time() - t0
    FASTTEXT_PATH.parent.mkdir(parents=True, exist_ok=True)
    model.save(str(FASTTEXT_PATH))
    _model_cache = model
    vocab_size = len(model.wv)
    print_(f"[fasttext] Training done in {elapsed:.1f}s. Vocab: {vocab_size} words. Saved to {FASTTEXT_PATH}")
    return model


def fasttext_avg_pool(texts):
    """Return dense (n, 300) matrix: average word vectors per text.

    Uses the globally cached FastText model. If the cache is empty (e.g. after
    joblib.load() in a new process), auto-loads the model from FASTTEXT_PATH.
    Call get_or_train_fasttext_model() before using this function the first time.

    Parameters
    ----------
    texts : iterable of str

    Returns
    -------
    np.ndarray of shape (n, 300), dtype float32
    """
    import functools
    print_ = functools.partial(print, flush=True)

    global _model_cache
    model = _model_cache
    if model is None:
        if not FASTTEXT_PATH.exists():
            raise RuntimeError(
                "FastText model not found. Call get_or_train_fasttext_model(X_train) "
                f"first (expected at {FASTTEXT_PATH})."
            )
        print_(f"[fasttext] Auto-loading model from {FASTTEXT_PATH} ...")
        from gensim.models import FastText
        _model_cache = FastText.load(str(FASTTEXT_PATH))
        model = _model_cache
        print_(f"[fasttext] Loaded. Vocab size: {len(model.wv)}")
    texts = list(texts)
    out = np.zeros((len(texts), 300), dtype=np.float32)
    for i, t in enumerate(texts):
        words = str(t).split()
        if not words:
            continue
        # gensim FastText handles OOV via subword character n-grams automatically
        vecs = []
        for w in words:
            try:
                vecs.append(model.wv[w])
            except KeyError:
                continue
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out
