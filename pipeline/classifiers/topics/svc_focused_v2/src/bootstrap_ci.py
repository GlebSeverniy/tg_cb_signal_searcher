"""
Bootstrap CI for the final mixture's macro F1 and per-topic F1.
Run: python -m src.bootstrap_ci
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from .data_loader import load_test_binary
from . import ALLOWED_TOPICS

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"


def main(exp_id: str = "final_mixture_v2_cv", n_boot: int = 1000, seed: int = 42):
    attempts_dir = PROJECT_ROOT / "results" / "attempts" / exp_id
    preds_path = attempts_dir / "test_predictions.json"
    if not preds_path.exists():
        raise SystemExit(f"No test_predictions.json in {attempts_dir} — run mixture first.")

    preds = json.loads(preds_path.read_text())
    rng = np.random.default_rng(seed)

    topics = list(ALLOWED_TOPICS)
    topic_arrays = {}
    n = None
    for t in topics:
        if t not in preds:
            continue
        y_pred = np.asarray(preds[t])
        _, y_true_series = load_test_binary(t)
        y_true = y_true_series.values.astype(int)
        if len(y_true) != len(y_pred):
            raise SystemExit(f"Length mismatch for {t}: y_true={len(y_true)} y_pred={len(y_pred)}")
        n = len(y_true) if n is None else n
        topic_arrays[t] = (y_true, y_pred)

    # Bootstrap (shared sample indices across topics — proper joint CI)
    macro_samples = []
    per_topic_samples = {t: [] for t in topic_arrays}

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        f1s = []
        for t, (yt, yp) in topic_arrays.items():
            f = f1_score(yt[idx], yp[idx], pos_label=1, zero_division=0)
            f1s.append(f)
            per_topic_samples[t].append(f)
        macro_samples.append(float(np.mean(f1s)))

    def ci(arr, q=(2.5, 97.5)):
        return [float(np.percentile(arr, q[0])), float(np.percentile(arr, q[1]))]

    out = {
        "exp_id": exp_id,
        "n_boot": n_boot,
        "seed": seed,
        "macro_f1_mean": float(np.mean(macro_samples)),
        "macro_f1_ci95": ci(macro_samples),
        "macro_f1_std": float(np.std(macro_samples)),
        "per_topic_ci95": {t: ci(per_topic_samples[t]) for t in topic_arrays},
        "per_topic_mean": {t: float(np.mean(per_topic_samples[t])) for t in topic_arrays},
    }

    out_path = RESULTS_DIR / "bootstrap_ci.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False))
    print(f"Bootstrap CI written to {out_path}")
    print(f"Macro F1: {out['macro_f1_mean']:.4f} [{out['macro_f1_ci95'][0]:.4f}, {out['macro_f1_ci95'][1]:.4f}] @ 95%")
    return out


if __name__ == "__main__":
    main()
