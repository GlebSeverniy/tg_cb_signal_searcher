"""eval analysis: recall дешёвого префильтра (тема ∪ эконом) на LLM-эталоне.
usage: eval_analyze.py <annotated.parquet> <prefix> <n_themes>
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd

TOPICS = ["Защита прав потребителей","ДКП","Финансовый рынок","НБП","Финтех","Геополитика","НДО"]
TC = ["topic_"+t.replace(" ","_") for t in TOPICS]


def main():
    inp, pref, nth = sys.argv[1], sys.argv[2], int(sys.argv[3])
    rel_cols = [f"t{i}_relevant" for i in range(1, nth+1)]
    ann = pd.read_parquet(inp, columns=rel_cols)
    tp = pd.read_parquet(f"{pref}_topics.parquet").set_index("row_idx").sort_index()
    ec = pd.read_parquet(f"{pref}_econom.parquet").set_index("row_idx").sort_index()
    n = len(ann)
    assert len(tp)==n and len(ec)==n, "row count mismatch"

    R = ann[rel_cols].fillna(False).astype(bool).any(axis=1).values      # эталон: ≥1 тема relevant
    theme = (tp[TC]==1).any(axis=1).values                                # наш: ≥1 из 7 тем
    econ  = (ec["is_econom"]==1).values                                   # наш: экономический
    union = theme | econ                                                  # «отсортированный»

    Rn = int(R.sum())
    print("="*66)
    print(f"{inp.split('/')[-1]}   N={n:,}   эталонно-релевантных (≥1 тема LLM): {Rn:,} ({100*Rn/n:.1f}%)")
    print("-"*66)
    # объём префильтра
    for name, mask in [("только темы", theme), ("только эконом", econ), ("тема ∪ эконом", union)]:
        keep = int(mask.sum())
        capt = int((mask & R).sum())
        recall = 100*capt/Rn
        prec = 100*capt/keep if keep else 0
        print(f"  {name:<16s}: оставляет {keep:>5,} ({100*keep/n:4.1f}% массива) | "
              f"ловит {capt:>4,}/{Rn} релев. | RECALL {recall:5.1f}% | точность {prec:4.1f}%")
    lost = int((R & ~union).sum())
    print("-"*66)
    print(f"  ИТОГ префильтра «тема ∪ эконом»: теряем {lost} релевантных из {Rn} "
          f"({100*lost/Rn:.1f}%), экономим {100*(1-union.mean()):.1f}% объёма LLM")

    # вклад: сколько релевантных ловится ТОЛЬКО темами, ТОЛЬКО экономом
    only_theme = int((R & theme & ~econ).sum())
    only_econ  = int((R & econ & ~theme).sum())
    both       = int((R & theme & econ).sum())
    print(f"  из пойманных релевантных: и тема и эконом={both}, только тема={only_theme}, только эконом={only_econ}")

    # recall по каждой LLM-теме (какие посылы теряются префильтром)
    print("  recall по LLM-темам:")
    for rc in rel_cols:
        rmask = ann[rc].fillna(False).astype(bool).values
        if rmask.sum()==0: continue
        cap = int((rmask & union).sum())
        print(f"    {rc}: {cap}/{int(rmask.sum())} ({100*cap/rmask.sum():.0f}%)")


if __name__ == "__main__":
    main()
