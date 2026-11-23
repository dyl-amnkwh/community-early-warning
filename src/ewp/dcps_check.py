"""DCPS-only check: does neighborhood data help predict next year's top-quarter status?

Charter schools draw students citywide, so a charter's tract says little about its
students. Here only DCPS (neighborhood-based) schools are used for training and testing.
Only logistic regression is used: unlike a forest, it cannot memorize school fingerprints.

Usage: python -m ewp.dcps_check
Writes: reports/dcps_check.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ewp.early_test import FIRST_TRAIN, TEST_YEARS, load
from ewp.metrics import recall_at_capacity
from ewp.run_models import FEATURES, TARGET, fit_score

CAPS = (0.10, 0.20, 0.30)
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def run(df, only_unflagged):
    parts = []
    for ty in TEST_YEARS:
        has_y = df[TARGET].notna()
        train = df[has_y & (df["target_year"] >= FIRST_TRAIN) & (df["target_year"] < ty)]
        test = df[has_y & (df["target_year"] == ty) & df["absent_pct"].notna()].copy()
        if only_unflagged:
            test = test[~test["now_top"]]
        test["test_year"] = ty
        test["baseline"] = test["absent_pct"] / 100
        for name in FEATURES:
            test[f"logreg {name}"] = fit_score(name, "logreg", train, test)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def report(res, title):
    y = res[TARGET].to_numpy()
    emit(f"\n=== {title} ===")
    for ty, g in res.groupby("test_year"):
        emit(f"  test year {ty}: {len(g)} schools, {int(g[TARGET].sum())} in the top quarter next year")
    emit(f"  pooled: {len(res)} school-years, {int(y.sum())} positives (base rate {y.mean():.2f})\n")
    models = ["baseline"] + [f"logreg {n}" for n in FEATURES]
    ranks = {m: res.groupby("test_year")[m].rank(pct=True).to_numpy() for m in models}
    rows = []
    for m in models:
        row = {"model": m, "pr_auc": round(average_precision_score(y, ranks[m]), 3)}
        for c in CAPS:
            row[f"recall@{round(c * 100)}%"] = round(recall_at_capacity(y, ranks[m], c), 3)
        rows.append(row)
    emit(pd.DataFrame(rows).to_string(index=False))

    codes = res["school_code"].to_numpy()
    groups = {k: np.flatnonzero(codes == k) for k in np.unique(codes)}
    keys = list(groups)
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(500)]
    a, c = ranks["logreg A_attendance"], ranks["logreg C_plus_neighborhood"]
    stats = {"pr_auc": average_precision_score,
             "recall@20%": lambda u, v: recall_at_capacity(u, v, 0.20)}
    emit("\nC minus A (neighborhood added to attendance); 95% interval, resampling schools")
    for stat, fn in stats.items():
        diff = fn(y, c) - fn(y, a)
        boot = [fn(y[i], c[i]) - fn(y[i], a[i]) for i in draws]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        emit(f"  {stat:10s} {diff:+.3f}  (interval {lo:+.3f} to {hi:+.3f})")


if __name__ == "__main__":
    df = load()
    df = df[df["sector"] == "dcps"]
    report(run(df, False), "DCPS schools: all schools scored")
    report(run(df, True), "DCPS schools not yet in the top quarter (small numbers)")
    emit("\nIf an interval includes 0, the difference could be chance.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/dcps_check.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/dcps_check.txt")
