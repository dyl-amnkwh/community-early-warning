"""Early-warning test: which schools NOT currently in the top quarter of chronic
absenteeism will be in it next year?

Rolling-origin evaluation: for each test year, train only on earlier target years
(from 2023 on, skipping the pandemic-affected 2022 rows), then score that year.

Usage: python -m ewp.early_test
Writes: reports/early_warning_test.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ewp.metrics import recall_at_capacity
from ewp.run_models import FEATURES, PANEL, TARGET, fit_score

TEST_YEARS = [2024, 2025]
FIRST_TRAIN = 2023
CAPS = (0.10, 0.20, 0.30)
BASE = "baseline: this year's rate"
MODELS = [BASE] + [f"{k} {n}" for k in ("logreg", "forest") for n in FEATURES]
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def load():
    df = pd.read_csv(PANEL, dtype={"school_code": str})
    df["is_dcps"] = (df["sector"] == "dcps").astype(float).where(df["sector"].notna())
    rank = df.groupby("school_year")["absent_pct"].rank(pct=True)
    df["now_top"] = rank > 0.75
    return df


def score_all(df):
    parts = []
    for ty in TEST_YEARS:
        has_y = df[TARGET].notna()
        train = df[has_y & (df["target_year"] >= FIRST_TRAIN) & (df["target_year"] < ty)]
        test = df[has_y & (df["target_year"] == ty) & df["absent_pct"].notna() & ~df["now_top"]].copy()
        test["test_year"] = ty
        test[BASE] = test["absent_pct"] / 100
        for kind in ("logreg", "forest"):
            for name in FEATURES:
                test[f"{kind} {name}"] = fit_score(name, kind, train, test)
        parts.append(test)
    return pd.concat(parts, ignore_index=True)


def recall(y, s, frac):
    return recall_at_capacity(y, s, frac)


if __name__ == "__main__":
    res = score_all(load())
    y = res[TARGET].to_numpy()
    emit("Schools NOT in the top quarter this year; will they be in it next year?\n")
    for ty, g in res.groupby("test_year"):
        emit(f"  test year {ty}: {len(g)} schools, {int(g[TARGET].sum())} moved into the top quarter")
    emit(f"  pooled: {len(res)} school-years, {int(y.sum())} moved in "
         f"(chance level for PR-AUC = base rate {y.mean():.2f})\n")

    ranks = {m: res.groupby("test_year")[m].rank(pct=True).to_numpy() for m in MODELS}
    rows = []
    for m in MODELS:
        row = {"model": m, "pr_auc": round(average_precision_score(y, ranks[m]), 3)}
        for c in CAPS:
            row[f"recall@{round(c * 100)}%"] = round(recall(y, ranks[m], c), 3)
        rows.append(row)
    emit(pd.DataFrame(rows).to_string(index=False))

    emit("\nDoes neighborhood data help? (C minus A; 95% interval, resampling schools)")
    codes = res["school_code"].to_numpy()
    groups = {k: np.flatnonzero(codes == k) for k in np.unique(codes)}
    keys = list(groups)
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(500)]
    stats = {"pr_auc": average_precision_score, "recall@20%": lambda a, b: recall(a, b, 0.20)}
    for kind in ("logreg", "forest"):
        a, c = ranks[f"{kind} A_attendance"], ranks[f"{kind} C_plus_neighborhood"]
        for stat, fn in stats.items():
            diff = fn(y, c) - fn(y, a)
            boot = [fn(y[i], c[i]) - fn(y[i], a[i]) for i in draws]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            emit(f"  {kind:7s} {stat:10s} {diff:+.3f}  (interval {lo:+.3f} to {hi:+.3f})")

    emit("\nSmall numbers of positives mean wide uncertainty. Intervals that include 0 could be chance.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/early_warning_test.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/early_warning_test.txt")
