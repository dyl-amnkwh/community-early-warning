"""XGBoost vs. simpler models for the early-warning target, plus SHAP explanations.

Same test as early_test.py (schools NOT yet in the top quarter; rolling origin on
2024 and 2025), with gradient-boosted trees added. SHAP then shows which inputs the
boosted model leans on. SHAP describes the model, not cause and effect.

Usage: python -m ewp.xgb_shap
Writes: reports/xgb_shap.txt, reports/figures/shap_importance.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import average_precision_score
from xgboost import XGBClassifier

from ewp.early_test import BASE, CAPS, FIRST_TRAIN, TEST_YEARS, load, recall, score_all
from ewp.run_models import FEATURES, TARGET

PARAMS = dict(n_estimators=200, max_depth=2, learning_rate=0.05, subsample=0.8,
              colsample_bytree=0.8, min_child_weight=5, reg_lambda=5.0,
              eval_metric="logloss", random_state=42, n_jobs=1)
COLS = {"A": FEATURES["A_attendance"], "C": FEATURES["C_plus_neighborhood"]}
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def train_rows(df, test_year):
    return df[df[TARGET].notna() & (df["target_year"] >= FIRST_TRAIN) & (df["target_year"] < test_year)]


def fit_xgb(cols, train):
    model = XGBClassifier(**PARAMS)
    model.fit(train[cols], train[TARGET].astype(int))
    return model


if __name__ == "__main__":
    df = load()
    res = score_all(df)
    for key, cols in COLS.items():
        res[f"xgboost {key}"] = np.nan
        for ty in TEST_YEARS:
            m = res["test_year"] == ty
            model = fit_xgb(cols, train_rows(df, ty))
            res.loc[m, f"xgboost {key}"] = model.predict_proba(res.loc[m, cols])[:, 1]

    y = res[TARGET].to_numpy()
    names = [BASE, "logreg A_attendance", "logreg C_plus_neighborhood", "forest A_attendance",
             "forest C_plus_neighborhood", "xgboost A", "xgboost C"]
    ranks = {m: res.groupby("test_year")[m].rank(pct=True).to_numpy() for m in names}
    emit("Schools NOT in the top quarter this year; will they be in it next year?")
    emit(f"Pooled test years {TEST_YEARS}: {len(res)} school-years, {int(y.sum())} moved in "
         f"(chance PR-AUC = {y.mean():.2f})\n")
    rows = []
    for m in names:
        row = {"model": m, "pr_auc": round(average_precision_score(y,
ranks[m]), 3)}
        for c in CAPS:
            row[f"recall@{round(c * 100)}%"] = round(recall(y, ranks[m], c), 3)
        rows.append(row)
    emit(pd.DataFrame(rows).to_string(index=False))

    emit("\nXGBoost: does neighborhood data help? (C minus A; 95% interval, resampling schools)")
    codes = res["school_code"].to_numpy()
    groups = {k: np.flatnonzero(codes == k) for k in np.unique(codes)}
    keys = list(groups)
    rng = np.random.default_rng(0)
    draws = [np.concatenate([groups[k] for k in rng.choice(keys, len(keys))]) for _ in range(500)]
    a, c = ranks["xgboost A"], ranks["xgboost C"]
    for stat, fn in (("pr_auc", average_precision_score), ("recall@20%", lambda u, v: recall(u, v, 0.20))):
        diff = fn(y, c) - fn(y, a)
        boot = [fn(y[i], c[i]) - fn(y[i], a[i]) for i in draws]
        lo, hi = np.percentile(boot, [2.5, 97.5])
        emit(f"  {stat:10s} {diff:+.3f}  (interval {lo:+.3f} to {hi:+.3f})")

    last = max(TEST_YEARS)
    model = fit_xgb(COLS["C"], train_rows(df, last))
    X = res.loc[res["test_year"] == last, COLS["C"]]
    sv = shap.TreeExplainer(model).shap_values(X)
    imp = pd.DataFrame({
        "feature": COLS["C"],
        "mean_abs_shap": np.abs(sv).mean(axis=0),
        "higher_value_pushes": ["risk up" if np.corrcoef(X[f].fillna(X[f].median()), sv[:, j])[0, 1] > 0
                                else "risk down" for j, f in enumerate(COLS["C"])],
    }).sort_values("mean_abs_shap", ascending=False)
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(3)
    emit(f"\nSHAP for the XGBoost model with all inputs, trained before {last}, explaining {len(X)} "
         f"{last} schools\n(mean absolute contribution, in log-odds; direction = how higher values move the score)")
    emit(imp.to_string(index=False))

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    top = imp.iloc[::-1]
    ax.barh(top["feature"], top["mean_abs_shap"], color="#5D3A9B")
    ax.set_xlabel("Mean absolute SHAP value (log-odds)")
    ax.set_title(f"What the XGBoost early-warning model leans on ({last})")
    fig.tight_layout()
    fig.savefig("reports/figures/shap_importance.png", dpi=150)
    plt.close(fig)

    emit("\nSHAP explains the model, not causes. Neighborhood values are the same for every school in a "
         "tract, so a flexible model can use them to recognize schools (see grouped_check). "
         "Small number of positives: treat differences as tentative.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/xgb_shap.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/xgb_shap.txt and reports/figures/shap_importance.png")
