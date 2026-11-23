"""Compare nested early-warning models on DC schools.

Target: will a school be in the top quarter of chronic absenteeism next year?
Train on earlier target years; test on 2025, which the models never see.

Usage: python -m ewp.run_models
Writes: reports/model_comparison.txt
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from ewp.metrics import evaluate, recall_at_capacity
from ewp.models import build_logreg, build_rf

PANEL = Path("data/processed/dc_panel.csv")
TARGET, TEST_YEAR = "next_top_quarter", 2025
CAPS = (0.10, 0.20, 0.25)

ATTEND = ["absent_pct", "prior_pct", "change_from_prior"]
NEIGH = ["poverty_rate", "median_hh_income", "rent_burden",
         "no_vehicle_hh", "no_internet_hh", "limited_english_hh"]
FEATURES = {
    "N_neighborhood_only": NEIGH,
    "A_attendance": ATTEND,
    "B_plus_school_type": ATTEND + ["is_dcps"],
    "C_plus_neighborhood": ATTEND + ["is_dcps"] + NEIGH,
}
STATS = {
    "pr_auc": average_precision_score,
    "recall@20%": lambda y, s: recall_at_capacity(y, s, 0.20),
}
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def load():
    df = pd.read_csv(PANEL, dtype={"school_code": str})
    df["is_dcps"] = (df["sector"] == "dcps").astype(float).where(df["sector"].notna())
    return df[df[TARGET].notna()].copy()


def fit_score(name, kind, train, test):
    cols = FEATURES[name]
    model = build_logreg() if kind == "logreg" else build_rf()
    model.fit(train[cols], train[TARGET])
    return model.predict_proba(test[cols])[:, 1]


def run(df, train_years):
    train = df[df["target_year"].isin(train_years)]
    test = df[df["target_year"] == TEST_YEAR]
    y = test[TARGET].to_numpy()
    emit(f"\n=== Train on target years {list(train_years)} ({len(train)} rows) | "
         f"test on {TEST_YEAR} ({len(test)} schools, {int(y.sum())} in the top quarter) ===")
    emit(f"Chance level for PR-AUC is the base rate: {y.mean():.2f}\n")

    rate = test["absent_pct"].fillna(test["absent_pct"].median())
    scores = {"baseline: this year's rate": (rate / 100).clip(0, 1).to_numpy()}
    for kind in ("logreg", "forest"):
        for name in FEATURES:
            scores[f"{kind} {name}"] = fit_score(name, kind, train, test)

    rows = []
    for label, s in scores.items():
        r = evaluate(y, s, CAPS)
        rows.append({"model": label, **{k: round(v, 3) for k, v in r.items()
                                        if k not in ("brier", "base_rate")}})
    emit(pd.DataFrame(rows).to_string(index=False))

    emit("\nDoes adding neighborhood data help? (C minus A; 95% bootstrap interval over schools)")
    rng = np.random.default_rng(0)
    draws = [rng.integers(0, len(y), len(y)) for _ in range(500)]
    for kind in ("logreg", "forest"):
        a, c = scores[f"{kind} A_attendance"], scores[f"{kind} C_plus_neighborhood"]
        for stat, fn in STATS.items():
            diff = fn(y, c) - fn(y, a)
            boot = [fn(y[i], c[i]) - fn(y[i], a[i]) for i in draws]
            lo, hi = np.percentile(boot, [2.5, 97.5])
            emit(f"  {kind:7s} {stat:10s} {diff:+.3f}  (interval {lo:+.3f} to {hi:+.3f})")


if __name__ == "__main__":
    data = load()
    run(data, [2023, 2024])
    run(data, [2022, 2023, 2024])
    emit("\nIf an interval includes 0, the difference could be chance.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/model_comparison.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/model_comparison.txt")
