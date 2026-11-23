"""Does neighborhood data predict absenteeism for schools in tracts the model never saw?

A school's tract features never change, so a forest can memorize them as a school ID
and recall that school's past rate. Holding out whole tracts removes that shortcut.

Usage: python -m ewp.grouped_check
Writes: reports/grouped_check.txt
"""
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from ewp.run_models import NEIGH, PANEL

MODELS = {
    "ridge (linear)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                    Ridge(alpha=10.0)),
    "forest": make_pipeline(SimpleImputer(strategy="median"),
                            RandomForestRegressor(n_estimators=300, min_samples_leaf=5,
                                                  random_state=42, n_jobs=-1)),
}
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


if __name__ == "__main__":
    df = pd.read_csv(PANEL, dtype={"school_code": str, "geoid": str})
    keep = df["school_year"].isin([2023, 2024, 2025]) & df["absent_pct"].notna() & df["geoid"].notna()
    df = df[keep].reset_index(drop=True)
    X, y, groups = df[NEIGH], df["absent_pct"], df["geoid"]
    emit(f"{len(df)} school-years in {df['geoid'].nunique()} tracts (2023-2025)")
    emit("R^2 for predicting a school's rate from its tract's indicators alone\n")

    rows = []
    for name, model in MODELS.items():
        preds = {
            "random split": cross_val_predict(model, X, y, cv=KFold(5, shuffle=True, random_state=0)),
            "held-out tracts": cross_val_predict(model, X, y, groups=groups, cv=GroupKFold(n_splits=5)),
        }
        for label, pred in preds.items():
            row = {"model": name, "split": label, "R2_all": round(r2_score(y, pred), 2)}
            for sec in ("dcps", "charter"):
                m = (df["sector"] == sec).to_numpy()
                row[f"R2_{sec}"] = round(r2_score(y[m], pred[m]), 2)
            rows.append(row)
    emit(pd.DataFrame(rows).to_string(index=False))
    emit("\nIf R^2 drops a lot when whole tracts are held out, the forest was recalling "
         "individual schools, not learning a neighborhood pattern.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/grouped_check.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/grouped_check.txt")
