"""Which neighborhood factors move with school absenteeism, and where do schools sit?

Usage: python -m ewp.associations
Writes: reports/associations.txt, reports/figures/school_map_2025.png,
        reports/figures/poverty_vs_absenteeism_2025.png
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.pipeline import make_pipeline

from ewp.run_models import NEIGH, PANEL

YEAR = 2025
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def spearman_table(d):
    rows = []
    for col in NEIGH:
        row = {"indicator": col}
        for label, g in (("all", d), ("dcps", d[d["sector"] == "dcps"]),
                         ("charter", d[d["sector"] == "charter"])):
            sub = g[[col, "absent_pct"]].dropna()
            row[f"rho_{label}"] = round(sub.corr(method="spearman").iloc[0, 1], 2)
            row[f"n_{label}"] = len(sub)
        rows.append(row)
    return pd.DataFrame(rows)


def forest_importance(df):
    train = df[df["school_year"].isin([2023, 2024])]
    test = df[df["school_year"] == YEAR]
    model = make_pipeline(
        SimpleImputer(strategy="median"),
        RandomForestRegressor(n_estimators=500, min_samples_leaf=5, random_state=42, n_jobs=-1),
    )
    model.fit(train[NEIGH], train["absent_pct"])
    r2 = r2_score(test["absent_pct"], model.predict(test[NEIGH]))
    imp = permutation_importance(model, test[NEIGH], test["absent_pct"],
                                 n_repeats=30, random_state=0, scoring="r2")
    out = pd.DataFrame({"indicator": NEIGH, "r2_drop": imp.importances_mean.round(3),
                        "sd": imp.importances_std.round(3)})
    return r2, out.sort_values("r2_drop", ascending=False)


def make_figures(d):
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    g = d.dropna(subset=["lat", "lon"])
    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(g["lon"], g["lat"], c=g["absent_pct"], cmap="YlOrRd", s=45,
                    edgecolor="black", linewidth=0.3)
    ax.set_aspect(1 / np.cos(np.radians(38.9)))
    ax.set_title(f"DC schools by chronic absenteeism, {YEAR} (dot map)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(sc, ax=ax, label="% of students chronically absent")
    fig.tight_layout()
    fig.savefig("reports/figures/school_map_2025.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for sector, color in (("dcps", "tab:blue"), ("charter", "tab:orange")):
        s = d[d["sector"] == sector].dropna(subset=["poverty_rate"])
        ax.scatter(s["poverty_rate"] * 100, s["absent_pct"], c=color, label=sector, alpha=0.7)
    ax.set_xlabel("Poverty rate of the school's census tract (%)")
    ax.set_ylabel("% of students chronically absent")
    ax.set_title(f"Neighborhood poverty vs. school absenteeism, {YEAR}")
    ax.legend()
    fig.tight_layout()
    fig.savefig("reports/figures/poverty_vs_absenteeism_2025.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    df = pd.read_csv(PANEL, dtype={"school_code": str})
    df = df[df["absent_pct"].notna()]
    d = df[df["school_year"] == YEAR]
    emit(f"Schools in {YEAR}: {len(d)} ({(d['sector'] == 'dcps').sum()} DCPS, "
         f"{(d['sector'] == 'charter').sum()} charter)\n")
    emit("Spearman correlation of each neighborhood indicator with the school's rate")
    emit("(+ means higher in tracts where more schools have high absenteeism)\n")
    emit(spearman_table(d).to_string(index=False))

    r2, imp = forest_importance(df)
    emit(f"\nNeighborhood-only forest (trained on 2023-24, tested on {YEAR}): R^2 = {r2:.2f}")
    emit("Permutation importance (drop in R^2 when the indicator is shuffled):\n")
    emit(imp.to_string(index=False))

    make_figures(d)
    emit("\nCaution: these are associations between a school's tract and the school's rate."
         "\nCharter schools draw students from across the city, so their tract is a weak proxy."
         "\nIndicators are correlated, so importance is shared among them.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/associations.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/associations.txt and 2 figures in reports/figures/")
