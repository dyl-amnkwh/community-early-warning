"""Build the school-year modeling table for DC.

Each row is one school in one year (t). Features come from year t; the
targets describe what happens in year t+1.

Usage: python -m ewp.panel
Writes: data/processed/dc_panel.csv (gitignored)
"""
from pathlib import Path

import pandas as pd

from ewp.schools import clean_code

PROC = Path("data/processed")
EXCLUDE = {"1214", "1220", "1176"}  # detention-site schools; Riverseed (location unverified, 2 years only)


def shifted(df, col, years, name):
    s = df[["school_code", "school_year", col]].copy()
    s["school_year"] = s["school_year"] + years
    return s.rename(columns={col: name})


def build() -> pd.DataFrame:
    ab = pd.read_csv(PROC / "dc_chronic_absenteeism_school_year.csv", dtype={"school_code": str})
    loc = pd.read_csv(PROC / "dc_school_locations.csv", dtype={"school_code": str, "geoid": str})
    acs = pd.read_csv(PROC / "dc_acs_tracts.csv", dtype={"geoid": str})

    ab["school_code"] = ab["school_code"].map(clean_code)
    loc["school_code"] = loc["school_code"].map(clean_code)
    ab = ab[~ab["school_code"].isin(EXCLUDE)].rename(columns={"chronic_absent_pct": "absent_pct"})

    df = ab.merge(shifted(ab, "absent_pct", 1, "prior_pct"), on=["school_code", "school_year"], how="left")
    df = df.merge(shifted(ab, "absent_pct", -1, "next_pct"), on=["school_code", "school_year"], how="left")
    df["change_from_prior"] = df["absent_pct"] - df["prior_pct"]
    df["target_year"] = df["school_year"] + 1
    has = df["next_pct"].notna()
    df["next_over_50"] = (df["next_pct"] >= 50).astype(float).where(has)
    df["next_rise"] = (df["next_pct"] > df["absent_pct"]).astype(float).where(has)
    rank = df.groupby("target_year")["next_pct"].rank(pct=True)
    df["next_top_quarter"] = (rank > 0.75).astype(float).where(has)

    keep = ["school_code", "sector", "lat", "lon", "geoid"]
    df = df.merge(loc[keep].drop_duplicates("school_code"), on="school_code", how="left")
    return df.merge(acs.drop(columns=["name"]), on="geoid", how="left")


if __name__ == "__main__":
    df = build()
    out = PROC / "dc_panel.csv"
    df.to_csv(out, index=False)
    print(f"Saved {len(df)} rows to {out}")
    ok = df[df["next_pct"].notna()]
    print(f"Rows with a next-year outcome: {len(ok)}\n")
    s = ok.groupby("target_year").agg(
        schools=("school_code", "nunique"),
        share_over_50=("next_over_50", "mean"),
        share_rise=("next_rise", "mean"),
        share_top_quarter=("next_top_quarter", "mean"),
        with_tract_data=("poverty_rate", "count"),
    ).round(2)
    print(s.to_string())
