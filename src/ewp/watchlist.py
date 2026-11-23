"""Watch-list logic for the dashboard (kept separate so it can be tested)."""
import pandas as pd

NEED_FLAGS = {
    "no_vehicle_hh": "transportation",
    "no_internet_hh": "internet access",
    "rent_burden": "housing cost",
    "poverty_rate": "economic hardship",
    "limited_english_hh": "language access",
}


def tract_cutoffs(acs: pd.DataFrame, q: float = 0.75) -> dict:
    """Top-quartile cutoff for each indicator across all DC tracts."""
    return {col: float(acs[col].quantile(q)) for col in NEED_FLAGS}


def needs_profile(row, cutoffs: dict) -> str:
    hits = [label for col, label in NEED_FLAGS.items()
            if pd.notna(row.get(col)) and row[col] >= cutoffs[col]]
    return ", ".join(hits) if hits else "none in the top quarter of DC tracts"


def add_status(panel: pd.DataFrame) -> pd.DataFrame:
    """Mark schools in the top quarter of chronic absenteeism for their year."""
    df = panel.copy()
    rank = df.groupby("school_year")["absent_pct"].rank(pct=True)
    df["in_top_quarter"] = rank > 0.75
    return df


def watch_list(df: pd.DataFrame, year: int, capacity: int, sector: str = "All"):
    """Schools not yet in the top quarter, ranked by this year's rate."""
    d = df[(df["school_year"] == year) & df["absent_pct"].notna()]
    if sector != "All":
        d = d[d["sector"] == sector]
    pool = d[~d["in_top_quarter"]].sort_values("absent_pct", ascending=False)
    return pool.head(capacity), d


def backtest(df: pd.DataFrame, year: int, capacity: int, sector: str = "All"):
    """How many schools that entered the top quarter next year were on the list?"""
    picked, d = watch_list(df, year, capacity, sector)
    pool = d[~d["in_top_quarter"]]
    if pool["next_top_quarter"].notna().sum() == 0:
        return None
    return {"entered": int((pool["next_top_quarter"] == 1).sum()),
            "caught": int((picked["next_top_quarter"] == 1).sum()),
            "pool": len(pool)}
