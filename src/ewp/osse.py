"""Combine OSSE DC School Report Card attendance files into one school-year table.

Usage: python -m ewp.osse
Reads:  data/raw/*Attendance*.xlsx
Writes: data/processed/dc_chronic_absenteeism_school_year.csv (gitignored)
"""
import re
from pathlib import Path

import pandas as pd

RAW = Path("data/raw")
OUT = Path("data/processed/dc_chronic_absenteeism_school_year.csv")


def school_sheet(path) -> str:
    for name in pd.ExcelFile(path).sheet_names:
        if name == "School Metric Scores" or (name.startswith("School") and "Overall" in name):
            return name
    raise ValueError(f"No school-level sheet found in {path}")


def end_year(label):
    """'2021'->2021, '2022-23'->2023, 'SY2023-24'->2024, '2024-25'->2025."""
    s = str(label)
    m = re.search(r"(\d{4})\s*-\s*(\d{2})", s)
    if m:
        return 2000 + int(m.group(2))
    m = re.search(r"(\d{4})", s)
    return int(m.group(1)) if m else None


def load_all() -> pd.DataFrame:
    frames = []
    for f in sorted(RAW.glob("*Attendance*.xlsx")):
        d = pd.read_excel(f, sheet_name=school_sheet(f))
        d.columns = [str(c).strip() for c in d.columns]
        metric = d["Metric"].astype(str).str.strip()
        group = d["Student Group"].astype(str).str.strip()
        d = d[(metric == "Chronic Absenteeism") & (group == "All Report Card Students")].copy()
        d["source_file"] = f.name
        frames.append(d)
    if not frames:
        raise SystemExit("No attendance files found in data/raw/")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({
        "lea_name": df["LEA Name"],
        "lea_code": df["LEA Code"].astype(str),
        "school_name": df["School Name"],
        "school_code": df["School Code"].astype(str),
        "school_year": df["School Year"].map(end_year),
        "chronic_absent_pct": pd.to_numeric(df["Metric Score"], errors="coerce"),
        "n_students": pd.to_numeric(df["Metric N"], errors="coerce"),
    })
    out["pandemic_year"] = out["school_year"] == 2021
    before = len(out)
    out = out.drop_duplicates(["school_code", "school_year"], keep="first")
    if len(out) < before:
        print(f"Note: dropped {before - len(out)} duplicate school-year rows")
    return out.sort_values(["school_year", "school_code"]).reset_index(drop=True)


if __name__ == "__main__":
    out = clean(load_all())
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(f"Saved {len(out)} school-year rows to {OUT}\n")
    summary = out.groupby("school_year").agg(
        schools=("school_code", "nunique"),
        with_rate=("chronic_absent_pct", "count"),
        mean_pct=("chronic_absent_pct", "mean"),
        median_pct=("chronic_absent_pct", "median"),
    ).round(1)
    print(summary.to_string())
