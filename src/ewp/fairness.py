"""Fairness check for the watch list: who gets listed, and who does it catch?

The list is the top 20% of schools not yet in the top quarter, ranked by this year's rate
(the rule the dashboard uses). We compare groups of schools on how often they are listed,
how many of the schools that later moved into the top quarter were caught (capture), and
how many listed schools did move in (precision). Years pooled: 2023 and 2024.

Usage: python -m ewp.fairness
Writes: reports/fairness_check.txt
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

from ewp.run_models import PANEL
from ewp.watchlist import add_status, watch_list

YEARS = [2023, 2024]
SHARE = 0.20
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def fmt(k, n):
    if n == 0:
        return "n/a"
    lo, hi = wilson(k, n)
    return f"{k / n:.2f} ({lo:.2f}-{hi:.2f})"


def grade_band(name):
    s = str(name).lower()
    if re.search(r"\b(high|senior)\b", s):
        return "high school"
    if re.search(r"\bmiddle\b", s):
        return "middle school"
    if re.search(r"\b(elementary|education campus)\b", s):
        return "elementary or education campus"
    return "unclear (no grade level in name)"


def collect(panel):
    parts = []
    for year in YEARS:
        _, d = watch_list(panel, year, 1)
        pool_n = int((~d["in_top_quarter"]).sum())
        picked, d = watch_list(panel, year, int(round(SHARE * pool_n)))
        pool = d[~d["in_top_quarter"]].copy()
        pool["on_list"] = pool["school_code"].isin(picked["school_code"])
        pool["moved_in"] = pool["next_top_quarter"] == 1
        parts.append(pool)
    return pd.concat(parts, ignore_index=True)


def table(res, col):
    rows = []
    for g, s in list(res.groupby(col, observed=True)) + [("ALL", res)]:
        listed, moved = int(s["on_list"].sum()), int(s["moved_in"].sum())
        caught = int((s["on_list"] & s["moved_in"]).sum())
        rows.append({
            "group": g, "schools": len(s), "listed": f"{listed / len(s):.0%}",
            "moved in": moved, "caught": caught,
            "capture (95% CI)": fmt(caught, moved) + ("*" if moved < 5 else ""),
            "precision (95% CI)": fmt(caught, listed),
        })
    return pd.DataFrame(rows).to_string(index=False)


if __name__ == "__main__":
    panel = add_status(pd.read_csv(PANEL, dtype={"school_code": str, "geoid": str}))
    res = collect(panel)
    res["sector_group"] = res["sector"].fillna("unknown (no location)")
    res["tract_poverty"] = pd.qcut(res["poverty_rate"], 3, labels=[
        "lower-poverty tract", "middle", "higher-poverty tract"]).astype(object).fillna("no tract data")
    res["grade_band"] = res["school_name"].map(grade_band)

    emit(f"Watch list = top {SHARE:.0%} of schools not yet in the top quarter, by this year's rate.")
    emit(f"School years {YEARS} pooled; outcomes are the following years. "
         f"{len(res)} school-years, {int(res['moved_in'].sum())} moved into the top quarter.")
    emit("listed = share of the group put on the list; capture = share of schools that moved in")
    emit("that were on the list; precision = share of listed schools that moved in.")
    emit("* = fewer than 5 schools moved in, so do not read much into the rate.\n")
    for title, col in (("By sector", "sector_group"),
                       ("By poverty of the school's tract (tertiles)", "tract_poverty"),
                       ("By grade level (inferred from the school name)", "grade_band")):
        emit(f"--- {title} ---")
        emit(table(res, col))
        emit("")
    emit("Ward is not in our data, so it is not checked. Small groups give wide intervals.")
    emit("Similar capture and precision across groups is reassuring; large gaps deserve a closer look.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/fairness_check.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/fairness_check.txt")
