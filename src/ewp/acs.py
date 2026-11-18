"""Download ACS 5-year tract indicators for Washington, DC from the Census API.

Usage: set CENSUS_API_KEY in your shell, then run `python -m ewp.acs`.
Output: data/processed/dc_acs_tracts.csv (gitignored).
"""
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

YEAR = 2023            # ACS 5-year vintage (2019-2023)
STATE, COUNTY = "11", "001"   # District of Columbia

VARS = {
    "B17001_001E": "pov_universe",
    "B17001_002E": "pov_below",
    "B19013_001E": "median_hh_income",
    "B25070_001E": "rent_universe",
    "B25070_007E": "rent_30_35",
    "B25070_008E": "rent_35_40",
    "B25070_009E": "rent_40_50",
    "B25070_010E": "rent_50plus",
    "B25044_001E": "veh_universe",
    "B25044_003E": "owner_no_vehicle",
    "B25044_010E": "renter_no_vehicle",
    "B28002_001E": "net_universe",
    "B28002_013E": "no_internet",
    "C16002_001E": "lang_universe",
    "C16002_004E": "lep_spanish",
    "C16002_007E": "lep_indoeuropean",
    "C16002_010E": "lep_asian",
    "C16002_013E": "lep_other",
}


def fetch(year: int = YEAR) -> pd.DataFrame:
    key = os.environ.get("CENSUS_API_KEY")
    if not key:
        sys.exit("Set your key first: export CENSUS_API_KEY=\"your-key-here\"")
    url = f"https://api.census.gov/data/{year}/acs/acs5"
    params = {
        "get": "NAME," + ",".join(VARS),
        "for": "tract:*",
        "in": f"state:{STATE} county:{COUNTY}",
        "key": key,
    }
    r = requests.get(url, params=params, timeout=60)
    r.raise_for_status()
    rows = r.json()
    df = pd.DataFrame(rows[1:], columns=rows[0]).rename(columns=VARS)
    for c in VARS.values():
        df[c] = pd.to_numeric(df[c], errors="coerce")
        df.loc[df[c] < 0, c] = np.nan   # Census uses large negatives for missing
    df["geoid"] = df["state"] + df["county"] + df["tract"]
    return df


def build_indicators(df: pd.DataFrame) -> pd.DataFrame:
    def rate(num, den):
        return (num / den).where(den > 0)

    out = pd.DataFrame({"geoid": df["geoid"], "name": df["NAME"]})
    out["poverty_rate"] = rate(df["pov_below"], df["pov_universe"])
    out["median_hh_income"] = df["median_hh_income"]
    out["rent_burden"] = rate(
        df[["rent_30_35", "rent_35_40", "rent_40_50", "rent_50plus"]].sum(axis=1),
        df["rent_universe"],
    )
    out["no_vehicle_hh"] = rate(df["owner_no_vehicle"] + df["renter_no_vehicle"], df["veh_universe"])
    out["no_internet_hh"] = rate(df["no_internet"], df["net_universe"])
    out["limited_english_hh"] = rate(
        df[["lep_spanish", "lep_indoeuropean", "lep_asian", "lep_other"]].sum(axis=1),
        df["lang_universe"],
    )
    return out


if __name__ == "__main__":
    out = build_indicators(fetch())
    path = Path("data/processed/dc_acs_tracts.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"Saved {len(out)} tracts to {path}")
    print(out.describe().round(3).to_string())
