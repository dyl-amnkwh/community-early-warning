"""Locate OSSE schools and assign each to a 2020 census tract.

Usage: python -m ewp.schools
Reads:  data/raw/DC_Charter_Schools.csv, data/raw/DC_Public_Schools.csv,
        data/processed/dc_chronic_absenteeism_school_year.csv
Writes: data/processed/dc_school_locations.csv
Cache:  data/interim/tract_lookup.csv (so reruns don't re-query the Census)
"""
import difflib
import re
import time
from pathlib import Path

import pandas as pd
import requests

RAW, PROC, INTERIM = Path("data/raw"), Path("data/processed"), Path("data/interim")
GEO_URL = "https://geocoding.geo.census.gov/geocoder/geographies/coordinates"


def clean_code(x):
    try:
        return str(int(float(x)))
    except (TypeError, ValueError):
        return None


def norm(s):
    s = re.sub(r"[^a-z0-9 ]", " ", str(s).lower())
    s = re.sub(r"\b(school|pcs|dcps|the|of|at|and|public|charter)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def sim(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio()


def load_locations() -> pd.DataFrame:
    ch = pd.read_csv(RAW / "DC_Charter_Schools.csv")
    dp = pd.read_csv(RAW / "DC_Public_Schools.csv")
    a = pd.DataFrame({"sector": "charter",
                      "loc_name": ch["SCHOOL_NAM"].fillna(ch["NAME"]),
                      "loc_code": ch["SCHOOLCODE"].map(clean_code),
                      "lat": ch["LATITUDE"], "lon": ch["LONGITUDE"]})
    b = pd.DataFrame({"sector": "dcps",
                      "loc_name": dp["SCHOOL_NAM"].fillna(dp["NAME"]),
                      "loc_code": dp["SCHOOL_ID"].map(clean_code),
                      "lat": dp["LATITUDE"], "lon": dp["LONGITUDE"]})
    locs = pd.concat([a, b], ignore_index=True).dropna(subset=["lat", "lon"])
    return locs.reset_index(drop=True)


def match_schools(schools: pd.DataFrame, locs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, s in schools.iterrows():
        best, method, score = None, None, 0.0
        same = locs[locs["loc_code"] == s["school_code"]]
        if len(same):
            sc = same["loc_name"].map(lambda n: sim(n, s["school_name"]))
            one_place = same[["lat", "lon"]].drop_duplicates().shape[0] == 1
            if one_place or sc.max() >= 0.5:
                best, method, score = same.loc[sc.idxmax()], "code", sc.max()
        if best is None:
            sc = locs["loc_name"].map(lambda n: sim(n, s["school_name"]))
            if sc.max() >= 0.8:
                best, method, score = locs.loc[sc.idxmax()], "name", sc.max()
        rows.append({
            "school_code": s["school_code"], "school_name": s["school_name"],
            "sector": best["sector"] if best is not None else None,
            "loc_name": best["loc_name"] if best is not None else None,
            "lat": best["lat"] if best is not None else None,
            "lon": best["lon"] if best is not None else None,
            "match_method": method, "match_sim": round(score, 2),
        })
    return pd.DataFrame(rows)


def tract_for(lat, lon):
    params = {"x": lon, "y": lat, "benchmark": "Public_AR_Current",
              "vintage": "Current_Current", "layers": "Census Tracts", "format": "json"}
    for _ in range(3):
        try:
            r = requests.get(GEO_URL, params=params, timeout=30)
            r.raise_for_status()
            return r.json()["result"]["geographies"]["Census Tracts"][0]["GEOID"]
        except Exception:
            time.sleep(2)
    return None


def add_tracts(df: pd.DataFrame) -> pd.DataFrame:
    cache_path = INTERIM / "tract_lookup.csv"
    cache = {}
    if cache_path.exists():
        c = pd.read_csv(cache_path, dtype={"geoid": str})
        cache = {(r.lat, r.lon): r.geoid for r in c.itertuples()}
    coords = df.dropna(subset=["lat", "lon"])[["lat", "lon"]].round(6).drop_duplicates()
    todo = [(r.lat, r.lon) for r in coords.itertuples() if (r.lat, r.lon) not in cache]
    print(f"Looking up {len(todo)} locations at the Census geocoder (this takes a few minutes)...")
    for i, (la, lo) in enumerate(todo, 1):
        g = tract_for(la, lo)
        if g:
            cache[(la, lo)] = g
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}")
        time.sleep(0.2)
    INTERIM.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([(k[0], k[1], v) for k, v in cache.items()],
                 columns=["lat", "lon", "geoid"]).to_csv(cache_path, index=False)
    df = df.copy()
    df["lat"], df["lon"] = df["lat"].round(6), df["lon"].round(6)
    df["geoid"] = [cache.get((a, b)) for a, b in zip(df["lat"], df["lon"])]
    return df


if __name__ == "__main__":
    osse = pd.read_csv(PROC / "dc_chronic_absenteeism_school_year.csv",
                       dtype={"school_code": str})
    osse["school_code"] = osse["school_code"].map(clean_code)
    schools = (osse.sort_values("school_year")
                   .drop_duplicates("school_code", keep="last")
                   [["school_code", "school_name"]])
    out = add_tracts(match_schools(schools, load_locations()))
    out.to_csv(PROC / "dc_school_locations.csv", index=False)

    n, m = len(out), out["lat"].notna().sum()
    print(f"\nSchools in OSSE data: {n}")
    print(f"Matched to a location: {m}  (by code: {(out.match_method == 'code').sum()}, "
          f"by name: {(out.match_method == 'name').sum()})")
    print(f"Assigned a census tract: {out['geoid'].notna().sum()}")
    acs = pd.read_csv(PROC / "dc_acs_tracts.csv", dtype={"geoid": str})
    print(f"Tracts found in ACS table: {out['geoid'].isin(acs['geoid']).sum()}")
    miss = out[out["lat"].isna()]
    if len(miss):
        print("\nUnmatched schools:")
        print(miss[["school_code", "school_name"]].head(30).to_string(index=False))
