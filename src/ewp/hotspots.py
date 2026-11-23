"""Hotspot analysis: where do community needs and school absenteeism concentrate?

Needs a one-time download of Census tract boundaries (about 1 MB) and the pyshp package.

Usage: python -m ewp.hotspots
Writes: reports/hotspots.txt, reports/figures/need_map.png,
        reports/figures/school_hotspots_2025.png
"""
import zipfile
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import shapefile
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon

RAW, PROC = Path("data/raw"), Path("data/processed")
YEAR = 2025
K_TRACT, K_SCHOOL = 6, 8
NEED_COLS = ["poverty_rate", "rent_burden", "no_vehicle_hh", "no_internet_hh"]
URLS = [
    "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_11_tract_500k.zip",
    "https://www2.census.gov/geo/tiger/GENZ2022/shp/cb_2022_11_tract_500k.zip",
]
LINES = []


def emit(text=""):
    print(text)
    LINES.append(text)


def boundaries():
    """Return {geoid: [ring, ...]}; each ring is a list of (lon, lat) points."""
    path = RAW / "tract_boundaries.zip"
    if not path.exists():
        for url in URLS:
            try:
                r = requests.get(url, timeout=120)
                r.raise_for_status()
                path.write_bytes(r.content)
                break
            except requests.RequestException as e:
                print(f"Could not download {url}: {e}")
        else:
            raise SystemExit("Download the DC tract file (cb_..._11_tract_500k.zip) from "
                             "https://www2.census.gov/geo/tiger/GENZ2023/shp/ and save it as "
                             "data/raw/tract_boundaries.zip")
    z = zipfile.ZipFile(path)
    part = {Path(n).suffix.lower(): n for n in z.namelist()}
    rd = shapefile.Reader(shp=BytesIO(z.read(part[".shp"])), shx=BytesIO(z.read(part[".shx"])),
                          dbf=BytesIO(z.read(part[".dbf"])))
    names = [f[0] for f in rd.fields[1:]]
    out = {}
    for sr in rd.shapeRecords():
        pts = sr.shape.points
        cuts = list(sr.shape.parts) + [len(pts)]
        out[dict(zip(names, sr.record))["GEOID"]] = [pts[cuts[i]:cuts[i + 1]]
                                                      for i in range(len(cuts) - 1)]
    return out


def centroid(rings):
    best, best_area = None, 0.0
    for ring in rings:
        a = np.asarray(ring, float)
        x, y = a[:, 0], a[:, 1]
        cross = x[:-1] * y[1:] - x[1:] * y[:-1]
        area = cross.sum() / 2
        if abs(area) > best_area:
            best_area = abs(area)
            best = (((x[:-1] + x[1:]) * cross).sum() / (6 * area),
                    ((y[:-1] + y[1:]) * cross).sum() / (6 * area))
    return best


def need_index(acs):
    """Equal-weight average of standardized poverty, rent burden, no-vehicle, no-internet."""
    z = (acs[NEED_COLS] - acs[NEED_COLS].mean()) / acs[NEED_COLS].std()
    return z.mean(axis=1).where(z.notna().sum(axis=1) >= 3)


def project(lonlat):
    a = np.asarray(lonlat, float)
    return np.column_stack([a[:, 0] * np.cos(np.radians(38.9)), a[:, 1]])


def knn_neighbors(xy, k):
    d = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1))
    np.fill_diagonal(d, np.inf)
    return np.argsort(d, axis=1)[:, :k]


def morans_i(x, nbrs, n_perm=999, seed=0):
    x = np.asarray(x, float)

    def stat(v):
        z = v - v.mean()
        return (z * z[nbrs].mean(axis=1)).sum() / (z * z).sum()

    obs = stat(x)
    rng = np.random.default_rng(seed)
    perm = np.array([stat(rng.permutation(x)) for _ in range(n_perm)])
    return obs, -1 / (len(x) - 1), (1 + (perm >= obs).sum()) / (n_perm + 1)


def gi_star(x, nbrs):
    x = np.asarray(x, float)
    n = len(x)
    idx = np.column_stack([np.arange(n), nbrs])
    w = idx.shape[1]
    s = np.sqrt((x ** 2).mean() - x.mean() ** 2)
    return (x[idx].sum(axis=1) - x.mean() * w) / (s * np.sqrt((n * w - w * w) / (n - 1)))


def draw_tracts(ax, tracts, values=None, cmap="Blues", label=None, fig=None):
    patches, vals = [], []
    for geoid, rings in tracts.items():
        for ring in rings:
            patches.append(Polygon(ring, closed=True))
            vals.append(np.nan if values is None else values.get(geoid, np.nan))
    pc = PatchCollection(patches, edgecolor="white", linewidth=0.4)
    if values is None:
        pc.set_facecolor("#EEEEEE")
    else:
        cm = plt.get_cmap(cmap).copy()
        cm.set_bad("#DDDDDD")
        pc.set_cmap(cm)
        pc.set_array(np.ma.masked_invalid(vals))
    ax.add_collection(pc)
    ax.autoscale_view()
    ax.set_aspect(1 / np.cos(np.radians(38.9)))
    ax.set_xticks([])
    ax.set_yticks([])
    if values is not None:
        fig.colorbar(pc, ax=ax, label=label, shrink=0.7)


if __name__ == "__main__":
    tracts = boundaries()
    acs = pd.read_csv(PROC / "dc_acs_tracts.csv", dtype={"geoid": str})
    acs["need"] = need_index(acs)
    need = dict(zip(acs["geoid"], acs["need"]))
    panel = pd.read_csv(PROC / "dc_panel.csv", dtype={"school_code": str, "geoid": str})
    s = panel[(panel["school_year"] == YEAR) & panel["absent_pct"].notna()
              & panel["lat"].notna()].copy().reset_index(drop=True)
    s["need"] = s["geoid"].map(need)

    ok = [g for g in tracts if pd.notna(need.get(g))]
    nb = knn_neighbors(project([centroid(tracts[g]) for g in ok]), K_TRACT)
    i, e, p = morans_i([need[g] for g in ok], nb)
    emit(f"Community need index, {len(ok)} tracts: Moran's I = {i:.2f} (random would be {e:.3f}), p = {p:.3f}")
    emit("(need index = equal-weight average of standardized poverty, rent burden, no-vehicle, no-internet)\n")
    for label, sub in (("all located schools", s), ("DCPS schools only", s[s["sector"] == "dcps"])):
        nbs = knn_neighbors(project(sub[["lon", "lat"]]), K_SCHOOL)
        i, e, p = morans_i(sub["absent_pct"], nbs)
        emit(f"{YEAR} absenteeism rate, {label} ({len(sub)}): Moran's I = {i:.2f} "
             f"(random would be {e:.3f}), p = {p:.3f}")

    z = gi_star(s["absent_pct"], knn_neighbors(project(s[["lon", "lat"]]), K_SCHOOL))
    s["spot"] = np.where(z >= 1.96, "hot spot", np.where(z <= -1.96, "cold spot", "not significant"))
    emit(f"\nGetis-Ord Gi* hot and cold spots of {YEAR} absenteeism (each school vs. its {K_SCHOOL} nearest)")
    emit(s.groupby("spot").agg(schools=("school_code", "count"), mean_rate=("absent_pct", "mean"),
                               mean_need=("need", "mean"), dcps=("sector", lambda v: (v == "dcps").sum()),
                               charter=("sector", lambda v: (v == "charter").sum())).round(2).to_string())
    emit("\nHot-spot schools (exploratory; no correction for multiple tests):")
    emit(s[s["spot"] == "hot spot"].sort_values("absent_pct", ascending=False)[
        ["school_name", "sector", "absent_pct"]].head(20).round(1).to_string(index=False))

    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 8))
    draw_tracts(ax, tracts, need, "Blues", "Community need index (higher = more need)", fig)
    ax.scatter(s["lon"], s["lat"], s=14, c="black", alpha=0.7, label="school")
    ax.set_title("Community need by census tract, with school locations")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("reports/figures/need_map.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    draw_tracts(ax, tracts)
    for name, color in (("not significant", "#999999"), ("hot spot", "#E66100"), ("cold spot", "#1F77B4")):
        g = s[s["spot"] == name]
        ax.scatter(g["lon"], g["lat"], s=40 if name != "not significant" else 18, c=color,
                   edgecolor="black", linewidth=0.3, label=f"{name} ({len(g)})")
    ax.set_title(f"Hot and cold spots of school absenteeism, {YEAR}")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig("reports/figures/school_hotspots_2025.png", dpi=150)
    plt.close(fig)

    emit("\nAssociation only. Charter schools draw students citywide, so their location is a weak proxy.")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/hotspots.txt").write_text("\n".join(LINES), encoding="utf-8")
    print("\nSaved reports/hotspots.txt and 2 figures in reports/figures/")
