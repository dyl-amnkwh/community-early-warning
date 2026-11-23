"""DC Community Early-Warning dashboard.

Run from the project folder:  streamlit run app/streamlit_app.py
"""
from pathlib import Path

import pandas as pd
import streamlit as st

from ewp.watchlist import add_status, backtest, needs_profile, tract_cutoffs, watch_list

PROC, REPORTS = Path("data/processed"), Path("reports")
ORANGE, PURPLE, GREY = "#E66100", "#5D3A9B", "#AAAAAA"

st.set_page_config(page_title="DC Community Early-Warning", layout="wide")


@st.cache_data
def load():
    panel = pd.read_csv(PROC / "dc_panel.csv", dtype={"school_code": str, "geoid": str})
    acs = pd.read_csv(PROC / "dc_acs_tracts.csv", dtype={"geoid": str})
    return add_status(panel), tract_cutoffs(acs)


panel, cutoffs = load()
years = sorted(int(y) for y in panel["school_year"].unique())

st.title("DC Community Early-Warning Prototype")
st.caption("Public school-level data (OSSE attendance, Census ACS). "
           "These are offers of support, not risk labels for schools or students.")

with st.sidebar:
    year = st.selectbox("School year (ending)", years, index=len(years) - 1)
    capacity = st.slider("Schools your team can support", 5, 60, 30)
    sector = st.radio("Sector", ["All", "dcps", "charter"], horizontal=True)
    st.caption("The watch list is schools NOT yet in the top quarter of chronic "
               "absenteeism, ranked by this year's rate.")

picked, d = watch_list(panel, year, capacity, sector)
tab_map, tab_list, tab_evidence, tab_limits = st.tabs(
    ["Where support may be needed", "Watch list", "What the evidence says", "Limits"])

with tab_map:
    if year == 2021:
        st.warning("2021 was a pandemic-disrupted year; compare with care.")
    bt = backtest(panel, year, capacity, sector)
    if bt is None:
        st.info("Latest year: this is the live list. Next year's outcomes are not known yet.")
    elif bt["entered"] == 0:
        st.info("No schools in this view moved into the top quarter the next year.")
    else:
        st.success(f"Backtest: {bt['caught']} of the {bt['entered']} schools that moved into "
                   f"the top quarter the next year were on this list of {len(picked)}.")
    m = d.dropna(subset=["lat", "lon"]).copy()
    m["color"] = GREY
    m["size"] = 60
    m.loc[m["in_top_quarter"], ["color", "size"]] = [PURPLE, 140]
    m.loc[m["school_code"].isin(picked["school_code"]), ["color", "size"]] = [ORANGE, 220]
    st.map(m, latitude="lat", longitude="lon", color="color", size="size")
    st.caption("Orange: on the watch list. Purple: already in the top quarter. Grey: other schools.")

with tab_list:
    t = picked.copy()
    t["needs"] = [needs_profile(r, cutoffs) if r["sector"] == "dcps"
                  else "n/a (charter tract is a weak proxy)" for _, r in t.iterrows()]
    out = t[["school_name", "sector", "absent_pct", "change_from_prior", "needs"]].rename(columns={
        "school_name": "School", "sector": "Sector", "absent_pct": "Chronically absent (%)",
        "change_from_prior": "Change vs. prior year (pts)", "needs": "Community needs nearby"})
    st.dataframe(out.round(1), hide_index=True)
    st.download_button("Download this list (CSV)", out.to_csv(index=False), "watch_list.csv")
    st.caption("Community needs: indicators in the top quarter of DC tracts. Shown for DCPS "
               "schools only; they describe the neighborhood, not individual students.")

with tab_evidence:
    st.markdown("""
- **Attendance history does most of the work.** Ranking DCPS schools by this year's rate predicted
  next year's top-quarter schools with PR-AUC of about 0.86 (chance is 0.26).
- **Neighborhood data did not improve on that.** Adding it changed PR-AUC by -0.008
  (95% interval -0.036 to +0.020) for DCPS schools.
- **Watching the 20% of not-yet-flagged schools with the highest rate caught 23 of the 28**
  that later moved into the top quarter (about 82%). Small numbers, wide uncertainty.
- **Tract indicators alone explain roughly 17% of the variation** in school rates for schools
  in tracts the model had not seen (23% for DCPS, 7% for charters).
""")
    for name in ("early_warning_test", "dcps_check", "grouped_check", "fairness_check", "associations"):
        f = REPORTS / f"{name}.txt"
        if f.exists():
            with st.expander(f"Full output: {name}"):
                st.code(f.read_text(encoding="utf-8"))

with tab_limits:
    st.markdown("""
- **Schools, not students.** No student records are used; patterns here may not hold for individuals.
- **Association, not cause.** Neighborhood indicators go with absenteeism; they do not explain it.
- **Charter schools draw students citywide,** so a charter's tract says little about its students.
- **Small samples.** About 230 schools and four year-to-year transitions; 2021 was a pandemic year.
- **Charter schools.** The list is less precise for charters: roughly one in eight listed
  charter schools later entered the top quarter, versus about half of listed DCPS schools.
- **A simple rule works.** The ranking here is just this year's rate; models did not beat it.
- **Human judgment first.** Use this list to start conversations and offer support, not to label schools.
""")
