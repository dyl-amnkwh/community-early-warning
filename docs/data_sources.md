# Data sources

| Source | Use | Notes |
|---|---|---|
| Civil Rights Data Collection (CRDC) | School-level chronic absenteeism | Biennial |
| NCES CCD and EDGE | School locations, neighborhood estimates | Link schools to tracts |
| ACS 5-year (Census Bureau) | Tract-level socioeconomic indicators | Heavily collinear; use margins of error |
| VDOE School Quality Profiles | Virginia school-level absenteeism | Case study |
| Fairfax County Public Schools open data | District-level context | Check current availability |
| Synthetic student-level data | Pipeline demo only | Must be labeled synthetic |

## Linking
School lat/long -> census tract (geopandas spatial join) -> ACS tract features.

## Rules
- Do not commit raw data. Record download dates and URLs here.
- Student-level data requires a data-use agreement (FERPA). Interview data requires IRB review.
