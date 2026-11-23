# Data sources

| Source | Use | Notes |
|---|---|---|
| OSSE DC School Report Card attendance files | School-level chronic absenteeism, 2021-2025 | Manual download from osse.dc.gov; school-year labels vary by file |
| Census ACS 5-year (2019-2023) | Tract indicators: poverty, income, rent burden, no vehicle, no internet, limited English | Census API; use margins of error in any further work |
| DC Open Data: DC Public Schools and DC Charter Schools | School coordinates | Matched to OSSE by school code |
| Census Geocoder | Assign each school's coordinates to a census tract | Current tract geography |

## Linking
OSSE school code -> location file (code match) -> coordinates -> census tract -> ACS indicators.

## Rules

## Rules
- Do not commit raw data. Record download dates and URLs here.
- Student-level data would need a data-use agreement; interview data would need IRB review.
