# Community Early-Warning: Predicting Chronic Absenteeism with Community Context

An early-warning prototype that asks: **do community-level indicators and interview-derived
themes improve how schools identify students who could benefit from support, before
attendance problems become chronic?**

> **Status:** early development. Results will be added as analyses are completed.
> Any synthetic-data output is a pipeline demonstration, not a research finding.

## Research questions

1. Can attendance, academic performance, and neighborhood socioeconomic indicators predict future chronic absenteeism?
2. Do models with community-level factors outperform models using traditional indicators alone?
3. Which factors are most associated with persistent attendance challenges?
4. Can predictive analytics help community schools allocate intervention resources more effectively? (simulation only)
5. Can sentence-embedding methods detect barrier themes in student interviews, and do those themes add predictive lift?

## Approach

Nested model comparison, evaluated on a strict temporal split:

| Model | Features |
|---|---|
| A | Prior attendance |
| B | A + academics + discipline |
| C | B + neighborhood (ACS) indicators |
| D | C + interview-derived theme scores |

- Models: regularized logistic regression, random forest, XGBoost
- Metrics: PR-AUC, recall at fixed review capacity, calibration (not accuracy)
- Explainability: SHAP on grouped features
- Geography: Moran's I, Getis-Ord Gi*, choropleths, spatial cross-validation
- Text: sentence-transformer embeddings, zero-shot then SetFit theme detection, validated against human coders
- Fairness: error-rate and calibration audits by subgroup

## Repo layout

```
configs/      project settings (years, thresholds, feature groups)
data/         raw / interim / processed / synthetic (gitignored except .gitkeep)
docs/         data sources, theme codebook, ethics and privacy
notebooks/    EDA, modeling, SHAP, spatial analysis
src/ewp/      reusable code: temporal split, metrics, models, theme detection
app/          Streamlit dashboard
tests/        unit tests
```

## Quickstart

```bash
git clone https://github.com/<your-username>/community-early-warning.git
cd community-early-warning
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pytest
streamlit run app/streamlit_app.py
```

## Data

See [docs/data_sources.md](docs/data_sources.md). No student-level or interview data is stored
in this repo. Never commit real student records or transcripts.

## Ethics

Predictions are framed as **offers of support**, not risk labels. See
[docs/ethics_and_privacy.md](docs/ethics_and_privacy.md).

## License

MIT
