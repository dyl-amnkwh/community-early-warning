"""Dashboard placeholder. Replace the demo data with processed outputs."""
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Community Early-Warning", layout="wide")
st.title("Community Early-Warning Prototype")
st.warning("Demo data. Synthetic values for layout only; not findings.")

capacity = st.slider("Review capacity (% of schools supported)", 1, 50, 10)

rng = np.random.default_rng(0)
df = pd.DataFrame({
    "school": [f"School {i}" for i in range(1, 41)],
    "support_priority": rng.random(40),
    "poverty_rate": rng.random(40),
})
n = max(1, int(len(df) * capacity / 100))
st.subheader(f"Top {n} schools by support priority")
st.dataframe(df.sort_values("support_priority", ascending=False).head(n), use_container_width=True)
