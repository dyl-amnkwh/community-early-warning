import numpy as np
import pandas as pd
import pytest

from ewp.metrics import recall_at_capacity
from ewp.split import temporal_split


def test_temporal_split_no_leakage():
    df = pd.DataFrame({"school_year": [2021, 2022, 2023, 2024, 2024], "x": range(5)})
    train, test = temporal_split(df, "school_year", 2024)
    assert train["school_year"].max() < test["school_year"].min()


def test_temporal_split_empty_raises():
    df = pd.DataFrame({"school_year": [2024], "x": [1]})
    with pytest.raises(ValueError):
        temporal_split(df, "school_year", 2024)


def test_recall_at_capacity_perfect_ranking():
    y = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    s = np.array([0.9, 0.8, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    assert recall_at_capacity(y, s, 0.2) == 1.0
