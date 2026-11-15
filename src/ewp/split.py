"""Temporal splitting. Never use random splits for early-warning evaluation."""
import pandas as pd


def temporal_split(df: pd.DataFrame, year_col: str, test_year: int):
    """Train on years strictly before test_year; test on test_year."""
    train = df[df[year_col] < test_year].copy()
    test = df[df[year_col] == test_year].copy()
    if train.empty or test.empty:
        raise ValueError("Empty train or test set; check year_col and test_year.")
    return train, test


def rolling_year_folds(df: pd.DataFrame, year_col: str, min_train_years: int = 2):
    """Yield (train_idx, val_idx) for expanding-window time-aware CV."""
    years = sorted(df[year_col].unique())
    for i in range(min_train_years, len(years)):
        train_idx = df.index[df[year_col].isin(years[:i])]
        val_idx = df.index[df[year_col] == years[i]]
        yield train_idx, val_idx
