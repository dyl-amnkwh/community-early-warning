import pandas as pd

from ewp.watchlist import add_status, backtest, needs_profile, watch_list


def make_panel():
    return add_status(pd.DataFrame({
        "school_year": [2024] * 8,
        "school_code": list("abcdefgh"),
        "sector": ["dcps"] * 8,
        "absent_pct": [10, 20, 30, 40, 50, 60, 70, 80],
        "next_top_quarter": [0, 0, 0, 1, 0, 1, 1, 1],
    }))


def test_watch_list_excludes_schools_already_in_top_quarter():
    picked, _ = watch_list(make_panel(), 2024, capacity=3)
    assert list(picked["school_code"]) == ["f", "e", "d"]


def test_backtest_counts_schools_that_entered():
    bt = backtest(make_panel(), 2024, capacity=3)
    assert bt == {"entered": 2, "caught": 2, "pool": 6}


def test_backtest_none_without_outcomes():
    df = make_panel().assign(next_top_quarter=float("nan"))
    assert backtest(df, 2024, capacity=3) is None


def test_needs_profile_flags_high_indicators():
    row = {"no_internet_hh": 0.4, "poverty_rate": 0.01}
    cut = {"no_vehicle_hh": 1, "no_internet_hh": 0.3, "rent_burden": 1,
           "poverty_rate": 0.2, "limited_english_hh": 1}
    assert needs_profile(row, cut) == "internet access"
