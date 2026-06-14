"""Tests for data_cleaner.py — market hours filter, holiday drop, gap fill."""
import pandas as pd
import pytest
import pytz
from src.data_cleaner import clean, get_session_bars

IST = pytz.timezone("Asia/Kolkata")


def make_df(times, closes=None):
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz=IST) for t in times])
    closes = closes or [100.0] * len(times)
    return pd.DataFrame({
        "open": closes, "high": closes, "low": closes, "close": closes, "volume": [1000] * len(idx)
    }, index=idx)


def test_pre_market_bars_dropped():
    df = make_df(["2025-06-09 09:00", "2025-06-09 09:10", "2025-06-09 09:15", "2025-06-09 09:20"])
    result = clean(df)
    assert all(result.index.time >= pd.Timestamp("09:15").time())


def test_post_market_bars_dropped():
    df = make_df(["2025-06-09 15:25", "2025-06-09 15:30", "2025-06-09 15:35"])
    result = clean(df)
    # 15:30 and 15:35 should be excluded (market closes at 15:30, filter is strict <)
    assert all(t < pd.Timestamp("15:30").time() for t in result.index.time)


def test_holiday_dropped():
    # 2025-04-14 is a known holiday
    df = make_df(["2025-04-14 09:15", "2025-04-14 09:20", "2025-06-09 09:15"])
    result = clean(df)
    assert not any(result.index.date == pd.Timestamp("2025-04-14").date())


def test_weekend_dropped():
    df = make_df(["2025-06-07 09:15", "2025-06-08 09:15", "2025-06-09 09:15"])
    result = clean(df)
    assert len(result) == 1
    assert result.index[0].weekday() == 0  # Monday


def test_nan_ohlc_dropped():
    df = make_df(["2025-06-09 09:15", "2025-06-09 09:20"])
    df.iloc[0, df.columns.get_loc("close")] = float("nan")
    result = clean(df)
    assert len(result) == 1


def test_sensex_zero_volume_kept():
    df = make_df(["2025-06-09 09:15"], closes=[80000.0])
    df["volume"] = 0
    result = clean(df, symbol="SENSEX")
    assert len(result) == 1   # Not dropped, just flagged
