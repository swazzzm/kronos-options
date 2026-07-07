"""
Data cleaning for Indian market 5-min bars.

Handles:
- Filter to market hours 9:15–15:30 IST
- Drop weekends and NSE/BSE holidays
- Forward-fill missing bars within a session (max 3 consecutive gaps)
- Index symbols (NIFTY / BANKNIFTY / SENSEX): volume is always 0 (no traded volume
  on index feed) — flag bars as volume_synthetic=True but NEVER drop them.
- Remove pre-market and post-market noise
"""
from __future__ import annotations
import logging
from datetime import time as dtime

import pandas as pd

from src.utils import IST, NSE_HOLIDAYS

logger = logging.getLogger(__name__)

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)

# Index symbols that carry no traded volume on the exchange feed.
# Zero-volume bars for these are EXPECTED and must be kept.
INDEX_SYMBOLS = {"NIFTY", "BANKNIFTY", "FINNIFTY", "MIDCPNIFTY", "SENSEX", "BANKEX"}


def clean(
    df: pd.DataFrame,
    symbol: str = "",
    fill_gaps: bool = True,
    max_fill: int = 3,
) -> pd.DataFrame:
    """
    Main entry point. Accepts a DataFrame with DatetimeIndex (IST or UTC).
    Returns cleaned DataFrame with IST tz-aware index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ── Ensure IST timezone ──────────────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)

    # ── Drop weekends ───────────────────────────────────────────────
    df = df[df.index.dayofweek < 5]

    # ── Drop holidays ──────────────────────────────────────────────
    import numpy as np
    df = df[~np.isin(df.index.date, list(NSE_HOLIDAYS))]

    # ── Filter to market hours ─────────────────────────────────────
    times = df.index.time
    df = df[(times >= MARKET_OPEN) & (times < MARKET_CLOSE)]

    # ── Zero-volume handling ───────────────────────────────────────
    # Index symbols (NIFTY, BANKNIFTY, SENSEX …) have no traded volume on
    # the exchange price feed. Zero volume is EXPECTED — keep all bars and
    # mark them so downstream code (e.g. Kronos) can exclude the volume
    # feature column rather than treating it as a signal.
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            is_index = symbol.upper() in INDEX_SYMBOLS
            if is_index:
                logger.debug(
                    "%s: %d bars with zero volume (expected — index feed has no traded volume)",
                    symbol, zero_vol,
                )
                # Mark synthetic so forecaster can drop the volume column
                df["volume_synthetic"] = df["volume"] == 0
            else:
                logger.warning("%s: %d bars with zero volume (unexpected)", symbol, zero_vol)

    # ── Drop bars with NaN OHLC ───────────────────────────────────
    df = df.dropna(subset=["open", "high", "low", "close"])

    # ── Sanity check: high >= low ──────────────────────────────────
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        logger.warning("%s: %d bars with high < low — dropping", symbol, bad_hl.sum())
        df = df[~bad_hl]

    # ── Fill gaps within sessions ─────────────────────────────────
    if fill_gaps:
        df = _fill_intraday_gaps(df, max_fill=max_fill, symbol=symbol)

    df.sort_index(inplace=True)
    return df


def _fill_intraday_gaps(df: pd.DataFrame, max_fill: int, symbol: str) -> pd.DataFrame:
    """
    Forward-fill missing 5-min bars within a trading session.
    Caps fill at max_fill consecutive missing bars (avoids filling across halts).
    """
    if df.empty:
        return df

    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return df
    mode_diff = diffs.mode()
    if mode_diff.empty:
        return df
    resolution = mode_diff.iloc[0]

    reindexed_frames = []
    for day, group in df.groupby(df.index.date):
        start = group.index[0].replace(hour=9, minute=15, second=0, microsecond=0)
        end   = group.index[-1]
        full_idx = pd.date_range(start=start, end=end, freq=resolution, tz=IST)
        group = group.reindex(full_idx)
        group = _capped_ffill(group, max_fill=max_fill)
        reindexed_frames.append(group)

    result = pd.concat(reindexed_frames)
    n_filled = result["close"].isna().sum()
    if n_filled:
        result = result.dropna(subset=["open", "close"])
        logger.debug("%s: dropped %d unfillable gap bars", symbol, n_filled)

    return result


def _capped_ffill(df: pd.DataFrame, max_fill: int) -> pd.DataFrame:
    return df.ffill(limit=max_fill)


# ── Convenience wrappers ────────────────────────────────────────

def get_session_bars(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    mask = df.index.strftime("%Y-%m-%d") == date_str
    return df[mask].copy()


def is_expiry_day(d, symbol: str, config: dict) -> bool:
    expiry_day_name = config["instruments"][symbol].get("expiry_day", "Thursday")
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4,
    }
    return d.weekday() == day_map.get(expiry_day_name, 3)
