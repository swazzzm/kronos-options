"""
Data fetcher — wraps broker API, fetches and caches OHLCV bars.

Key design decisions:
- Fetches 1-min bars from broker, resamples to 5-min locally (more reliable).
- Caches raw CSVs in data/historical/ so API calls are minimised on reruns.
- For backtesting with Zerodha, uses snapshot-based expired option data
  via ZerodhaBrokerWithHistory (see src/broker/zerodha_historical.py).
- For backtesting with Upstox, uses the expired-instruments API as before.
- For live trading, calls fetch_latest_bars().

Usage:
    from src.data_fetcher import DataFetcher
    from src.utils import get_broker
    fetcher = DataFetcher(broker=get_broker())
    df = fetcher.fetch_date_range("NIFTY", "2026-01-01", "2026-06-30")
"""
from __future__ import annotations
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.broker.base import BrokerInterface
from src.utils import IST, is_trading_day, load_config

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/historical")


class DataFetcher:
    def __init__(self, broker: BrokerInterface, config_path: str = "config.yaml"):
        self.broker = broker
        self.cfg = load_config(config_path)
        self._broker_name = self.cfg.get("broker", {}).get("primary", "upstox")
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _instrument_key(self, symbol: str) -> str:
        """Return the broker-appropriate instrument key for index data."""
        inst = self.cfg["instruments"][symbol]
        if self._broker_name == "zerodha":
            # Zerodha historical API uses integer token; return token as string
            token = inst.get("zerodha_instrument_token")
            if token:
                return str(token)
        return inst["upstox_key"]

    def _cache_path(self, symbol: str, resolution: str) -> Path:
        # Separate cache per broker so Upstox and Zerodha data don't mix
        return CACHE_DIR / f"{symbol}_{resolution}_{self._broker_name}.csv"

    # ── Core fetch ────────────────────────────────────────────────────

    def fetch_date_range(
        self,
        symbol: str,
        start: str,
        end: str,
        resolution: str = "5min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for symbol between start and end (inclusive).
        Merges cached data with any new dates to minimise API calls.
        Returns 5-min (or requested resolution) DataFrame, IST tz-aware index.
        """
        cache_path = self._cache_path(symbol, resolution)
        existing = pd.DataFrame()

        if use_cache and cache_path.exists():
            existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not existing.empty:
                existing.index = pd.to_datetime(existing.index, utc=True).tz_convert(IST)

        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

        dates_needed = [
            start_dt + timedelta(days=i)
            for i in range((end_dt - start_dt).days + 1)
            if is_trading_day(start_dt + timedelta(days=i))
        ]

        if not existing.empty:
            cached_dates = set(existing.index.date)
            dates_needed = [d for d in dates_needed if d not in cached_dates]

        if dates_needed:
            logger.info("Fetching %d missing dates for %s via %s", len(dates_needed), symbol, self._broker_name)
            new_frames = []
            for d in dates_needed:
                df_day = self._fetch_one_day(symbol, d.strftime("%Y-%m-%d"), resolution)
                if df_day is not None and not df_day.empty:
                    new_frames.append(df_day)
                time.sleep(self.cfg["broker"].get("request_delay_s", 0.35))

            if new_frames:
                new_data = pd.concat(new_frames).sort_index()
                existing = pd.concat([existing, new_data]).sort_index()
                existing = existing[~existing.index.duplicated(keep="last")]
                existing.to_csv(cache_path)
                logger.info("Cached %d new bars for %s → %s", len(new_data), symbol, cache_path)

        if existing.empty:
            logger.warning("No data returned for %s %s→%s", symbol, start, end)
            return pd.DataFrame()

        mask = (existing.index.date >= start_dt) & (existing.index.date <= end_dt)
        return existing[mask].copy()

    def _fetch_one_day(
        self,
        symbol: str,
        date_str: str,
        resolution: str = "5min",
    ) -> Optional[pd.DataFrame]:
        """Fetch 1-min bars for one day, resample to target resolution."""
        instrument_key = self._instrument_key(symbol)
        try:
            df = self.broker.get_historical_candles(
                instrument_key, interval="1minute",
                from_date=date_str, to_date=date_str,
            )
        except Exception as e:
            logger.error("Failed to fetch %s on %s via %s: %s", symbol, date_str, self._broker_name, e)
            return None

        if df is None or df.empty:
            logger.debug("No data for %s on %s", symbol, date_str)
            return None

        if resolution == "1min":
            return df
        return self._resample(df, resolution)

    @staticmethod
    def _resample(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        """Resample 1-min OHLCV to a coarser bar."""
        rule = resolution
        agg = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }
        if "oi" in df.columns:
            agg["oi"] = "last"
        resampled = df.resample(rule, label="left", closed="left").agg(agg)
        return resampled.dropna(subset=["open", "close"])

    # ── Live bar fetch ────────────────────────────────────────────────────

    def fetch_latest_bars(
        self,
        symbol: str,
        n_bars: int,
        resolution: str = "5min",
    ) -> pd.DataFrame:
        """
        Fetch the most recent n_bars for use in live forecasting.
        Combines cached history with today's live bars.
        """
        cache_path = self._cache_path(symbol, resolution)
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
        else:
            df = pd.DataFrame()

        today = date.today().strftime("%Y-%m-%d")
        today_df = self._fetch_one_day(symbol, today, resolution)
        if today_df is not None and not today_df.empty:
            df = pd.concat([df, today_df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]

        return df.tail(n_bars)

    # ── Option data helpers (for backtester) ────────────────────────────

    def get_option_candle_at(
        self,
        symbol: str,
        date_str: str,
        strike: int,
        option_type: str,
        time_str: str = "09:16",
        expiry: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Fetch the 1-min candle for an option contract at a specific time.
        Used by the backtester to get entry/exit prices from real option data.

        For Zerodha: uses snapshot-based token resolution via ZerodhaBrokerWithHistory.
        For Upstox: uses the expired-instruments API as before.

        Returns dict: {open, high, low, close, volume} or None.
        """
        # Use upstox_key for broker calls that still expect it (Upstox expired API)
        # Use zerodha_instrument_token for Zerodha snapshot resolution
        inst_cfg     = self.cfg["instruments"][symbol]
        instrument_key = inst_cfg["upstox_key"]  # both brokers use this as the logical key

        # Resolve expiry if not provided
        if expiry is None:
            try:
                expiries = self.broker.get_expired_expiries(instrument_key)
                valid = [e for e in expiries if e >= date_str]
                if not valid:
                    logger.warning("No expiry found for %s on %s", symbol, date_str)
                    return None
                expiry = sorted(valid)[0]
            except Exception as e:
                logger.warning("Could not fetch expiries for %s: %s", symbol, e)
                return None

        # Resolve option instrument key / token
        opt_key = self.broker.get_expired_option_key(instrument_key, expiry, strike, option_type)

        if opt_key is None:
            # Try adjacent strikes (ATM may be off by one step)
            atm_step = inst_cfg["atm_step"]
            for adj in [atm_step, -atm_step, atm_step * 2, -atm_step * 2]:
                adj_strike = strike + adj
                opt_key = self.broker.get_expired_option_key(
                    instrument_key, expiry, adj_strike, option_type
                )
                if opt_key:
                    logger.info(
                        "Strike adjusted %s %s: %d → %d",
                        symbol, date_str, strike, adj_strike,
                    )
                    break
                time.sleep(self.cfg["broker"].get("request_delay_s", 0.35))

        if opt_key is None:
            logger.debug(
                "Option key not found: %s %s %s%s expiry=%s — will use BS fallback",
                symbol, date_str, strike, option_type, expiry,
            )
            return None

        # Fetch candles for the day
        try:
            candles = self.broker.get_expired_option_candles(opt_key, "1minute", date_str)
        except Exception as e:
            logger.warning("Error fetching option candles for %s %s: %s", symbol, date_str, e)
            return None

        for c in candles:
            if time_str in str(c[0]):
                return {
                    "timestamp": c[0],
                    "open":      float(c[1]),
                    "high":      float(c[2]),
                    "low":       float(c[3]),
                    "close":     float(c[4]),
                    "volume":    c[5] if len(c) > 5 else 0,
                }
        return None
