"""
Upstox v2 broker implementation.

API reference: https://upstox.com/developer/api-documentation/
All endpoints require Bearer token in Authorization header.
Token is valid for one trading day — refresh daily via login flow.

Key instrument key formats:
  NSE index:  "NSE_INDEX|Nifty 50"
  BSE index:  "BSE_INDEX|SENSEX"
  NSE equity: "NSE_EQ|INFY"
  NSE option: "NSE_FO|NIFTY25JUN24000CE" (use expired-instruments API to look up)
"""
from __future__ import annotations
import os
import time
import urllib.parse
import logging
from typing import Optional

import requests
import pandas as pd
from tenacity import retry, stop_after_attempt, wait_fixed

from .base import BrokerInterface

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upstox.com/v2"


class UpstoxBroker(BrokerInterface):
    """
    Upstox v2 REST API implementation.

    Usage:
        broker = UpstoxBroker(access_token=os.getenv("UPSTOX_ACCESS_TOKEN"))
    """

    def __init__(
        self,
        access_token: str,
        request_delay_s: float = 0.35,
        timeout_s: int = 10,
        max_retries: int = 3,
    ):
        self.access_token = access_token
        self.delay = request_delay_s
        self.timeout = timeout_s
        self.max_retries = max_retries

    # ── Internal helpers ───────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    def _get(self, url: str) -> dict:
        """GET with retry and rate-limit delay. Returns parsed JSON data dict."""
        time.sleep(self.delay)
        for attempt in range(self.max_retries):
            try:
                r = requests.get(url, headers=self._headers(), timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                logger.warning("GET %s → HTTP %s (attempt %d)", url, r.status_code, attempt + 1)
            except requests.RequestException as e:
                logger.warning("GET %s failed: %s (attempt %d)", url, e, attempt + 1)
            time.sleep(1.0 * (attempt + 1))
        logger.error("All retries exhausted for %s", url)
        return {}

    @staticmethod
    def _encode(instrument_key: str) -> str:
        return urllib.parse.quote(instrument_key, safe="")

    # ── Market data ────────────────────────────────────────────────────

    def get_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles.
        interval: "1minute" | "30minute" | "day" | "week" | "month"
        Returns DataFrame with columns [open, high, low, close, volume, oi]
        and a DatetimeIndex in IST timezone.
        """
        encoded = self._encode(instrument_key)
        url = f"{BASE_URL}/historical-candle/{encoded}/{interval}/{to_date}/{from_date}"
        data = self._get(url)
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            logger.warning("No candles returned for %s %s %s→%s", instrument_key, interval, from_date, to_date)
            return pd.DataFrame()

        df = pd.DataFrame(candles, columns=["timestamp", "open", "high", "low", "close", "volume", "oi"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        df.set_index("timestamp", inplace=True)
        df = df.sort_index()
        return df

    def get_expired_expiries(self, instrument_key: str) -> list[str]:
        """Return list of past expiry dates as 'YYYY-MM-DD' strings, sorted ascending."""
        encoded = self._encode(instrument_key)
        url = f"{BASE_URL}/expired-instruments/expiries?instrument_key={encoded}"
        data = self._get(url)
        expiries = data.get("data", [])
        return sorted(expiries)

    def get_expired_option_key(
        self,
        instrument_key: str,
        expiry: str,
        strike: int,
        option_type: str,
    ) -> Optional[str]:
        """
        Find the Upstox instrument key for an expired option contract.
        Searches by strike and option_type (CE/PE) among all contracts for given expiry.
        Returns None if not found.
        """
        encoded = self._encode(instrument_key)
        url = (
            f"{BASE_URL}/expired-instruments/option/contract"
            f"?instrument_key={encoded}&expiry_date={expiry}"
        )
        data = self._get(url)
        for contract in data.get("data", []):
            sp = float(contract.get("strike_price") or contract.get("strikePrice", -1))
            itype = contract.get("instrument_type") or contract.get("instrumentType", "")
            if sp == float(strike) and itype == option_type:
                return contract.get("instrument_key") or contract.get("instrumentKey")
        return None

    def get_expired_option_candles(
        self,
        expired_option_key: str,
        interval: str,
        date_str: str,
    ) -> list:
        """
        Fetch raw candles for an expired option on a specific date.
        Returns raw list of [timestamp, open, high, low, close, volume, oi].
        """
        encoded = self._encode(expired_option_key)
        url = (
            f"{BASE_URL}/expired-instruments/historical-candle"
            f"/{encoded}/{interval}/{date_str}/{date_str}"
        )
        data = self._get(url)
        return data.get("data", {}).get("candles", [])

    def get_option_chain(self, instrument_key: str, expiry: str) -> pd.DataFrame:
        """
        Fetch live option chain for given expiry.
        Returns DataFrame with columns: strike, CE_ltp, CE_iv, CE_oi, PE_ltp, PE_iv, PE_oi
        NOTE: Only works for non-expired (current) expiries.
        """
        encoded = self._encode(instrument_key)
        url = f"{BASE_URL}/option/chain?instrument_key={encoded}&expiry_date={expiry}"
        data = self._get(url)
        rows = []
        for item in data.get("data", []):
            row = {
                "strike": item.get("strike_price"),
                "CE_ltp": item.get("call_options", {}).get("market_data", {}).get("ltp"),
                "CE_iv":  item.get("call_options", {}).get("option_greeks", {}).get("iv"),
                "CE_oi":  item.get("call_options", {}).get("market_data", {}).get("oi"),
                "PE_ltp": item.get("put_options", {}).get("market_data", {}).get("ltp"),
                "PE_iv":  item.get("put_options", {}).get("option_greeks", {}).get("iv"),
                "PE_oi":  item.get("put_options", {}).get("market_data", {}).get("oi"),
            }
            rows.append(row)
        return pd.DataFrame(rows)

    def get_live_quote(self, instrument_key: str) -> dict:
        """Fetch latest LTP and OHLCV for an instrument."""
        encoded = self._encode(instrument_key)
        url = f"{BASE_URL}/market-quote/quotes?instrument_key={encoded}"
        data = self._get(url)
        quotes = data.get("data", {})
        # Upstox returns dict keyed by instrument_key
        for key, val in quotes.items():
            return {
                "ltp":    val.get("last_price"),
                "open":   val.get("ohlc", {}).get("open"),
                "high":   val.get("ohlc", {}).get("high"),
                "low":    val.get("ohlc", {}).get("low"),
                "close":  val.get("ohlc", {}).get("close"),
                "volume": val.get("volume"),
            }
        return {}

    # ── Order management ───────────────────────────────────────────────

    def place_order(
        self,
        instrument_key: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0.0,
        tag: str = "",
    ) -> str:
        """
        Place a real order via Upstox.
        ONLY called by live_trader.py — paper_trader.py logs without calling this.
        """
        payload = {
            "quantity": quantity,
            "product": "D",                    # Intraday (MIS equivalent)
            "validity": "DAY",
            "price": price,
            "tag": tag,
            "instrument_token": instrument_key,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
        }
        r = requests.post(
            f"{BASE_URL}/order/place",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code == 200:
            return r.json().get("data", {}).get("order_id", "")
        raise RuntimeError(f"Order placement failed: {r.status_code} {r.text}")

    def get_positions(self) -> pd.DataFrame:
        data = self._get(f"{BASE_URL}/portfolio/short-term-positions")
        return pd.DataFrame(data.get("data", []))

    def get_order_status(self, order_id: str) -> dict:
        data = self._get(f"{BASE_URL}/order/details?order_id={order_id}")
        return data.get("data", {})
