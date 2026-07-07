"""
Zerodha Kite Connect v3 broker implementation.
Swap broker.primary to "zerodha" in config.yaml to activate.

Prerequisites:
  1. pip install kiteconnect pyotp
  2. Set in .env:
       ZERODHA_API_KEY=your_api_key
       ZERODHA_API_SECRET=your_api_secret
       ZERODHA_ACCESS_TOKEN=your_access_token   # refreshed daily via zerodha_auth.py

Note on instrument tokens:
  Zerodha uses integer instrument_token, not string keys.
  All methods accept either the string symbol (e.g. "NIFTY") or an integer token.
  Option chain lookup returns instrument_token in each row for order placement.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from .base import BrokerInterface

logger = logging.getLogger(__name__)

# Zerodha exchange + segment constants
_EXCHANGE_NFO = "NFO"       # options & futures
_EXCHANGE_NSE = "NSE"       # equity
_INDEX_TOKEN_MAP = {
    "NIFTY":     256265,
    "BANKNIFTY": 260105,
    "SENSEX":    265,
}


class ZerodhaBroker(BrokerInterface):
    """
    Full Zerodha Kite Connect v3 implementation.
    All methods match the BrokerInterface contract defined in base.py.
    """

    def __init__(self, api_key: str, access_token: str):
        try:
            from kiteconnect import KiteConnect
        except ImportError:
            raise ImportError("Run: pip install kiteconnect")

        self.kite = KiteConnect(api_key=api_key)
        self.kite.set_access_token(access_token)
        self._instruments_cache: Optional[pd.DataFrame] = None
        self._instruments_fetched_date: Optional[str] = None
        logger.info("ZerodhaBroker initialised with api_key=%s...", api_key[:6])

    # ── Instruments cache ──────────────────────────────────────────────

    def _get_instruments(self, exchange: str = "NFO") -> pd.DataFrame:
        """Fetch and cache NFO instruments list (refreshed once per trading day)."""
        today = datetime.now().strftime("%Y-%m-%d")
        if self._instruments_cache is not None and self._instruments_fetched_date == today:
            return self._instruments_cache

        logger.info("Fetching instruments list from Zerodha (exchange=%s)...", exchange)
        raw = self.kite.instruments(exchange=exchange)
        df = pd.DataFrame(raw)
        self._instruments_cache = df
        self._instruments_fetched_date = today
        return df

    def _resolve_token(self, symbol: str) -> int:
        """Resolve index symbol to Zerodha instrument_token."""
        symbol = symbol.upper()
        if symbol in _INDEX_TOKEN_MAP:
            return _INDEX_TOKEN_MAP[symbol]
        # Try NSE equity lookup
        instruments = self._get_instruments("NSE")
        row = instruments[instruments["tradingsymbol"] == symbol]
        if row.empty:
            raise ValueError(f"Cannot resolve instrument token for symbol: {symbol}")
        return int(row.iloc[0]["instrument_token"])

    # ── Market data ────────────────────────────────────────────────────

    def get_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV candles from Zerodha historical data API.
        instrument_key: symbol string (e.g. 'NIFTY') or integer token as string.
        interval: '1minute' | '5minute' | '15minute' | '30minute' | '60minute' | 'day'
        """
        try:
            token = int(instrument_key)
        except ValueError:
            token = self._resolve_token(instrument_key)

        # Kite interval mapping
        interval_map = {
            "1minute":  "minute",
            "5minute":  "5minute",
            "15minute": "15minute",
            "30minute": "30minute",
            "60minute": "60minute",
            "day":      "day",
        }
        kite_interval = interval_map.get(interval, interval)

        data = self.kite.historical_data(
            instrument_token=token,
            from_date=from_date,
            to_date=to_date,
            interval=kite_interval,
            continuous=False,
            oi=False,
        )
        df = pd.DataFrame(data)
        if df.empty:
            return df
        df.rename(columns={"date": "timestamp"}, inplace=True)
        df.set_index("timestamp", inplace=True)
        df.index = pd.to_datetime(df.index, utc=False)
        return df[["open", "high", "low", "close", "volume"]]

    def get_expired_expiries(self, instrument_key: str) -> list[str]:
        """
        Return sorted list of past weekly expiry dates for NIFTY/BANKNIFTY.
        Uses NFO instruments CSV to find distinct expiry dates for options contracts.
        """
        symbol = instrument_key.upper().replace("_INDEX", "")
        instruments = self._get_instruments("NFO")
        options = instruments[
            (instruments["name"] == symbol) &
            (instruments["instrument_type"].isin(["CE", "PE"]))
        ]
        if options.empty:
            logger.warning("No instruments found for %s in NFO", symbol)
            return []

        today = datetime.now().date()
        options = options.copy()
        options["expiry"] = pd.to_datetime(options["expiry"])
        past = options[options["expiry"].dt.date < today]
        expiries = sorted(past["expiry"].dt.strftime("%Y-%m-%d").unique().tolist())
        return expiries

    def get_expired_option_key(
        self,
        instrument_key: str,
        expiry: str,
        strike: int,
        option_type: str,
    ) -> Optional[str]:
        """
        Return instrument_token (as string) for an expired option contract.
        Returns None if not found in instruments cache.
        """
        symbol = instrument_key.upper().replace("_INDEX", "")
        instruments = self._get_instruments("NFO")
        mask = (
            (instruments["name"] == symbol) &
            (instruments["instrument_type"] == option_type.upper()) &
            (instruments["strike"] == float(strike)) &
            (pd.to_datetime(instruments["expiry"]).dt.strftime("%Y-%m-%d") == expiry)
        )
        matched = instruments[mask]
        if matched.empty:
            return None
        return str(int(matched.iloc[0]["instrument_token"]))

    def get_expired_option_candles(
        self,
        expired_option_key: str,
        interval: str,
        date_str: str,
    ) -> list:
        """Return raw candle list for an expired option on a given date."""
        df = self.get_historical_candles(
            instrument_key=expired_option_key,
            interval=interval,
            from_date=date_str,
            to_date=date_str,
        )
        if df.empty:
            return []
        df.reset_index(inplace=True)
        return df.to_dict(orient="records")

    def get_option_chain(
        self,
        instrument_key: str,
        expiry: str,
    ) -> pd.DataFrame:
        """
        Return live option chain for given symbol and expiry.
        Columns: strike, instrument_type, instrument_token, ltp, iv, oi, volume
        """
        symbol = instrument_key.upper().replace("_INDEX", "")
        instruments = self._get_instruments("NFO")
        mask = (
            (instruments["name"] == symbol) &
            (instruments["instrument_type"].isin(["CE", "PE"])) &
            (pd.to_datetime(instruments["expiry"]).dt.strftime("%Y-%m-%d") == expiry)
        )
        chain_instruments = instruments[mask].copy()
        if chain_instruments.empty:
            logger.warning("No option chain data for %s expiry=%s", symbol, expiry)
            return pd.DataFrame()

        tokens = chain_instruments["instrument_token"].astype(int).tolist()
        # Zerodha quote accepts max 500 instruments per call
        quotes = {}
        for i in range(0, len(tokens), 500):
            batch = tokens[i:i+500]
            q = self.kite.quote(instruments=batch)
            quotes.update(q)

        rows = []
        for _, row in chain_instruments.iterrows():
            token = str(int(row["instrument_token"]))
            q = quotes.get(token, {})
            rows.append({
                "strike":           row["strike"],
                "instrument_type":  row["instrument_type"],
                "instrument_token": int(row["instrument_token"]),
                "tradingsymbol":    row["tradingsymbol"],
                "ltp":              q.get("last_price", 0.0),
                "iv":               q.get("implied_volatility", 0.0),
                "oi":               q.get("oi", 0),
                "volume":           q.get("volume", 0),
            })

        return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)

    def get_live_quote(self, instrument_key: str) -> dict:
        """Return latest quote for a symbol or instrument token."""
        try:
            token = int(instrument_key)
        except ValueError:
            token = self._resolve_token(instrument_key)

        quotes = self.kite.quote(instruments=[token])
        q = quotes.get(str(token), {})
        return {
            "ltp":    q.get("last_price", 0.0),
            "open":   q.get("ohlc", {}).get("open", 0.0),
            "high":   q.get("ohlc", {}).get("high", 0.0),
            "low":    q.get("ohlc", {}).get("low", 0.0),
            "close":  q.get("ohlc", {}).get("close", 0.0),
            "volume": q.get("volume", 0),
        }

    # ── Order management ──────────────────────────────────────────────

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
        Place order on Zerodha via Kite Connect.
        instrument_key: integer instrument_token as string (from get_option_chain).
        transaction_type: "BUY" or "SELL".
        order_type: "MARKET" or "LIMIT".
        Returns Zerodha order_id as string.
        """
        from kiteconnect import KiteConnect

        kite_txn = (
            self.kite.TRANSACTION_TYPE_BUY
            if transaction_type.upper() == "BUY"
            else self.kite.TRANSACTION_TYPE_SELL
        )
        kite_order_type = (
            self.kite.ORDER_TYPE_MARKET
            if order_type.upper() == "MARKET"
            else self.kite.ORDER_TYPE_LIMIT
        )

        # Resolve tradingsymbol from instrument_token for NFO orders
        instruments = self._get_instruments("NFO")
        row = instruments[instruments["instrument_token"] == int(instrument_key)]
        if row.empty:
            raise ValueError(f"Instrument token {instrument_key} not found in NFO instruments")
        tradingsymbol = row.iloc[0]["tradingsymbol"]

        order_params = {
            "tradingsymbol":    tradingsymbol,
            "exchange":         _EXCHANGE_NFO,
            "transaction_type": kite_txn,
            "quantity":         quantity,
            "order_type":       kite_order_type,
            "product":          self.kite.PRODUCT_MIS,  # intraday margin
            "validity":         self.kite.VALIDITY_DAY,
            "tag":              tag[:20] if tag else "",  # Zerodha tag limit: 20 chars
        }
        if order_type.upper() == "LIMIT" and price > 0:
            order_params["price"] = price

        logger.info("Placing order: %s", order_params)
        order_id = self.kite.place_order(
            variety=self.kite.VARIETY_REGULAR,
            **order_params,
        )
        logger.info("Order placed. order_id=%s", order_id)
        return str(order_id)

    def get_positions(self) -> pd.DataFrame:
        """Return current day + net positions as a combined DataFrame."""
        positions = self.kite.positions()
        day_pos = positions.get("day", [])
        net_pos = positions.get("net", [])
        all_pos = day_pos + net_pos
        if not all_pos:
            return pd.DataFrame()
        return pd.DataFrame(all_pos)

    def get_order_status(self, order_id: str) -> dict:
        """Return full order detail dict for given order_id."""
        orders = self.kite.orders()
        for order in orders:
            if str(order.get("order_id")) == str(order_id):
                return order
        return {}

    # ── Margin helpers ────────────────────────────────────────────────

    def get_available_margin(self) -> float:
        """
        Return available cash margin for F&O segment (in ₹).
        Used by RiskManager to enforce ₹3L capital cap.
        """
        margins = self.kite.margins(segment="equity")
        fno = self.kite.margins(segment="commodity")  # F&O is under commodity segment in Kite
        # Zerodha returns margins per segment; use net available for F&O
        try:
            fno_margins = self.kite.margins()
            return float(fno_margins.get("equity", {}).get("available", {}).get("cash", 0.0))
        except Exception as e:
            logger.warning("Could not fetch F&O margins: %s", e)
            return 0.0
