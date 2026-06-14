"""
Zerodha Kite Connect broker implementation (STUB).
Swap broker.primary to "zerodha" in config.yaml to use this.

To activate:
  1. pip install kiteconnect
  2. Set ZERODHA_API_KEY, ZERODHA_API_SECRET, ZERODHA_ACCESS_TOKEN in .env
  3. Change broker.primary to "zerodha" in config.yaml
"""
from __future__ import annotations
from typing import Optional
import pandas as pd

from .base import BrokerInterface


class ZerodhaBroker(BrokerInterface):
    """
    Zerodha Kite Connect v3 implementation.
    Currently a STUB — methods raise NotImplementedError.
    Implement when switching from Upstox.
    """

    def __init__(self, api_key: str, access_token: str):
        # Lazy import so kiteconnect isn't required if using Upstox
        try:
            from kiteconnect import KiteConnect
            self.kite = KiteConnect(api_key=api_key)
            self.kite.set_access_token(access_token)
        except ImportError:
            raise ImportError("kiteconnect not installed. Run: pip install kiteconnect")

    def get_historical_candles(self, instrument_key, interval, from_date, to_date) -> pd.DataFrame:
        # Zerodha uses instrument_token (int), not string key.
        # Need to resolve symbol → token via kite.ltp() or instruments CSV first.
        raise NotImplementedError("Zerodha historical candles: TODO")

    def get_expired_expiries(self, instrument_key) -> list[str]:
        raise NotImplementedError("Zerodha expired expiries: TODO — use TrueData or manual CSV")

    def get_expired_option_key(self, instrument_key, expiry, strike, option_type) -> Optional[str]:
        raise NotImplementedError("Zerodha expired option key: TODO")

    def get_expired_option_candles(self, expired_option_key, interval, date_str) -> list:
        raise NotImplementedError("Zerodha expired option candles: TODO")

    def get_option_chain(self, instrument_key, expiry) -> pd.DataFrame:
        raise NotImplementedError("Zerodha option chain: TODO")

    def get_live_quote(self, instrument_key) -> dict:
        raise NotImplementedError("Zerodha live quote: TODO")

    def place_order(self, instrument_key, transaction_type, quantity, order_type, price=0.0, tag="") -> str:
        raise NotImplementedError("Zerodha place order: TODO")

    def get_positions(self) -> pd.DataFrame:
        raise NotImplementedError("Zerodha positions: TODO")

    def get_order_status(self, order_id) -> dict:
        raise NotImplementedError("Zerodha order status: TODO")
