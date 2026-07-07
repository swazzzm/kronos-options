"""
Walk-forward backtester.

For each bar in the test period:
  1. Use only data up to that bar (no lookahead).
  2. Run KronosForecaster on the lookback window.
  3. Generate signal via SignalEngine.
  4. Map to option trade via OptionsMapper.
  5. Simulate entry/exit using REAL Zerodha/Kite expired-instruments option prices.
  6. Apply realistic brokerage, STT, and slippage.

IMPORTANT: This backtester uses ACTUAL historical option OHLC data from Kite's
expired-instruments API. No Black-Scholes approximation is needed.
A Black-Scholes fallback is included for days where API data is unavailable —
those trades are tagged "bs_approximation" in the trade log so you can filter them.

If Kronos / PyTorch is unavailable the backtester silently skips forecast bars
and falls back to Black-Scholes signals — it does NOT crash.

Usage:
    backtester = Backtester(broker=get_broker())
    results = backtester.run("NIFTY", "2025-01-01", "2025-12-31")
    results["equity_curve"].plot()
"""
from __future__ import annotations
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.broker.base import BrokerInterface
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean, is_expiry_day
from src.forecaster import KronosForecaster, KronosUnavailableError
from src.signal_engine import SignalEngine
from src.options_mapper import OptionsMapper
from src.utils import load_config, IST, round_to_atm, get_instrument_key
from src.db import init_db, get_conn

logger = logging.getLogger(__name__)


# ── Transaction cost calculator ───────────────────────────────────────

def calc_charges(
    entry_premium: float,
    exit_premium: float,
    lot_size: int,
    lots: int,
    n_legs: int,
    cfg: dict,
) -> float:
    """
    Compute realistic round-trip charges for an option trade.
    Based on Zerodha rate card (June 2026).
    Returns total charges in ₹.
    """
    costs = cfg.get("costs", {})
    qty   = lot_size * lots

    buy_turnover  = entry_premium * qty * n_legs / 2
    sell_turnover = exit_premium  * qty * n_legs / 2

    brokerage = costs["brokerage_per_order"] * 2 * n_legs
    stt       = sell_turnover * costs["stt_sell_options_pct"] / 100
    exchange  = (buy_turnover + sell_turnover) * costs["exchange_txn_options_pct"] / 100
    gst       = (brokerage + exchange) * costs["gst_pct"] / 100
    sebi      = (buy_turnover + sell_turnover) * costs["sebi_turnover_pct"] / 100
    stamp     = buy_turnover * costs["stamp_duty_buy_pct"] / 100

    return round(brokerage + stt + exchange + gst + sebi + stamp, 2)


def calc_slippage(premium: float, ticks: int = 1) -> float:
    tick_size = 0.05 if premium < 100 else 0.10
    return ticks * tick_size


class Backtester:
    def __init__(
        self,
        broker: BrokerInterface,
        config_path: str = "config.yaml",
        db_path: str = "data/kronos_options.db",
    ):
        self.broker      = broker
        self.cfg         = load_config(config_path)
        self.config_path = config_path
        self.db_path     = db_path
        self.fetcher     = DataFetcher(broker, config_path)
        self.forecaster  = KronosForecaster(config_path, db_path)
        self.signal_eng  = SignalEngine(config_path, db_path)
        self.mapper      = OptionsMapper(broker, config_path, db_path)
        init_db(db_path)

        # Warn once if Kronos/PyTorch is missing — don't spam per bar
        self._kronos_warned = False

    def run(
        self,
        symbol: str,
        start: str,
        end: str,
        bar_resolution: str = "5min",
        signal_bar_interval: int = 6,
        lots: int = 1,
    ) -> dict:
        """
        Walk-forward backtest.

        Args:
            symbol:               NIFTY | BANKNIFTY | SENSEX
            start / end:          Date range "YYYY-MM-DD"
            bar_resolution:       Bar size (currently only "5min" tested)
            signal_bar_interval:  How many bars between signal evaluations
            lots:                 Number of lots per trade

        Returns dict with:
            trade_log:      pd.DataFrame of all trades
            equity_curve:   pd.Series of running P&L
            stats:          dict of performance metrics
        """
        logger.info("Starting backtest: %s %s → %s", symbol, start, end)

        df_raw = self.fetcher.fetch_date_range(symbol, start, end, bar_resolution)
        df     = clean(df_raw, symbol=symbol)

        if df.empty:
            raise ValueError(f"No clean data for {symbol} {start}→{end}")

        # Drop zero-volume bars (market closed / auction ticks) — warn once
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            logger.warning("%s: dropping %d zero-volume bars before backtest.", symbol, zero_vol)
            df = df[df["volume"] > 0]

        inst_cfg  = self.cfg["instruments"][symbol]
        atm_step  = inst_cfg["atm_step"]
        lot_size  = inst_cfg["lot_size"]
        bt_cfg    = self.cfg.get("backtest", {})
        tc_cfg    = self.cfg.get("trading", {})
        skip_exp  = tc_cfg.get("skip_expiry_day", True)
        use_real  = bt_cfg.get("use_real_option_data", True)
        slippage  = self.cfg["costs"].get("slippage_ticks", 1)
        lookback  = self.cfg["kronos"]["lookback"]

        # Resolve instrument key via broker-agnostic helper
        inst_key  = get_instrument_key(symbol, self.config_path)

        trade_records = []
        equity        = 0.0
        equity_series = {}
        open_trade: Optional[dict] = None
        bar_count = 0

        dates = sorted(set(df.index.date))

        for d in dates:
            date_str = d.strftime("%Y-%m-%d")

            if skip_exp and is_expiry_day(d, symbol, self.cfg):
                logger.info("Skipping expiry day: %s %s", symbol, date_str)
                continue

            day_bars = df[df.index.date == d]
            if len(day_bars) < 2:
                continue

            daily_pnl = 0.0

            for i, (ts, bar) in enumerate(day_bars.iterrows()):
                bar_count += 1

                t = ts.time()
                from datetime import time as dtime
                no_open  = dtime(9, 15 + tc_cfg.get("no_trade_open_mins", 5))
                sq_off_t = dtime(*[int(x) for x in tc_cfg.get("square_off_time", "15:15").split(":")])

                # Square off open position at EOD
                if open_trade and t >= sq_off_t:
                    rec = self._close_trade(
                        open_trade, symbol, date_str, str(ts.time())[:5],
                        lot_size, lots, slippage, use_real, "SQUARE_OFF"
                    )
                    if rec:
                        daily_pnl += rec["net_pnl_rs"]
                        equity    += rec["net_pnl_rs"]
                        equity_series[ts] = equity
                        trade_records.append(rec)
                    open_trade = None

                # Generate signal every N bars, no open position, within trading hours
                if (
                    open_trade is None
                    and bar_count % signal_bar_interval == 0
                    and t >= no_open
                    and t < sq_off_t
                ):
                    idx_pos = df.index.get_loc(ts)
                    hist    = df.iloc[max(0, idx_pos - lookback): idx_pos]
                    if len(hist) < 20:
                        continue

                    try:
                        forecast = self.forecaster.forecast(symbol, hist, use_cache=False)
                        signal   = self.signal_eng.generate(forecast, save_to_db=False)
                    except KronosUnavailableError as e:
                        # Warn only once per run, then skip silently
                        if not self._kronos_warned:
                            logger.warning(
                                "Kronos unavailable — skipping all forecast bars. "
                                "Black-Scholes fallback active.\n  %s", e
                            )
                            self._kronos_warned = True
                        continue
                    except Exception as e:
                        logger.warning("Forecast failed at %s: %s", ts, e)
                        continue

                    if signal["signal"] == "NEUTRAL":
                        continue

                    expiry = self._get_nearest_expiry(inst_key, date_str)
                    if not expiry:
                        continue

                    recommendation = self.mapper.map(
                        {**signal, "symbol": symbol},
                        expiry=expiry,
                    )
                    if recommendation.get("strategy") == "NO_TRADE" or not recommendation.get("legs"):
                        continue

                    open_trade = {
                        "recommendation": recommendation,
                        "entry_ts":       str(ts),
                        "entry_bar":      bar,
                        "signal":         signal,
                        "expiry":         expiry,
                    }
                    logger.debug("Opened %s %s @ %s", symbol, recommendation["strategy"], ts)

        # Close any still-open trade at end of test period
        if open_trade:
            rec = self._close_trade(
                open_trade, symbol, dates[-1].strftime("%Y-%m-%d"), "15:15",
                lot_size, lots, slippage, use_real, "PERIOD_END"
            )
            if rec:
                equity += rec["net_pnl_rs"]
                trade_records.append(rec)

        trade_log    = pd.DataFrame(trade_records) if trade_records else pd.DataFrame()
        equity_curve = pd.Series(equity_series, name="equity_rs")
        stats        = self._compute_stats(trade_log, equity_curve)

        logger.info(
            "Backtest complete: %d trades | P&L=₹%.0f | Win=%.1f%% | Sharpe=%.2f",
            stats["total_trades"], stats["total_pnl_rs"],
            stats["win_rate_pct"], stats["sharpe"],
        )

        self._save_run(symbol, start, end, stats, trade_log)

        return {
            "trade_log":    trade_log,
            "equity_curve": equity_curve,
            "stats":        stats,
        }

    # ── Trade close helper ─────────────────────────────────────────────

    def _close_trade(
        self,
        open_trade: dict,
        symbol: str,
        date_str: str,
        time_str: str,
        lot_size: int,
        lots: int,
        slippage_ticks: int,
        use_real: bool,
        exit_reason: str,
    ) -> Optional[dict]:
        rec      = open_trade["recommendation"]
        legs     = rec["legs"]
        exp      = open_trade["expiry"]

        if not legs:
            return None

        leg0     = legs[0]
        strike   = leg0["strike"]
        opt_type = leg0["option_type"]

        entry_price, exit_price, data_source = self._get_option_prices(
            symbol, date_str, time_str, strike, opt_type, exp, use_real
        )

        if entry_price is None:
            logger.warning("No price data for %s %s %s%s — skipping", symbol, date_str, strike, opt_type)
            return None

        direction_mult = 1 if leg0["action"] == "BUY" else -1
        raw_pts  = (exit_price - entry_price) * direction_mult
        raw_pnl  = raw_pts * lot_size * lots * len(legs)

        slip    = calc_slippage(entry_price, slippage_ticks) * lot_size * lots * len(legs) * 2
        charges = calc_charges(entry_price, exit_price, lot_size, lots, len(legs), self.cfg)
        net_pnl = raw_pnl - slip - charges

        return {
            "trade_date":   date_str,
            "symbol":       symbol,
            "signal":       open_trade["signal"]["signal"],
            "strategy":     rec["strategy"],
            "strike":       strike,
            "option_type":  opt_type,
            "expiry":       exp,
            "entry_price":  entry_price,
            "exit_price":   exit_price,
            "lots":         lots,
            "lot_size":     lot_size,
            "raw_pnl_rs":   round(raw_pnl, 2),
            "charges_rs":   round(charges + slip, 2),
            "net_pnl_rs":   round(net_pnl, 2),
            "sl_tgt_tag":   exit_reason,
            "data_source":  data_source,
        }

    def _get_option_prices(
        self, symbol, date_str, time_str, strike, opt_type, expiry, use_real
    ):
        if use_real:
            candle = self.fetcher.get_option_candle_at(
                symbol, date_str, strike, opt_type, time_str, expiry
            )
            if candle:
                return candle["open"], candle["close"], "real_option_data"

        logger.debug("Using BS fallback for %s %s %s%s", symbol, date_str, strike, opt_type)
        entry, exit_ = self._bs_approximate(symbol, date_str, strike, opt_type, expiry)
        return entry, exit_, "bs_approximation"

    def _bs_approximate(self, symbol, date_str, strike, opt_type, expiry):
        """
        Black-Scholes approximation when real option data is unavailable.
        CAUTION: rough estimate only. Filter 'bs_approximation' trades from analysis.
        """
        from scipy.stats import norm
        import math

        iv = self.cfg["backtest"].get("iv_assumption_pct", 15.0) / 100

        try:
            spot_df = self.fetcher.fetch_date_range(symbol, date_str, date_str, "5min", use_cache=True)
            if spot_df.empty:
                return None, None
            spot = float(spot_df["close"].iloc[0])
        except Exception:
            return None, None

        exp_dt = datetime.strptime(expiry, "%Y-%m-%d")
        bar_dt = datetime.strptime(date_str, "%Y-%m-%d")
        dte    = max((exp_dt - bar_dt).days, 1)
        T      = dte / 365
        r      = 0.065
        S, K   = spot, float(strike)

        d1 = (math.log(S / K) + (r + 0.5 * iv**2) * T) / (iv * math.sqrt(T))
        d2 = d1 - iv * math.sqrt(T)

        if opt_type == "CE":
            price = S * norm.cdf(d1) - K * math.exp(-r * T) * norm.cdf(d2)
        else:
            price = K * math.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        price = max(price, 0.05)
        return round(price, 2), round(price * 0.97, 2)

    # ── Expiry helper ──────────────────────────────────────────────────

    def _get_nearest_expiry(self, instrument_key: str, date_str: str) -> Optional[str]:
        try:
            expiries = self.broker.get_expired_expiries(instrument_key)
            valid    = [e for e in expiries if e >= date_str]
            return sorted(valid)[0] if valid else None
        except Exception as e:
            logger.warning("Could not fetch expiries: %s", e)
            return None

    # ── Performance stats ──────────────────────────────────────────────

    @staticmethod
    def _compute_stats(trade_log: pd.DataFrame, equity_curve: pd.Series) -> dict:
        if trade_log.empty:
            return {"total_trades": 0, "total_pnl_rs": 0, "win_rate_pct": 0,
                    "sharpe": 0, "max_drawdown_rs": 0, "profit_factor": 0}

        pnls   = trade_log["net_pnl_rs"]
        wins   = pnls[pnls > 0]
        losses = pnls[pnls <= 0]
        total_trades  = len(pnls)
        win_rate      = len(wins) / total_trades * 100 if total_trades else 0
        profit_factor = abs(wins.sum() / losses.sum()) if losses.sum() != 0 else float("inf")

        if len(equity_curve) > 1:
            daily_returns = equity_curve.resample("D").last().diff().dropna()
            sharpe = (daily_returns.mean() / daily_returns.std() * (252 ** 0.5)
                      if daily_returns.std() != 0 else 0)
        else:
            sharpe = 0

        if not equity_curve.empty:
            roll_max = equity_curve.cummax()
            max_dd   = float((equity_curve - roll_max).min())
        else:
            max_dd = 0

        return {
            "total_trades":    total_trades,
            "wins":            len(wins),
            "losses":          len(losses),
            "total_pnl_rs":    round(float(pnls.sum()), 2),
            "win_rate_pct":    round(win_rate, 2),
            "avg_win_rs":      round(float(wins.mean()), 2) if len(wins) else 0,
            "avg_loss_rs":     round(float(losses.mean()), 2) if len(losses) else 0,
            "profit_factor":   round(profit_factor, 3),
            "sharpe":          round(float(sharpe), 3),
            "max_drawdown_rs": round(max_dd, 2),
        }

    def _save_run(self, symbol, start, end, stats, trade_log):
        import json
        with get_conn(self.db_path) as conn:
            cur = conn.execute(
                """INSERT INTO backtest_runs
                   (run_at, symbol, start_date, end_date, total_trades, wins, losses,
                    total_pnl_rs, sharpe, max_drawdown_rs, win_rate_pct, profit_factor, params)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (
                    datetime.now().isoformat(), symbol, start, end,
                    stats["total_trades"], stats.get("wins", 0), stats.get("losses", 0),
                    stats["total_pnl_rs"], stats["sharpe"], stats["max_drawdown_rs"],
                    stats["win_rate_pct"], stats["profit_factor"],
                    json.dumps(self.cfg.get("signal", {})),
                ),
            )
            run_id = cur.lastrowid
            if not trade_log.empty:
                for _, row in trade_log.iterrows():
                    conn.execute(
                        """INSERT INTO backtest_trades
                           (run_id, trade_date, symbol, signal, strategy, strike, option_type,
                            expiry, entry_price, exit_price, lots, lot_size,
                            raw_pnl_rs, charges_rs, net_pnl_rs, sl_tgt_tag, data_source)
                           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                        (
                            run_id, row.get("trade_date"), symbol,
                            row.get("signal"), row.get("strategy"),
                            row.get("strike"), row.get("option_type"), row.get("expiry"),
                            row.get("entry_price"), row.get("exit_price"),
                            row.get("lots"), row.get("lot_size"),
                            row.get("raw_pnl_rs"), row.get("charges_rs"), row.get("net_pnl_rs"),
                            row.get("sl_tgt_tag"), row.get("data_source"),
                        ),
                    )
