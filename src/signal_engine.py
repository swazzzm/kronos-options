"""
Signal engine — converts Kronos forecast distribution into a trading signal.

Inputs:  forecast dict (from forecaster.py) + optional live IV
Outputs: signal dict with:
  signal:             STRONG_BULLISH | BULLISH | NEUTRAL | BEARISH | STRONG_BEARISH
  confidence:         0.0–1.0
  expected_move_pct:  % move forecast
  prob_up:            fraction of paths ending higher
  expected_range:     (p10_close, p90_close) — 10th/90th percentile
  dispersion_regime:  "high" | "low" | "medium"
  iv_regime:          "high" | "low" | "medium" | None (if no IV available)
"""
from __future__ import annotations
import logging
from typing import Optional

from src.db import save_signal
from src.utils import load_config, now_ist

logger = logging.getLogger(__name__)


class SignalEngine:
    def __init__(self, config_path: str = "config.yaml", db_path: str = "data/kronos_options.db"):
        self.cfg = load_config(config_path).get("signal", {})
        self.db_path = db_path

    def generate(
        self,
        forecast: dict,
        live_iv: Optional[float] = None,
        save_to_db: bool = True,
    ) -> dict:
        """
        Convert a forecast dict into a signal.

        Args:
            forecast:  Output of KronosForecaster.forecast()
            live_iv:   Current implied volatility (%) from option chain, or None
            save_to_db: Persist signal to SQLite

        Returns signal dict.
        """
        prob_up    = forecast["prob_up"]
        move_pct   = forecast["expected_move_pct"]
        dispersion = forecast["path_dispersion"]
        current_close = forecast["current_close"]
        expected_range = forecast.get("expected_range", (None, None))

        # ── Directional signal ─────────────────────────────────────────
        signal = self._classify_direction(prob_up, move_pct)

        # ── Confidence score ───────────────────────────────────────────
        # Combines directional conviction (prob_up distance from 0.5) with
        # magnitude of expected move.
        conviction = abs(prob_up - 0.5) * 2           # 0 → random, 1 → certain
        magnitude_score = min(abs(move_pct) / 1.0, 1.0)  # capped at 1% move → 1.0
        confidence = round(0.6 * conviction + 0.4 * magnitude_score, 3)

        # ── Dispersion regime ──────────────────────────────────────────
        high_disp = self.cfg.get("high_dispersion_pct", 0.40)
        low_disp  = self.cfg.get("low_dispersion_pct", 0.20)
        if dispersion >= high_disp:
            dispersion_regime = "high"
        elif dispersion <= low_disp:
            dispersion_regime = "low"
        else:
            dispersion_regime = "medium"

        # ── IV regime ──────────────────────────────────────────────────
        iv_regime = None
        if live_iv is not None:
            high_iv = self.cfg.get("high_iv_threshold", 20.0)
            low_iv  = self.cfg.get("low_iv_threshold", 12.0)
            if live_iv >= high_iv:
                iv_regime = "high"
            elif live_iv <= low_iv:
                iv_regime = "low"
            else:
                iv_regime = "medium"

        result = {
            "symbol":             forecast["symbol"],
            "signal_time":        now_ist().isoformat(),
            "signal":             signal,
            "confidence":         confidence,
            "expected_move_pct":  round(move_pct, 4),
            "prob_up":            round(prob_up, 4),
            "expected_range_low": expected_range[0],
            "expected_range_high": expected_range[1],
            "current_close":      current_close,
            "dispersion_regime":  dispersion_regime,
            "iv_regime":          iv_regime,
            "live_iv":            live_iv,
            "forecast_id":        forecast.get("forecast_id"),
        }

        if save_to_db:
            sig_id = save_signal(result, db_path=self.db_path)
            result["signal_id"] = sig_id

        logger.info(
            "%s signal: %s (confidence=%.2f, move=%.3f%%, prob_up=%.2f, dispersion=%s)",
            forecast["symbol"], signal, confidence, move_pct, prob_up, dispersion_regime,
        )
        return result

    def _classify_direction(self, prob_up: float, move_pct: float) -> str:
        """
        Classify signal based on probability and magnitude thresholds from config.

        The NEUTRAL zone is wide by design — we'd rather miss a trade than force one.
        Both prob_up AND move_pct must agree before going directional.
        """
        su_prob = self.cfg.get("prob_up_strong", 0.75)
        d_prob  = self.cfg.get("prob_up_directional", 0.65)
        sd_prob = self.cfg.get("prob_down_strong", 0.25)
        db_prob = self.cfg.get("prob_down_directional", 0.35)

        su_move = self.cfg.get("strong_bullish_move_pct", 0.50)
        b_move  = self.cfg.get("bullish_move_pct", 0.25)
        be_move = self.cfg.get("bearish_move_pct", -0.25)
        sb_move = self.cfg.get("strong_bearish_move_pct", -0.50)

        if prob_up >= su_prob and move_pct >= su_move:
            return "STRONG_BULLISH"
        elif prob_up >= d_prob and move_pct >= b_move:
            return "BULLISH"
        elif prob_up <= sd_prob and move_pct <= sb_move:
            return "STRONG_BEARISH"
        elif prob_up <= db_prob and move_pct <= be_move:
            return "BEARISH"
        else:
            return "NEUTRAL"
