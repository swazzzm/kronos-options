"""Tests for signal_engine.py — signal classification and confidence."""
import pytest
from unittest.mock import patch
from src.signal_engine import SignalEngine


def make_forecast(prob_up, move_pct, dispersion=0.3, symbol="NIFTY"):
    return {
        "symbol": symbol,
        "prob_up": prob_up,
        "expected_move_pct": move_pct,
        "path_dispersion": dispersion,
        "current_close": 24000.0,
        "expected_range": (23900.0, 24200.0),
        "expected_close": 24000.0 * (1 + move_pct / 100),
        "forecast_id": None,
    }


@pytest.fixture
def engine():
    return SignalEngine()


def test_strong_bullish(engine):
    sig = engine.generate(make_forecast(0.80, 0.60), save_to_db=False)
    assert sig["signal"] == "STRONG_BULLISH"
    assert sig["confidence"] > 0.5


def test_bullish(engine):
    sig = engine.generate(make_forecast(0.68, 0.30), save_to_db=False)
    assert sig["signal"] == "BULLISH"


def test_neutral(engine):
    sig = engine.generate(make_forecast(0.52, 0.10), save_to_db=False)
    assert sig["signal"] == "NEUTRAL"


def test_bearish(engine):
    sig = engine.generate(make_forecast(0.32, -0.30), save_to_db=False)
    assert sig["signal"] == "BEARISH"


def test_strong_bearish(engine):
    sig = engine.generate(make_forecast(0.20, -0.60), save_to_db=False)
    assert sig["signal"] == "STRONG_BEARISH"


def test_confidence_range(engine):
    for prob, move in [(0.80, 0.60), (0.52, 0.05), (0.20, -0.60)]:
        sig = engine.generate(make_forecast(prob, move), save_to_db=False)
        assert 0.0 <= sig["confidence"] <= 1.0


def test_conflicting_signals_neutral(engine):
    # High prob_up but tiny expected move → should be NEUTRAL
    sig = engine.generate(make_forecast(0.70, 0.05), save_to_db=False)
    assert sig["signal"] == "NEUTRAL"
