# Kronos Options Signal System

Intraday options signal system for NIFTY, BANKNIFTY, and SENSEX using the
[Kronos foundation model](https://github.com/shiyu-coder/Kronos) for probabilistic
index price forecasting. Converts index forecasts into concrete option trade
recommendations with realistic cost modelling.

**Data source: Upstox v2 API** (real historical option prices via expired-instruments API).

---

## Architecture

```
market data → DataFetcher → DataCleaner → KronosForecaster
                                                 ↓
                                         SignalEngine (BULLISH / BEARISH / NEUTRAL + confidence)
                                                 ↓
                                         OptionsMapper (pick strikes, strategy, lot sizes)
                                                 ↓
                                      ┌─── Backtester (historical walk-forward)
                                      └─── PaperTrader (live loop, no real orders)
                                                 ↓
                                          Streamlit Dashboard
```

---

## Windows Setup (8 GB RAM, no GPU)

### 1. Prerequisites

```
Python 3.10 or 3.11 (https://www.python.org/downloads/)
Git (https://git-scm.com/download/win)
```

### 2. Clone this repo and install dependencies

```powershell
cd C:\Users\YourName\Desktop
git clone <this-repo-url> kronos_options
cd kronos_options
pip install -r requirements.txt
```

### 3. Install Kronos model

```powershell
cd C:\Users\YourName\Desktop
git clone https://github.com/shiyu-coder/Kronos
cd Kronos
pip install -r requirements.txt
```

Note the full path to the Kronos folder — you'll need it in the next step.

### 4. Configure credentials

```powershell
copy .env.example .env
notepad .env
```

Fill in:
- `UPSTOX_ACCESS_TOKEN` — get from Upstox Developer Console after daily login
- `KRONOS_REPO_PATH` — full path to the cloned Kronos folder (e.g., `C:\Users\YourName\Desktop\Kronos`)

### 5. Initialise database

```powershell
python -c "from src.db import init_db; init_db()"
```

### 6. Test the setup

```powershell
pytest tests/ -v
```

---

## Daily Workflow

### Step 1 — Refresh Upstox token (every morning before 9:15)

Upstox access tokens expire daily. After logging in via the Upstox app or web:

```powershell
# Update UPSTOX_ACCESS_TOKEN in .env with today's token
notepad .env
```

### Step 2 — Fetch historical data (first time, or weekly top-up)

```powershell
python -c "
from src.data_fetcher import DataFetcher
from src.utils import get_broker
f = DataFetcher(get_broker())
for sym in ['NIFTY', 'BANKNIFTY', 'SENSEX']:
    df = f.fetch_date_range(sym, '2025-01-01', '2025-12-31')
    print(sym, len(df), 'bars')
"
```

### Step 3 — Run paper trader (market hours only)

```powershell
python -m src.paper_trader --symbols NIFTY BANKNIFTY
```

### Step 4 — Open dashboard

```powershell
streamlit run dashboard/app.py
```

Visit `http://localhost:8501` in your browser.

---

## Module-by-Module Testing

### data_fetcher + data_cleaner

```powershell
python -c "
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean
from src.utils import get_broker
f = DataFetcher(get_broker())
df = f.fetch_date_range('NIFTY', '2025-06-01', '2025-06-10')
print(clean(df, 'NIFTY').tail())
"
```
Expected: 5-min bars, IST index, no weekends or 9:15 pre-market candles.

### forecaster

```powershell
python -c "
from src.forecaster import KronosForecaster
from src.data_fetcher import DataFetcher
from src.data_cleaner import clean
from src.utils import get_broker
df = clean(DataFetcher(get_broker()).fetch_date_range('NIFTY', '2025-06-01', '2025-06-10'))
fc = KronosForecaster().forecast('NIFTY', df)
print('prob_up:', fc['prob_up'], 'move:', fc['expected_move_pct'])
"
```
Expected: dict with `prob_up`, `expected_move_pct`, `median_path` DataFrame.

### signal_engine

```powershell
pytest tests/test_signal_engine.py -v
```
Expected: all 7 tests pass.

### backtester

```powershell
python -c "
from src.backtester import Backtester
from src.utils import get_broker
bt = Backtester(get_broker())
r = bt.run('NIFTY', '2025-01-01', '2025-03-31')
print(r['stats'])
"
```
Expected: stats dict with total_trades, win_rate_pct, total_pnl_rs, sharpe.

---

## Backtesting Notes

The backtester uses **real Upstox historical option OHLC data** via the
`/expired-instruments/historical-candle` API endpoint — the same approach
as your existing `nifty_first_candle_v5.py` script. This is significantly
more accurate than Black-Scholes approximation.

When real data is unavailable for a bar, the backtester falls back to
Black-Scholes and tags the trade `data_source=bs_approximation`.
Filter these out of performance stats for clean analysis:

```python
real_trades = results["trade_log"][results["trade_log"]["data_source"] == "real_option_data"]
```

---

## Key Assumptions

| # | Assumption | Impact |
|---|-----------|--------|
| 1 | Lot sizes from config.yaml (NIFTY=75, BN=15, SENSEX=10) | Update when NSE revises; or fetch from broker at runtime |
| 2 | Kronos model API follows the pattern in its README | Adjust `_run_kronos()` in forecaster.py if API differs |
| 3 | SENSEX volume = 0 is normal (BSE index methodology) | Flagged but not dropped; volume feature ignored for SENSEX |
| 4 | Black-Scholes fallback uses IV=15% and r=6.5% | Only for missing data; clearly tagged in trade log |
| 5 | Upstox `expired-instruments` API covers all needed history | Verify for dates > 2 years old; may need TrueData for older data |
| 6 | Weekly expiry: NIFTY=Thursday, BANKNIFTY=Wednesday, SENSEX=Friday | Update if NSE/BSE changes expiry day |

---

## File Structure

```
kronos_options/
├── config.yaml           ← all settings, no credentials
├── .env                  ← credentials (gitignored)
├── requirements.txt
├── README.md
├── data/
│   ├── historical/       ← cached 5-min CSV per symbol
│   └── kronos_options.db ← SQLite: forecasts, signals, trades, P&L
├── src/
│   ├── broker/
│   │   ├── base.py       ← BrokerInterface ABC
│   │   ├── upstox.py     ← PRIMARY: Upstox v2 REST
│   │   └── zerodha.py    ← STUB: swap when needed
│   ├── data_fetcher.py   ← fetch + cache OHLCV + option candles
│   ├── data_cleaner.py   ← filter market hours, holidays, gaps
│   ├── forecaster.py     ← Kronos-small wrapper
│   ├── signal_engine.py  ← forecast → signal
│   ├── options_mapper.py ← signal → option trade legs
│   ├── backtester.py     ← walk-forward backtest, real option prices
│   ├── paper_trader.py   ← live loop, no real orders
│   ├── live_trader.py    ← real orders (triple-locked, disabled by default)
│   ├── db.py             ← SQLite schema and helpers
│   └── utils.py          ← IST helpers, holidays, config loader, broker factory
├── dashboard/
│   └── app.py            ← Streamlit: 4 pages
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_first_forecast.ipynb
│   └── 03_signal_tuning.ipynb
└── tests/
    ├── test_utils.py
    ├── test_data_cleaner.py
    └── test_signal_engine.py
```

---

## Live Trading Safety

Live trading has a **triple lock**:

1. `LIVE_TRADING=true` in `.env`
2. `--live` CLI flag
3. Typed confirmation: `I CONFIRM LIVE TRADING`

Plus hard limits in `config.yaml` → `live_trading`:
- `max_trades_per_day: 3`
- `max_capital_per_trade: 50000`
- `max_daily_loss_rs: -5000` → kills all trading for the day if hit

Do not enable live trading until you have run the backtest and paper trader
for at least 4 weeks and are comfortable with the signal quality.
