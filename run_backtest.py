"""
Backtest runner — quick CLI to run the Kronos walk-forward backtester.

Usage:
    python run_backtest.py
    python run_backtest.py --symbol BANKNIFTY --start 2025-06-01 --end 2025-12-31
    python run_backtest.py --symbol NIFTY --start 2025-01-01 --end 2025-12-31 --lots 2
"""
import argparse
import sys
from pathlib import Path

# Run from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.utils import get_broker, load_config, setup_logging
from src.backtester import Backtester


def main():
    setup_logging("backtest")

    cfg = load_config()
    default_start = cfg.get("backtest", {}).get("default_start", "2025-01-01")
    default_end   = cfg.get("backtest", {}).get("default_end",   "2025-12-31")

    parser = argparse.ArgumentParser(description="Kronos Options Backtester")
    parser.add_argument("--symbol", default="NIFTY",      choices=["NIFTY", "BANKNIFTY", "SENSEX"])
    parser.add_argument("--start",  default=default_start, help="Start date YYYY-MM-DD")
    parser.add_argument("--end",    default=default_end,   help="End date YYYY-MM-DD")
    parser.add_argument("--lots",   type=int, default=1,  help="Number of lots per trade")
    args = parser.parse_args()

    print(f"\nBacktest: {args.symbol}  {args.start} → {args.end}  lots={args.lots}\n")

    broker = get_broker()
    bt     = Backtester(broker=broker)
    result = bt.run(args.symbol, args.start, args.end, lots=args.lots)

    stats = result["stats"]
    print("\n── Results ─────────────────────────────────────")
    print(f"  Trades:       {stats['total_trades']} ({stats.get('wins',0)}W / {stats.get('losses',0)}L)")
    print(f"  Win rate:     {stats['win_rate_pct']:.1f}%")
    print(f"  Total P&L:    ₹{stats['total_pnl_rs']:,.0f}")
    print(f"  Avg win:      ₹{stats.get('avg_win_rs', 0):,.0f}")
    print(f"  Avg loss:     ₹{stats.get('avg_loss_rs', 0):,.0f}")
    print(f"  Profit factor:{stats['profit_factor']:.2f}")
    print(f"  Sharpe:       {stats['sharpe']:.2f}")
    print(f"  Max drawdown: ₹{stats['max_drawdown_rs']:,.0f}")
    print("────────────────────────────────────────────────\n")

    tl = result["trade_log"]
    if not tl.empty:
        bs_pct = (tl.get("data_source", "") == "bs_approximation").mean() * 100 if "data_source" in tl.columns else 0
        if bs_pct > 0:
            print(f"  ⚠  {bs_pct:.1f}% of trades used Black-Scholes fallback (real option data unavailable).")
        print(tl[["trade_date", "symbol", "signal", "strategy", "net_pnl_rs", "data_source"]].to_string(index=False))


if __name__ == "__main__":
    main()
