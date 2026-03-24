# =============================================================================
# main.py — Entry Point
# =============================================================================
# WHY: The entry point is kept minimal intentionally. Its only job is to
# initialize the engine and handle top-level errors. All logic lives in the
# modules. This mirrors how production trading systems are structured —
# the launcher does as little as possible; the engine does everything.

import sys
import os
from pathlib import Path


def check_dependencies():
    """Verify all required packages are installed before starting."""
    required = ['yfinance', 'pandas', 'numpy', 'rich', 'requests']
    missing = []
    for package in required:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print(f"Install with: pip install {' '.join(missing)}")
        sys.exit(1)


def print_banner():
    """Display startup banner with system configuration."""
    import config

    print("=" * 70)
    print("  MARKET RESEARCH ENGINE — Multi-Asset Paper Trading System")
    print("  Dezona Group — Internal Research Tooling")
    print("=" * 70)
    print(f"  Mode:          {'PAPER TRADING' if config.PAPER_TRADING else 'LIVE TRADING'}")
    print(f"  Starting Cap:  ${config.STARTING_CAPITAL:,.2f}")
    print(f"  Symbols:       {', '.join(config.SYMBOLS)}")
    print(f"  Strategies:    MA Crossover, RSI Momentum, Mean Reversion")
    print(f"  Vote Required: 2/3 strategies must agree")
    print(f"  Position Size: {config.POSITION_SIZE_PCT*100:.0f}% per trade")
    print(f"  Stop Loss:     {config.STOP_LOSS_PCT*100:.0f}%  |  Take Profit: {config.TAKE_PROFIT_PCT*100:.0f}%")
    print(f"  Kill Switch:   {config.MAX_DAILY_DRAWDOWN_PCT*100:.0f}% daily drawdown")
    print(f"  Poll Interval: {config.POLL_INTERVAL_SECONDS}s")
    print("=" * 70)
    print()


def main():
    """
    Main entry point.

    WHY: We check dependencies first (fail fast), then show configuration
    so the user can verify settings before the system begins trading.
    Transparency at startup is a professional practice — you always want
    to know exactly what parameters the system is running with.
    """
    check_dependencies()
    print_banner()

    # Ensure FRED key reminder
    import config
    if not config.FRED_API_KEY:
        print("[main] Note: FRED_API_KEY not set — Treasury yield data unavailable.")
        print("[main] Register free at fred.stlouisfed.org and add key to config.py")
        print()

    # Create logs directory
    Path('logs').mkdir(exist_ok=True)

    # Initialize and run the engine
    from engine import TradingEngine
    engine = TradingEngine()
    engine.run()


if __name__ == '__main__':
    main()
