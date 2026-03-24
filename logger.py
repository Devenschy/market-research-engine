# =============================================================================
# logger.py — Strategy Performance Logging to CSV + JSON
# =============================================================================
# WHY: The log files are more valuable than the live dashboard for learning.
# After running the system for a week, you can analyze:
# - Which strategy/regime combinations produce the best signals
# - Which signals fired but DIDN'T trigger a trade (and what happened to price after)
# - How your win rate varies across asset classes
# - What your actual Sharpe ratio is (not estimated, ACTUAL)
#
# The signal log is particularly valuable — reviewing signals that did NOT
# trigger trades teaches you about the majority vote filter and regime suppression.

import csv
import json
import os
import numpy as np
from datetime import datetime
from pathlib import Path
import config


def ensure_log_dirs():
    """Create log directories if they don't exist."""
    Path('logs').mkdir(exist_ok=True)


def log_signal(symbol: str, strategy: str, signal: str,
               price: float, regime: str, triggered_trade: bool,
               suppression_reason: str = None, votes: dict = None):
    """
    Log every signal generated — whether or not it triggered a trade.

    WHY: Signal logging is separated from trade logging intentionally.
    You want to review EVERY signal, including:
    - Signals blocked by majority vote filter (only 1 strategy agreed)
    - Signals suppressed by regime filter (wrong regime)
    - Signals blocked by risk limits (max positions, kill switch)

    Reviewing blocked signals is how you develop intuition for when the
    filters are working correctly vs when they're filtering out good trades.
    """
    ensure_log_dirs()
    file_exists = os.path.exists(config.SIGNALS_LOG)

    with open(config.SIGNALS_LOG, 'a', newline='') as f:
        fieldnames = ['timestamp', 'symbol', 'strategy', 'signal', 'price',
                      'regime', 'triggered_trade', 'suppression_reason',
                      'ma_vote', 'rsi_vote', 'mean_rev_vote']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        votes = votes or {}
        writer.writerow({
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'strategy': strategy,
            'signal': signal,
            'price': round(price, 6),
            'regime': regime,
            'triggered_trade': triggered_trade,
            'suppression_reason': suppression_reason or '',
            'ma_vote': votes.get('MA_Crossover', ''),
            'rsi_vote': votes.get('RSI_Momentum', ''),
            'mean_rev_vote': votes.get('Mean_Reversion', '')
        })


def log_trade(trade_record: dict):
    """
    Log every executed (paper) trade with full entry/exit/PnL details.

    WHY: The trade log is your audit trail. After 30 days, you can calculate
    your actual performance metrics:
    - Win rate by strategy
    - Average win vs average loss
    - Sharpe ratio (mean return / std of returns, annualized)
    - Maximum drawdown period
    - Performance by regime (did trend-following work in TRENDING regimes?)

    These calculations from real data are what you present in an interview,
    not just "I built a trading bot."
    """
    ensure_log_dirs()
    file_exists = os.path.exists(config.TRADES_LOG)

    with open(config.TRADES_LOG, 'a', newline='') as f:
        fieldnames = ['entry_time', 'exit_time', 'symbol', 'direction',
                      'entry_price', 'exit_price', 'quantity', 'pnl',
                      'pnl_pct', 'exit_reason', 'strategy', 'regime',
                      'ma_vote', 'rsi_vote', 'mean_rev_vote']
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        votes = trade_record.get('votes', {})
        writer.writerow({
            'entry_time': trade_record['entry_time'].isoformat() if hasattr(trade_record['entry_time'], 'isoformat') else trade_record['entry_time'],
            'exit_time': trade_record['exit_time'].isoformat() if hasattr(trade_record['exit_time'], 'isoformat') else trade_record['exit_time'],
            'symbol': trade_record['symbol'],
            'direction': trade_record['direction'],
            'entry_price': round(trade_record['entry_price'], 6),
            'exit_price': round(trade_record['exit_price'], 6),
            'quantity': round(trade_record['quantity'], 8),
            'pnl': round(trade_record['pnl'], 4),
            'pnl_pct': round(trade_record['pnl_pct'], 2),
            'exit_reason': trade_record['exit_reason'],
            'strategy': trade_record['strategy'],
            'regime': trade_record['regime'],
            'ma_vote': votes.get('MA_Crossover', ''),
            'rsi_vote': votes.get('RSI_Momentum', ''),
            'mean_rev_vote': votes.get('Mean_Reversion', '')
        })


def update_performance(risk_manager) -> dict:
    """
    Calculate and persist rolling performance metrics to performance.json.

    WHY: Performance attribution — knowing WHICH strategies contribute to
    returns and which detract — is a core quant skill. The Sharpe ratio
    in particular is the universal language of risk-adjusted returns.

    SHARPE RATIO = (Portfolio Return - Risk Free Rate) / Portfolio Std Dev
    - Sharpe > 1: Good (better than 1 unit of return per unit of risk)
    - Sharpe > 2: Excellent
    - Sharpe < 0: You're losing money risk-adjusted

    We calculate an approximate Sharpe from trade-level data since we don't
    have daily NAV history in this simplified implementation.
    """
    ensure_log_dirs()
    stats = risk_manager.get_stats()

    # Load existing trade data for Sharpe calculation
    pnl_pcts = []
    strategy_stats = {}

    if os.path.exists(config.TRADES_LOG):
        with open(config.TRADES_LOG, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    pnl_pct = float(row['pnl_pct'])
                    pnl_pcts.append(pnl_pct)
                    strat = row['strategy']
                    if strat not in strategy_stats:
                        strategy_stats[strat] = {'pnls': [], 'wins': 0, 'losses': 0}
                    strategy_stats[strat]['pnls'].append(pnl_pct)
                    if pnl_pct > 0:
                        strategy_stats[strat]['wins'] += 1
                    else:
                        strategy_stats[strat]['losses'] += 1
                except (ValueError, KeyError):
                    continue

    # Calculate Sharpe ratio (annualized, assuming 252 trading days)
    sharpe = None
    if len(pnl_pcts) > 5:
        mean_return = np.mean(pnl_pcts)
        std_return = np.std(pnl_pcts)
        if std_return > 0:
            # Annualize: multiply by sqrt(252) for daily, or sqrt(252*6.5*60) for hourly
            sharpe = round((mean_return / std_return) * np.sqrt(252), 3)

    # Per-strategy performance
    strategy_performance = {}
    for strat, data in strategy_stats.items():
        total = data['wins'] + data['losses']
        strategy_performance[strat] = {
            'total_trades': total,
            'wins': data['wins'],
            'losses': data['losses'],
            'win_rate': round(data['wins'] / total * 100, 1) if total > 0 else 0,
            'avg_pnl_pct': round(np.mean(data['pnls']), 3) if data['pnls'] else 0
        }

    performance = {
        'last_updated': datetime.now().isoformat(),
        'portfolio': {
            'equity': stats['equity'],
            'total_return_pct': stats['total_return_pct'],
            'realized_pnl': stats['realized_pnl'],
            'total_trades': stats['total_trades'],
            'win_rate': stats['win_rate'],
            'sharpe_ratio': sharpe
        },
        'by_strategy': strategy_performance
    }

    with open(config.PERFORMANCE_LOG, 'w') as f:
        json.dump(performance, f, indent=2)

    return performance
