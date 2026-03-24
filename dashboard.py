# =============================================================================
# dashboard.py — Live Terminal Dashboard using Rich
# =============================================================================
# WHY: The dashboard makes this feel like a real trading system, not a script.
# Staring at live data — watching regimes shift, signals fire, positions open
# and close — builds intuition faster than any book. The regime column is
# particularly important: over time you will start to see which regimes
# produce reliable signals and which produce noise. That is applied market knowledge.

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich import box
from datetime import datetime
from collections import deque
import config

console = Console()

# Signal event log (last 20 signals shown on dashboard)
signal_log = deque(maxlen=20)


def add_signal_event(symbol: str, strategy: str, signal: str,
                     price: float, regime: str, triggered: bool,
                     reason: str = None):
    """Add a new signal event to the rolling log for dashboard display."""
    signal_log.append({
        'timestamp': datetime.now().strftime('%H:%M:%S'),
        'symbol': symbol,
        'strategy': strategy[:12],
        'signal': signal,
        'price': price,
        'regime': regime,
        'triggered': triggered,
        'reason': reason or ''
    })


def _regime_color(regime: str) -> str:
    """
    Color-code regimes for instant visual identification.
    WHY: Color coding lets you identify the regime at a glance without reading text.
    After a few days, your brain pattern-matches automatically.
    """
    colors = {
        'TRENDING': 'green',
        'RANGING': 'yellow',
        'VOLATILE': 'red',
        'UNKNOWN': 'dim'
    }
    return colors.get(regime, 'white')


def _pnl_color(pnl: float) -> str:
    """Green for positive P&L, red for negative — universal trading convention."""
    return 'green' if pnl >= 0 else 'red'


def build_portfolio_panel(stats: dict, alt_data: dict) -> Panel:
    """
    Portfolio summary panel — equity, P&L, drawdown, kill switch status.

    WHY: The portfolio panel is the first thing you look at. It tells you
    the current state of the account at a glance. The kill switch indicator
    is prominently displayed so you always know if trading is halted.
    """
    table = Table(box=None, show_header=False, padding=(0, 1))
    table.add_column("Metric", style="bold")
    table.add_column("Value")

    equity = stats.get('equity', 0)
    capital = stats.get('capital', 0)
    total_return = stats.get('total_return_pct', 0)
    daily_dd = stats.get('daily_drawdown_pct', 0)
    win_rate = stats.get('win_rate', 0)
    halted = stats.get('trading_halted', False)

    status_text = Text("HALTED ⛔", style="bold red") if halted else Text("ACTIVE ✓", style="bold green")

    table.add_row("Status", status_text)
    table.add_row("Equity", Text(f"${equity:,.2f}", style=_pnl_color(equity - config.STARTING_CAPITAL)))
    table.add_row("Cash", f"${capital:,.2f}")
    table.add_row("Total Return", Text(f"{total_return:+.2f}%", style=_pnl_color(total_return)))
    table.add_row("Daily Drawdown", Text(f"{daily_dd:.2f}%", style='red' if daily_dd > 3 else 'white'))
    table.add_row("Win Rate", f"{win_rate:.1f}% ({stats.get('winning_trades',0)}W/{stats.get('losing_trades',0)}L)")
    table.add_row("Open Positions", str(stats.get('open_positions', 0)))

    # Alternative data summary
    fear_greed = alt_data.get('fear_greed')
    if fear_greed:
        fg_val = fear_greed.get('value', 'N/A')
        fg_class = fear_greed.get('classification', '')
        fg_color = 'red' if fg_val < 30 else ('green' if fg_val > 70 else 'yellow')
        table.add_row("Fear & Greed", Text(f"{fg_val} ({fg_class})", style=fg_color))

    treasury = alt_data.get('treasury_yield_10y')
    if treasury:
        table.add_row("10Y Treasury", f"{treasury:.2f}%")

    return Panel(table, title="[bold blue]Portfolio[/bold blue]",
                 border_style="blue", padding=(0, 1))


def build_positions_panel(open_positions: dict, prices: dict, regimes: dict) -> Panel:
    """
    Open positions panel — shows all active trades with live P&L.

    WHY: You want to see all open positions at once with their current P&L,
    how far from SL/TP they are, what regime they entered in, and which
    strategies voted. This builds awareness of your current risk exposure.
    """
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold magenta")
    table.add_column("Symbol", width=10)
    table.add_column("Dir", width=5)
    table.add_column("Entry", width=10)
    table.add_column("Current", width=10)
    table.add_column("P&L%", width=7)
    table.add_column("SL", width=10)
    table.add_column("TP", width=10)
    table.add_column("Regime", width=9)
    table.add_column("Strategies", width=30)

    if not open_positions:
        table.add_row("—", "—", "—", "—", "—", "—", "—", "—", "No open positions")
    else:
        for symbol, pos in open_positions.items():
            current_price = prices.get(symbol, pos.entry_price)
            pnl_pct = pos.unrealized_pnl_pct(current_price)

            regime_info = regimes.get(symbol, {})
            regime_name = regime_info.get('regime', type('', (), {'value': 'UNKNOWN'})()).value if hasattr(regime_info.get('regime', None), 'value') else 'UNKNOWN'

            votes_str = ' | '.join(
                f"{k.split('_')[0]}:{v}" for k, v in pos.votes.items() if v
            )

            table.add_row(
                symbol,
                Text(pos.direction, style='green' if pos.direction == 'LONG' else 'red'),
                f"{pos.entry_price:.4f}",
                f"{current_price:.4f}",
                Text(f"{pnl_pct:+.2f}%", style=_pnl_color(pnl_pct)),
                f"{pos.stop_loss:.4f}",
                f"{pos.take_profit:.4f}",
                Text(regime_name, style=_regime_color(regime_name)),
                votes_str[:30]
            )

    return Panel(table, title="[bold magenta]Open Positions[/bold magenta]",
                 border_style="magenta", padding=(0, 1))


def build_signal_log_panel() -> Panel:
    """
    Rolling signal log panel — last 20 signal events.

    WHY: The signal log is the learning tool. You want to see every signal that
    fired and whether it triggered a trade. If it didn't trigger, why? Was it
    blocked by the vote filter? Regime suppression? Risk limits?

    Reading this log daily builds pattern recognition faster than any textbook.
    Pay attention to:
    - Which regimes produce signals that DO trigger trades
    - Which asset classes generate the most signals
    - Whether buy signals cluster (risk-on sentiment across assets)
    """
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold cyan")
    table.add_column("Time", width=8)
    table.add_column("Symbol", width=9)
    table.add_column("Strategy", width=14)
    table.add_column("Signal", width=6)
    table.add_column("Price", width=10)
    table.add_column("Regime", width=9)
    table.add_column("Traded", width=7)
    table.add_column("Note", width=25)

    recent_signals = list(reversed(signal_log))

    if not recent_signals:
        table.add_row("—", "—", "—", "—", "—", "—", "—", "Waiting for signals...")
    else:
        for event in recent_signals:
            sig = event['signal']
            sig_color = 'green' if sig == 'BUY' else 'red'
            traded_text = Text("YES ✓", style='green') if event['triggered'] else Text("NO ✗", style='red dim')

            table.add_row(
                event['timestamp'],
                event['symbol'],
                event['strategy'],
                Text(sig, style=sig_color),
                f"{event['price']:.4f}",
                Text(event['regime'], style=_regime_color(event['regime'])),
                traded_text,
                event['reason'][:25]
            )

    return Panel(table, title="[bold cyan]Signal Log (Last 20)[/bold cyan]",
                 border_style="cyan", padding=(0, 1))


def build_regime_panel(regimes: dict) -> Panel:
    """
    Current regime classification for all symbols.

    WHY: Seeing all regimes at once lets you spot macro patterns.
    If all equity symbols shift to VOLATILE simultaneously, that's
    a market-wide risk event — not a symbol-specific issue.
    """
    table = Table(box=box.SIMPLE, show_header=True, header_style="bold white")
    table.add_column("Symbol", width=10)
    table.add_column("Regime", width=10)
    table.add_column("ADX", width=7)
    table.add_column("ATR", width=10)
    table.add_column("BB Width", width=9)
    table.add_column("Size Mod", width=9)

    for symbol in config.SYMBOLS:
        r = regimes.get(symbol, {})
        regime_obj = r.get('regime', None)
        regime_name = regime_obj.value if hasattr(regime_obj, 'value') else 'UNKNOWN'

        table.add_row(
            symbol,
            Text(regime_name, style=_regime_color(regime_name)),
            str(r.get('adx', '—')),
            str(r.get('atr', '—')),
            str(r.get('bb_width', '—')),
            f"{r.get('position_size_modifier', 1.0):.1f}x"
        )

    return Panel(table, title="[bold white]Market Regimes[/bold white]",
                 border_style="white", padding=(0, 1))


def render_dashboard(stats: dict, open_positions: dict, prices: dict,
                     regimes: dict, alt_data: dict, tick_count: int):
    """
    Render the full dashboard to the terminal.

    Called by the engine on every tick. Uses Rich's console.clear() + print
    approach for simplicity. For a smoother display, use Rich's Live context
    manager (shown in main.py).

    WHY: We refresh the entire screen on every tick rather than updating
    individual cells. This is simpler to implement and sufficient for a
    60-second polling interval. HFT systems use differential updates.
    """
    console.clear()

    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    console.print(
        Panel(
            f"[bold]Market Research Engine[/bold] | "
            f"[dim]Tick #{tick_count}[/dim] | "
            f"[dim]{now}[/dim] | "
            f"[{'red' if stats.get('trading_halted') else 'green'}]"
            f"{'PAPER TRADING HALTED' if stats.get('trading_halted') else 'PAPER TRADING ACTIVE'}[/]",
            style="bold white on blue"
        )
    )

    console.print(build_portfolio_panel(stats, alt_data))
    console.print(build_regime_panel(regimes))
    console.print(build_positions_panel(open_positions, prices, regimes))
    console.print(build_signal_log_panel())
