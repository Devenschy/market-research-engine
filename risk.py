# =============================================================================
# risk.py — Risk Management: Position Sizing, SL/TP, Kill Switch
# =============================================================================
# WHY: Risk management is where most retail traders fail. They focus obsessively
# on entry signals but ignore position sizing and loss limits. Professional
# trading desks treat risk management as equally important as alpha generation.
# This module enforces three non-negotiable rules on every trade.

import config
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Position:
    """
    Represents a single open paper trading position.

    WHY: We track all relevant data for each position so the engine can:
    1. Monitor SL/TP on every tick
    2. Calculate unrealized P&L for the dashboard
    3. Attribute closed trades to the right strategy for performance analysis
    """
    symbol: str
    entry_price: float
    quantity: float
    stop_loss: float
    take_profit: float
    direction: str          # 'LONG' or 'SHORT'
    strategy: str           # Which strategy generated the signal
    regime: str             # Market regime at time of entry
    entry_time: datetime = field(default_factory=datetime.now)
    votes: dict = field(default_factory=dict)  # Which strategies voted

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L at current market price.

        WHY: Tracking unrealized P&L lets you see if your open positions
        are working. If unrealized P&L deteriorates rapidly, it's information
        about whether the market is moving against you before SL is hit.
        """
        if self.direction == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct(self, current_price: float) -> float:
        """P&L as percentage of entry value."""
        entry_value = self.entry_price * self.quantity
        if entry_value == 0:
            return 0.0
        return self.unrealized_pnl(current_price) / entry_value * 100


class RiskManager:
    """
    Enforces all risk controls on every trade decision.

    Three layers of risk control:
    1. Position sizing: Fixed fractional (5% of capital per trade)
    2. Per-trade SL/TP: 2% stop-loss, 4% take-profit
    3. Daily kill switch: 5% daily drawdown halts all trading

    WHY: These three layers work together. Even if a strategy fires 10 bad
    signals in a row, the position sizing limits each loss to ~0.1% of capital
    (5% size * 2% stop), and the kill switch stops trading after losing 5% in a day.
    The combination makes catastrophic loss mathematically very difficult.
    """

    def __init__(self, starting_capital: float = config.STARTING_CAPITAL):
        self.starting_capital = starting_capital
        self.capital = starting_capital    # Current available cash
        self.equity = starting_capital     # Total equity including open positions
        self.open_positions: dict[str, Position] = {}  # symbol → Position

        # Daily tracking for kill switch
        self.day_start_equity = starting_capital
        self.current_date = date.today()
        self.trading_halted = False

        # Realized P&L tracking
        self.realized_pnl = 0.0
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0

    def calculate_position_size(self, price: float,
                                  regime_modifier: float = 1.0) -> float:
        """
        Calculate position size using fixed fractional sizing.

        WHY: Fixed fractional sizing (Kelly-adjacent) means:
        - Loss amount scales DOWN as account shrinks (protective)
        - Gain amount scales UP as account grows (compounding)
        - You can never lose more than the position_size_pct in any trade
          (assuming stop-loss is honored)

        The regime_modifier reduces size in VOLATILE regimes (0.5x) and
        slightly in RANGING regimes (0.75x) because signal reliability is lower.

        Quantity = (Capital * POSITION_SIZE_PCT * regime_modifier) / price
        """
        trade_capital = self.capital * config.POSITION_SIZE_PCT * regime_modifier
        quantity = trade_capital / price
        return round(quantity, 8)  # 8 decimal places for crypto compatibility

    def calculate_stop_loss(self, entry_price: float, direction: str) -> float:
        """
        Auto-calculate stop-loss 2% from entry price.

        WHY: Automatic SL placement removes emotional decision-making.
        The 2% level is chosen because it's wide enough to avoid noise
        (small random price fluctuations) while limiting loss to a known amount.
        In highly volatile assets like crypto, you may want to widen this.
        """
        if direction == 'LONG':
            return round(entry_price * (1 - config.STOP_LOSS_PCT), 8)
        else:
            return round(entry_price * (1 + config.STOP_LOSS_PCT), 8)

    def calculate_take_profit(self, entry_price: float, direction: str) -> float:
        """
        Auto-calculate take-profit 4% from entry price.

        WHY: 4% take-profit on 2% stop-loss = 2:1 reward-to-risk ratio.
        With a 40% win rate: (0.4 * 4%) - (0.6 * 2%) = +0.4% per trade.
        This means the strategy can be WRONG most of the time and still
        make money — a fundamental insight about professional trading.
        """
        if direction == 'LONG':
            return round(entry_price * (1 + config.TAKE_PROFIT_PCT), 8)
        else:
            return round(entry_price * (1 - config.TAKE_PROFIT_PCT), 8)

    def check_kill_switch(self) -> bool:
        """
        Check and enforce the daily drawdown kill switch.

        WHY: Prop trading desks call this the 'daily loss limit' or 'DLL'.
        If you've lost 5% of your starting equity today, you STOP trading.
        No exceptions. No 'I can recover it' rationalization.

        The psychological tendency to double down after losses is called
        'loss aversion' and it's the primary cause of trading blowups.
        The kill switch removes discretion — the system enforces the rule.

        Returns True if trading should continue, False if halted.
        """
        # Reset daily tracking at start of new day
        today = date.today()
        if today != self.current_date:
            self.current_date = today
            self.day_start_equity = self.equity
            self.trading_halted = False
            return True

        if self.trading_halted:
            return False

        # Calculate today's drawdown
        daily_drawdown = (self.day_start_equity - self.equity) / self.day_start_equity

        if daily_drawdown >= config.MAX_DAILY_DRAWDOWN_PCT:
            self.trading_halted = True
            print(f"[risk] KILL SWITCH ACTIVATED — Daily drawdown: {daily_drawdown:.1%}")
            return False

        return True

    def can_open_position(self, symbol: str) -> tuple[bool, str]:
        """
        Check all conditions before opening a new position.

        Returns (can_open: bool, reason: str)
        """
        if not self.check_kill_switch():
            return False, "Kill switch active — daily loss limit reached"

        if symbol in self.open_positions:
            return False, f"Already have open position in {symbol}"

        if len(self.open_positions) >= config.MAX_OPEN_POSITIONS:
            return False, f"Max open positions ({config.MAX_OPEN_POSITIONS}) reached"

        if self.capital <= 0:
            return False, "No available capital"

        return True, "OK"

    def open_position(self, symbol: str, price: float, direction: str,
                       strategy: str, regime: str, votes: dict,
                       regime_modifier: float = 1.0) -> Optional[Position]:
        """
        Open a new paper trading position with all risk parameters set.

        WHY: We set SL and TP at entry time, not reactively. This is important
        because it removes the temptation to 'move the stop' when a trade goes
        against you — one of the most common and costly retail trading errors.
        """
        can_open, reason = self.can_open_position(symbol)
        if not can_open:
            return None

        quantity = self.calculate_position_size(price, regime_modifier)
        if quantity <= 0:
            return None

        stop_loss = self.calculate_stop_loss(price, direction)
        take_profit = self.calculate_take_profit(price, direction)

        position = Position(
            symbol=symbol,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            direction=direction,
            strategy=strategy,
            regime=regime,
            votes=votes
        )

        cost = price * quantity
        self.capital -= cost
        self.open_positions[symbol] = position
        self.total_trades += 1

        return position

    def check_exits(self, prices: dict[str, float]) -> list[dict]:
        """
        Check all open positions against current prices for SL/TP triggers.

        WHY: SL/TP must be checked on EVERY tick. In paper trading this is
        simulated — in live trading this would be handled by the broker's
        order management system. Checking every minute is sufficient for our
        polling interval.

        Returns a list of closed position records for logging.
        """
        closed = []

        for symbol, position in list(self.open_positions.items()):
            if symbol not in prices:
                continue

            current_price = prices[symbol]
            exit_reason = None

            if position.direction == 'LONG':
                if current_price <= position.stop_loss:
                    exit_reason = 'STOP_LOSS'
                elif current_price >= position.take_profit:
                    exit_reason = 'TAKE_PROFIT'
            else:  # SHORT
                if current_price >= position.stop_loss:
                    exit_reason = 'STOP_LOSS'
                elif current_price <= position.take_profit:
                    exit_reason = 'TAKE_PROFIT'

            if exit_reason:
                closed_record = self._close_position(symbol, current_price, exit_reason)
                closed.append(closed_record)

        return closed

    def _close_position(self, symbol: str, exit_price: float,
                         exit_reason: str) -> dict:
        """
        Close a position and update capital/P&L tracking.

        WHY: When we close a position, we add back the original cost basis
        plus any realized profit/loss. This keeps the capital tracking accurate.
        """
        position = self.open_positions.pop(symbol)
        exit_value = exit_price * position.quantity
        entry_cost = position.entry_price * position.quantity

        if position.direction == 'LONG':
            pnl = exit_value - entry_cost
        else:
            pnl = entry_cost - exit_value

        self.capital += entry_cost + pnl
        self.realized_pnl += pnl

        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1

        return {
            'symbol': symbol,
            'direction': position.direction,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'pnl_pct': (pnl / entry_cost) * 100,
            'exit_reason': exit_reason,
            'strategy': position.strategy,
            'regime': position.regime,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'votes': position.votes
        }

    def update_equity(self, prices: dict[str, float]):
        """
        Recalculate total equity including unrealized P&L from open positions.

        WHY: The kill switch watches EQUITY, not just realized P&L. If open
        positions are underwater, that counts against the daily drawdown limit.
        This prevents the scenario where unrealized losses are ignored until
        they crystallize into actual losses.
        """
        open_pnl = sum(
            pos.unrealized_pnl(prices[sym])
            for sym, pos in self.open_positions.items()
            if sym in prices
        )
        position_cost = sum(
            pos.entry_price * pos.quantity
            for pos in self.open_positions.values()
        )
        self.equity = self.capital + position_cost + open_pnl

    def get_stats(self) -> dict:
        """Return current portfolio statistics for dashboard/logging."""
        win_rate = (self.winning_trades / self.total_trades * 100
                    if self.total_trades > 0 else 0)
        return {
            'equity': round(self.equity, 2),
            'capital': round(self.capital, 2),
            'realized_pnl': round(self.realized_pnl, 2),
            'total_return_pct': round((self.equity - self.starting_capital) /
                                       self.starting_capital * 100, 2),
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 1),
            'open_positions': len(self.open_positions),
            'trading_halted': self.trading_halted,
            'daily_drawdown_pct': round(
                (self.day_start_equity - self.equity) / self.day_start_equity * 100, 2
            ) if self.day_start_equity > 0 else 0
        }
