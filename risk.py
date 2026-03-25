# =============================================================================
# risk.py — Risk Management: Position Sizing, SL/TP, Kill Switch
# =============================================================================
# WHY: Risk management is where most retail traders fail. They focus obsessively
# on entry signals but ignore position sizing and loss limits. Professional
# trading desks treat risk management as equally important as alpha generation.
# This module enforces three non-negotiable rules on every trade.

import config
import json
import os
from datetime import datetime, date
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

STATE_FILE = 'logs/portfolio_state.json'


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

    def calculate_risk_parity_size(self, symbol: str, price: float,
                                     price_history: list,
                                     regime_modifier: float = 1.0) -> float:
        """
        Calculate position size using Risk Parity — equal risk contribution per position.

        WHY RISK PARITY BEATS FIXED FRACTIONAL FOR MULTI-ASSET SYSTEMS:
        Fixed fractional (5% per trade) ignores volatility. A 5% position in
        Bitcoin (daily vol ~3-4%) and a 5% position in EURUSD (daily vol ~0.3%)
        are wildly different risk exposures. Bitcoin position has 10x more risk.

        Risk parity fixes this by sizing so each position contributes EQUAL risk:
        Position Size = (Target Daily Vol $ Amount) / (Asset Daily Volatility)

        Example:
        - Target: $100 daily vol per position (1% of $10,000 capital)
        - Bitcoin daily vol: 3% → position = $100 / 0.03 = $3,333 (3.3% of capital)
        - EURUSD daily vol: 0.3% → position = $100 / 0.003 = $33,333 (capped at 10%)

        This means crypto gets SMALLER positions and forex/low-vol assets get LARGER
        ones — the opposite of what most retail traders do.

        Falls back to fixed fractional if not enough price history.
        """
        import numpy as np

        if not config.RISK_PARITY_ENABLED or len(price_history) < config.RISK_PARITY_VOL_WINDOW + 1:
            # Fall back to fixed fractional sizing if risk parity not enabled or insufficient data
            return self.calculate_position_size(price, regime_modifier)

        try:
            prices = np.array(price_history[-config.RISK_PARITY_VOL_WINDOW:])
            # Daily log returns: ln(P_t / P_{t-1})
            # WHY LOG RETURNS: Log returns are time-additive and normally distributed,
            # making them the correct input for volatility calculations.
            log_returns = np.diff(np.log(prices))
            daily_vol = np.std(log_returns)  # Daily volatility as a decimal

            if daily_vol <= 0:
                return self.calculate_position_size(price, regime_modifier)

            # Target dollar volatility per position
            target_dollar_vol = self.capital * config.RISK_PARITY_TARGET_VOL

            # Position value = target_dollar_vol / daily_vol
            position_value = target_dollar_vol / daily_vol

            # Apply regime modifier (reduces size in volatile regimes)
            position_value *= regime_modifier

            # Apply floor and ceiling as % of capital
            min_value = self.capital * config.RISK_PARITY_MIN_SIZE
            max_value = self.capital * config.RISK_PARITY_MAX_SIZE
            position_value = max(min_value, min(max_value, position_value))

            quantity = position_value / price
            return round(quantity, 8)

        except Exception:
            return self.calculate_position_size(price, regime_modifier)

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
                       regime_modifier: float = 1.0,
                       price_history: list = None) -> Optional[Position]:
        """
        Open a new paper trading position with all risk parameters set.

        WHY: We set SL and TP at entry time, not reactively. This is important
        because it removes the temptation to 'move the stop' when a trade goes
        against you — one of the most common and costly retail trading errors.
        """
        can_open, reason = self.can_open_position(symbol)
        if not can_open:
            return None

        if price_history and config.RISK_PARITY_ENABLED:
            quantity = self.calculate_risk_parity_size(symbol, price, price_history, regime_modifier)
        else:
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

    def save_state(self):
        """
        Persist the full portfolio state to disk after every tick.

        WHY: Without persistence, every restart (bug fix, deployment, crash)
        resets the portfolio to $10,000 with no positions. This makes it
        impossible to track long-term performance or maintain open positions
        across restarts. With persistence, the engine picks up exactly where
        it left off — capital, open positions, P&L history all intact.

        We save to JSON (human-readable) so you can inspect the state file
        directly if you ever need to debug or manually adjust a position.
        """
        Path('logs').mkdir(exist_ok=True)
        try:
            # Serialize open positions (dataclasses aren't JSON-serializable by default)
            positions_data = {}
            for symbol, pos in self.open_positions.items():
                positions_data[symbol] = {
                    'symbol': pos.symbol,
                    'entry_price': pos.entry_price,
                    'quantity': pos.quantity,
                    'stop_loss': pos.stop_loss,
                    'take_profit': pos.take_profit,
                    'direction': pos.direction,
                    'strategy': pos.strategy,
                    'regime': pos.regime,
                    'entry_time': pos.entry_time.isoformat(),
                    'votes': pos.votes,
                }

            state = {
                'capital': self.capital,
                'equity': self.equity,
                'realized_pnl': self.realized_pnl,
                'total_trades': self.total_trades,
                'winning_trades': self.winning_trades,
                'losing_trades': self.losing_trades,
                'day_start_equity': self.day_start_equity,
                'current_date': self.current_date.isoformat(),
                'trading_halted': self.trading_halted,
                'open_positions': positions_data,
                'last_saved': datetime.now().isoformat(),
            }

            with open(STATE_FILE, 'w') as f:
                json.dump(state, f, indent=2)

        except Exception as e:
            print(f"[risk] Warning: could not save state: {e}")

    def load_state(self) -> bool:
        """
        Restore portfolio state from the last save file on startup.

        WHY: This is called once when the engine initializes. If a state file
        exists from a previous run, we restore everything — capital, open
        positions, trade counts — so the engine continues seamlessly.
        If no state file exists (first ever run), we start fresh from
        STARTING_CAPITAL with no positions. Either way, the engine is ready.

        Returns True if state was loaded, False if starting fresh.
        """
        if not os.path.exists(STATE_FILE):
            print("[risk] No saved state found — starting fresh.")
            return False

        try:
            with open(STATE_FILE) as f:
                state = json.load(f)

            self.capital = state['capital']
            self.equity = state['equity']
            self.realized_pnl = state['realized_pnl']
            self.total_trades = state['total_trades']
            self.winning_trades = state['winning_trades']
            self.losing_trades = state['losing_trades']
            self.day_start_equity = state['day_start_equity']
            self.current_date = date.fromisoformat(state['current_date'])
            self.trading_halted = state['trading_halted']

            # Restore open positions as Position dataclass objects
            for symbol, pos_data in state.get('open_positions', {}).items():
                self.open_positions[symbol] = Position(
                    symbol=pos_data['symbol'],
                    entry_price=pos_data['entry_price'],
                    quantity=pos_data['quantity'],
                    stop_loss=pos_data['stop_loss'],
                    take_profit=pos_data['take_profit'],
                    direction=pos_data['direction'],
                    strategy=pos_data['strategy'],
                    regime=pos_data['regime'],
                    entry_time=datetime.fromisoformat(pos_data['entry_time']),
                    votes=pos_data.get('votes', {}),
                )

            last_saved = state.get('last_saved', 'unknown')
            print(f"[risk] State restored from {last_saved}")
            print(f"[risk] Capital: ${self.capital:,.2f} | "
                  f"Open positions: {len(self.open_positions)} | "
                  f"Realized P&L: ${self.realized_pnl:+,.2f}")
            return True

        except Exception as e:
            print(f"[risk] Warning: could not load state ({e}) — starting fresh.")
            return False
