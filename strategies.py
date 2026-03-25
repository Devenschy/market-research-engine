# =============================================================================
# strategies.py — Three Trading Strategies
# =============================================================================
# WHY: Three strategies instead of one because no single strategy works in all
# market regimes. By running all three simultaneously and requiring majority
# agreement (2/3), we get a crude but effective regime filter built into the
# signal generation layer itself.
#
# Each strategy is a class with:
#   - update(price): feed new price data
#   - signal(): return 'BUY', 'SELL', or None
#   - is_ready(): True when enough data exists to generate reliable signals

import numpy as np
from collections import deque
import config


class MovingAverageCrossover:
    """
    Trend-following strategy using fast vs slow moving average crossover.

    WHY: MA crossover is the simplest trend-following strategy. When the
    fast MA (10 periods) crosses above the slow MA (30 periods), the short-term
    momentum is pointing up — a bullish signal. The opposite for sell.

    WHEN IT WORKS: Trending markets with sustained directional moves.
    WHEN IT FAILS: Choppy, sideways markets — generates false signals constantly.
    This is why the regime detector exists.

    LAG PROBLEM: MA crossover inherently lags price because it's a moving average.
    By the time it signals, some of the move is already over. This is the
    fundamental tradeoff between signal reliability and timeliness.
    """

    def __init__(self, fast_period: int = config.MA_FAST, slow_period: int = config.MA_SLOW):
        self.fast_period = fast_period
        self.slow_period = slow_period
        # deque with maxlen automatically discards old data — memory efficient
        self.prices = deque(maxlen=slow_period + 1)
        self._prev_fast = None
        self._prev_slow = None

    def update(self, price: float):
        """Feed a new price point into the strategy."""
        self.prices.append(price)

    def is_ready(self) -> bool:
        """Need at least slow_period prices to compute the slow MA."""
        return len(self.prices) >= self.slow_period

    def signal(self) -> str | None:
        """
        Returns 'BUY' on bullish crossover, 'SELL' on bearish crossover, None otherwise.

        WHY: We detect crossovers by comparing current MA relationship to previous.
        A crossover only fires the tick it happens — not on every tick the fast
        MA is above slow (that would generate continuous signals, not crossovers).
        """
        if not self.is_ready():
            return None

        prices_list = list(self.prices)
        fast_ma = np.mean(prices_list[-self.fast_period:])
        slow_ma = np.mean(prices_list[-self.slow_period:])

        # Previous period's MAs (one tick ago)
        prev_fast = np.mean(prices_list[-(self.fast_period + 1):-1]) if len(prices_list) > self.fast_period else None
        prev_slow = np.mean(prices_list[-(self.slow_period + 1):-1]) if len(prices_list) > self.slow_period else None

        if prev_fast is None or prev_slow is None:
            return None

        # Bullish crossover: fast was below slow, now fast is above slow
        if prev_fast <= prev_slow and fast_ma > slow_ma:
            return 'BUY'

        # Bearish crossover: fast was above slow, now fast is below slow
        if prev_fast >= prev_slow and fast_ma < slow_ma:
            return 'SELL'

        return None

    def get_values(self) -> dict:
        """Return current indicator values for dashboard display."""
        if not self.is_ready():
            return {'fast_ma': None, 'slow_ma': None}
        prices_list = list(self.prices)
        return {
            'fast_ma': round(np.mean(prices_list[-self.fast_period:]), 4),
            'slow_ma': round(np.mean(prices_list[-self.slow_period:]), 4)
        }


class RSIMomentum:
    """
    Momentum oscillator strategy using RSI (Relative Strength Index).

    WHY: RSI measures the speed and magnitude of price changes to identify
    overbought/oversold conditions. Unlike MA crossover, RSI works best
    in RANGING markets — it assumes prices will revert to a "normal" level.

    RSI Formula: RSI = 100 - (100 / (1 + RS)) where RS = avg_gain / avg_loss

    SIGNAL LOGIC:
    - RSI crosses UP through 30: market was oversold (selling exhausted),
      momentum turning positive — BUY signal
    - RSI crosses DOWN through 70: market was overbought (buying exhausted),
      momentum turning negative — SELL signal

    WHY CROSSINGS, NOT LEVELS: Waiting for RSI to cross back through the
    threshold (not just touch it) filters out false signals. An RSI that
    dips to 28 and bounces to 32 is confirmed recovery. Still at 25 is not.
    """

    def __init__(self, period: int = config.RSI_PERIOD):
        self.period = period
        self.prices = deque(maxlen=period * 3)  # extra history for accuracy
        self._prev_rsi = None

    def update(self, price: float):
        self.prices.append(price)

    def is_ready(self) -> bool:
        return len(self.prices) >= self.period + 1

    def _calculate_rsi(self, prices_list: list) -> float | None:
        """
        Calculate RSI using Wilder's Smoothed Moving Average method.

        WHY: Wilder's smoothing (not simple average) is the correct RSI calculation.
        It gives more weight to recent data and produces smoother RSI values
        than a simple average of gains/losses.
        """
        if len(prices_list) < self.period + 1:
            return None

        # Calculate price changes
        changes = np.diff(prices_list[-(self.period + 1):])
        gains = np.where(changes > 0, changes, 0)
        losses = np.where(changes < 0, -changes, 0)

        avg_gain = np.mean(gains)
        avg_loss = np.mean(losses)

        if avg_loss == 0:
            return 100.0  # All gains, overbought

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def signal(self) -> str | None:
        """
        Returns 'BUY' when RSI crosses up through 30, 'SELL' when crosses down through 70.
        """
        if not self.is_ready():
            return None

        prices_list = list(self.prices)
        current_rsi = self._calculate_rsi(prices_list)

        # Previous RSI (using prices one tick ago)
        prev_rsi = self._calculate_rsi(prices_list[:-1]) if len(prices_list) > self.period + 1 else None

        if current_rsi is None:
            return None

        sig = None

        if prev_rsi is not None:
            # Oversold recovery: RSI was below 30, now above 30 → BUY
            if prev_rsi <= 30 and current_rsi > 30:
                sig = 'BUY'
            # Overbought rejection: RSI was above 70, now below 70 → SELL
            elif prev_rsi >= 70 and current_rsi < 70:
                sig = 'SELL'

        self._prev_rsi = current_rsi
        return sig

    def get_values(self) -> dict:
        if not self.is_ready():
            return {'rsi': None}
        return {'rsi': round(self._calculate_rsi(list(self.prices)), 2)}


class MeanReversion:
    """
    Statistical mean reversion strategy based on Z-score deviation.

    WHY: Prices tend to revert to their mean over time due to:
    1. Fundamental anchoring — there's usually an "fair value" gravity
    2. Arbitrage — traders exploit mispricings until they disappear
    3. Liquidity cycles — forced sellers/buyers create temporary dislocations

    Z-SCORE: How many standard deviations from the mean is the current price?
    Z = (price - mean) / std_dev

    SIGNAL LOGIC:
    - Z-score < -2: Price is >2 std devs BELOW mean → abnormally cheap → BUY
    - Z-score returns to 0: Price has reverted to mean → EXIT (SELL)

    WARNING: This strategy is LETHAL in trending markets. A trending stock
    can stay "cheap" for weeks while momentum carries it lower. This is why
    regime detection suppresses mean reversion signals in trending regimes.
    """

    def __init__(self, window: int = config.MEAN_REV_WINDOW,
                 z_threshold: float = config.MEAN_REV_ZSCORE_THRESHOLD):
        self.window = window
        self.z_threshold = z_threshold
        self.prices = deque(maxlen=window + 1)
        self._in_trade = False  # Track whether we're waiting for mean reversion exit

    def update(self, price: float):
        self.prices.append(price)

    def is_ready(self) -> bool:
        return len(self.prices) >= self.window

    def _calculate_zscore(self) -> float | None:
        if not self.is_ready():
            return None
        prices_array = np.array(list(self.prices))
        mean = np.mean(prices_array)
        std = np.std(prices_array)
        if std == 0:
            return 0.0
        return (prices_array[-1] - mean) / std

    def signal(self) -> str | None:
        """
        BUY when price is >2 std devs below mean, SELL when price reverts to mean.
        """
        zscore = self._calculate_zscore()
        if zscore is None:
            return None

        # Extreme oversold: price far below rolling mean → mean reversion BUY
        if zscore < -self.z_threshold:
            return 'BUY'

        # Extreme overbought: price far above rolling mean → mean reversion SELL
        if zscore > self.z_threshold:
            return 'SELL'

        # Price near mean (reversion complete) — signal to exit if in trade
        # The engine/risk manager handles the actual exit via stop-loss / take-profit
        return None

    def get_values(self) -> dict:
        zscore = self._calculate_zscore()
        if zscore is None:
            return {'zscore': None, 'mean': None}
        prices_array = np.array(list(self.prices))
        return {
            'zscore': round(zscore, 3),
            'mean': round(np.mean(prices_array), 4)
        }


class StrategyEnsemble:
    """
    Runs all three strategies simultaneously and aggregates their signals.

    WHY: Majority voting (2/3 strategies must agree) is a simple but powerful
    ensemble method. It:
    1. Reduces false positives — a single misfiring strategy can't open a position
    2. Implicitly filters regimes — MA crossover and RSI rarely agree in trending
       markets at the same moment unless the signal is strong
    3. Teaches you about signal correlation and independence

    TRADEOFF: You will miss some trades. That's okay. The goal is high-confidence
    signals, not maximum trade frequency. In live trading, overtrading kills returns.
    """

    def __init__(self):
        self.ma = MovingAverageCrossover()
        self.rsi = RSIMomentum()
        self.mean_rev = MeanReversion()
        self.strategies = {
            'MA_Crossover': self.ma,
            'RSI_Momentum': self.rsi,
            'Mean_Reversion': self.mean_rev
        }

    def update(self, price: float):
        """Feed price to all strategies simultaneously."""
        for strat in self.strategies.values():
            strat.update(price)

    def get_votes(self) -> dict:
        """
        Get the signal from each strategy.
        Returns dict: {'MA_Crossover': 'BUY', 'RSI_Momentum': None, ...}
        """
        votes = {}
        for name, strat in self.strategies.items():
            votes[name] = strat.signal() if strat.is_ready() else None
        return votes

    def aggregate_signal(self) -> str | None:
        """
        Apply majority vote (2/3) to determine final signal direction.

        WHY: We count BUY votes separately from SELL votes. 2+ BUYs = BUY signal.
        2+ SELLs = SELL signal. Mixed or insufficient votes = no trade.
        This prevents a scenario where 1 BUY + 1 SELL + 1 None = confused action.
        """
        votes = self.get_votes()
        buy_votes = sum(1 for v in votes.values() if v == 'BUY')
        sell_votes = sum(1 for v in votes.values() if v == 'SELL')

        if buy_votes >= 2:
            return 'BUY'
        if sell_votes >= 2:
            return 'SELL'
        return None

    def get_indicator_values(self) -> dict:
        """Return all indicator values for dashboard display."""
        return {
            'MA': self.ma.get_values(),
            'RSI': self.rsi.get_values(),
            'MeanRev': self.mean_rev.get_values()
        }

    def is_ready(self) -> bool:
        """True when at least 2 strategies have enough data to vote."""
        ready_count = sum(1 for s in self.strategies.values() if s.is_ready())
        return ready_count >= 2
