# =============================================================================
# regime.py — Market Regime Detection
# =============================================================================
# WHY: Regime detection is the most intellectually sophisticated component.
# Every strategy has an environment where it thrives and one where it destroys
# capital. Trading MA crossover in a ranging market, or mean reversion in a
# trending market, is a fast way to lose money systematically.
#
# Professional quant funds (Renaissance, Two Sigma, AQR) spend enormous
# resources on regime detection. This module gives you a simplified but
# structurally sound version of that thinking.
#
# THREE REGIMES:
# - TRENDING: Strong directional move. Favor MA Crossover. Suppress Mean Reversion.
# - RANGING: Choppy sideways. Favor RSI + Mean Reversion. Suppress MA Crossover.
# - VOLATILE: High uncertainty spike. Reduce all position sizes. Wait for clarity.

import numpy as np
import pandas as pd
from enum import Enum
import config


class Regime(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"   # Not enough data yet


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14) -> float | None:
    """
    Calculate ADX (Average Directional Index) — measures trend STRENGTH, not direction.

    WHY: ADX was created by J. Welles Wilder (same person who invented RSI).
    - ADX > 25: Strong trend (could be up or down — ADX is directionless)
    - ADX < 25: Weak trend / ranging market
    - Rising ADX: Trend strengthening
    - Falling ADX: Trend weakening

    The calculation involves True Range and Directional Movement (+DM, -DM).
    True Range captures the full price range including gaps from previous close.

    Returns ADX value (0-100), or None if insufficient data.
    """
    n = len(close)
    if n < period + 1:
        return None

    # True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    tr = np.zeros(n)
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)

    for i in range(1, n):
        hl = high[i] - low[i]
        hpc = abs(high[i] - close[i-1])
        lpc = abs(low[i] - close[i-1])
        tr[i] = max(hl, hpc, lpc)

        # +DM: today's high minus yesterday's high (if positive and bigger than -DM)
        up_move = high[i] - high[i-1]
        down_move = low[i-1] - low[i]

        plus_dm[i] = up_move if (up_move > down_move and up_move > 0) else 0
        minus_dm[i] = down_move if (down_move > up_move and down_move > 0) else 0

    # Wilder's smoothed averages (equivalent to EMA with alpha = 1/period)
    atr = np.zeros(n)
    plus_di = np.zeros(n)
    minus_di = np.zeros(n)

    atr[period] = np.sum(tr[1:period+1])
    plus_di[period] = np.sum(plus_dm[1:period+1])
    minus_di[period] = np.sum(minus_dm[1:period+1])

    for i in range(period+1, n):
        atr[i] = atr[i-1] - (atr[i-1] / period) + tr[i]
        plus_di[i] = plus_di[i-1] - (plus_di[i-1] / period) + plus_dm[i]
        minus_di[i] = minus_di[i-1] - (minus_di[i-1] / period) + minus_dm[i]

    # DI+ and DI- as percentages of ATR
    plus_di_pct = 100 * plus_di / np.where(atr != 0, atr, 1)
    minus_di_pct = 100 * minus_di / np.where(atr != 0, atr, 1)

    # DX = |DI+ - DI-| / (DI+ + DI-)
    dx = np.zeros(n)
    for i in range(period, n):
        denom = plus_di_pct[i] + minus_di_pct[i]
        if denom != 0:
            dx[i] = 100 * abs(plus_di_pct[i] - minus_di_pct[i]) / denom

    # ADX = smoothed DX
    adx = np.zeros(n)
    adx[2*period] = np.mean(dx[period:2*period+1])
    for i in range(2*period+1, n):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return float(adx[-1]) if adx[-1] != 0 else None


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
                  period: int = 14) -> tuple[float | None, float | None]:
    """
    Calculate ATR (Average True Range) — measures raw price volatility.

    WHY: ATR tells you the average size of price moves over the period.
    - High ATR: Large price swings, more uncertainty, wider stop-losses needed
    - Low ATR: Tight price action, potentially coiling before a breakout
    - ATR spike: Sudden volatility expansion — often signals regime shift

    We return both current ATR and the 20-period ATR average to detect spikes.
    A spike is defined as current ATR > 1.5x the rolling average.

    Returns (current_atr, atr_average)
    """
    n = len(close)
    if n < period + 1:
        return None, None

    tr = np.zeros(n)
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i-1]),
            abs(low[i] - close[i-1])
        )

    # Current ATR (period-average of True Range)
    current_atr = np.mean(tr[-period:])

    # Baseline ATR (20-period average for spike detection)
    baseline_period = min(20, n - 1)
    atr_avg = np.mean(tr[-baseline_period:]) if n > baseline_period else current_atr

    return float(current_atr), float(atr_avg)


def calculate_bollinger_width(close: np.ndarray, period: int = 20) -> float | None:
    """
    Calculate Bollinger Band Width — measures compression vs expansion.

    WHY: Bollinger Band Width = (Upper Band - Lower Band) / Middle Band
    - Narrow width (squeeze): Low volatility, market coiling — breakout likely soon
    - Wide width: High volatility, momentum move underway
    - Expanding width: Regime is becoming more directional (trending or volatile)
    - Contracting width: Regime is calming into a range

    Used alongside ADX and ATR for multi-confirmation regime classification.

    Returns width as a ratio (e.g., 0.05 = bands are 5% wide relative to price)
    """
    if len(close) < period:
        return None

    rolling = close[-period:]
    mean = np.mean(rolling)
    std = np.std(rolling)

    if mean == 0:
        return None

    upper = mean + (2 * std)
    lower = mean - (2 * std)
    width = (upper - lower) / mean

    return float(width)


def detect_regime(df: pd.DataFrame) -> dict:
    """
    Classify the current market regime for a symbol given its OHLCV history.

    WHY: We use three indicators in combination because each captures a different
    dimension of market structure:
    - ADX: Trend strength (is there a directional move?)
    - ATR spike: Volatility shock (is this an unusual volatility event?)
    - BB Width: Band expansion (is the range expanding or contracting?)

    The regime influences strategy weighting in the engine:
    - TRENDING → trust MA Crossover, suppress Mean Reversion
    - RANGING → trust RSI + Mean Reversion, suppress MA Crossover
    - VOLATILE → reduce position sizes across all strategies, wait for clarity

    Returns a dict with regime, indicator values, and a position size modifier.
    """
    if df.empty or len(df) < 30:
        return {
            'regime': Regime.UNKNOWN,
            'adx': None,
            'atr': None,
            'atr_avg': None,
            'bb_width': None,
            'position_size_modifier': 1.0
        }

    high = df['High'].values
    low = df['Low'].values
    close = df['Close'].values

    adx = calculate_adx(high, low, close)
    atr, atr_avg = calculate_atr(high, low, close)
    bb_width = calculate_bollinger_width(close)

    # --- Regime Classification Logic ---

    # Check for volatility spike first (takes priority over trend classification)
    # WHY: A volatile spike can occur within a trend or range. It signals that
    # normal signal reliability is reduced and capital should be protected.
    is_volatile_spike = (
        atr is not None and
        atr_avg is not None and
        atr_avg > 0 and
        atr > (atr_avg * config.ATR_SPIKE_MULTIPLIER)
    )

    if is_volatile_spike:
        regime = Regime.VOLATILE
        position_modifier = 0.5   # Half position size in volatile regimes
    elif adx is not None and adx > config.ADX_TREND_THRESHOLD:
        regime = Regime.TRENDING
        position_modifier = 1.0   # Full position size in clear trends
    else:
        regime = Regime.RANGING
        position_modifier = 0.75  # Slightly reduced in ranging (mean rev is less reliable)

    return {
        'regime': regime,
        'adx': round(adx, 2) if adx is not None else None,
        'atr': round(atr, 4) if atr is not None else None,
        'atr_avg': round(atr_avg, 4) if atr_avg is not None else None,
        'bb_width': round(bb_width, 4) if bb_width is not None else None,
        'position_size_modifier': position_modifier
    }


def filter_signal_by_regime(signal: str | None, strategy_name: str,
                              regime: Regime) -> str | None:
    """
    Suppress signals from strategies that are mismatched to the current regime.

    WHY: This is the regime filter applied on top of the majority vote. Even if
    2/3 strategies agree, a regime-mismatched signal should be suppressed.

    Rules:
    - TRENDING regime: suppress Mean Reversion signals (trend can extend far)
    - RANGING regime: suppress MA Crossover signals (whipsaws in sideways markets)
    - VOLATILE regime: suppress ALL signals (uncertainty too high for new positions)

    Returns the original signal if regime-compatible, None if suppressed.
    """
    if signal is None:
        return None

    if regime == Regime.VOLATILE:
        # In volatile regimes, no new positions — protect existing capital
        return None

    if regime == Regime.TRENDING and strategy_name == 'Mean_Reversion':
        # Mean reversion in a trend is fighting the tape — extremely dangerous
        return None

    if regime == Regime.RANGING and strategy_name == 'MA_Crossover':
        # MA crossover in a range generates false signals every few bars
        return None

    return signal
