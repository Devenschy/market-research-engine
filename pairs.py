# =============================================================================
# pairs.py — Statistical Arbitrage / Pairs Trading
# =============================================================================
#
# WHAT IS PAIRS TRADING?
# ---------------------
# Pairs trading was invented at Morgan Stanley in the 1980s by a quantitative
# research group led by Nunzio Tartaglia. The core insight is simple but powerful:
#
#   Two stocks that have historically moved together will CONTINUE to move
#   together — and when they temporarily diverge, they will snap back.
#
# You make money by:
#   1. Identifying a pair of assets that are statistically linked (cointegrated)
#   2. Measuring how far apart they currently are (the "spread")
#   3. When the spread widens abnormally: BUY the underperformer, SHORT the outperformer
#   4. Profit when the spread returns to its historical average
#
# WHY THIS STRATEGY IS POWERFUL:
# --------------------------------
# It is MARKET-NEUTRAL. You don't care if the market goes up or down.
# If both stocks crash together, you're long one and short the other — so you're
# roughly flat. You only profit from the RELATIVE move between them.
# This is why long/short equity hedge funds can survive market crashes.
#
# The strategy generates "statistical arbitrage alpha" — edge that comes from
# exploiting temporary mispricings rather than predicting market direction.
#
# HISTORICAL CONTEXT:
# -------------------
# Tartaglia's team at Morgan Stanley reportedly made $50 million in their first
# year with this strategy in 1987. It's now widely used by quant funds including
# Renaissance Technologies, Two Sigma, and D.E. Shaw.
#
# HOW TO READ THE SIGNALS:
# -------------------------
# z-score > +2.0 : Symbol1 is expensive relative to Symbol2
#                  → SHORT Symbol1, LONG Symbol2 (they will converge)
# z-score < -2.0 : Symbol1 is cheap relative to Symbol2
#                  → LONG Symbol1, SHORT Symbol2
# |z-score| < 0.5: Spread has converged → EXIT the position, take profit

import numpy as np
import pandas as pd
import yfinance as yf
import config

# Try to import statsmodels — it provides the Engle-Granger cointegration test.
# statsmodels is a standard scientific Python library (installed with pandas usually).
# If it's not installed: pip install statsmodels
try:
    from statsmodels.tsa.stattools import coint
    from statsmodels.regression.linear_model import OLS
    from statsmodels.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

# ---------------------------------------------------------------------------
# PAIRS UNIVERSE — The pairs we analyze
# ---------------------------------------------------------------------------
# Each pair was chosen for a different reason:
#
# AAPL / MSFT: Two tech giants with overlapping revenue drivers (enterprise,
#   consumer hardware, cloud). Their stocks have historically been highly
#   correlated. Institutional rotation between them creates tradeable spreads.
#
# BTC / ETH: Crypto pairs move together on macro crypto sentiment (risk-on/off,
#   regulatory news). ETH tends to outperform in "altcoin seasons" and
#   underperform when Bitcoin dominance rises. The spread is tradeable.
#
# GC=F / CL=F (Gold / Crude Oil): Both are inflation hedges and are priced
#   in USD. Rising USD weakens both. Geopolitical risk lifts oil more than gold.
#   The macro-driven correlation makes the spread statistically meaningful.
PAIRS = config.PAIRS


# =============================================================================
# FUNCTION 1: Fetch historical prices for both symbols
# =============================================================================

def fetch_pair_history(symbol1: str, symbol2: str, period: str = '90d') -> tuple:
    """
    Fetch closing price history for both symbols and align them on common trading days.

    WHY 90 DAYS?
    A longer window gives more reliable cointegration statistics, but the relationship
    may have changed if you go back too far. 90 days (≈3 months) balances recency
    and statistical power. Professional pairs traders often use 6-12 months.

    WHY INNER JOIN?
    Symbol1 and Symbol2 may have different trading calendars. For example:
    - US equities don't trade on US holidays
    - Crypto trades 24/7
    - Futures have different settlement days
    Inner join keeps only dates where BOTH symbols have prices, ensuring the
    spread calculation is always based on the same date.

    Returns:
        (prices1, prices2): Two pd.Series aligned on common dates.
        Returns (None, None) if fetch fails.
    """
    try:
        # Download both symbols in a single API call — more efficient
        data = yf.download(
            [symbol1, symbol2],
            period=period,
            auto_adjust=True,    # Adjusts for splits and dividends automatically
            progress=False       # Suppress download progress bar
        )

        # yfinance returns a MultiIndex DataFrame when fetching multiple symbols.
        # Structure: data['Close'][symbol1] gives the closing price series.
        if data.empty:
            return None, None

        # Extract closing prices for each symbol
        if isinstance(data.columns, pd.MultiIndex):
            prices1 = data['Close'][symbol1].dropna()
            prices2 = data['Close'][symbol2].dropna()
        else:
            # Single symbol case (shouldn't happen here, but defensive coding)
            return None, None

        # Align on common dates — inner join keeps only rows where both have data
        # This is critical: the spread only makes sense when both prices exist
        aligned = pd.concat([prices1, prices2], axis=1, join='inner')
        aligned.columns = [symbol1, symbol2]
        aligned = aligned.dropna()

        if len(aligned) < 30:
            # Not enough data for reliable statistics — return None
            # WHY 30? Standard minimum for statistical tests to have any power.
            return None, None

        return aligned[symbol1], aligned[symbol2]

    except Exception as e:
        print(f"[pairs] Error fetching {symbol1}/{symbol2}: {e}")
        return None, None


# =============================================================================
# FUNCTION 2: Pearson Correlation
# =============================================================================

def calculate_correlation(prices1: pd.Series, prices2: pd.Series) -> float:
    """
    Calculate the Pearson correlation coefficient between two price series.

    WHAT CORRELATION TELLS YOU:
    - +1.0: Perfect positive correlation — they move identically
    - +0.8 to +1.0: Strong positive correlation — good pairs trading candidate
    - +0.5 to +0.8: Moderate — less reliable spread behavior
    - Below +0.5: Weak or no correlation — not suitable for pairs trading

    CRITICAL WARNING — CORRELATION IS NOT ENOUGH:
    -----------------------------------------------
    Correlation > 0.8 is NECESSARY but NOT SUFFICIENT for pairs trading.
    Two stocks can both be in long-term uptrends (highly correlated) but their
    SPREAD keeps growing — it never reverts. This is called a "spurious correlation."

    Example: AAPL and any random tech stock in 2020 were both going up.
    Correlation was high, but the spread between them wasn't mean-reverting.

    You need COINTEGRATION (see test_cointegration below) to confirm the spread
    actually reverts to a mean. Use correlation as a quick filter, not a final test.

    Returns:
        float: Correlation coefficient between -1.0 and +1.0
    """
    try:
        # Pearson correlation on log returns is more statistically robust than
        # correlating raw prices, but for pairs trading we use prices directly
        # because we care about the LEVEL spread, not return correlation.
        correlation = prices1.corr(prices2)
        return round(float(correlation), 4)
    except Exception:
        return 0.0


# =============================================================================
# FUNCTION 3: Engle-Granger Cointegration Test
# =============================================================================

def test_cointegration(prices1: pd.Series, prices2: pd.Series) -> dict:
    """
    Test whether two price series are cointegrated using the Engle-Granger test.

    COINTEGRATION vs CORRELATION — THE FUNDAMENTAL DIFFERENCE:
    -----------------------------------------------------------
    Correlation: "Do these two series move in the same DIRECTION at the same TIME?"
        - Measured on returns (daily/weekly changes)
        - High correlation means they tend to go up/down together
        - Problem: You can have high correlation between two trending series
          whose SPREAD keeps growing forever (not suitable for pairs trading)

    Cointegration: "Does the SPREAD between these two series revert to a mean?"
        - Measured on PRICE LEVELS, not returns
        - A cointegrated pair has a stationary spread (bounded, mean-reverting)
        - The spread can diverge temporarily but is mathematically "pulled back"
        - This is exactly what you need for pairs trading to be profitable

    ANALOGY: A drunk man and his dog going for a walk.
        - Both wander randomly (non-stationary paths)
        - But they are COINTEGRATED — the leash keeps the distance bounded
        - If the dog runs too far ahead, the man will catch up (mean reversion)
        - If you only looked at their direction changes (correlation), you'd miss the leash!

    THE ENGLE-GRANGER TEST (1987 Nobel Prize):
    ------------------------------------------
    1. Regress prices1 on prices2 to find the linear relationship: prices1 = α + β * prices2
    2. Calculate the residuals (the "spread"): residuals = prices1 - α - β * prices2
    3. Test whether residuals are stationary (ADF test on residuals)
    4. If residuals are stationary → cointegrated → spread is mean-reverting → pairs trade!

    p_value < 0.05 means we can reject the null hypothesis (no cointegration)
    at the 95% confidence level. The pair IS cointegrated.

    THE HEDGE RATIO (β):
    --------------------
    The hedge ratio tells you HOW MANY UNITS of symbol2 to hold per unit of symbol1.
    Example: if β = 1.5, for every 1 share of AAPL you long, you short 1.5 shares of MSFT.
    This makes the position "dollar-neutral" relative to the statistical relationship.

    Returns:
        dict with keys: cointegrated (bool), p_value (float), hedge_ratio (float)
    """
    if not STATSMODELS_AVAILABLE:
        return {
            'cointegrated': False,
            'p_value': 1.0,
            'hedge_ratio': 1.0,
            'error': 'statsmodels not installed. Run: pip install statsmodels'
        }

    try:
        # STEP 1: Engle-Granger cointegration test
        # coint() tests: H0 = no cointegration, H1 = cointegrated
        # Returns: (test_statistic, p_value, critical_values)
        _, p_value, _ = coint(prices1, prices2)

        # STEP 2: Calculate the hedge ratio via OLS regression
        # We regress prices1 on prices2: prices1 = alpha + beta * prices2
        # The slope (beta) IS the hedge ratio — it tells us the proportional
        # relationship between the two price series at their "equilibrium"
        X = add_constant(prices2.values)   # Add intercept column
        y = prices1.values
        model = OLS(y, X).fit()
        hedge_ratio = model.params[1]       # Coefficient on prices2 = beta

        # p_value < 0.05 = statistically significant cointegration at 95% confidence
        is_cointegrated = bool(p_value < 0.05)

        return {
            'cointegrated': is_cointegrated,
            'p_value': round(float(p_value), 4),
            'hedge_ratio': round(float(hedge_ratio), 4)
        }

    except Exception as e:
        return {
            'cointegrated': False,
            'p_value': 1.0,
            'hedge_ratio': 1.0,
            'error': str(e)
        }


# =============================================================================
# FUNCTION 4: Calculate the Spread
# =============================================================================

def calculate_spread(prices1: pd.Series, prices2: pd.Series, hedge_ratio: float) -> pd.Series:
    """
    Calculate the hedge-ratio-adjusted spread between two price series.

    WHAT IS THE SPREAD?
    -------------------
    The spread is the "distance" between the two assets after accounting for
    their proportional relationship (hedge ratio).

    spread = prices1 - (hedge_ratio * prices2)

    WHY NOT JUST PRICES1 - PRICES2?
    --------------------------------
    Raw price difference ignores the economic relationship between the assets.
    If AAPL is at $180 and MSFT is at $420, the raw spread of -$240 is meaningless.
    The hedge ratio tells us that (for example) 1 share of AAPL corresponds to
    0.43 shares of MSFT. So we compare AAPL to 0.43 * MSFT.

    When the spread INCREASES (prices1 is expensive relative to prices2):
    → Short prices1, Long prices2 — expect convergence
    When the spread DECREASES (prices1 is cheap relative to prices2):
    → Long prices1, Short prices2 — expect convergence

    Returns:
        pd.Series: The spread time series (same index as input prices)
    """
    spread = prices1 - (hedge_ratio * prices2)
    spread.name = 'spread'
    return spread


# =============================================================================
# FUNCTION 5: Z-Score of the Spread
# =============================================================================

def calculate_zscore(spread: pd.Series, window: int = 30) -> pd.Series:
    """
    Calculate the rolling z-score of the spread.

    WHAT IS A Z-SCORE?
    ------------------
    Z-score measures how many standard deviations the current spread is from
    its recent rolling mean:

        z = (spread - rolling_mean) / rolling_std

    WHY ROLLING (not full-history)?
    --------------------------------
    The absolute level of the spread can drift over time due to structural
    changes (new products, management, macro shifts). A rolling window (30 days)
    means we're asking: "Is the spread unusual relative to the RECENT past?"
    This is more actionable than comparing to a 2-year average that may be stale.

    INTERPRETING THE Z-SCORE:
    -------------------------
    z > +2.0 : Spread is 2 standard deviations ABOVE its recent mean
               → prices1 is expensive relative to prices2
               → SHORT prices1, LONG prices2
    z < -2.0 : Spread is 2 standard deviations BELOW its recent mean
               → prices1 is cheap relative to prices2
               → LONG prices1, SHORT prices2
    |z| < 0.5: Spread is near its mean → convergence has occurred → EXIT

    WHY ±2.0 AS THRESHOLD?
    ----------------------
    ±2.0 standard deviations corresponds to the outermost ~5% of a normal
    distribution. By only trading when the spread is this extreme, we ensure
    we're acting on genuine statistical dislocations, not random noise.
    Most professional pairs traders use 2.0 as the entry trigger.

    Returns:
        pd.Series: Rolling z-score of the spread
    """
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()

    # Avoid division by zero — if std is 0, the spread hasn't moved (no signal)
    zscore = (spread - rolling_mean) / rolling_std.replace(0, np.nan)
    zscore.name = 'zscore'
    return zscore


# =============================================================================
# FUNCTION 6: Full Signal for a Single Pair
# =============================================================================

def get_pair_signal(symbol1: str, symbol2: str) -> dict:
    """
    Run the complete pairs trading analysis for a single pair.

    PIPELINE:
    1. Fetch price history for both symbols
    2. Calculate correlation (quick filter)
    3. Test cointegration (proper statistical test)
    4. Calculate spread and z-score
    5. Generate a trading signal based on z-score thresholds

    SIGNAL LOGIC:
    -------------
    We use z-score thresholds from config.py:
    - Entry: |z| > PAIRS_ZSCORE_ENTRY (default 2.0)
    - Exit:  |z| < PAIRS_ZSCORE_EXIT  (default 0.5)

    IMPORTANT: We only trade cointegrated pairs!
    Even if the z-score is extreme, without cointegration there's no statistical
    reason to expect mean reversion. Trading non-cointegrated pairs on z-score
    alone is like betting on coin flips — no edge.

    Returns:
        dict with full analysis results including signal and plain-English explanation
    """
    # Pull thresholds from config — single source of truth
    zscore_entry = config.PAIRS_ZSCORE_ENTRY
    zscore_exit = config.PAIRS_ZSCORE_EXIT

    # --- Step 1: Fetch data ---
    prices1, prices2 = fetch_pair_history(symbol1, symbol2, period=config.PAIRS_LOOKBACK)

    if prices1 is None or prices2 is None:
        return {
            'symbol1': symbol1,
            'symbol2': symbol2,
            'error': 'Failed to fetch price data',
            'cointegrated': False,
            'p_value': None,
            'correlation': None,
            'hedge_ratio': None,
            'current_spread': None,
            'zscore': None,
            'signal': None,
            'signal_reason': 'Could not fetch data',
            'zscore_entry': zscore_entry,
            'zscore_exit': zscore_exit,
        }

    # --- Step 2: Correlation (quick filter) ---
    correlation = calculate_correlation(prices1, prices2)

    # --- Step 3: Cointegration test (real statistical test) ---
    coint_result = test_cointegration(prices1, prices2)
    is_cointegrated = coint_result.get('cointegrated', False)
    p_value = coint_result.get('p_value', 1.0)
    hedge_ratio = coint_result.get('hedge_ratio', 1.0)

    # --- Step 4: Spread and z-score ---
    spread = calculate_spread(prices1, prices2, hedge_ratio)
    zscore_series = calculate_zscore(spread)

    # Get the most recent z-score and spread value
    current_zscore = float(zscore_series.dropna().iloc[-1]) if not zscore_series.dropna().empty else None
    current_spread = float(spread.iloc[-1]) if not spread.empty else None

    # --- Step 5: Generate signal ---
    signal = None
    signal_reason = 'No signal'

    if current_zscore is not None:
        if not is_cointegrated:
            # No cointegration = no statistical edge, don't trade
            signal_reason = (
                f"Pair not cointegrated (p-value={p_value:.3f} > 0.05). "
                f"Z-score={current_zscore:.2f} but without cointegration, "
                f"there is no statistical reason to expect mean reversion."
            )
        elif current_zscore > zscore_entry:
            # Spread is wide: symbol1 expensive relative to symbol2
            signal = 'SHORT_1_LONG_2'
            signal_reason = (
                f"Z-score={current_zscore:.2f} > {zscore_entry}. "
                f"{symbol1} is trading expensive relative to {symbol2} "
                f"(spread is {current_zscore:.1f} standard deviations above its mean). "
                f"Statistical expectation: spread will revert downward. "
                f"Action: SHORT {symbol1}, LONG {symbol2}."
            )
        elif current_zscore < -zscore_entry:
            # Spread is wide (other direction): symbol1 cheap relative to symbol2
            signal = 'LONG_1_SHORT_2'
            signal_reason = (
                f"Z-score={current_zscore:.2f} < -{zscore_entry}. "
                f"{symbol1} is trading cheap relative to {symbol2} "
                f"(spread is {abs(current_zscore):.1f} standard deviations below its mean). "
                f"Statistical expectation: spread will revert upward. "
                f"Action: LONG {symbol1}, SHORT {symbol2}."
            )
        elif abs(current_zscore) < zscore_exit:
            # Spread has converged — exit any open position
            signal_reason = (
                f"Z-score={current_zscore:.2f} is near zero (|z| < {zscore_exit}). "
                f"Spread has converged to its mean. "
                f"EXIT signal: close any open pairs position for this pair."
            )
        else:
            # In between entry and exit thresholds — no action
            signal_reason = (
                f"Z-score={current_zscore:.2f} is between entry ({zscore_entry}) "
                f"and exit ({zscore_exit}) thresholds. No new position, hold if open."
            )

    return {
        'symbol1': symbol1,
        'symbol2': symbol2,
        'cointegrated': is_cointegrated,
        'p_value': p_value,
        'correlation': correlation,
        'hedge_ratio': hedge_ratio,
        'current_spread': round(current_spread, 4) if current_spread is not None else None,
        'zscore': round(current_zscore, 3) if current_zscore is not None else None,
        'signal': signal,
        'signal_reason': signal_reason,
        'zscore_entry': zscore_entry,
        'zscore_exit': zscore_exit,
    }


# =============================================================================
# FUNCTION 7: Analyze All Pairs
# =============================================================================

def analyze_all_pairs() -> list:
    """
    Run the full pairs trading analysis on every pair in the PAIRS universe.

    Sorting by |z-score| descending means the most extreme (highest opportunity)
    pairs appear first. This is useful when scanning many pairs — you want to
    see the biggest dislocations at the top.

    WHY CATCH ERRORS PER PAIR?
    --------------------------
    If one pair fails (e.g., a crypto pair has a data outage), we don't want the
    entire analysis to crash. Graceful degradation — continue with the other pairs
    and report the error in that pair's result. This is essential for production code.

    Returns:
        list of dicts, sorted by |zscore| descending (biggest opportunity first)
    """
    results = []

    for symbol1, symbol2 in PAIRS:
        try:
            result = get_pair_signal(symbol1, symbol2)
            results.append(result)
        except Exception as e:
            results.append({
                'symbol1': symbol1,
                'symbol2': symbol2,
                'error': str(e),
                'cointegrated': False,
                'zscore': None,
                'signal': None,
                'signal_reason': f'Analysis failed: {e}'
            })

    # Sort by absolute z-score — biggest opportunities first
    # Pairs with no z-score (errors) go to the end
    results.sort(
        key=lambda x: abs(x.get('zscore') or 0),
        reverse=True
    )

    return results


# =============================================================================
# FUNCTION 8: Pair Performance Summary (Backtest-style)
# =============================================================================

def get_pair_performance_summary(prices1: pd.Series, prices2: pd.Series, hedge_ratio: float) -> dict:
    """
    Summarize historical spread behavior to evaluate the quality of the pair.

    WHY THIS MATTERS FOR SIZING AND TIMING:
    ----------------------------------------
    Before trading a pair, you want to know:
    1. How stable is the spread? (mean_spread, std_spread)
    2. How QUICKLY does it revert? (half_life)

    A pair that takes 60 days to revert has high capital tie-up risk.
    A pair that reverts in 5 days is much more capital-efficient.
    Professional pairs traders target half-lives of 5-20 days.

    THE HALF-LIFE CONCEPT:
    ----------------------
    If you treat the spread as an Ornstein-Uhlenbeck mean-reverting process,
    the "half-life" is how long it takes for a deviation to decay by half.

    Example: half-life = 7 days means:
    - Day 0: Spread is 2.0 std devs from mean (entry point)
    - Day 7: Spread is expected to be ~1.0 std devs from mean
    - Day 14: Spread is expected to be ~0.5 std devs from mean (near exit)

    So the expected hold time is approximately 2 × half-life.

    HALF-LIFE FORMULA:
    ------------------
    Fit an AR(1) model: Δspread_t = φ × spread_{t-1} + ε
    where Δspread_t = spread_t - spread_{t-1}

    The AR(1) coefficient φ (phi) tells you the "pull back" strength per period.
    φ = -1.0 means instant reversion, φ = 0.0 means random walk (no reversion).
    φ between -1 and 0 means mean-reverting with some persistence.

    half_life = -log(2) / log(1 + φ)

    A half-life of 5-20 days is ideal. >60 days is slow/risky.

    Returns:
        dict: mean_spread, std_spread, half_life (in days), num_days
    """
    try:
        spread = calculate_spread(prices1, prices2, hedge_ratio)

        mean_spread = float(spread.mean())
        std_spread = float(spread.std())

        # --- AR(1) regression to estimate half-life ---
        # Δspread_t = φ × spread_{t-1} + ε
        # We regress the CHANGE in spread on the LAGGED spread level

        spread_lag = spread.shift(1)           # spread at t-1
        delta_spread = spread.diff()           # spread_t - spread_{t-1}

        # Align — drop NaN from differencing/shifting
        reg_data = pd.concat([delta_spread, spread_lag], axis=1).dropna()
        reg_data.columns = ['delta', 'lag']

        if len(reg_data) < 10:
            half_life = None
        else:
            # OLS: delta_spread = phi * spread_lag (no intercept needed for mean-reverting process)
            if STATSMODELS_AVAILABLE:
                X = add_constant(reg_data['lag'].values)
                y = reg_data['delta'].values
                model = OLS(y, X).fit()
                phi = model.params[1]   # AR(1) coefficient

                # Half-life formula: how many periods for deviation to halve
                # Only valid when phi is negative (mean-reverting) and > -1 (not explosive)
                if -1.0 < phi < 0.0:
                    half_life = round(-np.log(2) / np.log(1 + phi), 1)
                else:
                    # phi >= 0: spread is diverging (not mean-reverting) — bad pair
                    # phi <= -1: explosive reversion — unusual / data issue
                    half_life = None
            else:
                # Fallback: simple estimate using correlation
                half_life = None

        return {
            'mean_spread': round(mean_spread, 4),
            'std_spread': round(std_spread, 4),
            'half_life': half_life,    # Days — None if not mean-reverting
            'num_days': len(spread),
            'interpretation': (
                f"Spread mean={mean_spread:.3f}, std={std_spread:.3f}. "
                + (f"Half-life ≈ {half_life} days — expect to hold position ~{int(half_life*2)} days on average."
                   if half_life else "Could not estimate half-life (spread may not be mean-reverting).")
            )
        }

    except Exception as e:
        return {
            'mean_spread': None,
            'std_spread': None,
            'half_life': None,
            'num_days': 0,
            'error': str(e)
        }


# =============================================================================
# QUICK TEST — Run this file directly to see output
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("PAIRS TRADING ANALYSIS")
    print("=" * 60)

    results = analyze_all_pairs()

    for r in results:
        print(f"\n{r['symbol1']} / {r['symbol2']}")
        print(f"  Cointegrated : {r.get('cointegrated')} (p={r.get('p_value')})")
        print(f"  Correlation  : {r.get('correlation')}")
        print(f"  Hedge Ratio  : {r.get('hedge_ratio')}")
        print(f"  Z-Score      : {r.get('zscore')}")
        print(f"  Signal       : {r.get('signal')}")
        print(f"  Reason       : {r.get('signal_reason', '')[:120]}")
