# =============================================================================
# factors.py — Factor Investing Module
# =============================================================================
#
# WHAT IS FACTOR INVESTING?
# ==========================
# Factor investing is how the world's largest quantitative hedge funds —
# AQR Capital, Renaissance Technologies, Two Sigma, Dimensional Fund Advisors —
# generate systematic, repeatable alpha (returns above the market benchmark).
#
# Instead of trying to predict whether AAPL will be up or down tomorrow (which
# is near-impossible), factor investors ask: "Across thousands of stocks and
# decades of data, which CHARACTERISTICS reliably predict future outperformance?"
#
# The answer from 50+ years of academic research: there are persistent market
# anomalies that survive out-of-sample testing across countries and time periods.
# These are called "factors." The four most researched and robustly documented:
#
#   1. MOMENTUM — Recent winners keep winning (short term)
#      Research: Jegadeesh & Titman (1993) — Nobel Prize level work
#      Why it persists: Investors underreact to good news. Institutions are
#      slow to build positions. Trend-following creates self-reinforcing flows.
#
#   2. VALUE — Cheap assets outperform expensive ones (long term)
#      Research: Fama & French (1992) — most cited paper in finance history
#      Why it persists: Investors systematically overpay for exciting "growth"
#      stories and underpay for boring, profitable companies. Behavioral bias.
#
#   3. QUALITY — Strong balance sheets, high ROE, low debt outperform
#      Research: Sloan (1996) accruals anomaly, Novy-Marx (2013) profitability
#      Why it persists: Earnings quality is hard to analyze — most investors
#      don't do deep fundamental work, so quality stocks are often underpriced.
#
#   4. LOW VOLATILITY — Lower-risk assets outperform higher-risk ones
#      Research: Black (1972) — documented before CAPM even took hold
#      Why it persists: Institutional managers benchmarked against indices
#      chase high-beta stocks for career reasons. This overprices risky stocks
#      and leaves low-volatility stocks systematically undervalued. Pure anomaly.
#
# HOW TO USE FACTORS PROFESSIONALLY:
# - Long/short equity: Long top-quintile by factor, short bottom-quintile
# - Factor tilts: Overweight factor-positive stocks in a passive portfolio
# - Signal confluence: Require multiple factors to agree before taking a position
# - Factor rotation: Different factors outperform in different macro regimes
#   (value does well in recovery, momentum in bull markets, quality in downturns)
#
# THE FACTOR ZOO WARNING:
# Academic researchers have published 300+ "factors." Most are false positives
# from data mining. The four implemented here have the strongest theoretical
# grounding, longest track records, and survive out-of-sample testing globally.
# ==============================================================================

import yfinance as yf
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import config


# =============================================================================
# HELPER: Identify asset class (equities-only factors need this guard)
# =============================================================================

def _is_equity(symbol: str) -> bool:
    """
    Rough heuristic to detect if a symbol is an equity (stock).

    WHY: The value and quality factors only make sense for equities. P/E ratios
    and ROE don't exist for Bitcoin, gold, or EUR/USD. Rather than calling the
    yfinance API and failing gracefully every time, we do a fast string check
    first as a pre-filter. This saves API calls and makes the logic explicit.

    The convention in yfinance:
    - Crypto: ends in '-USD' or '-USDT' (e.g., BTC-USD, ETH-USD)
    - Forex: ends in '=X' (e.g., EURUSD=X, GBPUSD=X)
    - Futures: ends in '=F' (e.g., GC=F for gold, CL=F for crude oil)
    - Equities: plain ticker symbols with no suffix (AAPL, MSFT, TSLA)
    """
    if symbol.endswith('-USD') or symbol.endswith('-USDT'):
        return False  # Crypto
    if symbol.endswith('=X'):
        return False  # Forex
    if symbol.endswith('=F'):
        return False  # Futures / Commodities
    return True       # Assume equity


# =============================================================================
# FACTOR 1: MOMENTUM
# =============================================================================

def calculate_momentum_score(symbol: str, period_months: int = 12, skip_months: int = 1) -> dict | None:
    """
    Calculate the classic academic momentum factor for a given symbol.

    WHY THE SKIP-MONTH MATTERS (this is the most important subtlety):
    ===================================================================
    The original Jegadeesh & Titman (1993) paper discovered that 12-month
    momentum is a strong predictor of future returns — BUT only if you skip
    the most recent month. This counterintuitive design feature exists because:

    - SHORT-TERM REVERSAL: At the 1-month horizon, returns REVERSE, not continue.
      A stock up 5% this month tends to be slightly DOWN next month. This is the
      opposite of momentum. It's caused by market microstructure effects like
      bid-ask bounce and short-term mean reversion by market makers.

    - If you include the most recent month in your momentum calculation, these
      short-term reversal effects PARTIALLY CANCEL OUT your 12-month momentum
      signal, making it weaker and noisier.

    - Professional quant funds all implement the skip-month by default.
      It's not optional — it's part of the canonical factor definition.

    CALCULATION:
    - Fetch ~14 months of daily price data
    - Return = price at (1 month ago) / price at (13 months ago) - 1
    - This gives the 12-month return that EXCLUDES the most recent month

    SIGNAL LOGIC:
    - momentum_return > 10%: BUY — strong positive trend, likely to continue
    - momentum_return < -10%: SELL — strong negative trend, likely to continue
    - Between -10% and +10%: No signal — insufficient directional evidence

    The 10% threshold filters out weak signals. Professional implementations
    often sort by momentum percentile and go long top 30%, short bottom 30%.

    Args:
        symbol: Ticker symbol (works for equities, crypto, commodities, forex)
        period_months: Lookback window in months (default 12 — the academic standard)
        skip_months: Recent months to skip to avoid short-term reversal (default 1)

    Returns:
        dict with momentum metrics, or None if data unavailable
    """
    try:
        ticker = yf.Ticker(symbol)

        # We need (period_months + skip_months + 1) months of data to be safe.
        # Add a buffer: fetch 14 months for a 12-month signal with 1-month skip.
        # Using 420 days (14 months * 30 days) as a conservative buffer.
        total_days = (period_months + skip_months + 1) * 31  # ~31 days/month buffer
        start_date = datetime.today() - timedelta(days=total_days)

        hist = ticker.history(start=start_date.strftime('%Y-%m-%d'), interval='1d')

        if hist.empty or len(hist) < 60:
            # Need at least 60 trading days to compute a meaningful signal
            print(f"[factors] Insufficient price history for {symbol}")
            return None

        closes = hist['Close'].dropna()

        # === SKIP-MONTH IMPLEMENTATION ===
        # "1 month ago" in trading days = approximately 21 trading days
        # "13 months ago" = approximately 13 * 21 = 273 trading days
        skip_days = skip_months * 21      # ~21 trading days per month
        period_days = period_months * 21  # ~252 trading days per year

        # Index from the end: skip the most recent 'skip_days' bars
        # Then look back 'period_days' more bars from that point
        if len(closes) < (skip_days + period_days):
            print(f"[factors] Not enough data for {symbol} momentum calculation")
            return None

        price_end = float(closes.iloc[-(skip_days + 1)])    # Price 1 month ago
        price_start = float(closes.iloc[-(skip_days + period_days + 1)])  # Price 13 months ago

        if price_start == 0:
            return None

        # Core momentum calculation: total return over the measurement window
        momentum_return = (price_end / price_start) - 1.0

        # === SCORE NORMALIZATION (-1 to +1) ===
        # We map momentum_return to a [-1, 1] scale using a simple sigmoid-like
        # compression. A return of +50% or more saturates to ~+1.0, -50% to ~-1.0.
        # This normalization makes scores comparable across asset classes
        # (a 50% crypto move and a 10% equity move shouldn't be treated the same).
        # Clamp to [-1, 1] range.
        momentum_score = max(-1.0, min(1.0, momentum_return / 0.5))

        # === SIGNAL GENERATION ===
        # The 10% threshold is derived from academic literature.
        # Jegadeesh & Titman found the signal is most reliable for the top and
        # bottom deciles of momentum — roughly corresponding to >10% and <-10%
        # over a 12-month period for large-cap equities.
        if momentum_return > 0.10:
            signal = 'BUY'
        elif momentum_return < -0.10:
            signal = 'SELL'
        else:
            signal = None

        return {
            'symbol': symbol,
            'momentum_return': round(momentum_return, 4),   # e.g., 0.1523 = 15.23%
            'momentum_score': round(momentum_score, 4),     # normalized -1 to 1
            'signal': signal,
            'period_months': period_months,
            'skip_months': skip_months,
        }

    except Exception as e:
        print(f"[factors] Error calculating momentum for {symbol}: {e}")
        return None


# =============================================================================
# FACTOR 2: VALUE
# =============================================================================

def calculate_value_score(symbol: str) -> dict | None:
    """
    Calculate the value factor for an equity using fundamental valuation metrics.

    WHY VALUE INVESTING WORKS (Behavioral Finance Explanation):
    ============================================================
    Value investing works because investors systematically OVERPAY for exciting
    growth stories and UNDERPAY for boring, profitable companies.

    This is the "glamour stock" premium documented by Lakonishok, Shleifer &
    Vishny (1994). Investors extrapolate recent high growth into the future —
    they expect the hot tech company to keep growing at 40% forever — and pay
    a massive premium for that growth. When growth inevitably slows (regression
    to the mean), glamour stocks crash.

    Meanwhile, "value stocks" (low P/E, low P/B) are boring companies —
    utilities, consumer staples, industrials — that grow slowly but generate
    steady cash flows. Investors ignore them, causing them to trade at a
    discount. As their cash flows compound over years, these discounted prices
    deliver superior long-run returns.

    THE METRICS EXPLAINED:
    - P/E (Price/Earnings): How much you pay for $1 of earnings.
      P/E of 15 means $15 per $1 of annual earnings (6.7% earnings yield).
      P/E of 35 means $35 per $1 of annual earnings (2.9% earnings yield).
      Lower P/E = cheaper relative to current profitability.

    - P/B (Price/Book): How much you pay for $1 of accounting net assets.
      P/B below 1.0 means the stock trades below its liquidation value —
      theoretically, you could buy the company and sell its assets for profit.
      P/B above 3-4 usually indicates either a franchise (AAPL ~40 P/B) or
      overvaluation.

    - Forward P/E: Uses analyst consensus EPS estimates for next year.
      Forward P/E < Trailing P/E implies expected earnings growth.
      Forward P/E > Trailing P/E implies expected earnings decline.

    IMPORTANT LIMITATION — ONLY WORKS FOR EQUITIES:
    This function returns None for crypto, forex, and commodities because
    those assets have no earnings, no book value, and no P/E ratio.
    Attempting to apply value metrics to Bitcoin is meaningless.

    Args:
        symbol: Equity ticker symbol (AAPL, MSFT, etc.)

    Returns:
        dict with value metrics, or None if not an equity or data unavailable
    """
    # Pre-filter: value factor only applies to equities
    if not _is_equity(symbol):
        return None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info  # Returns a large dict of fundamental data

        # yfinance uses 'trailingPE', 'forwardPE', 'priceToBook' as keys
        trailing_pe = info.get('trailingPE')
        forward_pe = info.get('forwardPE')
        pb_ratio = info.get('priceToBook')

        # If we can't get P/E at all, value scoring is impossible
        if trailing_pe is None and forward_pe is None:
            print(f"[factors] No P/E data available for {symbol} (may not be an equity with reported earnings)")
            return None

        # Use trailing P/E as primary metric; fall back to forward P/E if not available
        primary_pe = trailing_pe if trailing_pe is not None else forward_pe

        # === VALUE SCORE CALCULATION ===
        # We compute a simple score that maps P/E to a 0-100 scale (or negative for overvalued).
        #
        # Academic research (Fama & French) shows stocks with P/E < 15 tend to outperform
        # those with P/E > 35. These thresholds are approximate — the key insight is
        # RELATIVE cheapness (bottom quintile of P/E in the market) beats expensive stocks.
        #
        # For simplicity, we define:
        #   P/E <= 10: Very cheap     → score ~100 (extremely undervalued territory)
        #   P/E == 15: Cheap          → score ~75  (historically good value entry)
        #   P/E == 25: Fair value     → score ~50  (roughly market neutral)
        #   P/E == 35: Expensive      → score ~25  (overvalued territory)
        #   P/E >= 50: Very expensive → score ~0   (bubble/growth-at-any-price territory)
        #
        # This linear interpolation is a simplification — real factor models use
        # percentile ranks across the entire investable universe.

        if primary_pe <= 0:
            # Negative P/E means the company is losing money — not a value stock
            value_score = 0.0
        else:
            # Map P/E to a 0-1 score inversely (lower P/E = higher score)
            # Saturate at P/E = 5 (score = 1.0) and P/E = 60 (score = 0.0)
            value_score = max(0.0, min(1.0, (60.0 - primary_pe) / (60.0 - 5.0)))

        # === SIGNAL GENERATION ===
        if primary_pe < 15:
            signal = 'BUY'    # Historically cheap — value premium likely
        elif primary_pe > 35:
            signal = 'SELL'   # Historically expensive — mean reversion risk
        else:
            signal = None     # Fair value zone — no strong directional edge

        return {
            'symbol': symbol,
            'pe_ratio': round(trailing_pe, 2) if trailing_pe else None,
            'forward_pe': round(forward_pe, 2) if forward_pe else None,
            'pb_ratio': round(pb_ratio, 2) if pb_ratio else None,
            'value_score': round(value_score, 4),   # 0-1 scale (higher = cheaper)
            'signal': signal,
        }

    except Exception as e:
        print(f"[factors] Error calculating value score for {symbol}: {e}")
        return None


# =============================================================================
# FACTOR 3: QUALITY
# =============================================================================

def calculate_quality_score(symbol: str) -> dict | None:
    """
    Calculate the quality factor for an equity using fundamental metrics.

    WHY THE QUALITY FACTOR WORKS:
    ==============================
    The quality anomaly is one of the most puzzling in finance: companies with
    strong financials (high profitability, low leverage, stable earnings) earn
    HIGHER returns than weak companies — even after controlling for other factors.

    Standard theory (CAPM) predicts the OPPOSITE: more risk = more return.
    A highly leveraged company should compensate investors with higher expected
    returns for the higher bankruptcy risk. But empirically, it does the opposite.

    THE EXPLANATION from behavioral finance:
    Investors confuse ACCOUNTING COMPLEXITY with GENUINE UNCERTAINTY. Companies
    that manage earnings aggressively (Sloan's 1996 accruals anomaly), carry
    heavy debt, and have thin margins are RISKY in ways that aren't fully priced
    into current stock prices. When the balance sheet eventually cracks, the
    drawdown is severe and fast.

    Quality stocks, by contrast, are boring. Low debt means no refinancing
    crises. High margins mean pricing power. High ROE means management deploys
    capital efficiently. These companies compound quietly over decades.
    AQR calls this "Quality Minus Junk" (QMJ) — their most cited factor paper.

    THE FOUR QUALITY METRICS AND WHY EACH WAS CHOSEN:

    1. ROE (Return on Equity) > 15%:
       ROE measures how efficiently management converts shareholders' equity into
       profit. 15% is roughly the long-run S&P 500 average ROE. Companies above
       this are generating above-average returns on capital — a sign of durable
       competitive advantage (Warren Buffett's favorite metric for exactly this reason).

    2. Debt-to-Equity < 0.5:
       D/E below 0.5 means the company has twice as much equity as debt.
       It can service its debt comfortably in a downturn. High D/E (>2.0) means
       even a modest revenue decline can trigger a debt spiral. The 2008-2009
       crisis destroyed heavily leveraged companies while quality ones survived.

    3. Gross Margin > 30%:
       Gross margin = (Revenue - Cost of Goods Sold) / Revenue.
       A 30%+ gross margin indicates pricing power — the company can charge
       enough above its direct costs to fund R&D, marketing, and still profit.
       Commodity businesses (airlines, steel) often have <10% gross margins
       and consistently destroy shareholder value.

    4. Positive Earnings Growth:
       A company growing its earnings is allocating capital productively.
       Negative earnings growth (shrinkage) can indicate secular decline,
       competitive pressure, or management problems. Even flat is a yellow flag.

    Args:
        symbol: Equity ticker symbol

    Returns:
        dict with quality metrics (0-100 score), or None if not equity / no data
    """
    # Quality factor only applies to equities with fundamental data
    if not _is_equity(symbol):
        return None

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        roe = info.get('returnOnEquity')         # e.g., 0.45 = 45% ROE
        debt_to_equity = info.get('debtToEquity')  # e.g., 0.3 = 30% D/E ratio
        gross_margin = info.get('grossMargins')   # e.g., 0.43 = 43% gross margin
        earnings_growth = info.get('earningsGrowth')  # e.g., 0.12 = 12% growth

        # If we have no fundamental data at all, return None gracefully
        if all(v is None for v in [roe, debt_to_equity, gross_margin, earnings_growth]):
            print(f"[factors] No fundamental data available for {symbol}")
            return None

        # === SCORING RUBRIC (0 to 100 points) ===
        # Each of the four criteria contributes up to 25 points.
        # This equal weighting is a simplification — in practice, ROE and
        # gross margin tend to have the highest predictive power (Novy-Marx 2013).
        #
        # The thresholds are derived from academic literature and
        # professional screen criteria (Value Line, Morningstar, AQR research).

        quality_score = 0.0

        # Criterion 1: ROE > 15% (Warren Buffett's minimum for "excellent" businesses)
        if roe is not None and roe > 0.15:
            quality_score += 25.0

        # Criterion 2: Debt/Equity < 0.5 (conservative balance sheet)
        # Note: yfinance returns D/E as a number like 0.3 or 180 depending on version.
        # Some versions return it as a percentage (e.g., 30 for 30%). We normalize.
        if debt_to_equity is not None:
            # Normalize if returned as large number (yfinance quirk: sometimes 100x scale)
            de_normalized = debt_to_equity / 100.0 if debt_to_equity > 10 else debt_to_equity
            if de_normalized < 0.5:
                quality_score += 25.0

        # Criterion 3: Gross Margin > 30% (pricing power indicator)
        if gross_margin is not None and gross_margin > 0.30:
            quality_score += 25.0

        # Criterion 4: Positive Earnings Growth (compounding trajectory)
        if earnings_growth is not None and earnings_growth > 0:
            quality_score += 25.0

        # === SIGNAL GENERATION ===
        # Only generate a BUY signal for high-quality companies (75+ score).
        # We don't generate SELL signals from quality alone — a low-quality
        # company can still be a buy if it's cheap enough (the value factor handles that).
        # This asymmetry reflects how quality is used in practice:
        # as a FILTER (reject low-quality) more than a directional signal.
        signal = 'BUY' if quality_score > 75 else None

        return {
            'symbol': symbol,
            'roe': round(roe, 4) if roe is not None else None,
            'debt_to_equity': round(debt_to_equity, 4) if debt_to_equity is not None else None,
            'gross_margin': round(gross_margin, 4) if gross_margin is not None else None,
            'earnings_growth': round(earnings_growth, 4) if earnings_growth is not None else None,
            'quality_score': quality_score,   # 0-100 scale
            'signal': signal,
        }

    except Exception as e:
        print(f"[factors] Error calculating quality score for {symbol}: {e}")
        return None


# =============================================================================
# FACTOR 4: LOW VOLATILITY
# =============================================================================

def calculate_low_vol_score(symbol: str, period: str = '90d') -> dict | None:
    """
    Calculate the low-volatility factor — one of the most persistent and
    theoretically puzzling anomalies in all of finance.

    THE LOW-VOL ANOMALY EXPLAINED:
    ================================
    Standard CAPM theory (Sharpe, 1964) predicts: higher risk (beta/volatility)
    should be compensated with higher expected return. This is the foundational
    principle of modern portfolio theory.

    Fischer Black (1972) noticed the opposite empirically: low-beta stocks
    earned HIGHER risk-adjusted returns than high-beta stocks. Sixty years of
    data and dozens of replications across every major global market confirm this.

    WHY DOES THIS PARADOX PERSIST?
    The most accepted explanation (Baker, Bradley & Wurgler 2011):

    1. INSTITUTIONAL BENCHMARKING CONSTRAINT: Most fund managers are benchmarked
       against the S&P 500. If they want to beat the index, they can't just buy
       boring low-vol stocks and wait — they'll underperform in a bull market.
       So they systematically overweight HIGH-BETA stocks to get "more market
       exposure." This excess demand overprices high-beta stocks and leaves
       low-beta stocks undervalued relative to their true risk/return.

    2. LEVERAGE AVERSION: Individual investors who want more return typically
       can't (or won't) use leverage. Instead they buy high-volatility stocks
       as a proxy for "more upside." This lottery-ticket behavior overprices
       volatile stocks.

    3. CAREER RISK: No fund manager ever got fired for holding high-beta FAANG
       stocks that crashed. But holding boring utilities while FAANG rips is
       career-ending even if it was the right risk-adjusted call.

    WHAT WE MEASURE — REALIZED VOLATILITY:
    Realized volatility = annualized standard deviation of daily log returns.
    It's the most straightforward measure of historical price variability.

    FORMULA:
    - Daily log returns: r_t = ln(P_t / P_{t-1})
    - Daily std dev: σ_daily = std(r)
    - Annualized: σ_annual = σ_daily * sqrt(252)   [252 trading days/year]

    BENCHMARKS:
    - Low vol equities: σ_annual ~10-15%  (utilities, consumer staples)
    - Average S&P 500 stock: σ_annual ~20-25%
    - High vol growth stocks: σ_annual ~30-50%  (TSLA, meme stocks)
    - Crypto: σ_annual ~60-100%+  (BTC ~60-80%, altcoins 100-200%)

    Works for ALL asset classes — unlike value/quality which need fundamentals.

    Args:
        symbol: Any ticker symbol (equity, crypto, forex, commodity)
        period: Lookback window for volatility calculation (default '90d')

    Returns:
        dict with realized volatility and signal, or None if data unavailable
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval='1d')

        if hist.empty or len(hist) < 20:
            # Need at least 20 data points for a meaningful volatility estimate
            print(f"[factors] Insufficient data for volatility calculation: {symbol}")
            return None

        closes = hist['Close'].dropna()

        # === REALIZED VOLATILITY CALCULATION ===
        # We use LOG returns (not simple returns) because:
        # 1. They are time-additive: 10-day log return = sum of 10 daily log returns
        # 2. They are symmetric: a +10% and -10% move have the same absolute log return
        # 3. They are approximately normally distributed for short horizons
        # Simple returns: R_t = (P_t - P_{t-1}) / P_{t-1}
        # Log returns:    r_t = ln(P_t / P_{t-1}) = ln(1 + R_t)
        log_returns = np.log(closes / closes.shift(1)).dropna()

        if len(log_returns) < 10:
            return None

        # Annualize by multiplying daily std by sqrt(252)
        # This conversion assumes returns are i.i.d. (independent, identically distributed)
        # which is approximately true for daily returns
        daily_vol = float(log_returns.std())
        realized_vol = daily_vol * np.sqrt(252)  # Annualized realized volatility

        # === VOL PERCENTILE (pseudo-calculation) ===
        # In a real system, you'd calculate the percentile rank of this symbol's
        # vol relative to the full investable universe (e.g., Russell 1000).
        # Here we approximate using asset-class-specific benchmarks:
        #
        # The logic: if a stock's vol is lower than typical, it gets a high
        # percentile rank (meaning it's in the "low vol" bucket). We want
        # vol_percentile to be HIGH when volatility is LOW (inverse relationship).
        #
        # Benchmark annualized vols (approximate):
        #   - Equities: ~25% typical
        #   - Crypto: ~80% typical
        #   - Forex: ~8% typical
        #   - Commodities: ~25% typical
        if symbol.endswith('-USD') or symbol.endswith('-USDT'):
            benchmark_vol = 0.80   # Crypto is inherently high-vol
        elif symbol.endswith('=X'):
            benchmark_vol = 0.08   # Major forex pairs are low-vol
        else:
            benchmark_vol = 0.25   # Equities / commodities baseline

        # vol_percentile: 1.0 = lowest vol (best for low-vol factor), 0.0 = highest vol
        # Clamp to [0, 1]
        vol_percentile = max(0.0, min(1.0, 1.0 - (realized_vol / (benchmark_vol * 2.0))))

        # === SIGNAL GENERATION ===
        # Signal only fires for genuinely low-volatility assets.
        # Threshold: vol below 60% of the asset-class benchmark.
        # This selects the bottom two quintiles of vol — roughly matching
        # how professional low-vol strategies construct their long books.
        if realized_vol < benchmark_vol * 0.60:
            signal = 'BUY'   # Low vol relative to asset class — factor edge exists
        else:
            signal = None    # Above-average or typical vol — no low-vol premium

        return {
            'symbol': symbol,
            'realized_vol': round(realized_vol, 4),      # e.g., 0.18 = 18% annualized
            'vol_percentile': round(vol_percentile, 4),  # 0-1 (higher = lower vol = better)
            'signal': signal,
            'period': period,
        }

    except Exception as e:
        print(f"[factors] Error calculating low-vol score for {symbol}: {e}")
        return None


# =============================================================================
# AGGREGATOR: Run All Four Factors for a List of Symbols
# =============================================================================

def get_factor_signals(symbols: list[str]) -> dict:
    """
    Run all four factor models across all symbols and produce a composite signal.

    WHY FACTOR CONFLUENCE MATTERS:
    ================================
    No single factor is reliable all the time. Value had a brutal decade
    (2010-2020) of underperformance. Momentum crashes occasionally (March 2020:
    -34% in weeks). But when MULTIPLE factors agree on the same direction for
    the same asset, the probability of that view being correct increases
    substantially.

    This is called "multi-factor investing" and it's the backbone of most
    systematic strategies at firms like AQR and Dimensional Fund Advisors.
    Their research shows that combining uncorrelated factors (momentum + value
    have near-zero correlation) produces better risk-adjusted returns than
    any single factor in isolation.

    COMPOSITE SIGNAL LOGIC:
    - 3+ factors agree BUY  → STRONG_BUY  (high conviction)
    - 2 factors agree BUY   → BUY         (moderate conviction)
    - 2 factors agree SELL  → SELL        (moderate conviction)
    - 3+ factors agree SELL → STRONG_SELL (high conviction)
    - No clear agreement    → None        (no edge, stand aside)

    Args:
        symbols: List of ticker symbols to analyze

    Returns:
        Dict keyed by symbol, each containing all four factor results
        plus a composite signal and plain-English explanation
    """
    results = {}

    for symbol in symbols:
        print(f"[factors] Analyzing {symbol}...")

        # Run all four factors (each handles its own error gracefully)
        momentum = calculate_momentum_score(symbol)
        value = calculate_value_score(symbol)       # Returns None for non-equities
        quality = calculate_quality_score(symbol)   # Returns None for non-equities
        low_vol = calculate_low_vol_score(symbol)

        # === COMPOSITE SIGNAL AGGREGATION ===
        # Collect all non-None signals into a list for vote counting
        buy_votes = 0
        sell_votes = 0
        active_factors = []   # Track which factors had opinions

        for factor_name, factor_result in [
            ('momentum', momentum),
            ('value', value),
            ('quality', quality),
            ('low_vol', low_vol)
        ]:
            if factor_result is not None and factor_result.get('signal') is not None:
                signal_val = factor_result['signal']
                active_factors.append(f"{factor_name}={signal_val}")
                if signal_val == 'BUY':
                    buy_votes += 1
                elif signal_val == 'SELL':
                    sell_votes += 1

        # === COMPOSITE SIGNAL DECISION TREE ===
        # We require at least 2 factors to agree before generating any signal.
        # A single-factor signal is insufficient evidence — too many false positives.
        total_opinion_votes = buy_votes + sell_votes

        if buy_votes >= 3:
            composite_signal = 'STRONG_BUY'
            composite_reason = (
                f"{buy_votes} factors agree bullish ({', '.join(active_factors)}). "
                f"Multi-factor confluence is a high-conviction signal used by quant funds."
            )
        elif buy_votes == 2:
            composite_signal = 'BUY'
            composite_reason = (
                f"2 factors agree bullish ({', '.join(active_factors)}). "
                f"Moderate-conviction multi-factor signal."
            )
        elif sell_votes >= 3:
            composite_signal = 'STRONG_SELL'
            composite_reason = (
                f"{sell_votes} factors agree bearish ({', '.join(active_factors)}). "
                f"Multi-factor downside confluence — high conviction negative view."
            )
        elif sell_votes == 2:
            composite_signal = 'SELL'
            composite_reason = (
                f"2 factors agree bearish ({', '.join(active_factors)}). "
                f"Moderate-conviction multi-factor sell signal."
            )
        else:
            composite_signal = None
            composite_reason = (
                f"No clear factor consensus. Active factors: "
                f"{', '.join(active_factors) if active_factors else 'none had signals'}. "
                f"Standing aside — no edge identified."
            )

        results[symbol] = {
            'momentum': momentum,
            'value': value,
            'quality': quality,
            'low_vol': low_vol,
            'composite_signal': composite_signal,
            'composite_reason': composite_reason,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
        }

    return results


# =============================================================================
# PORTFOLIO CONSTRUCTION: Rank Symbols by a Single Factor
# =============================================================================

def rank_symbols_by_factor(symbols: list[str], factor: str = 'momentum') -> list[dict]:
    """
    Rank all symbols by a given factor score, from best to worst.

    WHY RANKING IS THE CORRECT WAY TO USE FACTORS:
    ================================================
    Factor investing is NOT about finding individual stocks that look attractive.
    It is about constructing PORTFOLIOS that systematically tilt toward a factor.

    The professional implementation is:
    1. Rank your entire investable universe by the factor score
    2. LONG the top 30% (best factor exposure)
    3. SHORT the bottom 30% (worst factor exposure)
    4. The middle 40% is ignored — no edge there

    This long/short construction is market-neutral: you profit from the
    SPREAD between top and bottom performers, regardless of whether the
    market goes up or down overall. This is the source of "pure alpha"
    uncorrelated to market beta.

    AQR's factor funds are constructed exactly this way. Their research
    shows the long/short construction has higher Sharpe ratio than simply
    buying the top-ranked stocks (long-only), because the short book
    hedges market exposure.

    SUPPORTED FACTORS:
    - 'momentum': Sort by momentum_return (higher = better recent trend)
    - 'value': Sort by value_score (higher = cheaper = better value)
    - 'quality': Sort by quality_score (higher = stronger fundamentals)
    - 'low_vol': Sort by vol_percentile (higher = lower vol = better)

    Args:
        symbols: List of ticker symbols to rank
        factor: Which factor to rank by ('momentum', 'value', 'quality', 'low_vol')

    Returns:
        Sorted list of {'symbol': str, 'score': float, 'rank': int},
        ordered from best (rank 1) to worst (rank N).
        Symbols with no data for the requested factor are excluded.
    """
    # Map factor name to the function and score key
    factor_functions = {
        'momentum': (calculate_momentum_score, 'momentum_score'),
        'value': (calculate_value_score, 'value_score'),
        'quality': (calculate_quality_score, 'quality_score'),
        'low_vol': (calculate_low_vol_score, 'vol_percentile'),
    }

    if factor not in factor_functions:
        print(f"[factors] Unknown factor '{factor}'. Valid options: {list(factor_functions.keys())}")
        return []

    calc_fn, score_key = factor_functions[factor]

    # Calculate factor score for each symbol
    scored = []
    for symbol in symbols:
        result = calc_fn(symbol)
        if result is not None and score_key in result and result[score_key] is not None:
            scored.append({
                'symbol': symbol,
                'score': float(result[score_key]),
            })
        else:
            # Gracefully skip symbols with no data for this factor
            print(f"[factors] No {factor} data for {symbol}, excluding from ranking")

    # Sort descending: highest score = rank 1 (best)
    scored.sort(key=lambda x: x['score'], reverse=True)

    # Assign rank (1-indexed: rank 1 = top performer)
    for i, item in enumerate(scored):
        item['rank'] = i + 1

    # Annotate with long/short designation
    # Top 30% = "LONG" candidates, Bottom 30% = "SHORT" candidates
    n = len(scored)
    top_cutoff = max(1, int(n * 0.30))
    bottom_cutoff = max(1, int(n * 0.30))

    for i, item in enumerate(scored):
        if i < top_cutoff:
            item['designation'] = 'LONG'     # Top 30% — long book
        elif i >= n - bottom_cutoff:
            item['designation'] = 'SHORT'    # Bottom 30% — short book
        else:
            item['designation'] = 'NEUTRAL'  # Middle 40% — no position

    return scored
