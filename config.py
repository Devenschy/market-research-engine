# =============================================================================
# config.py — Single Source of Truth for All Settings
# =============================================================================
# WHY: Centralizing all configuration prevents "magic numbers" scattered across
# the codebase. In a real trading firm, config changes go through a review
# process. Having them all in one place makes that audit easy.

# --- Asset Universe ---
# WHY: Cross-asset coverage is intentional. Equities (AAPL, MSFT), crypto
# (BTC-USD, ETH-USD), commodities (GC=F gold, CL=F crude oil), and forex
# (EURUSD=X) all respond differently to macro regimes. Watching them together
# teaches you how capital flows across asset classes.
SYMBOLS = ['AAPL', 'MSFT', 'BTC-USD', 'ETH-USD', 'GC=F', 'CL=F', 'EURUSD=X']

# --- Paper Trading Mode ---
# WHY: Never connect real capital until the system has proven itself in simulation.
# Paper trading lets you validate signal quality and risk controls without risk.
PAPER_TRADING = True

# --- Capital & Position Sizing ---
STARTING_CAPITAL = 10000.0

# WHY: 5% fixed fractional sizing means the maximum you can lose on any single
# trade is 5% * 2% stop = 0.1% of capital. This keeps individual losses trivial
# and prevents any single trade from being catastrophic.
POSITION_SIZE_PCT = 0.05   # 5% of current capital per trade

# --- Risk Controls ---
# WHY: The 2/4 ratio creates a 2:1 reward-to-risk. With a 40% win rate you
# still make money: (0.4 * 4%) - (0.6 * 2%) = +0.4%. Most retail traders
# do the opposite — they cut winners early and let losers run.
STOP_LOSS_PCT = 0.02       # 2% max loss per trade
TAKE_PROFIT_PCT = 0.04     # 4% target gain per trade

# WHY: The daily kill switch is used by every serious prop trading desk.
# A bad morning can become a catastrophic day if you keep trading through it.
# The discipline to stop is a psychological skill, not just a technical one.
MAX_DAILY_DRAWDOWN_PCT = 0.05   # 5% daily drawdown halts all trading

MAX_OPEN_POSITIONS = 5

# --- Polling Interval ---
# WHY: 60 seconds is sufficient for a learning/research system. We are not
# competing on latency — we are learning market structure.
POLL_INTERVAL_SECONDS = 30

# --- Strategy Parameters ---
# WHY: Fast MA (10) vs Slow MA (30) gives a responsive but not noisy crossover.
# These are standard parameters used in academic and professional backtests.
MA_FAST = 10
MA_SLOW = 30

# WHY: RSI-14 is the most widely used momentum oscillator parameter. Welles
# Wilder, who invented RSI, recommended 14 periods. It's standard on Bloomberg.
RSI_PERIOD = 14

# WHY: 20 periods for mean reversion matches one trading month of daily data.
# The 2 standard deviation threshold captures ~95% of price distribution,
# meaning a signal only fires during statistically unusual price dislocations.
MEAN_REV_WINDOW = 20
MEAN_REV_ZSCORE_THRESHOLD = 1.5

# --- Regime Detection Thresholds ---
# WHY: ADX > 25 is the widely accepted threshold for a "trending" market.
# Below 25, the market is directionless and trend-following strategies
# underperform. This cutoff was established in Wilder's original ADX work.
ADX_TREND_THRESHOLD = 20

# WHY: ATR spike (50% above 20-period average) signals unusual volatility.
# In volatile regimes, all strategies become less reliable and position
# sizes should be reduced to protect capital.
ATR_SPIKE_MULTIPLIER = 1.5

# --- Data History for Warm-up ---
# Strategies need historical data to initialize their indicators before trading
WARMUP_PERIOD = '60d'
WARMUP_INTERVAL = '1h'

# --- Logging Paths ---
TRADES_LOG = 'logs/trades.csv'
SIGNALS_LOG = 'logs/signals.csv'
PERFORMANCE_LOG = 'logs/performance.json'

# --- API Keys (optional — system degrades gracefully without them) ---
# Register free at fred.stlouisfed.org — takes 2 minutes.
# FRED has 800,000+ economic series: rates, inflation, employment, money supply.
import os
from dotenv import load_dotenv
load_dotenv()
FRED_API_KEY = os.environ.get('FRED_API_KEY', '')   # Stored in .env file — never committed to git

# --- Derivatives Settings ---
# WHY: Centralizing derivatives config here keeps all tunable parameters in one
# auditable location. The comments explain the reasoning behind each value so
# you understand what you're changing before you change it.

# OPTIONS CONFIG
# WHY AAPL AND MSFT: These are the two most liquid equity options markets in the
# world. AAPL and MSFT average millions of options contracts per day, giving tight
# bid-ask spreads and reliable price discovery. SPY, QQQ are also liquid but are
# ETFs with different tax treatment. Crypto options (on Deribit) are more complex.
OPTIONS_SYMBOLS = ['AAPL', 'MSFT']   # Equities with liquid options

# PERP / CRYPTO FUTURES CONFIG
PERP_SYMBOLS = ['BTC-USD', 'ETH-USD']  # Crypto perpetuals (paper trading only)

# WHY 2x MAX LEVERAGE:
# 2x leverage means you need a 50% adverse move to get liquidated.
# This is enough to feel the mechanics (funding rates, liq price, leveraged P&L)
# without the extreme liquidation risk of 5x, 10x, or 100x leverage.
# Professional prop shops often limit learning accounts to 2-3x for the same reason.
PERP_LEVERAGE = 2                      # 2x max for learning

# WHY 0.5% MAINTENANCE MARGIN:
# This matches Binance's standard maintenance margin for BTC/ETH perpetuals.
# It's the minimum equity buffer required before liquidation fires.
# Below 0.5% of notional remaining, the exchange forcibly closes the position.
PERP_MAINTENANCE_MARGIN = 0.005        # 0.5% maintenance margin

# OPTIONS EXPIRY WINDOW
# WHY 45 DTE MAX:
# Options with > 45 days to expiry have lower theta (time decay is slow).
# The "theta decay cliff" accelerates significantly inside 45 days.
# Most professional options sellers target 30-45 DTE for the theta sweet spot.
OPTIONS_MAX_DTE = 45                   # Max days to expiry for new positions

# WHY 7 DTE MIN:
# Options with < 7 days to expiry are "gamma bombs." Their delta can swing
# wildly with small price moves. Theta decay per day is extreme. The risk/reward
# profile for BUYING options this close to expiry is generally unfavorable.
# Professional rule of thumb: close or avoid < 21 DTE; never buy < 7 DTE.
OPTIONS_MIN_DTE = 7                    # Don't buy options with < 7 days (theta decay accelerates)

# IV RANK THRESHOLDS (0-100 scale)
# WHY THESE THRESHOLDS:
# IV Rank 70+ = options are in the top 30% of their yearly IV range.
# Premiums are expensive. Selling premium has better expected value than buying.
# IV Rank 30- = options are in the bottom 30% of their yearly IV range.
# Premiums are cheap. Buying options has better expected value than selling.
# The 30/70 split is a widely-used heuristic in professional options trading
# (Tastytrade, CBOE educational materials, etc.).
IV_RANK_HIGH = 70                      # Above this: IV is rich, favor selling premium
IV_RANK_LOW = 30                       # Below this: IV is cheap, favor buying options

# --- Pairs Trading Settings ---
# WHY THESE PAIRS:
# AAPL/MSFT: Two tech giants with correlated revenue drivers (enterprise, cloud, consumer).
# BTC/ETH: Crypto pair driven by the same macro sentiment and liquidity cycles.
# GC=F/CL=F: Gold and crude oil — both inflation hedges priced in USD, macro-correlated.
PAIRS = [
    ('AAPL', 'MSFT'),
    ('BTC-USD', 'ETH-USD'),
    ('GC=F', 'CL=F'),
]

# WHY 2.0 ENTRY THRESHOLD?
# ±2.0 standard deviations corresponds to the outer ~5% of a normal distribution.
# Only trading at this extreme ensures we're acting on genuine statistical dislocations,
# not routine daily noise. This is the industry-standard threshold for pairs entry.
PAIRS_ZSCORE_ENTRY = 2.0    # Open position when spread this many std devs from mean

# WHY 0.5 EXIT THRESHOLD?
# Once the spread is within 0.5 std devs of its mean, it has largely converged.
# We don't wait for perfect zero-crossing — that's too greedy and risks giving back
# gains as the spread overshoots. 0.5 captures ~80% of the expected convergence.
PAIRS_ZSCORE_EXIT = 0.5     # Close position when spread converges to this level

# WHY 90 DAYS?
# 90 days (≈3 months) gives enough data for the Engle-Granger cointegration test
# to have statistical power, while keeping the relationship estimate "recent" enough
# to reflect the current market structure (not stale 2-year-old relationships).
PAIRS_LOOKBACK = '90d'      # History window for cointegration testing

# --- Sentiment Settings ---
# WHY 0.3 BULLISH THRESHOLD (not 0.05)?
# VADER's compound score of 0.05 just means "slightly positive."
# For an AVERAGE across many headlines to reach 0.3, you need a sustained cluster
# of strongly positive articles — that's a real signal, not one good headline.
# Academic research (Tetlock 2007) shows strong sentiment clusters have the most
# predictive power for 1-3 day forward returns.
SENTIMENT_BULLISH_THRESHOLD = 0.3   # avg compound score above this = BUY signal

# WHY -0.3 BEARISH THRESHOLD?
# Symmetric to the bullish threshold. Studies show negative sentiment has
# STRONGER and MORE CONSISTENT predictive power than positive sentiment.
# (Bad news travels faster and has more asymmetric impact on prices.)
# The -0.3 threshold filters to genuinely alarming news clusters, not routine
# analyst concerns that appear in every stock's news feed daily.
SENTIMENT_BEARISH_THRESHOLD = -0.3  # avg compound score below this = SELL signal
