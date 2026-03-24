# =============================================================================
# options.py — Options Pricing, Greeks, Signals, and Paper P&L Tracking
# =============================================================================
# WHY OPTIONS MATTER:
# Options are not just "insurance" — they are precision instruments. A stock
# trader can only bet that a price goes up or down. An options trader can bet
# on direction, magnitude, timing, AND volatility itself. This is why
# professional desks use options for nearly every sophisticated trade.
#
# This module covers:
#   1. Data fetching (live options chains via yfinance)
#   2. Black-Scholes pricing from scratch (the mathematical engine)
#   3. Greeks calculation (how the option responds to market changes)
#   4. Signal generation (given a directional view, pick the right option)
#   5. Paper P&L tracking (simulate owning an option position)
#
# SCOPE: Only AAPL and MSFT — equities with the most liquid options markets.
# Crypto options exist but are more complex; commodities options (GC=F, CL=F)
# are futures options with different mechanics. We stay in liquid equity land.

import math
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional


# Plain-English explanation of every Greek — shown on the dashboard
# so you always see the definition next to the live number.
GREEKS_EXPLAINED = {
    'delta': (
        "How much the option price moves per $1 move in the stock. "
        "Delta 0.50 = option gains $0.50 if stock gains $1. "
        "Calls: 0 to +1. Puts: -1 to 0. ATM options are ~0.50."
    ),
    'gamma': (
        "How fast DELTA changes as the stock moves. "
        "High gamma = your delta exposure accelerates — options move faster in your favor (or against). "
        "Highest for ATM options near expiry. Gamma risk is why selling short-dated options is dangerous."
    ),
    'theta': (
        "Time decay — dollar value lost per day just from time passing. "
        "Always negative for buyers (you pay for time). Always positive for sellers (you collect time). "
        "Accelerates sharply inside the last 30 days before expiry."
    ),
    'vega': (
        "Sensitivity to implied volatility (IV). "
        "Vega 0.10 = option gains $0.10 per 1% rise in IV. "
        "Long options gain when IV spikes (fear/earnings). Short options lose. "
        "This is why buying options into earnings can be a trap — IV is already elevated."
    ),
    'rho': (
        "Sensitivity to interest rates. "
        "Minor for short-dated options but meaningful for LEAPS (1-2 year options). "
        "Rising rates slightly help calls, slightly hurt puts."
    ),
}


# =============================================================================
# CONSTANTS
# =============================================================================

# Risk-free rate — approximation using current short-term T-bill yield.
# WHY: Black-Scholes requires a risk-free rate to discount the option's
# expected payoff back to present value. We use a reasonable approximation.
# In a production system this would be fetched from FRED (DGS3MO series).
RISK_FREE_RATE = 0.05   # 5% — approximately the current Fed Funds rate

# Symbols for which we support options (liquid markets only)
# WHY: Options on illiquid underlyings have wide bid-ask spreads and poor
# price discovery. AAPL and MSFT have among the highest options volume globally.
OPTIONS_SYMBOLS = ['AAPL', 'MSFT']


# =============================================================================
# SECTION 1: DATA FETCHING
# =============================================================================

def fetch_options_chain(symbol: str) -> Optional[dict]:
    """
    Fetch the full options chain (calls and puts) for the nearest expiry date.

    WHY NEAREST EXPIRY:
    We focus on the front-month (nearest expiry) because it has the highest
    options volume and tightest bid-ask spreads. This is where most retail and
    institutional activity concentrates. Longer-dated options (LEAPS) have
    different characteristics and lower liquidity.

    HOW yfinance OPTIONS WORK:
    yfinance returns options data as a dict with 'calls' and 'puts' DataFrames.
    Each row is one strike price with columns: strike, lastPrice, bid, ask,
    impliedVolatility, volume, openInterest, etc.

    IMPORTANT: yfinance options data can be unreliable — the API sometimes
    returns stale data or times out. We wrap everything in try/except.

    Returns a dict with:
        - 'symbol': the ticker
        - 'expiry': the expiry date string
        - 'calls': DataFrame of call options
        - 'puts': DataFrame of put options
        - 'current_price': current stock price
    Or None if fetching fails.
    """
    # Guard: only process symbols with liquid options
    if symbol not in OPTIONS_SYMBOLS:
        print(f"[options] {symbol} not in supported options symbols: {OPTIONS_SYMBOLS}")
        return None

    try:
        ticker = yf.Ticker(symbol)

        # Fetch current stock price
        hist = ticker.history(period='1d', interval='1m')
        if hist.empty:
            print(f"[options] Could not fetch price for {symbol}")
            return None
        current_price = float(hist['Close'].iloc[-1])

        # Get available expiry dates
        # WHY: yfinance lists expiry dates as strings (e.g., '2024-01-19')
        expirations = ticker.options
        if not expirations:
            print(f"[options] No options expirations available for {symbol}")
            return None

        # Use the nearest expiry (first in the list — yfinance returns sorted)
        nearest_expiry = expirations[0]

        # Fetch the options chain for that expiry
        chain = ticker.option_chain(nearest_expiry)
        calls_df = chain.calls.copy()
        puts_df = chain.puts.copy()

        # Clean up: fill NaN implied volatility with 0
        # WHY: Deep ITM/OTM options sometimes have no reported IV from the exchange
        calls_df['impliedVolatility'] = calls_df['impliedVolatility'].fillna(0)
        puts_df['impliedVolatility'] = puts_df['impliedVolatility'].fillna(0)

        # Add useful derived columns for display
        # Moneyness: how far the strike is from current price as a percentage
        calls_df['moneyness_pct'] = ((calls_df['strike'] - current_price) / current_price * 100).round(2)
        puts_df['moneyness_pct'] = ((puts_df['strike'] - current_price) / current_price * 100).round(2)

        # Calculate days to expiry for reference
        expiry_date = datetime.strptime(nearest_expiry, '%Y-%m-%d').date()
        today = date.today()
        dte = (expiry_date - today).days

        return {
            'symbol': symbol,
            'expiry': nearest_expiry,
            'days_to_expiry': dte,
            'current_price': round(current_price, 4),
            'calls': calls_df,
            'puts': puts_df
        }

    except Exception as e:
        print(f"[options] Error fetching options chain for {symbol}: {e}")
        return None


def fetch_iv_rank(symbol: str) -> Optional[float]:
    """
    Calculate IV Rank: where is today's implied volatility relative to its
    52-week high and low?

    WHY IV RANK MATTERS:
    IV Rank (IVR) tells you whether options are currently cheap or expensive
    relative to their historical range.

    IVR = (Current IV - 52-week Low IV) / (52-week High IV - 52-week Low IV) * 100

    - IVR near 100: IV is near its yearly HIGH → options are EXPENSIVE
      → Favor SELLING premium (selling calls, puts, spreads)
      → You collect more premium and theta (time decay) works for you faster
    - IVR near 0: IV is near its yearly LOW → options are CHEAP
      → Favor BUYING options (long calls, long puts)
      → You pay less for the option and have a better risk/reward

    HOW WE APPROXIMATE IV RANK:
    We use 252 days of historical daily closes to estimate realized volatility
    as a proxy for implied volatility history. True IVR requires historical
    IV data (e.g., from CBOE or a paid data provider). Our approximation
    uses 30-day rolling realized volatility as a proxy.

    Returns a 0-100 score where higher means richer IV.
    """
    if symbol not in OPTIONS_SYMBOLS:
        return None

    try:
        ticker = yf.Ticker(symbol)
        # Fetch 1 year of daily data for vol calculation
        hist = ticker.history(period='1y', interval='1d')

        if hist.empty or len(hist) < 30:
            print(f"[options] Insufficient history for IV rank calculation: {symbol}")
            return None

        close = hist['Close'].values

        # Calculate 30-day rolling realized volatility (annualized)
        # WHY: Log returns are used because they're additive and give a better
        # statistical distribution than simple returns. Annualized by sqrt(252).
        log_returns = np.diff(np.log(close))
        window = 21  # ~1 trading month

        rolling_vol = []
        for i in range(window, len(log_returns) + 1):
            daily_vol = np.std(log_returns[i-window:i])
            annualized_vol = daily_vol * math.sqrt(252)
            rolling_vol.append(annualized_vol)

        if len(rolling_vol) < 2:
            return None

        current_vol = rolling_vol[-1]
        vol_52w_low = min(rolling_vol)
        vol_52w_high = max(rolling_vol)

        # Avoid division by zero in pathologically low volatility environments
        if vol_52w_high == vol_52w_low:
            return 50.0

        # IV Rank formula: 0 = historically cheap, 100 = historically expensive
        iv_rank = (current_vol - vol_52w_low) / (vol_52w_high - vol_52w_low) * 100
        return round(iv_rank, 1)

    except Exception as e:
        print(f"[options] Error calculating IV rank for {symbol}: {e}")
        return None


# =============================================================================
# SECTION 2: BLACK-SCHOLES PRICING (FROM SCRATCH)
# =============================================================================

def _norm_cdf(x: float) -> float:
    """
    Standard normal cumulative distribution function.

    WHY: Black-Scholes requires evaluating probabilities under the normal
    distribution. We use Python's math.erfc for accuracy.
    math.erf(x) = (2/sqrt(pi)) * integral from 0 to x of e^(-t^2) dt
    CDF(x) = 0.5 * erfc(-x / sqrt(2))

    This is mathematically equivalent to scipy.stats.norm.cdf but we
    avoid the scipy dependency to keep this module lightweight.
    """
    return 0.5 * math.erfc(-x / math.sqrt(2))


def _norm_pdf(x: float) -> float:
    """
    Standard normal probability density function.

    WHY: The PDF is needed for gamma and vega calculations.
    PDF(x) = (1 / sqrt(2*pi)) * e^(-0.5 * x^2)
    """
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def black_scholes(S: float, K: float, T: float, r: float, sigma: float,
                  option_type: str = 'call') -> float:
    """
    Price a European call or put option using the Black-Scholes formula.

    WHAT BLACK-SCHOLES ASSUMES (and why each matters):
    1. Log-normal stock prices: Prices can't go below zero, returns are normally
       distributed. Works reasonably for liquid equities.
    2. Constant volatility: WRONG in practice — IV smiles and skews exist.
       Black-Scholes underprices tail risk (fat tails).
    3. No dividends: We ignore dividends for simplicity. Use Black-76 for futures.
    4. Continuous hedging possible: Markets are always open, any position size.
       Real markets have gaps and liquidity constraints.
    5. No transaction costs: Pure theoretical price.

    Despite these assumptions, Black-Scholes is still the industry standard
    because (a) it gives a consistent pricing language, and (b) traders use
    the implied volatility surface to capture deviations from BS assumptions.

    THE FORMULA INTUITION:
    A call option is worth the probability-weighted expected value of receiving
    the stock (delta * S * N(d1)) minus the probability-weighted cost of paying
    the strike (K * e^(-rT) * N(d2)).

    d1 and d2 represent how many standard deviations the stock needs to move
    for the option to be in-the-money at expiry.

    Parameters:
        S: Current stock price (e.g., 180.50)
        K: Strike price (e.g., 182.00)
        T: Time to expiry in YEARS (e.g., 30 days = 30/365 = 0.082)
        r: Risk-free rate as decimal (e.g., 0.05 = 5%)
        sigma: Implied/realized volatility as decimal (e.g., 0.25 = 25%)
        option_type: 'call' or 'put'

    Returns: Theoretical option price (premium) in dollars per share.
    Note: Options contracts = 100 shares. Multiply by 100 for contract cost.
    """
    # Guard against invalid inputs
    if T <= 0:
        # Option has expired — intrinsic value only
        if option_type.lower() == 'call':
            return max(0.0, S - K)
        else:
            return max(0.0, K - S)

    if sigma <= 0:
        # Zero volatility edge case — just discounted intrinsic value
        if option_type.lower() == 'call':
            return max(0.0, S - K * math.exp(-r * T))
        else:
            return max(0.0, K * math.exp(-r * T) - S)

    # -------------------------------------------------------------------------
    # STEP 1: Calculate d1 and d2
    # -------------------------------------------------------------------------
    # d1 represents the "adjusted" number of standard deviations the stock
    # needs to be in the money at expiry, accounting for drift (r + sigma^2/2)
    # and the distribution spread (sigma * sqrt(T))
    #
    # d1 = [ln(S/K) + (r + sigma^2/2) * T] / (sigma * sqrt(T))
    #
    # Intuition:
    # - ln(S/K) = how far in/out of the money we currently are (log scale)
    # - (r + sigma^2/2) * T = expected drift of log-price over time T
    # - sigma * sqrt(T) = the "width" of the uncertainty cone
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))

    # d2 = d1 minus one standard deviation unit
    # d2 represents the probability (in normal space) of the option expiring ITM
    # without the adjustment for the expected stock value
    d2 = d1 - sigma * math.sqrt(T)

    # -------------------------------------------------------------------------
    # STEP 2: Calculate option price using CDF of normal distribution
    # -------------------------------------------------------------------------
    if option_type.lower() == 'call':
        # Call price = S * N(d1) - K * e^(-rT) * N(d2)
        # N(d1): probability-weighted expected stock price (above strike)
        # K * e^(-rT): present value of the strike price
        # N(d2): risk-neutral probability that call expires ITM
        price = S * _norm_cdf(d1) - K * math.exp(-r * T) * _norm_cdf(d2)
    else:
        # Put price = K * e^(-rT) * N(-d2) - S * N(-d1)
        # This is derived from call-put parity: C - P = S - K*e^(-rT)
        # N(-d2): probability that put expires ITM
        price = K * math.exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)

    return round(max(0.0, price), 4)


def calculate_greeks(S: float, K: float, T: float, r: float, sigma: float,
                     option_type: str = 'call') -> dict:
    """
    Calculate all four primary option Greeks using Black-Scholes formulas.

    WHAT ARE THE GREEKS?
    Greeks measure the sensitivity of the option's price to changes in
    the underlying variables. They are the core risk management tools
    for options traders — every options desk monitors Greek exposures in
    real-time.

    Parameters: Same as black_scholes()
    Returns: dict with delta, gamma, theta, vega values (and the BS price).
    """
    if T <= 0 or sigma <= 0:
        # Expired or zero-vol option — minimal greeks
        intrinsic = max(0.0, S - K) if option_type.lower() == 'call' else max(0.0, K - S)
        delta = 1.0 if (option_type.lower() == 'call' and S > K) else (
                -1.0 if (option_type.lower() == 'put' and S < K) else 0.0)
        return {'delta': delta, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0,
                'price': intrinsic}

    # Re-calculate d1 and d2 (same as black_scholes above)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)

    # -------------------------------------------------------------------------
    # DELTA — The Hedge Ratio and Directional Exposure
    # -------------------------------------------------------------------------
    # WHAT IT IS: Delta measures how much the option price changes for a $1
    # move in the underlying stock price.
    #
    # WHY TRADERS WATCH IT:
    # - Call delta ranges from 0 to 1. Delta 0.5 means if stock goes up $1,
    #   your call gains ~$0.50.
    # - Put delta ranges from -1 to 0. Delta -0.5 means if stock goes up $1,
    #   your put loses ~$0.50.
    # - ATM options: ~0.5 delta (coin flip on expiry direction)
    # - Deep ITM: delta approaches 1 (behaves like owning the stock)
    # - Deep OTM: delta approaches 0 (lottery ticket)
    #
    # DELTA AS PROBABILITY: N(d2) is roughly the probability of expiring ITM.
    # Delta (N(d1)) is slightly higher due to the stock price weighting.
    #
    # DELTA HEDGING: Market makers "delta hedge" by buying/selling shares
    # to offset their options exposure. This is why options activity
    # can drive stock price movement.
    if option_type.lower() == 'call':
        delta = _norm_cdf(d1)
    else:
        # Put delta = call delta - 1 (by put-call parity)
        # Put delta is always negative: if stock rises, put loses value
        delta = _norm_cdf(d1) - 1.0

    # -------------------------------------------------------------------------
    # GAMMA — Rate of Change of Delta (Convexity)
    # -------------------------------------------------------------------------
    # WHAT IT IS: Gamma measures how fast delta changes as the stock price moves.
    # It is the second derivative of option price with respect to stock price.
    #
    # WHY TRADERS WATCH IT:
    # - High gamma = your delta exposure changes rapidly. A 0.5 delta option
    #   with high gamma might become 0.7 delta after a $5 stock move.
    # - ATM options near expiry have the HIGHEST gamma — their delta can flip
    #   from 0.3 to 0.7 with small price moves. This is "gamma risk."
    # - Options buyers love gamma (convexity works for them — wins are bigger
    #   than losses in $ terms). Sellers fear it.
    #
    # GAMMA IS IDENTICAL FOR CALLS AND PUTS with the same strike/expiry.
    # This is because gamma measures price curvature, not direction.
    gamma = _norm_pdf(d1) / (S * sigma * math.sqrt(T))

    # -------------------------------------------------------------------------
    # THETA — Time Decay (the "Rent" You Pay for Owning an Option)
    # -------------------------------------------------------------------------
    # WHAT IT IS: Theta measures how much option value is lost PER DAY from
    # the passage of time alone, holding everything else constant.
    #
    # WHY TRADERS WATCH IT:
    # - Theta is ALWAYS negative for options buyers. You are paying rent.
    #   Buy a call for $5.00 with theta = -0.05: tomorrow it's worth ~$4.95,
    #   all else equal, just because one day has passed.
    # - Theta accelerates as expiry approaches (the "theta decay cliff" in
    #   the last 30 days). This is why buying weekly options is expensive —
    #   time decay is rapid.
    # - Options SELLERS receive positive theta. They collect premium daily.
    #   This is the core business model of covered call writers.
    #
    # FORMULA NOTE: Theta is divided by 365 to express as daily decay.
    # (Some traders use 252 trading days; 365 calendar days is more standard.)
    theta_call = (
        -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
        - r * K * math.exp(-r * T) * _norm_cdf(d2)
    )
    if option_type.lower() == 'call':
        theta = theta_call / 365.0
    else:
        # Put theta slightly differs from call theta due to the interest component
        theta_put = (
            -(S * _norm_pdf(d1) * sigma) / (2 * math.sqrt(T))
            + r * K * math.exp(-r * T) * _norm_cdf(-d2)
        )
        theta = theta_put / 365.0

    # -------------------------------------------------------------------------
    # VEGA — Sensitivity to Implied Volatility
    # -------------------------------------------------------------------------
    # WHAT IT IS: Vega measures how much the option price changes for a 1%
    # (0.01) change in implied volatility (sigma).
    #
    # WHY TRADERS WATCH IT:
    # - Long options have POSITIVE vega. When uncertainty/fear spikes (VIX
    #   jumps), IV rises, and your options gain value even without a stock move.
    #   This is why buying puts before earnings is expensive — you pay for vega.
    # - Short options have NEGATIVE vega. If you sold a covered call and the
    #   stock makes a big earnings gap, IV expands and your short option gains
    #   value (bad for you).
    # - ATM options have the highest vega. Far OTM/ITM options have lower vega.
    #
    # VEGA IS IDENTICAL FOR CALLS AND PUTS with the same strike/expiry.
    # (Like gamma, vega is a symmetric measure of uncertainty sensitivity)
    #
    # FORMULA NOTE: We divide by 100 to express vega as "per 1% IV change"
    # rather than "per 100% IV change" (which would be useless).
    vega = S * _norm_pdf(d1) * math.sqrt(T) / 100.0

    # Calculate theoretical price while we have d1/d2
    bs_price = black_scholes(S, K, T, r, sigma, option_type)

    return {
        'delta': round(delta, 4),
        'gamma': round(gamma, 6),
        'theta': round(theta, 4),   # Daily dollar decay per share
        'vega': round(vega, 4),     # Dollar change per 1% IV move
        'price': bs_price
    }


# =============================================================================
# SECTION 3: OPTIONS SIGNALS
# =============================================================================

@dataclass
class OptionSignal:
    """
    Represents a specific options trade recommendation.

    WHY A DATACLASS:
    Dataclasses give us a clean, self-documenting structure with automatic
    __init__, __repr__, and __eq__. This is the standard Python pattern for
    value objects.

    FIELDS EXPLAINED:
    - symbol: The underlying stock (AAPL or MSFT)
    - option_type: 'call' or 'put'
    - strike: The strike price of the recommended contract
    - expiry: Expiry date string (e.g., '2024-01-19')
    - premium: Estimated cost to BUY the option (or premium RECEIVED if selling)
    - greeks: dict of delta, gamma, theta, vega from black_scholes
    - action: 'BUY' or 'SELL' (we recommend buying for directional, selling for premium)
    - signal_reason: Plain English explanation of WHY this option was selected
    - breakeven: Price at which the trade breaks even at expiry
    - days_to_expiry: Countdown clock — options have a fixed lifespan
    - iv_rank_at_entry: Record of IV regime at time of signal
    """
    symbol: str
    option_type: str            # 'call' or 'put'
    action: str                 # 'BUY' or 'SELL'
    strike: float
    expiry: str
    premium: float              # Option price per share (contract = premium * 100)
    greeks: dict
    signal_reason: str
    breakeven: float
    days_to_expiry: int
    iv_rank_at_entry: float
    timestamp: datetime = field(default_factory=datetime.now)


def generate_options_signal(symbol: str, price: float, direction: str,
                            iv_rank: float) -> Optional[OptionSignal]:
    """
    Given a directional signal from the strategy ensemble and the current IV regime,
    recommend the BEST options contract to express that view.

    THE CORE INSIGHT — VOLATILITY REGIME MATTERS AS MUCH AS DIRECTION:
    Options have two components of value: intrinsic value and time value.
    Time value is driven by implied volatility. When IV is high, options are
    expensive. When IV is low, they're cheap. This changes the OPTIMAL STRATEGY:

    --- FOUR SCENARIOS ---

    1. BUY signal + LOW IV (IV Rank < 30):
       → BUY an ATM CALL
       WHY: Direction is bullish. IV is cheap, so the premium is affordable.
       Risk: Limited to premium paid. If wrong, you lose the premium.
       Breakeven: Strike + Premium

    2. BUY signal + HIGH IV (IV Rank > 70):
       → SELL an OTM PUT instead of buying a call
       WHY: Bullish directional view, but buying calls when IV is expensive means
       you're paying a high premium. Instead, sell a put below current price —
       you COLLECT that rich premium. If the stock stays above your put strike,
       you keep all the premium. Even if the stock drops somewhat, you profit.
       This is how professional traders express "moderately bullish" in high-IV.
       Risk: Put assignment if stock falls below strike (you're forced to buy).
       Breakeven: Strike - Premium received

    3. SELL signal + LOW IV:
       → BUY an ATM PUT
       WHY: Directional is bearish. IV is cheap, so put premium is affordable.
       Same logic as scenario 1 but in the other direction.

    4. SELL signal + HIGH IV:
       → SELL an OTM CALL (covered call logic)
       WHY: Bearish directional view + expensive options = sell the OTM call,
       collect premium. The call only gets exercised if the stock rallies
       significantly above the strike (which your bearish thesis says won't happen).
       Breakeven: Strike + Premium received

    PARAMETERS:
    - symbol: AAPL or MSFT
    - price: Current stock price
    - direction: 'BUY' (bullish) or 'SELL' (bearish) from the strategy ensemble
    - iv_rank: 0-100 score from fetch_iv_rank()

    Returns an OptionSignal dataclass or None if no suitable contract found.
    """
    if symbol not in OPTIONS_SYMBOLS:
        return None

    # Fetch the options chain to find real contracts
    chain_data = fetch_options_chain(symbol)
    if chain_data is None:
        print(f"[options] Cannot generate signal for {symbol} — no options chain data")
        return None

    dte = chain_data['days_to_expiry']

    # Skip options with very short or no time remaining
    # WHY: Options with < 7 days to expiry have extreme theta decay and gamma
    # risk. The risk/reward profile becomes unfavorable for buying.
    # Short-dated options can lose 50% of their value in a day on no news.
    if dte < 7:
        print(f"[options] {symbol} nearest expiry has only {dte} days — too close")
        return None

    expiry = chain_data['expiry']
    current_price = chain_data['current_price']
    T = dte / 365.0   # Convert days to years for Black-Scholes

    # -------------------------------------------------------------------------
    # DETERMINE STRATEGY BASED ON DIRECTION + IV REGIME
    # -------------------------------------------------------------------------
    low_iv = iv_rank is not None and iv_rank < 30    # IV cheap → buy options
    high_iv = iv_rank is not None and iv_rank > 70   # IV rich → sell options
    iv_rank_val = iv_rank if iv_rank is not None else 50.0

    if direction == 'BUY' and not high_iv:
        # ---- SCENARIO 1: BUY ATM CALL ----
        # ATM = At-The-Money = strike closest to current price
        # WHY ATM: ATM calls have ~0.5 delta (biggest bang for the buck per
        # dollar spent), and fair premium relative to OTM/ITM.
        option_type = 'call'
        action = 'BUY'

        # Find the ATM strike from the real options chain
        calls_df = chain_data['calls']
        if calls_df.empty:
            return None

        atm_idx = (calls_df['strike'] - current_price).abs().idxmin()
        atm_row = calls_df.loc[atm_idx]
        strike = float(atm_row['strike'])

        # Use implied vol from market if available, else estimate
        sigma = float(atm_row['impliedVolatility']) if atm_row['impliedVolatility'] > 0 else 0.25

        # Calculate premium and greeks using our Black-Scholes engine
        premium = black_scholes(current_price, strike, T, RISK_FREE_RATE, sigma, 'call')
        greeks = calculate_greeks(current_price, strike, T, RISK_FREE_RATE, sigma, 'call')

        breakeven = round(strike + premium, 4)
        reason = (
            f"BUY ATM CALL: Bullish signal + IV Rank {iv_rank_val:.0f} (cheap premium). "
            f"Buy ATM call at strike {strike}, premium ~${premium:.2f}/share (${premium*100:.0f}/contract). "
            f"Breakeven at expiry: ${breakeven:.2f}. Limited risk, unlimited upside."
        )

    elif direction == 'BUY' and high_iv:
        # ---- SCENARIO 2: SELL OTM PUT ----
        # OTM put = put with strike BELOW current price
        # WHY: We choose a strike about 5% below current price for the put.
        # This means the stock would need to fall 5%+ before we suffer a loss.
        # Meanwhile we COLLECT the rich premium upfront.
        option_type = 'put'
        action = 'SELL'

        puts_df = chain_data['puts']
        if puts_df.empty:
            return None

        # Target: OTM put ~5% below current price
        target_strike = current_price * 0.95
        otm_idx = (puts_df['strike'] - target_strike).abs().idxmin()
        otm_row = puts_df.loc[otm_idx]
        strike = float(otm_row['strike'])

        sigma = float(otm_row['impliedVolatility']) if otm_row['impliedVolatility'] > 0 else 0.30
        premium = black_scholes(current_price, strike, T, RISK_FREE_RATE, sigma, 'put')
        greeks = calculate_greeks(current_price, strike, T, RISK_FREE_RATE, sigma, 'put')

        # Breakeven for SELLING a put: you make money if stock stays above (strike - premium)
        breakeven = round(strike - premium, 4)
        reason = (
            f"SELL OTM PUT: Bullish signal but IV Rank {iv_rank_val:.0f} (expensive premium). "
            f"Sell {current_price*5:.0f}-delta put at strike {strike}, collect ~${premium:.2f}/share (${premium*100:.0f}/contract). "
            f"Profit if stock stays above ${breakeven:.2f} at expiry. Premium is your max profit."
        )

    elif direction == 'SELL' and not high_iv:
        # ---- SCENARIO 3: BUY ATM PUT ----
        option_type = 'put'
        action = 'BUY'

        puts_df = chain_data['puts']
        if puts_df.empty:
            return None

        atm_idx = (puts_df['strike'] - current_price).abs().idxmin()
        atm_row = puts_df.loc[atm_idx]
        strike = float(atm_row['strike'])

        sigma = float(atm_row['impliedVolatility']) if atm_row['impliedVolatility'] > 0 else 0.25
        premium = black_scholes(current_price, strike, T, RISK_FREE_RATE, sigma, 'put')
        greeks = calculate_greeks(current_price, strike, T, RISK_FREE_RATE, sigma, 'put')

        breakeven = round(strike - premium, 4)
        reason = (
            f"BUY ATM PUT: Bearish signal + IV Rank {iv_rank_val:.0f} (cheap premium). "
            f"Buy ATM put at strike {strike}, premium ~${premium:.2f}/share (${premium*100:.0f}/contract). "
            f"Breakeven at expiry: ${breakeven:.2f}. Profit if stock falls below breakeven."
        )

    else:
        # ---- SCENARIO 4: SELL OTM CALL (high IV + bearish) ----
        option_type = 'call'
        action = 'SELL'

        calls_df = chain_data['calls']
        if calls_df.empty:
            return None

        # Target: OTM call ~5% above current price
        target_strike = current_price * 1.05
        otm_idx = (calls_df['strike'] - target_strike).abs().idxmin()
        otm_row = calls_df.loc[otm_idx]
        strike = float(otm_row['strike'])

        sigma = float(otm_row['impliedVolatility']) if otm_row['impliedVolatility'] > 0 else 0.30
        premium = black_scholes(current_price, strike, T, RISK_FREE_RATE, sigma, 'call')
        greeks = calculate_greeks(current_price, strike, T, RISK_FREE_RATE, sigma, 'call')

        # Breakeven for SELLING a call: you lose money if stock rises above (strike + premium)
        breakeven = round(strike + premium, 4)
        reason = (
            f"SELL OTM CALL: Bearish signal + IV Rank {iv_rank_val:.0f} (expensive premium). "
            f"Sell OTM call at strike {strike}, collect ~${premium:.2f}/share (${premium*100:.0f}/contract). "
            f"Profit if stock stays below ${breakeven:.2f} at expiry (theta works for you)."
        )

    return OptionSignal(
        symbol=symbol,
        option_type=option_type,
        action=action,
        strike=strike,
        expiry=expiry,
        premium=round(premium, 4),
        greeks=greeks,
        signal_reason=reason,
        breakeven=breakeven,
        days_to_expiry=dte,
        iv_rank_at_entry=iv_rank_val
    )


def put_call_ratio(options_chain: dict) -> Optional[dict]:
    """
    Calculate the volume-weighted Put/Call Ratio (PCR) as a sentiment signal.

    WHAT IS PUT/CALL RATIO:
    PCR = Total Put Volume / Total Call Volume

    It tells you the RATIO of bearish bets to bullish bets in the options market.
    Options market participants tend to be more sophisticated than stock buyers/
    sellers (institutional, hedge funds, etc.), so PCR is considered a reliable
    sentiment indicator.

    INTERPRETATION:
    - PCR > 1.2: Bearish sentiment dominates.
      More puts being bought than calls. Fear or hedging activity elevated.
      CONTRARIAN signal: Extreme fear can precede a rally (overcrowded short).
    - PCR < 0.7: Bullish sentiment dominates.
      More calls being bought than puts. Complacency or speculative froth.
      CONTRARIAN signal: Extreme greed can precede a pullback.
    - PCR 0.7 - 1.2: Neutral zone — neither extreme.

    WHY VOLUME NOT OPEN INTEREST:
    Volume = contracts TRADED TODAY (fresh sentiment)
    Open Interest = all outstanding contracts (includes old positions, stale)
    Volume PCR is more timely.

    Parameters:
        options_chain: The dict returned by fetch_options_chain()

    Returns dict with pcr value, interpretation, and sentiment classification.
    """
    if options_chain is None:
        return None

    calls_df = options_chain.get('calls')
    puts_df = options_chain.get('puts')

    if calls_df is None or puts_df is None:
        return None

    # Sum all call and put volumes for the expiry
    # WHY: We look at total volume across all strikes, not just ATM.
    # Institutional hedgers often buy OTM puts — they count in total PCR.
    total_call_volume = calls_df['volume'].fillna(0).sum()
    total_put_volume = puts_df['volume'].fillna(0).sum()

    if total_call_volume == 0:
        # No call volume — can't compute meaningful ratio
        return {'pcr': None, 'interpretation': 'No call volume data', 'sentiment': 'UNKNOWN'}

    pcr = total_put_volume / total_call_volume

    # Classify sentiment based on PCR thresholds
    # These thresholds are commonly used in professional options analytics
    if pcr > 1.2:
        sentiment = 'BEARISH'
        interpretation = (
            f"PCR {pcr:.2f} > 1.2: Heavy put buying relative to calls. "
            "Elevated fear or hedging activity. Contrarian note: extreme "
            "pessimism can precede a short squeeze or relief rally."
        )
    elif pcr < 0.7:
        sentiment = 'BULLISH'
        interpretation = (
            f"PCR {pcr:.2f} < 0.7: Heavy call buying relative to puts. "
            "Speculative or bullish optimism dominates. Contrarian note: "
            "extreme complacency can precede corrections."
        )
    else:
        sentiment = 'NEUTRAL'
        interpretation = (
            f"PCR {pcr:.2f} in neutral zone (0.7-1.2): Balanced put and "
            "call activity. No strong sentiment extreme detected."
        )

    return {
        'pcr': round(pcr, 3),
        'total_call_volume': int(total_call_volume),
        'total_put_volume': int(total_put_volume),
        'sentiment': sentiment,
        'interpretation': interpretation
    }


# =============================================================================
# SECTION 4: PAPER OPTIONS P&L TRACKING
# =============================================================================

@dataclass
class PaperOptionsPosition:
    """
    Tracks a paper options position from entry to expiry.

    WHY TRACK OPTIONS POSITIONS SEPARATELY FROM STOCK POSITIONS:
    Options have fundamentally different P&L mechanics than stocks:
    1. Limited life — they expire worthless or have intrinsic value
    2. Non-linear P&L — the relationship between stock price and option value
       is curved (gamma), not straight
    3. Time decay — the position loses value every day even if stock doesn't move
    4. Vega exposure — IV changes affect the position's mark-to-market value

    PAPER ONLY: These positions exist in simulation only. No real orders are placed.
    """
    symbol: str
    option_type: str            # 'call' or 'put'
    action: str                 # 'BUY' (long) or 'SELL' (short)
    strike: float
    expiry: str
    entry_premium: float        # Premium paid (BUY) or received (SELL) per share
    num_contracts: int          # Each contract = 100 shares
    entry_greeks: dict
    entry_iv_rank: float
    entry_price: float          # Underlying stock price at entry
    entry_time: datetime = field(default_factory=datetime.now)

    def breakeven_at_expiry(self) -> float:
        """
        Calculate the underlying stock price at which this position breaks even.

        BREAKEVEN LOGIC:
        - Long Call: Stock must rise above (strike + premium) to profit.
          The premium is your total cost — you need to recoup it.
        - Long Put: Stock must fall below (strike - premium) to profit.
        - Short Call: Stock must stay below (strike + premium received).
          You keep premium as long as stock doesn't rally above your breakeven.
        - Short Put: Stock must stay above (strike - premium received).

        This is a key number to know before entering any options trade.
        """
        if self.option_type == 'call':
            if self.action == 'BUY':
                return round(self.strike + self.entry_premium, 2)
            else:
                # Short call: you profit below (strike + premium received)
                return round(self.strike + self.entry_premium, 2)
        else:  # put
            if self.action == 'BUY':
                return round(self.strike - self.entry_premium, 2)
            else:
                # Short put: you profit above (strike - premium received)
                return round(self.strike - self.entry_premium, 2)

    def days_remaining(self) -> int:
        """Calculate days left until option expiry (the countdown clock)."""
        expiry_date = datetime.strptime(self.expiry, '%Y-%m-%d').date()
        today = date.today()
        return max(0, (expiry_date - today).days)

    def current_value(self, current_stock_price: float,
                      current_sigma: float = 0.25) -> dict:
        """
        Mark the position to current market value using Black-Scholes.

        WHY MARK-TO-MARKET:
        Options positions need to be revalued constantly because their price
        changes with stock price (delta), time passing (theta), and volatility
        changes (vega). A snapshot of current value shows:
        1. Current P&L (unrealized gain/loss)
        2. Updated Greeks (your risk exposure has changed since entry)
        3. Whether to consider closing early (profit taking or cutting losses)

        PARAMETERS:
        - current_stock_price: Live stock price for underlying
        - current_sigma: Current implied volatility (0.25 = 25% annualized)

        Returns dict with mark price, P&L, updated greeks.
        """
        dte = self.days_remaining()

        if dte <= 0:
            # Option has expired — calculate final intrinsic value
            if self.option_type == 'call':
                intrinsic = max(0.0, current_stock_price - self.strike)
            else:
                intrinsic = max(0.0, self.strike - current_stock_price)
            current_premium = intrinsic
            is_expired = True
        else:
            T = dte / 365.0
            current_premium = black_scholes(
                current_stock_price, self.strike, T,
                RISK_FREE_RATE, current_sigma, self.option_type
            )
            is_expired = False

        # Calculate P&L per share and total P&L (per contract = 100 shares)
        if self.action == 'BUY':
            # Long option: profit = (current premium - entry premium) * 100 * contracts
            pnl_per_share = current_premium - self.entry_premium
        else:
            # Short option: profit = (entry premium - current premium) * 100 * contracts
            # You collected premium upfront; if it's worth less now, you profit
            pnl_per_share = self.entry_premium - current_premium

        total_pnl = pnl_per_share * 100 * self.num_contracts
        pnl_pct = (pnl_per_share / self.entry_premium * 100) if self.entry_premium > 0 else 0

        # Recalculate updated greeks
        if dte > 0:
            T = dte / 365.0
            updated_greeks = calculate_greeks(
                current_stock_price, self.strike, T,
                RISK_FREE_RATE, current_sigma, self.option_type
            )
        else:
            updated_greeks = {'delta': 0.0, 'gamma': 0.0, 'theta': 0.0, 'vega': 0.0}

        return {
            'current_premium': round(current_premium, 4),
            'pnl_per_share': round(pnl_per_share, 4),
            'total_pnl': round(total_pnl, 2),
            'pnl_pct': round(pnl_pct, 2),
            'days_remaining': dte,
            'is_expired': is_expired,
            'updated_greeks': updated_greeks,
            'breakeven': self.breakeven_at_expiry()
        }


# =============================================================================
# MODULE-LEVEL PAPER POSITIONS REGISTRY
# =============================================================================
# WHY: A simple list stored at module level lets the dashboard and other modules
# read current paper positions without needing to pass state objects around.
# This is a simplified approach — a production system would use a database.

paper_positions: list[PaperOptionsPosition] = []


def add_paper_position(signal: OptionSignal, num_contracts: int = 1,
                       current_sigma: float = 0.25) -> PaperOptionsPosition:
    """
    Open a new paper options position based on a generated signal.

    WHY 1 CONTRACT DEFAULT:
    Each contract controls 100 shares. At $180/share AAPL with a $5 ATM call,
    1 contract costs $500. For a learning system, 1 contract is plenty to feel
    the real dollar P&L while staying in a manageable size.

    Returns the new PaperOptionsPosition (also stored in paper_positions list).
    """
    position = PaperOptionsPosition(
        symbol=signal.symbol,
        option_type=signal.option_type,
        action=signal.action,
        strike=signal.strike,
        expiry=signal.expiry,
        entry_premium=signal.premium,
        num_contracts=num_contracts,
        entry_greeks=signal.greeks,
        entry_iv_rank=signal.iv_rank_at_entry,
        entry_price=signal.greeks.get('price', 0.0)
    )
    paper_positions.append(position)
    print(f"[options] Paper position opened: {signal.action} {num_contracts}x {signal.symbol} "
          f"${signal.strike} {signal.option_type.upper()} exp {signal.expiry} "
          f"@ ${signal.premium:.2f}/share")
    return position


def get_all_positions_summary(current_prices: dict) -> list[dict]:
    """
    Get a summary of all paper options positions with current P&L.

    Used by the Streamlit dashboard to display options portfolio status.

    Parameters:
        current_prices: dict mapping symbol → current stock price

    Returns list of dicts with position details and current marks.
    """
    summaries = []
    for pos in paper_positions:
        current_stock_price = current_prices.get(pos.symbol, pos.entry_price)
        mark = pos.current_value(current_stock_price)

        summaries.append({
            'symbol': pos.symbol,
            'type': pos.option_type.upper(),
            'action': pos.action,
            'strike': pos.strike,
            'expiry': pos.expiry,
            'dte': mark['days_remaining'],
            'entry_premium': pos.entry_premium,
            'current_premium': mark['current_premium'],
            'contracts': pos.num_contracts,
            'total_pnl': mark['total_pnl'],
            'pnl_pct': mark['pnl_pct'],
            'breakeven': mark['breakeven'],
            'delta': mark['updated_greeks'].get('delta', 0),
            'theta': mark['updated_greeks'].get('theta', 0),
            'is_expired': mark['is_expired']
        })
    return summaries
