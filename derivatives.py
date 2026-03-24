# =============================================================================
# derivatives.py — Crypto Perpetuals and Commodity Futures
# =============================================================================
# WHY DERIVATIVES BEYOND OPTIONS:
# Options are one type of derivative. Crypto perpetuals and commodity futures
# are two more — each with distinct mechanics that shape professional trading.
#
# This module covers three areas:
#
# 1. CRYPTO PERPETUALS (BTC-USD, ETH-USD via Binance)
#    Perpetuals are futures contracts with NO expiry date. Instead of expiring,
#    they use a "funding rate" mechanism to keep the perp price anchored to spot.
#    Every major crypto exchange (Binance, Bybit, OKX) offers these.
#    They can be traded with leverage — amplifying both gains AND losses.
#
# 2. COMMODITY FUTURES CURVES (GC=F gold, CL=F crude oil via yfinance)
#    Real commodity futures have fixed expiry. The relationship between
#    the front-month price and next-month price tells you about supply/demand
#    expectations. This is called the "futures curve" or "term structure."
#
# 3. GREEKS EXPLAINER DICT
#    A plain-English reference for the Greeks, importable anywhere.
#
# ALL POSITIONS ARE PAPER ONLY. No real orders, no real risk.

import math
import requests
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# =============================================================================
# SECTION 1: GREEKS EXPLAINER REFERENCE DICTIONARY
# =============================================================================
# WHY A STATIC DICT:
# The Greeks are abstract mathematical concepts that intimidate beginners.
# Having a plain-English explainer importable from this module means
# any part of the system (dashboard, CLI, notebook) can display it easily.
# This is "educational scaffolding" — the system teaches you as you use it.

GREEKS_EXPLAINED = {
    'delta': (
        'How much the option price moves per $1 move in the stock. '
        'Delta 0.5 = option gains $0.50 if stock gains $1. '
        'Ranges from 0 to 1 for calls, -1 to 0 for puts. '
        'ATM options have ~0.5 delta. Deep ITM approaches 1.0 (moves like the stock). '
        'Delta also approximates the probability of expiring in-the-money.'
    ),
    'gamma': (
        'How fast delta changes. High gamma = your delta exposure changes rapidly as price moves. '
        'Example: gamma 0.05 means if stock moves $1, your delta changes by 0.05. '
        'ATM options near expiry have the highest gamma — this is "gamma risk." '
        'Options buyers love gamma (convexity works in their favor). '
        'Options sellers fear gamma (losses accelerate as stock moves against them).'
    ),
    'theta': (
        'Time decay — how much value the option loses per day just from time passing. '
        'Always negative for buyers (you are paying rent on the option). '
        'Example: theta -0.05 means the option loses $0.05 per share per day. '
        'Theta decay accelerates dramatically in the last 30 days before expiry. '
        'This is why weekly option buyers need to be right quickly. '
        'Option sellers receive positive theta — they profit from time passing.'
    ),
    'vega': (
        'Sensitivity to implied volatility. Long options gain value when IV rises (fear/uncertainty spikes). '
        'Example: vega 0.10 means option gains $0.10 per 1% rise in IV. '
        'Buying options before earnings is expensive because vega is already priced in. '
        'After earnings, IV typically collapses ("IV crush") — vega buyers get hurt. '
        'Selling options before earnings collects that rich vega premium.'
    ),
    'rho': (
        'Sensitivity to interest rate changes. Usually the smallest Greek for equities. '
        'Long calls benefit slightly from rising rates (cost of carry increases call value). '
        'Rho matters more for long-dated options (LEAPS) and interest-rate-sensitive assets. '
        'For short-dated stock options, rho is rarely traded actively.'
    )
}


# =============================================================================
# SECTION 2: CRYPTO PERPETUALS
# =============================================================================

# Maintenance margin rate for Binance-style perpetuals
# WHY 0.5%: This is Binance's standard maintenance margin for BTC/ETH perps.
# If your unrealized loss reaches 99.5% of your initial margin, liquidation fires.
MAINTENANCE_MARGIN = 0.005   # 0.5%

# Supported perp symbols (crypto only — equity and commodity perps are different)
PERP_SYMBOLS = ['BTC-USD', 'ETH-USD']

# Binance symbol mapping (perp uses USDT pairs, not USD)
# WHY: yfinance uses 'BTC-USD' but Binance perps trade as 'BTCUSDT'
BINANCE_SYMBOL_MAP = {
    'BTC-USD': 'BTCUSDT',
    'ETH-USD': 'ETHUSDT'
}


@dataclass
class PerpPosition:
    """
    Represents a paper crypto perpetual futures position.

    WHY PERPS ARE DIFFERENT FROM SPOT:
    When you buy BTC spot, you own the coin outright. Zero liquidation risk.
    When you trade BTC perpetuals with leverage, you post a small margin deposit
    and control a much larger position. This amplifies both profits AND losses.

    Example: $1,000 capital at 2x leverage controls a $2,000 BTC position.
    - If BTC rises 5%, you profit $100 (10% on your capital).
    - If BTC falls 50% from entry, you lose $1,000 — your ENTIRE margin.
      The exchange liquidates you before you lose more than you deposited.

    FIELDS EXPLAINED:
    - symbol: 'BTC-USD' or 'ETH-USD'
    - direction: 'LONG' (profit when price rises) or 'SHORT' (profit when falls)
    - entry_price: Price at which the perp position was opened
    - leverage: Multiplier on capital (2x = control 2x your margin)
    - notional_value: Total position size = entry_price * quantity (before leverage)
    - margin_posted: Capital locked up as collateral for the leveraged position
    - liquidation_price: If price reaches this, position is forcibly closed at a loss
    - funding_accumulated: Total funding payments paid/received since opening
    """
    symbol: str
    direction: str              # 'LONG' or 'SHORT'
    entry_price: float
    leverage: float             # e.g., 2.0 for 2x
    quantity: float             # Amount of the asset (e.g., 0.01 BTC)
    notional_value: float       # entry_price * quantity
    margin_posted: float        # notional_value / leverage (capital at risk)
    liquidation_price: float    # Forced close if price reaches this
    funding_accumulated: float = 0.0
    entry_time: datetime = field(default_factory=datetime.now)
    open: bool = True

    def unrealized_pnl(self, current_price: float) -> float:
        """
        Calculate unrealized P&L on the leveraged position.

        WHY SIMPLE BUT POWERFUL:
        The leverage is already baked in via the notional_value.
        P&L = (price change) * quantity
        The quantity was sized at entry to give the leveraged exposure.

        Example: 2x long BTC at $40,000, qty = 0.05 BTC (notional = $2,000)
        Margin posted = $1,000 (half the notional).
        If BTC rises to $42,000: unrealized P&L = $100 (10% on $1,000 margin).
        """
        if self.direction == 'LONG':
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity

    def unrealized_pnl_pct_of_margin(self, current_price: float) -> float:
        """
        P&L as percentage of MARGIN POSTED (not notional) — the relevant measure.

        WHY MARGIN: Leverage means a 5% notional move = 10% margin move at 2x.
        Expressing P&L as % of margin shows you the REAL impact on your capital.
        This is how institutional traders report leveraged P&L.
        """
        if self.margin_posted <= 0:
            return 0.0
        return self.unrealized_pnl(current_price) / self.margin_posted * 100


def calculate_liquidation_price(entry_price: float, leverage: float,
                                direction: str,
                                maintenance_margin: float = MAINTENANCE_MARGIN) -> float:
    """
    Calculate the liquidation price for a leveraged perpetual position.

    WHAT IS LIQUIDATION:
    When you trade with leverage, you borrow capital from the exchange.
    If the trade goes against you far enough, the exchange forcibly closes your
    position to protect itself from losses. This is "liquidation."

    You lose your ENTIRE margin — a complete loss of deposited capital.
    This is the single most dangerous aspect of leveraged trading.

    FORMULA DERIVATION:
    For a LONG position at 2x leverage:
    - You control $2,000 of BTC with $1,000 margin.
    - If BTC drops 50%, the position loses $1,000 — your entire margin.
    - But exchange liquidates BEFORE 100% loss, at the maintenance margin threshold.
    - Liquidation = entry * (1 - 1/leverage + maintenance_margin)

    For SHORT at 2x:
    - You profit from falls, but lose if price rises.
    - Liquidation = entry * (1 + 1/leverage - maintenance_margin)

    WHY 2x MAX FOR LEARNING:
    At 2x leverage, you need a 50% adverse move to get liquidated (minus MM).
    At 10x, you need only a 10% adverse move. At 100x (available on Binance),
    a 1% adverse move wipes you out. The learning value is in understanding
    the mechanism, not in maximizing leverage.

    Parameters:
        entry_price: The price at which the position was opened
        leverage: Leverage multiplier (e.g., 2.0 for 2x)
        direction: 'LONG' or 'SHORT'
        maintenance_margin: Exchange's minimum margin requirement (0.5% = 0.005)

    Returns: Liquidation price (the price at which you get wiped out)
    """
    if leverage <= 0:
        raise ValueError("Leverage must be positive")

    if direction == 'LONG':
        # LONG liquidation: price falls until your margin is consumed
        # Formula: entry * (1 - 1/leverage + maintenance_margin)
        #
        # Breaking this down:
        # - 1/leverage = the fraction of notional you own as margin (e.g., 1/2 = 50% at 2x)
        # - When price falls by (1/leverage - maintenance_margin), your remaining
        #   equity equals just the maintenance margin — exchange liquidates here
        #
        # At 2x with 0.5% MM: entry * (1 - 0.5 + 0.005) = entry * 0.505
        # i.e., a 49.5% price drop from entry liquidates a 2x long
        liq_price = entry_price * (1.0 - (1.0 / leverage) + maintenance_margin)

    else:  # SHORT
        # SHORT liquidation: price rises until your margin is consumed
        # Formula: entry * (1 + 1/leverage - maintenance_margin)
        #
        # At 2x with 0.5% MM: entry * (1 + 0.5 - 0.005) = entry * 1.495
        # i.e., a 49.5% price rise from entry liquidates a 2x short
        liq_price = entry_price * (1.0 + (1.0 / leverage) - maintenance_margin)

    return round(liq_price, 2)


def calculate_funding_cost(position: 'PerpPosition', funding_rate: float) -> float:
    """
    Calculate the funding payment for a perpetual position over one funding period.

    WHAT IS FUNDING IN PERPETUALS:
    Without an expiry date, how does a perpetual futures price stay close to spot?
    The answer is the FUNDING RATE mechanism.

    Every 8 hours, one side pays the other:
    - POSITIVE funding rate (common in bull markets):
      LONGS pay SHORTS. This incentivizes short selling to bring perp price
      down toward spot (when perp trades at a premium to spot).
    - NEGATIVE funding rate (common in bear markets / capitulation):
      SHORTS pay LONGS. Incentivizes buying to bring perp price up toward spot.

    FUNDING COST FORMULA:
    Funding Payment = Position Notional * Funding Rate
    (Positive = you pay, Negative = you receive, depending on direction)

    PRACTICAL IMPACT:
    - BTC funding rate is typically 0.01% per 8 hours = 0.03% per day = ~11% annualized.
    - During bull markets, funding can spike to 0.1%+ per 8h = 1% per day.
    - A leveraged long held through sustained positive funding slowly gets eaten.
    - Professional traders track CUMULATIVE funding as part of total P&L.

    Parameters:
        position: The open PerpPosition
        funding_rate: Current 8-hour funding rate as decimal (e.g., 0.0001 = 0.01%)

    Returns: Dollar amount of funding (positive = you pay, negative = you receive).
    """
    # Funding is calculated on the NOTIONAL value, not just the margin
    # WHY: You borrowed capital to achieve the notional — you pay/receive on the full position
    base_funding = position.notional_value * abs(funding_rate)

    if funding_rate >= 0:
        # Positive funding: longs pay shorts
        if position.direction == 'LONG':
            return base_funding      # Positive = cost for long
        else:
            return -base_funding     # Negative = income for short
    else:
        # Negative funding: shorts pay longs
        if position.direction == 'SHORT':
            return base_funding      # Positive = cost for short
        else:
            return -base_funding     # Negative = income for long


def open_perp_position(symbol: str, direction: str, price: float,
                       leverage: float = 2.0,
                       capital_to_risk: float = 500.0) -> Optional[PerpPosition]:
    """
    Open a paper perpetual futures position.

    WHY PAPER ONLY:
    Perpetuals with leverage carry liquidation risk (complete capital loss).
    We simulate the mechanics to learn without real financial consequences.

    POSITION SIZING LOGIC:
    - capital_to_risk = margin you're posting (e.g., $500)
    - notional_value = capital_to_risk * leverage (e.g., $1,000 at 2x)
    - quantity = notional_value / entry_price (e.g., $1,000 / $50,000 = 0.02 BTC)

    LEVERAGE CAP:
    We hard-cap at 2x leverage for this learning system. The purpose is to
    feel the mechanics (funding rates, liquidation prices, leveraged P&L)
    without the extreme risk of high leverage.

    Parameters:
        symbol: 'BTC-USD' or 'ETH-USD'
        direction: 'LONG' or 'SHORT'
        price: Current market price of the perpetual
        leverage: Multiplier (capped at 2.0 in this system)
        capital_to_risk: Margin amount in USD (dollars pledged as collateral)

    Returns: PerpPosition dataclass or None on invalid inputs.
    """
    if symbol not in PERP_SYMBOLS:
        print(f"[derivatives] {symbol} not in supported perp symbols: {PERP_SYMBOLS}")
        return None

    # Hard cap leverage at 2x for safety
    leverage = min(leverage, 2.0)

    if leverage <= 0 or price <= 0 or capital_to_risk <= 0:
        print(f"[derivatives] Invalid perp position parameters for {symbol}")
        return None

    # Calculate position sizing
    notional_value = capital_to_risk * leverage
    quantity = notional_value / price
    liq_price = calculate_liquidation_price(price, leverage, direction, MAINTENANCE_MARGIN)

    position = PerpPosition(
        symbol=symbol,
        direction=direction,
        entry_price=price,
        leverage=leverage,
        quantity=round(quantity, 8),
        notional_value=round(notional_value, 2),
        margin_posted=round(capital_to_risk, 2),
        liquidation_price=liq_price
    )

    print(f"[derivatives] Paper perp opened: {direction} {quantity:.6f} {symbol} @ ${price:,.2f} | "
          f"Leverage: {leverage}x | Liq: ${liq_price:,.2f} | Margin: ${capital_to_risk:,.2f}")

    return position


def check_liquidation(position: PerpPosition, current_price: float) -> bool:
    """
    Check if a leveraged position should be liquidated at the current price.

    HOW LIQUIDATION WORKS IN PRACTICE:
    The exchange continuously monitors every open position against real-time prices.
    When the mark price (exchange's reference price) touches or crosses the
    liquidation price, the exchange's "auto-deleveraging" engine immediately
    closes the position at market price.

    Liquidation often happens at a WORSE price than the calculated liquidation
    price because the market moves fast. This is called "liquidation slippage."
    In crypto crashes (e.g., May 2021, Nov 2022), billions of leveraged positions
    were liquidated in cascading waves, further driving down prices.

    Parameters:
        position: The open PerpPosition
        current_price: Current market price

    Returns True if the position should be liquidated (margin is depleted).
    """
    if not position.open:
        return False   # Already closed

    liq = position.liquidation_price

    if position.direction == 'LONG' and current_price <= liq:
        print(f"[derivatives] LIQUIDATION TRIGGERED: {position.direction} {position.symbol} "
              f"@ ${current_price:,.2f} (liq price: ${liq:,.2f})")
        return True

    if position.direction == 'SHORT' and current_price >= liq:
        print(f"[derivatives] LIQUIDATION TRIGGERED: {position.direction} {position.symbol} "
              f"@ ${current_price:,.2f} (liq price: ${liq:,.2f})")
        return True

    return False


def fetch_funding_history(symbol: str = 'BTCUSDT', limit: int = 24) -> Optional[list]:
    """
    Fetch historical funding rate data from Binance public API.

    WHY FUNDING HISTORY MATTERS:
    Looking at the last 24 funding rates (covering 8 days at 3 per day) tells you:
    1. Whether funding has been consistently positive (longs paying — bullish speculative excess)
    2. Whether a funding rate spike is a one-off or sustained
    3. How much a long position has paid cumulatively (total carry cost)

    Sustained high positive funding (>0.05% per 8h) is a warning sign:
    - Leveraged longs are heavily positioned
    - A price drop would cascade into forced liquidations
    - This is often seen RIGHT BEFORE major crypto crashes

    Parameters:
        symbol: Binance perp symbol (e.g., 'BTCUSDT')
        limit: Number of historical funding periods to fetch (each period = 8 hours)

    Returns list of dicts with 'fundingTime' and 'fundingRate', or None on error.
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': limit}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse into clean records
        history = []
        for item in data:
            history.append({
                'timestamp': datetime.fromtimestamp(item['fundingTime'] / 1000),
                'funding_rate': float(item['fundingRate']),
                'funding_rate_pct': float(item['fundingRate']) * 100,  # As percentage
                'annualized_pct': float(item['fundingRate']) * 3 * 365 * 100  # 3 periods/day * 365 days
            })

        return history

    except Exception as e:
        print(f"[derivatives] Error fetching funding history for {symbol}: {e}")
        return None


def get_current_funding_rate(symbol: str = 'BTCUSDT') -> Optional[dict]:
    """
    Fetch the most recent funding rate and provide context for interpretation.

    Returns a dict with the rate and plain-English interpretation.
    """
    try:
        url = "https://fapi.binance.com/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': 1}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        if not data:
            return None

        rate = float(data[0]['fundingRate'])
        rate_pct = rate * 100
        annualized_pct = rate * 3 * 365 * 100   # 3 funding periods per day

        # Interpret the funding rate
        if rate > 0.001:    # > 0.1% per 8h — extremely high
            sentiment = 'EXTREME LONG BIAS'
            interpretation = (
                f"Funding {rate_pct:.4f}% ({annualized_pct:.1f}% annualized): DANGEROUSLY HIGH. "
                "Longs paying shorts heavily. Leveraged long exposure is extreme. "
                "This level precedes cascading long liquidations — contrarian SHORT signal."
            )
        elif rate > 0.0003:   # 0.03-0.1% per 8h — high
            sentiment = 'BULLISH EXCESS'
            interpretation = (
                f"Funding {rate_pct:.4f}% ({annualized_pct:.1f}% annualized): elevated. "
                "Longs paying shorts. Mild overheating in long leverage. "
                "Not yet extreme but worth monitoring."
            )
        elif rate > 0:        # Low positive
            sentiment = 'MILD LONG BIAS'
            interpretation = (
                f"Funding {rate_pct:.4f}% ({annualized_pct:.1f}% annualized): normal range. "
                "Slight long bias — not a concern for positioning."
            )
        elif rate < -0.0003:   # Negative — shorts paying longs
            sentiment = 'BEARISH EXCESS'
            interpretation = (
                f"Funding {rate_pct:.4f}% ({annualized_pct:.1f}% annualized): negative. "
                "Shorts paying longs. Leveraged short position crowded. "
                "Potential short squeeze risk — contrarian LONG signal."
            )
        else:
            sentiment = 'NEUTRAL'
            interpretation = (
                f"Funding {rate_pct:.4f}% ({annualized_pct:.1f}% annualized): neutral. "
                "Balanced long/short positioning."
            )

        return {
            'symbol': symbol,
            'funding_rate': rate,
            'funding_rate_pct': round(rate_pct, 6),
            'annualized_pct': round(annualized_pct, 2),
            'sentiment': sentiment,
            'interpretation': interpretation
        }

    except Exception as e:
        print(f"[derivatives] Error fetching current funding rate for {symbol}: {e}")
        return None


# =============================================================================
# SECTION 3: COMMODITY FUTURES CURVES
# =============================================================================

def fetch_futures_curve(symbol: str) -> Optional[dict]:
    """
    Fetch and compare front-month vs next-month futures prices to identify
    whether the market is in contango or backwardation.

    WHAT IS A FUTURES CURVE:
    A commodity futures "curve" plots the price of the same commodity at
    different delivery dates. For example:
    - February gold: $1,950/oz
    - March gold: $1,965/oz
    - April gold: $1,980/oz

    The SHAPE of this curve reveals fundamental supply/demand dynamics.

    HOW WE APPROXIMATE WITH yfinance:
    yfinance provides the front-month futures (e.g., GC=F for gold, CL=F for crude).
    For the "next month," we look at recent price and compare to spot close.
    In a full implementation, you would fetch GCG24, GCH24, etc. (specific monthly contracts).
    Our approach compares recent price action to approximate the curve shape.

    SUPPORTED SYMBOLS:
    - GC=F: COMEX Gold futures (front month)
    - CL=F: NYMEX WTI Crude Oil futures (front month)

    Returns dict with prices, roll yield, and curve shape classification.
    """
    supported = ['GC=F', 'CL=F']
    if symbol not in supported:
        print(f"[derivatives] {symbol} not in supported futures symbols: {supported}")
        return None

    try:
        ticker = yf.Ticker(symbol)

        # Fetch recent daily data
        hist = ticker.history(period='3mo', interval='1d')

        if hist.empty or len(hist) < 30:
            print(f"[derivatives] Insufficient history for {symbol} futures curve")
            return None

        front_price = float(hist['Close'].iloc[-1])

        # Approximate the "next month" price using a 22-day rolling average
        # WHY: In a true futures market, next-month price differs from front-month
        # by storage cost (contango) or supply premium (backwardation).
        # Since yfinance only provides front-month directly, we approximate
        # next-month as the 22-day-ago price of the same contract.
        # This is an approximation — real next-month data requires futures data subscriptions.
        next_price_approx = float(hist['Close'].iloc[-22]) if len(hist) >= 22 else front_price

        roll_yield = calculate_roll_yield(front_price, next_price_approx)
        curve_info = classify_curve_shape(roll_yield)

        # Additional context: 30-day realized volatility
        log_returns = (hist['Close'].pct_change().dropna())
        realized_vol = float(log_returns.tail(30).std() * (252 ** 0.5) * 100)

        return {
            'symbol': symbol,
            'front_price': round(front_price, 4),
            'next_price_approx': round(next_price_approx, 4),
            'roll_yield_pct': round(roll_yield * 100, 4),
            'curve_shape': curve_info['shape'],
            'explanation': curve_info['explanation'],
            'etf_impact': curve_info['etf_impact'],
            'realized_vol_30d': round(realized_vol, 2)
        }

    except Exception as e:
        print(f"[derivatives] Error fetching futures curve for {symbol}: {e}")
        return None


def calculate_roll_yield(front_price: float, next_price: float) -> float:
    """
    Calculate the roll yield from moving between front-month and next-month contracts.

    WHAT IS ROLL YIELD:
    Futures contracts expire. When a contract approaches expiry, holders must
    "roll" — close the expiring contract and open the next one.
    If the next contract is cheaper than the front (backwardation), rolling is
    PROFITABLE (you sell expensive, buy cheap = positive roll yield).
    If next is more expensive (contango), rolling COSTS money (negative roll yield).

    FORMULA:
    Roll Yield = (Front Price - Next Price) / Front Price
    - Positive (front > next): Backwardation — profitable for longs
    - Negative (front < next): Contango — costly for longs

    REAL-WORLD SIGNIFICANCE:
    The United States Oil Fund (USO ETF) famously lost value not because
    crude oil fell but because of persistent contango. In 2020, USO lost
    ~80% of value vs ~40% for spot crude. This "contango drag" is a major
    reason commodity ETFs underperform physical commodities.

    Parameters:
        front_price: Current front-month futures price
        next_price: Next month futures price (or approximation)

    Returns: Roll yield as decimal (e.g., 0.01 = 1% positive = backwardation)
    """
    if front_price <= 0:
        return 0.0
    return (front_price - next_price) / front_price


def classify_curve_shape(roll_yield: float) -> dict:
    """
    Classify the futures curve as BACKWARDATION or CONTANGO and explain why it matters.

    BACKWARDATION — The "Normal" Oil Market:
    Front month price > Next month price → Positive roll yield
    WHY IT HAPPENS:
    - Near-term supply is scarce or demand is urgent
    - Physical buyers pay a premium for immediate delivery
    - Common in oil markets during geopolitical tensions, refinery outages
    - IMPLICATIONS: Longs are REWARDED for holding (positive carry)
    - Example: May 2022 oil crisis — Brent front month traded $5+ above deferred

    CONTANGO — The "Storage Glut" Market:
    Front month price < Next month price → Negative roll yield
    WHY IT HAPPENS:
    - Excess supply in the near term, storage costs money
    - The market prices in carrying costs (storage + insurance + financing)
    - Very common in gold (which has storage costs and no consumable demand)
    - IMPLICATIONS: Longs are PENALIZED for holding (negative carry)
    - Example: April 2020 — WTI crude briefly went NEGATIVE as storage was full

    ETF DECAY IN CONTANGO:
    This is the most important practical lesson from futures curves.
    Commodity ETFs that roll futures monthly (USO, UNG) continuously sell the
    cheap expiring contract and buy the expensive next contract.
    Over time, this "selling low, buying high" erodes the ETF's value
    even if the spot commodity price is flat. Contango = passive long tax.

    Returns a dict with shape, explanation, and ETF impact warning.
    """
    if roll_yield > 0.005:     # > 0.5% positive
        shape = 'BACKWARDATION'
        explanation = (
            f"Roll yield {roll_yield*100:.3f}%: BACKWARDATION — near-term supply is tight "
            "or demand is urgent. Front-month premium indicates immediate delivery scarcity. "
            "Positive for longs: rolling contracts generates income (sell expensive, buy cheaper). "
            "Common during supply disruptions, geopolitical crises, unexpected demand surges."
        )
        etf_impact = (
            "FAVORABLE for commodity ETF holders: positive roll yield adds to returns "
            "beyond spot price changes. Physical commodity longs and commodity ETFs "
            "both benefit from backwardation."
        )
    elif roll_yield < -0.005:   # > 0.5% negative
        shape = 'CONTANGO'
        explanation = (
            f"Roll yield {roll_yield*100:.3f}%: CONTANGO — near-term oversupply or "
            "high storage costs. Each month's contract prices in storage and carry costs. "
            "Negative for longs: rolling contracts is a continual cost (sell cheap, buy expensive). "
            "Common in gold (permanent storage cost) and oil during glut periods."
        )
        etf_impact = (
            "WARNING — CONTANGO DRAG on commodity ETFs: "
            "ETFs like USO (oil) and UNG (natural gas) roll futures monthly. "
            "In steep contango, this rolling cost can subtract 2-10%+ annually "
            "from returns even if the spot commodity is flat. "
            "This is why USO dramatically underperformed crude oil in 2020. "
            "Consider direct commodity exposure or backwardation-focused strategies."
        )
    else:                       # Near-flat curve
        shape = 'FLAT / NEUTRAL'
        explanation = (
            f"Roll yield {roll_yield*100:.3f}%: Approximately flat curve. "
            "Front and next month prices are nearly equal. "
            "No strong supply/demand imbalance signal from the curve alone. "
            "Monitor for curve shifts as a leading supply indicator."
        )
        etf_impact = (
            "Minimal roll yield impact: ETF holders experience near-zero carry cost "
            "from rolling. Performance tracks spot price changes closely."
        )

    return {
        'shape': shape,
        'explanation': explanation,
        'etf_impact': etf_impact,
        'roll_yield_pct': round(roll_yield * 100, 4)
    }


# =============================================================================
# MODULE-LEVEL PERP POSITIONS REGISTRY
# =============================================================================
# WHY MODULE-LEVEL: Mirrors the pattern used in options.py — a simple list
# that can be read by the dashboard without passing state objects.

perp_positions: list[PerpPosition] = []


def get_perp_summary(current_prices: dict) -> list[dict]:
    """
    Get a summary of all open paper perp positions with P&L and liquidation status.

    Used by the Streamlit dashboard to display derivatives portfolio.

    Parameters:
        current_prices: dict mapping symbol → current price

    Returns list of dicts suitable for Streamlit display.
    """
    summaries = []
    for pos in perp_positions:
        if not pos.open:
            continue

        current_price = current_prices.get(pos.symbol, pos.entry_price)
        is_liquidated = check_liquidation(pos, current_price)

        if is_liquidated:
            pos.open = False
            upnl = -pos.margin_posted   # Complete margin loss on liquidation
            upnl_pct = -100.0
        else:
            upnl = pos.unrealized_pnl(current_price)
            upnl_pct = pos.unrealized_pnl_pct_of_margin(current_price)

        # Distance to liquidation as % — how close are we to the cliff?
        if pos.direction == 'LONG':
            distance_to_liq_pct = (current_price - pos.liquidation_price) / current_price * 100
        else:
            distance_to_liq_pct = (pos.liquidation_price - current_price) / current_price * 100

        summaries.append({
            'symbol': pos.symbol,
            'direction': pos.direction,
            'leverage': pos.leverage,
            'entry_price': pos.entry_price,
            'current_price': current_price,
            'liquidation_price': pos.liquidation_price,
            'margin_posted': pos.margin_posted,
            'notional_value': pos.notional_value,
            'unrealized_pnl': round(upnl, 2),
            'unrealized_pnl_pct': round(upnl_pct, 2),
            'funding_accumulated': round(pos.funding_accumulated, 4),
            'distance_to_liq_pct': round(distance_to_liq_pct, 2),
            'is_liquidated': is_liquidated,
            'entry_time': pos.entry_time.strftime('%m/%d %H:%M')
        })

    return summaries
