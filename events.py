# =============================================================================
# events.py — Event-Driven Trading: Earnings, Dividends, Corporate Actions
# =============================================================================
# WHY EVENT-DRIVEN TRADING:
# Some of the most predictable (and dangerous) price moves happen around
# scheduled corporate events. Earnings announcements cause the largest single-day
# moves for most stocks. Dividend ex-dates create mechanical price adjustments.
# Understanding WHEN these events occur lets you:
#
#   1. AVOID trading into earnings (IV crush destroys options buyers)
#   2. EXPLOIT post-earnings drift (stocks often continue trending after a surprise)
#   3. TIME dividend strategies (capture dividend vs avoid ex-date price drop)
#   4. SIZE positions smaller when uncertainty is high (pre-earnings)
#
# This is "event-driven" investing — one of the most consistently profitable
# hedge fund strategies. SAC Capital, Third Point, and Greenlight Capital all
# use event catalysts as primary trade triggers. The key is identifying WHICH
# events the market has mispriced.

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional
import config


# =============================================================================
# EARNINGS CALENDAR
# =============================================================================
# WHY EARNINGS MATTER SO MUCH:
# Earnings announcements are the single most important scheduled event for
# equities. They reveal the fundamental truth about a company: did it grow?
# Did it beat expectations? Is the business accelerating or decelerating?
#
# The "Earnings Surprise Effect" (aka SUE — Standardized Unexpected Earnings):
# Academic research shows stocks that beat earnings estimates by more than
# 2 standard deviations continue to DRIFT higher for 60+ days after the
# announcement (Bernard & Thomas 1989). This "Post-Earnings Announcement Drift"
# (PEAD) is one of the most replicated findings in finance.
#
# Why does it persist? Markets underreact to earnings surprises. Analysts
# are slow to revise estimates. Institutional investors can't instantly
# deploy capital. The information diffuses slowly — and that creates alpha.
#
# THE OPTIONS TRAP (IV Crush):
# Before earnings, implied volatility (IV) spikes as options buyers seek
# protection or speculation. After earnings, regardless of direction,
# IV collapses back to normal ("IV crush"). Options bought before earnings
# often LOSE money even when the stock moves in the right direction —
# because the IV collapse destroys more value than the directional move gains.
# This is why experienced traders SELL premium into earnings, not buy it.

def fetch_earnings_calendar(symbols: list = None) -> list[dict]:
    """
    Fetch upcoming earnings announcement dates for a list of equity symbols.

    Returns a list of event dicts sorted by date ascending.
    Non-equity symbols (crypto, forex, futures) are automatically skipped.

    Each event dict contains:
    {
        'symbol': str,
        'event_type': 'EARNINGS',
        'date': str (YYYY-MM-DD),
        'days_until': int,
        'estimated_eps': float or None,   # Analyst consensus EPS estimate
        'surprise_pct': float or None,    # Prior quarter EPS surprise %
        'risk_level': str,                # 'HIGH' | 'MEDIUM' | 'LOW'
        'action': str                     # What the engine should do
    }
    """
    if symbols is None:
        symbols = config.SYMBOLS

    events = []
    today = date.today()

    for symbol in symbols:
        # Skip non-equities — no earnings calendar for crypto/forex/futures
        if (symbol.endswith('-USD') or symbol.endswith('=X') or
                symbol.endswith('=F') or symbol.endswith('-USDT')):
            continue

        try:
            ticker = yf.Ticker(symbol)
            cal = ticker.calendar

            if cal is None or cal.empty:
                continue

            # yfinance calendar has 'Earnings Date' as a column or row
            # The structure changed between yfinance versions, handle both
            earnings_date = None

            if isinstance(cal, pd.DataFrame):
                if 'Earnings Date' in cal.columns:
                    val = cal['Earnings Date'].iloc[0]
                    if pd.notna(val):
                        earnings_date = pd.to_datetime(val).date()
                elif 'Earnings Date' in cal.index:
                    val = cal.loc['Earnings Date'].iloc[0]
                    if pd.notna(val):
                        earnings_date = pd.to_datetime(val).date()

            if earnings_date is None:
                continue

            days_until = (earnings_date - today).days

            # Only care about upcoming events (next 30 days)
            if days_until < 0 or days_until > 30:
                continue

            # Get analyst estimates
            info = ticker.info
            estimated_eps = info.get('epsCurrentYear') or info.get('epsForward')

            # Determine risk level based on how close the earnings are
            if days_until <= 1:
                risk_level = 'CRITICAL'    # Earnings today or tomorrow
                action = 'CLOSE_OR_HEDGE'  # Close positions to avoid binary event
            elif days_until <= 5:
                risk_level = 'HIGH'        # Within a week
                action = 'REDUCE_SIZE'     # Cut position size in half
            elif days_until <= 14:
                risk_level = 'MEDIUM'      # Within two weeks
                action = 'CAUTION'         # Trade with awareness of upcoming event
            else:
                risk_level = 'LOW'         # More than 2 weeks away
                action = 'MONITOR'         # Normal trading, watch calendar

            events.append({
                'symbol': symbol,
                'event_type': 'EARNINGS',
                'date': earnings_date.isoformat(),
                'days_until': days_until,
                'estimated_eps': round(estimated_eps, 4) if estimated_eps else None,
                'risk_level': risk_level,
                'action': action,
            })

        except Exception as e:
            print(f"[events] Could not fetch earnings for {symbol}: {e}")
            continue

    # Sort by date ascending (soonest first)
    events.sort(key=lambda x: x['days_until'])
    return events


# =============================================================================
# DIVIDEND CALENDAR
# =============================================================================
# WHY DIVIDENDS CREATE TRADING OPPORTUNITIES:
# Dividends create MECHANICAL price effects that are entirely predictable:
#
# 1. EX-DATE PRICE DROP: On the ex-dividend date, the stock price drops
#    by approximately the dividend amount at market open. This is not a
#    loss — you receive the dividend — but the price adjustment is mechanical.
#    SHORT SELLERS must pay the dividend on borrowed shares (a cost).
#
# 2. DIVIDEND CAPTURE STRATEGY: Buy before ex-date, receive dividend, sell
#    after. In theory this is zero-sum (price drops by dividend amount).
#    In practice, tax effects and short-term price dynamics create small edges.
#
# 3. HIGH-YIELD SCREENING: Companies paying sustainable dividends are often
#    financially healthy (quality factor overlap). But dividend yield above
#    8-10% often signals a "yield trap" — the dividend is unsustainable and
#    will be cut, destroying the thesis.
#
# 4. AVOID SHORTING BEFORE EX-DATE: A short position before the ex-dividend
#    date owes the dividend payment, increasing the cost of carry.

def fetch_dividend_calendar(symbols: list = None) -> list[dict]:
    """
    Fetch upcoming ex-dividend dates for equity symbols.

    Returns list of dicts containing dividend events in the next 30 days.
    Each dict contains the ex-date, dividend amount, and trading implications.
    """
    if symbols is None:
        symbols = config.SYMBOLS

    events = []
    today = date.today()

    for symbol in symbols:
        # Dividends only apply to equities (and some ETFs)
        if (symbol.endswith('-USD') or symbol.endswith('=X') or
                symbol.endswith('=F') or symbol.endswith('-USDT')):
            continue

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Ex-dividend date
            ex_date_ts = info.get('exDividendDate')
            dividend_rate = info.get('dividendRate')     # Annual dividend $
            dividend_yield = info.get('dividendYield')   # Annual yield %

            if ex_date_ts is None or dividend_rate is None:
                continue

            # Convert timestamp to date
            if isinstance(ex_date_ts, (int, float)):
                ex_date = date.fromtimestamp(ex_date_ts)
            else:
                ex_date = pd.to_datetime(ex_date_ts).date()

            days_until = (ex_date - today).days

            # Only upcoming events
            if days_until < 0 or days_until > 30:
                continue

            # Determine if yield looks sustainable or is a "yield trap"
            if dividend_yield and dividend_yield > 0.10:
                risk_note = 'HIGH YIELD — potential yield trap, check payout ratio'
            elif dividend_yield and dividend_yield > 0.06:
                risk_note = 'Elevated yield — monitor sustainability'
            else:
                risk_note = 'Normal dividend'

            events.append({
                'symbol': symbol,
                'event_type': 'DIVIDEND',
                'date': ex_date.isoformat(),
                'days_until': days_until,
                'dividend_rate': round(dividend_rate, 4),
                'dividend_yield': round(dividend_yield, 4) if dividend_yield else None,
                'risk_note': risk_note,
                'action': 'AVOID_NEW_SHORTS' if days_until <= 3 else 'MONITOR',
            })

        except Exception as e:
            print(f"[events] Could not fetch dividend info for {symbol}: {e}")
            continue

    events.sort(key=lambda x: x['days_until'])
    return events


# =============================================================================
# UNIFIED EVENT FEED
# =============================================================================

def get_all_events(symbols: list = None) -> dict:
    """
    Fetch all upcoming events (earnings + dividends) for all symbols.

    Returns a unified dict:
    {
        'earnings': [...],
        'dividends': [...],
        'all_events': [...],   # merged and sorted
        'high_risk_symbols': [...],   # symbols with events in next 5 days
        'last_updated': str
    }
    """
    if symbols is None:
        symbols = config.SYMBOLS

    earnings = fetch_earnings_calendar(symbols)
    dividends = fetch_dividend_calendar(symbols)

    all_events = earnings + dividends
    all_events.sort(key=lambda x: x['days_until'])

    # Flag symbols with high-risk events in next 5 days
    high_risk = set()
    for event in all_events:
        if event['days_until'] <= 5:
            high_risk.add(event['symbol'])
            if event.get('risk_level') in ('CRITICAL', 'HIGH'):
                high_risk.add(event['symbol'])

    return {
        'earnings': earnings,
        'dividends': dividends,
        'all_events': all_events,
        'high_risk_symbols': list(high_risk),
        'last_updated': datetime.now().isoformat(),
    }


# =============================================================================
# TRADE BLOCKER — Used by engine.py to gate trade execution
# =============================================================================

def should_block_trade(symbol: str, direction: str,
                        events_data: dict = None) -> tuple[bool, str]:
    """
    Determine whether a trade should be blocked due to an upcoming event.

    WHY THIS MATTERS FOR TRADE EXECUTION:
    ========================================
    Trading into a known binary event (like earnings) is gambling, not investing.
    The DIRECTION of the move is random — even if you know the company will beat
    earnings, the stock can sell off ("sell the news") because expectations
    were too high. Trading on rumors works until the news is confirmed and
    everyone rushes for the exit simultaneously.

    Professional risk managers implement "event blackout windows":
    - No new positions within N days of earnings
    - Reduce existing positions before binary events
    - Close positions if earnings are tomorrow

    This function implements that logic for the engine.

    Args:
        symbol: Ticker symbol to check
        direction: 'LONG' or 'SHORT' (some event rules are directional)
        events_data: Pre-fetched events dict (to avoid redundant API calls)
                     If None, will fetch fresh data for just this symbol

    Returns:
        (should_block: bool, reason: str)
        If should_block is True, the engine should NOT open a new position.
    """
    try:
        if events_data is None:
            events_data = get_all_events([symbol])

        # Check all events for this specific symbol
        for event in events_data.get('all_events', []):
            if event['symbol'] != symbol:
                continue

            days = event['days_until']
            event_type = event['event_type']

            # EARNINGS BLACKOUT RULES
            if event_type == 'EARNINGS':
                if days <= 1:
                    return True, (
                        f"EARNINGS TOMORROW — binary event risk. "
                        f"Stock can gap 5-20% in either direction. "
                        f"Position blocked until after announcement."
                    )
                elif days <= 3:
                    return True, (
                        f"EARNINGS IN {days} DAYS — IV elevated, direction uncertain. "
                        f"Risk/reward unfavorable for new positions. Blocked."
                    )
                # Don't block for earnings 4+ days away — just caution

            # DIVIDEND RULES (directional — relevant for shorts)
            if event_type == 'DIVIDEND' and direction == 'SHORT':
                if days <= 2:
                    return True, (
                        f"EX-DIVIDEND IN {days} DAYS — short seller owes dividend. "
                        f"Short position cost of carry is elevated. Blocked."
                    )

        # No blocking events found
        return False, 'OK'

    except Exception as e:
        # NEVER block on an error — fail open (allow trades) to avoid false negatives
        print(f"[events] Error in should_block_trade for {symbol}: {e}")
        return False, 'OK'


# =============================================================================
# IV CRUSH DETECTOR (Pre-earnings Options Warning)
# =============================================================================
# WHY IV CRUSH IS THE #1 OPTIONS TRADING MISTAKE:
# When a stock has earnings approaching, market makers RAISE implied volatility
# on all options. They do this because earnings create uncertainty — the stock
# COULD move 10% in either direction. The options market prices this in.
#
# After earnings, regardless of what the stock does, that uncertainty is RESOLVED.
# Market makers immediately lower IV back to normal levels. This IV collapse can
# destroy 40-60% of an option's value EVEN IF the stock moves in your direction.
#
# EXAMPLE: AAPL reports earnings, stock gaps up 3%
# - If you bought a call option before earnings at high IV: might LOSE value
# - The delta gain from 3% upside = +$0.30 (if delta 0.10)
# - The IV crush on the vega = -$0.50 (if IV drops 10 points)
# - Net: -$0.20 even though you were right on direction
#
# This is why professional options traders SELL into high IV before earnings
# (collect the elevated premium) rather than BUY (pay the elevated premium).

def get_iv_crush_warnings(symbols: list = None) -> list[dict]:
    """
    Generate warnings for symbols with upcoming earnings where IV crush risk
    would affect any open options positions.

    Returns list of warning dicts for symbols with earnings in next 7 days.
    """
    if symbols is None:
        symbols = config.SYMBOLS

    warnings = []
    earnings_events = fetch_earnings_calendar(symbols)

    for event in earnings_events:
        if event['days_until'] <= 7 and event['event_type'] == 'EARNINGS':
            warnings.append({
                'symbol': event['symbol'],
                'earnings_date': event['date'],
                'days_until': event['days_until'],
                'warning': (
                    f"IV CRUSH RISK: {event['symbol']} earnings in {event['days_until']} days. "
                    f"Options buyers face IV collapse after announcement regardless of direction. "
                    f"Consider: (1) Close options before earnings, "
                    f"(2) Sell premium to exploit elevated IV, "
                    f"(3) Use stock instead of options for directional bets."
                ),
                'risk_level': event['risk_level'],
            })

    return warnings
