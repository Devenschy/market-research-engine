# =============================================================================
# events.py — Earnings Calendar + Corporate Events Module
# =============================================================================
#
# WHY EVENT RISK MANAGEMENT IS CRITICAL:
# ========================================
# Event-driven strategies profit from predictable price moves around scheduled
# corporate events. But the flip side is equally important: EVENT RISK is the
# single biggest source of catastrophic losses for unprepared retail traders.
#
# THE KEY EVENTS AND THEIR MARKET IMPACT:
#
# 1. EARNINGS ANNOUNCEMENTS (highest impact):
#    - Stocks typically move 3-8% on earnings day for large-cap stocks
#    - Small/mid-cap stocks often move 10-20%+ on earnings surprises
#    - Options market PRICES IN this expected move via elevated Implied Volatility (IV)
#      in the days before earnings
#    - After earnings, IV COLLAPSES (the event is resolved, uncertainty gone)
#    - This IV crush destroys option buyers even if they got direction right
#
# 2. EX-DIVIDEND DATE:
#    - If you own the stock BEFORE this date, you receive the upcoming dividend
#    - The stock price DROPS by approximately the dividend amount on ex-div date
#      because the value of the future dividend is "removed" from the stock
#    - This is mechanical, not speculative — it's an accounting certainty
#    - Buying a stock the day before ex-div to capture the dividend is called
#      "dividend capture" — usually not profitable after taxes and bid-ask spread
#
# 3. ANALYST UPGRADES/DOWNGRADES:
#    - A major bank upgrade (e.g., Goldman Sachs upgrading AAPL to Buy)
#      can move a stock 2-5% on the day
#    - These are hard to predict in advance but explain many "random" gaps
#
# 4. M&A ANNOUNCEMENTS:
#    - Acquisition TARGETS typically jump 20-40% instantly on announcement day
#    - This is called the "acquisition premium" — buyers pay above market price
#    - Acquirers often fall 2-5% (market worried about overpayment)
#
# THE TWO MOST IMPORTANT RULES FOR RETAIL TRADERS:
#
#   RULE 1 — EARNINGS RULE (used by virtually all professional options traders):
#   "Do not hold short-term options through earnings. The IV crush will kill you."
#
#   Here is exactly what happens:
#   - Stock is at $100. Earnings in 2 days. Options pricing in +-$5 expected move.
#   - You buy a call option for $3.00 (IV is elevated at 80% annualized)
#   - Earnings come out: stock goes UP $3 (direction correct!)
#   - But IV collapses from 80% to 25% after the event resolves
#   - Your call is now worth $1.50 despite being right on direction
#   - You LOSE 50% even though you correctly predicted the stock would go up
#   - This is IV crush. It is real, repeatable, and devastating.
#
#   RULE 2 — EX-DIVIDEND RULE (used by options traders and long stock holders):
#   "Do not buy new LONG positions the day before ex-dividend.
#    The expected drop in stock price equals your expected dividend gain."
#   You are not getting "free money." You're paying for the dividend in price drop.
#
# WHY THIS MODULE EXISTS IN THE TRADING ENGINE:
# The engine needs a gatekeeper function (should_block_trade) that is called
# BEFORE any trade is executed. Professional trading desks have exactly this:
# a pre-trade check against the earnings calendar and dividend calendar.
# Systematic risk management > emotional decision-making.
#
# =============================================================================

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta, date
from typing import Optional
import config


# =============================================================================
# HELPER: Identify asset class (events only apply to equities)
# =============================================================================

def _is_equity(symbol: str) -> bool:
    """
    Rough heuristic to detect if a symbol is an equity (stock).

    WHY: Earnings calendars and dividend schedules only exist for publicly
    traded equities. Bitcoin has no quarterly earnings. Gold futures pay
    no dividends. EUR/USD has no ex-dividend date.

    Rather than making an API call that would fail for these asset classes,
    we pre-filter using yfinance's naming convention:
    - Crypto: ends in '-USD' or '-USDT' (e.g., BTC-USD, ETH-USD)
    - Forex: ends in '=X' (e.g., EURUSD=X, GBPUSD=X)
    - Futures: ends in '=F' (e.g., GC=F for gold, CL=F for crude oil)
    - Equities: plain ticker symbols with no special suffix (AAPL, MSFT, TSLA)
    """
    if symbol.endswith('-USD') or symbol.endswith('-USDT'):
        return False  # Crypto — no corporate events
    if symbol.endswith('=X'):
        return False  # Forex pair — no earnings or dividends
    if symbol.endswith('=F'):
        return False  # Futures contract — no corporate events
    return True       # Assume equity with potential corporate events


# =============================================================================
# FUNCTION 1: EARNINGS CALENDAR (single symbol)
# =============================================================================

def fetch_earnings_calendar(symbol: str) -> dict:
    """
    Fetch the next earnings date for a single equity and assess the risk level.

    THE MECHANICS OF EARNINGS RISK:
    =================================
    Every publicly traded company must report quarterly earnings (10-Q) and
    annual results (10-K) to the SEC. These reports are scheduled weeks in advance
    and are among the most anticipated events in markets.

    WHY THE 2-DAY RULE EXISTS:
    --------------------------
    The "2 days before earnings" rule is the most widely used heuristic among
    professional options traders (Tastytrade, CBOE educational materials,
    every major options desk). Here is the detailed mechanics:

    TIMELINE OF AN EARNINGS EVENT:
    - Day -14 to -3: IV gradually rises as uncertainty about earnings approaches.
      Smart money begins reducing exposure. Some speculators buy options.
    - Day -2 to -1: IV spikes sharply. Options become very expensive.
      New positions have terrible risk/reward: you're paying maximum premium
      for options that will lose 60-80% of value after the event.
    - Day 0 (Earnings day): Price gaps significantly on open.
      IV immediately collapses back to normal (event is resolved).
    - Day +1: Normal trading resumes. IV is low again.

    THE KEY INSIGHT: You can be RIGHT on direction and STILL lose money if you
    bought options in the 2 days before earnings. The IV crush is that severe.

    For STOCK positions (not options), the risk is different:
    - Gap risk: The stock can open 10-20% lower than your stop-loss price.
      Your stop-loss order only triggers at the NEXT available price,
      which could be far below where you set it.
    - Example: You own AAPL with a stop at $180. Earnings miss.
      Stock opens at $165. You get filled at $165, not $180.
      "Gap risk" = the difference between your stop and actual fill.

    RISK LEVELS:
    - NONE: No upcoming earnings scheduled
    - LOW: 3-14 days away — be aware, but normal position sizing OK
    - MEDIUM: 3-5 days away — reduce position size, tighten stops
    - HIGH: 0-2 days away — do not open new positions, consider closing

    The 2-day rule before earnings is used by most professional options traders.
    IV spikes in the 2 days before earnings and collapses after — this IV crush
    destroys option buyers even when they get direction right.

    Args:
        symbol: Equity ticker symbol (e.g., 'AAPL', 'MSFT')

    Returns:
        dict with earnings date, days until earnings, risk level, and recommendation.
        Returns NONE risk level for non-equities (crypto, forex, commodities).
    """
    # Earnings only exist for equities
    if not _is_equity(symbol):
        return {
            'symbol': symbol,
            'next_earnings': None,
            'days_until_earnings': None,
            'earnings_risk': 'NONE',
            'recommendation': f"{symbol} is not an equity — no earnings calendar applies.",
        }

    try:
        ticker = yf.Ticker(symbol)
        today = date.today()
        next_earnings_date = None

        # === METHOD 1: ticker.calendar ===
        # This is the most reliable method. Returns a dict-like object with
        # 'Earnings Date' as a list of timestamps (sometimes 2 dates for a range).
        try:
            cal = ticker.calendar
            if cal is not None:
                if isinstance(cal, dict):
                    earnings_dates = cal.get('Earnings Date', [])
                    if earnings_dates:
                        for ed in earnings_dates:
                            if hasattr(ed, 'date'):
                                ed_date = ed.date()
                            elif hasattr(ed, 'to_pydatetime'):
                                ed_date = ed.to_pydatetime().date()
                            else:
                                ed_date = pd.Timestamp(ed).date()
                            if ed_date >= today:
                                next_earnings_date = ed_date
                                break
                elif isinstance(cal, pd.DataFrame):
                    if 'Earnings Date' in cal.columns:
                        val = cal['Earnings Date'].iloc[0]
                        if pd.notna(val):
                            ts = pd.Timestamp(val)
                            if ts.date() >= today:
                                next_earnings_date = ts.date()
                    elif 'Earnings Date' in cal.index:
                        val = cal.loc['Earnings Date'].iloc[0]
                        if pd.notna(val):
                            ts = pd.Timestamp(val)
                            if ts.date() >= today:
                                next_earnings_date = ts.date()
        except Exception as cal_err:
            print(f"[events] calendar method failed for {symbol}: {cal_err}")

        # === METHOD 2: ticker.earnings_dates (fallback) ===
        # earnings_dates is a DataFrame indexed by date, sorted newest first.
        # We look for the nearest FUTURE date.
        if next_earnings_date is None:
            try:
                earnings_df = ticker.earnings_dates
                if earnings_df is not None and not earnings_df.empty:
                    future_earnings = earnings_df[
                        earnings_df.index.normalize() >= pd.Timestamp(today)
                    ]
                    if not future_earnings.empty:
                        next_earnings_date = future_earnings.index[-1].date()
            except Exception as ed_err:
                print(f"[events] earnings_dates method failed for {symbol}: {ed_err}")

        # === COMPUTE DAYS UNTIL EARNINGS ===
        if next_earnings_date is not None:
            days_until = (next_earnings_date - today).days
        else:
            days_until = None

        # === RISK LEVEL CLASSIFICATION ===
        # These thresholds are industry-standard heuristics, not arbitrary.
        # See: Tastytrade "Managing Earnings Risk" and CBOE options education.
        if days_until is None:
            earnings_risk = 'NONE'
            recommendation = "No earnings date found in the next reporting period. Safe to trade normally."
        elif days_until <= 2:
            earnings_risk = 'HIGH'
            recommendation = (
                f"EARNINGS IN {days_until} DAY(S) — HIGH RISK. "
                f"Do NOT open new positions. Consider closing existing positions. "
                f"IV is spiking and will crush option buyers. Gap risk for stock holders. "
                f"The 2-day rule: professional options traders universally avoid new "
                f"positions within 2 days of earnings due to IV crush risk."
            )
        elif days_until <= 5:
            earnings_risk = 'MEDIUM'
            recommendation = (
                f"Earnings in {days_until} days — MEDIUM RISK. "
                f"Reduce position size by 50%. Tighten stop-loss to minimize gap risk. "
                f"IV is beginning to rise — option premium is becoming expensive."
            )
        elif days_until <= 14:
            earnings_risk = 'LOW'
            recommendation = (
                f"Earnings in {days_until} days — LOW RISK. "
                f"Normal trading OK, but be aware of the upcoming event. "
                f"Start planning your exit strategy now."
            )
        else:
            earnings_risk = 'NONE'
            recommendation = (
                f"Earnings in {days_until} days — no immediate risk. "
                f"Normal position sizing and risk management apply."
            )

        return {
            'symbol': symbol,
            'next_earnings': next_earnings_date.isoformat() if next_earnings_date else None,
            'days_until_earnings': days_until,
            'earnings_risk': earnings_risk,
            'recommendation': recommendation,
        }

    except Exception as e:
        print(f"[events] Error fetching earnings calendar for {symbol}: {e}")
        return {
            'symbol': symbol,
            'next_earnings': None,
            'days_until_earnings': None,
            'earnings_risk': 'NONE',
            'recommendation': f"Could not retrieve earnings data for {symbol}: {e}",
        }


# =============================================================================
# FUNCTION 2: DIVIDEND CALENDAR (single symbol)
# =============================================================================

def fetch_dividend_calendar(symbol: str) -> dict:
    """
    Fetch the next ex-dividend date and dividend amount for a single equity.

    THE EX-DIVIDEND MECHANICS (one of the most misunderstood topics):
    ==================================================================
    The ex-dividend date is the cutoff date for receiving an upcoming dividend.
    To receive the dividend, you must own the stock BEFORE this date.

    TIMELINE:
    - Declaration Date: Company announces the dividend (e.g., "We will pay $0.25/share")
    - Ex-Dividend Date: The cutoff. Buy before this date = you get the dividend.
    - Record Date: Usually 1 business day after ex-div. Company takes a snapshot.
    - Payment Date: The actual cash arrives in your account (typically 3-4 weeks later).

    THE PRICE MECHANICS (why this matters for trading):
    ---------------------------------------------------
    On the ex-dividend date, the stock price is EXPECTED to drop by approximately
    the dividend amount at market open. This is not arbitrary — it's mathematical:

    Before ex-div: Stock price includes the right to receive the upcoming dividend.
    After ex-div: New buyers do NOT receive the dividend. So the "right to receive"
    is no longer embedded in the share price. The stock is worth less by exactly
    that amount (in an efficient market).

    Example:
    - AAPL closes at $180.00 on the day before ex-div, with a $0.25 dividend
    - On ex-div date, AAPL opens at approximately $179.75 (adjusted)
    - If you owned the stock before ex-div, you receive $0.25/share in cash
    - Net result: you have $179.75 stock + $0.25 cash = $180.00 value
    - You didn't gain anything from the dividend — the price adjusted

    THE DIVIDEND CAPTURE MYTH:
    Many retail investors think they can buy a stock the day before ex-div,
    collect the dividend, then sell. In practice:
    1. The price drops by the dividend amount — net gain is near zero
    2. Dividends may be taxed as ordinary income depending on holding period
    3. Bid-ask spread and commissions erode the tiny theoretical gain
    Dividend capture strategies only work at institutional scale.

    WHY WE BLOCK LONGS BEFORE EX-DIV:
    Opening a new LONG position the day before ex-div means you are paying the
    full current price (which includes the upcoming dividend's value), then
    receiving a dividend that exactly offsets the price drop. The risk/reward
    for a new entry here is unfavorable.

    Ex-dividend date: if you own the stock BEFORE this date, you get the dividend.
    The stock price typically falls by ~the dividend amount on this date as the
    value has been paid out.

    Args:
        symbol: Equity ticker symbol

    Returns:
        dict with ex-dividend date, dividend amount, and days until ex-div.
        Returns None fields for non-equities or symbols with no dividend history.
    """
    # Dividends only exist for equities (stocks and some ETFs)
    if not _is_equity(symbol):
        return {
            'symbol': symbol,
            'next_ex_dividend': None,
            'dividend_amount': None,
            'days_until_ex_div': None,
        }

    try:
        ticker = yf.Ticker(symbol)
        today = date.today()
        next_ex_div_date = None
        dividend_amount = None

        # === METHOD 1: ticker.info exDividendDate field ===
        # info['exDividendDate'] is returned as a Unix timestamp (integer seconds)
        # for most equity symbols. This is the most direct source.
        try:
            info = ticker.info
            ex_date_ts = info.get('exDividendDate')
            dividend_rate = info.get('dividendRate')  # Annual dividend in dollars

            if ex_date_ts is not None:
                if isinstance(ex_date_ts, (int, float)):
                    candidate = date.fromtimestamp(ex_date_ts)
                else:
                    candidate = pd.Timestamp(ex_date_ts).date()
                if candidate >= today:
                    next_ex_div_date = candidate

            if dividend_rate is not None:
                # dividendRate is annual; divide by 4 for typical quarterly payment
                dividend_amount = float(dividend_rate) / 4.0
        except Exception:
            pass  # Fall through to method 2

        # === METHOD 2: ticker.dividends history (fallback) ===
        # If the info field didn't give us a future ex-div date, we estimate
        # the next one from historical dividend payment data.
        if next_ex_div_date is None:
            try:
                dividends = ticker.dividends
                if dividends is not None and not dividends.empty:
                    # Most recent actual payment amount
                    dividend_amount = float(dividends.iloc[-1])

                    # Estimate next payment date from average payment frequency
                    if len(dividends) >= 2:
                        dates = dividends.index.to_list()
                        gaps = [(dates[i+1] - dates[i]).days
                                for i in range(len(dates) - 1)]
                        avg_gap = sum(gaps[-4:]) / len(gaps[-4:])  # Last 4 gaps
                    else:
                        avg_gap = 90  # Default assumption: quarterly

                    last_payment_date = dividends.index[-1].date()
                    estimated_next = last_payment_date + timedelta(days=int(avg_gap))

                    if estimated_next >= today:
                        next_ex_div_date = estimated_next
            except Exception as div_err:
                print(f"[events] Dividend history fallback failed for {symbol}: {div_err}")

        # === DAYS UNTIL EX-DIV ===
        days_until_ex_div = (next_ex_div_date - today).days if next_ex_div_date else None

        return {
            'symbol': symbol,
            'next_ex_dividend': next_ex_div_date.isoformat() if next_ex_div_date else None,
            'dividend_amount': round(dividend_amount, 4) if dividend_amount else None,
            'days_until_ex_div': days_until_ex_div,
        }

    except Exception as e:
        print(f"[events] Error fetching dividend calendar for {symbol}: {e}")
        return {
            'symbol': symbol,
            'next_ex_dividend': None,
            'dividend_amount': None,
            'days_until_ex_div': None,
        }


# =============================================================================
# FUNCTION 3: AGGREGATE ALL EVENTS FOR ALL SYMBOLS
# =============================================================================

def get_all_events(symbols: list = None) -> dict:
    """
    Fetch both earnings and dividend calendars for all symbols in one call.

    WHY A BATCH FUNCTION:
    =====================
    The engine's main loop needs to check events for all symbols at startup
    and periodically during the trading day. Having a single function that
    returns a clean, structured summary is easier to integrate than calling
    individual functions per symbol in the main loop.

    The 'high_risk_symbols' key is the most important output: it's a list
    of symbols that should be flagged for immediate attention by the trader.
    Any symbol here should have positions reviewed NOW.

    Args:
        symbols: List of ticker symbols to check. Defaults to config.SYMBOLS.

    Returns:
        dict with:
        - 'earnings': {symbol: earnings_calendar_dict}
        - 'dividends': {symbol: dividend_calendar_dict}
        - 'high_risk_symbols': list of symbols with earnings within 2 days
        - 'last_updated': ISO timestamp of when this was fetched
    """
    if symbols is None:
        symbols = config.SYMBOLS

    earnings_results = {}
    dividend_results = {}
    high_risk_symbols = []

    for symbol in symbols:
        print(f"[events] Fetching calendar events for {symbol}...")

        earnings_data = fetch_earnings_calendar(symbol)
        dividend_data = fetch_dividend_calendar(symbol)

        earnings_results[symbol] = earnings_data
        dividend_results[symbol] = dividend_data

        # Flag HIGH earnings risk symbols for immediate attention
        if earnings_data.get('earnings_risk') == 'HIGH':
            high_risk_symbols.append(symbol)

    return {
        'earnings': earnings_results,
        'dividends': dividend_results,
        'high_risk_symbols': high_risk_symbols,
        'last_updated': datetime.utcnow().isoformat() + 'Z',
    }


# =============================================================================
# FUNCTION 4: PRE-TRADE EVENT CHECK GATE
# =============================================================================

def should_block_trade(symbol: str, signal_direction: str,
                        events_data: dict = None) -> tuple[bool, str]:
    """
    Master pre-trade risk gate: should we block this trade due to an upcoming event?

    WHY THIS FUNCTION IS THE MOST IMPORTANT ONE IN THIS MODULE:
    ============================================================
    Every professional trading system has a set of PRE-TRADE CHECKS that run
    before any order is sent to the market. This function is one of those checks.

    Think of it as the "compliance officer" sitting between the signal generator
    and the order router. The signal generator says "BUY AAPL." This function
    checks: "Is there an earnings announcement in 2 days? If yes, block the trade."

    This is not optional risk management. At professional desks, traders who
    hold through earnings without explicit approval face immediate disciplinary
    action. The systematic version of that rule is this function.

    BLOCKING RULES:

    RULE 1 — EARNINGS WITHIN 2 DAYS: Block ALL new positions (both long and short).
    WHY: Whether you're long or short, the binary outcome of an earnings surprise
    creates unacceptable gap risk. You cannot know if the stock will open +15%
    or -15%. Any directional position is essentially a coin flip at unfavorable
    odds (you're paying elevated IV premium for options, or gap risk for stock).
    The only safe action is no action.

    RULE 2 — EX-DIVIDEND WITHIN 1 DAY: Block new LONG positions only.
    WHY: If you buy a stock today and tomorrow is ex-div, you ARE buying before
    the ex-div date (so you get the dividend), but you're paying the pre-ex-div
    price that already includes the dividend's value. When the stock drops by
    the dividend amount tomorrow at open, your position is immediately under water.
    Short positions benefit from this drop, so they are NOT blocked.
    Note: we do NOT block shorts — ex-div is actually favorable for short sellers
    because the expected price drop aligns with their position direction.

    Args:
        symbol: The ticker symbol to check
        signal_direction: 'BUY' or 'SELL' — direction of the proposed trade
        events_data: Pre-fetched events dict (to avoid redundant API calls).
                     If None, will fetch fresh data for just this symbol.

    Returns:
        tuple: (should_block: bool, reason: str)
        - (True, "reason string") if the trade should be blocked
        - (False, "OK") if safe to proceed
    """
    # Non-equities have no earnings/dividend risk — always allow
    if not _is_equity(symbol):
        return (False, 'OK')

    try:
        # --- CHECK 1: EARNINGS PROXIMITY ---
        # Higher-priority check — earnings risk overrides dividend risk.
        if events_data is not None:
            # Re-use pre-fetched data if available (avoids extra API call)
            earnings_info = events_data.get('earnings', {}).get(symbol)
        else:
            earnings_info = None

        if earnings_info is None:
            earnings_info = fetch_earnings_calendar(symbol)

        days_until_earnings = earnings_info.get('days_until_earnings')
        earnings_risk = earnings_info.get('earnings_risk', 'NONE')

        if earnings_risk == 'HIGH':
            reason = (
                f"BLOCKED — Earnings in {days_until_earnings} day(s) for {symbol}. "
                f"The 2-day earnings rule prohibits opening new positions. "
                f"IV is elevated (options are expensive) and gap risk is extreme. "
                f"Wait until after earnings are reported and IV normalizes."
            )
            return (True, reason)

        # --- CHECK 2: EX-DIVIDEND PROXIMITY ---
        # Only block LONG (BUY) positions before ex-dividend.
        # Short positions benefit from the ex-div price drop, so they are not blocked.
        if signal_direction.upper() == 'BUY':
            if events_data is not None:
                dividend_info = events_data.get('dividends', {}).get(symbol)
            else:
                dividend_info = None

            if dividend_info is None:
                dividend_info = fetch_dividend_calendar(symbol)

            days_until_ex_div = dividend_info.get('days_until_ex_div')
            dividend_amount = dividend_info.get('dividend_amount')

            if days_until_ex_div is not None and days_until_ex_div <= 1:
                div_str = f"${dividend_amount:.4f}" if dividend_amount else "an expected amount"
                reason = (
                    f"BLOCKED — Ex-dividend date for {symbol} is in {days_until_ex_div} day(s). "
                    f"Stock price will drop by approximately {div_str} on ex-div date. "
                    f"Opening a new LONG position now means buying at the pre-ex-div price "
                    f"and absorbing the price drop. Wait until after the ex-dividend date."
                )
                return (True, reason)

    except Exception as e:
        # NEVER block on an error — fail open (allow trades) to avoid false negatives.
        # A false negative (allowing a trade that should be blocked) is better than
        # a false positive (blocking a trade that should be allowed) when caused by
        # a data fetch error.
        print(f"[events] Error in should_block_trade for {symbol}: {e}")

    # All checks passed — trade is safe to execute
    return (False, 'OK')


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
    Each warning explains the IV crush mechanics and suggested actions.
    """
    if symbols is None:
        symbols = config.SYMBOLS

    warnings = []

    for symbol in symbols:
        earnings_info = fetch_earnings_calendar(symbol)
        days_until = earnings_info.get('days_until_earnings')
        earnings_risk = earnings_info.get('earnings_risk', 'NONE')

        if days_until is not None and days_until <= 7 and earnings_risk != 'NONE':
            warnings.append({
                'symbol': symbol,
                'earnings_date': earnings_info.get('next_earnings'),
                'days_until': days_until,
                'risk_level': earnings_risk,
                'warning': (
                    f"IV CRUSH RISK: {symbol} earnings in {days_until} day(s). "
                    f"Options buyers face IV collapse after announcement regardless of direction. "
                    f"Consider: (1) Close options before earnings, "
                    f"(2) Sell premium to exploit elevated IV, "
                    f"(3) Use stock instead of options for directional bets."
                ),
            })

    return warnings
