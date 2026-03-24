# =============================================================================
# data.py — All Data Fetching Lives Here
# =============================================================================
# WHY: Isolating data fetching into one module means:
#   1. You can swap out data sources without touching strategy logic
#   2. You can add caching, rate limiting, and error handling in one place
#   3. Every other module treats data as a clean interface, not a raw API call

import yfinance as yf
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import config


def fetch_price(symbol: str) -> float | None:
    """
    Fetch the latest price for a single symbol.

    WHY: We use yfinance's period='1d' with the most recent close because
    intraday 'price' from yfinance can be unreliable for some asset classes.
    For a 60-second polling system, last close is sufficient for signal generation.
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period='2d', interval='1m')
        if hist.empty:
            return None
        return float(hist['Close'].iloc[-1])
    except Exception as e:
        print(f"[data] Error fetching price for {symbol}: {e}")
        return None


def fetch_prices(symbols: list[str]) -> dict[str, float]:
    """
    Batch fetch latest prices for all symbols.

    WHY: Batch fetching is more efficient than individual calls.
    Returns a dict so callers can look up by symbol name cleanly.
    """
    prices = {}
    for symbol in symbols:
        price = fetch_price(symbol)
        if price is not None:
            prices[symbol] = price
    return prices


def fetch_history(symbol: str, period: str = '60d', interval: str = '1h') -> pd.DataFrame:
    """
    Fetch OHLCV history for a symbol — used for strategy warm-up and regime detection.

    WHY: Strategies need historical data to initialize indicators. For example,
    a 30-period MA needs 30 price points before it can generate a signal.
    'Warm-up' means feeding historical data to indicators before live trading begins.

    Returns a DataFrame with columns: Open, High, Low, Close, Volume
    """
    try:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        if hist.empty:
            print(f"[data] No history returned for {symbol}")
            return pd.DataFrame()
        return hist[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    except Exception as e:
        print(f"[data] Error fetching history for {symbol}: {e}")
        return pd.DataFrame()


def fetch_fear_greed_index() -> dict | None:
    """
    Fetch the CNN Fear & Greed Index from alternative.me (free, no API key).

    WHY: Fear & Greed is real alternative data used by professional traders.
    - Score below 20: Extreme Fear — historically a mean-reversion BUY signal.
      Prices have usually been beaten down beyond what fundamentals justify.
    - Score above 80: Extreme Greed — historically a fade/sell signal.
      Retail sentiment peaks before corrections.

    This is how hedge funds extract alpha without paying Bloomberg $24k/year.

    Returns dict with 'value' (0-100) and 'classification' (e.g., 'Fear')
    """
    try:
        url = "https://api.alternative.me/fng/"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        entry = data['data'][0]
        return {
            'value': int(entry['value']),
            'classification': entry['value_classification'],
            'timestamp': entry['timestamp']
        }
    except Exception as e:
        print(f"[data] Error fetching Fear & Greed Index: {e}")
        return None


def fetch_crypto_funding_rate(symbol: str = 'BTCUSDT') -> float | None:
    """
    Fetch the perpetual futures funding rate from Binance public API (free, no key).

    WHY: Funding rates are one of the most important signals in crypto markets.
    - Highly positive funding (>0.1%): Longs are paying shorts heavily.
      This signals overleveraged bullish positioning — a contrarian SHORT setup.
    - Highly negative funding (<-0.05%): Shorts are paying longs.
      Overleveraged short squeeze risk — a contrarian LONG setup.

    Funding is charged every 8 hours. Rates above 0.3% per 8h are extreme.

    Returns the current funding rate as a decimal (e.g., 0.0001 = 0.01%)
    """
    try:
        # Binance public endpoint — no authentication required
        url = f"https://fapi.binance.com/fapi/v1/fundingRate"
        params = {'symbol': symbol, 'limit': 1}
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data:
            return float(data[0]['fundingRate'])
        return None
    except Exception as e:
        # Binance may be geo-restricted; degrade gracefully
        print(f"[data] Error fetching funding rate for {symbol}: {e}")
        return None


def fetch_treasury_yield() -> float | None:
    """
    Fetch the 10-Year US Treasury yield from FRED (free, API key required).

    WHY: The 10Y Treasury yield is the most important number in global finance.
    - Rising yields → tighter financial conditions → headwind for equities/crypto
    - Falling yields → easier conditions → tailwind for risk assets
    - The yield curve (10Y - 2Y spread) predicts recessions with ~18mo lead time

    FRED series: DGS10 (10-Year Treasury Constant Maturity Rate)
    Register at fred.stlouisfed.org for a free API key.

    Returns the yield as a percentage (e.g., 4.25 means 4.25%)
    """
    if not config.FRED_API_KEY:
        # Graceful degradation — system works without FRED, just without macro context
        return None

    try:
        from fredapi import Fred
        fred = Fred(api_key=config.FRED_API_KEY)
        series = fred.get_series('DGS10', observation_start='2020-01-01')
        latest = series.dropna().iloc[-1]
        return float(latest)
    except Exception as e:
        print(f"[data] Error fetching Treasury yield from FRED: {e}")
        return None


def fetch_crypto_dominance() -> dict | None:
    """
    Fetch Bitcoin and Ethereum dominance from CoinGecko (free, no key required).

    WHY: BTC dominance tells you about risk appetite within crypto.
    - Rising BTC dominance: capital flowing into "safer" crypto, risk-off within crypto
    - Falling BTC dominance (alt season): risk-on, retail money flooding into alts

    Returns dict with 'btc_dominance' and 'eth_dominance' as percentages.
    """
    try:
        url = "https://api.coingecko.com/api/v3/global"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()['data']
        market_cap_pct = data.get('market_cap_percentage', {})
        return {
            'btc_dominance': market_cap_pct.get('btc', None),
            'eth_dominance': market_cap_pct.get('eth', None),
            'total_market_cap_usd': data.get('total_market_cap', {}).get('usd', None)
        }
    except Exception as e:
        print(f"[data] Error fetching crypto dominance: {e}")
        return None


def get_alt_data_summary() -> dict:
    """
    Aggregate all alternative data signals into one dict for the engine.

    WHY: The engine needs a single clean interface to alt data. This function
    handles all the API calls and returns a unified snapshot. If any source
    fails, its value is None and the engine handles the missing data gracefully.
    """
    return {
        'fear_greed': fetch_fear_greed_index(),
        'btc_funding_rate': fetch_crypto_funding_rate('BTCUSDT'),
        'eth_funding_rate': fetch_crypto_funding_rate('ETHUSDT'),
        'treasury_yield_10y': fetch_treasury_yield(),
        'crypto_dominance': fetch_crypto_dominance(),
        'timestamp': datetime.now().isoformat()
    }
