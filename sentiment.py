# =============================================================================
# sentiment.py — News Sentiment Analysis + SEC Filing NLP
# =============================================================================
#
# WHY SENTIMENT MATTERS IN FINANCE:
# -----------------------------------
# Price is driven by human decisions. Human decisions are driven by information
# AND emotion. Sentiment analysis tries to quantify the "emotion" component of
# the information flow that moves prices.
#
# Academic research (e.g., Tetlock 2007, "Giving Content to Investor Sentiment")
# shows that negative words in Wall Street Journal articles predict:
#   - NEXT DAY: Small negative returns
#   - Next 1-3 days: Continued pressure as other traders read the same news
#   - Week+: Price fully adjusts
#
# This 1-3 day lag exists because:
#   1. Not everyone reads every article immediately
#   2. Institutional investors take time to act on news
#   3. Market microstructure (position limits, fund redemptions) delays reaction
#
# The edge comes from being FAST — analyzing news before it's fully priced in.
#
# HOW WE SCORE SENTIMENT — VADER:
# --------------------------------
# VADER (Valence Aware Dictionary and sEntiment Reasoner) was developed at
# Georgia Tech specifically for SHORT, informal text like:
#   - Social media posts (Twitter/Reddit)
#   - News headlines
#   - Earnings call soundbites
#
# VADER understands financial-specific language better than generic NLP because:
#   1. It handles capitalization ("RECORD EARNINGS" vs "record earnings")
#   2. It handles punctuation emphasis ("Earnings BEAT!!!")
#   3. It has a finance-aware lexicon built in
#
# The COMPOUND score ranges from -1.0 (most negative) to +1.0 (most positive).
# Thresholds: compound > 0.05 = positive, < -0.05 = negative, in between = neutral.
#
# HOW TO USE THE OUTPUT:
# -----------------------
# avg_compound > +0.3: BULLISH — strong positive news flow → BUY signal consideration
# avg_compound < -0.3: BEARISH — strong negative news flow → SELL signal consideration
# avg_compound between -0.3 and +0.3: Neutral — no strong sentiment signal
#
# IMPORTANT CAVEAT:
# -----------------
# News sentiment is ONE signal — not a standalone strategy. Combine with:
#   - Price action (is the stock reacting to the news?)
#   - Volume (is there institutional confirmation?)
#   - Sector sentiment (is it specific to the stock or industry-wide noise?)

import nltk
import yfinance as yf
import config
from datetime import datetime

# Download VADER lexicon if not already present.
# VADER's lexicon is a dictionary of ~7,500 words with sentiment scores.
# quiet=True suppresses the "downloading..." output to keep logs clean.
# This only downloads once — NLTK caches it locally after the first download.
try:
    nltk.download('vader_lexicon', quiet=True)
    from nltk.sentiment.vader import SentimentIntensityAnalyzer

    # Initialize VADER once at module level — creating it repeatedly is wasteful
    # since it loads the entire lexicon from disk each time.
    _vader = SentimentIntensityAnalyzer()
    VADER_AVAILABLE = True
except Exception as e:
    VADER_AVAILABLE = False
    _vader = None
    print(f"[sentiment] VADER not available: {e}. Run: pip install nltk")


# ---------------------------------------------------------------------------
# Finance-specific vocabulary for boosted scoring
# ---------------------------------------------------------------------------
# WHY CUSTOM WORD LISTS?
# VADER's general lexicon doesn't give extra weight to finance-specific terms.
# "Restatement" is a huge red flag in finance but neutral in everyday language.
# "Exceeded" (earnings beat) is mildly positive generally but VERY positive in finance.
# These lists boost the signal strength for finance-specific events.

_NEGATIVE_FINANCE_WORDS = [
    'lawsuit', 'investigation', 'probe', 'subpoena',     # Legal risk
    'decline', 'miss', 'missed', 'shortfall', 'fell',    # Earnings misses
    'loss', 'losses', 'deficit', 'impairment',           # Financial deterioration
    'risk', 'uncertainty', 'headwind', 'headwinds',      # Forward guidance risk
    'restatement', 'restate', 'fraud', 'misconduct',     # Accounting / fraud
    'layoffs', 'restructuring', 'cut', 'cuts',           # Cost reduction
    'downgrade', 'lowered', 'reduced', 'below',          # Analyst actions
    'recall', 'default', 'bankruptcy', 'insolvency'      # Severe events
]

_POSITIVE_FINANCE_WORDS = [
    'record', 'records', 'all-time',                     # New highs
    'growth', 'grew', 'expanding', 'expansion',          # Positive trajectory
    'beat', 'beats', 'exceeded', 'surpassed',            # Earnings beats
    'raised', 'raises', 'lifted', 'increased',           # Guidance raises
    'guidance', 'outlook', 'forecast',                   # Forward-looking positivity
    'momentum', 'acceleration', 'accelerating',          # Business velocity
    'outperform', 'upgrade', 'overweight', 'buy',        # Analyst upgrades
    'strong', 'strength', 'robust', 'resilient',         # Qualitative positivity
    'dividend', 'buyback', 'repurchase',                 # Capital return to shareholders
    'partnership', 'contract', 'wins', 'award'           # Business wins
]


# =============================================================================
# FUNCTION 1: Fetch News Headlines via yfinance
# =============================================================================

def fetch_news_headlines(symbol: str, max_articles: int = 20) -> list:
    """
    Fetch recent news headlines for a symbol using yfinance's built-in news feed.

    WHY YFINANCE FOR NEWS?
    ----------------------
    yfinance aggregates financial news from Yahoo Finance's news feed.
    It's completely free and requires no API key. The tradeoff:
    - Coverage: Good for major US stocks and crypto
    - Latency: Not real-time — typically 15-30 minutes delayed
    - Quality: Yahoo Finance aggregates from Reuters, AP, Benzinga, etc.
    - Limitation: Small/illiquid stocks may have few or no articles

    For a production system you'd use:
    - Benzinga Pro API (fast, structured, $$$)
    - NewsAPI.org (free tier available)
    - SEC EDGAR for official filings (free but requires parsing)
    - Alpha Vantage News Sentiment API (free tier)

    Returns:
        list of dicts: [{title, publisher, timestamp, url}, ...]
        Returns empty list on error (never crashes).
    """
    try:
        ticker = yf.Ticker(symbol)

        # ticker.news returns a list of article dicts from Yahoo Finance
        # Each dict has keys: title, publisher, link, providerPublishTime, type, thumbnail
        raw_news = ticker.news

        if not raw_news:
            return []

        articles = []
        for item in raw_news[:max_articles]:   # Limit to max_articles
            try:
                # Convert Unix timestamp to readable string
                publish_time = item.get('providerPublishTime', 0)
                if publish_time:
                    timestamp_str = datetime.utcfromtimestamp(publish_time).strftime('%Y-%m-%d %H:%M')
                else:
                    timestamp_str = 'Unknown'

                # Extract the article content from nested structure
                # yfinance returns content in 'content' dict with 'title' key
                content = item.get('content', {})
                title = content.get('title', '') if isinstance(content, dict) else item.get('title', '')

                # Try to get publisher info
                provider = content.get('provider', {}) if isinstance(content, dict) else {}
                publisher = provider.get('displayName', 'Unknown') if isinstance(provider, dict) else 'Unknown'

                if not title:
                    continue

                articles.append({
                    'title': title,
                    'publisher': publisher,
                    'timestamp': timestamp_str,
                    'url': content.get('canonicalUrl', {}).get('url', '') if isinstance(content, dict) else '',
                })
            except Exception:
                # Skip malformed articles rather than crashing
                continue

        return articles

    except Exception as e:
        # yfinance can fail with network errors, rate limiting, or for obscure symbols
        print(f"[sentiment] Failed to fetch news for {symbol}: {e}")
        return []


# =============================================================================
# FUNCTION 2: Score Sentiment of a Text String
# =============================================================================

def score_sentiment(text: str) -> dict:
    """
    Score the sentiment of a text string using VADER.

    VADER OUTPUT EXPLANATION:
    -------------------------
    VADER returns four scores that always sum to 1.0 (except compound):

    'positive' (0 to 1): Fraction of text with positive valence
    'negative' (0 to 1): Fraction of text with negative valence
    'neutral'  (0 to 1): Fraction of text with neutral valence
    'compound' (-1 to 1): Normalized, weighted overall score
        - The compound score is the most useful single number
        - It aggregates all sentiment signals into one value
        - > +0.05 = positive sentiment
        - < -0.05 = negative sentiment
        - Between -0.05 and +0.05 = neutral

    WHY VADER WORKS WELL ON HEADLINES:
    ------------------------------------
    Headlines are short, punchy, and use emphasis for a reason.
    "AAPL CRUSHES EARNINGS" vs "Apple earnings miss estimates"
    VADER picks up on capitalization, exclamation marks, and sentiment words
    in context. Generic BERT/GPT sentiment models often underperform VADER
    on headline-length text because they were trained on longer documents.

    Returns:
        dict: {compound, positive, negative, neutral}
        Returns neutral (all zeros) if VADER is unavailable or text is empty.
    """
    if not VADER_AVAILABLE or not text or not text.strip():
        return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}

    try:
        scores = _vader.polarity_scores(text)
        return {
            'compound': round(scores['compound'], 4),
            'positive': round(scores['pos'], 4),
            'negative': round(scores['neg'], 4),
            'neutral':  round(scores['neu'], 4),
        }
    except Exception:
        return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}


# =============================================================================
# FUNCTION 3: Analyze All Headlines for a Single Symbol
# =============================================================================

def analyze_symbol_sentiment(symbol: str) -> dict:
    """
    Fetch and score all recent headlines for a symbol, then aggregate the results.

    AGGREGATION LOGIC:
    ------------------
    We average the compound scores across all headlines. This gives equal weight
    to each article regardless of length. An improvement would be to weight
    by recency (newer articles weighted higher) or by source credibility.

    SIGNAL THRESHOLDS (from config.py):
    -------------------------------------
    avg_compound > SENTIMENT_BULLISH_THRESHOLD (default 0.3):
        Strong consensus positive news → BUY signal consideration
    avg_compound < SENTIMENT_BEARISH_THRESHOLD (default -0.3):
        Strong consensus negative news → SELL signal consideration

    WHY 0.3 AND NOT 0.05?
    ----------------------
    The 0.05 threshold is for individual words. For an average across many
    headlines to reach 0.3, you need consistently strong positive/negative
    language across multiple articles — that's a real signal, not noise.
    A single "decent" headline might push the average to 0.1. You need
    a cluster of strong headlines to hit 0.3.

    Returns:
        dict with sentiment statistics and BUY/SELL/None signal
    """
    articles = fetch_news_headlines(symbol)

    if not articles:
        return {
            'symbol': symbol,
            'avg_compound': 0.0,
            'sentiment_label': 'NEUTRAL',
            'article_count': 0,
            'bullish_count': 0,
            'bearish_count': 0,
            'headlines': [],
            'signal': None,
        }

    # Score each headline individually
    scored_headlines = []
    compound_scores = []

    for article in articles:
        title = article.get('title', '')
        if not title:
            continue

        scores = score_sentiment(title)
        compound = scores['compound']
        compound_scores.append(compound)

        # Label each individual headline
        if compound > 0.05:
            label = 'BULLISH'
        elif compound < -0.05:
            label = 'BEARISH'
        else:
            label = 'NEUTRAL'

        scored_headlines.append({
            'title': title,
            'publisher': article.get('publisher', 'Unknown'),
            'timestamp': article.get('timestamp', ''),
            'compound': compound,
            'label': label,
        })

    if not compound_scores:
        avg_compound = 0.0
    else:
        avg_compound = round(sum(compound_scores) / len(compound_scores), 4)

    # Count bullish/bearish articles
    bullish_count = sum(1 for s in compound_scores if s > 0.05)
    bearish_count = sum(1 for s in compound_scores if s < -0.05)

    # Overall sentiment label
    if avg_compound > 0.05:
        sentiment_label = 'BULLISH'
    elif avg_compound < -0.05:
        sentiment_label = 'BEARISH'
    else:
        sentiment_label = 'NEUTRAL'

    # Strong signal thresholds — pulled from config
    bullish_threshold = config.SENTIMENT_BULLISH_THRESHOLD   # default 0.3
    bearish_threshold = config.SENTIMENT_BEARISH_THRESHOLD   # default -0.3

    if avg_compound > bullish_threshold:
        signal = 'BUY'
    elif avg_compound < bearish_threshold:
        signal = 'SELL'
    else:
        signal = None

    return {
        'symbol': symbol,
        'avg_compound': avg_compound,
        'sentiment_label': sentiment_label,
        'article_count': len(scored_headlines),
        'bullish_count': bullish_count,
        'bearish_count': bearish_count,
        'headlines': scored_headlines,
        'signal': signal,
    }


# =============================================================================
# FUNCTION 4: Analyze Multiple Symbols
# =============================================================================

def analyze_all_symbols_sentiment(symbols: list) -> dict:
    """
    Run sentiment analysis for every symbol in the provided list.

    WHY PER-SYMBOL TRY/EXCEPT?
    ---------------------------
    Some symbols have no news (small caps, obscure ETFs, certain forex pairs).
    Without individual error handling, one missing symbol would abort the entire
    analysis run. By catching per-symbol, we always get results for symbols that
    work, and record the error for those that don't.

    Returns:
        dict keyed by symbol: {symbol: sentiment_result_dict, ...}
    """
    results = {}

    for symbol in symbols:
        try:
            results[symbol] = analyze_symbol_sentiment(symbol)
        except Exception as e:
            # Return a neutral placeholder rather than crashing
            results[symbol] = {
                'symbol': symbol,
                'avg_compound': 0.0,
                'sentiment_label': 'NEUTRAL',
                'article_count': 0,
                'bullish_count': 0,
                'bearish_count': 0,
                'headlines': [],
                'signal': None,
                'error': str(e),
            }

    return results


# =============================================================================
# FUNCTION 5: Finance-Boosted Sentiment Scoring
# =============================================================================

def score_filing_text(text: str) -> dict:
    """
    Score sentiment of SEC filings or earnings call text with finance-specific boosting.

    WHY DIFFERENT FROM score_sentiment()?
    ----------------------------------------
    Standard VADER scores casual/news text well, but SEC filings use corporate
    boilerplate language that VADER may underweight. Words like "restatement"
    or "impairment charge" are highly negative financial events, but VADER
    may see them as mildly negative at best.

    HOW BOOSTING WORKS:
    -------------------
    1. Run standard VADER scoring
    2. Scan for finance-specific negative/positive words in the text
    3. Adjust the compound score: each negative hit reduces it by 0.05,
       each positive hit increases it by 0.05
    4. Clamp the final score to [-1.0, +1.0]

    This makes the scorer more sensitive to finance-specific language patterns
    that VADER's general lexicon underweights.

    EXAMPLE USE CASES:
    ------------------
    - Scoring 10-K Risk Factors section (should be bearish if lots of risk language)
    - Scoring earnings call transcript for tone vs. content
    - Scoring press releases for actual substance vs. PR spin

    Returns:
        dict: standard sentiment scores plus 'finance_words_found' list
    """
    base_scores = score_sentiment(text)

    if not text or not VADER_AVAILABLE:
        return {**base_scores, 'finance_words_found': []}

    try:
        text_lower = text.lower()

        # Find which finance-specific words appear in the text
        found_words = []
        adjustment = 0.0

        for word in _NEGATIVE_FINANCE_WORDS:
            if word in text_lower:
                found_words.append(f"-{word}")     # Prefix with '-' to indicate negative
                adjustment -= 0.05                 # Each negative word pulls compound down

        for word in _POSITIVE_FINANCE_WORDS:
            if word in text_lower:
                found_words.append(f"+{word}")     # Prefix with '+' to indicate positive
                adjustment += 0.05                 # Each positive word pushes compound up

        # Apply adjustment to the base compound score, clamped to [-1.0, +1.0]
        adjusted_compound = base_scores['compound'] + adjustment
        adjusted_compound = max(-1.0, min(1.0, adjusted_compound))

        return {
            'compound': round(adjusted_compound, 4),
            'positive': base_scores['positive'],
            'negative': base_scores['negative'],
            'neutral':  base_scores['neutral'],
            'finance_words_found': found_words,
            'raw_compound': base_scores['compound'],   # Keep original for comparison
            'adjustment': round(adjustment, 4),
        }

    except Exception as e:
        return {**base_scores, 'finance_words_found': [], 'error': str(e)}


# =============================================================================
# FUNCTION 6: Aggregate Sentiment Summary Across All Symbols
# =============================================================================

def get_sentiment_summary(symbols: list) -> dict:
    """
    Build an aggregate sentiment view across all symbols.

    WHY AGGREGATE?
    ---------------
    Individual stock sentiment is noisy. Aggregating across your full symbol
    universe gives you a "market sentiment barometer." If 7 out of 7 symbols
    are showing negative sentiment, that's a meaningful macro signal.
    If only 1 out of 7 is negative, it may be idiosyncratic company news.

    HOW TO INTERPRET overall_market_sentiment:
    -------------------------------------------
    > +0.15: Broadly positive news flow across your universe → risk-on
    < -0.15: Broadly negative news flow → risk-off
    Near 0:  Mixed or neutral news environment

    Returns:
        dict: {
            overall_market_sentiment: float,    # Average compound across all symbols
            most_bullish: str,                  # Symbol with highest avg_compound
            most_bearish: str,                  # Symbol with lowest avg_compound
            by_symbol: dict,                    # Full per-symbol results
        }
    """
    by_symbol = analyze_all_symbols_sentiment(symbols)

    compound_scores = {
        sym: data['avg_compound']
        for sym, data in by_symbol.items()
        if 'error' not in data
    }

    if not compound_scores:
        return {
            'overall_market_sentiment': 0.0,
            'most_bullish': None,
            'most_bearish': None,
            'by_symbol': by_symbol,
        }

    overall = round(sum(compound_scores.values()) / len(compound_scores), 4)

    # Find the most bullish and most bearish symbols
    most_bullish = max(compound_scores, key=compound_scores.get)
    most_bearish = min(compound_scores, key=compound_scores.get)

    # Overall label
    if overall > 0.05:
        overall_label = 'BULLISH'
    elif overall < -0.05:
        overall_label = 'BEARISH'
    else:
        overall_label = 'NEUTRAL'

    return {
        'overall_market_sentiment': overall,
        'overall_label': overall_label,
        'most_bullish': most_bullish,
        'most_bullish_score': compound_scores[most_bullish],
        'most_bearish': most_bearish,
        'most_bearish_score': compound_scores[most_bearish],
        'by_symbol': by_symbol,
    }


# =============================================================================
# QUICK TEST — Run this file directly to see output
# =============================================================================
if __name__ == '__main__':
    print("=" * 60)
    print("NEWS SENTIMENT ANALYSIS")
    print("=" * 60)

    test_symbols = ['AAPL', 'MSFT', 'BTC-USD']

    for sym in test_symbols:
        result = analyze_symbol_sentiment(sym)
        print(f"\n{sym}: {result['sentiment_label']} ({result['avg_compound']:+.3f}) "
              f"| {result['article_count']} articles "
              f"| Signal: {result['signal']}")
        for h in result['headlines'][:3]:
            print(f"   [{h['label']:7s}] {h['compound']:+.3f} | {h['title'][:80]}")

    print("\n--- Finance Filing Scorer Test ---")
    sample_filing = (
        "The company faces significant risk and uncertainty related to the ongoing "
        "investigation. Revenue declined and the board approved a restatement of "
        "earnings. Despite these headwinds, record customer growth and strong guidance "
        "for next quarter demonstrate resilient momentum."
    )
    filing_scores = score_filing_text(sample_filing)
    print(f"Raw compound:      {filing_scores['raw_compound']}")
    print(f"Adjusted compound: {filing_scores['compound']}")
    print(f"Finance words:     {filing_scores['finance_words_found']}")
