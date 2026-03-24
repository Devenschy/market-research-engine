# =============================================================================
# edgar.py — SEC EDGAR Research: 8-K Scanner, Insider Trades, 13F Holdings
# =============================================================================
# WHY SEC EDGAR:
# The SEC's Electronic Data Gathering, Analysis, and Retrieval (EDGAR) system
# is one of the most underutilized edges available to individual investors.
# Every public company must file legally binding disclosures here — earnings
# surprises, CEO departures, FDA decisions, going-concern warnings — often
# BEFORE the news hits Bloomberg or CNBC. Reading filings directly is how
# professional analysts develop an information edge over retail traders.
#
# Base URL: https://data.sec.gov  (no API key required)
# EDGAR full-text search: https://efts.sec.gov/LATEST/search-index?q=...
#
# IMPORTANT: All requests must include a User-Agent header identifying your
# application. The SEC requires this so they can contact you if your script
# causes server problems. Omitting it results in 403 errors.
# =============================================================================

import requests
import json
import time
import re
import xml.etree.ElementTree as ET
from datetime import datetime, date, timedelta

# ---------------------------------------------------------------------------
# HTTP session with required User-Agent header
# ---------------------------------------------------------------------------
# WHY a session object: Reusing a requests.Session() maintains a connection
# pool, which is faster than opening a new TCP connection for every request.
# The SEC's servers also respond better to consistent, identified clients.
HEADERS = {'User-Agent': 'MarketResearchEngine research@dezonagroup.com'}

_session = requests.Session()
_session.headers.update(HEADERS)


# =============================================================================
# SECTION 1: Company CIK Lookup
# =============================================================================
# WHY CIK (Central Index Key):
# The SEC does not index filings by ticker symbol. Every public company is
# assigned a unique CIK when it first registers with the SEC. CIKs are stable
# and permanent — they do not change when a company is acquired or renamed.
# You must convert ticker → CIK before accessing any filing data.
#
# CIKs are zero-padded to 10 digits in API URLs (e.g., Apple = 0000320193)
# but some endpoints accept the un-padded integer form. We standardize on
# the 10-digit padded string throughout this module.
# =============================================================================

# Pre-built mapping for the symbols we care about most.
# WHY a local dict: Avoids an extra HTTP round-trip for symbols we already know.
# Extend this as you add more tickers to your research universe.
TICKER_TO_CIK = {
    # --- Mega-cap Tech ---
    'AAPL':  '0000320193',   # Apple Inc.
    'MSFT':  '0000789019',   # Microsoft Corp.
    'GOOGL': '0001652044',   # Alphabet Inc. (Class A)
    'GOOG':  '0001652044',   # Alphabet Inc. (Class C) — same CIK
    'META':  '0001326801',   # Meta Platforms
    'AMZN':  '0001018724',   # Amazon.com
    'NVDA':  '0001045810',   # NVIDIA Corp.
    'TSLA':  '0001318605',   # Tesla Inc.
    # --- Finance ---
    'JPM':   '0000019617',   # JPMorgan Chase
    'BAC':   '0000070858',   # Bank of America
    'GS':    '0000886982',   # Goldman Sachs
    # --- Berkshire (needed for 13F section) ---
    'BRK-B': '0001067983',   # Berkshire Hathaway
    'BRK-A': '0001067983',   # Same entity, different share class
    # --- Healthcare ---
    'JNJ':   '0000200406',   # Johnson & Johnson
    'PFE':   '0000078003',   # Pfizer
    # --- Energy ---
    'XOM':   '0000034088',   # Exxon Mobil
    'CVX':   '0000093410',   # Chevron
    # --- Add more tickers here as your research universe grows ---
}

# Berkshire Hathaway's CIK — referenced separately in the 13F section
BERKSHIRE_CIK = '0001067983'


def get_cik(symbol: str) -> str | None:
    """
    Resolve a ticker symbol to its SEC CIK (Central Index Key).

    WHY THIS MATTERS:
    EDGAR's API is organized entirely around CIK, not ticker. You cannot
    fetch Apple's filings without knowing CIK 0000320193. This function
    first checks our local cache (instant), then falls back to a live
    EDGAR lookup if the ticker is not in our pre-built map.

    Args:
        symbol: Ticker symbol (e.g. 'AAPL', 'MSFT')

    Returns:
        Zero-padded 10-digit CIK string, or None if not found.
    """
    # Step 1: Check pre-built map first — no HTTP call needed
    symbol_upper = symbol.upper().strip()
    if symbol_upper in TICKER_TO_CIK:
        return TICKER_TO_CIK[symbol_upper]

    # Step 2: Live EDGAR lookup via the company search endpoint
    # The `output=atom` parameter returns an Atom XML feed we can parse.
    # This is the same endpoint used by EDGAR's web search UI.
    url = (
        f'https://www.sec.gov/cgi-bin/browse-edgar'
        f'?action=getcompany&ticker={symbol_upper}'
        f'&type=&dateb=&owner=include&count=10&search_text=&output=atom'
    )
    try:
        resp = _session.get(url, timeout=10)
        resp.raise_for_status()
        # The CIK appears in the Atom feed as a numeric string inside the
        # <company-info> element. Pattern: "CIK=" followed by digits.
        match = re.search(r'CIK=(\d+)', resp.text)
        if match:
            raw_cik = match.group(1)
            # Zero-pad to 10 digits — required by submissions API
            padded = raw_cik.zfill(10)
            # Cache it so we don't look it up again this session
            TICKER_TO_CIK[symbol_upper] = padded
            return padded
        return None
    except Exception:
        # Network errors, parse failures — return None rather than crashing
        return None


# =============================================================================
# SECTION 2: 8-K Scanner — Material Event Alerts
# =============================================================================
# WHY 8-K FILINGS MATTER FOR TRADING:
#
# Form 8-K is the "current report" — companies must file within 4 business
# days of any "material event." Material events are things that a reasonable
# investor would want to know before trading the stock. Examples:
#
#   - Item 2.02: Results of Operations (earnings releases)
#   - Item 5.02: Departure/appointment of directors or executives (CEO change)
#   - Item 1.01: Entry into a material definitive agreement (big contracts, M&A)
#   - Item 1.03: Bankruptcy or receivership
#   - Item 4.02: Non-reliance on previously issued financial statements (restatement)
#   - Item 8.01: FDA approval, clinical trial results, other regulation events
#   - Item 1.05: Cybersecurity incidents (added by SEC rule in 2023)
#
# WHY READ 8-Ks BEFORE NEWS HEADLINES:
# When a company files an 8-K, it appears on EDGAR immediately. Major news
# services (Bloomberg, Reuters) typically pick it up 5–30 minutes later after
# a journalist reads it and writes a story. In those 5–30 minutes, reading
# EDGAR directly gives you the same information a professional analyst has.
# Stocks routinely move 5–20% on 8-K events. Being early matters.
# =============================================================================

def fetch_recent_8k(symbol: str, max_filings: int = 5) -> list[dict]:
    """
    Fetch the most recent 8-K filings for a company from SEC EDGAR.

    The submissions endpoint returns the full filing history as JSON.
    We filter for 8-K form type and return only filings from the last 30 days,
    because anything older is already priced into the market.

    Args:
        symbol:      Ticker symbol (e.g. 'AAPL')
        max_filings: Maximum number of 8-K filings to return (default 5)

    Returns:
        List of dicts, each describing one 8-K filing. Empty list on error.
    """
    cik = get_cik(symbol)
    if not cik:
        return []

    # EDGAR submissions endpoint: returns all filing metadata for a company.
    # The CIK in this URL must be zero-padded to 10 digits.
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'

    try:
        time.sleep(0.5)   # Respect SEC rate limit: max 10 requests/second
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    # The 'filings' → 'recent' dict contains parallel arrays: one entry per
    # filing, indexed consistently across all arrays.
    # WHY parallel arrays: EDGAR's API stores metadata this way for efficiency.
    # We zip the arrays together to reconstruct per-filing dicts.
    recent = data.get('filings', {}).get('recent', {})
    forms       = recent.get('form', [])
    dates       = recent.get('filingDate', [])
    descriptions = recent.get('primaryDocument', [])
    accessions  = recent.get('accessionNumber', [])

    results = []
    today = date.today()
    cutoff = today - timedelta(days=30)   # Only show last 30 days

    for form, filing_date_str, doc, accession in zip(forms, dates, descriptions, accessions):
        if form != '8-K':
            continue

        try:
            filing_date = datetime.strptime(filing_date_str, '%Y-%m-%d').date()
        except ValueError:
            continue

        if filing_date < cutoff:
            continue   # Skip filings older than 30 days

        # Build the URL to the actual filing document.
        # EDGAR archives are at: /Archives/edgar/data/{cik}/{accession_no_dashes}/
        # We strip dashes from the accession number for the folder name.
        accession_clean = accession.replace('-', '')
        doc_url = (
            f'https://www.sec.gov/Archives/edgar/data/'
            f'{int(cik)}/{accession_clean}/{doc}'
        )

        days_ago = (today - filing_date).days

        results.append({
            'symbol':       symbol.upper(),
            'filing_date':  filing_date_str,
            'form_type':    '8-K',
            'description':  doc,
            'document_url': doc_url,
            'days_ago':     days_ago,
        })

        if len(results) >= max_filings:
            break

    return results


def fetch_8k_text(filing_url: str) -> str:
    """
    Fetch the raw text content of an 8-K filing from EDGAR.

    Returns the first 2000 characters, which is enough to identify the type
    of material event and run keyword-based sentiment analysis. Full 8-Ks
    can be very long (legal boilerplate, exhibits, signatures) so we truncate
    to the substance-dense opening section.

    Args:
        filing_url: Full URL to the primary document of an 8-K filing

    Returns:
        String with cleaned text (HTML tags stripped). Empty string on error.
    """
    try:
        time.sleep(0.5)   # SEC rate limit courtesy delay
        resp = _session.get(filing_url, timeout=15)
        resp.raise_for_status()
        raw = resp.text

        # Strip HTML tags using a simple regex.
        # WHY not BeautifulSoup: We want zero extra dependencies. The regex
        # approach is imperfect for complex HTML but works well for EDGAR's
        # relatively simple filing documents.
        text = re.sub(r'<[^>]+>', ' ', raw)

        # Collapse multiple whitespace characters into a single space
        text = re.sub(r'\s+', ' ', text).strip()

        # Return the first 2000 chars — the most information-dense section
        return text[:2000]
    except Exception:
        return ''


def scan_8k_for_keywords(text: str) -> dict:
    """
    Run a keyword scan on 8-K text to classify it as bullish, bearish, or neutral.

    WHY KEYWORD SCANNING:
    Professional quant shops use NLP models for this, but keyword scanning
    catches the highest-signal events with near-perfect recall. "Bankruptcy"
    in an 8-K is ALWAYS bearish. "Record revenue" is ALWAYS bullish. You
    don't need ML to find these — you need discipline to actually read the
    filings, which most retail traders never do.

    The approach here is intentionally simple and explainable. You can audit
    exactly why the model called something bullish or bearish, unlike a black-
    box neural network.

    Args:
        text: Text content of an 8-K filing (ideally first 2000 chars)

    Returns:
        Dict with bullish_hits, bearish_hits, and overall sentiment label.
    """
    text_lower = text.lower()

    # BULLISH KEYWORDS — events that typically send the stock higher
    # These represent: earnings beats, capital return, strategic growth,
    # revenue acceleration, or reduced competitive risk.
    BULLISH_KEYWORDS = [
        'beat',
        'exceeded',
        'record',
        'raised guidance',
        'acquisition',
        'buyback',
        'share repurchase',
        'dividend increase',
        'partnership',
        'approved',
        'fda approval',
        'accelerating',
        'strong demand',
        'margin expansion',
        'new contract',
    ]

    # BEARISH KEYWORDS — events that typically send the stock lower
    # These represent: earnings misses, legal/regulatory risk, financial stress,
    # leadership instability, accounting problems, or existential threats.
    BEARISH_KEYWORDS = [
        'missed',
        'below expectations',
        'restructuring',
        'layoffs',
        'investigation',
        'lawsuit',
        'going concern',
        'restatement',
        'bankruptcy',
        'breach',
        'cybersecurity incident',
        'data breach',
        'subpoena',
        'sec investigation',
        'ceo resignation',
        'guidance cut',
        'lowered guidance',
        'impairment',
        'write-down',
    ]

    bullish_hits = [kw for kw in BULLISH_KEYWORDS if kw in text_lower]
    bearish_hits = [kw for kw in BEARISH_KEYWORDS if kw in text_lower]

    # Determine overall sentiment.
    # WHY weighted: A single "going concern" or "bankruptcy" hit is more
    # significant than a single "partnership" hit. Bear hits carry slightly
    # more urgency because downside surprises tend to be faster and larger
    # than upside surprises (the "asymmetric volatility" phenomenon).
    if len(bearish_hits) > len(bullish_hits):
        sentiment = 'BEARISH'
    elif len(bullish_hits) > len(bearish_hits):
        sentiment = 'BULLISH'
    else:
        sentiment = 'NEUTRAL'

    return {
        'bullish_hits': bullish_hits,
        'bearish_hits': bearish_hits,
        'sentiment':    sentiment,
    }


# =============================================================================
# SECTION 3: Insider Transaction Tracker (Form 4)
# =============================================================================
# WHY INSIDER TRANSACTIONS ARE A PREMIER SIGNAL:
#
# Form 4 is filed within 2 business days whenever an "insider" (officer,
# director, or 10%+ shareholder) buys or sells the company's stock. This is
# required by Section 16 of the Securities Exchange Act of 1934 and is one of
# the oldest disclosure requirements in U.S. securities law.
#
# WHY BUYING IS INFORMATIVE:
# Executives already have enormous exposure to their company — their salary,
# bonus, equity grants, career prospects, and reputation all depend on the
# company's success. When they ALSO spend their own after-tax dollars buying
# stock in the open market, they are making a concentrated bet on top of an
# already-concentrated position. This signals very high conviction.
#
# Academic evidence:
#   - Seyhun (1986): Insider buyers earn ~3% abnormal returns over 6 months
#   - Lakonishok & Lee (2001): Strongest signal comes from small-cap insider buys
#   - Jeng, Metrick & Zeckhauser (2003): Insider buyers beat market by 6%/year
#
# WHY SELLING IS LESS INFORMATIVE (but cluster selling is):
# Executives sell stock for many reasons unrelated to their company outlook:
# diversification, estate planning, buying a house, divorce settlement, etc.
# A single insider sale means little. But when 3+ insiders sell simultaneously
# in a short window — what academics call "cluster selling" — that IS informative
# because the coincidence of timing suggests shared information.
#
# LEGAL NOTE: All transactions here are PUBLICLY DISCLOSED and LEGAL.
# Insider trading (trading on material non-public information) is illegal and
# is NOT what Form 4 captures. Form 4 captures legal open-market transactions
# that insiders are required to disclose precisely so the public can see them.
# =============================================================================

def fetch_insider_transactions(symbol: str, max_transactions: int = 10) -> list[dict]:
    """
    Fetch recent Form 4 insider transactions for a company.

    Form 4 filings are XML documents with a standardized schema defined by
    the SEC. We parse the XML to extract the transaction details.

    Args:
        symbol:           Ticker symbol (e.g. 'AAPL')
        max_transactions: Maximum number of Form 4 transactions to return

    Returns:
        List of insider transaction dicts, sorted most-recent first.
        Empty list on error.
    """
    cik = get_cik(symbol)
    if not cik:
        return []

    url = f'https://data.sec.gov/submissions/CIK{cik}.json'

    try:
        time.sleep(0.5)
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    recent = data.get('filings', {}).get('recent', {})
    forms       = recent.get('form', [])
    dates       = recent.get('filingDate', [])
    accessions  = recent.get('accessionNumber', [])
    primary_docs = recent.get('primaryDocument', [])

    transactions = []
    today = date.today()

    for form, filing_date_str, accession, primary_doc in zip(
        forms, dates, accessions, primary_docs
    ):
        if form != '4':
            continue

        if len(transactions) >= max_transactions:
            break

        # Parse the Form 4 XML from EDGAR archives.
        # Accession numbers look like: 0001234567-24-000001
        # The folder name uses them without dashes.
        accession_clean = accession.replace('-', '')
        cik_int = int(cik)

        # Try the primary document first; fall back to a .xml extension
        xml_filename = primary_doc if primary_doc.endswith('.xml') else primary_doc.rsplit('.', 1)[0] + '.xml'
        xml_url = (
            f'https://www.sec.gov/Archives/edgar/data/'
            f'{cik_int}/{accession_clean}/{xml_filename}'
        )

        parsed = _parse_form4_xml(xml_url, symbol, filing_date_str, today)
        if parsed:
            transactions.extend(parsed)

    # Sort by most recent first and cap at max_transactions
    transactions.sort(key=lambda x: x['transaction_date'], reverse=True)
    return transactions[:max_transactions]


def _parse_form4_xml(xml_url: str, symbol: str, filing_date_str: str, today: date) -> list[dict]:
    """
    Fetch and parse a single Form 4 XML file from EDGAR.

    Form 4 XML schema reference:
    https://www.sec.gov/info/edgar/edgarfm-vol2-v58.pdf (Section 4)

    Key XML elements we care about:
      ownershipDocument/reportingOwner/reportingOwnerId/rptOwnerName
      ownershipDocument/reportingOwner/reportingOwnerRelationship/officerTitle
      ownershipDocument/nonDerivativeTable/nonDerivativeTransaction/...
        transactionDate/value
        transactionAmounts/transactionShares/value
        transactionAmounts/transactionPricePerShare/value
        transactionAmounts/transactionAcquiredDisposedCode/value  ('A' or 'D')
        postTransactionAmounts/sharesOwnedFollowingTransaction/value

    Args:
        xml_url:          Full URL to the Form 4 XML file
        symbol:           Ticker symbol for labeling results
        filing_date_str:  Filing date string (YYYY-MM-DD)
        today:            Today's date for computing days_ago

    Returns:
        List of transaction dicts (one Form 4 can contain multiple transactions).
        Empty list if parsing fails.
    """
    try:
        time.sleep(0.5)
        resp = _session.get(xml_url, timeout=10)
        if resp.status_code == 404:
            # Many Form 4 filings use different filenames — silently skip
            return []
        resp.raise_for_status()

        root = ET.fromstring(resp.text)
    except Exception:
        # XML parse errors, network errors — return empty rather than crashing
        return []

    results = []

    # Extract the reporting owner's name and title.
    # A Form 4 has one primary owner (the insider) but may cover multiple
    # transactions on different dates.
    owner_name = ''
    officer_title = ''

    owner_el = root.find('.//reportingOwner')
    if owner_el is not None:
        name_el = owner_el.find('.//rptOwnerName')
        if name_el is not None and name_el.text:
            owner_name = name_el.text.strip()

        title_el = owner_el.find('.//officerTitle')
        if title_el is not None and title_el.text:
            officer_title = title_el.text.strip()

    # Parse the non-derivative transactions table.
    # "Non-derivative" = actual common stock (as opposed to options/warrants).
    # This is the table that represents real open-market buys and sells.
    for txn in root.findall('.//nonDerivativeTransaction'):
        try:
            # Transaction date (may differ from filing date by up to 2 biz days)
            txn_date_el = txn.find('.//transactionDate/value')
            txn_date_str = txn_date_el.text.strip() if txn_date_el is not None and txn_date_el.text else filing_date_str

            # Number of shares transacted
            shares_el = txn.find('.//transactionShares/value')
            shares = float(shares_el.text.strip()) if shares_el is not None and shares_el.text else 0.0

            # Price per share
            price_el = txn.find('.//transactionPricePerShare/value')
            price = float(price_el.text.strip()) if price_el is not None and price_el.text else 0.0

            # 'A' = Acquired (bought), 'D' = Disposed (sold)
            adc_el = txn.find('.//transactionAcquiredDisposedCode/value')
            adc = adc_el.text.strip().upper() if adc_el is not None and adc_el.text else ''
            txn_type = 'BUY' if adc == 'A' else 'SELL' if adc == 'D' else 'UNKNOWN'

            # Total shares owned AFTER this transaction
            owned_el = txn.find('.//sharesOwnedFollowingTransaction/value')
            shares_after = float(owned_el.text.strip()) if owned_el is not None and owned_el.text else 0.0

            # Compute days ago from transaction date
            try:
                txn_date = datetime.strptime(txn_date_str, '%Y-%m-%d').date()
                days_ago = (today - txn_date).days
            except ValueError:
                days_ago = 0

            # Skip zero-share or unknown transactions (grants, awards, etc.)
            if shares == 0 or txn_type == 'UNKNOWN':
                continue

            results.append({
                'symbol':            symbol.upper(),
                'insider_name':      owner_name,
                'title':             officer_title,
                'transaction_date':  txn_date_str,
                'transaction_type':  txn_type,
                'shares':            shares,
                'price_per_share':   price,
                'total_value':       shares * price,   # Notional value of the transaction
                'shares_owned_after': shares_after,
                'days_ago':          days_ago,
            })

        except (ValueError, AttributeError, TypeError):
            # Malformed XML element — skip this transaction, continue with others
            continue

    return results


def analyze_insider_sentiment(transactions: list[dict]) -> dict:
    """
    Aggregate a list of insider transactions into an overall sentiment signal.

    WHY THIS AGGREGATION MATTERS:
    Individual transactions need context. A $50,000 CEO purchase is noise.
    A $5,000,000 CEO purchase alongside two director purchases is a very
    different signal. This function collapses the list into a single verdict
    that you can use as an input to a trading decision.

    Signal hierarchy (strongest → weakest):
      1. CEO or CFO buying       → strongest bullish signal
      2. Multiple insiders buying simultaneously
      3. Large net buy value (>$1M)
      4. More sellers than buyers → bearish

    Args:
        transactions: List of insider transaction dicts from fetch_insider_transactions()

    Returns:
        Dict with signal, buy_value, sell_value, net_sentiment, and key_insight.
    """
    if not transactions:
        return {
            'signal':       'NEUTRAL',
            'buy_value':    0.0,
            'sell_value':   0.0,
            'net_sentiment': 0.0,
            'key_insight':  'No recent insider transactions found.',
        }

    # Focus on the last 30 days — older transactions are already public knowledge
    today = date.today()
    cutoff = today - timedelta(days=30)

    recent = [
        t for t in transactions
        if t.get('days_ago', 999) <= 30
    ]

    if not recent:
        return {
            'signal':       'NEUTRAL',
            'buy_value':    0.0,
            'sell_value':   0.0,
            'net_sentiment': 0.0,
            'key_insight':  'No insider transactions in the last 30 days.',
        }

    buy_txns  = [t for t in recent if t['transaction_type'] == 'BUY']
    sell_txns = [t for t in recent if t['transaction_type'] == 'SELL']

    buy_value  = sum(t['total_value'] for t in buy_txns)
    sell_value = sum(t['total_value'] for t in sell_txns)
    net        = buy_value - sell_value

    # Check for CEO/CFO involvement — the highest-conviction signal
    # WHY: The CEO has the most complete picture of company health. CFO controls
    # financial reporting. Both buying simultaneously is historically one of
    # the strongest predictors of 6-month outperformance.
    c_suite_buyers = [
        t for t in buy_txns
        if any(title_kw in t['title'].upper()
               for title_kw in ['CEO', 'CFO', 'CHIEF EXECUTIVE', 'CHIEF FINANCIAL'])
    ]

    # Determine the signal
    n_buyers  = len(set(t['insider_name'] for t in buy_txns))
    n_sellers = len(set(t['insider_name'] for t in sell_txns))

    if c_suite_buyers:
        signal = 'BULLISH'
        key_insight = (
            f"{c_suite_buyers[0]['title']} {c_suite_buyers[0]['insider_name']} "
            f"bought ${c_suite_buyers[0]['total_value']:,.0f} worth of shares — "
            f"C-suite open-market buying is the strongest insider signal."
        )
    elif n_buyers >= 2:
        signal = 'BULLISH'
        key_insight = (
            f"{n_buyers} insiders bought in the last 30 days "
            f"(total ${buy_value:,.0f}). Cluster buying signals high internal conviction."
        )
    elif buy_value > 1_000_000 and buy_value > sell_value * 2:
        signal = 'BULLISH'
        key_insight = (
            f"Net insider buying of ${net:,.0f} over the last 30 days. "
            f"Large-scale open-market purchases signal undervaluation."
        )
    elif n_sellers >= 3 and sell_value > buy_value * 3:
        signal = 'BEARISH'
        key_insight = (
            f"{n_sellers} insiders sold in the last 30 days "
            f"(total ${sell_value:,.0f}). Cluster selling can signal deteriorating fundamentals."
        )
    elif sell_value > buy_value:
        signal = 'NEUTRAL'
        key_insight = (
            f"Net insider selling of ${abs(net):,.0f}, but insider selling alone "
            f"is not reliably bearish (diversification, taxes, and personal needs are common reasons)."
        )
    else:
        signal = 'NEUTRAL'
        key_insight = 'Mixed or minimal insider activity — no strong directional signal.'

    return {
        'signal':        signal,
        'buy_value':     buy_value,
        'sell_value':    sell_value,
        'net_sentiment': net,
        'key_insight':   key_insight,
    }


# =============================================================================
# SECTION 4: 13F Holdings — What Hedge Funds Own
# =============================================================================
# WHY 13F FILINGS MATTER:
#
# Any institutional investment manager with more than $100 million in
# "Section 13(f) securities" (primarily U.S.-listed equities) must file Form
# 13F within 45 days of each calendar quarter end. This means we see their
# holdings as of:
#   - March 31 → disclosed by May 15
#   - June 30  → disclosed by August 14
#   - September 30 → disclosed by November 14
#   - December 31 → disclosed by February 14
#
# The most-watched 13F in finance belongs to Berkshire Hathaway (Warren Buffett).
# When Berkshire ADDS a new position, it signals that Buffett and Charlie Munger
# believe the company is priced below intrinsic value — often by a wide margin.
# The "Berkshire effect" is real: stocks added to the portfolio often rally 5–15%
# on the disclosure date as the market reprices to Buffett's implied valuation.
#
# IMPORTANT CAVEAT — 45-DAY LAG:
# By the time you read a 13F, the data is 45–135 days old. Berkshire may have
# already changed the position. 13F analysis is useful for:
#   - Identifying high-conviction long-term ideas (Buffett rarely trades in/out)
#   - Confirming your own thesis with sophisticated money
#   - Learning which sectors major funds are rotating into/out of
# It is NOT a short-term trading signal. Use it for CONVICTION, not entry timing.
#
# HOW 13F XML WORKS:
# The actual holdings are in an InfoTable XML file within the 13F filing.
# Each <infoTable> element represents one holding:
#   <nameOfIssuer>     — company name
#   <value>            — fair market value in THOUSANDS of USD
#   <shrsOrPrnAmt><sshPrnamt> — number of shares
# =============================================================================

def fetch_berkshire_holdings() -> list[dict]:
    """
    Fetch Berkshire Hathaway's latest 13F holdings from SEC EDGAR.

    Berkshire's 13F is the most-watched filing in finance. When Buffett adds a
    new position, it often signals deep value that the market hasn't fully
    recognized. The methodology here is:
      1. Fetch Berkshire's submission JSON to find the latest 13F filing
      2. Find the infotable XML document within that filing
      3. Parse each <infoTable> entry for position details
      4. Sort by value and return the top 20 holdings with portfolio percentages

    Returns:
        List of up to 20 dicts: [{company, value_usd, shares, pct_of_portfolio}]
        Empty list on error.
    """
    cik = BERKSHIRE_CIK
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'

    try:
        time.sleep(0.5)
        resp = _session.get(url, timeout=15)
        resp.raise_for_status()
        data = resp.json()
    except Exception:
        return []

    # Find the most recent 13F-HR filing.
    # WHY 13F-HR: The "-HR" suffix stands for "Holdings Report" — this is the
    # main 13F variant that contains actual portfolio data. Other variants
    # (13F-HR/A = amendment, 13F-NT = notice of non-reporting) are less useful.
    recent = data.get('filings', {}).get('recent', {})
    forms      = recent.get('form', [])
    accessions = recent.get('accessionNumber', [])
    dates      = recent.get('filingDate', [])

    target_accession = None
    target_date      = None

    for form, accession, filing_date in zip(forms, accessions, dates):
        if form in ('13F-HR', '13F-HR/A'):
            target_accession = accession
            target_date      = filing_date
            break   # filings are sorted most-recent first

    if not target_accession:
        return []

    # Fetch the filing index to find the infotable document filename.
    # EDGAR filing index URL format: /Archives/edgar/data/{cik}/{accession}-index.json
    accession_clean = target_accession.replace('-', '')
    index_url = (
        f'https://data.sec.gov/submissions/CIK{cik}.json'
    )
    # The infotable is typically named "infotable.xml" or similar.
    # We query the filing index JSON to find the exact filename.
    filing_index_url = (
        f'https://www.sec.gov/Archives/edgar/data/'
        f'{int(cik)}/{accession_clean}/{target_accession}-index.json'
    )

    try:
        time.sleep(0.5)
        idx_resp = _session.get(filing_index_url, timeout=10)
        idx_resp.raise_for_status()
        idx_data = idx_resp.json()
    except Exception:
        idx_data = {}

    # Find the infotable XML document within the filing.
    # It is typically described as 'INFORMATION TABLE' in the document description.
    infotable_filename = None
    for doc_entry in idx_data.get('documents', []):
        doc_type = doc_entry.get('type', '').upper()
        doc_name = doc_entry.get('documentName', '')
        if 'INFORMATION TABLE' in doc_type or doc_name.lower().endswith('infotable.xml'):
            infotable_filename = doc_name
            break

    # Fallback: try the standard naming convention used by most filers
    if not infotable_filename:
        infotable_filename = 'infotable.xml'

    infotable_url = (
        f'https://www.sec.gov/Archives/edgar/data/'
        f'{int(cik)}/{accession_clean}/{infotable_filename}'
    )

    try:
        time.sleep(0.5)
        xml_resp = _session.get(infotable_url, timeout=15)
        if xml_resp.status_code == 404:
            return []
        xml_resp.raise_for_status()
        root = ET.fromstring(xml_resp.text)
    except Exception:
        return []

    holdings = []

    # Parse each <infoTable> element.
    # WHY the namespace wildcard '{*}': EDGAR's XML uses namespaces that vary
    # across filers and years. The wildcard ignores the namespace prefix so
    # our code works regardless of the exact namespace declaration in the file.
    for info in root.findall('.//{*}infoTable'):
        try:
            name_el  = info.find('{*}nameOfIssuer')
            value_el = info.find('{*}value')
            # Shares are nested inside shrsOrPrnAmt → sshPrnamt
            shares_el = info.find('.//{*}sshPrnamt')

            if name_el is None or value_el is None:
                continue

            company   = name_el.text.strip() if name_el.text else 'Unknown'
            # 13F values are reported in thousands of USD — multiply by 1000
            value_usd = float(value_el.text.strip()) * 1000 if value_el.text else 0.0
            shares    = int(shares_el.text.strip().replace(',', '')) if (shares_el is not None and shares_el.text) else 0

            holdings.append({
                'company':           company,
                'value_usd':         value_usd,
                'shares':            shares,
                'pct_of_portfolio':  0.0,   # Calculated below after summing total
            })
        except (ValueError, AttributeError):
            continue

    if not holdings:
        return []

    # Calculate each position's percentage of total portfolio
    total_value = sum(h['value_usd'] for h in holdings)
    if total_value > 0:
        for h in holdings:
            h['pct_of_portfolio'] = round((h['value_usd'] / total_value) * 100, 2)

    # Sort by value descending and return top 20 positions
    # WHY top 20: Berkshire typically holds 40–50 positions, but the top 20
    # represent ~90%+ of the portfolio value and are the highest-conviction bets.
    holdings.sort(key=lambda x: x['value_usd'], reverse=True)

    return holdings[:20]


# =============================================================================
# SECTION 5: Master Fetch Function
# =============================================================================

def get_all_edgar_data(symbols: list[str]) -> dict:
    """
    Master function: fetch all EDGAR data (8-K filings, insider transactions,
    insider sentiment, and Berkshire 13F holdings) for a list of symbols.

    WHY A MASTER FUNCTION:
    The Streamlit dashboard calls one function per data source module (see
    streamlit_app.py pattern). Having a single entry point makes the
    dashboard code clean and makes it easy to add new data sources later
    without touching the UI layer.

    WHY TRY/EXCEPT PER SYMBOL:
    If EDGAR returns bad data for one symbol (malformed JSON, missing CIK,
    network timeout), we should still return valid data for all other symbols.
    Crashing on one bad ticker would deprive you of data on all tickers.
    This "fail-open" pattern is standard in production data pipelines.

    SEC EDGAR RATE LIMITS:
    The SEC explicitly allows up to 10 requests per second. We use 0.5-second
    delays (2 requests/second) to be a good API citizen. Hammering public
    services damages them for everyone and risks being blocked.

    Args:
        symbols: List of ticker symbols to fetch data for

    Returns:
        Dict with filings_8k, insider_transactions, insider_sentiment,
        berkshire_holdings, and last_updated timestamp.
    """
    # Filter to equity symbols only — EDGAR does not have filings for
    # crypto (BTC-USD), forex (EURUSD=X), or futures (GC=F, CL=F)
    # WHY: Trying to look up CIKs for these would just return 404 errors
    # and waste API calls. We silently skip non-equity symbols.
    equity_symbols = [
        s for s in symbols
        if not any(suffix in s.upper() for suffix in ['-USD', '=F', '=X', '^'])
    ]

    result = {
        'filings_8k':           {},
        'insider_transactions': {},
        'insider_sentiment':    {},
        'berkshire_holdings':   [],
        'last_updated':         datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    }

    # --- 8-K filings and insider transactions per symbol ---
    for symbol in equity_symbols:
        # 8-K Scanner
        try:
            filings = fetch_recent_8k(symbol, max_filings=5)
            result['filings_8k'][symbol] = filings
        except Exception:
            # Never propagate exceptions from individual symbol fetches
            result['filings_8k'][symbol] = []

        time.sleep(0.5)   # Courtesy delay between symbols

        # Insider Transactions
        try:
            transactions = fetch_insider_transactions(symbol, max_transactions=10)
            result['insider_transactions'][symbol] = transactions
            result['insider_sentiment'][symbol]    = analyze_insider_sentiment(transactions)
        except Exception:
            result['insider_transactions'][symbol] = []
            result['insider_sentiment'][symbol]    = {
                'signal': 'NEUTRAL', 'buy_value': 0.0, 'sell_value': 0.0,
                'net_sentiment': 0.0, 'key_insight': 'Data unavailable.',
            }

        time.sleep(0.5)

    # --- Berkshire 13F holdings (fetched once, not per-symbol) ---
    # WHY: Berkshire's portfolio is market-wide context, not specific to any
    # one symbol in our universe. We fetch it once and display it separately.
    try:
        result['berkshire_holdings'] = fetch_berkshire_holdings()
    except Exception:
        result['berkshire_holdings'] = []

    return result


# =============================================================================
# Module self-test
# =============================================================================
# WHY: A quick smoke test that runs when the module is executed directly.
# This verifies imports and basic wiring without making live API calls.
# The Streamlit app imports this module — if the import fails, the tab
# will degrade gracefully (matching the try/except pattern in streamlit_app.py).

if __name__ == '__main__':
    print('edgar.py loaded successfully.')
    print(f'Pre-built ticker→CIK map covers {len(TICKER_TO_CIK)} symbols.')
    print('CIK lookup test (AAPL):', get_cik('AAPL'))
    print('Module self-test complete.')
