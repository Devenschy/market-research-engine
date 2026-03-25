# =============================================================================
# streamlit_app.py — Browser Dashboard for Market Research Engine
# =============================================================================
# Run with: streamlit run streamlit_app.py
# Then open: http://localhost:8501
#
# TABS:
#   1. Trading     — original dashboard (portfolio KPIs, prices, signals, P&L chart)
#   2. Options     — live options chains, IV Rank, put/call ratio, greeks explainer
#   3. Derivatives — crypto perp positions, funding rates, commodity futures curves

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import yfinance as yf
import config

# Import new derivatives modules — wrapped in try/except so the app
# continues to work even if these modules have an import error.
# This is "graceful degradation" — a core principle of resilient systems.
try:
    import options as options_module
    OPTIONS_AVAILABLE = True
except ImportError as e:
    OPTIONS_AVAILABLE = False
    OPTIONS_IMPORT_ERROR = str(e)

try:
    import derivatives as derivatives_module
    DERIVATIVES_AVAILABLE = True
except ImportError as e:
    DERIVATIVES_AVAILABLE = False
    DERIVATIVES_IMPORT_ERROR = str(e)

try:
    import pairs as pairs_module
    PAIRS_AVAILABLE = True
except ImportError as e:
    PAIRS_AVAILABLE = False

try:
    import sentiment as sentiment_module
    import nltk
    nltk.download('vader_lexicon', quiet=True)
    SENTIMENT_AVAILABLE = True
except ImportError as e:
    SENTIMENT_AVAILABLE = False

try:
    import edgar as edgar_module
    EDGAR_AVAILABLE = True
except ImportError as e:
    EDGAR_AVAILABLE = False

try:
    import factors as factors_module
    FACTORS_AVAILABLE = True
    FACTORS_IMPORT_ERROR = None
except Exception as e:
    FACTORS_AVAILABLE = False
    FACTORS_IMPORT_ERROR = str(e)

try:
    import events as events_module
    EVENTS_AVAILABLE = True
    EVENTS_IMPORT_ERROR = None
except Exception as e:
    EVENTS_AVAILABLE = False
    EVENTS_IMPORT_ERROR = str(e)


st.set_page_config(
    page_title="Market Research Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Auto-refresh every 60 seconds
st.markdown(
    '<meta http-equiv="refresh" content="60">',
    unsafe_allow_html=True
)

# --- Header ---
st.title("📈 Market Research Engine")
st.caption(f"Dezona Group — Paper Trading | Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
st.divider()


# =============================================================================
# HELPER FUNCTIONS — Data Loading
# =============================================================================

def load_performance():
    if os.path.exists(config.PERFORMANCE_LOG):
        with open(config.PERFORMANCE_LOG) as f:
            return json.load(f)
    return None


def load_trades():
    if os.path.exists(config.TRADES_LOG):
        try:
            df = pd.read_csv(config.TRADES_LOG)
            if not df.empty:
                df['entry_time'] = pd.to_datetime(df['entry_time'], errors='coerce')
                df['exit_time'] = pd.to_datetime(df['exit_time'], errors='coerce')
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def load_signals():
    if os.path.exists(config.SIGNALS_LOG):
        try:
            df = pd.read_csv(config.SIGNALS_LOG)
            if not df.empty:
                df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            return df
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def fetch_live_prices():
    prices = {}
    for symbol in config.SYMBOLS:
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                prices[symbol] = round(float(hist['Close'].iloc[-1]), 4)
        except Exception:
            prices[symbol] = None
    return prices


# =============================================================================
# TOP-LEVEL TAB LAYOUT
# =============================================================================
# WHY THREE TABS:
# Putting all information in one scrollable page creates cognitive overload.
# Tabs separate concerns cleanly:
# - Trading: The core engine outputs (what the system is doing)
# - Options: Derivatives intelligence layered on top of spot signals
# - Derivatives: Perps and futures — additional market structure signals

main_tab, options_tab, derivatives_tab, pairs_tab, sentiment_tab, edgar_tab, factors_tab, events_tab = st.tabs([
    "📊 Trading",
    "🎯 Options",
    "⛓ Derivatives",
    "🔗 Pairs / Stat Arb",
    "📰 Sentiment",
    "🏛 SEC / EDGAR",
    "📐 Factor Investing",
    "📅 Events Calendar"
])


# =============================================================================
# TAB 1: TRADING (Original Dashboard Content)
# =============================================================================

with main_tab:

    # --- Load data ---
    perf = load_performance()
    trades_df = load_trades()
    signals_df = load_signals()

    # --- Row 1: Portfolio KPIs ---
    col1, col2, col3, col4, col5 = st.columns(5)

    if perf:
        port = perf.get('portfolio', {})
        equity = port.get('equity', config.STARTING_CAPITAL)
        total_return = port.get('total_return_pct', 0.0)
        realized_pnl = port.get('realized_pnl', 0.0)
        win_rate = port.get('win_rate', 0.0)
        sharpe = port.get('sharpe_ratio', None)
        total_trades = port.get('total_trades', 0)
    else:
        equity = config.STARTING_CAPITAL
        total_return = 0.0
        realized_pnl = 0.0
        win_rate = 0.0
        sharpe = None
        total_trades = 0

    col1.metric("Equity", f"${equity:,.2f}",
                delta=f"{total_return:+.2f}%")
    col2.metric("Realized P&L", f"${realized_pnl:+,.2f}")
    col3.metric("Win Rate", f"{win_rate:.1f}%")
    col4.metric("Sharpe Ratio", f"{sharpe:.3f}" if sharpe else "—")
    col5.metric("Total Trades", total_trades)

    st.divider()

    # --- Row 2: Live Prices ---
    st.subheader("Live Prices")

    with st.spinner("Fetching live prices..."):
        live_prices = fetch_live_prices()

    price_cols = st.columns(len(config.SYMBOLS))
    for i, symbol in enumerate(config.SYMBOLS):
        price = live_prices.get(symbol)
        price_cols[i].metric(
            label=symbol,
            value=f"{price:,.4f}" if price else "—"
        )

    st.divider()

    # --- Row 3: Trades + Strategy Performance side by side ---
    left, right = st.columns([2, 1])

    with left:
        st.subheader("Trade History")
        if trades_df.empty:
            st.info("No trades yet. The engine is running and waiting for 2/3 strategy agreement.")
        else:
            display_df = trades_df.copy()

            # Color P&L column
            def color_pnl(val):
                color = "green" if val > 0 else "red"
                return f"color: {color}"

            # Format columns
            if 'pnl' in display_df.columns:
                display_df['pnl'] = display_df['pnl'].apply(lambda x: f"${x:+.2f}")
            if 'pnl_pct' in display_df.columns:
                display_df['pnl_pct'] = display_df['pnl_pct'].apply(lambda x: f"{x:+.2f}%")
            if 'entry_time' in display_df.columns:
                display_df['entry_time'] = display_df['entry_time'].dt.strftime('%m/%d %H:%M')
            if 'exit_time' in display_df.columns:
                display_df['exit_time'] = display_df['exit_time'].dt.strftime('%m/%d %H:%M')

            cols_to_show = ['exit_time', 'symbol', 'direction', 'entry_price',
                            'exit_price', 'pnl', 'pnl_pct', 'exit_reason', 'strategy', 'regime']
            cols_to_show = [c for c in cols_to_show if c in display_df.columns]

            st.dataframe(
                display_df[cols_to_show].sort_values('exit_time', ascending=False).head(50),
                use_container_width=True,
                hide_index=True
            )

    with right:
        st.subheader("By Strategy")
        if perf and perf.get('by_strategy'):
            for strat_name, stats in perf['by_strategy'].items():
                with st.container(border=True):
                    st.markdown(f"**{strat_name.replace('_', ' ')}**")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Trades", stats.get('total_trades', 0))
                    s2.metric("Win Rate", f"{stats.get('win_rate', 0):.1f}%")
                    s3.metric("Avg P&L", f"{stats.get('avg_pnl_pct', 0):+.2f}%")
        else:
            st.info("Strategy stats appear after first trades close.")

    st.divider()

    # --- Row 4: Signal Log ---
    st.subheader("Signal Log")

    if signals_df.empty:
        st.info("No signals yet — engine is warming up or waiting for conditions.")
    else:
        # Tabs: All signals vs Traded only vs Blocked only
        sig_tab1, sig_tab2, sig_tab3 = st.tabs(["All Signals", "Traded ✓", "Blocked ✗"])

        def format_signals(df):
            display = df.copy()
            if 'timestamp' in display.columns:
                display['timestamp'] = display['timestamp'].dt.strftime('%m/%d %H:%M:%S')
            cols = ['timestamp', 'symbol', 'strategy', 'signal', 'price',
                    'regime', 'triggered_trade', 'suppression_reason']
            cols = [c for c in cols if c in display.columns]
            return display[cols].sort_values('timestamp', ascending=False).head(100)

        with sig_tab1:
            st.dataframe(format_signals(signals_df), use_container_width=True, hide_index=True)
        with sig_tab2:
            traded = signals_df[signals_df['triggered_trade'] == True]
            st.dataframe(format_signals(traded), use_container_width=True, hide_index=True)
        with sig_tab3:
            blocked = signals_df[signals_df['triggered_trade'] == False]
            st.dataframe(format_signals(blocked), use_container_width=True, hide_index=True)

    # --- P&L Chart ---
    if not trades_df.empty and 'pnl' in trades_df.columns:
        st.divider()
        st.subheader("Cumulative P&L")
        chart_df = trades_df.dropna(subset=['exit_time']).sort_values('exit_time').copy()
        chart_df['pnl_raw'] = pd.to_numeric(
            chart_df['pnl'].astype(str).str.replace('$', '').str.replace('+', ''), errors='coerce'
        ) if chart_df['pnl'].dtype == object else chart_df['pnl']
        chart_df['cumulative_pnl'] = chart_df['pnl_raw'].cumsum()
        st.line_chart(chart_df.set_index('exit_time')['cumulative_pnl'])

    # --- Footer ---
    st.divider()
    st.caption(
        "Paper trading only — no real capital at risk. "
        "Engine must be running separately (`python main.py`) to generate live signals. "
        "This dashboard reads from logs/ and refreshes every 60 seconds."
    )


# =============================================================================
# TAB 2: OPTIONS
# =============================================================================

with options_tab:
    st.header("Options Intelligence")
    st.caption(
        "Covers AAPL and MSFT only — the most liquid equity options markets. "
        "Data via yfinance. Black-Scholes pricing calculated in-engine."
    )

    if not OPTIONS_AVAILABLE:
        st.error(f"options.py module could not be imported: {OPTIONS_IMPORT_ERROR}")
        st.stop()

    # -------------------------------------------------------------------------
    # SYMBOL SELECTOR
    # -------------------------------------------------------------------------
    selected_symbol = st.selectbox(
        "Select Symbol",
        options=options_module.OPTIONS_SYMBOLS,
        key="options_symbol_select"
    )

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION A: IV RANK GAUGE
    # -------------------------------------------------------------------------
    # WHY IV RANK FIRST:
    # Before looking at any specific contract, you need to know the IV regime.
    # IV Rank tells you whether options are cheap or expensive right now.
    # This is the FIRST thing professional options traders check every morning.
    st.subheader(f"IV Rank — {selected_symbol}")
    st.caption(
        "IV Rank (0-100) shows where current implied volatility sits within its "
        "52-week range. High IVR = expensive options (favor selling). "
        "Low IVR = cheap options (favor buying)."
    )

    with st.spinner(f"Calculating IV Rank for {selected_symbol}..."):
        iv_rank = options_module.fetch_iv_rank(selected_symbol)

    if iv_rank is not None:
        # Color-code the metric by IV regime
        # Green < 30 (cheap), Yellow 30-70 (normal), Red > 70 (expensive)
        if iv_rank < 30:
            iv_color = "normal"
            iv_label = f"IV Rank: {iv_rank:.0f} — LOW (Options are CHEAP → Favor BUYING)"
            iv_advice = (
                "IV is in the bottom 30% of its yearly range. Option premiums are relatively "
                "inexpensive. This is the better environment for buying calls or puts, as you "
                "pay less for the same leverage. Selling premium in low IV captures less income."
            )
        elif iv_rank > 70:
            iv_color = "inverse"
            iv_label = f"IV Rank: {iv_rank:.0f} — HIGH (Options are EXPENSIVE → Favor SELLING)"
            iv_advice = (
                "IV is in the top 30% of its yearly range. Option premiums are elevated. "
                "This is the environment for selling premium — covered calls, cash-secured puts, "
                "or credit spreads. Theta (time decay) works faster for sellers when IV is high. "
                "Avoid buying single options when IV is rich unless you have a strong directional view."
            )
        else:
            iv_color = "off"
            iv_label = f"IV Rank: {iv_rank:.0f} — NEUTRAL"
            iv_advice = (
                "IV is in the middle of its yearly range. No strong premium-buying or "
                "premium-selling edge from IV alone. Use directional conviction to guide strategy choice."
            )

        iv_col1, iv_col2 = st.columns([1, 3])
        with iv_col1:
            # Large visual progress bar as IV gauge
            st.metric("IV Rank", f"{iv_rank:.0f} / 100")
            st.progress(int(iv_rank) / 100)
            if iv_rank < 30:
                st.success("LOW IV — Buy Options")
            elif iv_rank > 70:
                st.error("HIGH IV — Sell Premium")
            else:
                st.warning("NEUTRAL IV")

        with iv_col2:
            st.markdown(f"**{iv_label}**")
            st.info(iv_advice)
            st.caption(
                f"Thresholds — Low: < {config.IV_RANK_LOW} | High: > {config.IV_RANK_HIGH}"
            )
    else:
        st.warning(f"Could not calculate IV Rank for {selected_symbol}. yfinance data may be unavailable.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION B: LIVE OPTIONS CHAIN
    # -------------------------------------------------------------------------
    st.subheader(f"Live Options Chain — {selected_symbol}")
    st.caption(
        "Nearest expiry only (highest liquidity). "
        "IV = Implied Volatility. OI = Open Interest. "
        "Moneyness % = how far the strike is from current stock price."
    )

    with st.spinner(f"Fetching options chain for {selected_symbol}..."):
        chain_data = options_module.fetch_options_chain(selected_symbol)

    if chain_data is not None:
        current_price = chain_data['current_price']
        expiry = chain_data['expiry']
        dte = chain_data['days_to_expiry']

        # Summary metrics
        meta_c1, meta_c2, meta_c3 = st.columns(3)
        meta_c1.metric("Current Price", f"${current_price:,.2f}")
        meta_c2.metric("Nearest Expiry", expiry)
        meta_c3.metric("Days to Expiry", dte,
                        delta=f"{'⚠ Short DTE' if dte < 14 else 'OK'}")

        # Display calls and puts side by side
        chain_col1, chain_col2 = st.columns(2)

        with chain_col1:
            st.markdown("**Calls (Bullish)**")
            calls = chain_data['calls'].copy()

            # Select and rename columns for clean display
            display_cols = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility',
                            'volume', 'openInterest', 'moneyness_pct']
            display_cols = [c for c in display_cols if c in calls.columns]
            calls_display = calls[display_cols].copy()

            # Format implied volatility as percentage
            if 'impliedVolatility' in calls_display.columns:
                calls_display['impliedVolatility'] = (
                    calls_display['impliedVolatility'] * 100
                ).round(1).astype(str) + '%'

            calls_display.columns = [c.replace('impliedVolatility', 'IV%')
                                       .replace('lastPrice', 'Last')
                                       .replace('openInterest', 'OI')
                                       .replace('moneyness_pct', 'Money%')
                                       for c in calls_display.columns]

            # Highlight ATM row
            atm_strike_idx = (calls['strike'] - current_price).abs().idxmin()
            atm_strike = calls.loc[atm_strike_idx, 'strike']

            st.dataframe(
                calls_display.sort_values('strike').reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

        with chain_col2:
            st.markdown("**Puts (Bearish)**")
            puts = chain_data['puts'].copy()

            display_cols = ['strike', 'lastPrice', 'bid', 'ask', 'impliedVolatility',
                            'volume', 'openInterest', 'moneyness_pct']
            display_cols = [c for c in display_cols if c in puts.columns]
            puts_display = puts[display_cols].copy()

            if 'impliedVolatility' in puts_display.columns:
                puts_display['impliedVolatility'] = (
                    puts_display['impliedVolatility'] * 100
                ).round(1).astype(str) + '%'

            puts_display.columns = [c.replace('impliedVolatility', 'IV%')
                                      .replace('lastPrice', 'Last')
                                      .replace('openInterest', 'OI')
                                      .replace('moneyness_pct', 'Money%')
                                      for c in puts_display.columns]

            st.dataframe(
                puts_display.sort_values('strike', ascending=False).reset_index(drop=True),
                use_container_width=True,
                hide_index=True
            )

        st.caption(f"ATM Strike (nearest to ${current_price:.2f}): ${atm_strike:.2f}")

    else:
        st.warning(
            f"Could not fetch options chain for {selected_symbol}. "
            "yfinance options data can be intermittently unavailable. Try again in a moment."
        )

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION C: PUT/CALL RATIO
    # -------------------------------------------------------------------------
    # WHY PUT/CALL RATIO:
    # The PCR is one of the oldest and most reliable sentiment indicators.
    # When options market participants (typically more sophisticated than average
    # stock traders) are heavily buying puts, it signals elevated fear or hedging.
    # When call buying dominates, it can signal speculative excess.
    st.subheader(f"Put/Call Ratio — {selected_symbol}")

    if chain_data is not None:
        pcr_data = options_module.put_call_ratio(chain_data)

        if pcr_data and pcr_data.get('pcr') is not None:
            pcr = pcr_data['pcr']

            pcr_c1, pcr_c2, pcr_c3 = st.columns(3)
            pcr_c1.metric("Put/Call Ratio", f"{pcr:.3f}")
            pcr_c2.metric("Total Call Volume", f"{pcr_data.get('total_call_volume', 0):,}")
            pcr_c3.metric("Total Put Volume", f"{pcr_data.get('total_put_volume', 0):,}")

            sentiment = pcr_data.get('sentiment', 'UNKNOWN')
            if sentiment == 'BEARISH':
                st.error(f"Sentiment: {sentiment} — {pcr_data.get('interpretation', '')}")
            elif sentiment == 'BULLISH':
                st.success(f"Sentiment: {sentiment} — {pcr_data.get('interpretation', '')}")
            else:
                st.info(f"Sentiment: {sentiment} — {pcr_data.get('interpretation', '')}")

            # Reference thresholds
            st.caption(
                "PCR Thresholds: > 1.2 = Bearish (put-heavy) | < 0.7 = Bullish (call-heavy) | "
                "0.7-1.2 = Neutral. Remember: PCR is a contrarian indicator at extremes."
            )
        else:
            st.info("Put/Call ratio data unavailable (volume data may be empty for this expiry).")
    else:
        st.info("Options chain required for PCR calculation.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION D: GREEKS EXPLAINER TABLE
    # -------------------------------------------------------------------------
    # WHY AN EXPLAINER TABLE IN THE UI:
    # The Greeks are abstract until you see them in context. Having the
    # plain-English explainer always visible means you can cross-reference
    # as you look at the options chain above. Learning happens through repetition
    # and contextual association, not from reading a textbook once.
    st.subheader("Options Greeks — Plain English Reference")
    st.caption(
        "These four numbers control everything about how an option behaves. "
        "Professional traders think in terms of Greek exposures, not just price."
    )

    if OPTIONS_AVAILABLE:
        greeks_data = []
        for greek, explanation in options_module.GREEKS_EXPLAINED.items():
            greeks_data.append({'Greek': greek.upper(), 'What It Means': explanation})

        greeks_df = pd.DataFrame(greeks_data)
        st.dataframe(
            greeks_df,
            use_container_width=True,
            hide_index=True,
            column_config={
                'Greek': st.column_config.TextColumn('Greek', width='small'),
                'What It Means': st.column_config.TextColumn('What It Means', width='large')
            }
        )

        # Also show the Black-Scholes Greeks for the ATM option if we have chain data
        if chain_data is not None and iv_rank is not None:
            st.divider()
            st.markdown("**Live Greeks for ATM Option**")
            st.caption(
                "Greeks calculated in-engine using Black-Scholes. "
                "IV from market; r = 5% risk-free rate."
            )

            try:
                calls = chain_data['calls']
                atm_idx = (calls['strike'] - chain_data['current_price']).abs().idxmin()
                atm_row = calls.loc[atm_idx]
                atm_strike = float(atm_row['strike'])
                atm_iv = float(atm_row['impliedVolatility']) if atm_row['impliedVolatility'] > 0 else 0.25
                dte_T = chain_data['days_to_expiry'] / 365.0

                if dte_T > 0:
                    call_greeks = options_module.calculate_greeks(
                        chain_data['current_price'], atm_strike, dte_T,
                        options_module.RISK_FREE_RATE, atm_iv, 'call'
                    )
                    put_greeks = options_module.calculate_greeks(
                        chain_data['current_price'], atm_strike, dte_T,
                        options_module.RISK_FREE_RATE, atm_iv, 'put'
                    )

                    greek_live_cols = st.columns(4)
                    greek_names = ['delta', 'gamma', 'theta', 'vega']
                    for i, g in enumerate(greek_names):
                        with greek_live_cols[i]:
                            st.metric(
                                f"Call {g.title()}",
                                f"{call_greeks.get(g, 0):.4f}",
                                help=options_module.GREEKS_EXPLAINED.get(g, '')
                            )

                    st.caption(
                        f"Strike: ${atm_strike:.2f} | IV: {atm_iv*100:.1f}% | "
                        f"DTE: {chain_data['days_to_expiry']} | "
                        f"ATM Call BS Price: ${call_greeks.get('price', 0):.2f} | "
                        f"ATM Put BS Price: ${put_greeks.get('price', 0):.2f}"
                    )
            except Exception as e:
                st.caption(f"Live greeks calculation error: {e}")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION E: PAPER OPTIONS POSITIONS
    # -------------------------------------------------------------------------
    st.subheader("Active Paper Options Positions")
    st.caption(
        "Paper positions opened by the generate_options_signal() function. "
        "P&L marked to market using Black-Scholes at current price."
    )

    if OPTIONS_AVAILABLE and options_module.paper_positions:
        # Get current prices for mark-to-market
        current_prices_for_options = {}
        for sym in options_module.OPTIONS_SYMBOLS:
            try:
                ticker = yf.Ticker(sym)
                hist = ticker.history(period='1d', interval='1m')
                if not hist.empty:
                    current_prices_for_options[sym] = float(hist['Close'].iloc[-1])
            except Exception:
                pass

        positions_summary = options_module.get_all_positions_summary(current_prices_for_options)

        if positions_summary:
            pos_df = pd.DataFrame(positions_summary)

            # Color-code P&L
            def style_pnl(val):
                if isinstance(val, (int, float)):
                    return 'color: green' if val >= 0 else 'color: red'
                return ''

            st.dataframe(
                pos_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'total_pnl': st.column_config.NumberColumn('P&L ($)', format='$%.2f'),
                    'pnl_pct': st.column_config.NumberColumn('P&L %', format='%.2f%%'),
                    'entry_premium': st.column_config.NumberColumn('Entry Premium', format='$%.4f'),
                    'current_premium': st.column_config.NumberColumn('Current Premium', format='$%.4f'),
                }
            )
        else:
            st.info("No active paper positions.")
    else:
        st.info(
            "No paper options positions open. Positions are created by "
            "generate_options_signal() when the engine produces directional signals "
            "for AAPL or MSFT."
        )


# =============================================================================
# TAB 3: DERIVATIVES (Perps + Futures Curves)
# =============================================================================

with derivatives_tab:
    st.header("Derivatives Dashboard")
    st.caption(
        "Crypto perpetuals (BTC, ETH via Binance) and commodity futures curves "
        "(Gold, Crude Oil via yfinance). All positions are paper only."
    )

    if not DERIVATIVES_AVAILABLE:
        st.error(f"derivatives.py module could not be imported: {DERIVATIVES_IMPORT_ERROR}")
        st.stop()

    # -------------------------------------------------------------------------
    # SECTION A: CRYPTO PERP POSITIONS
    # -------------------------------------------------------------------------
    st.subheader("Crypto Perpetual Positions")
    st.caption(
        "Paper perpetual futures. Leverage: 2x max. "
        "Liquidation price highlighted when close — this is where your margin gets wiped."
    )

    # Get current crypto prices
    crypto_prices = {}
    for sym in derivatives_module.PERP_SYMBOLS:
        try:
            ticker = yf.Ticker(sym)
            hist = ticker.history(period='1d', interval='1m')
            if not hist.empty:
                crypto_prices[sym] = round(float(hist['Close'].iloc[-1]), 2)
        except Exception:
            pass

    # Display current crypto prices
    price_cols_perp = st.columns(len(derivatives_module.PERP_SYMBOLS))
    for i, sym in enumerate(derivatives_module.PERP_SYMBOLS):
        price = crypto_prices.get(sym)
        price_cols_perp[i].metric(sym, f"${price:,.2f}" if price else "—")

    # Perp positions table
    if derivatives_module.perp_positions:
        perp_summary = derivatives_module.get_perp_summary(crypto_prices)

        if perp_summary:
            perp_df = pd.DataFrame(perp_summary)

            st.dataframe(
                perp_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    'liquidation_price': st.column_config.NumberColumn(
                        'Liq Price ⚠',
                        format='$%.2f',
                        help='If price reaches this level, your margin is wiped. Keep distance > 10%.'
                    ),
                    'unrealized_pnl': st.column_config.NumberColumn('Unr. P&L ($)', format='$%.2f'),
                    'unrealized_pnl_pct': st.column_config.NumberColumn('P&L %', format='%.2f%%'),
                    'distance_to_liq_pct': st.column_config.NumberColumn(
                        'Dist to Liq %',
                        format='%.2f%%',
                        help='How far price needs to move against you before liquidation.'
                    ),
                }
            )

            # Warning if any position is close to liquidation (< 10% distance)
            for pos_info in perp_summary:
                if pos_info['is_liquidated']:
                    st.error(
                        f"LIQUIDATED: {pos_info['direction']} {pos_info['symbol']} — "
                        f"Margin of ${pos_info['margin_posted']:,.2f} lost."
                    )
                elif pos_info['distance_to_liq_pct'] < 10:
                    st.warning(
                        f"WARNING: {pos_info['direction']} {pos_info['symbol']} is "
                        f"{pos_info['distance_to_liq_pct']:.1f}% from liquidation at "
                        f"${pos_info['liquidation_price']:,.2f}. Consider reducing size."
                    )
    else:
        st.info(
            "No paper perp positions open. Use derivatives.open_perp_position() to open one. "
            "Positions are tracked in derivatives.perp_positions[]."
        )

    # -------------------------------------------------------------------------
    # Educational: Liquidation Calculator
    # -------------------------------------------------------------------------
    with st.expander("Liquidation Price Calculator (Educational)"):
        st.caption(
            "Enter a hypothetical position to understand where liquidation would occur. "
            "This is the single most important concept in leveraged trading — "
            "always know your liquidation price BEFORE entering a trade."
        )

        liq_col1, liq_col2, liq_col3 = st.columns(3)
        liq_entry = liq_col1.number_input("Entry Price ($)", value=50000.0, step=1000.0)
        liq_leverage = liq_col2.slider("Leverage", min_value=1, max_value=10, value=2)
        liq_direction = liq_col3.selectbox("Direction", ["LONG", "SHORT"])

        if st.button("Calculate Liquidation Price"):
            try:
                liq_price = derivatives_module.calculate_liquidation_price(
                    liq_entry, float(liq_leverage), liq_direction
                )
                pct_move = abs(liq_price - liq_entry) / liq_entry * 100
                st.metric(
                    "Liquidation Price",
                    f"${liq_price:,.2f}",
                    delta=f"{'-' if liq_direction == 'LONG' else '+'}{pct_move:.1f}% from entry"
                )

                liq_explanation = (
                    f"At {liq_leverage}x leverage, you control ${liq_entry * liq_leverage:,.0f} "
                    f"in notional with ${liq_entry:,.0f} margin. "
                    f"A {pct_move:.1f}% adverse move liquidates you. "
                    f"At 10x leverage that same {pct_move:.1f}% would become a mere {100/liq_leverage:.1f}% — "
                    "why high leverage is so dangerous."
                )
                st.info(liq_explanation)
            except Exception as e:
                st.error(f"Calculation error: {e}")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION B: FUNDING RATE HISTORY
    # -------------------------------------------------------------------------
    st.subheader("Funding Rate History")
    st.caption(
        "Crypto perpetuals use funding rates (paid every 8h) to keep the perp "
        "price anchored to spot. Persistent positive funding = longs paying shorts = "
        "sign of over-leveraged bullish positioning."
    )

    funding_symbol_select = st.selectbox(
        "Select Crypto",
        options=['BTCUSDT', 'ETHUSDT'],
        key="funding_symbol"
    )

    with st.spinner(f"Fetching funding rate data for {funding_symbol_select}..."):
        # Current funding rate
        current_funding = derivatives_module.get_current_funding_rate(funding_symbol_select)
        # Historical funding (last 24 periods = ~8 days)
        funding_history = derivatives_module.fetch_funding_history(funding_symbol_select, limit=24)

    if current_funding:
        fund_c1, fund_c2, fund_c3 = st.columns(3)
        fund_c1.metric(
            "Current Funding Rate",
            f"{current_funding['funding_rate_pct']:.4f}%",
            help="Paid every 8 hours. Positive = longs pay shorts."
        )
        fund_c2.metric(
            "Annualized Rate",
            f"{current_funding['annualized_pct']:.2f}%",
            help="If this rate persisted all year. Used to estimate carry cost for long holders."
        )
        fund_c3.metric("Market Sentiment", current_funding['sentiment'])

        sentiment = current_funding['sentiment']
        if 'EXTREME' in sentiment:
            st.error(current_funding['interpretation'])
        elif 'EXCESS' in sentiment or 'BEARISH' in sentiment:
            st.warning(current_funding['interpretation'])
        else:
            st.info(current_funding['interpretation'])

    if funding_history:
        # Build a chart from funding history
        hist_df = pd.DataFrame(funding_history)
        hist_df['timestamp'] = pd.to_datetime(hist_df['timestamp'])
        hist_df = hist_df.sort_values('timestamp')

        st.markdown("**Funding Rate History (Last 8 Days)**")
        st.line_chart(
            hist_df.set_index('timestamp')['funding_rate_pct'],
            use_container_width=True
        )

        # Summary stats
        avg_rate = hist_df['funding_rate_pct'].mean()
        max_rate = hist_df['funding_rate_pct'].max()
        min_rate = hist_df['funding_rate_pct'].min()

        hist_c1, hist_c2, hist_c3 = st.columns(3)
        hist_c1.metric("Avg Funding (8d)", f"{avg_rate:.4f}%")
        hist_c2.metric("Max (Long most expensive)", f"{max_rate:.4f}%")
        hist_c3.metric("Min (Short most expensive)", f"{min_rate:.4f}%")

        st.caption(
            "Cumulative cost for a long held 8 days at current rate: "
            f"{avg_rate * 24:.3f}% of notional (3 payments/day × 8 days)"
        )
    else:
        st.info("Funding history unavailable — Binance API may be geo-restricted.")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION C: COMMODITY FUTURES CURVE
    # -------------------------------------------------------------------------
    st.subheader("Commodity Futures Curve")
    st.caption(
        "Contango vs Backwardation — the curve shape tells you about supply/demand dynamics "
        "and has major implications for commodity ETF returns (contango drag)."
    )

    futures_col1, futures_col2 = st.columns(2)

    for i, futures_symbol in enumerate(['GC=F', 'CL=F']):
        label = "Gold (GC=F)" if futures_symbol == 'GC=F' else "Crude Oil (CL=F)"

        target_col = futures_col1 if i == 0 else futures_col2

        with target_col:
            with st.spinner(f"Fetching {label} futures data..."):
                futures_data = derivatives_module.fetch_futures_curve(futures_symbol)

            st.markdown(f"**{label}**")

            if futures_data:
                roll_yield = futures_data['roll_yield_pct']
                curve_shape = futures_data['curve_shape']

                f1, f2, f3 = st.columns(3)
                f1.metric("Front Month", f"${futures_data['front_price']:,.2f}")
                f2.metric("~Next Month", f"${futures_data['next_price_approx']:,.2f}")
                f3.metric(
                    "Roll Yield",
                    f"{roll_yield:+.3f}%",
                    delta=f"{'+ Backwardation' if roll_yield > 0 else '- Contango'}"
                )

                if 'BACKWARDATION' in curve_shape:
                    st.success(f"Curve: {curve_shape}")
                elif 'CONTANGO' in curve_shape:
                    st.error(f"Curve: {curve_shape}")
                else:
                    st.info(f"Curve: {curve_shape}")

                with st.expander("Explanation"):
                    st.write(futures_data['explanation'])
                    st.divider()
                    st.markdown("**ETF Impact:**")
                    st.write(futures_data['etf_impact'])

                st.metric("30-Day Realized Volatility", f"{futures_data['realized_vol_30d']:.1f}%")

            else:
                st.warning(f"Could not fetch futures data for {futures_symbol}")

    st.divider()

    # -------------------------------------------------------------------------
    # SECTION D: GREEKS REFERENCE (from derivatives module)
    # -------------------------------------------------------------------------
    st.subheader("Greeks Reference — from derivatives.GREEKS_EXPLAINED")
    st.caption(
        "The same Greeks reference dictionary from derivatives.py. "
        "Importable anywhere in the codebase for educational scaffolding."
    )

    greeks_ref_data = []
    for greek, explanation in derivatives_module.GREEKS_EXPLAINED.items():
        greeks_ref_data.append({'Greek': greek.upper(), 'Plain English Explanation': explanation})

    st.dataframe(
        pd.DataFrame(greeks_ref_data),
        use_container_width=True,
        hide_index=True,
        column_config={
            'Greek': st.column_config.TextColumn('Greek', width='small'),
            'Plain English Explanation': st.column_config.TextColumn(
                'Plain English Explanation', width='large'
            )
        }
    )

    # -------------------------------------------------------------------------
    # Footer
    # -------------------------------------------------------------------------
    st.divider()
    st.caption(
        "All derivatives data is for educational purposes. Positions are paper only. "
        "Crypto perp data from Binance public API (may be geo-restricted). "
        "Commodity futures data from yfinance (front-month approximation only). "
        "Options data from yfinance (may be delayed or unavailable outside market hours)."
    )

# =============================================================================
# TAB 4: PAIRS / STATISTICAL ARBITRAGE
# =============================================================================
with pairs_tab:
    st.header("Pairs Trading — Statistical Arbitrage")
    st.caption(
        "Pairs trading finds two correlated assets whose price relationship has "
        "temporarily diverged. Buy the cheap one, short the expensive one, profit "
        "when they converge. Market-neutral — direction doesn't matter, only relative value."
    )

    if not PAIRS_AVAILABLE:
        st.error("pairs.py failed to import. Run: pip install statsmodels")
    else:
        with st.spinner("Running cointegration tests on all pairs..."):
            try:
                pair_results = pairs_module.analyze_all_pairs()
            except Exception as e:
                pair_results = []
                st.error(f"Pairs analysis error: {e}")

        if not pair_results:
            st.info("No pair results returned — check your internet connection.")
        else:
            for pair in pair_results:
                sym1 = pair.get('symbol1', '')
                sym2 = pair.get('symbol2', '')
                zscore = pair.get('zscore', 0) or 0
                signal = pair.get('signal')
                cointegrated = pair.get('cointegrated', False)
                p_value = pair.get('p_value', 1.0) or 1.0
                correlation = pair.get('correlation', 0) or 0
                hedge_ratio = pair.get('hedge_ratio', 1.0) or 1.0

                # Color-code the container border by signal strength
                with st.container(border=True):
                    col1, col2, col3, col4, col5 = st.columns([2, 1.5, 1.5, 1.5, 3])

                    col1.metric("Pair", f"{sym1} / {sym2}")
                    col2.metric("Z-Score", f"{zscore:+.2f}",
                                delta="SIGNAL" if signal else "no signal",
                                delta_color="normal" if signal else "off")
                    col3.metric("Correlation", f"{correlation:.2f}")
                    col4.metric("P-Value", f"{p_value:.3f}",
                                delta="Cointegrated ✓" if cointegrated else "Not cointegrated",
                                delta_color="normal" if cointegrated else "inverse")
                    col5.metric("Hedge Ratio", f"{hedge_ratio:.4f}",
                                help=f"Hold {hedge_ratio:.2f} units of {sym2} per 1 unit of {sym1}")

                    if signal:
                        direction = "LONG" if "LONG_1" in signal else "SHORT"
                        st.success(f"**SIGNAL: {signal.replace('_', ' ')}** — {pair.get('signal_reason', '')}")
                    elif not cointegrated:
                        st.warning("Pair not cointegrated — no statistical basis for mean reversion trade")
                    else:
                        st.info(f"Cointegrated but z-score {zscore:+.2f} not extreme enough. Entry threshold: ±{pair.get('zscore_entry', 2.0)}")

        st.divider()
        st.subheader("How Pairs Trading Works")
        with st.expander("Click to learn the mechanics"):
            st.markdown("""
**The core idea:** Two stocks that have moved together historically will continue to do so.
When they temporarily diverge, the spread will revert — giving you a profit opportunity.

**Cointegration vs Correlation:**
- *Correlation* = they move in the same direction (e.g. both go up on good days)
- *Cointegration* = their SPREAD is stationary (it reverts to a mean)
- You need cointegration, not just correlation. Two trending stocks can be highly correlated without their spread ever reverting.

**The Z-Score:**
- Measures how many standard deviations the current spread is from its historical mean
- Z > +2.0 → spread is unusually wide → stock 1 is expensive vs stock 2 → SHORT stock 1, LONG stock 2
- Z < -2.0 → spread is unusually narrow → stock 1 is cheap vs stock 2 → LONG stock 1, SHORT stock 2
- |Z| < 0.5 → spread has converged → close the trade

**The Hedge Ratio:**
The number of shares of stock 2 to hold per share of stock 1.
Calculated via OLS regression so the pair is dollar-neutral.
A hedge ratio of 1.3 means: for every 1 share of AAPL, hold 1.3 shares of MSFT (short).

**Half-Life:**
How long it typically takes the spread to revert halfway to its mean.
A 5-day half-life means you expect to hold for ~10 days on average.
Shorter half-life = faster mean reversion = better for trading.
            """)


# =============================================================================
# TAB 5: NEWS SENTIMENT
# =============================================================================
with sentiment_tab:
    st.header("News Sentiment Analysis")
    st.caption(
        "Sentiment analysis scores financial headlines from -1 (most bearish) to +1 (most bullish). "
        "Academic research shows negative news sentiment precedes stock declines by 1-3 days."
    )

    if not SENTIMENT_AVAILABLE:
        st.error("sentiment.py failed to import. Run: pip install nltk")
    else:
        with st.spinner("Fetching and scoring headlines for all symbols..."):
            try:
                sentiment_data = sentiment_module.analyze_all_symbols_sentiment(config.SYMBOLS)
                summary = sentiment_module.get_sentiment_summary(config.SYMBOLS)
            except Exception as e:
                sentiment_data = {}
                summary = {}
                st.error(f"Sentiment error: {e}")

        # Market-wide sentiment bar
        if summary:
            overall = summary.get('overall_market_sentiment', 0)
            most_bullish = summary.get('most_bullish', '—')
            most_bearish = summary.get('most_bearish', '—')

            s1, s2, s3 = st.columns(3)
            s1.metric("Overall Market Sentiment", f"{overall:+.3f}",
                      delta="BULLISH" if overall > 0.05 else ("BEARISH" if overall < -0.05 else "NEUTRAL"),
                      delta_color="normal" if overall > 0.05 else ("inverse" if overall < -0.05 else "off"))
            s2.metric("Most Bullish Symbol", most_bullish)
            s3.metric("Most Bearish Symbol", most_bearish)

        st.divider()

        # Per-symbol sentiment
        for symbol in config.SYMBOLS:
            data = sentiment_data.get(symbol, {})
            if not data:
                continue

            avg_compound = data.get('avg_compound', 0) or 0
            label = data.get('sentiment_label', 'NEUTRAL')
            articles = data.get('article_count', 0)
            headlines = data.get('headlines', [])

            label_color = "green" if label == "BULLISH" else ("red" if label == "BEARISH" else "gray")

            with st.expander(f"**{symbol}** — Sentiment: :{label_color}[{label}] ({avg_compound:+.3f}) | {articles} articles"):
                if headlines:
                    for h in headlines[:10]:
                        compound = h.get('compound', 0)
                        title = h.get('title', '')
                        bar = "🟢" if compound > 0.05 else ("🔴" if compound < -0.05 else "⚪")
                        st.write(f"{bar} `{compound:+.2f}` {title}")
                else:
                    st.write("No recent headlines found for this symbol.")

        st.divider()
        st.subheader("How Sentiment Analysis Works")
        with st.expander("Learn about VADER NLP"):
            st.markdown("""
**VADER** (Valence Aware Dictionary and sEntiment Reasoner) was built specifically for
short social media and news text. Unlike generic NLP, it understands:
- Capitalization ("GREAT" scores higher than "great")
- Punctuation ("Great!!!" scores higher than "Great.")
- Negation ("not good" correctly scores negative)
- Finance-specific amplifiers

**Compound Score:**
- Above +0.05 = Positive (BULLISH)
- Below -0.05 = Negative (BEARISH)
- In between = Neutral

**How to use it:**
Watch for sudden swings in sentiment score for a symbol. A stock that's been
neutral for weeks suddenly going to -0.5 compound often precedes a down move.
Cross-reference with your strategy signals — a BUY signal on a stock with strongly
bearish sentiment is a lower-conviction trade.
            """)


# =============================================================================
# TAB 6: SEC / EDGAR RESEARCH
# =============================================================================
with edgar_tab:
    st.header("SEC EDGAR Research")
    st.caption(
        "Direct access to SEC filings: 8-K material events, insider buying/selling (Form 4), "
        "and institutional holdings (13F). All data is public and free from EDGAR."
    )

    if not EDGAR_AVAILABLE:
        st.error("edgar.py failed to import.")
    else:
        # Auto-load EDGAR data once per session — avoids tab-reset bug from button clicks
        # Data is cached in session_state so it only fetches once, not on every rerun
        if 'edgar_data' not in st.session_state:
            with st.spinner("Loading SEC filings... (30-60 seconds, respecting SEC rate limits)"):
                try:
                    st.session_state['edgar_data'] = edgar_module.get_all_edgar_data(config.SYMBOLS)
                except Exception as e:
                    st.error(f"EDGAR fetch error: {e}")
                    st.session_state['edgar_data'] = None

        # Manual refresh button — uses a key so it doesn't reset tabs
        if st.button("Refresh EDGAR Data", key="edgar_refresh"):
            with st.spinner("Refreshing SEC filings..."):
                try:
                    st.session_state['edgar_data'] = edgar_module.get_all_edgar_data(config.SYMBOLS)
                except Exception as e:
                    st.error(f"EDGAR fetch error: {e}")

        edgar_data = st.session_state.get('edgar_data', None)

        if edgar_data:
            # --- 8-K Filings ---
            st.subheader("Recent 8-K Filings — Material Events")
            st.caption("8-K = current report. Filed within 4 business days of any material event (earnings, lawsuits, CEO changes, mergers).")

            filings_8k = edgar_data.get('filings_8k', {})
            any_filings = False
            for symbol, filings in filings_8k.items():
                if filings:
                    any_filings = True
                    with st.expander(f"**{symbol}** — {len(filings)} recent filing(s)"):
                        for f in filings:
                            sentiment = f.get('keyword_sentiment', 'NEUTRAL')
                            s_color = "green" if sentiment == "BULLISH" else ("red" if sentiment == "BEARISH" else "gray")
                            st.write(f"📄 **{f.get('filing_date', '')}** — :{s_color}[{sentiment}]")
                            st.write(f"   {f.get('description', 'No description')}")
                            bullish_hits = f.get('bullish_keywords', [])
                            bearish_hits = f.get('bearish_keywords', [])
                            if bullish_hits:
                                st.write(f"   🟢 Bullish keywords: {', '.join(bullish_hits)}")
                            if bearish_hits:
                                st.write(f"   🔴 Bearish keywords: {', '.join(bearish_hits)}")
            if not any_filings:
                st.info("No 8-K filings in the last 30 days for tracked symbols.")

            st.divider()

            # --- Insider Transactions ---
            st.subheader("Insider Transactions — Form 4")
            st.caption(
                "Form 4 = insider trading report. Filed within 2 days of any executive/director buy or sell. "
                "Insider BUYING is one of the strongest bullish signals in finance."
            )

            insider_data = edgar_data.get('insider_transactions', {})
            insider_sentiment = edgar_data.get('insider_sentiment', {})

            for symbol in ['AAPL', 'MSFT']:
                transactions = insider_data.get(symbol, [])
                sentiment_info = insider_sentiment.get(symbol, {})

                with st.expander(f"**{symbol}** Insider Activity — {sentiment_info.get('signal', 'NEUTRAL')}"):
                    if sentiment_info:
                        ins1, ins2, ins3 = st.columns(3)
                        ins1.metric("Insider Signal", sentiment_info.get('signal', '—'))
                        ins2.metric("Buy Value (30d)", f"${sentiment_info.get('buy_value', 0):,.0f}")
                        ins3.metric("Sell Value (30d)", f"${sentiment_info.get('sell_value', 0):,.0f}")
                        if sentiment_info.get('key_insight'):
                            st.info(sentiment_info['key_insight'])

                    if transactions:
                        tx_df = pd.DataFrame(transactions)
                        for col in ['total_value', 'price_per_share', 'shares']:
                            if col in tx_df.columns:
                                tx_df[col] = pd.to_numeric(tx_df[col], errors='coerce')
                        st.dataframe(tx_df, use_container_width=True, hide_index=True)
                    else:
                        st.write("No recent Form 4 filings found.")

            st.divider()

            # --- Berkshire 13F ---
            st.subheader("Berkshire Hathaway 13F — Latest Holdings")
            st.caption(
                "Warren Buffett's portfolio, filed quarterly with the SEC. "
                "New positions signal deep value conviction. Exits signal concerns. "
                "45-day filing lag — these are positions as of last quarter end."
            )

            bh_holdings = edgar_data.get('berkshire_holdings', [])
            if bh_holdings:
                bh_df = pd.DataFrame(bh_holdings)
                if 'value_usd' in bh_df.columns:
                    bh_df['value_usd'] = bh_df['value_usd'].apply(lambda x: f"${x/1e9:.2f}B" if x >= 1e9 else f"${x/1e6:.0f}M")
                if 'pct_of_portfolio' in bh_df.columns:
                    bh_df['pct_of_portfolio'] = bh_df['pct_of_portfolio'].apply(lambda x: f"{x:.1f}%")
                st.dataframe(bh_df, use_container_width=True, hide_index=True)
            else:
                st.info("Click 'Fetch EDGAR Data' to load Berkshire's latest 13F holdings.")

            st.divider()
            st.subheader("Why SEC Filings Matter for Trading")
            with st.expander("Learn about EDGAR as an alpha source"):
                st.markdown("""
**8-K Filings** appear on EDGAR before most news services pick them up.
Reading filings directly gives you an information edge over traders who wait for headlines.
Material events in 8-Ks regularly cause 5-20% moves.

**Form 4 Insider Buying** is one of the most researched alpha signals:
- Academic studies show insider buying predicts 6-8% annualized outperformance
- The signal is strongest when: CEO/CFO buys (not just a director), multiple insiders buy simultaneously,
  the purchase is large relative to their salary, and the stock has recently underperformed

**Insider Selling** is much less informative:
- Executives sell for many reasons: diversification, a new house, taxes, divorce
- Only cluster selling (multiple insiders selling simultaneously) is a reliable bearish signal

**13F Holdings** show institutional conviction but with a 45-day lag:
- Best used to understand what smart money thinks, not for timing
- A new Berkshire position validates your own thesis — confirmation, not alpha
                """)
        else:
            st.info("Click **'Fetch EDGAR Data'** above to load SEC research. Takes ~30-60 seconds due to SEC rate limits.")


# =============================================================================
# TAB 7: FACTOR INVESTING
# =============================================================================

with factors_tab:
    st.header("📐 Factor Investing")
    st.caption(
        "Momentum · Value · Quality · Low Volatility — the four most research-backed "
        "return predictors in academic finance. Used by AQR, Dimensional Fund Advisors, "
        "and most systematic hedge funds."
    )

    if not FACTORS_AVAILABLE:
        err = FACTORS_IMPORT_ERROR or "factors.py not found"
        st.error(f"Factor Investing unavailable: {err}")
    else:
        # Factor explainer
        with st.expander("What is Factor Investing? (Click to learn)"):
            st.markdown("""
**Factor investing** is the systematic pursuit of return premiums that have been documented
across 50+ years of academic research and live trading.

| Factor | What It Measures | Why It Works |
|--------|-----------------|--------------|
| **Momentum** | 12-1 month return | Markets underreact to good news; trends persist |
| **Value** | P/E ratio (inverse) | Investors overpay for "exciting" growth stocks |
| **Quality** | ROE, low debt, margins | High-quality companies trade at structural discounts |
| **Low Vol** | Realized volatility | Institutions overweight risky stocks, leaving low-vol underpriced |

**Signal logic:** 2+ factors agreeing = BUY or SELL. 3+ = STRONG signal.
Only composite multi-factor signals are high conviction — single-factor signals are noise.

**Note:** Value and Quality only apply to equities (AAPL, MSFT, ASML.AS, SAP.DE).
Momentum and Low Vol apply to all asset classes including crypto and forex.
            """)

        factor_symbols = [s for s in config.SYMBOLS
                          if not s.endswith('=X')]  # Exclude pure forex for cleaner display

        col_btn, col_select = st.columns([1, 2])
        with col_btn:
            run_factors = st.button("🔍 Run Factor Analysis", type="primary",
                                    help="Takes 30-60 seconds — fetches data from Yahoo Finance")
        with col_select:
            rank_by = st.selectbox("Rank symbols by:",
                                   ['composite', 'momentum', 'value', 'quality', 'low_vol'],
                                   key='factor_rank_by')

        if run_factors or 'factor_results' in st.session_state:
            if run_factors:
                with st.spinner("Running factor analysis across all symbols..."):
                    try:
                        st.session_state['factor_results'] = factors_module.get_factor_signals(factor_symbols)
                        st.session_state['factor_ranking'] = factors_module.rank_symbols_by_factor(
                            factor_symbols, factor=rank_by
                        )
                    except Exception as e:
                        st.error(f"Factor analysis error: {e}")

            factor_results = st.session_state.get('factor_results', {})
            factor_ranking = st.session_state.get('factor_ranking', [])

            if factor_results:
                # --- Composite Signal Summary ---
                st.subheader("Composite Factor Signals")

                signal_rows = []
                for sym, data in factor_results.items():
                    composite_sig = data.get('composite_signal') or 'NEUTRAL'
                    buy_v = data.get('buy_votes', 0)
                    sell_v = data.get('sell_votes', 0)

                    # Individual factor scores
                    mom = data.get('momentum')
                    val = data.get('value')
                    qual = data.get('quality')
                    lv = data.get('low_vol')

                    signal_rows.append({
                        'Symbol': sym,
                        'Signal': composite_sig,
                        'Buy Votes': buy_v,
                        'Sell Votes': sell_v,
                        'Momentum': f"{mom.get('momentum_return', 0)*100:+.1f}%" if mom else '—',
                        'Value Score': f"{val.get('value_score', 0):.2f}" if val else '—',
                        'Quality': f"{qual.get('quality_score', 0):.0f}/100" if qual else '—',
                        'Low Vol': f"{lv.get('vol_percentile', 0):.2f}" if lv else '—',
                        'Reasoning': data.get('composite_reason', '')[:80],
                    })

                sig_df = pd.DataFrame(signal_rows)

                # Color-code signals
                def color_signal(val):
                    if 'STRONG_BUY' in str(val):
                        return 'background-color: #0a4f0a; color: #00ff88'
                    elif 'BUY' in str(val):
                        return 'background-color: #1a3d1a; color: #88ff88'
                    elif 'STRONG_SELL' in str(val):
                        return 'background-color: #4f0a0a; color: #ff8888'
                    elif 'SELL' in str(val):
                        return 'background-color: #3d1a1a; color: #ffaaaa'
                    return ''

                styled = sig_df.style.applymap(color_signal, subset=['Signal'])
                st.dataframe(styled, use_container_width=True, hide_index=True)

                # --- Factor Rankings ---
                if factor_ranking:
                    st.subheader(f"Symbol Rankings by {rank_by.title()}")
                    rank_rows = []
                    for item in factor_ranking:
                        rank_rows.append({
                            'Rank': item['rank'],
                            'Symbol': item['symbol'],
                            'Score': round(item['score'], 4),
                            'Designation': item.get('designation', '—'),
                        })

                    rank_df = pd.DataFrame(rank_rows)

                    def color_designation(val):
                        if val == 'LONG':
                            return 'color: #00ff88; font-weight: bold'
                        elif val == 'SHORT':
                            return 'color: #ff6666; font-weight: bold'
                        return 'color: #888888'

                    styled_rank = rank_df.style.applymap(color_designation, subset=['Designation'])
                    st.dataframe(styled_rank, use_container_width=True, hide_index=True)

                    st.caption(
                        "**LONG** = top 30% by factor (long book) | "
                        "**SHORT** = bottom 30% (short book) | "
                        "**NEUTRAL** = middle 40% (no edge)"
                    )

                # --- Deep Dive per Symbol ---
                st.subheader("Deep Dive by Symbol")
                selected_sym = st.selectbox("Select symbol for factor details:",
                                             list(factor_results.keys()),
                                             key='factor_deep_dive')

                if selected_sym and selected_sym in factor_results:
                    sym_data = factor_results[selected_sym]
                    c1, c2, c3, c4 = st.columns(4)

                    mom_data = sym_data.get('momentum')
                    val_data = sym_data.get('value')
                    qual_data = sym_data.get('quality')
                    lv_data = sym_data.get('low_vol')

                    with c1:
                        st.metric("Momentum Signal",
                                  mom_data.get('signal', 'N/A') if mom_data else 'N/A',
                                  f"{mom_data.get('momentum_return', 0)*100:+.1f}% 12-1mo" if mom_data else None)
                    with c2:
                        st.metric("Value Signal",
                                  val_data.get('signal', 'N/A') if val_data else 'N/A (non-equity)',
                                  f"P/E {val_data.get('pe_ratio', '—')}" if val_data else None)
                    with c3:
                        st.metric("Quality Signal",
                                  qual_data.get('signal', 'N/A') if qual_data else 'N/A (non-equity)',
                                  f"Score {qual_data.get('quality_score', '—')}/100" if qual_data else None)
                    with c4:
                        st.metric("Low-Vol Signal",
                                  lv_data.get('signal', 'N/A') if lv_data else 'N/A',
                                  f"Vol {lv_data.get('realized_vol', 0)*100:.1f}% ann." if lv_data else None)

                    st.info(f"**Composite reasoning:** {sym_data.get('composite_reason', '—')}")
        else:
            st.info("Click **'Run Factor Analysis'** above to score all 13 symbols. Takes ~45 seconds (fetches data from Yahoo Finance for each symbol).")
            st.markdown("**Symbols that will be analyzed:**")
            sym_df = pd.DataFrame([
                {'Symbol': s,
                 'Asset Class': ('Equity' if not any(s.endswith(x) for x in ['-USD','-USDT','=X','=F']) else
                                 'Crypto' if s.endswith('-USD') else
                                 'Forex' if s.endswith('=X') else 'Futures/Commodity'),
                 'Factors': ('All 4' if not any(s.endswith(x) for x in ['-USD','-USDT','=X','=F']) else 'Momentum + Low Vol')}
                for s in factor_symbols
            ])
            st.dataframe(sym_df, use_container_width=True, hide_index=True)


# =============================================================================
# TAB 8: EVENTS CALENDAR
# =============================================================================

with events_tab:
    st.header("📅 Events Calendar")
    st.caption(
        "Upcoming earnings, dividends, and corporate actions. "
        "The engine automatically blocks new trades into known binary events."
    )

    if not EVENTS_AVAILABLE:
        err = EVENTS_IMPORT_ERROR or "events.py not found"
        st.error(f"Events Calendar unavailable: {err}")
    else:
        with st.expander("Why Events Matter for Trading (Click to learn)"):
            st.markdown("""
**Earnings Announcements** are the most important scheduled events in equity markets:
- Stocks can gap 5-20% overnight in either direction
- Trading into earnings is essentially a coin flip — even if you're right on fundamentals, the stock can "sell the news"
- **IV Crush**: Options premiums collapse after earnings regardless of direction — options buyers often lose even when the stock moves their way

**The engine blocks new positions within 3 days of earnings** to avoid binary event risk.

**Dividend Ex-Dates** create mechanical price adjustments:
- Stock price drops by ~dividend amount at ex-date open
- Shorting before ex-date means you owe the dividend (increased cost)
- The engine blocks new SHORT positions within 2 days of an ex-dividend date

**Post-Earnings Drift (PEAD)**: After a large earnings surprise, stocks tend to continue drifting in the surprise direction for 30-60 days. This is one of the most replicated anomalies in finance. Future version: auto-generate signals based on earnings surprise magnitude.
            """)

        col_e1, col_e2 = st.columns([1, 3])
        with col_e1:
            fetch_events = st.button("🔄 Refresh Events", type="primary")

        # Auto-load on first visit; refresh button forces a re-fetch
        if 'events_data' not in st.session_state or fetch_events:
            with st.spinner("Fetching earnings and dividend calendars..."):
                try:
                    st.session_state['events_data'] = events_module.get_all_events(config.SYMBOLS)
                except Exception as e:
                    st.error(f"Events fetch error: {e}")

        if 'events_data' in st.session_state:
            events_data = st.session_state.get('events_data', {})
            if events_data:
                high_risk = events_data.get('high_risk_symbols', [])
                if high_risk:
                    st.warning(f"⚠️ High-risk event symbols (next 5 days): **{', '.join(high_risk)}** — engine will block new trades.")

                # --- Earnings ---
                # events_data['earnings'] = {symbol: {next_earnings, days_until_earnings, earnings_risk, ...}}
                st.subheader("Upcoming Earnings Announcements")
                earnings_dict = events_data.get('earnings', {})
                earnings_with_dates = [v for v in earnings_dict.values()
                                       if v.get('next_earnings') is not None]
                if earnings_with_dates:
                    e_rows = []
                    for ev in sorted(earnings_with_dates, key=lambda x: x.get('days_until_earnings', 999)):
                        e_rows.append({
                            'Symbol': ev['symbol'],
                            'Date': ev.get('next_earnings', '—'),
                            'Days Until': ev.get('days_until_earnings', '—'),
                            'Risk Level': ev.get('earnings_risk', '—'),
                            'Recommendation': ev.get('recommendation', '')[:80],
                        })
                    e_df = pd.DataFrame(e_rows)

                    def color_risk(val):
                        if val == 'HIGH':
                            return 'background-color: #5c0000; color: #ff4444; font-weight: bold'
                        elif val == 'MEDIUM':
                            return 'background-color: #3d1a00; color: #ff8800'
                        elif val == 'LOW':
                            return 'background-color: #3d3d00; color: #ffff44'
                        return ''

                    st.dataframe(e_df.style.applymap(color_risk, subset=['Risk Level']),
                                 use_container_width=True, hide_index=True)
                else:
                    st.success("No upcoming earnings found for tracked symbols (or none scheduled in yfinance).")

                # --- IV Crush Warnings ---
                try:
                    iv_warnings = events_module.get_iv_crush_warnings(config.SYMBOLS)
                    if iv_warnings:
                        st.subheader("⚡ IV Crush Risk Warnings (Options)")
                        for w in iv_warnings:
                            st.warning(w['warning'])
                except Exception:
                    pass

                # --- Dividends ---
                # events_data['dividends'] = {symbol: {next_ex_dividend, dividend_amount, days_until_ex_div}}
                st.subheader("Upcoming Dividend Ex-Dates")
                dividends_dict = events_data.get('dividends', {})
                divs_with_dates = [v for v in dividends_dict.values()
                                   if v.get('next_ex_dividend') is not None]
                if divs_with_dates:
                    d_rows = []
                    for ev in sorted(divs_with_dates, key=lambda x: x.get('days_until_ex_div', 999)):
                        d_rows.append({
                            'Symbol': ev['symbol'],
                            'Ex-Date': ev.get('next_ex_dividend', '—'),
                            'Days Until': ev.get('days_until_ex_div', '—'),
                            'Quarterly Dividend': f"${ev.get('dividend_amount', 0):.4f}" if ev.get('dividend_amount') else '—',
                        })
                    st.dataframe(pd.DataFrame(d_rows), use_container_width=True, hide_index=True)
                else:
                    st.info("No upcoming dividend ex-dates found for tracked symbols.")

                last_updated = events_data.get('last_updated', '')
                if last_updated:
                    st.caption(f"Last updated: {last_updated[:19]}")
