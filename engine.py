# =============================================================================
# engine.py — Main Trading Loop
# =============================================================================
# WHY: The engine is the orchestrator — it coordinates all other modules without
# implementing any logic itself. This is the 'single responsibility' principle:
# the engine knows WHEN to call things and in WHAT ORDER, but not HOW they work.
#
# The order of operations matters:
# 1. Fetch prices (you need current data before anything else)
# 2. Update equity (know your current financial state)
# 3. Check exits (close positions before opening new ones — capital efficiency)
# 4. Detect regimes (know the environment before generating signals)
# 5. Generate + filter signals (strategies + majority vote + regime filter)
# 6. Check risk limits (validate before executing)
# 7. Execute orders (the output of all the above)
# 8. Log + display (record and present what happened)

import time
from datetime import datetime
from collections import defaultdict

import config
import data as data_module
from strategies import StrategyEnsemble
from regime import detect_regime, filter_signal_by_regime, Regime
from risk import RiskManager
from broker import PaperBroker
import logger as logger_module
import dashboard


class TradingEngine:
    """
    The main engine that orchestrates the entire trading system.

    WHY: Keeping all orchestration in one class makes the system's flow
    readable top-to-bottom. A new engineer can read run_tick() and understand
    the entire system in 5 minutes. This is what 'explainable' means in practice.
    """

    def __init__(self):
        # Initialize strategy ensembles — one per symbol
        # WHY: Each symbol needs its own independent strategy state.
        # AAPL's MA crossover is completely independent from BTC-USD's.
        self.ensembles = {symbol: StrategyEnsemble() for symbol in config.SYMBOLS}

        # Regime state — updated every tick for each symbol
        self.regimes = {}

        # Risk manager — single instance tracks all positions and capital
        self.risk = RiskManager()

        # Paper broker — executes simulated orders
        self.broker = PaperBroker()

        # Price history for regime detection (OHLCV DataFrames)
        self.price_history = {}

        # Alternative data (fetched less frequently — every 5 ticks)
        self.alt_data = {}
        self.alt_data_tick = 0
        self.ALT_DATA_INTERVAL = 5   # Fetch alt data every 5 ticks

        # Current live prices
        self.current_prices = {}

        self.tick_count = 0
        self.is_initialized = False

    def initialize(self):
        """
        Warm up all strategies with historical data before live trading.

        WHY: Strategies need historical data to initialize their indicators.
        A 30-period MA can't generate a signal until it has 30 price points.
        The warm-up feeds historical OHLCV data to each strategy so all
        indicators are 'ready' from the first live tick.

        This is analogous to how a quant fund backtests before going live.
        """
        print("[engine] Warming up strategies with historical data...")

        for symbol in config.SYMBOLS:
            print(f"[engine] Loading history for {symbol}...")
            df = data_module.fetch_history(
                symbol,
                period=config.WARMUP_PERIOD,
                interval=config.WARMUP_INTERVAL
            )

            if df.empty:
                print(f"[engine] Warning: No history for {symbol}, skipping warm-up")
                continue

            self.price_history[symbol] = df

            # Feed historical closes to strategy ensemble
            ensemble = self.ensembles[symbol]
            for price in df['Close'].values:
                ensemble.update(price)

            # Calculate initial regime from history
            self.regimes[symbol] = detect_regime(df)
            regime_name = self.regimes[symbol]['regime'].value
            print(f"[engine] {symbol}: regime={regime_name}, "
                  f"adx={self.regimes[symbol]['adx']}, "
                  f"ready={ensemble.is_ready()}")

        # Load initial alt data
        print("[engine] Fetching alternative data...")
        self.alt_data = data_module.get_alt_data_summary()

        self.is_initialized = True
        print("[engine] Initialization complete. Starting live trading loop...")

    def run_tick(self):
        """
        Execute one complete trading cycle. Called every POLL_INTERVAL_SECONDS.

        This is the heart of the system — everything flows through here.
        Read this method first when trying to understand the system as a whole.
        """
        self.tick_count += 1

        # STEP 1: Fetch current prices for all symbols
        # WHY: All subsequent steps depend on having fresh price data.
        self.current_prices = data_module.fetch_prices(config.SYMBOLS)
        if not self.current_prices:
            print("[engine] No prices fetched this tick — API issue")
            return

        # STEP 2: Update equity with current mark-to-market prices
        # WHY: We need to know current equity before checking the kill switch.
        # Unrealized losses count against the daily drawdown limit.
        self.risk.update_equity(self.current_prices)

        # STEP 3: Check open positions for SL/TP exits
        # WHY: Exit management happens BEFORE signal generation. This ensures
        # we don't generate new signals while exiting old positions.
        closed_positions = self.risk.check_exits(self.current_prices)
        for closed in closed_positions:
            logger_module.log_trade(closed)
            print(f"[engine] CLOSED {closed['symbol']} {closed['exit_reason']}: "
                  f"PnL {closed['pnl']:+.2f} ({closed['pnl_pct']:+.2f}%)")

        # STEP 4: Refresh alt data periodically (not every tick — API rate limits)
        if self.tick_count % self.ALT_DATA_INTERVAL == 0:
            self.alt_data = data_module.get_alt_data_summary()

        # STEP 5: Process each symbol
        for symbol in config.SYMBOLS:
            if symbol not in self.current_prices:
                continue

            price = self.current_prices[symbol]
            self._process_symbol(symbol, price)

        # STEP 6: Update performance log every 10 ticks
        if self.tick_count % 10 == 0:
            logger_module.update_performance(self.risk)

        # STEP 7: Refresh dashboard
        stats = self.risk.get_stats()
        dashboard.render_dashboard(
            stats=stats,
            open_positions=self.risk.open_positions,
            prices=self.current_prices,
            regimes=self.regimes,
            alt_data=self.alt_data,
            tick_count=self.tick_count
        )

    def _process_symbol(self, symbol: str, price: float):
        """
        Process one symbol through the full signal pipeline.

        Pipeline:
        1. Update regime detection with latest history
        2. Update strategy ensemble with new price
        3. Get votes from all strategies
        4. Apply majority vote filter (2/3 required)
        5. Apply regime filter (suppress mismatched signals)
        6. Check risk limits
        7. Execute if all checks pass
        """
        ensemble = self.ensembles[symbol]

        # Update price history for regime detection
        # WHY: We append the latest close to maintain a rolling window.
        # Full OHLCV is needed for regime detection (ATR, ADX require high/low).
        import pandas as pd
        new_row = pd.DataFrame([{
            'Open': price, 'High': price, 'Low': price,
            'Close': price, 'Volume': 0
        }])

        if symbol in self.price_history:
            self.price_history[symbol] = pd.concat(
                [self.price_history[symbol], new_row], ignore_index=True
            ).tail(500)  # Keep last 500 candles
        else:
            self.price_history[symbol] = new_row

        # Recalculate regime
        self.regimes[symbol] = detect_regime(self.price_history[symbol])
        regime_info = self.regimes[symbol]
        regime = regime_info['regime']
        regime_name = regime.value

        # Update strategy ensemble
        ensemble.update(price)

        if not ensemble.is_ready():
            return  # Not enough data yet

        # Get individual strategy votes
        votes = ensemble.get_votes()

        # Log all individual signals for analysis
        for strategy_name, vote in votes.items():
            if vote is not None:
                logger_module.log_signal(
                    symbol=symbol,
                    strategy=strategy_name,
                    signal=vote,
                    price=price,
                    regime=regime_name,
                    triggered_trade=False,  # Will update if trade executes
                    suppression_reason='individual_vote',
                    votes=votes
                )

        # Apply majority vote (2/3 strategies must agree)
        # WHY: A single strategy firing is not enough conviction to trade.
        # We need directional agreement across multiple, independent signals.
        aggregate_signal = ensemble.aggregate_signal()

        if aggregate_signal is None:
            return  # No majority agreement

        # Determine which strategy "owns" this signal for attribution
        dominant_strategy = self._get_dominant_strategy(votes, aggregate_signal)

        # Apply regime filter — suppress signals mismatched to regime
        # WHY: Even with majority agreement, if the signal conflicts with the
        # regime, the probability of success drops significantly.
        filtered_signal = filter_signal_by_regime(
            aggregate_signal, dominant_strategy, regime
        )

        if filtered_signal is None:
            suppression = f"Suppressed by {regime_name} regime"
            dashboard.add_signal_event(symbol, dominant_strategy, aggregate_signal,
                                        price, regime_name, False, suppression)
            logger_module.log_signal(symbol, dominant_strategy, aggregate_signal,
                                      price, regime_name, False, suppression, votes)
            return

        # Check if we already have an open position
        if symbol in self.risk.open_positions:
            return  # One position per symbol — simplifies risk management

        # Map signal to trade direction
        direction = 'LONG' if filtered_signal == 'BUY' else 'SHORT'

        # Check risk limits
        can_open, reason = self.risk.can_open_position(symbol)
        if not can_open:
            dashboard.add_signal_event(symbol, dominant_strategy, filtered_signal,
                                        price, regime_name, False, reason)
            return

        # Execute the paper trade
        position = self.risk.open_position(
            symbol=symbol,
            price=price,
            direction=direction,
            strategy=dominant_strategy,
            regime=regime_name,
            votes=votes,
            regime_modifier=regime_info.get('position_size_modifier', 1.0)
        )

        if position:
            self.broker.submit_order(symbol, direction, position.quantity, price)
            dashboard.add_signal_event(symbol, dominant_strategy, filtered_signal,
                                        price, regime_name, True)
            logger_module.log_signal(symbol, dominant_strategy, filtered_signal,
                                      price, regime_name, True, None, votes)
            print(f"[engine] OPENED {direction} {symbol} @ {price:.4f} | "
                  f"Regime: {regime_name} | Strategy: {dominant_strategy}")

    def _get_dominant_strategy(self, votes: dict, signal: str) -> str:
        """
        Identify which strategy to attribute this signal to.

        WHY: For performance attribution, we need to know which strategy
        "led" the majority vote. We pick the strategy whose signal matches
        the aggregate, using a priority order based on regime appropriateness.
        Priority: MA Crossover > RSI > Mean Reversion (arbitrary tie-break).
        """
        priority = ['MA_Crossover', 'RSI_Momentum', 'Mean_Reversion']
        for strat in priority:
            if votes.get(strat) == signal:
                return strat
        return 'Ensemble'

    def run(self):
        """
        Main loop — runs forever until interrupted.

        WHY: Infinite loop with sleep is the simplest approach for a
        research-grade system. Production systems use event loops or
        scheduled tasks, but for learning, a simple loop is clearest.
        """
        if not self.is_initialized:
            self.initialize()

        print(f"[engine] Starting trading loop — polling every {config.POLL_INTERVAL_SECONDS}s")
        print("[engine] Press Ctrl+C to stop")

        while True:
            try:
                start_time = time.time()
                self.run_tick()
                elapsed = time.time() - start_time
                sleep_time = max(0, config.POLL_INTERVAL_SECONDS - elapsed)
                time.sleep(sleep_time)
            except KeyboardInterrupt:
                print("\n[engine] Shutting down gracefully...")
                logger_module.update_performance(self.risk)
                final_stats = self.risk.get_stats()
                print(f"[engine] Final equity: ${final_stats['equity']:,.2f}")
                print(f"[engine] Total return: {final_stats['total_return_pct']:+.2f}%")
                print(f"[engine] Total trades: {final_stats['total_trades']}")
                print(f"[engine] Win rate: {final_stats['win_rate']:.1f}%")
                break
            except Exception as e:
                print(f"[engine] Tick error: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(10)   # Brief pause before retrying
