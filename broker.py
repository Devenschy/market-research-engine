# =============================================================================
# broker.py — Paper Trading Order Execution
# =============================================================================
# WHY: Isolating order execution into its own module means:
# 1. You can swap from paper trading to a real broker API by changing only this file
# 2. The engine never needs to know HOW orders are executed, just that they were
# 3. In a real system, this would connect to Alpaca, Interactive Brokers, or similar
#
# In paper trading mode, orders are executed instantly at the current market price.
# This is optimistic — real execution involves slippage, spreads, and latency.
# A more sophisticated version would simulate these market microstructure effects.

from datetime import datetime
from risk import Position
import config


class PaperBroker:
    """
    Simulated broker for paper trading.

    WHY: Paper trading with a simulated broker lets you test strategy logic
    and risk controls without financial risk. The discipline of treating paper
    trades as real — not changing rules mid-trade, honoring stop-losses —
    is what makes the simulation valuable for learning.

    LIMITATION: Paper trading assumes instant fill at quoted price.
    Real trading has:
    - Bid-ask spread: You buy at ask, sell at bid (hidden cost)
    - Slippage: Large orders move the market against you
    - Latency: Time between signal and fill, price may move
    For learning purposes, these simplifications are acceptable.
    """

    def __init__(self):
        self.order_history = []

    def submit_order(self, symbol: str, direction: str, quantity: float,
                     price: float, order_type: str = 'MARKET') -> dict:
        """
        Submit a paper trading order — fills instantly at market price.

        WHY: Market orders are used because we are not optimizing for execution
        quality — we are testing signal logic. In live trading, limit orders
        would reduce slippage costs.

        Returns an order record dict for logging.
        """
        if not config.PAPER_TRADING:
            raise RuntimeError("Live trading not implemented — PAPER_TRADING must be True")

        order = {
            'order_id': f"PAPER_{len(self.order_history)+1:05d}",
            'symbol': symbol,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'order_type': order_type,
            'status': 'FILLED',
            'fill_price': price,   # Paper trading: filled exactly at current price
            'timestamp': datetime.now().isoformat(),
            'paper_trade': True
        }

        self.order_history.append(order)
        return order

    def cancel_order(self, order_id: str) -> bool:
        """
        In paper trading, orders fill instantly so cancellation is not applicable.
        This method exists to make the interface compatible with a real broker.
        """
        return False  # Can't cancel instantly-filled paper orders
