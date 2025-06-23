import pandas as pd
import numpy as np
import datetime
import time
import pytz
from ib_insync import *

class SPYREVStrategy:
    """RSI Reversal strategy for SPY 0-DTE options with Keltner Channel-based stop loss."""

    def __init__(self, **kwargs):
        self.ticker = kwargs.get("ticker", "SPY")
        self.contracts = kwargs.get("contracts", 2)
        self.market_open = kwargs.get("market_open", "08:30:00")
        self.market_close = kwargs.get("market_close", "15:00:00")
        self.force_close_time = kwargs.get("force_close_time", "14:55:00")
        self.bar_size = kwargs.get("bar_size", "5 mins")
        self.rsi_period = kwargs.get("rsi_period", 14)
        self.ema_period = kwargs.get("ema_period", 9)
        self.rsi_oversold = kwargs.get("rsi_oversold", 30.0)
        self.rsi_overbought = kwargs.get("rsi_overbought", 70.0)
        self.paper_trading = kwargs.get("paper_trading", True)
        self.port = kwargs.get("port", 7497)
        
        # New partial sell targets
        self.partial_sell_targets = kwargs.get("partial_sell_targets", [1.00, 2.00])  # Example targets: $1.00 and $2.00
        
        # Trading state
        self.positions = []  # List of active positions
        self.half_position_closed = False  # Track if half position is closed
        
        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    def place_order(self, contract, action: str, qty: int):
        """Place an order."""
        order = MarketOrder(action, qty)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        print(f"{datetime.datetime.now(self.tz)} - {action} {qty} {contract.localSymbol}")
        return trade

    def monitor_price_move(self, underlying_price):
        """Monitor price move to trigger partial sell conditions."""
        if self.position == 0:
            return  # No position to monitor

        # Check for partial sell conditions based on underlying price movement
        price_movement = underlying_price - self.entry_underlying_price
        for target in self.partial_sell_targets:
            if price_movement >= target and target not in self.sold_targets:
                self.sell_partial_position(target)
                self.sold_targets.append(target)  # Ensure we only sell once per target

    def sell_partial_position(self, target):
        """Sell half of the position based on target."""
        qty_to_sell = self.contracts // 2
        action = "SELL"
        print(f"Selling {qty_to_sell} contracts at {target} price move target")
        self.place_order(self.option_contract, action, qty_to_sell)

    def run(self):
        if not self.connect_to_ib():
            return

        try:
            print("Starting SPY REV strategy...")
            while True:
                df = self.get_intraday_5min()
                if df is None or len(df) < self.rsi_period + self.ema_period:
                    time.sleep(30)
                    continue

                # Calculate RSI and 9 EMA
                df = self.calculate_indicators(df)
                
                # Calculate Keltner Channels
                df = self.calculate_keltner_channels(df)

                last_candle = df.iloc[-1]
                
                # Monitor price for partial sell conditions
                self.monitor_price_move(self.get_underlying_price())

                # Manage positions with stop loss and breakeven logic
                for position in self.positions[:]:
                    # Example: Adding stop loss logic here
                    if self.check_initial_stop_loss(position, last_candle):
                        self.exit_position(position, "Initial Stop Loss")
                        continue
                    
                    option_price = self.ib.reqTickers(position['contract'])[0].marketPrice()
                    if self.check_breakeven_stop_loss(position, option_price):
                        self.exit_position(position, "Breakeven Stop Loss")

                # Force close at 2:55 PM
                self.check_force_close()

                time.sleep(5)

        except KeyboardInterrupt:
            print("User interrupted - shutting down.")
        except Exception as exc:
            print(f"Unhandled error: {exc}")
        finally:
            self.ib.disconnect()
            print("Disconnected from Interactive Brokers.")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPY REV (Reversal) trading strategy")
    parser.add_argument("--ticker", type=str, default="SPY", help="Underlying ticker symbol")
    parser.add_argument("--contracts", type=int, default=2, help="Number of option contracts to trade")
    parser.add_argument("--paper_trading", action="store_true", help="Use paper trading account")
    parser.add_argument("--port", type=int, default=7497, help="Port number")
    parser.add_argument("--partial_sell_targets", type=float, nargs='+', default=[1.00, 2.00], help="Partial sell targets based on price movement")
    args = parser.parse_args()

    strategy = SPYREVStrategy(
        ticker=args.ticker,
        contracts=args.contracts,
        paper_trading=args.paper_trading,
        port=args.port,
        partial_sell_targets=args.partial_sell_targets  # New logic for partial sell targets
    )
    strategy.run()
