#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPY REV CHAD: SPY Reversal Highly Automated Dealer
# An automated trading system that implements a 5-minute reversal strategy for SPY options.

import pandas as pd
import numpy as np
import datetime
import time
import pytz
import yaml
from ib_insync import *

class SPYREVStrategy:
    """RSI Reversal strategy for SPY 0-DTE options with Keltner Channel-based stop loss."""

    def __init__(self, config):
        # Load configuration from the YAML file
        self.ticker = config.get("ticker", "SPY")
        self.contracts = config.get("contracts", 2)
        self.market_open = config.get("market_open", "08:30:00")
        self.market_close = config.get("market_close", "15:00:00")
        self.force_close_time = config.get("force_close_time", "14:55:00")
        self.bar_size = config.get("bar_size", "5 mins")
        self.rsi_period = config.get("rsi_period", 14)
        self.ema_period = config.get("ema_period", 9)
        self.rsi_oversold = config.get("rsi_oversold", 30.0)
        self.rsi_overbought = config.get("rsi_overbought", 70.0)
        self.paper_trading = config.get("paper_trading", True)
        self.port = config.get("port", 7497)

        # Partial sell targets (loaded from YAML)
        self.partial_sell_targets = config.get("partial_sell_targets", [1.00, 2.00])  # Target price movement

        # Trading state - can have multiple positions
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
        if not self.positions:  # No positions to monitor
            return

        for position in self.positions[:]:
            # Calculate price movement from entry price
            price_movement = underlying_price - position['entry_underlying_price']
            
            for target in self.partial_sell_targets:
                if price_movement >= target and target not in position.get('sold_targets', []):
                    self.sell_partial_position(position, target)
                    position.setdefault('sold_targets', []).append(target)  # Track sold targets

    def sell_partial_position(self, position, target):
        """Sell half of the position based on target price movement."""
        qty_to_sell = position['contracts'] // 2
        action = "SELL"
        print(f"Selling {qty_to_sell} contracts at {target} price move target")
        self.place_order(position['contract'], action, qty_to_sell)

    def exit_all(self, reason: str):
        """Exit all positions based on a given reason."""
        print(f"Exiting all positions due to: {reason}")
        for position in self.positions[:]:
            self.exit_position(position, reason)

    def exit_position(self, position: dict, reason: str):
        """Exit a specific position."""
        if not position.get('half_sold', False):
            quantity = position['contracts'] // 2
            position['half_sold'] = True
            self.place_order(position['contract'], "SELL", quantity)
            self.half_position_closed = True
        else:
            quantity = position['contracts_remaining']
            self.place_order(position['contract'], "SELL", quantity)

        # Log the reason for exiting the position
        print(f"Exited {position['type']} position | Reason: {reason}")

        # Remove position from active positions list
        self.positions.remove(position)

    def is_force_close_time(self) -> bool:
        """Check if it's time to force close positions (2:55 PM)."""
        now = datetime.datetime.now(self.tz)
        force_close_time = self.tz.localize(
            datetime.datetime.combine(now.date(), datetime.datetime.strptime(self.force_close_time, "%H:%M:%S").time())
        )
        return now >= force_close_time

    # Main loop and strategy execution
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

                # Manage stop loss and breakeven conditions
                for position in self.positions[:]:
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

# Load configuration from YAML
def load_config():
    with open("config.yaml", 'r') as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    import argparse

    # Load config.yaml
    config = load_config()['spy_rev']

    strategy = SPYREVStrategy(config=config['args'])
    strategy.run()
