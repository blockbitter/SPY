#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPY ORB CHAD: SPY Opening Range Breakout Highly Automated Dealer
# An automated trading system that implements a 5-minute opening range breakout strategy for SPY options.

import pandas as pd
import numpy as np
import datetime
import time
import pytz
from ib_insync import *

class SPYORBStrategy:
    def __init__(
        self,
        ticker: str = "SPY",
        contracts: int = 2,
        market_open: str = "08:30:00",
        market_close: str = "15:00:00",
        force_close_time: str = "14:55:00",
        bar_size: str = "5 mins",
        paper_trading: bool = True,
        port: int = 7497,
        partial_sell_targets: list = [1.00, 2.00],  # New logic: targets for partial sell
    ):
        self.ticker = ticker
        self.contracts = contracts
        self.market_open = market_open
        self.market_close = market_close
        self.force_close_time = force_close_time
        self.bar_size = bar_size
        self.paper_trading = paper_trading
        self.port = port

        self.partial_sell_targets = partial_sell_targets  # List of price move targets for partial sell
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set = False

        self.position = None
        self.option_contract = None
        self.entry_underlying_price = None
        self.entry_option_price = None
        self.entry_strike = None
        self.sold_targets = []  # Track which partial sell targets have been hit

        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    def connect_to_ib(self, host: str = "127.0.0.1", client_id: int = 9, max_retries: int = 3) -> bool:
        port = self.port
        for attempt in range(1, max_retries + 1):
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
                    time.sleep(1)
                self.ib.connect(host, port, clientId=client_id)
                print(f"Connected to Interactive Brokers {'Paper' if self.paper_trading else 'Live'} trading")
                return True
            except Exception as exc:
                print(f"Connection attempt {attempt}/{max_retries} failed: {exc}")
                time.sleep(2)
        print("Unable to connect after maximum retries – exiting.")
        return False

    def get_stock_contract(self):
        return Stock(self.ticker, "SMART", "USD")

    def get_underlying_price(self) -> float:
        ticker = self.ib.reqTickers(self.get_stock_contract())[0]
        return ticker.marketPrice()

    def get_option_contract(self, right: str) -> Option:
        today = datetime.datetime.now(self.tz).date()
        expiry_str = today.strftime("%Y%m%d")
        spot = self.get_underlying_price()
        strike = round(spot)
        contract = Option(
            symbol=self.ticker,
            lastTradeDateOrContractMonth=expiry_str,
            strike=strike,
            right=right,
            exchange="SMART",
            multiplier="100",
            currency="USD",
        )
        details = self.ib.reqContractDetails(contract)
        if details:
            contract = details[0].contract
            self.ib.qualifyContracts(contract)
        return contract

    def place_order(self, action: str, quantity: int):
        if self.option_contract is None:
            raise RuntimeError("Option contract not initialised before order placement.")
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(self.option_contract, order)
        self.ib.sleep(1)
        print(f"{datetime.datetime.now(self.tz)} - {action} {quantity} {self.option_contract.localSymbol}")
        return trade

    def enter_position(self, position_type: str):
        right = "C" if position_type == "CALL" else "P"
        self.option_contract = self.get_option_contract(right)
        self.place_order("BUY", self.contracts)
        self.position = position_type
        self.entry_underlying_price = self.get_underlying_price()
        opt_ticker = self.ib.reqTickers(self.option_contract)[0]
        self.entry_option_price = opt_ticker.marketPrice()
        self.entry_strike = self.option_contract.strike
        print(f"Entered {position_type} - Underlying: {self.entry_underlying_price:.2f}, Option: {self.entry_option_price:.2f}, Strike: {self.entry_strike}")

    def exit_all(self, reason: str):
        if self.position is None or self.option_contract is None:
            return
        remaining = self.contracts // 2 if self.half_position_closed else self.contracts
        self.place_order("SELL", remaining)
        opt_price = self.ib.reqTickers(self.option_contract)[0].marketPrice()
        pnl_per_contract = (opt_price - self.entry_option_price) * 100
        direction = "CALL" if self.position == "CALL" else "PUT"
        self.position = None
        self.option_contract = None
        self.entry_underlying_price = None
        self.entry_option_price = None
        self.entry_strike = None
        self.half_position_closed = False

    def monitor_price_move(self, underlying_price):
        if self.position == 0:
            return  # No position to monitor

        # Check for partial sell conditions based on underlying price movement
        price_movement = underlying_price - self.entry_underlying_price
        for target in self.partial_sell_targets:
            if price_movement >= target and target not in self.sold_targets:
                self.sell_partial_position(target)
                self.sold_targets.append(target)  # Ensure we only sell once per target

    def sell_partial_position(self, target):
        # Sell half of the position based on target
        qty_to_sell = self.contracts // 2
        action = "SELL"
        print(f"Selling {qty_to_sell} contracts at {target} move target")
        
        # You would need to determine the appropriate option contract to sell based on your strategy.
        self.place_order(action, qty_to_sell)

    def run(self):
        if not self.connect_to_ib():
            return
        try:
            daily_trade_done = False
            print("Starting SPY ORB strategy ...")
            while True:
                now = datetime.datetime.now(self.tz)
                if not self.is_market_open():
                    if self.position is not None:
                        print("Market closed - force exiting open position.")
                        self.exit_all("Market closed")
                    daily_trade_done = False
                    time.sleep(60)
                    continue
                if self.is_force_close_time() and self.position is not None:
                    print("Force-close time reached - closing position.")
                    self.exit_all("14:55 force close")
                df = self.get_intraday_5min()
                if df is None or df.empty:
                    print("No historical data - waiting...")
                    time.sleep(30)
                    continue
                if not self.opening_range_set:
                    self.calculate_opening_range(df)
                    time.sleep(5)
                    continue
                if not daily_trade_done and self.position is None:
                    last_closed = df.iloc[-2]
                    if last_closed["close"] > self.opening_range_high:
                        self.enter_position("CALL")
                        daily_trade_done = True
                    elif last_closed["close"] < self.opening_range_low:
                        self.enter_position("PUT")
                        daily_trade_done = True
                if self.position is not None:
                    underlying_price = self.get_underlying_price()
                    option_price = self.ib.reqTickers(self.option_contract)[0].marketPrice()
                    df["ema9"] = df["close"].ewm(span=9, adjust=False).mean()
                    df["ema20"] = df["close"].ewm(span=20, adjust=False).mean()
                    last_closed = df.iloc[-2]
                    ema9 = df.iloc[-2]["ema9"]
                    ema20 = df.iloc[-2]["ema20"]
                    if self.position == "CALL" and last_closed["close"] < ema9 and last_closed["close"] < ema20:
                        self.exit_all("EMA stop loss (CALL)")
                        time.sleep(5)
                        continue
                    if self.position == "PUT" and last_closed["close"] > ema9 and last_closed["close"] > ema20:
                        self.exit_all("EMA stop loss (PUT)")
                        time.sleep(5)
                        continue

                    # Monitor price for partial sell conditions
                    self.monitor_price_move(underlying_price)
                time.sleep(5)
        except KeyboardInterrupt:
            print("User interrupted - shutting down.")
        except Exception as exc:
            print(f"Unhandled error: {exc}")
        finally:
            if self.position is not None:
                self.exit_all("Shutdown")
            self.ib.disconnect()
            print("Disconnected from Interactive Brokers.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="SPY Opening Range Breakout strategy")
    parser.add_argument("--ticker", type=str, default="SPY", help="Underlying ticker symbol")
    parser.add_argument("--contracts", type=int, default=2, help="Number of option contracts to trade")
    parser.add_argument("--paper_trading", action="store_true", help="Use paper trading account (7497)")
    parser.add_argument("--port", type=int, default=7497, help="Port number")
    args = parser.parse_args()
    strategy = SPYORBStrategy(
        ticker=args.ticker,
        contracts=args.contracts,
        paper_trading=args.paper_trading,
        port=args.port,
    )
    strategy.run()
