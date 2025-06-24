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
    """Opening Range Breakout strategy for SPY 0-DTE options.

    The logic follows the specification supplied by the user.  It is intentionally
    written in a similar style to `spy_ema_chad.py` so that maintainers can easily
    jump between the two files.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        contracts: int = 2,
        underlying_move_target: float = 1.0,
        itm_offset: float = 1.05,
        market_open: str = "08:30:00",
        market_close: str = "15:00:00",
        force_close_time: str = "14:55:00",
        bar_size: str = "5 mins",
        paper_trading: bool = True,
        port: int = 7497,
    ):
        self.ticker = ticker
        self.contracts = contracts
        self.underlying_move_target = underlying_move_target
        self.itm_offset = itm_offset
        self.market_open = market_open
        self.market_close = market_close
        self.force_close_time = force_close_time
        self.bar_size = bar_size
        self.paper_trading = paper_trading
        self.port = port
        # Trading state
        self.opening_range_high = None
        self.opening_range_low = None
        self.opening_range_set = False

        self.position = None  # "CALL" or "PUT"
        self.option_contract = None
        self.entry_underlying_price = None
        self.entry_option_price = None
        self.entry_strike = None
        self.half_position_closed = False

        # IB & timezone
        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    # ---------------------------------------------------------------------
    # Interactive Brokers helpers
    # ---------------------------------------------------------------------
    def connect_to_ib(self, host: str = "127.0.0.1", client_id: int = 9, max_retries: int = 3) -> bool:
        """Connect to TWS / IB Gateway. Re-tries a few times for resiliency."""
        port = self.port
        for attempt in range(1, max_retries + 1):
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
                    time.sleep(1)
                self.ib.connect(host, port, clientId=client_id)
                print(
                    f"Connected to Interactive Brokers {'Paper' if self.paper_trading else 'Live'} trading"
                )
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
        """Return the ATM 0-DTE option contract for SPY (right="C" or "P")."""
        today = datetime.datetime.now(self.tz).date()
        expiry_str = today.strftime("%Y%m%d")  # 0-DTE (same-day) expiry for SPY

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

    # ---------------------------------------------------------------------
    # Market & timing helpers
    # ---------------------------------------------------------------------
    def is_market_open(self) -> bool:
        now = datetime.datetime.now(self.tz)
        today = now.date()
        mo = self.tz.localize(
            datetime.datetime.combine(today, datetime.datetime.strptime(self.market_open, "%H:%M:%S").time())
        )
        mc = self.tz.localize(
            datetime.datetime.combine(today, datetime.datetime.strptime(self.market_close, "%H:%M:%S").time())
        )
        return mo <= now <= mc

    def is_force_close_time(self) -> bool:
        now = datetime.datetime.now(self.tz)
        today = now.date()
        fct = self.tz.localize(
            datetime.datetime.combine(today, datetime.datetime.strptime(self.force_close_time, "%H:%M:%S").time())
        )
        return now >= fct

    # ---------------------------------------------------------------------
    # Data helpers
    # ---------------------------------------------------------------------
    def get_intraday_5min(self, duration: str = "1 D") -> pd.DataFrame | None:
        """Fetch 5-minute historical data for the underlying."""
        contract = self.get_stock_contract()
        bars = self.ib.reqHistoricalData(
            contract,
            endDateTime="",
            durationStr=duration,
            barSizeSetting=self.bar_size,
            whatToShow="TRADES",
            useRTH=True,
            formatDate=1,
        )
        if not bars:
            return None
        df = util.df(bars)
        df["date"] = pd.to_datetime(df["date"])
        return df

    def calculate_opening_range(self, df: pd.DataFrame):
        today = datetime.datetime.now(self.tz).date()
        # Filter today & first 15 minutes (3 * 5-minute candles)
        today_df = df[df["date"].dt.date == today]
        if today_df.empty:
            return  # Wait until we have today's data

        market_open_dt = self.tz.localize(
            datetime.datetime.combine(today, datetime.datetime.strptime(self.market_open, "%H:%M:%S").time())
        )
        range_end = market_open_dt + datetime.timedelta(minutes=15)
        opening_df = today_df[today_df["date"] < range_end]
        if len(opening_df) < 3:
            return  # Need 3 complete candles

        self.opening_range_high = opening_df["high"].max()
        self.opening_range_low = opening_df["low"].min()
        self.opening_range_set = True
        print(
            f"Opening range set - High: {self.opening_range_high:.2f}, Low: {self.opening_range_low:.2f}"
        )

    # ---------------------------------------------------------------------
    # Order helpers
    # ---------------------------------------------------------------------
    def place_order(self, action: str, quantity: int):
        if self.option_contract is None:
            raise RuntimeError("Option contract not initialised before order placement.")
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(self.option_contract, order)
        self.ib.sleep(1)
        print(f"{datetime.datetime.now(self.tz)} - {action} {quantity} {self.option_contract.localSymbol}")
        return trade

    def enter_position(self, position_type: str):
        """Enter CALL or PUT position (always buying options)."""
        right = "C" if position_type == "CALL" else "P"
        self.option_contract = self.get_option_contract(right)
        self.place_order("BUY", self.contracts)

        # Record entry stats
        self.position = position_type
        self.entry_underlying_price = self.get_underlying_price()
        opt_ticker = self.ib.reqTickers(self.option_contract)[0]
        self.entry_option_price = opt_ticker.marketPrice()
        self.entry_strike = self.option_contract.strike
        print(
            f"Entered {position_type} - Underlying: {self.entry_underlying_price:.2f}, Option: {self.entry_option_price:.2f}, Strike: {self.entry_strike}"
        )

    def exit_all(self, reason: str):
        if self.position is None or self.option_contract is None:
            return
        remaining = self.contracts // 2 if self.half_position_closed else self.contracts
        self.place_order("SELL", remaining)
        # Log P/L (rough, per option contract)
        opt_price = self.ib.reqTickers(self.option_contract)[0].marketPrice()
        pnl_per_contract = (opt_price - self.entry_option_price) * 100
        direction = "CALL" if self.position == "CALL" else "PUT"
        # Reset
        self.position = None
        self.option_contract = None
        self.entry_underlying_price = None
        self.entry_option_price = None
        self.entry_strike = None
        self.half_position_closed = False

    # ---------------------------------------------------------------------
    # Core loop
    # ---------------------------------------------------------------------
    def run(self):
        if not self.connect_to_ib():
            return

        try:
            daily_trade_done = False
            print("Starting SPY ORB strategy ...")
            while True:
                now = datetime.datetime.now(self.tz)

                # Handle market hours
                if not self.is_market_open():
                    if self.position is not None:
                        print("Market closed - force exiting open position.")
                        self.exit_all("Market closed")
                    daily_trade_done = False  # Reset for next day
                    time.sleep(60)
                    continue

                # Force-close time
                if self.is_force_close_time() and self.position is not None:
                    print("Force-close time reached - closing position.")
                    self.exit_all("14:55 force close")

                # Historical bars - used for signals
                df = self.get_intraday_5min()
                if df is None or df.empty:
                    print("No historical data - waiting...")
                    time.sleep(30)
                    continue

                # Ensure opening range captured
                if not self.opening_range_set:
                    self.calculate_opening_range(df)
                    time.sleep(5)
                    continue  # Need the range before anything else

                # Entry check (one trade per day)
                if not daily_trade_done and self.position is None:
                    last_closed = df.iloc[-2]  # Last *completed* 5-minute bar
                    if last_closed["close"] > self.opening_range_high:
                        self.enter_position("CALL")
                        daily_trade_done = True
                    elif last_closed["close"] < self.opening_range_low:
                        self.enter_position("PUT")
                        daily_trade_done = True

                # Manage open position
                if self.position is not None:
                    underlying_price = self.get_underlying_price()
                    option_price = self.ib.reqTickers(self.option_contract)[0].marketPrice()

                    # Initial stop loss (based on opening range)
                    last_closed = df.iloc[-2]
                    if self.position == "CALL" and last_closed["close"] < self.opening_range_low:
                        self.exit_all("Initial stop loss (CALL)")
                        time.sleep(5)
                        continue
                    if self.position == "PUT" and last_closed["close"] > self.opening_range_high:
                        self.exit_all("Initial stop loss (PUT)")
                        time.sleep(5)
                        continue

                    # Profit target 1 - underlying +/- $1
                    if not self.half_position_closed:
                        if self.position == "CALL" and underlying_price >= self.entry_underlying_price + self.underlying_move_target:
                            self.place_order("SELL", self.contracts // 2)
                            self.half_position_closed = True
                            print("First profit target hit - sold half, stop moved to breakeven.")
                        elif self.position == "PUT" and underlying_price <= self.entry_underlying_price - self.underlying_move_target:
                            self.place_order("SELL", self.contracts // 2)
                            self.half_position_closed = True
                            print("First profit target hit - sold half, stop moved to breakeven.")

                    # Breakeven stop on remaining half
                    if self.half_position_closed:
                        if option_price <= self.entry_option_price:
                            self.exit_all("Breakeven stop (remaining half)")
                            time.sleep(5)
                            continue

                    # Profit target 2 - option 1.05 ITM
                    if self.position == "CALL" and underlying_price >= self.entry_strike + self.itm_offset:
                        self.exit_all("Second profit target (CALL)")
                        time.sleep(5)
                        continue
                    if self.position == "PUT" and underlying_price <= self.entry_strike - self.itm_offset:
                        self.exit_all("Second profit target (PUT)")
                        time.sleep(5)
                        continue

                # Loop nap - 5-sec granularity is more than enough for 5-min bars
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
    parser.add_argument("--underlying_move_target", type=float, default=1.0, help="First profit target (underlying $ move)")
    parser.add_argument("--itm_offset", type=float, default=1.05, help="Underlying distance beyond strike for second target")
    parser.add_argument("--paper_trading", action="store_true", help="Use paper trading account (7497)")
    parser.add_argument("--port", type=int, default=7497, help="Port number")

    args = parser.parse_args()

    strategy = SPYORBStrategy(
        ticker=args.ticker,
        contracts=args.contracts,
        underlying_move_target=args.underlying_move_target,
        itm_offset=args.itm_offset,
        paper_trading=args.paper_trading,
        port=args.port,
    )
    strategy.run() 
