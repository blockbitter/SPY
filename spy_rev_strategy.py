#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPY REV CHAD: SPY Reversal Highly Automated Dealer
# An automated trading system for RSI reversal trading with 9 EMA confirmation

import pandas as pd
import numpy as np
import datetime
import time
import pytz
from ib_insync import *


class SPYREVStrategy:
    """RSI Reversal strategy for SPY 0-DTE options.

    The strategy looks for RSI extremes (below 30 for longs, above 70 for shorts)
    followed by price confirmation relative to 9 EMA for entry signals. Once in
    a trade, it only monitors profit targets and stop losses.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        contracts: int = 2,
        underlying_move_target: float = 1.0,
        itm_offset: float = 1.05,
        market_open: str = "08:30:00",
        market_close: str = "15:00:00",
        monitor_start: str = "08:25:00",
        no_new_trades_time: str = "14:30:00",
        force_close_time: str = "14:55:00",
        bar_size: str = "5 mins",
        rsi_period: int = 14,
        ema_period: int = 9,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        paper_trading: bool = True,
        port: int = 7497,
        rsi_threshold: float = 0.01,
        ema_threshold: float = 0.01,
    ):
        # Strategy parameters
        self.ticker = ticker
        self.contracts = contracts
        self.underlying_move_target = underlying_move_target
        self.itm_offset = itm_offset
        self.market_open = market_open
        self.market_close = market_close
        self.monitor_start = monitor_start
        self.no_new_trades_time = no_new_trades_time
        self.force_close_time = force_close_time
        self.bar_size = bar_size
        self.rsi_period = rsi_period
        self.ema_period = ema_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.paper_trading = paper_trading
        self.port = port
        self.rsi_threshold = rsi_threshold
        self.ema_threshold = ema_threshold

        # Runtime state
        self.positions = []            # Active positions
        self.rsi_signal = None         # "LONG_SETUP" or "SHORT_SETUP"
        self.rsi_signal_price = None   # Underlying price when RSI condition met
        self.in_trade = False          # True once a trade is opened
        self.monitoring_started = False

        # IB connection & timezone
        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    # ---------------------------------------------------------------------
    # IB and market timing helpers
    # ---------------------------------------------------------------------
    def connect_to_ib(self, host="127.0.0.1", client_id=10, max_retries=3) -> bool:
        for attempt in range(max_retries):
            try:
                if self.ib.isConnected():
                    self.ib.disconnect()
                    time.sleep(1)
                self.ib.connect(host, self.port, clientId=client_id)
                print(f"[{datetime.datetime.now(self.tz)}] Connected to IB ({'Paper' if self.paper_trading else 'Live'})")
                return True
            except Exception as e:
                print(f"IB connect attempt {attempt+1} failed: {e}")
                time.sleep(2)
        print("Failed to connect to IB.")
        return False

    def is_market_open(self) -> bool:
        now = datetime.datetime.now(self.tz)
        today = now.date()
        open_time = self.tz.localize(datetime.datetime.combine(today, datetime.datetime.strptime(self.market_open, "%H:%M:%S").time()))
        close_time = self.tz.localize(datetime.datetime.combine(today, datetime.datetime.strptime(self.market_close, "%H:%M:%S").time()))
        return open_time <= now <= close_time

    def should_start_monitoring(self) -> bool:
        now = datetime.datetime.now(self.tz)
        start_time = self.tz.localize(datetime.datetime.combine(now.date(), datetime.datetime.strptime(self.monitor_start, "%H:%M:%S").time()))
        return now >= start_time

    def can_open_new_trades(self) -> bool:
        now = datetime.datetime.now(self.tz)
        cutoff = self.tz.localize(datetime.datetime.combine(now.date(), datetime.datetime.strptime(self.no_new_trades_time, "%H:%M:%S").time()))
        return now < cutoff

    def is_force_close_time(self) -> bool:
        now = datetime.datetime.now(self.tz)
        fct = self.tz.localize(datetime.datetime.combine(now.date(), datetime.datetime.strptime(self.force_close_time, "%H:%M:%S").time()))
        return now >= fct

    # ---------------------------------------------------------------------
    # Data retrieval and indicator calculation
    # ---------------------------------------------------------------------
    def get_stock_contract(self):
        return Stock(self.ticker, "SMART", "USD")

    def get_underlying_price(self) -> float:
        ticker = self.ib.reqTickers(self.get_stock_contract())[0]
        return ticker.marketPrice()

    def get_option_contract(self, right: str) -> Option:
        today = datetime.datetime.now(self.tz).date()
        expiry = today.strftime("%Y%m%d")
        spot = self.get_underlying_price()
        strike = round(spot)
        contract = Option(self.ticker, expiry, strike, right, "SMART", "100", "USD")
        details = self.ib.reqContractDetails(contract)
        if details:
            contract = details[0].contract
            self.ib.qualifyContracts(contract)
        return contract

    def get_intraday_5min(self, duration="1 D") -> pd.DataFrame | None:
        bars = self.ib.reqHistoricalData(
            self.get_stock_contract(),
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df["rsi"] = 100 - (100 / (1 + rs))
        df["ema_9"] = df["close"].ewm(span=self.ema_period, adjust=False).mean()
        return df

    # ---------------------------------------------------------------------
    # Signal detection and entry logic
    # ---------------------------------------------------------------------
    def check_rsi_signal(self, df: pd.DataFrame):
        """Detect when RSI crosses above/below thresholds, then wait for EMA confirmation."""
        if self.in_trade or self.rsi_signal is not None:
            return

        last = df.iloc[-1]
        rsi = last["rsi"]
        if pd.isna(rsi):
            return

        # SHORT setup: RSI crosses above overbought
        if rsi > self.rsi_overbought + self.rsi_threshold:
            self.rsi_signal = "SHORT_SETUP"
            self.rsi_signal_price = last["close"]
            print(f"[{last['date']}] RSI > {self.rsi_overbought}: signal SHORT at {self.rsi_signal_price:.2f}")

        # LONG setup: RSI crosses below oversold
        elif rsi < self.rsi_oversold - self.rsi_threshold:
            self.rsi_signal = "LONG_SETUP"
            self.rsi_signal_price = last["close"]
            print(f"[{last['date']}] RSI < {self.rsi_oversold}: signal LONG at {self.rsi_signal_price:.2f}")

    def check_entry_conditions(self, df: pd.DataFrame) -> str | None:
        """Once RSI signal is set, wait for any candle to close across the 9 EMA."""
        if self.in_trade or self.rsi_signal is None:
            return None

        last = df.iloc[-1]
        price = last["close"]
        ema = last["ema_9"]
        if pd.isna(ema):
            return None

        if self.rsi_signal == "SHORT_SETUP" and price < ema - self.ema_threshold:
            return "ENTER_SHORT"
        if self.rsi_signal == "LONG_SETUP" and price > ema + self.ema_threshold:
            return "ENTER_LONG"
        return None

    # ---------------------------------------------------------------------
    # Position management
    # ---------------------------------------------------------------------
    def place_order(self, contract, action: str, qty: int):
        order = MarketOrder(action, qty)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        print(f"[{datetime.datetime.now(self.tz)}] {action} {qty} {contract.localSymbol}")
        return trade

    def enter_position(self, pos_type: str):
        """Open a CALL or PUT trade."""
        if self.in_trade:
            return
        signal = self.rsi_signal
        # Mark in-trade and reset signal tracking
        self.in_trade = True
        self.rsi_signal = None

        right = "C" if pos_type == "CALL" else "P"
        contract = self.get_option_contract(right)
        self.place_order(contract, "BUY", self.contracts)

        pos = {
            "type": pos_type,
            "contract": contract,
            "entry_underlying": self.get_underlying_price(),
            "entry_option": self.ib.reqTickers(contract)[0].marketPrice(),
            "entry_strike": contract.strike,
            "stop_loss_price": self.rsi_signal_price,
            "contracts_remaining": self.contracts,
            "half_sold": False,
            "entry_time": datetime.datetime.now(self.tz),
        }
        self.positions.append(pos)
        print(f"Entered {pos_type} @ underlying {pos['entry_underlying']:.2f}, stop @ {pos['stop_loss_price']:.2f}")

    def check_stop_loss(self, pos: dict, last_close: float) -> bool:
        if pos["type"] == "CALL":
            return last_close < pos["stop_loss_price"]
        else:
            return last_close > pos["stop_loss_price"]

    def check_profit_targets(self, pos: dict) -> str | None:
        spot = self.get_underlying_price()
        opt_price = self.ib.reqTickers(pos["contract"])[0].marketPrice()
        # First target on underlying move
        if not pos["half_sold"]:
            if pos["type"] == "CALL" and spot >= pos["entry_underlying"] + self.underlying_move_target:
                return "FIRST_TARGET"
            if pos["type"] == "PUT"  and spot <= pos["entry_underlying"] - self.underlying_move_target:
                return "FIRST_TARGET"
        # Breakeven stop on second half
        if pos["half_sold"] and opt_price <= pos["entry_option"]:
            return "BREAKEVEN_STOP"
        # Second target: $1.05 ITM
        if pos["type"] == "CALL" and spot >= pos["entry_strike"] + self.itm_offset:
            return "SECOND_TARGET"
        if pos["type"] == "PUT"  and spot <= pos["entry_strike"] - self.itm_offset:
            return "SECOND_TARGET"
        return None

    def exit_position(self, pos: dict, reason: str, partial: bool = False):
        """Close all or half of a position, then reset to look for new signals."""
        qty = pos["contracts_remaining"] if not partial else self.contracts // 2
        if partial:
            pos["contracts_remaining"] -= qty
            pos["half_sold"] = True

        self.place_order(pos["contract"], "SELL", qty)
        pnl = (self.ib.reqTickers(pos["contract"])[0].marketPrice() - pos["entry_option"]) * 100
        print(f"Exited {qty} {pos['type']} | Reason: {reason} | P/L per contract: {pnl:.2f}")

        # Remove fully closed positions
        if pos["contracts_remaining"] == 0:
            self.positions.remove(pos)
            # Reset strategy to look for new RSI/EMA signals
            self.in_trade = False
            self.rsi_signal = None
            self.rsi_signal_price = None

    def close_all_positions(self, reason: str):
        for pos in self.positions[:]:
            self.exit_position(pos, reason)

    # ---------------------------------------------------------------------
    # Main execution loop
    # ---------------------------------------------------------------------
    def reset_daily(self):
        self.positions.clear()
        self.rsi_signal = None
        self.rsi_signal_price = None
        self.in_trade = False
        self.monitoring_started = False

    def run(self):
        if not self.connect_to_ib():
            return

        try:
            print("Starting SPY REV strategy...")
            while True:
                now = datetime.datetime.now(self.tz)

                # market closed
                if not self.is_market_open():
                    if self.positions:
                        self.close_all_positions("Market closed")
                    self.reset_daily()
                    time.sleep(60)
                    continue

                # force close all at 14:55
                if self.is_force_close_time() and self.positions:
                    self.close_all_positions("Force close time")

                # start monitoring after 08:25
                if not self.monitoring_started and self.should_start_monitoring():
                    self.monitoring_started = True
                    print(f"Monitoring from {now}")

                # fetch data & indicators
                df = self.get_intraday_5min()
                if df is None or len(df) < max(self.rsi_period, self.ema_period) + 1:
                    time.sleep(30)
                    continue
                df = self.calculate_indicators(df)
                last_close = df.iloc[-1]["close"]

                # manage open positions
                if self.positions:
                    for pos in self.positions[:]:
                        if self.check_stop_loss(pos, last_close):
                            self.exit_position(pos, "Stop loss")
                            continue
                        tgt = self.check_profit_targets(pos)
                        if tgt == "FIRST_TARGET":
                            self.exit_position(pos, "First target", partial=True)
                        elif tgt == "BREAKEVEN_STOP":
                            self.exit_position(pos, "Breakeven stop")
                        elif tgt == "SECOND_TARGET":
                            self.exit_position(pos, "Second target")
                    time.sleep(5)
                    continue

                # signal & entry (only if within allowed trade window)
                if self.monitoring_started and self.can_open_new_trades():
                    self.check_rsi_signal(df)
                    entry = self.check_entry_conditions(df)
                    if entry == "ENTER_LONG":
                        self.enter_position("CALL")
                    elif entry == "ENTER_SHORT":
                        self.enter_position("PUT")

                time.sleep(5)

        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            if self.positions:
                self.close_all_positions("Shutdown")
            self.ib.disconnect()
            print("Disconnected from IB.")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="SPY")
    p.add_argument("--contracts", type=int, default=2)
    p.add_argument("--underlying_move_target", type=float, default=1.0)
    p.add_argument("--itm_offset", type=float, default=1.05)
    p.add_argument("--rsi_period", type=int, default=14)
    p.add_argument("--ema_period", type=int, default=9)
    p.add_argument("--rsi_oversold", type=float, default=30.0)
    p.add_argument("--rsi_overbought", type=float, default=70.0)
    p.add_argument("--paper_trading", action="store_true")
    p.add_argument("--port", type=int, default=7497)
    args = p.parse_args()

    strat = SPYREVStrategy(
        ticker=args.ticker,
        contracts=args.contracts,
        underlying_move_target=args.underlying_move_target,
        itm_offset=args.itm_offset,
        rsi_period=args.rsi_period,
        ema_period=args.ema_period,
        rsi_oversold=args.rsi_oversold,
        rsi_overbought=args.rsi_overbought,
        paper_trading=args.paper_trading,
        port=args.port,
    )
    strat.run()

