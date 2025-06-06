#!/usr/bin/env python
# -*- coding: utf-8 -*-
# SPY BOSK CHAD: Break-Of-Structure & Keltner Channel Highly Automated Dealer
# An automated trading system that implements the BOSK strategy for SPY options.

import pandas as pd
import numpy as np
import datetime
import time
import pytz
from ib_insync import *


class SPYBOSKStrategy:
    """Break-of-Structure + Keltner Channel exit strategy for SPY 0-DTE options.

    The strategy looks for a simultaneous bullish / bearish break of structure on the
    5-minute chart *and* a Keltner Channel cross on the same candle.  Position
    management (profit targets, stop, force close) mirrors the other CHAD
    strategies for consistency.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        contracts: int = 2,
        underlying_move_target: float = 1.0,
        itm_offset: float = 1.05,
        market_open: str = "08:30:00",
        market_close: str = "15:00:00",
        monitor_start: str = "08:30:00",
        no_new_trades_time: str = "14:00:00",
        force_close_time: str = "14:55:00",
        bar_size: str = "5 mins",
        ema9_period: int = 9,
        ema20_period: int = 20,
        atr_period: int = 20,
        kc_mult: float = 1.5,
        paper_trading: bool = True,
        port: int = 7497,
    ):
        # Core parameters
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
        self.ema9_period = ema9_period
        self.ema20_period = ema20_period
        self.atr_period = atr_period
        self.kc_mult = kc_mult
        self.paper_trading = paper_trading
        self.port = port
        # Trading state
        self.positions: list[dict] = []  # Active positions
        self.wait_for_ema20_cross = False  # Prevent re-entry after profitable trade
        self.last_profit_side: str | None = None  # "LONG" or "SHORT"

        # IB / timezone helpers
        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    # ------------------------------------------------------------------
    # Interactive Brokers helpers
    # ------------------------------------------------------------------
    def connect_to_ib(self, host: str = "127.0.0.1", client_id: int = 11, max_retries: int = 3) -> bool:
        """Connect to TWS / IB Gateway."""
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
        print("Unable to connect after maximum retries - exiting.")
        return False


    def has_position_type(self, position_type: str) -> bool:
        """Check if there's already a position of the specified type (CALL or PUT)."""
        return any(pos['type'] == position_type for pos in self.positions)


    def get_stock_contract(self):
        return Stock(self.ticker, "SMART", "USD")

    def get_underlying_price(self) -> float:
        ticker = self.ib.reqTickers(self.get_stock_contract())[0]
        return ticker.marketPrice()

    def get_option_contract(self, right: str) -> Option:
        """Return the ATM 0-DTE option contract (right="C" or "P")."""
        today = datetime.datetime.now(self.tz).date()
        expiry_str = today.strftime("%Y%m%d")
        strike = round(self.get_underlying_price())
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

    # ------------------------------------------------------------------
    # Time helpers
    # ------------------------------------------------------------------
    def _today_dt(self, time_str: str) -> datetime.datetime:
        today = datetime.datetime.now(self.tz).date()
        return self.tz.localize(
            datetime.datetime.combine(today, datetime.datetime.strptime(time_str, "%H:%M:%S").time())
        )

    def is_market_open(self) -> bool:
        now = datetime.datetime.now(self.tz)
        return self._today_dt(self.market_open) <= now <= self._today_dt(self.market_close)

    def should_start_monitoring(self) -> bool:
        now = datetime.datetime.now(self.tz)
        return now >= self._today_dt(self.monitor_start)

    def can_open_new_trades(self) -> bool:
        """Return True if we are allowed to open *new* positions right now."""
        now = datetime.datetime.now(self.tz)
        if now >= self._today_dt(self.no_new_trades_time):
            return False
        if self.wait_for_ema20_cross:
            return False
        return True

    def is_force_close_time(self) -> bool:
        now = datetime.datetime.now(self.tz)
        return now >= self._today_dt(self.force_close_time)

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------
    def get_intraday_5min(self, duration: str = "1 D") -> pd.DataFrame | None:
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

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add EMA(9), EMA(20), ATR and Keltner Channel bands to *df*."""
        # EMA 9 / EMA 20
        df["ema9"] = df["close"].ewm(span=self.ema9_period, adjust=False).mean()
        df["ema20"] = df["close"].ewm(span=self.ema20_period, adjust=False).mean()

        # ATR (true range)
        high_low = df["high"] - df["low"]
        high_close = (df["high"] - df["close"].shift(1)).abs()
        low_close = (df["low"] - df["close"].shift(1)).abs()
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.ewm(span=self.atr_period, adjust=False).mean()
        df["atr"] = atr

        # Keltner Channel
        df["kc_upper"] = df["ema20"] + self.kc_mult * df["atr"]
        df["kc_lower"] = df["ema20"] - self.kc_mult * df["atr"]
        return df

    # ------------------------------------------------------------------
    # Signal helpers
    # ------------------------------------------------------------------
    def _break_of_structure(self, df: pd.DataFrame, last_idx: int) -> str | None:
        """Return 'LONG' or 'SHORT' if candle at *last_idx* breaks structure."""
        if last_idx < 3:
            return None
        prev_high = df["high"].iloc[last_idx - 3:last_idx].max()
        prev_low = df["low"].iloc[last_idx - 3:last_idx].min()
        close = df["close"].iloc[last_idx]
        if close > prev_high:
            return "LONG"
        if close < prev_low:
            return "SHORT"
        return None

    def check_entry_signal(self, df: pd.DataFrame) -> str | None:
        """Evaluate the last *completed* candle (index -2) for entry signal."""
        if len(df) < 4:
            return None
        idx = len(df) - 2
        candle = df.iloc[idx]
        kc_lower = candle["kc_lower"]
        kc_upper = candle["kc_upper"]
        candle_open = candle["open"]
        candle_close = candle["close"]

        bos_side = self._break_of_structure(df, idx)
        if bos_side == "LONG":
            # Need open below KC lower and close above it (cross upwards)
            if candle_open < kc_lower and candle_close > kc_lower:
                return "ENTER_LONG"
        elif bos_side == "SHORT":
            # Need open above KC upper and close below it (cross downwards)
            if candle_open > kc_upper and candle_close < kc_upper:
                return "ENTER_SHORT"
        return None

    # ------------------------------------------------------------------
    # Position helpers
    # ------------------------------------------------------------------
    def place_order(self, contract, action: str, quantity: int):
        order = MarketOrder(action, quantity)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        print(f"{datetime.datetime.now(self.tz)} - {action} {quantity} {contract.localSymbol}")
        return trade

    def enter_position(self, position_type: str):
        """Enter CALL (long) or PUT (short)."""
        if self.has_position_type(position_type):
            print(f"Already have a {position_type} position - skipping entry")
            return
        right = "C" if position_type == "CALL" else "P"
        option_contract = self.get_option_contract(right)
        self.place_order(option_contract, "BUY", self.contracts)

        entry_underlying = self.get_underlying_price()
        entry_option_price = self.ib.reqTickers(option_contract)[0].marketPrice()
        position = {
            "type": position_type,  # CALL / PUT
            "contract": option_contract,
            "entry_underlying_price": entry_underlying,
            "entry_option_price": entry_option_price,
            "entry_strike": option_contract.strike,
            "contracts_remaining": self.contracts,
            "half_sold": False,
            "entry_time": datetime.datetime.now(self.tz),
        }
        self.positions.append(position)
        print(
            f"Entered {position_type} - Underlying: {entry_underlying:.2f}, Option: {entry_option_price:.2f}, Strike: {position['entry_strike']}"
        )

    def check_stop_loss(self, position: dict, last_candle: pd.Series) -> bool:
        close_price = last_candle["close"]
        ema9 = last_candle["ema9"]
        if np.isnan(ema9):
            return False
        if position["type"] == "CALL":
            return close_price < ema9
        else:
            return close_price > ema9

    def check_profit_targets(self, position: dict) -> str | None:
        underlying_price = self.get_underlying_price()
        option_price = self.ib.reqTickers(position["contract"])[0].marketPrice()
        # First target: underlying ± $1
        if not position["half_sold"]:
            if position["type"] == "CALL":
                if underlying_price >= position["entry_underlying_price"] + self.underlying_move_target:
                    return "FIRST_TARGET"
            else:
                if underlying_price <= position["entry_underlying_price"] - self.underlying_move_target:
                    return "FIRST_TARGET"
        # Breakeven stop after half sold
        if position["half_sold"] and option_price <= position["entry_option_price"]:
            return "BREAKEVEN_STOP"
        # Second target: option 1.05 ITM on underlying
        if position["type"] == "CALL":
            if underlying_price >= position["entry_strike"] + self.itm_offset:
                return "SECOND_TARGET"
        else:
            if underlying_price <= position["entry_strike"] - self.itm_offset:
                return "SECOND_TARGET"
        return None

    def exit_position(self, position: dict, reason: str, partial: bool = False):
        if partial and not position["half_sold"]:
            qty = self.contracts // 2
            position["contracts_remaining"] -= qty
            position["half_sold"] = True
        else:
            qty = position["contracts_remaining"]
        self.place_order(position["contract"], "SELL", qty)
        option_price = self.ib.reqTickers(position["contract"])[0].marketPrice()
        pnl = (option_price - position["entry_option_price"]) * 100
        print(f"Closed {qty} {position['type']} | Reason: {reason} | P/L: ${pnl:.2f}/contract")
        if not partial or position["contracts_remaining"] == 0:
            # Determine if profitable for re-entry guard
            self.wait_for_ema20_cross = pnl > 0
            self.last_profit_side = "LONG" if position["type"] == "CALL" else "SHORT"
            self.positions.remove(position)

    def close_all_positions(self, reason: str):
        for pos in self.positions[:]:
            self.exit_position(pos, reason)

    # ------------------------------------------------------------------
    # Daily reset helpers
    # ------------------------------------------------------------------
    def reset_daily_state(self):
        self.positions = []
        self.wait_for_ema20_cross = False
        self.last_profit_side = None

    def _check_ema20_cross_reset(self, last_candle: pd.Series):
        if not self.wait_for_ema20_cross:
            return
        close_price = last_candle["close"]
        ema20 = last_candle["ema20"]
        if np.isnan(ema20):
            return
        if self.last_profit_side == "LONG" and close_price < ema20:
            self.wait_for_ema20_cross = False
        elif self.last_profit_side == "SHORT" and close_price > ema20:
            self.wait_for_ema20_cross = False

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------
    def run(self):
        if not self.connect_to_ib():
            return
        try:
            print("Starting SPY BOSK strategy …")
            monitoring_started = False
            while True:
                now = datetime.datetime.now(self.tz)

                # Market hours check
                if not self.is_market_open():
                    if self.positions:
                        print("Market closed - exiting all positions.")
                        self.close_all_positions("Market closed")
                    self.reset_daily_state()
                    time.sleep(60)
                    continue

                # Force-close time
                if self.is_force_close_time() and self.positions:
                    print("Force-close time reached - closing all positions.")
                    self.close_all_positions("14:55 force close")

                # Start monitoring after monitor_start
                if not monitoring_started and self.should_start_monitoring():
                    monitoring_started = True
                    print("Started monitoring BOSK signals …")
                if not monitoring_started:
                    time.sleep(30)
                    continue

                # Fetch data
                df = self.get_intraday_5min()
                if df is None or len(df) < max(self.ema9_period, self.ema20_period, self.atr_period) + 5:
                    print("Insufficient historical data - waiting...")
                    time.sleep(30)
                    continue
                df = self.calculate_indicators(df)
                last_candle = df.iloc[-2]  # Last completed candle

                # Reset re-entry guard based on EMA20 cross
                self._check_ema20_cross_reset(last_candle)

                # Entry logic
                if self.can_open_new_trades():
                    entry_signal = self.check_entry_signal(df)
                    if entry_signal == "ENTER_LONG":
                        self.enter_position("CALL")
                    elif entry_signal == "ENTER_SHORT":
                        self.enter_position("PUT")

                # Manage positions
                for pos in self.positions[:]:
                    # Stop loss
                    if self.check_stop_loss(pos, last_candle):
                        self.exit_position(pos, "Stop loss")
                        continue
                    # Profit targets
                    tgt = self.check_profit_targets(pos)
                    if tgt == "FIRST_TARGET":
                        self.exit_position(pos, "First profit target", partial=True)
                    elif tgt == "BREAKEVEN_STOP":
                        self.exit_position(pos, "Breakeven stop")
                    elif tgt == "SECOND_TARGET":
                        self.exit_position(pos, "Second profit target")

                # Pace loop
                time.sleep(5)
        except KeyboardInterrupt:
            print("User interrupted - shutting down.")
        except Exception as exc:
            print(f"Unhandled error: {exc}")
        finally:
            if self.positions:
                self.close_all_positions("Shutdown")
            self.ib.disconnect()
            print("Disconnected from Interactive Brokers.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPY BOSK trading strategy")
    parser.add_argument("--ticker", type=str, default="SPY", help="Underlying ticker symbol")
    parser.add_argument("--contracts", type=int, default=2, help="Number of option contracts to trade")
    parser.add_argument("--underlying_move_target", type=float, default=1.0, help="First profit target (underlying $ move)")
    parser.add_argument("--itm_offset", type=float, default=1.05, help="Underlying distance beyond strike for second target")
    parser.add_argument("--paper_trading", action="store_true", help="Use paper trading account (7497)")
    parser.add_argument("--port", type=int, default=7497, help="Port number")
    args = parser.parse_args()

    strategy = SPYBOSKStrategy(
        ticker=args.ticker,
        contracts=args.contracts,
        underlying_move_target=args.underlying_move_target,
        itm_offset=args.itm_offset,
        paper_trading=args.paper_trading,
        port=args.port,
    )
    strategy.run() 
