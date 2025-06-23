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

    The strategy looks for a candle that leaves the Keltner Channel, and has 1 confirmation indicator (ADX >= 25).  
    Position management (profit targets, stop, force close) mirrors the other CHAD strategies for consistency.
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
        adx_period: int = 14,  # ADX period (standard is 14)
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
        self.adx_period = adx_period  # ADX period
        self.paper_trading = paper_trading
        self.port = port
        
        # Trading state
        self.positions: list[dict] = []  # Active positions
        self.wait_for_ema20_cross = False  # Prevent re-entry after profitable trade
        self.last_profit_side: str | None = None  # "LONG" or "SHORT"

        # IB / timezone helpers
        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    # Interactive Brokers helpers
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
        """Add EMA(9), EMA(20), ATR, Keltner Channel, and ADX to *df*."""
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

        # Calculate ADX (Average Directional Index)
        df["adx"] = self.calculate_adx(df)

        return df

    def calculate_adx(self, df: pd.DataFrame) -> pd.Series:
        """Calculate ADX (Average Directional Index) using the standard period of 14."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate the True Range (TR)
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs()
        ], axis=1).max(axis=1)
        
        # Calculate the +DI and -DI
        plus_dm = high.diff()
        minus_dm = -low.diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        smoothed_tr = tr.rolling(window=self.adx_period, min_periods=1).sum()
        smoothed_plus_dm = plus_dm.rolling(window=self.adx_period, min_periods=1).sum()
        smoothed_minus_dm = minus_dm.rolling(window=self.adx_period, min_periods=1).sum()
        
        plus_di = 100 * smoothed_plus_dm / smoothed_tr
        minus_di = 100 * smoothed_minus_dm / smoothed_tr
        
        # Calculate ADX
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
        adx = dx.rolling(window=self.adx_period, min_periods=1).mean()
        
        return adx

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
        adx = candle["adx"]

        # Entry condition: ADX >= 25 and crossing the Keltner Channel
        if adx >= 25.0:
            if candle_open < kc_lower and candle_close > kc_upper:
                return "ENTER_LONG"
            elif candle_open > kc_upper and candle_close < kc_lower:
                return "ENTER_SHORT"

        return None

    def check_stop_loss(self, position: dict, last_candle: pd.Series) -> bool:
        """Check if the position should be stopped out based on the 20 EMA."""
        close_price = last_candle["close"]
        ema20 = last_candle["ema20"]  # Using 20 EMA instead of 9 EMA
        
        if np.isnan(ema20):
            return False
        
        if position["type"] == "CALL":
            # For long positions: Exit if price closes below the 20 EMA
            if close_price < ema20:
                return True
        else:
            # For short positions: Exit if price closes above the 20 EMA
            if close_price > ema20:
                return True
        
        return False

    def exit_position(self, position: dict, reason: str, partial: bool = False):
        """Exit the position."""
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

    def run(self):
        if not self.connect_to_ib():
            return
        try:
            print("Starting SPY BOSK strategy...")
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
                    print("Started monitoring BOSK signals...")
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
                    # Profit targets (rest of your position management code)

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
