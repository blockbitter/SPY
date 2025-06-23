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
        self.underlying_move_target = kwargs.get("underlying_move_target", 1.0)
        self.itm_offset = kwargs.get("itm_offset", 1.05)
        self.market_open = kwargs.get("market_open", "08:30:00")
        self.market_close = kwargs.get("market_close", "15:00:00")
        self.monitor_start = kwargs.get("monitor_start", "08:25:00")
        self.no_new_trades_time = kwargs.get("no_new_trades_time", "15:30:00")
        self.force_close_time = kwargs.get("force_close_time", "14:55:00")
        self.bar_size = kwargs.get("bar_size", "5 mins")
        self.rsi_period = kwargs.get("rsi_period", 14)
        self.ema_period = kwargs.get("ema_period", 9)
        self.rsi_oversold = kwargs.get("rsi_oversold", 30.0)
        self.rsi_overbought = kwargs.get("rsi_overbought", 70.0)
        self.paper_trading = kwargs.get("paper_trading", True)
        self.port = kwargs.get("port", 7497)
        self.rsi_threshold = kwargs.get("rsi_threshold", 0.01)
        self.ema_threshold = kwargs.get("ema_threshold", 0.01)
        
        # Trading state - can have multiple positions
        self.positions = []  # List of active positions
        self.rsi_signal = None  # "LONG_SETUP" or "SHORT_SETUP" or None
        self.rsi_signal_price = None  # Price when RSI signal occurred
        self.half_position_closed = False  # Track if half position is closed
        self.monitoring_started = False
        
        # IB & timezone
        self.tz = pytz.timezone("US/Central")
        self.ib = IB()

    # Keltner Channel Calculation
    def calculate_keltner_channels(self, df: pd.DataFrame, period: int = 20, multiplier: float = 1.5):
        """Calculate the Keltner Channels."""
        df['ma'] = df['close'].rolling(window=period).mean()  # Moving average
        df['tr'] = np.maximum(df['high'] - df['low'], 
                             np.abs(df['high'] - df['close'].shift()), 
                             np.abs(df['low'] - df['close'].shift()))  # True range
        df['atr'] = df['tr'].rolling(window=period).mean()  # Average true range
        df['upper_keltner'] = df['ma'] + (multiplier * df['atr'])
        df['lower_keltner'] = df['ma'] - (multiplier * df['atr'])
        return df

    # ---------------------------------------------------------------------
    # Position Management with Stop Loss Logic
    # ---------------------------------------------------------------------
    def check_initial_stop_loss(self, position: dict, last_candle: pd.Series) -> bool:
        """Check if initial stop loss should be triggered (based on Keltner Channel)."""
        close_price = last_candle['close']
        
        # For CALL (Long positions): stop if price equals the lower Keltner Channel
        if position['type'] == "CALL":
            lower_keltner = last_candle['lower_keltner']
            if close_price <= lower_keltner:
                print(f"Stop-loss triggered for CALL at {close_price:.2f}, Keltner lower: {lower_keltner:.2f}")
                return True
        
        # For PUT (Short positions): stop if price equals the upper Keltner Channel
        elif position['type'] == "PUT":
            upper_keltner = last_candle['upper_keltner']
            if close_price >= upper_keltner:
                print(f"Stop-loss triggered for PUT at {close_price:.2f}, Keltner upper: {upper_keltner:.2f}")
                return True
        
        return False

    def check_breakeven_stop_loss(self, position: dict, option_price: float) -> bool:
        """Breakeven stop loss for second half of position (after first profit target is hit)."""
        if self.half_position_closed and option_price <= position['entry_option_price']:
            print(f"Breakeven stop triggered for {position['type']} at {option_price:.2f}")
            return True
        return False

    def check_force_close(self):
        """Force close positions at 2:55 PM."""
        if self.is_force_close_time():
            self.close_all_positions("14:55 force close")

    def exit_all(self, reason: str):
        """Exit all positions based on a given reason."""
        print(f"Exiting all positions due to: {reason}")
        for position in self.positions[:]:
            self.exit_position(position, reason)

    def exit_position(self, position: dict, reason: str):
        """Exit a specific position."""
        # Sell remaining or half depending on the state of the position
        if not position['half_sold']:
            quantity = self.contracts // 2
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

    def place_order(self, contract, action: str, qty: int):
        """Place an order."""
        order = MarketOrder(action, qty)
        trade = self.ib.placeOrder(contract, order)
        self.ib.sleep(1)
        print(f"{datetime.datetime.now(self.tz)} - {action} {qty} {contract.localSymbol}")
        return trade

    def is_force_close_time(self) -> bool:
        """Check if it's time to force close positions (2:55 PM)."""
        now = datetime.datetime.now(self.tz)
        force_close_time = self.tz.localize(
            datetime.datetime.combine(now.date(), datetime.datetime.strptime("14:55:00", "%H:%M:%S").time())
        )
        return now >= force_close_time

    # ---------------------------------------------------------------------
    # Main loop and strategy execution
    # ---------------------------------------------------------------------
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

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SPY REV (Reversal) trading strategy")
    parser.add_argument("--ticker", type=str, default="SPY", help="Underlying ticker symbol")
    parser.add_argument("--contracts", type=int, default=2, help="Number of option contracts to trade")
    parser.add_argument("--underlying_move_target", type=float, default=1.0, help="First profit target (underlying $ move)")
    parser.add_argument("--itm_offset", type=float, default=1.05, help="Underlying distance beyond strike for second target")
    parser.add_argument("--rsi_period", type=int, default=14, help="RSI calculation period")
    parser.add_argument("--ema_period", type=int, default=9, help="EMA calculation period")
    parser.add_argument("--rsi_oversold", type=float, default=30.0, help="RSI oversold level")
    parser.add_argument("--rsi_overbought", type=float, default=70.0, help="RSI overbought level")
    parser.add_argument("--paper_trading", action="store_true", help="Use paper trading account")
    parser.add_argument("--port", type=int, default=7497, help="Port number")
    args = parser.parse_args()

    strategy = SPYREVStrategy(
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
    strategy.run()

