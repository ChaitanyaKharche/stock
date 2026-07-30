import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from zoneinfo import ZoneInfo

# You need ALPACA_API_KEY/SECRET set in your environment
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

class BreakoutBacktestSimulator:
    """Production-grade CSV loader - handles ALL edge cases + premarket check"""
    
    def __init__(self, symbols, cash_per_trade=1000):
        if isinstance(symbols, str):
            symbols = [s.strip() for s in symbols.split(',')]
        self.symbols = symbols
        self.cash_per_trade = cash_per_trade

        # Alpaca client for premarket bars
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.tz = ZoneInfo("America/New_York")
        
        print(f"\n{'='*70}")
        print(f"BACKTEST SIMULATOR - Testing Trading Logic w/ Premarket")
        print(f"{'='*70}")
        print(f"Symbols: {', '.join(symbols)}")
        print(f"Cash per trade: ${cash_per_trade}")
        print(f"Target: +5.0%, Stop: -2.0%\n")
    
    def load_historical_csv(self, symbol):
        csv_path = Path(f"historical_data/{symbol}_20y.csv")
        if not csv_path.exists():
            return None
        try:
            df = pd.read_csv(csv_path, index_col=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = ['_'.join(col).strip('_') for col in df.columns.values]
            df = df.reset_index(drop=True)
            df = df.loc[:, ~df.columns.duplicated(keep='first')]
            df.columns = df.columns.str.lower().str.strip()
            date_col = None
            for col in df.columns:
                if any(x in col for x in ['date', 'timestamp', 'time']):
                    date_col = col
                    break
            if date_col is None:
                return None
            df['date_parsed'] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
            df = df.dropna(subset=['date_parsed'])
            df = df.drop(columns=[date_col])
            df = df.rename(columns={'date_parsed': 'date'})
            ohlcv_mapping = {'open': None, 'high': None, 'low': None, 'close': None}
            for col in df.columns:
                if col not in ['date']:
                    if 'open' in col and ohlcv_mapping['open'] is None:
                        ohlcv_mapping['open'] = col
                    elif 'high' in col and ohlcv_mapping['high'] is None:
                        ohlcv_mapping['high'] = col
                    elif 'low' in col and ohlcv_mapping['low'] is None:
                        ohlcv_mapping['low'] = col
                    elif 'close' in col and ohlcv_mapping['close'] is None:
                        ohlcv_mapping['close'] = col
            if any(v is None for v in ohlcv_mapping.values()):
                return None
            keep_cols = ['date'] + list(ohlcv_mapping.values())
            df = df[keep_cols]
            rename_dict = {v: k for k, v in ohlcv_mapping.items()}
            df = df.rename(columns=rename_dict)
            for col in ['open', 'high', 'low', 'close']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            df = df.dropna()
            df = df.sort_values('date').reset_index(drop=True)
            if len(df) < 2:
                return None
            return df
        except Exception as e:
            print(f"    ERROR loading {symbol}: {str(e)[:50]}")
            return None

    def get_premarket_high_low(self, symbol):
        try:
            now = datetime.now(self.tz)
            today_date = now.date()
            today_start = datetime.combine(today_date, dt_time(4, 0), tzinfo=self.tz)
            today_930 = datetime.combine(today_date, dt_time(9, 30), tzinfo=self.tz)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=today_start,
                end=today_930,
                feed="sip"
            )
            bars_response = self.data_client.get_stock_bars(request)
            if bars_response is None or bars_response.df.empty:
                return None, None
            df = bars_response.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            high = float(df['high'].max())
            low = float(df['low'].min())
            return high, low
        except Exception as e:
            return None, None

    def simulate_trading_day(self):
        print(f"Simulating Trading Day: {datetime.now().strftime('%Y-%m-%d')}")
        print(f"{'-'*70}")
        wins = 0
        losses = 0
        total_pnl = 0
        trades = 0
        for symbol in self.symbols:
            df = self.load_historical_csv(symbol)
            if df is None or len(df) < 2:
                print(f"  {symbol}: No data")
                continue
            yesterday = df.iloc[-2]
            today = df.iloc[-1]
            y_high = yesterday['high']
            t_close = today['close']
            t_open = today['open']
            t_high = today['high']
            t_low = today['low']

            # Get today's premarket high/low from Alpaca
            pm_high, pm_low = self.get_premarket_high_low(symbol)

            print(f"\n  {symbol}:")
            print(f"    Y-High: ${y_high:.2f} ", end='')
            if pm_high is not None:
                print(f"| PM-High: ${pm_high:.2f} ", end='')
            print(f"| Today: O=${t_open:.2f} H=${t_high:.2f} L=${t_low:.2f} C=${t_close:.2f}")

            # Signal: today close > yesterday high OR premarket high
            has_signal = (t_close > y_high) or (pm_high is not None and t_close > pm_high)

            if not has_signal:
                print(f"    ✗ NO SIGNAL")
                continue

            print(f"    ✓ SIGNAL")
            entry = max(y_high, pm_high if pm_high is not None else y_high) * 1.001
            position_size = self.cash_per_trade
            target = entry * 1.05
            stop = entry * 0.98
            if t_high >= target:
                exit_price = target
                exit_reason = "TARGET"
            elif t_low <= stop:
                exit_price = stop
                exit_reason = "STOP"
            else:
                exit_price = t_close
                exit_reason = "EOD"
            pnl_pct = (exit_price / entry - 1) * 100
            pnl_amt = position_size * (pnl_pct / 100)
            print(f"    Entry: ${entry:.2f} (${position_size}) | Exit ({exit_reason}): ${exit_price:.2f} | P&L: {pnl_pct:+.2f}%")
            trades += 1
            total_pnl += pnl_amt
            if pnl_pct > 0:
                wins += 1
            else:
                losses += 1

        print(f"\n{'='*70}")
        print(f"SUMMARY")
        print(f"{'='*70}")
        print(f"Signals: {trades}")
        if trades > 0:
            print(f"Wins: {wins} | Losses: {losses} | Win Rate: {wins / trades * 100:.0f}%")
            print(f"Total P&L: ${total_pnl:+.2f}")
        else:
            print("No signals today")

if __name__ == "__main__":
    symbols = ['TSLA', 'NVDA', 'SPY', 'QQQ',]
    sim = BreakoutBacktestSimulator(symbols, cash_per_trade=1000)
    sim.simulate_trading_day()
