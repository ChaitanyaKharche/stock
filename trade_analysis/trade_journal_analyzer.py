"""
Trade Journal Analyzer v1.0
Fetches technical indicators at exact trade timestamps for post-mortem analysis.

Usage:
1. Edit the TRADES list at the bottom with your actual trades
2. Run: python trade_journal_analyzer.py
3. Output: trade_analysis_YYYYMMDD_HHMMSS.csv

Requires:
- pip install alpaca-py pandas numpy ta yfinance python-dateutil --break-system-packages
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from dateutil import parser as date_parser
import csv
import yfinance as yf

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Technical indicator imports
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    sys.exit(1)

TZ = ZoneInfo("America/New_York")

# ============================================================================
# DATA CLIENT
# ============================================================================

class TradeJournalAnalyzer:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.vix_cache = {}  # Cache VIX data to avoid repeated yfinance calls
        
    def get_bars(self, symbol, timeframe, start, end):
        """Fetch historical bars from Alpaca"""
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed="sip"  # Paid subscription = SIP feed
            )
            bars_response = self.data_client.get_stock_bars(request)
            
            if bars_response is None or bars_response.df.empty:
                return None
            
            df = bars_response.df
            if isinstance(df.index, pd.MultiIndex):
                if 'symbol' in df.index.names:
                    df = df.reset_index(level='symbol', drop=True)
                else:
                    df = df.droplevel(0)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(TZ)
            
            return df
        except Exception as e:
            print(f"[ERROR] get_bars {symbol} {timeframe}: {e}")
            return None

    def get_vix_at_time(self, timestamp):
        """Get VIX value at a specific timestamp using yfinance"""
        date_str = timestamp.strftime('%Y-%m-%d')
        
        if date_str in self.vix_cache:
            vix_df = self.vix_cache[date_str]
        else:
            try:
                # Fetch VIX data for the day (and day before for safety)
                start_date = (timestamp - timedelta(days=2)).strftime('%Y-%m-%d')
                end_date = (timestamp + timedelta(days=1)).strftime('%Y-%m-%d')
                
                vix = yf.Ticker("^VIX")
                vix_df = vix.history(start=start_date, end=end_date, interval="1h")
                
                if vix_df.empty:
                    # Fallback to daily
                    vix_df = vix.history(start=start_date, end=end_date, interval="1d")
                
                self.vix_cache[date_str] = vix_df
            except Exception as e:
                print(f"[WARN] Could not fetch VIX: {e}")
                return None
        
        if vix_df.empty:
            return None
        
        # Find closest VIX value
        try:
            target_ts = timestamp.replace(tzinfo=None)
            if vix_df.index.tz is not None:
                vix_df.index = vix_df.index.tz_localize(None)
            
            # Get closest value
            idx = vix_df.index.get_indexer([target_ts], method='nearest')[0]
            if idx >= 0 and idx < len(vix_df):
                return round(vix_df.iloc[idx]['Close'], 2)
        except:
            pass
        
        return None

    def calculate_technicals(self, df):
        """Calculate all technical indicators on a dataframe"""
        if df is None or len(df) < 20:
            return None
        
        df = df.copy()
        
        # EMAs
        df['ema9'] = EMAIndicator(close=df['close'], window=9).ema_indicator()
        df['ema20'] = EMAIndicator(close=df['close'], window=20).ema_indicator()
        
        # RSI
        df['rsi'] = RSIIndicator(close=df['close'], window=14).rsi()
        
        # MACD
        macd = MACD(close=df['close'])
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_hist'] = macd.macd_diff()
        
        # VWAP (cumulative from start of data)
        try:
            vwap = VolumeWeightedAveragePrice(
                high=df['high'],
                low=df['low'],
                close=df['close'],
                volume=df['volume'],
                window=len(df)
            )
            df['vwap'] = vwap.volume_weighted_average_price()
        except:
            df['vwap'] = None
        
        return df

    def get_key_levels(self, symbol, trade_date):
        """Get key support/resistance levels for a given date"""
        levels = {}
        
        # Get daily bars for previous days
        start = trade_date - timedelta(days=10)
        end = trade_date
        
        daily_df = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), start, end)
        
        if daily_df is not None and len(daily_df) >= 2:
            # Yesterday's high/low
            yesterday = daily_df.iloc[-2] if len(daily_df) >= 2 else daily_df.iloc[-1]
            levels['prev_day_high'] = round(yesterday['high'], 2)
            levels['prev_day_low'] = round(yesterday['low'], 2)
            
            # 2 days ago
            if len(daily_df) >= 3:
                two_days_ago = daily_df.iloc[-3]
                levels['two_days_ago_high'] = round(two_days_ago['high'], 2)
                levels['two_days_ago_low'] = round(two_days_ago['low'], 2)
        
        # Get premarket levels for trade date
        pm_start = datetime.combine(trade_date.date(), dt_time(4, 0), tzinfo=TZ)
        pm_end = datetime.combine(trade_date.date(), dt_time(9, 30), tzinfo=TZ)
        
        pm_df = self.get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), pm_start, pm_end)
        
        if pm_df is not None and not pm_df.empty:
            levels['pm_high'] = round(pm_df['high'].max(), 2)
            levels['pm_low'] = round(pm_df['low'].min(), 2)
        
        # Calculate $50 multiples near current price
        if daily_df is not None and not daily_df.empty:
            current_price = daily_df.iloc[-1]['close']
            lower_50 = int(current_price / 50) * 50
            upper_50 = lower_50 + 50
            levels['lower_50_multiple'] = lower_50
            levels['upper_50_multiple'] = upper_50
        
        return levels

    def get_orb_levels(self, symbol, trade_date):
        """Get Opening Range Breakout levels (first 15 min)"""
        orb_start = datetime.combine(trade_date.date(), dt_time(9, 30), tzinfo=TZ)
        orb_end = datetime.combine(trade_date.date(), dt_time(9, 45), tzinfo=TZ)
        
        orb_df = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), orb_start, orb_end)
        
        if orb_df is not None and not orb_df.empty:
            return {
                'orb_high': round(orb_df['high'].max(), 2),
                'orb_low': round(orb_df['low'].min(), 2)
            }
        return {}

    def detect_market_regime(self, symbol, timestamp):
        """Detect market regime at trade time"""
        # Get 1h bars for the day up to trade time
        start = timestamp - timedelta(hours=6)
        end = timestamp
        
        hourly_df = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Hour), start, end)
        
        if hourly_df is None or len(hourly_df) < 3:
            return "UNKNOWN"
        
        up_hours = len(hourly_df[hourly_df['close'] > hourly_df['open']])
        down_hours = len(hourly_df[hourly_df['close'] < hourly_df['open']])
        
        if down_hours > up_hours * 1.5:
            return "DOWNTREND"
        elif up_hours > down_hours * 1.5:
            return "UPTREND"
        else:
            return "CHOP"

    def analyze_single_trade(self, trade):
        """
        Analyze a single trade and return all relevant data.
        
        trade dict should contain:
        - symbol: str (e.g., 'QQQ', 'SPY')
        - underlying: str (e.g., 'QQQ' for options on QQQ)
        - action: str ('BUY' or 'SELL')
        - strike: float
        - option_type: str ('CALL' or 'PUT')
        - expiry: str (e.g., '12/4')
        - contracts: int
        - price_per_contract: float
        - timestamp: str (e.g., '2024-12-02 10:30:00' or 'Dec 2' for EOD)
        - notes: str (optional)
        """
        result = {}
        
        # Parse timestamp
        ts_str = trade['timestamp']
        underlying = trade.get('underlying', trade['symbol'])
        
        try:
            # Try parsing with time
            if ':' in ts_str:
                ts = date_parser.parse(ts_str)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=TZ)
            else:
                # Date only - assume market open
                ts = date_parser.parse(ts_str)
                ts = datetime.combine(ts.date(), dt_time(9, 30), tzinfo=TZ)
        except Exception as e:
            print(f"[ERROR] Could not parse timestamp '{ts_str}': {e}")
            return None
        
        # Basic trade info
        result['timestamp'] = ts.strftime('%Y-%m-%d %H:%M:%S')
        result['date'] = ts.strftime('%Y-%m-%d')
        result['time'] = ts.strftime('%H:%M:%S')
        result['symbol'] = trade['symbol']
        result['underlying'] = underlying
        result['action'] = trade['action']
        result['strike'] = trade.get('strike', '')
        result['option_type'] = trade.get('option_type', '')
        result['expiry'] = trade.get('expiry', '')
        result['contracts'] = trade.get('contracts', 1)
        result['price_per_contract'] = trade.get('price_per_contract', 0)
        result['total_cost'] = result['contracts'] * result['price_per_contract'] * 100
        result['notes'] = trade.get('notes', '')
        
        print(f"Analyzing: {result['action']} {result['symbol']} @ {result['timestamp']}")
        
        # Get price data at trade time
        lookback_start = ts - timedelta(hours=2)
        lookback_end = ts + timedelta(minutes=5)
        
        # 1-minute bars
        df_1m = self.get_bars(underlying, TimeFrame(1, TimeFrameUnit.Minute), lookback_start, lookback_end)
        
        if df_1m is not None and not df_1m.empty:
            df_1m = self.calculate_technicals(df_1m)
            
            # Get values at trade time (closest bar)
            try:
                idx = df_1m.index.get_indexer([ts], method='nearest')[0]
                if idx >= 0 and idx < len(df_1m):
                    bar = df_1m.iloc[idx]
                    result['underlying_price'] = round(bar['close'], 2)
                    result['underlying_open'] = round(bar['open'], 2)
                    result['underlying_high'] = round(bar['high'], 2)
                    result['underlying_low'] = round(bar['low'], 2)
                    result['volume_1m'] = int(bar['volume'])
                    result['ema9_1m'] = round(bar['ema9'], 2) if pd.notna(bar['ema9']) else None
                    result['ema20_1m'] = round(bar['ema20'], 2) if pd.notna(bar['ema20']) else None
                    result['rsi_1m'] = round(bar['rsi'], 2) if pd.notna(bar['rsi']) else None
                    result['macd_1m'] = round(bar['macd'], 4) if pd.notna(bar['macd']) else None
                    result['macd_signal_1m'] = round(bar['macd_signal'], 4) if pd.notna(bar['macd_signal']) else None
                    result['macd_hist_1m'] = round(bar['macd_hist'], 4) if pd.notna(bar['macd_hist']) else None
                    result['vwap_1m'] = round(bar['vwap'], 2) if pd.notna(bar.get('vwap')) else None
                    
                    # Price relative to VWAP
                    if result['vwap_1m']:
                        result['price_vs_vwap'] = round(result['underlying_price'] - result['vwap_1m'], 2)
                        result['price_vs_vwap_pct'] = round((result['underlying_price'] / result['vwap_1m'] - 1) * 100, 3)
                    
                    # Price relative to EMAs
                    if result['ema9_1m']:
                        result['price_vs_ema9'] = round(result['underlying_price'] - result['ema9_1m'], 2)
                    if result['ema20_1m']:
                        result['price_vs_ema20'] = round(result['underlying_price'] - result['ema20_1m'], 2)
            except Exception as e:
                print(f"[WARN] Error extracting 1m data: {e}")
        
        # 5-minute bars for higher timeframe context
        df_5m = self.get_bars(underlying, TimeFrame(5, TimeFrameUnit.Minute), lookback_start, lookback_end)
        
        if df_5m is not None and not df_5m.empty:
            df_5m = self.calculate_technicals(df_5m)
            try:
                idx = df_5m.index.get_indexer([ts], method='nearest')[0]
                if idx >= 0 and idx < len(df_5m):
                    bar = df_5m.iloc[idx]
                    result['rsi_5m'] = round(bar['rsi'], 2) if pd.notna(bar['rsi']) else None
                    result['macd_hist_5m'] = round(bar['macd_hist'], 4) if pd.notna(bar['macd_hist']) else None
            except:
                pass
        
        # Get VIX
        vix_val = self.get_vix_at_time(ts)
        result['vix'] = vix_val
        
        # VIX regime
        if vix_val:
            if vix_val < 14:
                result['vix_regime'] = 'VERY_LOW'
            elif vix_val < 17:
                result['vix_regime'] = 'LOW'
            elif vix_val < 22:
                result['vix_regime'] = 'NORMAL'
            elif vix_val < 30:
                result['vix_regime'] = 'HIGH'
            else:
                result['vix_regime'] = 'VERY_HIGH'
        
        # Key levels
        levels = self.get_key_levels(underlying, ts)
        for k, v in levels.items():
            result[k] = v
        
        # ORB levels
        orb = self.get_orb_levels(underlying, ts)
        for k, v in orb.items():
            result[k] = v
        
        # Market regime
        result['market_regime'] = self.detect_market_regime(underlying, ts)
        
        # Distance to key levels (if we have price)
        if 'underlying_price' in result:
            price = result['underlying_price']
            
            if 'orb_high' in result:
                result['dist_to_orb_high'] = round(result['orb_high'] - price, 2)
                result['dist_to_orb_high_pct'] = round((result['orb_high'] / price - 1) * 100, 3)
            
            if 'orb_low' in result:
                result['dist_to_orb_low'] = round(price - result['orb_low'], 2)
                result['dist_to_orb_low_pct'] = round((price / result['orb_low'] - 1) * 100, 3)
            
            if 'prev_day_high' in result:
                result['dist_to_prev_high'] = round(result['prev_day_high'] - price, 2)
            
            if 'prev_day_low' in result:
                result['dist_to_prev_low'] = round(price - result['prev_day_low'], 2)
            
            if 'pm_high' in result:
                result['dist_to_pm_high'] = round(result['pm_high'] - price, 2)
            
            if 'pm_low' in result:
                result['dist_to_pm_low'] = round(price - result['pm_low'], 2)
            
            # Option moneyness
            if result['strike'] and result['option_type']:
                strike = float(result['strike'])
                if result['option_type'] == 'CALL':
                    result['moneyness'] = round(price - strike, 2)
                    result['moneyness_pct'] = round((price / strike - 1) * 100, 2)
                else:  # PUT
                    result['moneyness'] = round(strike - price, 2)
                    result['moneyness_pct'] = round((strike / price - 1) * 100, 2)
                
                # Classify
                if abs(result['moneyness_pct']) < 0.3:
                    result['option_status'] = 'ATM'
                elif result['moneyness'] > 0:
                    result['option_status'] = 'ITM'
                else:
                    result['option_status'] = 'OTM'
        
        # Time of day classification
        trade_time = ts.time()
        if trade_time < dt_time(10, 0):
            result['session'] = 'OPEN_30MIN'
        elif trade_time < dt_time(10, 30):
            result['session'] = 'MORNING_CHOP'
        elif trade_time < dt_time(12, 0):
            result['session'] = 'MID_MORNING'
        elif trade_time < dt_time(14, 0):
            result['session'] = 'LUNCH'
        elif trade_time < dt_time(15, 30):
            result['session'] = 'AFTERNOON'
        else:
            result['session'] = 'CLOSE'
        
        return result

    def analyze_trades(self, trades):
        """Analyze a list of trades and return results"""
        results = []
        
        for i, trade in enumerate(trades):
            print(f"\n[{i+1}/{len(trades)}] Processing trade...")
            result = self.analyze_single_trade(trade)
            if result:
                results.append(result)
        
        return results

    def export_to_csv(self, results, filename=None):
        """Export results to CSV"""
        if not results:
            print("No results to export")
            return
        
        if filename is None:
            filename = f"trade_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        # Get all unique keys
        all_keys = set()
        for r in results:
            all_keys.update(r.keys())
        
        # Order columns logically
        priority_cols = [
            'timestamp', 'date', 'time', 'session',
            'symbol', 'underlying', 'action', 'option_type', 'strike', 'expiry',
            'contracts', 'price_per_contract', 'total_cost',
            'underlying_price', 'option_status', 'moneyness', 'moneyness_pct',
            'vix', 'vix_regime', 'market_regime',
            'rsi_1m', 'rsi_5m',
            'price_vs_vwap', 'price_vs_vwap_pct',
            'price_vs_ema9', 'price_vs_ema20',
            'macd_hist_1m', 'macd_hist_5m',
            'orb_high', 'orb_low', 'dist_to_orb_high_pct', 'dist_to_orb_low_pct',
            'prev_day_high', 'prev_day_low', 'pm_high', 'pm_low',
            'notes'
        ]
        
        # Build final column order
        columns = [c for c in priority_cols if c in all_keys]
        columns += [c for c in sorted(all_keys) if c not in columns]
        
        df = pd.DataFrame(results)
        df = df.reindex(columns=columns)
        df.to_csv(filename, index=False)
        
        print(f"\n{'='*60}")
        print(f"Exported {len(results)} trades to: {filename}")
        print(f"{'='*60}")
        
        return filename


# ============================================================================
# TRADE INPUT SECTION - EDIT YOUR TRADES HERE
# ============================================================================

def parse_trade_string(trade_str):
    """
    Parse a trade string like:
    "Buy QQQ $627 Call 12/4, 3 contracts at $0.68, 9h ago"
    
    Returns a trade dict
    """
    import re
    
    trade = {}
    
    # Extract action (Buy/Sell)
    if trade_str.lower().startswith('buy'):
        trade['action'] = 'BUY'
    elif trade_str.lower().startswith('sell'):
        trade['action'] = 'SELL'
    else:
        return None
    
    # Extract symbol and strike
    match = re.search(r'(SPY|QQQ|AAPL|MSFT|NVDA|TSLA)\s+\$?(\d+(?:\.\d+)?)\s+(Call|Put)', trade_str, re.IGNORECASE)
    if match:
        trade['symbol'] = match.group(1).upper()
        trade['underlying'] = trade['symbol']
        trade['strike'] = float(match.group(2))
        trade['option_type'] = match.group(3).upper()
    
    # Extract expiry
    match = re.search(r'(Call|Put)\s+(\d+/\d+)', trade_str, re.IGNORECASE)
    if match:
        trade['expiry'] = match.group(2)
    
    # Extract contracts and price
    match = re.search(r'(\d+)\s+contracts?\s+at\s+\$?([\d.]+)', trade_str, re.IGNORECASE)
    if match:
        trade['contracts'] = int(match.group(1))
        trade['price_per_contract'] = float(match.group(2))
    
    return trade


# ============================================================================
# YOUR TRADES - EDIT THIS LIST
# ============================================================================

# Format your trades here. You can use either dict format or the parser.
# For timestamp, use format: '2024-12-02 10:30:00' for specific times
# or just the date '2024-12-02' and it will assume market open

TRADES = [
    # Monday Dec 1
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 620,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 4,
        'price_per_contract': 1.24,
        'timestamp': '2024-12-01 09:45:00',
        'notes': 'Entry on morning push'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 620,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 2,
        'price_per_contract': 1.06,
        'timestamp': '2024-12-01 10:30:00',
        'notes': 'Partial exit T1'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 620,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 1,
        'price_per_contract': 1.59,
        'timestamp': '2024-12-01 12:00:00',
        'notes': 'Partial exit T2'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 620,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 1,
        'price_per_contract': 1.79,
        'timestamp': '2024-12-01 14:00:00',
        'notes': 'Final exit T3'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 625,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 3,
        'price_per_contract': 0.39,
        'timestamp': '2024-12-01 11:00:00',
        'notes': 'Chasing move - OTM'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 625,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 3,
        'price_per_contract': 0.17,
        'timestamp': '2024-12-01 14:30:00',
        'notes': 'Stop loss hit'
    },
    
    # Tuesday Dec 2
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 625,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 4,
        'price_per_contract': 0.60,
        'timestamp': '2024-12-02 10:00:00',
        'notes': 'Morning breakout attempt'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 625,
        'option_type': 'CALL',
        'expiry': '12/2',
        'contracts': 4,
        'price_per_contract': 0.27,
        'timestamp': '2024-12-02 11:30:00',
        'notes': 'Stop loss - reversal'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 627,
        'option_type': 'CALL',
        'expiry': '12/3',
        'contracts': 4,
        'price_per_contract': 0.85,
        'timestamp': '2024-12-02 10:30:00',
        'notes': 'Chasing after first stop'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 627,
        'option_type': 'CALL',
        'expiry': '12/3',
        'contracts': 4,
        'price_per_contract': 0.50,
        'timestamp': '2024-12-02 13:00:00',
        'notes': 'Stop loss - revenge trade'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 615,
        'option_type': 'PUT',
        'expiry': '12/2',
        'contracts': 3,
        'price_per_contract': 0.64,
        'timestamp': '2024-12-02 13:30:00',
        'notes': 'Flipped to puts after losses'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 615,
        'option_type': 'PUT',
        'expiry': '12/2',
        'contracts': 3,
        'price_per_contract': 0.30,
        'timestamp': '2024-12-02 15:00:00',
        'notes': 'Stop loss - wrong direction again'
    },
    
    # Wednesday Dec 3
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 625,
        'option_type': 'CALL',
        'expiry': '12/3',
        'contracts': 5,
        'price_per_contract': 0.36,
        'timestamp': '2024-12-03 10:15:00',
        'notes': 'Morning entry'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 625,
        'option_type': 'CALL',
        'expiry': '12/3',
        'contracts': 5,
        'price_per_contract': 0.19,
        'timestamp': '2024-12-03 12:00:00',
        'notes': 'Stop loss - chop'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'BUY',
        'strike': 627,
        'option_type': 'CALL',
        'expiry': '12/4',
        'contracts': 3,
        'price_per_contract': 0.68,
        'timestamp': '2024-12-03 14:00:00',
        'notes': 'Afternoon scalp attempt'
    },
    {
        'symbol': 'QQQ',
        'underlying': 'QQQ',
        'action': 'SELL',
        'strike': 627,
        'option_type': 'CALL',
        'expiry': '12/4',
        'contracts': 3,
        'price_per_contract': 0.41,
        'timestamp': '2024-12-03 15:30:00',
        'notes': 'EOD exit - loss'
    },
]


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("TRADE JOURNAL ANALYZER v1.0")
    print("="*60)
    print(f"Loaded {len(TRADES)} trades to analyze")
    print()
    
    analyzer = TradeJournalAnalyzer()
    results = analyzer.analyze_trades(TRADES)
    
    if results:
        output_file = analyzer.export_to_csv(results)
        
        # Print summary
        print("\n" + "="*60)
        print("QUICK SUMMARY")
        print("="*60)
        
        df = pd.DataFrame(results)
        
        # Win/Loss by regime
        if 'vix_regime' in df.columns:
            print("\nTrades by VIX Regime:")
            print(df['vix_regime'].value_counts().to_string())
        
        if 'market_regime' in df.columns:
            print("\nTrades by Market Regime:")
            print(df['market_regime'].value_counts().to_string())
        
        if 'option_status' in df.columns:
            print("\nTrades by Option Moneyness:")
            print(df['option_status'].value_counts().to_string())
        
        if 'session' in df.columns:
            print("\nTrades by Session:")
            print(df['session'].value_counts().to_string())
        
        print(f"\nFull analysis saved to: {output_file}")
    else:
        print("No trades analyzed successfully.")
