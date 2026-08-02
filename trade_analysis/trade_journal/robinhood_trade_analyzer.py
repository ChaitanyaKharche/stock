"""
Robinhood Trade Journal Analyzer v2.0
Parses Robinhood CSV exports and fetches market technicals for each trade day.

Since Robinhood doesn't provide exact timestamps, we fetch daily data and
key levels for each trading day to identify patterns.

Usage:
1. Export your Robinhood trade history as CSV
2. Set your Alpaca API keys
3. Run: python robinhood_trade_analyzer.py your_trades.csv
4. Output: trade_analysis_YYYYMMDD_HHMMSS.csv

Requires:
pip install alpaca-py pandas numpy ta yfinance python-dateutil --break-system-packages
"""

import os
import sys
import pandas as pd
import numpy as np
import re
import csv
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo
from collections import defaultdict
import yfinance as yf

# Alpaca imports
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Technical indicator imports
from ta.momentum import RSIIndicator
from ta.trend import MACD, EMAIndicator
from ta.volume import VolumeWeightedAveragePrice

from ..paths import LOGS_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY environment variables")
    print("  export ALPACA_API_KEY='your_key'")
    print("  export ALPACA_SECRET_KEY='your_secret'")
    sys.exit(1)

TZ = ZoneInfo("America/New_York")

# ============================================================================
# ROBINHOOD CSV PARSER
# ============================================================================

def parse_robinhood_csv(filepath):
    """
    Parse Robinhood CSV export and extract option trades.
    
    Returns list of trade dicts with:
    - date, symbol, underlying, strike, option_type, expiry
    - action (BUY/SELL), quantity, price, amount
    """
    trades = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            # Skip non-trade rows
            trans_code = (row.get('Trans Code') or '').strip()
            if trans_code not in ['BTO', 'STC']:
                continue
            
            instrument = (row.get('Instrument') or '').strip()
            description = (row.get('Description') or '').strip()
            
            if not instrument or not description:
                continue
            
            # Parse option description: "QQQ 11/26/2025 Call $615.00"
            match = re.match(
                r'(\w+)\s+(\d+/\d+/\d+)\s+(Call|Put)\s+\$?([\d.]+)',
                description,
                re.IGNORECASE
            )
            
            if not match:
                # Try alternative format: "SPY 5/30/2025 Put $582.00"
                match = re.match(
                    r'(\w+)\s+(\d+/\d+/\d+)\s+(Call|Put)\s+\$?([\d.]+)',
                    description,
                    re.IGNORECASE
                )
            
            if not match:
                print(f"[WARN] Could not parse: {description}")
                continue
            
            underlying = match.group(1).upper()
            expiry_str = match.group(2)
            option_type = match.group(3).upper()
            strike = float(match.group(4))
            
            # Parse date
            date_str = (row.get('Activity Date') or '').strip()
            try:
                trade_date = datetime.strptime(date_str, '%m/%d/%Y')
            except:
                print(f"[WARN] Could not parse date: {date_str}")
                continue
            
            # Parse quantity and price
            try:
                qty_str = (row.get('Quantity') or '0').strip()
                quantity = int(qty_str) if qty_str else 0
                price_str = (row.get('Price') or '0').strip().replace('$', '').replace(',', '')
                price = float(price_str) if price_str else 0
                amount_str = (row.get('Amount') or '0').strip().replace('$', '').replace(',', '').replace('(', '-').replace(')', '')
                amount = float(amount_str) if amount_str else 0
            except Exception as e:
                print(f"[WARN] Could not parse numbers: {e}")
                continue
            
            trade = {
                'date': trade_date.strftime('%Y-%m-%d'),
                'trade_date': trade_date,
                'underlying': underlying,
                'strike': strike,
                'option_type': option_type,
                'expiry': expiry_str,
                'action': 'BUY' if trans_code == 'BTO' else 'SELL',
                'trans_code': trans_code,
                'quantity': quantity,
                'price': price,
                'amount': amount,
                'description': description
            }
            
            trades.append(trade)
    
    print(f"Parsed {len(trades)} option trades from CSV")
    return trades


# ============================================================================
# DATA CLIENT
# ============================================================================

class MarketDataFetcher:
    def __init__(self):
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.vix_cache = {}
        self.daily_cache = {}
        
    def get_bars(self, symbol, timeframe, start, end):
        """Fetch historical bars from Alpaca"""
        cache_key = f"{symbol}_{timeframe}_{start}_{end}"
        if cache_key in self.daily_cache:
            return self.daily_cache[cache_key]
        
        try:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=end,
                feed="sip"
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
            
            self.daily_cache[cache_key] = df
            return df
        except Exception as e:
            print(f"[ERROR] get_bars {symbol}: {e}")
            return None

    def get_vix_for_date(self, date):
        """Get VIX value for a specific date using yfinance"""
        date_str = date.strftime('%Y-%m-%d')
        
        if date_str in self.vix_cache:
            return self.vix_cache[date_str]
        
        try:
            start_date = (date - timedelta(days=5)).strftime('%Y-%m-%d')
            end_date = (date + timedelta(days=2)).strftime('%Y-%m-%d')
            
            vix = yf.Ticker("^VIX")
            vix_df = vix.history(start=start_date, end=end_date, interval="1d")
            
            if vix_df.empty:
                return None
            
            # Find closest date
            target = pd.Timestamp(date_str)
            if vix_df.index.tz is not None:
                vix_df.index = vix_df.index.tz_localize(None)
            
            # Get value for the date or closest before
            mask = vix_df.index <= target
            if mask.any():
                closest = vix_df[mask].iloc[-1]
                vix_val = round(closest['Close'], 2)
                self.vix_cache[date_str] = vix_val
                return vix_val
            
            return None
        except Exception as e:
            print(f"[WARN] VIX fetch error: {e}")
            return None

    def get_day_stats(self, symbol, date):
        """Get daily OHLCV and technicals for a trading day"""
        result = {}
        
        # Get 1-min bars for the trading day
        start = datetime.combine(date, dt_time(9, 30), tzinfo=TZ)
        end = datetime.combine(date, dt_time(16, 0), tzinfo=TZ)
        
        df_1m = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), start, end)
        
        if df_1m is not None and not df_1m.empty:
            result['day_open'] = round(df_1m.iloc[0]['open'], 2)
            result['day_high'] = round(df_1m['high'].max(), 2)
            result['day_low'] = round(df_1m['low'].min(), 2)
            result['day_close'] = round(df_1m.iloc[-1]['close'], 2)
            result['day_volume'] = int(df_1m['volume'].sum())
            
            # Day range
            result['day_range'] = round(result['day_high'] - result['day_low'], 2)
            result['day_range_pct'] = round((result['day_range'] / result['day_open']) * 100, 2)
            
            # Day direction
            if result['day_close'] > result['day_open']:
                result['day_direction'] = 'UP'
                result['day_change_pct'] = round((result['day_close'] / result['day_open'] - 1) * 100, 2)
            else:
                result['day_direction'] = 'DOWN'
                result['day_change_pct'] = round((result['day_close'] / result['day_open'] - 1) * 100, 2)
            
            # Calculate technicals at various times
            # Morning (10:00 AM)
            morning_df = df_1m[df_1m.index.time <= dt_time(10, 30)]
            if not morning_df.empty:
                result['morning_high'] = round(morning_df['high'].max(), 2)
                result['morning_low'] = round(morning_df['low'].min(), 2)
            
            # Lunch (12:00 - 13:00)
            lunch_df = df_1m[(df_1m.index.time >= dt_time(12, 0)) & (df_1m.index.time <= dt_time(13, 0))]
            if not lunch_df.empty:
                result['lunch_high'] = round(lunch_df['high'].max(), 2)
                result['lunch_low'] = round(lunch_df['low'].min(), 2)
            
            # Calculate EMAs and RSI on daily
            if len(df_1m) > 20:
                ema9 = EMAIndicator(close=df_1m['close'], window=9).ema_indicator()
                ema20 = EMAIndicator(close=df_1m['close'], window=20).ema_indicator()
                rsi = RSIIndicator(close=df_1m['close'], window=14).rsi()
                
                # Mid-day values (around 12:00)
                mid_idx = len(df_1m) // 2
                if mid_idx > 0:
                    result['midday_rsi'] = round(rsi.iloc[mid_idx], 2) if pd.notna(rsi.iloc[mid_idx]) else None
                
                # Close RSI
                result['close_rsi'] = round(rsi.iloc[-1], 2) if pd.notna(rsi.iloc[-1]) else None
        
        # Get ORB levels (first 15 min)
        orb_start = datetime.combine(date, dt_time(9, 30), tzinfo=TZ)
        orb_end = datetime.combine(date, dt_time(9, 45), tzinfo=TZ)
        
        if df_1m is not None:
            orb_df = df_1m[(df_1m.index >= orb_start) & (df_1m.index <= orb_end)]
            if not orb_df.empty:
                result['orb_high'] = round(orb_df['high'].max(), 2)
                result['orb_low'] = round(orb_df['low'].min(), 2)
                result['orb_range'] = round(result['orb_high'] - result['orb_low'], 2)
        
        # Get previous day levels
        prev_start = datetime.combine(date - timedelta(days=5), dt_time(9, 30), tzinfo=TZ)
        prev_end = datetime.combine(date - timedelta(days=1), dt_time(16, 0), tzinfo=TZ)
        
        df_daily = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Day), prev_start, prev_end)
        
        if df_daily is not None and len(df_daily) >= 1:
            prev_day = df_daily.iloc[-1]
            result['prev_day_high'] = round(prev_day['high'], 2)
            result['prev_day_low'] = round(prev_day['low'], 2)
            result['prev_day_close'] = round(prev_day['close'], 2)
            
            # Gap analysis
            if 'day_open' in result:
                gap = result['day_open'] - result['prev_day_close']
                result['gap'] = round(gap, 2)
                result['gap_pct'] = round((gap / result['prev_day_close']) * 100, 2)
                
                if abs(result['gap_pct']) < 0.1:
                    result['gap_type'] = 'FLAT'
                elif result['gap_pct'] > 0.5:
                    result['gap_type'] = 'GAP_UP'
                elif result['gap_pct'] < -0.5:
                    result['gap_type'] = 'GAP_DOWN'
                else:
                    result['gap_type'] = 'SMALL_GAP'
        
        # Get premarket levels
        pm_start = datetime.combine(date, dt_time(4, 0), tzinfo=TZ)
        pm_end = datetime.combine(date, dt_time(9, 30), tzinfo=TZ)
        
        pm_df = self.get_bars(symbol, TimeFrame(15, TimeFrameUnit.Minute), pm_start, pm_end)
        
        if pm_df is not None and not pm_df.empty:
            result['pm_high'] = round(pm_df['high'].max(), 2)
            result['pm_low'] = round(pm_df['low'].min(), 2)
        
        # VIX
        vix = self.get_vix_for_date(date)
        if vix:
            result['vix'] = vix
            if vix < 14:
                result['vix_regime'] = 'VERY_LOW'
            elif vix < 17:
                result['vix_regime'] = 'LOW'
            elif vix < 22:
                result['vix_regime'] = 'NORMAL'
            elif vix < 30:
                result['vix_regime'] = 'HIGH'
            else:
                result['vix_regime'] = 'VERY_HIGH'
        
        # Detect regime from price action
        if df_1m is not None and not df_1m.empty:
            # Count up vs down 5-min periods
            df_5m = df_1m.resample('5min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            if len(df_5m) > 0:
                up_bars = len(df_5m[df_5m['close'] > df_5m['open']])
                down_bars = len(df_5m[df_5m['close'] < df_5m['open']])
                total = up_bars + down_bars
                
                if total > 0:
                    result['up_bar_pct'] = round(up_bars / total * 100, 1)
                    
                    if down_bars > up_bars * 1.5:
                        result['day_regime'] = 'DOWNTREND'
                    elif up_bars > down_bars * 1.5:
                        result['day_regime'] = 'UPTREND'
                    else:
                        result['day_regime'] = 'CHOP'
        
        return result


# ============================================================================
# TRADE ANALYZER
# ============================================================================

def analyze_trades(trades, fetcher):
    """Analyze all trades and return results with market context"""
    results = []
    
    # Get unique date/symbol combinations
    date_symbols = set()
    for trade in trades:
        date_symbols.add((trade['trade_date'].date(), trade['underlying']))
    
    print(f"\nFetching market data for {len(date_symbols)} unique date/symbol combinations...")
    
    # Cache day stats
    day_stats_cache = {}
    
    for i, (date, symbol) in enumerate(sorted(date_symbols)):
        print(f"  [{i+1}/{len(date_symbols)}] {date} {symbol}")
        key = (date, symbol)
        day_stats_cache[key] = fetcher.get_day_stats(symbol, date)
    
    print(f"\nAnalyzing {len(trades)} trades...")
    
    for trade in trades:
        result = {}
        
        # Trade info
        result['date'] = trade['date']
        result['underlying'] = trade['underlying']
        result['action'] = trade['action']
        result['trans_code'] = trade['trans_code']
        result['option_type'] = trade['option_type']
        result['strike'] = trade['strike']
        result['expiry'] = trade['expiry']
        result['quantity'] = trade['quantity']
        result['price'] = trade['price']
        result['amount'] = trade['amount']
        result['description'] = trade['description']
        
        # Get day stats
        key = (trade['trade_date'].date(), trade['underlying'])
        day_stats = day_stats_cache.get(key, {})
        
        # Add all day stats to result
        for k, v in day_stats.items():
            result[k] = v
        
        # Calculate moneyness using day's range
        if 'day_open' in day_stats and 'day_close' in day_stats:
            avg_price = (day_stats['day_open'] + day_stats['day_close']) / 2
            
            if trade['option_type'] == 'CALL':
                result['moneyness'] = round(avg_price - trade['strike'], 2)
                result['moneyness_pct'] = round((avg_price / trade['strike'] - 1) * 100, 2)
            else:  # PUT
                result['moneyness'] = round(trade['strike'] - avg_price, 2)
                result['moneyness_pct'] = round((trade['strike'] / avg_price - 1) * 100, 2)
            
            # Classify
            if abs(result['moneyness_pct']) < 0.5:
                result['option_status'] = 'ATM'
            elif result['moneyness'] > 0:
                result['option_status'] = 'ITM'
            else:
                result['option_status'] = 'OTM'
            
            # Distance to key levels from strike
            if 'orb_high' in day_stats:
                result['strike_vs_orb_high'] = round(trade['strike'] - day_stats['orb_high'], 2)
            if 'orb_low' in day_stats:
                result['strike_vs_orb_low'] = round(trade['strike'] - day_stats['orb_low'], 2)
            if 'prev_day_high' in day_stats:
                result['strike_vs_prev_high'] = round(trade['strike'] - day_stats['prev_day_high'], 2)
        
        # Determine if trade was profitable (for SELL after BUY)
        # This would need matching logic which is complex, skip for now
        
        results.append(result)
    
    return results


def calculate_pnl(trades):
    """
    Match BUY and SELL trades to calculate P&L.
    Uses FIFO matching within same day/symbol/strike/type.
    """
    # Group trades by option contract
    contracts = defaultdict(list)
    
    for trade in trades:
        key = (trade['date'], trade['underlying'], trade['strike'], 
               trade['option_type'], trade['expiry'])
        contracts[key].append(trade)
    
    pnl_results = []
    
    for key, contract_trades in contracts.items():
        date, underlying, strike, opt_type, expiry = key
        
        buys = [t for t in contract_trades if t['action'] == 'BUY']
        sells = [t for t in contract_trades if t['action'] == 'SELL']
        
        total_buy_cost = sum(abs(t['amount']) for t in buys)
        total_sell_proceeds = sum(t['amount'] for t in sells)
        total_buy_qty = sum(t['quantity'] for t in buys)
        total_sell_qty = sum(t['quantity'] for t in sells)
        
        if total_buy_qty > 0 and total_sell_qty > 0:
            # Calculate P&L
            matched_qty = min(total_buy_qty, total_sell_qty)
            avg_buy_price = total_buy_cost / total_buy_qty if total_buy_qty > 0 else 0
            avg_sell_price = total_sell_proceeds / total_sell_qty if total_sell_qty > 0 else 0
            
            pnl = (avg_sell_price - avg_buy_price) * matched_qty
            pnl_pct = ((avg_sell_price / avg_buy_price) - 1) * 100 if avg_buy_price > 0 else 0
            
            pnl_results.append({
                'date': date,
                'underlying': underlying,
                'strike': strike,
                'option_type': opt_type,
                'expiry': expiry,
                'buy_qty': total_buy_qty,
                'sell_qty': total_sell_qty,
                'avg_buy_price': round(avg_buy_price / 100, 2),  # Per contract
                'avg_sell_price': round(avg_sell_price / 100, 2),
                'pnl': round(pnl, 2),
                'pnl_pct': round(pnl_pct, 2),
                'win': pnl > 0
            })
    
    return pnl_results


def export_results(results, pnl_results, output_prefix=None):
    """Export results to CSV files"""
    output_prefix = output_prefix or str(LOGS_DIR / 'trade_analysis')
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Main trades file
    trades_file = f"{output_prefix}_trades_{timestamp}.csv"
    df_trades = pd.DataFrame(results)
    
    # Reorder columns
    priority_cols = [
        'date', 'underlying', 'action', 'trans_code', 'option_type', 'strike', 'expiry',
        'quantity', 'price', 'amount', 'option_status', 'moneyness', 'moneyness_pct',
        'vix', 'vix_regime', 'day_regime', 'gap_type', 'gap_pct',
        'day_open', 'day_high', 'day_low', 'day_close', 'day_range_pct', 'day_direction',
        'orb_high', 'orb_low', 'orb_range',
        'prev_day_high', 'prev_day_low', 'pm_high', 'pm_low',
        'midday_rsi', 'close_rsi', 'up_bar_pct'
    ]
    
    cols = [c for c in priority_cols if c in df_trades.columns]
    cols += [c for c in df_trades.columns if c not in cols]
    df_trades = df_trades[cols]
    
    df_trades.to_csv(trades_file, index=False)
    print(f"\nExported trades to: {trades_file}")
    
    # P&L summary file
    if pnl_results:
        pnl_file = f"{output_prefix}_pnl_{timestamp}.csv"
        df_pnl = pd.DataFrame(pnl_results)
        df_pnl.to_csv(pnl_file, index=False)
        print(f"Exported P&L to: {pnl_file}")
        
        # Print summary
        wins = len([p for p in pnl_results if p['win']])
        losses = len([p for p in pnl_results if not p['win']])
        total_pnl = sum(p['pnl'] for p in pnl_results)
        
        print(f"\n{'='*60}")
        print("P&L SUMMARY")
        print(f"{'='*60}")
        print(f"Total Trades: {len(pnl_results)}")
        print(f"Wins: {wins} ({wins/len(pnl_results)*100:.1f}%)")
        print(f"Losses: {losses} ({losses/len(pnl_results)*100:.1f}%)")
        print(f"Total P&L: ${total_pnl:,.2f}")
        
        # By underlying
        print(f"\nBy Underlying:")
        for underlying in set(p['underlying'] for p in pnl_results):
            und_trades = [p for p in pnl_results if p['underlying'] == underlying]
            und_pnl = sum(p['pnl'] for p in und_trades)
            und_wins = len([p for p in und_trades if p['win']])
            print(f"  {underlying}: {len(und_trades)} trades, {und_wins} wins, ${und_pnl:,.2f}")
    
    return trades_file


def generate_summary_stats(results, pnl_results):
    """Generate summary statistics for pattern analysis"""
    df = pd.DataFrame(results)
    
    print(f"\n{'='*60}")
    print("PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    # Only analyze BUY entries
    buys = df[df['action'] == 'BUY']
    
    if len(buys) == 0:
        print("No BUY trades to analyze")
        return
    
    # By VIX regime
    if 'vix_regime' in buys.columns:
        print("\n📊 Trades by VIX Regime:")
        print(buys['vix_regime'].value_counts().to_string())
    
    # By day regime
    if 'day_regime' in buys.columns:
        print("\n📊 Trades by Day Regime:")
        print(buys['day_regime'].value_counts().to_string())
    
    # By option status (moneyness)
    if 'option_status' in buys.columns:
        print("\n📊 Trades by Moneyness:")
        print(buys['option_status'].value_counts().to_string())
    
    # By gap type
    if 'gap_type' in buys.columns:
        print("\n📊 Trades by Gap Type:")
        print(buys['gap_type'].value_counts().to_string())
    
    # By underlying
    print("\n📊 Trades by Underlying:")
    print(buys['underlying'].value_counts().to_string())
    
    # By option type
    print("\n📊 Trades by Option Type:")
    print(buys['option_type'].value_counts().to_string())
    
    # Average moneyness
    if 'moneyness_pct' in buys.columns:
        avg_moneyness = buys['moneyness_pct'].mean()
        print(f"\n📊 Average Moneyness at Entry: {avg_moneyness:.2f}%")
        
        otm_trades = buys[buys['option_status'] == 'OTM']
        if len(otm_trades) > 0:
            avg_otm = otm_trades['moneyness_pct'].mean()
            print(f"   Average OTM distance: {avg_otm:.2f}%")
    
    # VIX distribution
    if 'vix' in buys.columns:
        avg_vix = buys['vix'].mean()
        print(f"\n📊 Average VIX at Entry: {avg_vix:.2f}")
    
    # Day range
    if 'day_range_pct' in buys.columns:
        avg_range = buys['day_range_pct'].mean()
        print(f"\n📊 Average Day Range: {avg_range:.2f}%")
    
    # P&L by pattern if we have pnl_results
    if pnl_results:
        df_pnl = pd.DataFrame(pnl_results)
        
        # Merge with trade data for analysis
        # This is simplified - would need proper matching in production
        
        print(f"\n📊 Win Rate by Underlying:")
        for underlying in df_pnl['underlying'].unique():
            und_trades = df_pnl[df_pnl['underlying'] == underlying]
            win_rate = len(und_trades[und_trades['win']]) / len(und_trades) * 100
            print(f"   {underlying}: {win_rate:.1f}%")


# ============================================================================
# MAIN
# ============================================================================

def main():
    if len(sys.argv) < 2:
        print("Usage: python robinhood_trade_analyzer.py <trades.csv>")
        print("\nExample:")
        print("  python robinhood_trade_analyzer.py robinhood_trades.csv")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    
    if not os.path.exists(csv_file):
        print(f"ERROR: File not found: {csv_file}")
        sys.exit(1)
    
    print("="*60)
    print("ROBINHOOD TRADE JOURNAL ANALYZER v2.0")
    print("="*60)
    
    # Parse CSV
    print(f"\nParsing: {csv_file}")
    trades = parse_robinhood_csv(csv_file)
    
    if not trades:
        print("No trades found in CSV")
        sys.exit(1)
    
    # Calculate P&L
    print("\nCalculating P&L...")
    pnl_results = calculate_pnl(trades)
    
    # Fetch market data
    fetcher = MarketDataFetcher()
    results = analyze_trades(trades, fetcher)
    
    # Export
    output_file = export_results(results, pnl_results)
    
    # Summary
    generate_summary_stats(results, pnl_results)
    
    print(f"\n{'='*60}")
    print("DONE!")
    print(f"{'='*60}")
    print(f"\nOutput file: {output_file}")
    print("\nSend me this CSV and I'll analyze your patterns!")


if __name__ == "__main__":
    main()
