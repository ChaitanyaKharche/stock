import os
import sys
import pandas as pd
import numpy as np
import pandas_ta as ta
from datetime import datetime, time as dt_time, timedelta
from zoneinfo import ZoneInfo
from backtesting import Backtest, Strategy
from backtesting.lib import resample_apply

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..paths import LOGS_DIR

# --- ALPACA DATA LOADER ---
API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')
TZ = ZoneInfo("America/New_York")

def fetch_alpaca_data(symbol, start_date_str, end_date_str):
    """
    Fetches 1-minute bar data from Alpaca and enriches it
    with resampled 2-minute data for the backtest.
    """
    print(f"Fetching 1-min data for {symbol} from {start_date_str} to {end_date_str}...")
    client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
    
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=TimeFrame(1, TimeFrameUnit.Minute),
        start=datetime.strptime(start_date_str, '%Y-%m-%d').astimezone(TZ),
        end=datetime.strptime(end_date_str, '%Y-%m-%d').astimezone(TZ),
        feed="sip" 
    )
    
    bars = client.get_stock_bars(request)
    if bars.df.empty:
        raise ValueError(f"No data returned for {symbol}.")
        
    df = bars.df
    if isinstance(df.index, pd.MultiIndex):
        df = df.reset_index(level='symbol', drop=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')
    df.index = df.index.tz_convert(TZ)
    
    # Format for backtesting.py
    df.rename(columns={
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }, inplace=True)
    
    # Filter for market hours only
    df = df.between_time('09:30', '16:00')
    df = df.dropna()
    
    if df.empty:
        raise ValueError("No market-hour data found.")

    print(f"Fetch complete. {len(df)} bars loaded. Now enriching data...")

    # --- Pre-calculate VWAP (on 1-min data) ---
    df['VWAP'] = ta.vwap(df.High, df.Low, df.Close, df.Volume)

    # --- Pre-calculate 2-Min data ---
    df_2m = df.resample('2min').agg({
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'VWAP': 'last' # Use the last 1-min VWAP in the 2-min window
    })
    
    # Add _2m suffix
    df_2m = df_2m.add_suffix('_2m')
    
    # Merge 2-min data back into 1-min data (forward-fill)
    df = pd.merge_asof(df, df_2m, left_index=True, right_index=True)
    
    # Forward fill the resampled data
    df.fillna(method='ffill', inplace=True)
    df = df.dropna() # Drop any remaining NaNs at the beginning

    print(f"Enrichment complete.")
    return df

# --- STRATEGY DEFINITION ---

class MultiStrategyBT(Strategy):
    """
    Backtesting class for our Hourly Retest and ORB Reversal strategies.
    """
    retest_dist_pct = 0.15   # 0.15% retest zone
    cooldown_minutes = 5     # 5-min cooldown after exit
    default_stop_pct = 0.3   # 0.3% stop loss
    default_target_pct = 0.4 # 0.4% take profit

    def init(self):
        """
        Called once at the start. We use this to pre-calculate
        all our daily/hourly/ORB levels.
        """
        print("Initializing strategy... Pre-calculating levels...")
        
        # --- Create pd.Series from raw data ---
        high_series = pd.Series(self.data.High, index=self.data.index)
        low_series = pd.Series(self.data.Low, index=self.data.index)
        close_series = pd.Series(self.data.Close, index=self.data.index)
        volume_series = pd.Series(self.data.Volume, index=self.data.index)
        open_series = pd.Series(self.data.Open, index=self.data.index)
        
        # --- 1. Manually calculate VWAP ---
        vwap_series = ta.vwap(high=high_series, low=low_series, close=close_series, volume=volume_series)
        self.vwap = self.I(lambda: vwap_series, name="VWAP")

        # --- 2. Pre-calculate 2-Min data ---
        self.data_2m = {
            'Open': self.I(lambda: open_series.resample('2min').first().reindex(self.data.index, method='ffill'), name="Open_2m"),
            'High': self.I(lambda: high_series.resample('2min').max().reindex(self.data.index, method='ffill'), name="High_2m"),
            'Low': self.I(lambda: low_series.resample('2min').min().reindex(self.data.index, method='ffill'), name="Low_2m"),
            'Close': self.I(lambda: close_series.resample('2min').last().reindex(self.data.index, method='ffill'), name="Close_2m"),
            'VWAP': self.I(lambda: vwap_series.resample('2min').last().reindex(self.data.index, method='ffill'), name="VWAP_2m"),
        }

        # --- 3. Pre-calculate Daily & Hourly Levels ---
        
        # Daily
        daily_data = high_series.resample('D')
        y_high = daily_data.max().shift(1).reindex(self.data.index, method='ffill')
        y_low = low_series.resample('D').min().shift(1).reindex(self.data.index, method='ffill')

        # Hourly
        hourly_data = high_series.resample('H')
        h1_high = hourly_data.max().shift(1).reindex(self.data.index, method='ffill')
        h1_low = low_series.resample('H').min().shift(1).reindex(self.data.index, method='ffill')
        
        # ORB (9:30 - 9:45)
        df_orb = pd.DataFrame({'High': high_series, 'Low': low_series})
        orb_data = df_orb.between_time('09:30', '09:45').resample('D').agg(High=('High', 'max'), Low=('Low', 'min'))
        orb_high = orb_data.High.reindex(self.data.index, method='ffill')
        orb_low = orb_data.Low.reindex(self.data.index, method='ffill')

        # Store all levels in a single, accessible dict
        self.levels = {
            'support': [
                ('Y-Low', self.I(lambda: y_low, name="Y_Low")),
                ('H1-Low', self.I(lambda: h1_low, name="H1_Low")),
                ('ORB-Low', self.I(lambda: orb_low, name="ORB_Low")),
            ],
            'resistance': [
                ('Y-High', self.I(lambda: y_high, name="Y_High")),
                ('H1-High', self.I(lambda: h1_high, name="H1_High")),
                ('ORB-High', self.I(lambda: orb_high, name="ORB_High")),
            ]
        }
        
        # --- FIX: Cast cooldown to int() to prevent numpy type error ---
        self.last_exit_time = self.data.index[0] - timedelta(minutes=int(self.cooldown_minutes) + 1)




    def next(self):
        """
        This function is the "time machine."
        It runs once for *every single 1-minute bar* in our dataset.
        """
        
        current_time = self.data.index[-1]
        
        # --- 0. EOD LIQUIDATION ---
        if current_time.time() >= dt_time(15, 55):
            if self.position:
                self.position.close()
                self.last_exit_time = current_time # Set cooldown
            return # Stop trading for the day

        # --- 1. CHECK FOR RECENT EXIT (to set cooldown) ---
        if self.trades:
            # Check if the last closed trade exited on the *previous* bar
            if self.trades[-1].exit_time == self.data.index[-2]:
                self.last_exit_time = self.data.index[-2]
        
        # --- 2. CHECK COOLDOWN & POSITION ---
        if self.position:
            return # Already in a trade
            
        if (current_time - self.last_exit_time).total_seconds() < self.cooldown_minutes * 60:
            return # In cooldown

        # --- 3. PREPARE STOPS/TARGETS ---
        current_close = self.data.Close[-1]
        sl_pct = self.default_stop_pct / 100
        tp_pct = self.default_target_pct / 100
        
        buy_stop = current_close * (1 - sl_pct)
        buy_target = current_close * (1 + tp_pct)
        sell_stop = current_close * (1 + sl_pct)
        sell_target = current_close * (1 - tp_pct)

        # --- 4. RUN STRATEGY LOGIC ---
        
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]
        current_open = self.data.Open[-1]

        # === STRATEGY 1: HOURLY RETEST ===
        
        # Check for Retest of Support (for CALLS)
        for level_name, level_data in self.levels['support']:
            level_price = level_data[-1] 
            if pd.isna(level_price): continue
            
            retest_zone_high = level_price * (1 + self.retest_dist_pct / 100)
            retest_zone_low = level_price * (1 - self.retest_dist_pct / 100)
            
            if retest_zone_low <= current_low <= retest_zone_high:
                is_bullish_rejection = current_close > current_open and \
                                       current_close > (current_high + current_low) / 2
                if is_bullish_rejection:
                    self.buy(sl=buy_stop, tp=buy_target)
                    return 

        # Check for Retest of Resistance (for PUTS)
        for level_name, level_data in self.levels['resistance']:
            level_price = level_data[-1]
            if pd.isna(level_price): continue
            
            retest_zone_high = level_price * (1 + self.retest_dist_pct / 100)
            retest_zone_low = level_price * (1 - self.retest_dist_pct / 100)
            
            if retest_zone_low <= current_high <= retest_zone_high:
                is_bearish_rejection = current_close < current_open and \
                                       current_close < (current_high + current_low) / 2
                if is_bearish_rejection:
                    self.sell(sl=sell_stop, tp=sell_target)
                    return
        
        # === STRATEGY 2: ORB REVERSAL (2-Min Chart) ===
        
        # Check if the 2-min bar just changed
        if len(self.data) < 2 or self.data.Close_2m[-1] != self.data.Close_2m[-2]:
            prev_2m_high = self.data.High_2m[-2]
            prev_2m_low = self.data.Low_2m[-2]
            prev_2m_close = self.data.Close_2m[-2]
        else:
            return # Still in the same 2-min candle
            
        orb_high = self.levels['resistance'][2][1][-1] # ORB-High data
        orb_low = self.levels['support'][2][1][-1]    # ORB-Low data
        if pd.isna(orb_high) or pd.isna(orb_low): return 

        target_vwap = self.vwap[-1]

        # PUT Signal
        if (prev_2m_high > orb_high and prev_2m_close < orb_high):
            self.sell(sl=orb_high, tp=target_vwap)
            return
            
        # CALL Signal
        if (prev_2m_low < orb_low and prev_2m_close > orb_low):
            self.buy(sl=orb_low, tp=target_vwap)
            return
# --- BACKTEST EXECUTION ---

if __name__ == "__main__":
    
    # --- 1. SET PARAMETERS ---
    SYMBOL = 'QQQ'
    START_DATE = '2025-10-17' 
    END_DATE = '2025-11-16'
    CASH = 100_000
    COMMISSION_BPS = 0.001 
    
    # --- 2. LOAD DATA ---
    try:
        data = fetch_alpaca_data(SYMBOL, START_DATE, END_DATE)
    except Exception as e:
        print(f"\nFATAL ERROR: Could not load data. {e}")
        sys.exit(1)

    # --- 3. RUN BACKTEST (with our default guesses) ---
    print("\nRunning backtest with default parameters...")
    bt = Backtest(
        data,
        MultiStrategyBT,
        cash=CASH,
        commission=COMMISSION_BPS / 100, 
        trade_on_close=False, 
        exclusive_orders=True 
    )
    
    stats = bt.run()
    print("\n" + "="*70)
    print(f"DEFAULT RESULTS (Our Guesses)")
    print("="*70)
    print(stats)
    
    # --- 4. RUN OPTIMIZATION ---
    print("\n" + "="*70)
    print(f"RUNNING OPTIMIZATION... (This will take a few minutes)")
    print("="*70)
    
    opt_stats = bt.optimize(
        # We tell it which parameters to test and what range
        retest_dist_pct=[0.1, 0.15, 0.2],         # Test 3 values for retest zone
        cooldown_minutes=[3, 5, 10],             # Test 3 values for cooldown
        default_stop_pct=[0.2, 0.3, 0.5],        # Test 3 values for stop-loss
        default_target_pct=[0.3, 0.4, 0.6],      # Test 3 values for profit-target
        maximize='Sharpe Ratio',                 # Find the best risk-adjusted return
        constraint=lambda p: p.default_target_pct > p.default_stop_pct # Only test if target > stop
    )
    
    # --- 5. SHOW OPTIMIZED RESULTS ---
    print("\n" + "="*70)
    print(f"OPTIMIZED RESULTS")
    print("="*70)
    print(opt_stats)
    
    print("\n" + "="*70)
    print("BEST PARAMETERS FOUND:")
    print("="*70)
    print(opt_stats._strategy)

    print("\n" + "="*70)
    print("OPTIMIZED TRADES")
    print("="*70)
    print(opt_stats['_trades'])

    # --- 6. PLOT THE OPTIMIZED STRATEGY ---
    print("\nGenerating plot for *Optimized* Strategy...")
    opt_stats.plot(filename=str(LOGS_DIR / "multi_strategy_optimized_backtest"))