import pandas as pd
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
import time

from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

# Import your live trader class
from ..live_trading.multi_strategy_trader import MultiStrategyTrader, API_KEY, SECRET_KEY

class ReplaySimulator:
    def __init__(self, start_date, end_date, speed_multiplier=10):
        """
        speed_multiplier: 10 = 10x faster than real-time
        """
        self.start_date = start_date
        self.end_date = end_date
        self.speed = speed_multiplier
        self.tz = ZoneInfo("America/New_York")
        
        # Use your existing trader but in "replay mode"
        self.trader = MultiStrategyTrader(paper=True)
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        
    def fetch_replay_data(self):
        """Pre-fetch all 1-min bars for replay period"""
        print(f"Fetching data from {self.start_date} to {self.end_date}...")
        
        self.replay_bars = {}
        for symbol in self.trader.symbols:
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Minute),
                start=self.start_date,
                end=self.end_date,
                feed="sip"
            )
            bars = self.data_client.get_stock_bars(request)
            df = bars.df
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index(level='symbol', drop=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(self.tz)
            
            # Filter market hours only (9:30-16:00)
            df = df.between_time('09:30', '16:00')
            self.replay_bars[symbol] = df
            
        print(f"✓ Loaded {len(self.replay_bars['SPY'])} bars per symbol")
    
    def run_replay(self):
        self.fetch_replay_data()
        
        all_timestamps = pd.DatetimeIndex([])
        for df in self.replay_bars.values():
            all_timestamps = all_timestamps.union(df.index)
        all_timestamps = all_timestamps.sort_values()
        
        print(f"\n{'='*70}")
        print(f"REPLAY SIMULATION STARTED ({self.speed}x speed)")
        print(f"Days to simulate: {len(all_timestamps.date.unique())}")
        print(f"{'='*70}\n")
        
        # Track daily resets
        last_date = None
        orb_calculated = False
        eod_liquidated = False
        
        for current_time in all_timestamps:
            sim_time = current_time.time()
            current_date = current_time.date()
            
            # NEW DAY RESET
            if sim_time == datetime.strptime("09:30", "%H:%M").time():
                if last_date != current_date:
                    print(f"\n{'='*70}")
                    print(f"NEW TRADING DAY: {current_date}")
                    print(f"{'='*70}")
                    self.trader.daily_pnl = 0.0
                    self.trader.positions = {}
                    self.trader.entry_cooldown = {}
                    orb_calculated = False
                    eod_liquidated = False
                    last_date = current_date
                    self.trader.get_key_levels(force_update=True)
            
            # Skip pre-market
            if sim_time < datetime.strptime("09:30", "%H:%M").time():
                continue
            
            # Update bars
            self._update_trader_bars(current_time)
            
            # ORB calculation
            if sim_time >= datetime.strptime("09:45", "%H:%M").time() and not orb_calculated:
                print(f"\n[REPLAY] Calculating ORB at {current_time.strftime('%H:%M')}")
                self.trader.calculate_orb_levels()
                orb_calculated = True
            
            # EOD liquidation (ONCE per day)
            if sim_time >= datetime.strptime("15:55", "%H:%M").time() and not eod_liquidated:
                print(f"\n[REPLAY] EOD liquidation at {current_time.strftime('%H:%M')}")
                self.trader.liquidate_all_positions()
                eod_liquidated = True
            
            # Market close - continue to next day
            if sim_time >= datetime.strptime("16:00", "%H:%M").time():
                continue  # Don't break - process next day
            
            # Skip if already liquidated
            if eod_liquidated:
                continue
            
            # Exit checks
            self.trader.check_exits()
            
            # Strategy scans
            for symbol in self.trader.symbols:
                if not self.trader.can_trade(symbol, "any"):
                    continue
                
                for strategy_func in self.trader.strategy_functions:
                    signal = strategy_func(symbol)
                    if signal and self.trader.can_trade(symbol, signal['strategy_name']):
                        self.trader.place_trade(symbol, signal)
                        break
            
            # Simulate time passing (speed up replay)
            time.sleep(0.001)  # 1ms per bar = ~6.5 min per day at 390 bars/day
        
        print(f"\n{'='*70}")
        print(f"REPLAY COMPLETE - {len(all_timestamps.date.unique())} days simulated")
        print(f"{'='*70}\n")




def _update_trader_bars(self, current_time):
    for symbol in self.trader.symbols:
        bars_so_far = self.replay_bars[symbol].loc[:current_time]
        self.trader.bar_data[f"{symbol}_1m"] = bars_so_far
        
        # FIX: Use 'min' instead of 'T'
        bars_2m = bars_so_far.resample('2min').agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'vwap': 'mean'
        }).dropna()
        self.trader.bar_data[f"{symbol}_2m"] = bars_2m




    def _update_trader_bars(self, current_time):
        """Inject bars up to current_time into trader's bar_data"""
        for symbol in self.trader.symbols:
            # Get all bars up to current_time (simulates "data available so far")
            bars_so_far = self.replay_bars[symbol].loc[:current_time]
            
            # Store in trader's expected format
            self.trader.bar_data[f"{symbol}_1m"] = bars_so_far
            
            # Also create 2m bars for ORB strategy
            bars_2m = bars_so_far.resample('2min').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'vwap': 'mean'
            }).dropna()
            self.trader.bar_data[f"{symbol}_2m"] = bars_2m

# ===== USAGE =====
if __name__ == "__main__":
    # Test last week's trading
    # end = datetime.now(ZoneInfo("America/New_York"))
    # start = end - timedelta(days=7)
    

    # Option 2: Specific date range
    start = datetime(2024, 11, 1, tzinfo=ZoneInfo("America/New_York"))
    end = datetime(2025, 11, 18, tzinfo=ZoneInfo("America/New_York"))
    simulator = ReplaySimulator(
        start_date=start,
        end_date=end,
        speed_multiplier=60  # 60x = 1 trading day in ~6 minutes
    )
    
    simulator.run_replay()
