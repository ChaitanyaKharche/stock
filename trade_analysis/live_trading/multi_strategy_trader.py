import os
import sys
import time
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo
from ta.momentum import RSIIndicator

from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

from ..signals import gamma_exposure
from ..paths import LOGS_DIR

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

if not API_KEY or not SECRET_KEY:
    print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
    sys.exit(1)

class MultiStrategyTrader:
    def __init__(self, paper=True):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=paper)
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.paper = paper
        self.log_file = LOGS_DIR / "strategy_trader_log.txt"
        self.tz = ZoneInfo("America/New_York")
        
        # Watchlist
        self.symbols = ['SPY', 'QQQ']
        
        # Position tracking
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = -200.0 # Daily stop
        
        # --- OPTIMIZED PARAMETERS ---
        self.entry_cooldown_minutes = 3 # OPTIMIZED from 5
        self.entry_cooldown = {} # {symbol: timestamp}
        
        # Strategy Parameters (From Optimization)
        self.default_target_pct = 0.6 # OPTIMIZED from 0.4
        self.default_stop_pct = 0.5   # OPTIMIZED from 0.3
        self.retest_dist_pct = 0.2    # OPTIMIZED from 0.15
        # --- END OPTIMIZED PARAMETERS ---

        # ATR-based risk (scales target/stop to current volatility instead
        # of the fixed percentages above, which are used only as a fallback
        # when ATR isn't available yet)
        self.atr_period = 14
        self.target_atr_mult = 2.5
        self.stop_atr_mult = 1.0
        self.atr = {}            # {symbol: atr_value}
        self.yesterday_close = {}  # {symbol: float}, gap-regime baseline

        # Multi-day S/R confluence: a breakout level that lines up with
        # another recent level (session or premarket, last N days) is
        # higher-conviction and gets sized up
        self.confluence_lookback_days = 3
        self.confluence_zone_pct = 0.15  # % tolerance to count as the same zone
        self.confluence_size_mult = 1.5

        # Gap / volatility regime filter: a headline/news-driven gap well
        # beyond the recent ATR derates size instead of trusting static
        # S/R levels to hold
        self.gap_atr_mult_threshold = 2.0
        self.elevated_regime_size_mult = 0.5
        self.gap_regime = {}      # {symbol: "NORMAL" | "ELEVATED"}
        self.regime_checked_today = False

        # Dealer gamma-exposure (GEX) regime: NET_LONG_GAMMA dealers
        # dampen/pin price (derate breakout size), NET_SHORT_GAMMA dealers
        # amplify moves (breakouts more likely to run, keep full size)
        self.gex_long_gamma_size_mult = 0.6
        self.gex_regime = {}      # {symbol: "NET_LONG_GAMMA" | "NET_SHORT_GAMMA" | "UNKNOWN"}

        self.base_shares = 100

        # Cached Data
        self.key_levels = {} # {symbol: {'support': [], 'resistance': []}}
        self.last_level_update = None
        self.orb_levels = {} # {symbol: {'high': float, 'low': float}}
        self.last_orb_calc = None
        
        # Multi-Timeframe bar data
        self.bar_data = {} # { 'QQQ_1m': df, 'QQQ_2m': df, ... }
        
        # List of strategies to run
        self.strategy_functions = [
            self.strategy_1_hourly_retest,
            self.strategy_2_orb_reversal,
            self.strategy_3_false_breakout_fade
        ]
        
        self.log("="*70)
        self.log(f"MULTI-STRATEGY TRADER (v3 - Optimized) STARTED")
        self.log(f"Mode: {'PAPER' if paper else 'LIVE'}")
        self.log(f"Symbols: {', '.join(self.symbols)}")
        self.log(f"Params: TP={self.default_target_pct}% | SL={self.default_stop_pct}% | Cooldown={self.entry_cooldown_minutes}min | RetestZone={self.retest_dist_pct}%")
        self.log("="*70)

    def log(self, message):
        timestamp = datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{timestamp}] {message}"
        print(msg)
        with open(self.log_file, 'a', encoding='utf-8') as f:
            f.write(msg + "\n")

    def get_bars(self, symbol, timeframe, lookback_days=5):
        """Flexible bar getter"""
        try:
            now = datetime.now(self.tz)
            start = now - timedelta(days=lookback_days)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=timeframe,
                start=start,
                end=now,
                feed="sip" 
            )
            bars_response = self.data_client.get_stock_bars(request)
            
            if bars_response is None or bars_response.df.empty:
                return None
            
            df = bars_response.df
            # Handle multi-index if present
            if isinstance(df.index, pd.MultiIndex):
                if 'symbol' in df.index.names:
                    df = df.reset_index(level='symbol', drop=True)
                else:
                    # Fallback for unexpected multi-index
                    df = df.droplevel(0)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(self.tz)
            
            return df
        
        except Exception as e:
            self.log(f"[ERROR {symbol}] get_bars {timeframe}: {str(e)[:80]}")
            return None

    def update_bar_data(self):
        """Updates ALL timeframes for all symbols and calculates VWAP"""
        from ta.volume import VolumeWeightedAveragePrice # Import here or at top
        
        for symbol in self.symbols:
            # 1-Minute data
            df_1m = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), lookback_days=1)
            if df_1m is not None:
                # --- NEW: Calculate VWAP ---
                vwap = VolumeWeightedAveragePrice(
                    high=df_1m['high'], 
                    low=df_1m['low'], 
                    close=df_1m['close'], 
                    volume=df_1m['volume'],
                    window=len(df_1m) # Cumulative VWAP for the loaded data
                )
                df_1m['vwap'] = vwap.volume_weighted_average_price()
                # ---------------------------
                self.bar_data[f"{symbol}_1m"] = df_1m
            
            # 2-Minute data (for ORB)
            df_2m = self.get_bars(symbol, TimeFrame(2, TimeFrameUnit.Minute), lookback_days=1)
            if df_2m is not None:
                # Add VWAP to 2m as well for ORB targets
                vwap_2m = VolumeWeightedAveragePrice(
                    high=df_2m['high'], low=df_2m['low'], close=df_2m['close'], volume=df_2m['volume'], window=len(df_2m)
                )
                df_2m['vwap'] = vwap_2m.volume_weighted_average_price()
                self.bar_data[f"{symbol}_2m"] = df_2m

    def get_key_levels(self, force_update=False):
        """Gets Pre-Market H/L, Yesterday's H/L, and 60-Min levels."""
        now = datetime.now(self.tz)
        if (not force_update and self.last_level_update and
            (now - self.last_level_update).total_seconds() < 3600):
            return 

        self.log("Fetching key levels (PM, Daily, 60-Min)...")
        
        for symbol in self.symbols:
            self.key_levels[symbol] = {'support': [], 'resistance': []}
            
            # 1. Daily bars for PM, Yesterday's, and multi-day confluence levels
            daily_df = self.get_bars(
                symbol, TimeFrame(1, TimeFrameUnit.Day),
                lookback_days=max(5, self.confluence_lookback_days + 3)
            )
            if daily_df is None or len(daily_df) < 2:
                continue

            yesterday = daily_df.iloc[-2]
            self.key_levels[symbol]['resistance'].append((f"Y-High", yesterday['high']))
            self.key_levels[symbol]['support'].append((f"Y-Low", yesterday['low']))
            self.yesterday_close[symbol] = yesterday['close']

            # 1b. Prior sessions (day -2, -3, ...) for multi-day S/R confluence
            for i in range(2, self.confluence_lookback_days + 1):
                idx = -(i + 1)
                if abs(idx) <= len(daily_df):
                    session = daily_df.iloc[idx]
                    self.key_levels[symbol]['resistance'].append((f"Session High -{i}d", session['high']))
                    self.key_levels[symbol]['support'].append((f"Session Low -{i}d", session['low']))

            # 2. Premarket Levels
            try:
                today_date = now.date()
                pm_start = datetime.combine(today_date, dt_time(4, 0), tzinfo=self.tz)
                pm_end = datetime.combine(today_date, dt_time(9, 30), tzinfo=self.tz)
                
                pm_request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=TimeFrame(15, TimeFrameUnit.Minute), # 15min is efficient
                    start=pm_start,
                    end=pm_end,
                    feed="sip"
                )
                pm_response = self.data_client.get_stock_bars(pm_request)
                if pm_response is not None and not pm_response.df.empty:
                    pm_df = pm_response.df
                    if isinstance(pm_df.index, pd.MultiIndex):
                        pm_df = pm_df.reset_index(level='symbol', drop=True)
                    
                    self.key_levels[symbol]['resistance'].append((f"PM-High", pm_df['high'].max()))
                    self.key_levels[symbol]['support'].append((f"PM-Low", pm_df['low'].min()))
            except Exception as e:
                self.log(f"[WARN {symbol}] Could not get PM levels: {e}")

            # 3. Hourly Levels
            hourly_df = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Hour), lookback_days=10)
            if hourly_df is not None:
                for i in range(1, 6):
                    if i < len(hourly_df):
                        self.key_levels[symbol]['resistance'].append((f"H1 High {-i}", hourly_df.iloc[-i]['high']))
                        self.key_levels[symbol]['support'].append((f"H1 Low {-i}", hourly_df.iloc[-i]['low']))

            self.key_levels[symbol]['support'] = sorted(list(set(self.key_levels[symbol]['support'])), key=lambda x: x[1], reverse=True)
            self.key_levels[symbol]['resistance'] = sorted(list(set(self.key_levels[symbol]['resistance'])), key=lambda x: x[1])
            
            self.log(f"--- {symbol} Levels ---")
            for name, price in self.key_levels[symbol]['resistance'][:3]:
                self.log(f"  RES: {name} @ {price:.2f}")
            for name, price in self.key_levels[symbol]['support'][:3]:
                self.log(f"  SUP: {name} @ {price:.2f}")

        self.last_level_update = now

    def calculate_orb_levels(self):
        """Calculates the 15-min ORB H/L once per day."""
        now = datetime.now(self.tz)
        if self.last_orb_calc and self.last_orb_calc.date() == now.date():
            return # Already calculated today
        
        self.log("Calculating 15-min ORB levels (9:30 - 9:45)...")
        orb_start = datetime.combine(now.date(), dt_time(9, 30), tzinfo=self.tz)
        orb_end = datetime.combine(now.date(), dt_time(9, 45), tzinfo=self.tz)
        
        for symbol in self.symbols:
            try:
                # Use 1-min bars to get precise range
                bars_1m = self.bar_data.get(f"{symbol}_1m")
                if bars_1m is None:
                    # Fetch if not in cache
                    bars_1m = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Minute), lookback_days=1)
                
                orb_bars = bars_1m.between_time(orb_start.time(), orb_end.time())
                
                if not orb_bars.empty:
                    orb_high = orb_bars['high'].max()
                    orb_low = orb_bars['low'].min()
                    self.orb_levels[symbol] = {'high': orb_high, 'low': orb_low}
                    self.log(f"  {symbol} ORB: High ${orb_high:.2f}, Low ${orb_low:.2f}")
                else:
                    self.log(f"[WARN {symbol}] No bars found for 9:30-9:45 to calc ORB.")
            
            except Exception as e:
                self.log(f"[ERROR {symbol}] Failed to calc ORB: {e}")
                
        self.last_orb_calc = now

    def update_atr(self):
        """5-min ATR(14) per symbol, so target/stop/retest zones scale with
        current volatility instead of the fixed default_target_pct/stop_pct."""
        for symbol in self.symbols:
            df_5m = self.get_bars(symbol, TimeFrame(5, TimeFrameUnit.Minute), lookback_days=3)
            if df_5m is None or len(df_5m) < 2:
                continue

            high, low, close = df_5m['high'], df_5m['low'], df_5m['close']
            prev_close = close.shift(1)
            true_range = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1
            ).max(axis=1)

            atr = true_range.tail(self.atr_period).mean()
            if pd.notna(atr) and atr > 0:
                self.atr[symbol] = atr

    def check_confluence(self, symbol, level_price, side):
        """Does level_price line up with another recent S/R zone (within
        confluence_zone_pct)? Multiple touches = higher-conviction breakout."""
        levels = self.key_levels.get(symbol, {}).get(side, [])
        if not levels or level_price is None:
            return False, []

        matches = [
            name for name, price in levels
            if price != level_price and abs(price - level_price) / level_price <= self.confluence_zone_pct / 100
        ]
        return len(matches) > 0, matches

    def check_gap_regime(self, symbol, current_price):
        """Classify today's gap vs yesterday's close relative to ATR. A
        headline/news-driven gap well beyond the recent ATR flags an
        ELEVATED regime so size gets derated instead of trusting static
        S/R levels to hold."""
        y_close = self.yesterday_close.get(symbol)
        atr = self.atr.get(symbol)

        if y_close is None or not atr:
            self.gap_regime[symbol] = "NORMAL"
            return self.gap_regime[symbol]

        gap = abs(current_price - y_close)
        self.gap_regime[symbol] = "ELEVATED" if gap > self.gap_atr_mult_threshold * atr else "NORMAL"
        return self.gap_regime[symbol]

    def fetch_gex_regime(self, symbol):
        """Pulls today's dealer gamma-exposure regime from the live 0DTE
        option chain. Also folds the top gamma-concentration strikes into
        both key_levels sides, so a breakout aligning with a major dealer
        hedging strike scores as confluence too."""
        try:
            result = gamma_exposure.compute_gex(symbol)
            if 'error' in result:
                print(f"[GEX] {symbol}: {result['error']}")
                self.gex_regime[symbol] = "UNKNOWN"
                return self.gex_regime[symbol]

            self.gex_regime[symbol] = result['regime']
            for strike, gex in result['top_gamma_strikes']:
                label = f"Gamma strike ${strike:.2f} ({gex:+,.0f})"
                self.key_levels[symbol]['resistance'].append((label, strike))
                self.key_levels[symbol]['support'].append((label, strike))

            return self.gex_regime[symbol]
        except Exception as e:
            print(f"[ERROR {symbol}] fetch_gex_regime: {e}")
            self.gex_regime[symbol] = "UNKNOWN"
            return self.gex_regime[symbol]

    def score_signal_size(self, symbol, signal):
        """Combine multi-day confluence + gap regime into a size multiplier
        for a detected signal, and attach it to the signal dict."""
        side = 'support' if signal['direction'] == 'UP' else 'resistance'
        level_price = signal.get('level_price')

        has_confluence, matches = self.check_confluence(symbol, level_price, side)
        size_mult = self.confluence_size_mult if has_confluence else 1.0

        notes = []
        if self.gap_regime.get(symbol) == "ELEVATED":
            size_mult *= self.elevated_regime_size_mult
            notes.append("ELEVATED gap regime, size derated")
        elif has_confluence:
            notes.append(f"confluence with {matches}")

        if self.gex_regime.get(symbol) == "NET_LONG_GAMMA":
            size_mult *= self.gex_long_gamma_size_mult
            notes.append("NET_LONG_GAMMA (dealer pinning), size derated")
        elif self.gex_regime.get(symbol) == "NET_SHORT_GAMMA":
            notes.append("NET_SHORT_GAMMA (dealer amplifying)")

        signal['size_multiplier'] = size_mult
        if notes:
            signal['level_name'] = f"{signal['level_name']} | {' | '.join(notes)}"

        return signal

    def can_trade(self, symbol, strategy_name):
        """Check position and cooldown"""
        if self.daily_pnl <= self.max_daily_loss:
            return False 
        
        if symbol in self.positions:
            return False 
        
        if symbol in self.entry_cooldown:
            last_entry_time = self.entry_cooldown[symbol]
            minutes_since = (datetime.now(self.tz) - last_entry_time).total_seconds() / 60
            if minutes_since < self.entry_cooldown_minutes:
                return False
        
        return True

    def place_trade(self, symbol, signal):
        """Universal trade placement. Target/stop scale with ATR (fixed
        default_target_pct/stop_pct are only a fallback), and shares scale
        with the confluence/gap-regime size_multiplier from score_signal_size."""
        try:
            entry_price = signal['entry_price']
            atr = self.atr.get(symbol)
            size_mult = signal.get('size_multiplier', 1.0)

            if 'target_price' in signal:
                target = signal['target_price']
            else:
                target_dist = (self.target_atr_mult * atr) if atr else entry_price * (self.default_target_pct / 100.0)
                target = entry_price + target_dist if signal['direction'] == "UP" else entry_price - target_dist

            if 'stop_price' in signal:
                stop = signal['stop_price']
            else:
                stop_dist = (self.stop_atr_mult * atr) if atr else entry_price * (self.default_stop_pct / 100.0)
                stop = entry_price - stop_dist if signal['direction'] == "UP" else entry_price + stop_dist

            shares = int(self.base_shares * size_mult)

            self.positions[symbol] = {
                'type': signal['type'],
                'direction': signal['direction'],
                'entry': entry_price,
                'target': target,
                'stop': stop,
                'shares': shares,
                'entry_time': signal['time'],
                'strategy': signal['strategy_name']
            }
            self.entry_cooldown[symbol] = datetime.now(self.tz)

            self.log("\n" + "="*70)
            self.log(f"🚀 [{signal['strategy_name']} {signal['type']} ENTRY]")
            self.log(f"  Symbol: {symbol} @ ${entry_price:.2f}")
            self.log(f"  Trigger: {signal['level_name']}")
            self.log(f"  Shares: {shares} (size x{size_mult:.2f}, regime: {self.gap_regime.get(symbol, 'NORMAL')})")
            self.log(f"  Target: ${target:.2f} | Stop: ${stop:.2f}" + (f" (ATR: {atr:.3f})" if atr else ""))
            self.log("="*70 + "\n")

            # TODO: Add actual Alpaca order execution here

        except Exception as e:
            self.log(f"[ERROR] place_trade: {e}")

    def check_exits(self):
        """Universal exit checker, now with cooldown on exit."""
        if not self.positions:
            return

        for symbol in list(self.positions.keys()):
            try:
                pos = self.positions[symbol]
                bars = self.bar_data.get(f"{symbol}_1m")
                if bars is None or bars.empty:
                    continue

                current_price = bars.iloc[-1]['close']
                shares = pos.get('shares', self.base_shares)

                realized_pnl = 0.0
                exit_type = None

                if pos['direction'] == "UP":
                    if current_price >= pos['target']:
                        exit_type = "TARGET"
                        realized_pnl = (pos['target'] - pos['entry']) * shares
                    elif current_price <= pos['stop']:
                        exit_type = "STOP"
                        realized_pnl = (pos['stop'] - pos['entry']) * shares

                else: # DOWN
                    if current_price <= pos['target']:
                        exit_type = "TARGET"
                        realized_pnl = (pos['entry'] - pos['target']) * shares
                    elif current_price >= pos['stop']:
                        exit_type = "STOP"
                        realized_pnl = (pos['entry'] - pos['stop']) * shares

                if exit_type:
                    self.daily_pnl += realized_pnl

                    self.log("\n" + "="*70)
                    self.log(f"🛑 [{exit_type} EXIT] ({pos['strategy']})")
                    self.log(f"  Symbol: {symbol} @ ${current_price:.2f}")
                    self.log(f"  Entry: ${pos['entry']:.2f}")
                    self.log(f"  P&L: ${realized_pnl:+.2f}")
                    self.log(f"  Daily Total P&L: ${self.daily_pnl:.2f}")
                    self.log("="*70 + "\n")

                    # --- THIS IS THE FIX ---
                    del self.positions[symbol]
                    self.entry_cooldown[symbol] = datetime.now(self.tz)
                    # ---------------------

                    # TODO: Add actual Alpaca exit order here

            except Exception as e:
                self.log(f"[ERROR] check_exits {symbol}: {e}")

    # --- STRATEGY FUNCTIONS ---
    
    def strategy_1_hourly_retest(self, symbol):
        """
        YOUR CORE STRATEGY (Now with VWAP Trend Filter):
        1. Identifies key 60min + Daily levels
        2. Checks VWAP trend (Don't buy calls below VWAP)
        3. Enters on a 1-min rejection candle
        """
        bars_1m = self.bar_data.get(f"{symbol}_1m")
        if bars_1m is None or len(bars_1m) < 2 or symbol not in self.key_levels:
            return None 

        latest_bar = bars_1m.iloc[-1]
        current_close = latest_bar['close']
        current_low = latest_bar['low']
        current_high = latest_bar['high']
        current_vwap = latest_bar['vwap']
        
        # --- VWAP FILTER ---
        # If close > VWAP, we prefer Calls. If close < VWAP, we prefer Puts.
        # We will STRICTLY enforce this to prevent catching falling knives.
        trend_is_bullish = current_close > current_vwap
        trend_is_bearish = current_close < current_vwap
        # -------------------

        # Check for Retest of Support (for CALLS)
        # ONLY if we are above VWAP or very close to it (Trend is acceptable)
        if trend_is_bullish: 
            for level_name, level_price in self.key_levels[symbol]['support']:
                retest_zone_high = level_price * (1 + self.retest_dist_pct / 100) 
                retest_zone_low = level_price * (1 - self.retest_dist_pct / 100)
                
                if retest_zone_low <= current_low <= retest_zone_high:
                    is_bullish_rejection = current_close > latest_bar['open'] and \
                                           current_close > (current_high + current_low) / 2
                    if is_bullish_rejection:
                        return {
                            'type': 'CALL',
                            'direction': 'UP',
                            'entry_price': current_close,
                            'level_name': f"Retest of {level_name} (${level_price:.2f})",
                            'level_price': level_price,
                            'time': latest_bar.name,
                            'strategy_name': 'Hourly_Retest'
                        }

        # Check for Retest of Resistance (for PUTS)
        # ONLY if we are below VWAP (Trend is acceptable)
        if trend_is_bearish:
            for level_name, level_price in self.key_levels[symbol]['resistance']:
                retest_zone_high = level_price * (1 + self.retest_dist_pct / 100)
                retest_zone_low = level_price * (1 - self.retest_dist_pct / 100)
                
                if retest_zone_low <= current_high <= retest_zone_high:
                    is_bearish_rejection = current_close < latest_bar['open'] and \
                                           current_close < (current_high + current_low) / 2
                    if is_bearish_rejection:
                        return {
                            'type': 'PUT',
                            'direction': 'DOWN',
                            'entry_price': current_close,
                            'level_name': f"Retest of {level_name} (${level_price:.2f})",
                            'level_price': level_price,
                            'time': latest_bar.name,
                            'strategy_name': 'Hourly_Retest'
                        }
        return None


    def strategy_2_orb_reversal(self, symbol):
        bars_2m = self.bar_data.get(f"{symbol}_2m")
        if bars_2m is None or len(bars_2m) < 2 or symbol not in self.orb_levels:
            return None 
        
        orb_high = self.orb_levels[symbol]['high']
        orb_low = self.orb_levels[symbol]['low']
        
        prev_bar = bars_2m.iloc[-2]
        latest_bar = bars_2m.iloc[-1]
        current_price = latest_bar['close']  # ← Use close, not open
        
        # PUT Signal: Failed break above ORB high
        if (prev_bar['high'] > orb_high and prev_bar['close'] < orb_high):
            return {
                'type': 'PUT',
                'direction': 'DOWN',
                'entry_price': current_price,
                'level_name': f"ORB Reversal (Fade High ${orb_high:.2f})",
                'time': latest_bar.name,
                'strategy_name': 'ORB_Reversal',
                'stop_price': orb_high * 1.002,  # ← Stop 0.2% ABOVE entry (allows wiggle room)
                'target_price': orb_low           # ← Target = return to ORB low
            }
            
        # CALL Signal: Failed break below ORB low
        if (prev_bar['low'] < orb_low and prev_bar['close'] > orb_low):
            return {
                'type': 'CALL',
                'direction': 'UP',
                'entry_price': current_price,
                'level_name': f"ORB Reversal (Fade Low ${orb_low:.2f})",
                'time': latest_bar.name,
                'strategy_name': 'ORB_Reversal',
                'stop_price': orb_low * 0.998,   # ← Stop 0.2% BELOW entry
                'target_price': orb_high          # ← Target = return to ORB high
            }
        
        return None

    def detect_market_regime(self, symbol):
        """Determine if we're in up/down/chop regime"""
        bars_1h = self.get_bars(symbol, TimeFrame(1, TimeFrameUnit.Hour), lookback_days=1)
        
        # Count up vs down hours
        up_hours = len(bars_1h[bars_1h['close'] > bars_1h['open']])
        down_hours = len(bars_1h[bars_1h['close'] < bars_1h['open']])
        
        if down_hours > up_hours * 1.5:
            return "DOWNTREND"
        elif up_hours > down_hours * 1.5:
            return "UPTREND"
        else:
            return "CHOP"



    def strategy_3_false_breakout_fade(self, symbol):
        """
        STRATEGY 3: Algorithmic translation of "False Breakout"
        """
        bars_1m = self.bar_data.get(f"{symbol}_1m")
        if bars_1m is None or len(bars_1m) < 2 or symbol not in self.key_levels:
            return None 

        prev_bar = bars_1m.iloc[-2]
        latest_bar = bars_1m.iloc[-1]
        
        # PUT Signal: Failed break of Resistance
        for level_name, level_price in self.key_levels[symbol]['resistance']:
            if (prev_bar.high > level_price and prev_bar.close < level_price):
                return {
                    'type': 'PUT',
                    'direction': 'DOWN',
                    'entry_price': latest_bar['open'],
                    'level_name': f"False Breakout Fade (Resist {level_name})",
                    'level_price': level_price,
                    'time': latest_bar.name,
                    'strategy_name': 'Fade_Key_Level'
                }

        # CALL Signal: Failed break of Support
        for level_name, level_price in self.key_levels[symbol]['support']:
            if (prev_bar.low < level_price and prev_bar.close > level_price):
                return {
                    'type': 'CALL',
                    'direction': 'UP',
                    'entry_price': latest_bar['open'],
                    'level_name': f"False Breakout Fade (Support {level_name})",
                    'level_price': level_price,
                    'time': latest_bar.name,
                    'strategy_name': 'Fade_Key_Level'
                }
        
        return None

    def liquidate_all_positions(self):
        """Closes all open positions, e.g., for EOD."""
        self.log(f"\n{'='*70}")
        self.log(f"--- EOD LIQUIDATION INITIATED ---")
        if not self.positions:
            self.log("No open positions to liquidate.")
            self.log(f"----------------------------------\n")
            return

        for symbol in list(self.positions.keys()):
            pos = self.positions[symbol]
            bars = self.bar_data.get(f"{symbol}_1m")
            current_price = bars.iloc[-1]['close'] if bars is not None and not bars.empty else pos['entry']
            shares = pos.get('shares', self.base_shares)

            realized_pnl = 0.0
            if pos['direction'] == "UP":
                realized_pnl = (current_price - pos['entry']) * shares
            else: # DOWN
                realized_pnl = (pos['entry'] - current_price) * shares
                
            self.daily_pnl += realized_pnl

            self.log(f"🛑 [EOD LIQUIDATE] ({pos['strategy']})")
            self.log(f"  Symbol: {symbol} @ ${current_price:.2f} (Market Close)")
            self.log(f"  Entry: ${pos['entry']:.2f}")
            self.log(f"  P&L: ${realized_pnl:+.2f}")
            
            del self.positions[symbol]
            # TODO: Add actual Alpaca market order to close position
            
        self.log(f"All positions liquidated. Daily Total P&L: ${self.daily_pnl:.2f}")
        self.log(f"{'='*70}\n")

    def _liquidate_orphaned_positions_on_startup(self):
        """
        Queries Alpaca on startup for any open positions on our watchlist
        and liquidates them to ensure a clean state.
        THIS IS THE FIX FOR THE STATE-LOSS (RESTART) BUG.
        """
        self.log("--- Checking for orphaned positions on startup ---")
        try:
            positions = self.trading_client.get_all_positions()
            
            orphans_found = 0
            for pos in positions:
                if pos.symbol in self.symbols:
                    orphans_found += 1
                    self.log(f"![ORPHAN DETECTED] Found open {pos.symbol} position ({pos.qty} shares).")
                    self.log(f"![ORPHAN DETECTED] Closing {pos.symbol} at market...")
                    # Use Alpaca's close_position() for a simple market close
                    self.trading_client.close_position(pos.symbol)
                    self.log(f"![ORPHAN DETECTED] {pos.symbol} position closed.")
            
            if orphans_found == 0:
                self.log("No orphaned positions found. Starting with a clean slate.")
                
        except Exception as e:
            self.log(f"[ERROR] Failed to liquidate orphaned positions: {e}")
        self.log("----------------------------------------------------")

    # --- MAIN RUN LOOP ---
    def run(self):
        self._liquidate_orphaned_positions_on_startup()

        last_regime_check = None
        market_regime = None

        # Run once at start
        self.get_key_levels(force_update=True)
        self.update_atr()
        self.update_bar_data() # Initial data load

        self.log("\n" + "="*70)
        self.log("LIVE SCANNING STARTED")
        self.log("="*70 + "\n")

        last_bar_update = datetime.now(self.tz)
        last_level_update = datetime.now(self.tz)

        orb_calculated_today = False
        eod_liquidated = False # --- NEW FLAG for EOD ---
        gap_regime_checked_today = False

        while True:
            now = datetime.now(self.tz)
            
            if last_regime_check is None or (now - last_regime_check).total_seconds() > 1800:
                # Check regime for each symbol
                for sym in self.symbols:
                    regime = self.detect_market_regime(sym)
                    self.log(f"[REGIME] {sym} market regime: {regime}")
                last_regime_check = now
            # --- EOD LIQUIDATION LOGIC (BUG 3 FIX) ---
            eod_liquidation_time = dt_time(15, 55)
            if now.time() >= eod_liquidation_time and not eod_liquidated:
                self.log(f"\n[EOD] It's {now.strftime('%H:%M')}, liquidating all open positions.")
                self.update_bar_data() # Get latest price for P&L calc
                self.liquidate_all_positions()
                eod_liquidated = True # Set flag so it only runs once

            # --- MARKET CLOSE LOGIC ---
            if now.hour >= 16:
                self.log(f"\nMarket closed. Daily P&L: ${self.daily_pnl:.2f}")
                break
            
            # --- After liquidating, stop trading and wait for market close ---
            if eod_liquidated:
                time.sleep(10)
                continue
                
            # --- Reset flags pre-market ---
            if now.hour < 9:
                orb_calculated_today = False
                eod_liquidated = False
                gap_regime_checked_today = False
                self.log("Pre-market. Sleeping...")
                time.sleep(60)
                continue

            if now.hour == 9 and now.minute < 30:
                self.log("Pre-market. Sleeping...")
                time.sleep(30)
                continue

            try:
                # --- DATA & STATE UPDATES ---

                # --- GAP/VOLATILITY + GEX REGIME CHECK (once, right at the open) ---
                if not gap_regime_checked_today:
                    self.update_bar_data()
                    for symbol in self.symbols:
                        bars_1m = self.bar_data.get(f"{symbol}_1m")
                        if bars_1m is not None and not bars_1m.empty:
                            regime = self.check_gap_regime(symbol, bars_1m.iloc[-1]['close'])
                            self.log(f"[GAP REGIME] {symbol}: {regime}")
                        gex_regime = self.fetch_gex_regime(symbol)
                        self.log(f"[GEX REGIME] {symbol}: {gex_regime}")
                    gap_regime_checked_today = True

                # --- ORB CALCULATION (BUG 1 FIX) ---
                if now.time() >= dt_time(9, 45) and not orb_calculated_today:
                    self.log("\n[INFO] Market open past 9:45. Updating bar data and calculating ORB...")
                    self.update_bar_data()
                    self.calculate_orb_levels()
                    orb_calculated_today = True
                    self.log("[INFO] ORB calculation complete.\n")

                # --- DATA REFRESH & EXIT CHECK (BUG 2 FIX IS IN check_exits) ---
                if (now - last_bar_update).total_seconds() >= 30:
                    self.update_bar_data()
                    last_bar_update = now

                    self.check_exits()

                # Update key levels + ATR every 30 minutes
                if (now - last_level_update).total_seconds() >= 1800:
                    self.get_key_levels()
                    self.update_atr()
                    last_level_update = now

                # --- SIGNAL SCANNING ---
                for symbol in self.symbols:
                    if not self.can_trade(symbol, "any"):
                        continue

                    for strategy_func in self.strategy_functions:
                        signal = strategy_func(symbol)

                        if signal:
                            if self.can_trade(symbol, signal['strategy_name']):
                                signal = self.score_signal_size(symbol, signal)
                                self.place_trade(symbol, signal)
                                break
                
                time.sleep(2) 

            except KeyboardInterrupt:
                self.log("\nSTOPPED BY USER")
                self.liquidate_all_positions() 
                break
            except Exception as e:
                self.log(f"\n[FATAL ERROR] Main loop: {e}")
                time.sleep(10)

def main():
    trader = MultiStrategyTrader(paper=True)
    trader.run()

if __name__ == "__main__":
    main()