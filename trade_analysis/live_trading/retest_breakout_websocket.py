import os
import sys
import time
import json
import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from zoneinfo import ZoneInfo
import websockets
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

class RetestBreakoutSystemV2:
    """
    FIXED VERSION - Your actual edge:
    1. Pre-market boot (9:15 ready)
    2. WebSocket real-time (no 2min lag)
    3. Adaptive volume filters
    4. Extended expiry windows
    5. Opening range bias
    """
    
    def __init__(self, paper=True):
        self.trading_client = TradingClient(API_KEY, SECRET_KEY, paper=paper)
        self.data_client = StockHistoricalDataClient(API_KEY, SECRET_KEY)
        self.paper = paper
        self.log_file = LOGS_DIR / "retest_trading_log_v2.txt"
        self.tz = ZoneInfo("America/New_York")
        
        # Watchlist
        self.symbols = ['SPY', 'QQQ', 'TSLA']
        
        # Position tracking
        self.positions = {}
        self.daily_pnl = 0.0
        self.max_daily_loss = -200.0
        
        # ADAPTIVE FILTERS - Fixed for your trading style
        self.volume_filters = {
            'opening': 0.8,   # 9:30-10:00: More lenient
            'mid_day': 1.0,   # 10:00-14:00: Standard
            'power_hour': 0.9 # 14:00-16:00: Slightly lenient
        }
        
        self.target_pct = 0.4
        self.stop_pct = 0.3

        # RETEST LOGIC - Extended windows
        self.level_breaks = {}
        self.break_expiry = {
            'opening': 30,  # 30min for opening range
            'regular': 20,  # 20min rest of day
            'power_hour': 25 # 25min for close
        }
        self.retest_distance_pct = 0.15

        # Scale out targets
        self.scale_out_1 = 0.2  # 50% at 0.2%
        self.scale_out_2 = 0.3  # 25% at 0.3%

        # Historical levels cache
        self.cached_levels = {}
        self.last_level_update = None

        # ATR-based risk: target/stop/retest zone scale with current
        # volatility (fixed target_pct/stop_pct/retest_distance_pct above
        # are used only as a fallback when ATR isn't available yet)
        self.atr_period = 14
        self.target_atr_mult = 2.5
        self.stop_atr_mult = 1.0
        self.retest_atr_mult = 0.15
        self.atr = {}             # {symbol: atr_value}
        self.yesterday_close = {}  # {symbol: float}, gap-regime baseline

        # Multi-day S/R confluence: get_historical_levels already tracks
        # 3 days of session + premarket highs/lows - a broken level that
        # clusters with another one of those (within confluence_zone_pct)
        # is higher-conviction and gets sized up
        self.confluence_zone_pct = 0.15
        self.confluence_size_mult = 1.5

        # Gap / volatility regime filter: a headline/news-driven gap well
        # beyond the recent ATR derates size instead of trusting static
        # S/R levels to hold
        self.gap_atr_mult_threshold = 2.0
        self.elevated_regime_size_mult = 0.5
        self.gap_regime = {}          # {symbol: "NORMAL" | "ELEVATED"}
        self.gap_regime_checked = {}  # {symbol: date_str} - once per day

        # Dealer gamma-exposure (GEX) regime: NET_LONG_GAMMA dealers
        # dampen/pin price (derate breakout size), NET_SHORT_GAMMA dealers
        # amplify moves (breakouts more likely to run, keep full size)
        self.gex_long_gamma_size_mult = 0.6
        self.gex_regime = {}          # {symbol: "NET_LONG_GAMMA" | "NET_SHORT_GAMMA" | "UNKNOWN"}
        self.gex_checked = {}         # {symbol: date_str} - once per day
        
        # WebSocket tracking
        self.ws_bars = {sym: [] for sym in self.symbols}
        self.ws_connected = False
        
        print(f"\n{'='*70}")
        print(f"RETEST BREAKOUT SYSTEM V2 - FIXED & OPTIMIZED")
        print(f"{'='*70}")
        print(f"Mode: {'PAPER' if paper else 'LIVE'}")
        print(f"WebSocket: wss://stream.data.alpaca.markets/v2/sip")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"\nNEW FEATURES:")
        print(f"  ✓ Pre-market boot (9:15 AM)")
        print(f"  ✓ WebSocket real-time (no lag)")
        print(f"  ✓ Adaptive volume (0.8x opening, 1.0x mid, 0.9x power)")
        print(f"  ✓ Extended expiry (30min opening, 20min regular)")
        print(f"  ✓ Opening range bias detection")
        print(f"\nSTRATEGY:")
        print(f"  1. Level breaks → Track")
        print(f"  2. Retest + rejection → Enter")
        print(f"  3. Quick profit-taking @ 0.2%/0.3%/0.4%\n")
        
        self.log("="*70)
        self.log("RETEST BREAKOUT SYSTEM V2 STARTED")
    
    def log(self, message):
        timestamp = datetime.now(self.tz).strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{timestamp}] {message}"
        print(msg)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
        except:
            pass
    
    def get_volume_threshold(self):
        """Adaptive volume based on time of day"""
        now = datetime.now(self.tz)
        hour = now.hour
        
        if 9 <= hour < 10:
            return self.volume_filters['opening']
        elif 14 <= hour < 16:
            return self.volume_filters['power_hour']
        else:
            return self.volume_filters['mid_day']
    
    def get_expiry_window(self):
        """Adaptive expiry based on time"""
        now = datetime.now(self.tz)
        hour = now.hour
        
        if 9 <= hour < 10:
            return self.break_expiry['opening']
        elif 14 <= hour < 16:
            return self.break_expiry['power_hour']
        else:
            return self.break_expiry['regular']
    
    def get_bars(self, symbol, lookback_minutes=120):
        """Get recent bars for indicators"""
        try:
            now = datetime.now(self.tz)
            market_open = datetime.combine(now.date(), dt_time(9, 30), tzinfo=self.tz)
            
            if now < market_open:
                return None
            
            start = max(market_open, now - timedelta(minutes=lookback_minutes))
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=start,
                end=now,
                feed="sip"
            )
            
            bars_response = self.data_client.get_stock_bars(request)
            if bars_response is None or bars_response.df.empty:
                return None
            
            df = bars_response.df
            
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
                if 'timestamp' in df.columns:
                    df = df[df['symbol'] == symbol].copy()
                    df.set_index('timestamp', inplace=True)
            
            if df.index.tz is None:
                df.index = df.index.tz_localize('UTC')
            df.index = df.index.tz_convert(self.tz)
            
            return df
        
        except Exception as e:
            self.log(f"[ERROR {symbol}] get_bars: {str(e)[:80]}")
            return None
    
    def get_historical_levels(self, symbol, days_back=3):
        """Get market and premarket levels - CACHED"""
        try:
            now = datetime.now(self.tz)
            
            # Cache for 30 minutes
            if (self.last_level_update and 
                (now - self.last_level_update).total_seconds() < 1800 and
                symbol in self.cached_levels):
                return self.cached_levels[symbol]
            
            start_date = now - timedelta(days=days_back + 5)
            
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(1, TimeFrameUnit.Day),
                start=start_date,
                end=now,
                feed="sip"
            )
            
            bars_response = self.data_client.get_stock_bars(request)
            if bars_response is None or bars_response.df.empty:
                return None
            
            df = bars_response.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()
            
            if len(df) < 2:
                return None
            
            recent_days = df.iloc[-(days_back+1):-1]

            if len(recent_days) > 0:
                self.yesterday_close[symbol] = float(recent_days.iloc[-1]['close'])

            levels = {
                'market_highs': [],
                'market_lows': [],
                'pm_highs': [],
                'pm_lows': []
            }
            
            # Market session levels
            for idx, day in recent_days.iterrows():
                try:
                    date = day['timestamp'].date() if 'timestamp' in day else idx.date()
                    market_high = float(day['high'])
                    market_low = float(day['low'])
                    
                    levels['market_highs'].append((f"Market High {date.strftime('%m/%d')}", market_high))
                    levels['market_lows'].append((f"Market Low {date.strftime('%m/%d')}", market_low))
                    
                    # Premarket
                    pm_start = datetime.combine(date, dt_time(4, 0), tzinfo=self.tz)
                    pm_end = datetime.combine(date, dt_time(9, 30), tzinfo=self.tz)
                    
                    pm_request = StockBarsRequest(
                        symbol_or_symbols=symbol,
                        timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                        start=pm_start,
                        end=pm_end,
                        feed="sip"
                    )
                    
                    pm_response = self.data_client.get_stock_bars(pm_request)
                    if pm_response is not None and not pm_response.df.empty:
                        pm_df = pm_response.df
                        if isinstance(pm_df.index, pd.MultiIndex):
                            pm_df = pm_df.reset_index()
                        
                        pm_high = float(pm_df['high'].max())
                        pm_low = float(pm_df['low'].min())
                        
                        levels['pm_highs'].append((f"PM High {date.strftime('%m/%d')}", pm_high))
                        levels['pm_lows'].append((f"PM Low {date.strftime('%m/%d')}", pm_low))
                except:
                    continue
            
            # Today's premarket
            today_date = now.date()
            pm_start = datetime.combine(today_date, dt_time(4, 0), tzinfo=self.tz)
            pm_end = datetime.combine(today_date, dt_time(9, 30), tzinfo=self.tz)
            
            pm_request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=pm_start,
                end=pm_end,
                feed="sip"
            )
            
            try:
                pm_response = self.data_client.get_stock_bars(pm_request)
                if pm_response is not None and not pm_response.df.empty:
                    pm_df = pm_response.df
                    if isinstance(pm_df.index, pd.MultiIndex):
                        pm_df = pm_df.reset_index()
                    
                    pm_high = float(pm_df['high'].max())
                    pm_low = float(pm_df['low'].min())
                    
                    levels['pm_highs'].append((f"Today PM High", pm_high))
                    levels['pm_lows'].append((f"Today PM Low", pm_low))
            except:
                pass
            
            self.cached_levels[symbol] = levels
            self.last_level_update = now
            
            return levels
            
        except Exception as e:
            self.log(f"[ERROR {symbol}] get_historical_levels: {str(e)[:80]}")
            return None

    def update_atr(self, symbol, lookback_days=3):
        """5-min ATR from the past few sessions, so target/stop/retest
        scale with current volatility instead of the fixed pct fallbacks."""
        try:
            now = datetime.now(self.tz)
            request = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame(5, TimeFrameUnit.Minute),
                start=now - timedelta(days=lookback_days + 1),
                end=now,
                feed="sip"
            )
            bars_response = self.data_client.get_stock_bars(request)
            if bars_response is None or bars_response.df.empty:
                return None

            df = bars_response.df
            if isinstance(df.index, pd.MultiIndex):
                df = df.reset_index()

            if len(df) < 2:
                return None

            high, low, close = df['high'], df['low'], df['close']
            prev_close = close.shift(1)
            true_range = pd.concat(
                [high - low, (high - prev_close).abs(), (low - prev_close).abs()],
                axis=1
            ).max(axis=1)

            atr = true_range.tail(self.atr_period).mean()
            if pd.notna(atr) and atr > 0:
                self.atr[symbol] = float(atr)
                return self.atr[symbol]
            return None
        except Exception as e:
            self.log(f"[ERROR {symbol}] update_atr: {str(e)[:80]}")
            return None

    def check_confluence(self, levels, level_price):
        """Does level_price line up with another recent S/R zone (within
        confluence_zone_pct) across the full multi-day level set, including
        dealer gamma-concentration strikes? Multiple touches = higher-
        conviction breakout."""
        if not levels or level_price is None:
            return False, 0

        all_prices = (
            [p for _, p in levels.get('market_highs', [])] +
            [p for _, p in levels.get('market_lows', [])] +
            [p for _, p in levels.get('pm_highs', [])] +
            [p for _, p in levels.get('pm_lows', [])] +
            [p for _, p in levels.get('gamma_strikes', [])]
        )

        matches = [
            p for p in all_prices
            if p != level_price and abs(p - level_price) / level_price <= self.confluence_zone_pct / 100
        ]
        return len(matches) > 0, len(matches)

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
        option chain, and folds the top gamma-concentration strikes into
        the cached levels dict so check_confluence sees them too."""
        try:
            result = gamma_exposure.compute_gex(symbol)
            if 'error' in result:
                self.log(f"[GEX] {symbol}: {result['error']}")
                self.gex_regime[symbol] = "UNKNOWN"
                return self.gex_regime[symbol]

            self.gex_regime[symbol] = result['regime']
            if symbol in self.cached_levels:
                self.cached_levels[symbol]['gamma_strikes'] = [
                    (f"Gamma strike ${strike:.2f} ({gex:+,.0f})", strike)
                    for strike, gex in result['top_gamma_strikes']
                ]

            return self.gex_regime[symbol]
        except Exception as e:
            self.log(f"[ERROR {symbol}] fetch_gex_regime: {str(e)[:80]}")
            self.gex_regime[symbol] = "UNKNOWN"
            return self.gex_regime[symbol]

    async def handle_ws_message(self, message):
        """Handle WebSocket bar data"""
        try:
            data = json.loads(message)
            
            if isinstance(data, list):
                for msg in data:
                    await self.process_bar(msg)
            else:
                await self.process_bar(data)
        
        except Exception as e:
            self.log(f"[WS ERROR] {str(e)[:60]}")
    
    async def process_bar(self, bar_data):
        """Process individual bar"""
        try:
            msg_type = bar_data.get('T')
            
            if msg_type == 'success':
                self.log(f"[WS] {bar_data.get('msg', 'Connected')}")
                self.ws_connected = True
                return
            
            if msg_type == 'subscription':
                self.log(f"[WS] Subscribed: {bar_data.get('bars', [])}")
                return
            
            if msg_type != 'b':  # bar
                return
            
            symbol = bar_data.get('S')
            if symbol not in self.symbols:
                return
            
            # Store bar
            bar = {
                'time': bar_data.get('t'),
                'open': bar_data.get('o'),
                'high': bar_data.get('h'),
                'low': bar_data.get('l'),
                'close': bar_data.get('c'),
                'volume': bar_data.get('v')
            }
            
            self.ws_bars[symbol].append(bar)

            # Keep last 100 bars
            if len(self.ws_bars[symbol]) > 100:
                self.ws_bars[symbol] = self.ws_bars[symbol][-100:]

            # Gap/volatility regime check, once per symbol per day at the open
            now = datetime.now(self.tz)
            today_str = now.date().isoformat()
            if self.gap_regime_checked.get(symbol) != today_str and now.time() >= dt_time(9, 30):
                atr = self.update_atr(symbol)
                regime = self.check_gap_regime(symbol, float(bar['close']))
                self.log(f"[GAP REGIME] {symbol}: {regime}" + (f" (ATR: {atr:.3f})" if atr else " (ATR unavailable)"))
                self.gap_regime_checked[symbol] = today_str

            if self.gex_checked.get(symbol) != today_str and now.time() >= dt_time(9, 30):
                gex_regime = self.fetch_gex_regime(symbol)
                self.log(f"[GEX REGIME] {symbol}: {gex_regime}")
                self.gex_checked[symbol] = today_str

            # Check for signals every bar
            await self.check_retest_signal_ws(symbol)
        
        except Exception as e:
            self.log(f"[ERROR] process_bar: {str(e)[:60]}")
    
    async def check_retest_signal_ws(self, symbol):
        """Check retest using WebSocket data"""
        try:
            if len(self.ws_bars[symbol]) < 10:
                return
            
            levels = self.get_historical_levels(symbol, days_back=3)
            if levels is None:
                return
            
            # Current bar
            latest = self.ws_bars[symbol][-1]
            current_close = float(latest['close'])
            current_high = float(latest['high'])
            current_low = float(latest['low'])
            current_volume = float(latest['volume'])
            current_time = datetime.now(self.tz)
            
            # Previous bar
            if len(self.ws_bars[symbol]) >= 2:
                prev = self.ws_bars[symbol][-2]
                prev_close = float(prev['close'])
            else:
                prev_close = current_close
            
            # Volume check
            recent_vols = [float(b['volume']) for b in self.ws_bars[symbol][-20:]]
            vol_avg = np.mean(recent_vols) if recent_vols else 1
            vol_ratio = current_volume / vol_avg if vol_avg > 0 else 0
            
            vol_threshold = self.get_volume_threshold()
            expiry_window = self.get_expiry_window()
            
            # Build levels
            resistance_levels = []
            for name, price in levels['market_highs']:
                resistance_levels.append((name, price, 'STRONG'))
            for name, price in levels['pm_highs']:
                resistance_levels.append((name, price, 'MEDIUM'))
            resistance_levels.sort(key=lambda x: x[1], reverse=True)
            
            support_levels = []
            for name, price in levels['market_lows']:
                support_levels.append((name, price, 'STRONG'))
            for name, price in levels['pm_lows']:
                support_levels.append((name, price, 'MEDIUM'))
            support_levels.sort(key=lambda x: x[1])
            
            # Track breaks
            for level_name, level_price, strength in resistance_levels:
                if current_close > level_price:
                    level_key = f"{symbol}_{level_name}"
                    
                    if level_key not in self.level_breaks:
                        self.level_breaks[level_key] = {
                            'level': level_price,
                            'type': 'CALL',
                            'break_time': current_time,
                            'level_name': level_name,
                            'strength': strength
                        }
                        self.log(f"[BREAK] {symbol} {level_name} @ ${level_price:.2f} (now ${current_close:.2f})")
            
            for level_name, level_price, strength in support_levels:
                if current_close < level_price:
                    level_key = f"{symbol}_{level_name}"
                    
                    if level_key not in self.level_breaks:
                        self.level_breaks[level_key] = {
                            'level': level_price,
                            'type': 'PUT',
                            'break_time': current_time,
                            'level_name': level_name,
                            'strength': strength
                        }
                        self.log(f"[BREAK] {symbol} {level_name} @ ${level_price:.2f} (now ${current_close:.2f})")
            
            # Check retests
            for level_key, break_info in list(self.level_breaks.items()):
                if not level_key.startswith(symbol):
                    continue
                
                minutes_since_break = (current_time - break_info['break_time']).total_seconds() / 60
                
                if minutes_since_break > expiry_window:
                    self.log(f"[EXPIRED] {level_key} (>{expiry_window}min)")
                    del self.level_breaks[level_key]
                    continue
                
                level_price = break_info['level']
                level_type = break_info['type']
                atr = self.atr.get(symbol)
                retest_dist_pct = (self.retest_atr_mult * atr / level_price * 100) if atr else self.retest_distance_pct

                if level_type == 'CALL':
                    retest_range_low = level_price * (1 - retest_dist_pct / 100)
                    retest_range_high = level_price * (1 + retest_dist_pct / 100)

                    in_retest_zone = retest_range_low <= current_low <= retest_range_high
                    rejecting_up = current_close > prev_close and current_close > current_low

                    if in_retest_zone and rejecting_up:
                        if vol_ratio < vol_threshold:
                            self.log(f"[SKIP] {symbol}: Vol {vol_ratio:.2f}x < {vol_threshold}x")
                            continue

                        del self.level_breaks[level_key]

                        has_confluence, confluence_count = self.check_confluence(levels, level_price)
                        size_mult = self.confluence_size_mult if has_confluence else 1.0
                        if self.gap_regime.get(symbol) == "ELEVATED":
                            size_mult *= self.elevated_regime_size_mult
                        if self.gex_regime.get(symbol) == "NET_LONG_GAMMA":
                            size_mult *= self.gex_long_gamma_size_mult

                        signal = {
                            'type': 'CALL',
                            'direction': 'UP',
                            'level': level_price,
                            'level_name': break_info['level_name'],
                            'entry_price': current_close,
                            'time': current_time,
                            'strength': break_info['strength'],
                            'volume': current_volume,
                            'vol_ratio': vol_ratio,
                            'confluence_count': confluence_count,
                            'size_multiplier': size_mult
                        }

                        await self.place_entry(symbol, signal)

                else:  # PUT
                    retest_range_low = level_price * (1 - retest_dist_pct / 100)
                    retest_range_high = level_price * (1 + retest_dist_pct / 100)

                    in_retest_zone = retest_range_low <= current_high <= retest_range_high
                    rejecting_down = current_close < prev_close and current_close < current_high

                    if in_retest_zone and rejecting_down:
                        if vol_ratio < vol_threshold:
                            self.log(f"[SKIP] {symbol}: Vol {vol_ratio:.2f}x < {vol_threshold}x")
                            continue

                        del self.level_breaks[level_key]

                        has_confluence, confluence_count = self.check_confluence(levels, level_price)
                        size_mult = self.confluence_size_mult if has_confluence else 1.0
                        if self.gap_regime.get(symbol) == "ELEVATED":
                            size_mult *= self.elevated_regime_size_mult
                        if self.gex_regime.get(symbol) == "NET_LONG_GAMMA":
                            size_mult *= self.gex_long_gamma_size_mult

                        signal = {
                            'type': 'PUT',
                            'direction': 'DOWN',
                            'level': level_price,
                            'level_name': break_info['level_name'],
                            'entry_price': current_close,
                            'time': current_time,
                            'strength': break_info['strength'],
                            'volume': current_volume,
                            'vol_ratio': vol_ratio,
                            'confluence_count': confluence_count,
                            'size_multiplier': size_mult
                        }

                        await self.place_entry(symbol, signal)
        
        except Exception as e:
            self.log(f"[ERROR] check_retest_ws: {str(e)[:80]}")
    
    async def place_entry(self, symbol, signal):
        """Log entry"""
        try:
            if symbol in self.positions:
                return
            
            if self.daily_pnl <= self.max_daily_loss:
                self.log(f"[STOPPED] Max loss: ${self.daily_pnl:.2f}")
                return
            
            entry_price = signal['entry_price']
            atr = self.atr.get(symbol)

            # ATR-based target/stop (fallback to fixed pct if ATR unavailable);
            # scale-outs stay proportional (50%/75%) to the ATR-based final target
            target_pct = (self.target_atr_mult * atr / entry_price * 100) if atr else self.target_pct
            stop_pct = (self.stop_atr_mult * atr / entry_price * 100) if atr else self.stop_pct
            scale_out_1_pct = target_pct * 0.5
            scale_out_2_pct = target_pct * 0.75

            if signal['direction'] == "UP":
                target_final = entry_price * (1.0 + target_pct / 100.0)
                stop = entry_price * (1.0 - stop_pct / 100.0)
                target_1 = entry_price * (1.0 + scale_out_1_pct / 100.0)
                target_2 = entry_price * (1.0 + scale_out_2_pct / 100.0)
            else:
                target_final = entry_price * (1.0 - target_pct / 100.0)
                stop = entry_price * (1.0 + stop_pct / 100.0)
                target_1 = entry_price * (1.0 - scale_out_1_pct / 100.0)
                target_2 = entry_price * (1.0 - scale_out_2_pct / 100.0)

            size_multiplier = signal.get('size_multiplier', 1.0)

            self.positions[symbol] = {
                'type': signal['type'],
                'direction': signal['direction'],
                'entry': entry_price,
                'target_1': target_1,
                'target_2': target_2,
                'target_final': target_final,
                'stop': stop,
                'time': signal['time'],
                'level_name': signal['level_name'],
                'strength': signal['strength'],
                'size': 100,
                'size_multiplier': size_multiplier,
                'scaled_out': []
            }

            self.log(f"\n{'='*70}")
            self.log(f"[{signal['strength']} {signal['type']} ENTRY]")
            self.log(f"  {symbol} @ ${entry_price:.2f}")
            self.log(f"  Level: {signal['level_name']} (${signal['level']:.2f})")
            self.log(f"  Vol: {signal['vol_ratio']:.2f}x | Confluence: {signal.get('confluence_count', 0)} | Regime: {self.gap_regime.get(symbol, 'NORMAL')}")
            self.log(f"  Size multiplier: x{size_multiplier:.2f}")
            self.log(f"  Targets: ${target_1:.2f} / ${target_2:.2f} / ${target_final:.2f}")
            self.log(f"  Stop: ${stop:.2f}" + (f" (ATR: {atr:.3f})" if atr else ""))
            self.log(f"{'='*70}\n")
        
        except Exception as e:
            self.log(f"[ERROR] place_entry: {e}")
    
    async def check_exits(self):
        """Check exits"""
        for symbol in list(self.positions.keys()):
            try:
                if len(self.ws_bars[symbol]) == 0:
                    continue
                
                current_price = float(self.ws_bars[symbol][-1]['close'])
                pos = self.positions[symbol]
                size_mult = pos.get('size_multiplier', 1.0)

                if pos['direction'] == "UP":
                    pnl_pct = ((current_price - pos['entry']) / pos['entry']) * 100

                    if 1 not in pos['scaled_out'] and current_price >= pos['target_1']:
                        self.daily_pnl += pnl_pct * 0.5 * size_mult
                        pos['scaled_out'].append(1)
                        pos['size'] = 50
                        self.log(f"[EXIT-1] {symbol} 50% @ ${current_price:.2f} (+{pnl_pct:.2f}%)")

                    elif 2 not in pos['scaled_out'] and current_price >= pos['target_2']:
                        self.daily_pnl += pnl_pct * 0.25 * size_mult
                        pos['scaled_out'].append(2)
                        pos['size'] = 25
                        self.log(f"[EXIT-2] {symbol} 25% @ ${current_price:.2f} (+{pnl_pct:.2f}%)")

                    elif current_price >= pos['target_final']:
                        self.daily_pnl += pnl_pct * (pos['size'] / 100.0) * size_mult
                        self.log(f"[EXIT-FINAL] {symbol} @ ${current_price:.2f} | P&L: ${self.daily_pnl:.2f}")
                        del self.positions[symbol]

                    elif current_price <= pos['stop']:
                        self.daily_pnl += pnl_pct * (pos['size'] / 100.0) * size_mult
                        self.log(f"[STOP] {symbol} @ ${current_price:.2f} ({pnl_pct:.2f}%) | P&L: ${self.daily_pnl:.2f}")
                        del self.positions[symbol]

                else:  # DOWN
                    pnl_pct = ((pos['entry'] - current_price) / pos['entry']) * 100

                    if 1 not in pos['scaled_out'] and current_price <= pos['target_1']:
                        self.daily_pnl += pnl_pct * 0.5 * size_mult
                        pos['scaled_out'].append(1)
                        pos['size'] = 50
                        self.log(f"[EXIT-1] {symbol} 50% @ ${current_price:.2f} (+{pnl_pct:.2f}%)")

                    elif 2 not in pos['scaled_out'] and current_price <= pos['target_2']:
                        self.daily_pnl += pnl_pct * 0.25 * size_mult
                        pos['scaled_out'].append(2)
                        pos['size'] = 25
                        self.log(f"[EXIT-2] {symbol} 25% @ ${current_price:.2f} (+{pnl_pct:.2f}%)")

                    elif current_price <= pos['target_final']:
                        self.daily_pnl += pnl_pct * (pos['size'] / 100.0) * size_mult
                        self.log(f"[EXIT-FINAL] {symbol} @ ${current_price:.2f} | P&L: ${self.daily_pnl:.2f}")
                        del self.positions[symbol]

                    elif current_price >= pos['stop']:
                        self.daily_pnl += pnl_pct * (pos['size'] / 100.0) * size_mult
                        self.log(f"[STOP] {symbol} @ ${current_price:.2f} ({pnl_pct:.2f}%) | P&L: ${self.daily_pnl:.2f}")
                        del self.positions[symbol]
            
            except:
                pass



    async def run_websocket(self):
        """Run WebSocket stream"""
        uri = "wss://stream.data.alpaca.markets/v2/sip"
        
        try:
            self.log("="*70)
            self.log("CONNECTING TO WEBSOCKET")
            self.log("="*70)
            
            async with websockets.connect(uri) as websocket:
                # Auth
                auth_msg = {
                    "action": "auth",
                    "key": API_KEY,
                    "secret": SECRET_KEY
                }
                await websocket.send(json.dumps(auth_msg))
                
                # Wait for auth
                response = await websocket.recv()
                await self.handle_ws_message(response)
                
                # Subscribe to bars
                sub_msg = {
                    "action": "subscribe",
                    "bars": self.symbols
                }
                await websocket.send(json.dumps(sub_msg))
                
                self.log(f"[WS] Subscribed to: {self.symbols}")
                self.log("="*70 + "\n")
                
                # Listen
                while True:
                    message = await websocket.recv()
                    await self.handle_ws_message(message)
                    await self.check_exits()
        
        except Exception as e:
            self.log(f"[WS ERROR] {e}")
    
    async def pre_market_boot(self):
        """Boot at 9:15, load levels, be ready at 9:30"""
        now = datetime.now(self.tz)
        target_time = datetime.combine(now.date(), dt_time(9, 15), tzinfo=self.tz)
        
        if now < target_time:
            wait_seconds = (target_time - now).total_seconds()
            self.log(f"Waiting until 9:15 AM ({wait_seconds:.0f}s)...")
            await asyncio.sleep(wait_seconds)
        
        self.log("="*70)
        self.log("PRE-MARKET BOOT - LOADING LEVELS")
        self.log("="*70)
        
        for symbol in self.symbols:
            levels = self.get_historical_levels(symbol, days_back=3)
            if levels:
                self.log(f"{symbol}: {len(levels['market_highs'])} resistance, {len(levels['market_lows'])} support")

            atr = self.update_atr(symbol)
            self.log(f"{symbol}: ATR(5min) = {atr:.3f}" if atr else f"{symbol}: ATR unavailable, using fixed pct fallback")

            gex_regime = self.fetch_gex_regime(symbol)
            self.log(f"{symbol}: GEX regime = {gex_regime}")
            self.gex_checked[symbol] = now.date().isoformat()

        self.log("\n[READY] Waiting for 9:30 market open...")
        
        market_open = datetime.combine(now.date(), dt_time(9, 30), tzinfo=self.tz)
        now = datetime.now(self.tz)
        
        if now < market_open:
            wait_seconds = (market_open - now).total_seconds()
            await asyncio.sleep(wait_seconds)
        
        self.log("="*70)
        self.log("MARKET OPEN - LIVE TRADING")
        self.log("="*70 + "\n")
    
    async def run(self):
        """Main run loop"""
        # Pre-market boot
        await self.pre_market_boot()
        
        # Start WebSocket
        await self.run_websocket()

async def main():
    system = RetestBreakoutSystemV2(paper=True)
    
    try:
        await system.run()
    except KeyboardInterrupt:
        system.log("\nSTOPPED BY USER")
    except Exception as e:
        system.log(f"\nFATAL ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())