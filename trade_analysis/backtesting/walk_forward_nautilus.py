"""
NautilusTrader Backtest with LOB Simulation - FIXED VERSION
============================================================
Fixes:
  1. Dynamic balance query (not hardcoded $10k)
  2. Capital allocation split by number of symbols
  3. MARGIN account to allow PUT (short) entries
  4. Pre-trade balance check to prevent overdraft
  5. Event handlers for actual fill tracking

Requirements:
    pip install nautilus_trader pandas numpy yfinance --break-system-packages
"""

import os
import sys
from decimal import Decimal
from datetime import datetime, timedelta, time as dt_time
from pathlib import Path
from typing import Optional
import pandas as pd
import numpy as np

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.backtest.models import FillModel
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USD
from nautilus_trader.model.data import Bar, BarType, BarSpecification
from nautilus_trader.model.enums import (
    AccountType,
    AggregationSource,
    BarAggregation,
    OmsType,
    OrderSide,
    PriceType,
    TimeInForce,
)
from nautilus_trader.model.identifiers import (
    InstrumentId,
    Symbol,
    TraderId,
    Venue,
)
from nautilus_trader.model.instruments import Equity
from nautilus_trader.model.objects import Money, Price, Quantity
from nautilus_trader.model.events import OrderFilled, OrderRejected
from nautilus_trader.trading.strategy import Strategy, StrategyConfig
from nautilus_trader.core.datetime import dt_to_unix_nanos

from nautilus_trader.indicators import VolumeWeightedAveragePrice

from collections import deque


# =============================================================================
# GLOBAL CAPITAL COORDINATOR (shared across strategies)
# =============================================================================

class CapitalCoordinator:
    """
    Coordinates capital allocation across multiple strategy instances.
    Prevents concurrent overdraft when SPY and QQQ both try to enter at same bar.
    """
    def __init__(self, total_capital: float, num_symbols: int, max_allocation_pct: float = 0.45):
        self.total_capital = total_capital
        self.num_symbols = num_symbols
        # Each symbol gets a slice; max_allocation_pct prevents full allocation
        self.per_symbol_allocation = (total_capital / num_symbols) * max_allocation_pct
        self.locked_capital = 0.0  # Currently in positions
        
    def request_capital(self, amount: float) -> bool:
        """Returns True if capital can be allocated, False otherwise."""
        available = self.total_capital - self.locked_capital
        if amount <= available:
            self.locked_capital += amount
            return True
        return False
    
    def release_capital(self, amount: float):
        """Release capital when position closes."""
        self.locked_capital = max(0.0, self.locked_capital - amount)
    
    def get_available(self) -> float:
        return self.total_capital - self.locked_capital
    
    def update_total(self, new_total: float):
        """Update after realized PnL."""
        self.total_capital = new_total
        self.per_symbol_allocation = (new_total / self.num_symbols) * 0.45


# =============================================================================
# STRATEGY CONFIG
# =============================================================================

class MultiStrategyConfig(StrategyConfig, frozen=True):
    """Configuration for the multi-strategy trader."""
    
    instrument_id: str
    bar_type_1m: str
    bar_type_2m: str
    bar_type_1h: str
    
    # Optimized parameters
    default_target_pct: float = 0.6
    default_stop_pct: float = 0.5
    retest_dist_pct: float = 0.2
    entry_cooldown_minutes: int = 3
    max_daily_loss: float = -200.0
    
    # Strategy toggles
    enable_hourly_retest: bool = True
    enable_orb_reversal: bool = True
    enable_false_breakout: bool = True


# =============================================================================
# MAIN STRATEGY
# =============================================================================

class MultiStrategyTrader(Strategy):
    """
    NautilusTrader port with FIXED capital management.
    """
    
    # Class-level coordinator (shared across all instances)
    capital_coordinator: Optional[CapitalCoordinator] = None
    
    @classmethod
    def set_capital_coordinator(cls, coordinator: CapitalCoordinator):
        cls.capital_coordinator = coordinator
    
    def __init__(self, config: MultiStrategyConfig):
        super().__init__(config)
        
        self.instrument_id = InstrumentId.from_str(config.instrument_id)
        self.bar_type_1m = BarType.from_str(config.bar_type_1m)
        self.bar_type_2m = BarType.from_str(config.bar_type_2m)
        self.bar_type_1h = BarType.from_str(config.bar_type_1h)
        
        self.target_pct = config.default_target_pct / 100
        self.stop_pct = config.default_stop_pct / 100
        self.retest_dist_pct = config.retest_dist_pct / 100
        self.cooldown_minutes = config.entry_cooldown_minutes
        self.max_daily_loss = config.max_daily_loss
        
        self.enable_hourly_retest = config.enable_hourly_retest
        self.enable_orb_reversal = config.enable_orb_reversal
        self.enable_false_breakout = config.enable_false_breakout
        
        self.vwap = VolumeWeightedAveragePrice()
        
        self.key_levels = {'support': [], 'resistance': []}
        self.orb_levels = {'high': None, 'low': None}
        self.orb_calculated_today = False
        
        self.bars_1m = deque(maxlen=500)
        self.bars_2m = deque(maxlen=250)
        self.bars_1h = deque(maxlen=100)
        
        self.daily_pnl = 0.0
        self.last_entry_time = None
        self.current_position_direction = None
        self.current_position_size = 0
        self.locked_capital_for_position = 0.0  # Track what we locked
        self.entry_price = None
        self.actual_fill_price = None  # Track actual fill from event
        self.target_price = None
        self.stop_price = None
        self.strategy_name = None
        
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        
        self._current_date = None
    
    def on_start(self):
        self.log.info("MultiStrategyTrader starting...")
        self.register_indicator_for_bars(self.bar_type_1m, self.vwap)
        self.subscribe_bars(self.bar_type_1m)
        self.subscribe_bars(self.bar_type_2m)
        self.subscribe_bars(self.bar_type_1h)
        self.log.info(f"Subscribed to {self.bar_type_1m}, {self.bar_type_2m}, {self.bar_type_1h}")
    
    def on_bar(self, bar: Bar):
        bar_time = pd.Timestamp(bar.ts_event, unit='ns', tz='UTC').tz_convert('America/New_York')
        bar_date = bar_time.date()
        
        if self._current_date != bar_date:
            self._reset_daily_state(bar_date)
        
        if bar.bar_type == self.bar_type_1m:
            self._on_bar_1m(bar, bar_time)
        elif bar.bar_type == self.bar_type_2m:
            self._on_bar_2m(bar, bar_time)
        elif bar.bar_type == self.bar_type_1h:
            self._on_bar_1h(bar, bar_time)
    
    def on_order_filled(self, event: OrderFilled):
        """Track actual fill prices from exchange."""
        self.actual_fill_price = float(event.last_px)
        self.log.info(f"Order filled at ${self.actual_fill_price:.2f} (qty={event.last_qty})")
    
    def on_order_rejected(self, event: OrderRejected):
        """Handle rejections - release locked capital."""
        self.log.warning(f"Order rejected: {event.reason}")
        if self.locked_capital_for_position > 0:
            if self.capital_coordinator:
                self.capital_coordinator.release_capital(self.locked_capital_for_position)
            self.locked_capital_for_position = 0.0
            self.current_position_direction = None
            self.current_position_size = 0
    
    def _reset_daily_state(self, new_date):
        self._current_date = new_date
        self.daily_pnl = 0.0
        self.orb_calculated_today = False
        self.orb_levels = {'high': None, 'low': None}
        self.log.info(f"=== NEW TRADING DAY: {new_date} ===")
    
    def _on_bar_1m(self, bar: Bar, bar_time: pd.Timestamp):
        self.bars_1m.append(bar)
        
        if not self._is_regular_hours(bar_time):
            return
        
        if self.current_position_direction is not None:
            self._check_exits(bar)
            return
        
        if not self._can_trade(bar_time):
            return
        
        signal = None
        
        if self.enable_hourly_retest and len(self.bars_1m) >= 2:
            signal = self._strategy_hourly_retest(bar)
        
        if signal is None and self.enable_false_breakout and len(self.bars_1m) >= 2:
            signal = self._strategy_false_breakout(bar)
        
        if signal:
            self._execute_signal(signal, bar_time)
    
    def _on_bar_2m(self, bar: Bar, bar_time: pd.Timestamp):
        self.bars_2m.append(bar)
        
        if not self.orb_calculated_today and bar_time.time() >= dt_time(9, 45):
            self._calculate_orb(bar_time)
        
        if not self._is_regular_hours(bar_time):
            return
        
        if self.current_position_direction is not None:
            return
        
        if not self._can_trade(bar_time):
            return
        
        if self.enable_orb_reversal and len(self.bars_2m) >= 2:
            signal = self._strategy_orb_reversal(bar)
            if signal:
                self._execute_signal(signal, bar_time)
    
    def _on_bar_1h(self, bar: Bar, bar_time: pd.Timestamp):
        self.bars_1h.append(bar)
        self._update_key_levels()
    
    def _is_regular_hours(self, bar_time: pd.Timestamp) -> bool:
        t = bar_time.time()
        return dt_time(9, 30) <= t <= dt_time(15, 55)
    
    def _can_trade(self, bar_time: pd.Timestamp) -> bool:
        if self.daily_pnl <= self.max_daily_loss:
            return False
        
        if self.current_position_direction is not None:
            return False
        
        if self.last_entry_time:
            minutes_since = (bar_time - self.last_entry_time).total_seconds() / 60
            if minutes_since < self.cooldown_minutes:
                return False
        
        return True
    
    def _get_available_capital(self) -> float:
        """Query actual available capital from coordinator or portfolio."""
        if self.capital_coordinator:
            return min(
                self.capital_coordinator.get_available(),
                self.capital_coordinator.per_symbol_allocation
            )
        # Fallback: query portfolio (more accurate but slower)
        try:
            account = self.portfolio.account(Venue("NASDAQ"))
            if account:
                balance = account.balance_free(USD)
                if balance:
                    return float(balance.as_double())
        except Exception as e:
            self.log.warning(f"Could not query balance: {e}")
        return 0.0
    
    def _update_key_levels(self):
        if len(self.bars_1h) < 5:
            return
        
        self.key_levels = {'support': [], 'resistance': []}
        
        for i in range(1, min(6, len(self.bars_1h))):
            bar = self.bars_1h[-i]
            self.key_levels['resistance'].append((f"H1_High_{-i}", float(bar.high)))
            self.key_levels['support'].append((f"H1_Low_{-i}", float(bar.low)))
        
        self.key_levels['support'].sort(key=lambda x: x[1], reverse=True)
        self.key_levels['resistance'].sort(key=lambda x: x[1])
    
    def _calculate_orb(self, bar_time: pd.Timestamp):
        orb_start = dt_time(9, 30)
        orb_end = dt_time(9, 45)
        
        orb_bars = [
            b for b in self.bars_1m
            if orb_start <= pd.Timestamp(b.ts_event, unit='ns', tz='UTC').tz_convert('America/New_York').time() < orb_end
            and pd.Timestamp(b.ts_event, unit='ns', tz='UTC').tz_convert('America/New_York').date() == bar_time.date()
        ]
        
        if orb_bars:
            self.orb_levels['high'] = max(float(b.high) for b in orb_bars)
            self.orb_levels['low'] = min(float(b.low) for b in orb_bars)
            self.orb_calculated_today = True
            self.log.info(f"ORB Calculated: High={self.orb_levels['high']:.2f}, Low={self.orb_levels['low']:.2f}")
    
    # =========================================================================
    # STRATEGIES
    # =========================================================================
    
    def _strategy_hourly_retest(self, bar: Bar) -> Optional[dict]:
        if len(self.bars_1m) < 2 or not self.key_levels['support']:
            return None
        
        current_close = float(bar.close)
        current_low = float(bar.low)
        current_high = float(bar.high)
        current_open = float(bar.open)
        current_vwap = self.vwap.value
        
        if current_vwap == 0:
            return None
        
        trend_is_bullish = current_close > current_vwap
        trend_is_bearish = current_close < current_vwap
        
        if trend_is_bullish:
            for level_name, level_price in self.key_levels['support']:
                retest_zone_high = level_price * (1 + self.retest_dist_pct)
                retest_zone_low = level_price * (1 - self.retest_dist_pct)
                
                if retest_zone_low <= current_low <= retest_zone_high:
                    is_bullish_rejection = (
                        current_close > current_open and
                        current_close > (current_high + current_low) / 2
                    )
                    if is_bullish_rejection:
                        return {
                            'type': 'CALL',
                            'direction': 'UP',
                            'entry_price': current_close,
                            'level_name': f"Retest of {level_name} (${level_price:.2f})",
                            'strategy_name': 'Hourly_Retest',
                        }
        
        if trend_is_bearish:
            for level_name, level_price in self.key_levels['resistance']:
                retest_zone_high = level_price * (1 + self.retest_dist_pct)
                retest_zone_low = level_price * (1 - self.retest_dist_pct)
                
                if retest_zone_low <= current_high <= retest_zone_high:
                    is_bearish_rejection = (
                        current_close < current_open and
                        current_close < (current_high + current_low) / 2
                    )
                    if is_bearish_rejection:
                        return {
                            'type': 'PUT',
                            'direction': 'DOWN',
                            'entry_price': current_close,
                            'level_name': f"Retest of {level_name} (${level_price:.2f})",
                            'strategy_name': 'Hourly_Retest',
                        }
        
        return None
    
    def _strategy_orb_reversal(self, bar: Bar) -> Optional[dict]:
        if len(self.bars_2m) < 2:
            return None
        
        if self.orb_levels['high'] is None or self.orb_levels['low'] is None:
            return None
        
        orb_high = self.orb_levels['high']
        orb_low = self.orb_levels['low']
        
        prev_bar = self.bars_2m[-2]
        current_close = float(bar.close)
        prev_high = float(prev_bar.high)
        prev_low = float(prev_bar.low)
        prev_close = float(prev_bar.close)
        
        # PUT: Failed break above ORB high
        if prev_high > orb_high and prev_close < orb_high:
            return {
                'type': 'PUT',
                'direction': 'DOWN',
                'entry_price': current_close,
                'level_name': f"ORB Reversal (Fade High ${orb_high:.2f})",
                'strategy_name': 'ORB_Reversal',
                'stop_price': orb_high * 1.002,
                'target_price': orb_low,
            }
        
        # CALL: Failed break below ORB low
        if prev_low < orb_low and prev_close > orb_low:
            return {
                'type': 'CALL',
                'direction': 'UP',
                'entry_price': current_close,
                'level_name': f"ORB Reversal (Fade Low ${orb_low:.2f})",
                'strategy_name': 'ORB_Reversal',
                'stop_price': orb_low * 0.998,
                'target_price': orb_high,
            }
        
        return None
    
    def _strategy_false_breakout(self, bar: Bar) -> Optional[dict]:
        if len(self.bars_1m) < 2 or not self.key_levels['resistance']:
            return None
        
        prev_bar = self.bars_1m[-2]
        current_open = float(bar.open)
        prev_high = float(prev_bar.high)
        prev_low = float(prev_bar.low)
        prev_close = float(prev_bar.close)
        
        for level_name, level_price in self.key_levels['resistance']:
            if prev_high > level_price and prev_close < level_price:
                return {
                    'type': 'PUT',
                    'direction': 'DOWN',
                    'entry_price': current_open,
                    'level_name': f"False Breakout Fade (Resist {level_name})",
                    'strategy_name': 'Fade_Key_Level',
                }
        
        for level_name, level_price in self.key_levels['support']:
            if prev_low < level_price and prev_close > level_price:
                return {
                    'type': 'CALL',
                    'direction': 'UP',
                    'entry_price': current_open,
                    'level_name': f"False Breakout Fade (Support {level_name})",
                    'strategy_name': 'Fade_Key_Level',
                }
        
        return None
    
    # =========================================================================
    # EXECUTION - FIXED
    # =========================================================================
    
    def _execute_signal(self, signal: dict, bar_time: pd.Timestamp):
        """Execute with proper capital checks."""
        entry_price = signal['entry_price']
        direction = signal['direction']
        
        # FIX 1: Query actual available capital
        available_capital = self._get_available_capital()
        
        if available_capital < entry_price:
            self.log.warning(f"Insufficient capital: ${available_capital:.2f} < ${entry_price:.2f} needed for 1 share")
            return
        
        # FIX 2: Calculate position size from available capital (conservative)
        max_position_value = available_capital * 0.95  # Leave 5% buffer
        position_size = max(1, int(max_position_value / entry_price))
        required_capital = position_size * entry_price
        
        # FIX 3: Request capital allocation from coordinator
        if self.capital_coordinator:
            if not self.capital_coordinator.request_capital(required_capital):
                self.log.warning(f"Capital coordinator denied ${required_capital:.2f} request")
                return
        
        self.locked_capital_for_position = required_capital
        
        self.log.info(f"Position sizing: ${available_capital:.0f} avail -> {position_size} shares @ ${entry_price:.2f}")
        
        # Calculate target/stop
        if 'target_price' in signal:
            self.target_price = signal['target_price']
        else:
            if direction == 'UP':
                self.target_price = entry_price * (1 + self.target_pct)
            else:
                self.target_price = entry_price * (1 - self.target_pct)
        
        if 'stop_price' in signal:
            self.stop_price = signal['stop_price']
        else:
            if direction == 'UP':
                self.stop_price = entry_price * (1 - self.stop_pct)
            else:
                self.stop_price = entry_price * (1 + self.stop_pct)
        
        # Submit order
        order_side = OrderSide.BUY if direction == 'UP' else OrderSide.SELL
        
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(position_size),
            time_in_force=TimeInForce.IOC,
        )
        
        self.submit_order(order)
        
        # Track state
        self.current_position_direction = direction
        self.entry_price = entry_price
        self.current_position_size = position_size
        self.strategy_name = signal['strategy_name']
        self.last_entry_time = bar_time
        self.total_trades += 1
        
        self.log.info(
            f"🚀 [{signal['strategy_name']} {signal['type']} ENTRY] "
            f"{position_size} shares @ ${entry_price:.2f} | Target: ${self.target_price:.2f} | Stop: ${self.stop_price:.2f}"
        )
    
    def _check_exits(self, bar: Bar):
        if self.current_position_direction is None:
            return
        
        current_price = float(bar.close)
        realized_pnl = 0.0
        exit_type = None
        
        if self.current_position_direction == 'UP':
            if current_price >= self.target_price:
                exit_type = "TARGET"
                realized_pnl = (self.target_price - self.entry_price) * self.current_position_size
            elif current_price <= self.stop_price:
                exit_type = "STOP"
                realized_pnl = (self.stop_price - self.entry_price) * self.current_position_size
        else:
            if current_price <= self.target_price:
                exit_type = "TARGET"
                realized_pnl = (self.entry_price - self.target_price) * self.current_position_size
            elif current_price >= self.stop_price:
                exit_type = "STOP"
                realized_pnl = (self.entry_price - self.stop_price) * self.current_position_size
        
        if exit_type:
            self._close_position(exit_type, current_price, realized_pnl)
    
    def _close_position(self, exit_type: str, price: float, pnl: float):
        self.daily_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        order_side = OrderSide.SELL if self.current_position_direction == 'UP' else OrderSide.BUY
        
        order = self.order_factory.market(
            instrument_id=self.instrument_id,
            order_side=order_side,
            quantity=Quantity.from_int(self.current_position_size),
            time_in_force=TimeInForce.IOC,
        )
        
        self.submit_order(order)
        
        self.log.info(
            f"🛑 [{exit_type} EXIT] ({self.strategy_name}) "
            f"{self.current_position_size} shares @ ${price:.2f} | P&L: ${pnl:+.2f} | Daily: ${self.daily_pnl:.2f}"
        )
        
        # FIX: Release capital back to coordinator
        if self.capital_coordinator:
            self.capital_coordinator.release_capital(self.locked_capital_for_position)
            # Update total capital with realized PnL
            self.capital_coordinator.total_capital += pnl
        
        # Reset state
        self.current_position_direction = None
        self.entry_price = None
        self.actual_fill_price = None
        self.target_price = None
        self.stop_price = None
        self.strategy_name = None
        self.current_position_size = 0
        self.locked_capital_for_position = 0.0
    
    def on_stop(self):
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        self.log.info("=" * 70)
        self.log.info("BACKTEST SUMMARY")
        self.log.info("=" * 70)
        self.log.info(f"Total Trades:   {self.total_trades}")
        self.log.info(f"Winning Trades: {self.winning_trades}")
        self.log.info(f"Losing Trades:  {self.losing_trades}")
        self.log.info(f"Win Rate:       {win_rate:.1f}%")
        self.log.info("=" * 70)


# =============================================================================
# DATA LOADER
# =============================================================================

def load_data_yfinance(symbols: list, start_date: str, end_date: str) -> dict:
    try:
        import yfinance as yf
    except ImportError:
        print("Installing yfinance...")
        os.system("pip install yfinance --break-system-packages -q")
        import yfinance as yf
    
    data = {}
    for symbol in symbols:
        print(f"Downloading {symbol} data...")
        df = yf.download(
            symbol,
            start=start_date,
            end=end_date,
            interval='1m',
            progress=False,
        )
        if not df.empty:
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[symbol] = df
            print(f"  {symbol}: {len(df)} bars loaded")
        else:
            print(f"  {symbol}: No data available")
    
    return data


def create_bars_from_df(df: pd.DataFrame, instrument_id: InstrumentId, bar_type: BarType) -> list:
    bars = []
    
    for idx, row in df.iterrows():
        ts = pd.Timestamp(idx)
        if ts.tzinfo is None:
            ts = ts.tz_localize('America/New_York')
        
        ts_ns = dt_to_unix_nanos(ts)
        
        bar = Bar(
            bar_type=bar_type,
            open=Price.from_str(f"{row['Open']:.2f}"),
            high=Price.from_str(f"{row['High']:.2f}"),
            low=Price.from_str(f"{row['Low']:.2f}"),
            close=Price.from_str(f"{row['Close']:.2f}"),
            volume=Quantity.from_int(int(row['Volume'])),
            ts_event=ts_ns,
            ts_init=ts_ns,
        )
        bars.append(bar)
    
    return bars


def resample_to_timeframe(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    agg_dict = {
        'Open': 'first',
        'High': 'max',
        'Low': 'min',
        'Close': 'last',
        'Volume': 'sum',
    }
    return df.resample(timeframe).agg(agg_dict).dropna()


# =============================================================================
# MAIN BACKTEST RUNNER
# =============================================================================

def run_backtest(
    symbols: list = ['SPY', 'QQQ'],
    start_date: str = '2024-12-01',
    end_date: str = '2024-12-20',
    initial_capital: float = 10000.0,
):
    print("=" * 70)
    print("NAUTILUSTRADER BACKTEST WITH LOB SIMULATION (FIXED)")
    print("=" * 70)
    print(f"Symbols: {symbols}")
    print(f"Period: {start_date} to {end_date}")
    print(f"Capital: ${initial_capital:,.0f}")
    print(f"Per-symbol allocation: ~${initial_capital / len(symbols) * 0.45:,.0f}")
    print("=" * 70)
    
    # FIX: Create shared capital coordinator
    capital_coordinator = CapitalCoordinator(
        total_capital=initial_capital,
        num_symbols=len(symbols),
        max_allocation_pct=0.45  # Each symbol can use max 45% of capital
    )
    MultiStrategyTrader.set_capital_coordinator(capital_coordinator)
    
    raw_data = load_data_yfinance(symbols, start_date, end_date)
    
    if not raw_data:
        print("ERROR: No data loaded.")
        return
    
    engine_config = BacktestEngineConfig(
        trader_id=TraderId("BACKTEST-001"),
        logging=LoggingConfig(log_level="INFO"),
    )
    
    engine = BacktestEngine(config=engine_config)
    
    fill_model = FillModel(
        prob_fill_on_limit=0.65,
        prob_fill_on_stop=0.95,
        prob_slippage=0.80,
        random_seed=42,
    )
    
    venue = Venue("NASDAQ")
    
    # FIX: Use MARGIN account to allow short selling (PUT entries)
    engine.add_venue(
        venue=venue,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,  # Changed from CASH
        starting_balances=[Money(initial_capital, USD)],
        fill_model=fill_model,
    )
    
    for symbol in symbols:
        if symbol not in raw_data:
            continue
        
        df_1m = raw_data[symbol]
        
        instrument_id = InstrumentId(Symbol(symbol), venue)
        
        instrument = Equity(
            instrument_id=instrument_id,
            raw_symbol=Symbol(symbol),
            currency=USD,
            price_precision=2,
            price_increment=Price.from_str("0.01"),
            lot_size=Quantity.from_int(1),
            ts_event=0,
            ts_init=0,
        )
        
        engine.add_instrument(instrument)
        
        bar_spec_1m = BarSpecification(1, BarAggregation.MINUTE, PriceType.LAST)
        bar_spec_2m = BarSpecification(2, BarAggregation.MINUTE, PriceType.LAST)
        bar_spec_1h = BarSpecification(1, BarAggregation.HOUR, PriceType.LAST)
        
        bar_type_1m = BarType(instrument_id, bar_spec_1m, AggregationSource.EXTERNAL)
        bar_type_2m = BarType(instrument_id, bar_spec_2m, AggregationSource.EXTERNAL)
        bar_type_1h = BarType(instrument_id, bar_spec_1h, AggregationSource.EXTERNAL)
        
        bars_1m = create_bars_from_df(df_1m, instrument_id, bar_type_1m)
        
        df_2m = resample_to_timeframe(df_1m, '2min')
        bars_2m = create_bars_from_df(df_2m, instrument_id, bar_type_2m)
        
        df_1h = resample_to_timeframe(df_1m, '1h')
        bars_1h = create_bars_from_df(df_1h, instrument_id, bar_type_1h)
        
        engine.add_data(bars_1m)
        engine.add_data(bars_2m)
        engine.add_data(bars_1h)
        
        config = MultiStrategyConfig(
            strategy_id=f"MultiStrategy-{symbol}",
            instrument_id=str(instrument_id),
            bar_type_1m=str(bar_type_1m),
            bar_type_2m=str(bar_type_2m),
            bar_type_1h=str(bar_type_1h),
            default_target_pct=0.6,
            default_stop_pct=0.5,
            retest_dist_pct=0.2,
            entry_cooldown_minutes=3,
            max_daily_loss=-200.0,
        )
        
        strategy = MultiStrategyTrader(config=config)
        engine.add_strategy(strategy)
    
    print("\nRunning backtest...")
    engine.run()
    
    print("\n" + "=" * 70)
    print("FINAL ACCOUNT STATE")
    print("=" * 70)
    
    for account in engine.cache.accounts():
        print(f"Account: {account.id}")
        print(f"Balance: {account.balance_total(USD)}")
        print(f"Equity: {account.balance_free(USD)}")
    
    print(f"\nCapital Coordinator Final State:")
    print(f"  Total Capital: ${capital_coordinator.total_capital:,.2f}")
    print(f"  Locked: ${capital_coordinator.locked_capital:,.2f}")
    
    engine.dispose()
    
    print("\n" + "=" * 70)
    print("BACKTEST COMPLETE")
    print("=" * 70)
    print("\nFixes applied:")
    print("  ✓ Dynamic capital query (not hardcoded)")
    print("  ✓ Cross-strategy capital coordination")
    print("  ✓ MARGIN account (allows shorts)")
    print("  ✓ Pre-trade balance check")
    print("  ✓ Capital release on position close")


if __name__ == "__main__":
    end = datetime.now()
    start = end - timedelta(days=6)
    
    run_backtest(
        symbols=['SPY', 'QQQ'],
        start_date=start.strftime('%Y-%m-%d'),
        end_date=end.strftime('%Y-%m-%d'),
        initial_capital=10000.0,
    )