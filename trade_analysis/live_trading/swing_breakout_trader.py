
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json

from ..signals import gamma_exposure


class LiveBreakoutTrader:
    """
    Production trader with morning confirmation filter, ATR-adaptive risk,
    multi-day support/resistance confluence, a gap/volatility regime filter,
    and a dealer gamma-exposure (GEX) regime filter.

    KEY DIFFERENCE: Don't enter on first breakout signal.
    Instead: wait for 9:30-9:45 confirmation that buyers are serious,
    score that confirmation against multi-day S/R confluence plus dealer
    gamma positioning, and size risk off current volatility (ATR) instead
    of fixed percentages.
    """

    def __init__(self, symbol, alpaca_client, cash_per_trade=250):
        self.symbol = symbol
        self.client = alpaca_client
        self.cash_per_trade = cash_per_trade
        self.risk_fraction = 0.02  # Max 2% of cash_per_trade risked per trade

        # ATR-based risk (replaces fixed target_pct/stop_pct - adapts to regime)
        self.atr_period = 14
        self.stop_atr_mult = 1.0
        self.target_atr_mult = 2.5
        self.retest_atr_mult = 0.15  # confirmation retest zone, in ATRs

        # Multi-day S/R confluence
        self.confluence_lookback_days = 3
        self.confluence_zone_pct = 0.0015  # levels within 0.15% count as the same zone
        self.confluence_size_mult = 1.5    # size boost when breakout aligns with confluence

        # Gap / volatility regime filter
        self.gap_atr_mult_threshold = 2.0     # gap beyond this many ATRs = elevated regime
        self.elevated_regime_size_mult = 0.5  # derate size in elevated regime

        # Dealer gamma-exposure (GEX) regime filter: NET_LONG_GAMMA dealers
        # dampen/pin price (derate breakout size), NET_SHORT_GAMMA dealers
        # amplify moves (breakouts more likely to run, keep full size)
        self.gex_long_gamma_size_mult = 0.6
        self.gex_regime = "UNKNOWN"
        self.gex_top_strikes = []

        # Morning confirmation window
        self.morning_start = 9.5   # 9:30 AM
        self.morning_end = 9.75    # 9:45 AM

        self.position = None        # "LONG" or None
        self.position_size = 0
        self.entry_price = None
        self.entry_time = None
        self.stop_price = None
        self.target_price = None

        self.yesterday_high = None
        self.yesterday_low = None
        self.yesterday_close = None
        self.premarket_high = None
        self.atr = None
        self.confluence_levels = []   # [(label, price), ...]
        self.gap_regime = "NORMAL"    # NORMAL | ELEVATED
        self.size_multiplier = 1.0

    def get_market_hours_from_date(self, date_str):
        """Convert market time to decimal hours"""
        try:
            dt = datetime.fromisoformat(date_str)
            hours = dt.hour + dt.minute / 60 + dt.second / 3600
            return hours
        except:
            return None

    def fetch_yesterday_levels(self):
        """Get yesterday's high/low/close (signal trigger + gap baseline)"""
        try:
            bars = self.client.get_stock_bars(
                self.symbol,
                timeframe='day',
                limit=5
            )

            if len(bars) < 2:
                return None, None

            # Yesterday = second to last bar
            yesterday = bars[-2]
            self.yesterday_high = yesterday.high
            self.yesterday_low = yesterday.low
            self.yesterday_close = yesterday.close

            return self.yesterday_high, self.yesterday_low
        except Exception as e:
            print(f"Error fetching levels: {e}")
            return None, None

    def fetch_premarket_and_atr(self):
        """
        Fetch premarket high (4:00-9:30 AM) and compute a 5-min ATR from
        recent intraday bars, so the retest/target/stop scale with current
        volatility instead of a fixed percentage.
        """
        try:
            bars = self.client.get_stock_bars(
                self.symbol,
                timeframe='5Min',
                start=datetime.now() - timedelta(days=2),
                end=datetime.now()
            )

            premarket_bars = [b for b in bars if 4 <= self.get_market_hours_from_date(b.timestamp) <= 9.5]

            if premarket_bars:
                self.premarket_high = max(b.high for b in premarket_bars)
            else:
                self.premarket_high = self.yesterday_high

            self.atr = self._compute_atr(bars, self.atr_period)

            return self.premarket_high, self.atr
        except Exception as e:
            print(f"Error fetching premarket/ATR: {e}")
            self.premarket_high = self.yesterday_high
            return self.premarket_high, self.atr

    def _compute_atr(self, bars, period):
        """Average true range over the most recent `period` bars"""
        if not bars or len(bars) < 2:
            return None

        recent = bars[-(period + 1):]
        true_ranges = []

        for i in range(1, len(recent)):
            high = recent[i].high
            low = recent[i].low
            prev_close = recent[i - 1].close
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            true_ranges.append(tr)

        if not true_ranges:
            return None

        return sum(true_ranges) / len(true_ranges)

    def fetch_confluence_levels(self):
        """
        Build support/resistance zones from the past N sessions' highs/lows
        and premarket highs/lows, so a breakout that lines up with a level
        touched multiple times recently is treated as higher-conviction.
        """
        self.confluence_levels = []
        try:
            daily_bars = self.client.get_stock_bars(
                self.symbol,
                timeframe='day',
                limit=self.confluence_lookback_days + 2
            )

            session_bars = daily_bars[-(self.confluence_lookback_days + 1):-1]
            for bar in session_bars:
                self.confluence_levels.append((f"Session high {bar.timestamp}", bar.high))
                self.confluence_levels.append((f"Session low {bar.timestamp}", bar.low))

            intraday_bars = self.client.get_stock_bars(
                self.symbol,
                timeframe='5Min',
                start=datetime.now() - timedelta(days=self.confluence_lookback_days + 1),
                end=datetime.now()
            )

            by_day = {}
            for b in intraday_bars:
                hour = self.get_market_hours_from_date(b.timestamp)
                if hour is None or not (4 <= hour <= 9.5):
                    continue
                day_key = b.timestamp[:10] if isinstance(b.timestamp, str) else b.timestamp.date()
                by_day.setdefault(day_key, []).append(b)

            for day_key, day_bars in by_day.items():
                self.confluence_levels.append((f"PM high {day_key}", max(b.high for b in day_bars)))
                self.confluence_levels.append((f"PM low {day_key}", min(b.low for b in day_bars)))

            return self.confluence_levels
        except Exception as e:
            print(f"Error fetching confluence levels: {e}")
            return self.confluence_levels

    def check_confluence(self, level_price):
        """Does `level_price` line up with another recent S/R zone within tolerance?"""
        if not self.confluence_levels or level_price is None:
            return False, []

        matches = [
            label for label, price in self.confluence_levels
            if abs(price - level_price) / level_price <= self.confluence_zone_pct
        ]
        return len(matches) > 0, matches

    def check_gap_regime(self, current_price):
        """
        Flag an elevated-volatility regime when price has moved well beyond
        the recent ATR from yesterday's close (headline/gap-driven moves),
        so size gets derated instead of trusting static S/R levels to hold.
        """
        if self.yesterday_close is None or not self.atr:
            self.gap_regime = "NORMAL"
            return self.gap_regime

        gap = abs(current_price - self.yesterday_close)
        self.gap_regime = "ELEVATED" if gap > self.gap_atr_mult_threshold * self.atr else "NORMAL"

        return self.gap_regime

    def fetch_gex_regime(self):
        """
        Pulls today's dealer gamma-exposure regime from the live 0DTE option
        chain (see gamma_exposure.py). NET_LONG_GAMMA days tend to see
        breakouts fade/pin near dealer hedging levels; NET_SHORT_GAMMA days
        tend to see them run. Also folds the top gamma-concentration strikes
        into the confluence levels, so a breakout aligning with a major
        dealer hedging strike scores as confluence too.
        """
        try:
            result = gamma_exposure.compute_gex(self.symbol)
            if 'error' in result:
                print(f"[GEX] {self.symbol}: {result['error']}")
                self.gex_regime = "UNKNOWN"
                return self.gex_regime

            self.gex_regime = result['regime']
            self.gex_top_strikes = result['top_gamma_strikes']

            for strike, gex in self.gex_top_strikes:
                self.confluence_levels.append((f"Gamma strike ${strike:.2f} ({gex:+,.0f})", strike))

            return self.gex_regime
        except Exception as e:
            print(f"Error fetching GEX regime: {e}")
            self.gex_regime = "UNKNOWN"
            return self.gex_regime

    def check_morning_confirmation(self, current_price, current_time):
        """
        Morning confirmation filter (9:30-9:45), retest zone scaled to ATR.

        Entry signals:
        1. Price retests yesterday's high (within retest_atr_mult * ATR)
        2. Price gaps up and stays above yesterday's high
        3. Any price touches premarket high (strongest confirmation)

        This removes the "shakeout fear" by confirming buyers are interested.
        """

        if self.yesterday_high is None:
            return False, "No yesterday levels"

        current_hours = self.get_market_hours_from_date(datetime.now().isoformat())

        if not (self.morning_start <= current_hours <= self.morning_end):
            return False, f"Not in morning window (current: {current_hours:.2f}h)"

        retest_buffer = (self.retest_atr_mult * self.atr) if self.atr else self.yesterday_high * 0.003
        retest_level = self.yesterday_high - retest_buffer

        if current_price >= retest_level:
            return True, "Retest confirmed"

        if current_price > self.premarket_high * 0.998:
            return True, "Premarket high touch confirmed"

        return False, "No confirmation yet"

    def check_entry_signal(self, current_price, current_time):
        """
        Daily signal (premarket): close yesterday above yesterday_high.
        Intraday confirmation (9:30-9:45): retest or gap-hold.
        On confirmation, scores position size via confluence + gap regime.
        """

        if self.yesterday_high is None:
            self.fetch_yesterday_levels()
            return False, "Fetching yesterday levels"

        daily_signal = current_price > self.yesterday_high

        if not daily_signal:
            return False, "No daily breakout signal yet"

        confirmed, reason = self.check_morning_confirmation(current_price, current_time)
        if not confirmed:
            return False, reason

        has_confluence, matches = self.check_confluence(self.yesterday_high)
        self.size_multiplier = self.confluence_size_mult if has_confluence else 1.0

        if self.gap_regime == "ELEVATED":
            self.size_multiplier *= self.elevated_regime_size_mult
            reason = f"{reason} | ELEVATED gap regime, size derated"
        elif has_confluence:
            reason = f"{reason} | confluence with {matches}"

        if self.gex_regime == "NET_LONG_GAMMA":
            self.size_multiplier *= self.gex_long_gamma_size_mult
            reason = f"{reason} | NET_LONG_GAMMA (dealer pinning), size derated"
        elif self.gex_regime == "NET_SHORT_GAMMA":
            reason = f"{reason} | NET_SHORT_GAMMA (dealer amplifying)"

        return True, reason

    def calculate_position_size(self):
        """Size from % risk, ATR-based stop distance, and the confluence/regime multiplier"""
        stop_distance = self.stop_atr_mult * self.atr if self.atr else self.entry_price * 0.02
        stop_pct = stop_distance / self.entry_price

        position_pct = self.risk_fraction / stop_pct
        num_shares = int((self.cash_per_trade * position_pct * self.size_multiplier) / self.entry_price)

        return num_shares

    def place_entry_order(self):
        """Place market buy order, set ATR-based target/stop"""
        try:
            num_shares = self.calculate_position_size()

            order = self.client.submit_order(
                symbol=self.symbol,
                qty=num_shares,
                side="buy",
                type="market",
                time_in_force="day"
            )

            self.position = "LONG"
            self.position_size = num_shares
            self.entry_price = order.filled_avg_price
            self.entry_time = datetime.now()

            stop_distance = self.stop_atr_mult * self.atr if self.atr else self.entry_price * 0.02
            target_distance = self.target_atr_mult * self.atr if self.atr else self.entry_price * 0.05

            self.stop_price = self.entry_price - stop_distance
            self.target_price = self.entry_price + target_distance

            print(f"\n✓ ENTRY: {self.symbol} @ {self.entry_price:.2f}")
            print(f"  Shares: {num_shares} (size x{self.size_multiplier:.2f}, "
                  f"gap regime: {self.gap_regime}, GEX regime: {self.gex_regime})")
            print(f"  Target: {self.target_price:.2f} | Stop: {self.stop_price:.2f}"
                  + (f" (ATR: {self.atr:.3f})" if self.atr else ""))

            return True
        except Exception as e:
            print(f"Error placing entry order: {e}")
            return False

    def check_exit_signals(self, current_price):
        """Check if position should exit against ATR-based target/stop"""
        if self.position is None or self.entry_price is None:
            return False, None

        if self.target_price is not None and current_price >= self.target_price:
            return True, "TARGET"

        if self.stop_price is not None and current_price <= self.stop_price:
            return True, "STOP"

        return False, None

    def close_position(self, reason):
        """Close existing position"""
        try:
            if self.position is None:
                return

            quote = self.client.get_latest_bar(self.symbol)
            exit_price = quote.close

            self.client.submit_order(
                symbol=self.symbol,
                qty=self.position_size,
                side="sell",
                type="market",
                time_in_force="day"
            )

            pnl_pct = (exit_price / self.entry_price - 1) * 100
            pnl_amount = (exit_price - self.entry_price) * self.position_size

            print(f"\n✓ EXIT ({reason}): {self.symbol} @ {exit_price:.2f}")
            print(f"  PnL: {pnl_pct:+.2f}% (${pnl_amount:+.2f})")
            print(f"  Held: {(datetime.now() - self.entry_time).seconds / 60:.1f} minutes")

            self.position = None
            self.position_size = 0
            self.entry_price = None
            self.stop_price = None
            self.target_price = None

            return True
        except Exception as e:
            print(f"Error closing position: {e}")
            return False

    def run_market_day(self):
        """
        Main trading loop for a single market day

        Flow:
        1. Before 9:30: fetch yesterday/premarket levels, multi-day confluence, ATR
        2. At open: classify gap regime (normal vs elevated volatility)
        3. 9:30-9:45: wait for confirmation (retest or gap-hold), sized by confluence/regime
        4. Throughout day: monitor position, exit on ATR-based target/stop
        """

        print(f"\nStarting {self.symbol} trading day: {datetime.now()}")

        self.fetch_yesterday_levels()
        self.fetch_premarket_and_atr()
        self.fetch_confluence_levels()
        self.fetch_gex_regime()

        if self.yesterday_high is None:
            print("Could not fetch yesterday levels, skipping")
            return

        print(f"Yesterday high: {self.yesterday_high:.2f}")
        print(f"Premarket high: {self.premarket_high:.2f}")
        print(f"ATR({self.atr_period}, 5min): {self.atr:.3f}" if self.atr else "ATR: unavailable")
        print(f"GEX regime: {self.gex_regime}")
        print(f"Confluence levels tracked: {len(self.confluence_levels)} (incl. {len(self.gex_top_strikes)} gamma strikes)")

        gap_checked = False
        max_iterations = 500  # ~8 hours * 60 min / 1 min per check

        for i in range(max_iterations):
            current_time = datetime.now()

            # Market closed
            if current_time.hour >= 16:
                if self.position:
                    print("\nMarket closing, closing position...")
                    self.close_position("MARKET_CLOSE")
                print("Market closed, ending session")
                break

            # Get current price
            try:
                quote = self.client.get_latest_bar(self.symbol)
                current_price = quote.close
            except:
                print(f"Error getting quote, retrying...")
                time.sleep(5)
                continue

            if not gap_checked and self.get_market_hours_from_date(current_time.isoformat()) >= self.morning_start:
                self.check_gap_regime(current_price)
                print(f"Gap regime: {self.gap_regime}")
                gap_checked = True

            # Check entry
            if self.position is None:
                should_enter, reason = self.check_entry_signal(current_price, current_time)

                if should_enter:
                    self.entry_price = current_price
                    self.place_entry_order()

            # Check exit
            if self.position is not None:
                should_exit, reason = self.check_exit_signals(current_price)

                if should_exit:
                    self.close_position(reason)

            # Wait before next check
            time.sleep(60)  # Check every minute


# Example usage (requires Alpaca client)
if __name__ == "__main__":
    print("""
    Production Breakout Trader - Morning Confirmation + ATR Risk +
    Multi-Day S/R Confluence + Gap/Volatility Regime + Dealer GEX Regime

    - Entry trigger stays anchored to previous-day high/low + premarket
      (fastest-reacting, most-watched levels)
    - Target/stop/retest zone scale with 5-min ATR instead of fixed %,
      so risk adapts to the current volatility regime
    - Breakouts that align with a 3-day support/resistance zone, or with
      a major dealer gamma-concentration strike, get sized up (confluence)
    - Breakouts during an elevated gap regime (headline/news-driven) or a
      NET_LONG_GAMMA dealer-positioning day (dealers dampen/pin price) get
      sized down instead of held to a static level

    NOTE: prior fixed-percentage backtest numbers no longer apply to this
    version - re-backtest before live use.
    """)

    # from alpaca.trading.client import TradingClient
    #
    # API_KEY = "YOUR_API_KEY"
    # SECRET_KEY = "YOUR_SECRET_KEY"
    # client = TradingClient(API_KEY, SECRET_KEY, paper=True)
    #
    # trader = LiveBreakoutTrader("TSLA", client, cash_per_trade=250)
    # trader.run_market_day()
