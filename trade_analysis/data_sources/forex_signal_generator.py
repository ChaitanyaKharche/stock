import os
import sys
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

from ..paths import LOGS_DIR

API_KEY = os.getenv('ALPACA_API_KEY')
SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

class ForexSignalGenerator:
    """
    Forex signal generator using Alpaca forex rates API
    Pairs: EUR/USD, GBP/JPY, USD/JPY
    """

    def __init__(self):
        self.pairs = ['EUR/USD', 'GBP/JPY', 'USD/JPY']
        self.log_file = LOGS_DIR / "forex_signals.log"
        
        # Technical thresholds
        self.rsi_buy = 30
        self.rsi_sell = 70
        self.momentum_threshold = 0.1  # 0.1% move
        
        # Rate history tracking
        self.rate_history = {pair: [] for pair in self.pairs}
        
        print(f"\n{'='*70}")
        print(f"FOREX SIGNAL GENERATOR")
        print(f"{'='*70}")
        print(f"Pairs: {', '.join(self.pairs)}")
        print(f"API: Alpaca Forex Rates")
        print(f"Poll interval: 60 seconds\n")
    
    def log(self, message):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{timestamp}] {message}"
        print(msg)
        
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
        except:
            pass
    
    def fetch_rates(self):
        """Fetch latest forex rates"""
        try:
            url = "https://data.alpaca.markets/v1beta1/forex/latest/rates"
            headers = {
                "APCA-API-KEY-ID": API_KEY,
                "APCA-API-SECRET-KEY": SECRET_KEY
            }
            
            # Format pairs for API (e.g., EURUSD, GBPJPY)
            api_pairs = ','.join([p.replace('/', '') for p in self.pairs])
            params = {"currency_pairs": api_pairs}
            
            response = requests.get(url, headers=headers, params=params)
            
            if response.status_code == 200:
                data = response.json()
                rates = {}
                
                for pair in self.pairs:
                    api_pair = pair.replace('/', '')
                    if api_pair in data.get('rates', {}):
                        rate_data = data['rates'][api_pair]
                        rates[pair] = {
                            'bid': float(rate_data.get('BidPrice', 0)),
                            'ask': float(rate_data.get('AskPrice', 0)),
                            'mid': (float(rate_data.get('BidPrice', 0)) + float(rate_data.get('AskPrice', 0))) / 2,
                            'time': rate_data.get('Timestamp')
                        }
                
                return rates
            else:
                self.log(f"API ERROR: {response.status_code} - {response.text[:100]}")
                return {}
        
        except Exception as e:
            self.log(f"ERROR fetching rates: {str(e)[:80]}")
            return {}
    
    def calculate_rsi(self, prices, period=14):
        """Calculate RSI"""
        try:
            if len(prices) < period + 1:
                return 50
            
            deltas = np.diff(prices)
            seed = deltas[-period:]
            up = seed[seed >= 0].sum() / period if len(seed) > 0 else 0
            down = -seed[seed < 0].sum() / period if len(seed) > 0 else 0
            rs = up / down if down != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 50
            
            return rsi
        except:
            return 50
    
    def generate_signal(self, pair, rate_data):
        """Generate trading signal"""
        try:
            mid_price = rate_data['mid']
            
            # Store in history
            self.rate_history[pair].append({
                'time': datetime.now(),
                'price': mid_price
            })
            
            # Keep last 100 rates
            if len(self.rate_history[pair]) > 100:
                self.rate_history[pair] = self.rate_history[pair][-100:]
            
            # Need at least 20 rates for indicators
            if len(self.rate_history[pair]) < 20:
                self.log(f"[{pair}] Warming up... ({len(self.rate_history[pair])}/20)")
                return
            
            # Extract prices
            prices = np.array([r['price'] for r in self.rate_history[pair]])
            
            # RSI
            rsi = self.calculate_rsi(prices, period=14)
            
            # Momentum (last 5 bars)
            momentum_pct = ((prices[-1] - prices[-5]) / prices[-5] * 100) if prices[-5] > 0 else 0
            
            # Trend (SMA cross)
            sma_fast = prices[-10:].mean()
            sma_slow = prices[-20:].mean()
            trend = "UP" if sma_fast > sma_slow else "DOWN"
            
            # Signal generation
            signal = "HOLD"
            strength = "NEUTRAL"
            
            if rsi < self.rsi_buy and momentum_pct < -self.momentum_threshold and trend == "DOWN":
                signal = "BUY"
                strength = "STRONG" if rsi < 25 else "MEDIUM"
            elif rsi > self.rsi_sell and momentum_pct > self.momentum_threshold and trend == "UP":
                signal = "SELL"
                strength = "STRONG" if rsi > 75 else "MEDIUM"
            elif rsi < 40 and trend == "UP":
                signal = "BUY"
                strength = "WEAK"
            elif rsi > 60 and trend == "DOWN":
                signal = "SELL"
                strength = "WEAK"
            
            # Log signal
            self.log(
                f"[{pair}] {signal} ({strength}) | "
                f"Price: {mid_price:.5f} | "
                f"RSI: {rsi:.1f} | "
                f"Momentum: {momentum_pct:+.2f}% | "
                f"Trend: {trend}"
            )
        
        except Exception as e:
            self.log(f"ERROR generating signal for {pair}: {str(e)[:60]}")
    
    def run(self):
        """Main polling loop"""
        try:
            self.log("="*70)
            self.log("STARTING FOREX SIGNAL GENERATOR")
            self.log("="*70)
            self.log("")
            
            poll_count = 0
            
            while True:
                try:
                    poll_count += 1
                    self.log(f"--- Poll #{poll_count} ---")
                    
                    # Fetch rates
                    rates = self.fetch_rates()
                    
                    if not rates:
                        self.log("No rate data received")
                        time.sleep(60)
                        continue
                    
                    # Generate signals
                    for pair, rate_data in rates.items():
                        self.generate_signal(pair, rate_data)
                    
                    self.log("")
                    time.sleep(60)  # Poll every minute
                
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    self.log(f"POLL ERROR: {str(e)[:80]}")
                    time.sleep(60)
        
        except KeyboardInterrupt:
            self.log("\nSTOPPED BY USER")
        except Exception as e:
            self.log(f"FATAL ERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    if not API_KEY or not SECRET_KEY:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        sys.exit(1)
    
    generator = ForexSignalGenerator()
    generator.run()

if __name__ == "__main__":
    main()