import os
import sys
import pandas as pd
import numpy as np
import json
import asyncio
import websockets
from datetime import datetime, timedelta
from pathlib import Path
import logging
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)

ALPACA_API_KEY = os.getenv('ALPACA_API_KEY')
ALPACA_SECRET_KEY = os.getenv('ALPACA_SECRET_KEY')

class CryptoMarketPredictor:
    """
    Live crypto predictor - DIRECT WebSocket (bypasses alpaca-trade-api library)
    Connects directly to: wss://stream.data.alpaca.markets/v1beta3/crypto/us
    """
    
    def __init__(self):
        self.symbols = ['BTC/USD', 'ETH/USD', 'SOL/USD']
        self.data_dir = Path("crypto_historical_data")
        self.event_log = Path("crypto_events.json")
        self.log_file = Path("crypto_predictions.log")
        
        # Load historical patterns
        self.event_patterns = self.load_event_patterns()
        self.historical_data = self.load_historical_data()
        
        # Real-time tracking
        self.latest_bars = {}
        self.session_trades = {sym: [] for sym in self.symbols}
        
        print(f"\n{'='*70}")
        print(f"LIVE CRYPTO MARKET PREDICTOR (DIRECT WebSocket)")
        print(f"{'='*70}")
        print(f"Mode: LIVE STREAMING (Direct WebSocket)")
        print(f"Endpoint: wss://stream.data.alpaca.markets/v1beta3/crypto/us")
        print(f"Symbols: {', '.join([s.replace('/USD', '') for s in self.symbols])}")
        print(f"Loaded patterns from: {self.event_log}")
        print(f"API Key: {'SET' if ALPACA_API_KEY else 'MISSING'}\n")
    
    def load_event_patterns(self):
        """Load historical event patterns"""
        try:
            if not self.event_log.exists():
                print(f"WARNING: Event log not found ({self.event_log})")
                return {}
            
            with open(self.event_log, 'r') as f:
                data = json.load(f)
                print(f"[LOAD] Loaded event patterns: {len(data.get('events', {}))} symbols analyzed")
                return data.get('patterns', {})
        except Exception as e:
            print(f"ERROR loading patterns: {e}")
            return {}
    
    def load_historical_data(self):
        """Load historical price data"""
        data = {}
        try:
            for ticker in ['BTC', 'ETH', 'SOL']:
                csv_path = self.data_dir / f"{ticker}_historical.csv"
                
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    data[ticker] = df
                    print(f"[LOAD] {ticker}: {len(df)} historical days")
        except Exception as e:
            print(f"ERROR loading historical data: {e}")
        
        return data
    
    def log_prediction(self, message):
        """Log predictions to file + console"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        msg = f"[{timestamp}] {message}"
        print(msg)
        try:
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(msg + "\n")
        except:
            pass
    
    def calculate_technical_score(self, symbol):
        """Calculate technical indicator score (0-100)"""
        try:
            if len(self.session_trades[symbol]) < 5:
                return {'score': 50, 'rsi': 50, 'momentum_pct': 0}
            
            trades = self.session_trades[symbol][-20:]  # Last 20 bars
            closes = np.array([t['close'] for t in trades])
            volumes = np.array([t['volume'] for t in trades])
            
            # RSI (5-period)
            deltas = np.diff(closes)
            seed = deltas[-5:]
            up = seed[seed >= 0].sum() / 5 if len(seed) > 0 else 0
            down = -seed[seed < 0].sum() / 5 if len(seed) > 0 else 0
            rs = up / down if down != 0 else 0
            rsi = 100 - (100 / (1 + rs)) if rs >= 0 else 50
            
            # Momentum (2-bar)
            momentum = ((closes[-1] - closes[-2]) / closes[-2] * 100) if closes[-2] > 0 else 0
            
            # Volume trend
            avg_vol = volumes[-3:].mean()
            vol_spike = (volumes[-1] / avg_vol - 1) * 100 if avg_vol > 0 else 0
            
            # Combine
            score = 50
            score += rsi - 50
            score += momentum
            score += min(vol_spike, 30)
            score = max(0, min(100, score))
            
            return {'score': score, 'rsi': rsi, 'momentum_pct': momentum}
        except:
            return {'score': 50, 'rsi': 50, 'momentum_pct': 0}
    
    def calculate_event_risk(self, symbol):
        """Calculate event-driven risk"""
        upcoming_events = {
            'BTC/USD': [
                {'days': 2, 'event': 'Market activity', 'sentiment': 'VARIABLE'},
                {'days': 50, 'event': 'Fed decision', 'sentiment': 'VARIABLE'},
            ],
            'ETH/USD': [
                {'days': 2, 'event': 'Altseason momentum', 'sentiment': 'POSITIVE'},
                {'days': 50, 'event': 'Fed decision', 'sentiment': 'VARIABLE'},
            ],
            'SOL/USD': [
                {'days': 2, 'event': 'Altseason strength', 'sentiment': 'POSITIVE'},
                {'days': 7, 'event': 'Catalyst events', 'sentiment': 'POSITIVE'},
            ]
        }
        
        if symbol not in upcoming_events:
            return 50, 'NEUTRAL'
        
        catalyst_score = 50
        strength = 'NEUTRAL'
        
        for event in upcoming_events[symbol]:
            if event['days'] <= 7:
                if event['sentiment'] == 'POSITIVE':
                    catalyst_score += 15
                    strength = 'STRONG_BUY' if catalyst_score > 70 else 'BUY'
        
        return catalyst_score, strength
    
    async def handle_message(self, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)
            
            # Handle list of messages
            if isinstance(data, list):
                data = data[0] if data else {}
            
            msg_type = data.get('T')
            
            # Connection/Auth messages
            if msg_type == 'success':
                self.log_prediction(data.get('msg', 'Success'))
                return
            
            if msg_type == 'subscription':
                self.log_prediction(f"Subscribed to: {data}")
                return
            
            # Bar data (minute bars)
            if msg_type == 'b':
                symbol = data.get('S')
                if symbol not in self.symbols:
                    return
                
                bar = {
                    'time': data.get('t'),
                    'open': data.get('o'),
                    'high': data.get('h'),
                    'low': data.get('l'),
                    'close': data.get('c'),
                    'volume': data.get('v')
                }
                
                if symbol not in self.session_trades:
                    self.session_trades[symbol] = []
                
                self.session_trades[symbol].append(bar)
                
                # Keep last 100 bars
                if len(self.session_trades[symbol]) > 100:
                    self.session_trades[symbol] = self.session_trades[symbol][-100:]
                
                # Generate prediction every 5th bar
                if len(self.session_trades[symbol]) % 5 == 0:
                    await self.generate_prediction(symbol, bar['close'])
        
        except Exception as e:
            self.log_prediction(f"ERROR handling message: {str(e)[:60]}")
    
    async def generate_prediction(self, symbol, close_price):
        """Generate combined prediction"""
        try:
            tech_data = self.calculate_technical_score(symbol)
            tech_score = tech_data['score']
            
            event_score, catalyst = self.calculate_event_risk(symbol)
            
            combined_score = (tech_score * 0.4 + event_score * 0.6)
            combined_score = max(0, min(100, combined_score))
            
            if combined_score >= 65:
                signal = 'BUY'
                conviction = 'HIGH' if combined_score >= 75 else 'MEDIUM'
            elif combined_score <= 35:
                signal = 'SELL'
                conviction = 'HIGH' if combined_score <= 25 else 'MEDIUM'
            else:
                signal = 'HOLD'
                conviction = 'NEUTRAL'
            
            clean_symbol = symbol.replace('/USD', '')
            self.log_prediction(
                f"[{clean_symbol}] {signal} | Price ${close_price:.2f} | "
                f"Tech: {tech_score:.0f} | Event: {event_score:.0f} | "
                f"Combined: {combined_score:.0f} | RSI: {tech_data.get('rsi', 50):.0f} | "
                f"Catalyst: {catalyst}"
            )
        
        except Exception as e:
            self.log_prediction(f"ERROR generating prediction: {str(e)[:60]}")
    
    async def run_live(self):
        """Run live WebSocket stream"""
        uri = "wss://stream.data.alpaca.markets/v1beta3/crypto/us"
        
        try:
            self.log_prediction("="*70)
            self.log_prediction("STARTING LIVE CRYPTO WebSocket STREAM")
            self.log_prediction("="*70)
            self.log_prediction(f"Connecting to: {uri}")
            self.log_prediction("")
            
            async with websockets.connect(uri) as websocket:
                # Authenticate
                auth_msg = {
                    "action": "auth",
                    "key": ALPACA_API_KEY,
                    "secret": ALPACA_SECRET_KEY
                }
                await websocket.send(json.dumps(auth_msg))
                self.log_prediction("Sent authentication")
                
                # Handle auth response
                response = await websocket.recv()
                await self.handle_message(response)
                
                # Subscribe to bars
                sub_msg = {
                    "action": "subscribe",
                    "bars": self.symbols
                }
                await websocket.send(json.dumps(sub_msg))
                self.log_prediction(f"Subscribed to: {self.symbols}")
                self.log_prediction("")
                
                # Listen for messages
                async for message in websocket:
                    await self.handle_message(message)
        
        except KeyboardInterrupt:
            self.log_prediction("\nSTOPPED BY USER")
        except Exception as e:
            self.log_prediction(f"WebSocket ERROR: {e}")
            import traceback
            traceback.print_exc()


async def main():
    if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
        print("ERROR: Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars")
        sys.exit(1)
    
    predictor = CryptoMarketPredictor()
    await predictor.run_live()


if __name__ == "__main__":
    asyncio.run(main())