# trade_analysis/agent.py
"""
Autonomous agent that uses your existing enhanced_api endpoints
This way you don't need to refactor everything
"""

import asyncio
import httpx
import json
import sqlite3
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd

class TradingAgent:
    """
    Agent that calls your existing API endpoints for analysis
    Adds autonomous decision-making on top of your current system
    """
    
    def __init__(self, api_url: str = "http://localhost:8000"):
        self.api_url = api_url
        self.memory_db = "/tmp/agent_memory.db"
        self.positions = {}
        self.daily_trades = 0
        self.max_daily_trades = 3
        
        # Decision thresholds
        self.thresholds = {
            'entry_confidence': 75,
            'stop_loss': -0.15,
            'take_profit': 0.30,
            'max_hold_minutes': 30
        }
        
        self._init_memory()
    
    def _init_memory(self):
        """Initialize SQLite for learning"""
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TIMESTAMP,
                symbol TEXT,
                signal TEXT,
                confidence INTEGER,
                entry_price REAL,
                exit_price REAL,
                pnl REAL,
                reasoning TEXT,
                api_response TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conditions TEXT,
                success_rate REAL,
                avg_return REAL,
                occurrences INTEGER
            )
        ''')
        
        conn.commit()
        conn.close()
    
    async def run(self):
        """Main agent loop using your API"""
        print("🤖 Agent Started - Using Enhanced API")
        
        while True:
            try:
                # Check if your API is running
                async with httpx.AsyncClient() as client:
                    health = await client.get(f"{self.api_url}/")
                    if health.status_code != 200:
                        print("API not available. Waiting...")
                        await asyncio.sleep(900)
                        continue
                
                # Get signals from your API
                await self.scan_with_api()
                
                # Manage any open positions
                await self.manage_positions()
                
                # Learn from recent trades
                if datetime.now().hour == 16:  # After market close
                    await self.learn_from_trades()
                
                # Rate limit friendly
                await asyncio.sleep(900)  # Check every minute
                
            except Exception as e:
                print(f"Agent error: {e}")
                await asyncio.sleep(900)
    
    async def scan_with_api(self):
        """Call your enhanced API for signals"""
        
        symbols = ['QQQ', 'NVDA']
        signals = []
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            for symbol in symbols:
                try:
                    # Call YOUR existing endpoint
                    response = await client.post(
                        f"{self.api_url}/predict/enhanced/",
                        params={
                            "symbol": symbol,
                            "timeframe": "5m",
                            "strategy_mode": "momentum"
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        
                        # Extract signal from YOUR API response
                        signal = data.get('signal', 'HOLD')
                        confidence = data.get('confidence', 0)
                        reasoning = data.get('reasoning', '')
                        
                        if signal != 'HOLD' and confidence >= self.thresholds['entry_confidence']:
                            signals.append({
                                'symbol': symbol,
                                'signal': signal,
                                'confidence': confidence,
                                'reasoning': reasoning,
                                'full_response': data
                            })
                            
                            print(f"📊 Signal from API: {symbol} - {signal} ({confidence}%)")
                    
                except Exception as e:
                    print(f"Error getting signal for {symbol}: {e}")
        
        # Execute best signal if any
        if signals and self.daily_trades < self.max_daily_trades:
            # Sort by confidence
            signals.sort(key=lambda x: x['confidence'], reverse=True)
            best = signals[0]
            
            await self.execute_trade(best)
    
    async def execute_trade(self, signal_data: Dict):
        """Execute trade based on API signal"""
        
        symbol = signal_data['symbol']
        
        # Check if already in position
        if symbol in self.positions:
            print(f"Already in position for {symbol}")
            return
        
        # Get current price
        import yfinance as yf
        ticker = yf.Ticker(symbol)
        current_price = ticker.history(period='1d')['Close'].iloc[-1]
        
        # Record position
        self.positions[symbol] = {
            'entry_price': current_price,
            'entry_time': datetime.now(),
            'signal': signal_data['signal'],
            'confidence': signal_data['confidence'],
            'reasoning': signal_data['reasoning']
        }
        
        # Save to database
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO trades (timestamp, symbol, signal, confidence, entry_price, reasoning, api_response)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(),
            symbol,
            signal_data['signal'],
            signal_data['confidence'],
            current_price,
            signal_data['reasoning'],
            json.dumps(signal_data.get('full_response', {}))
        ))
        
        conn.commit()
        conn.close()
        
        self.daily_trades += 1
        
        print(f"✅ EXECUTED: {symbol} {signal_data['signal']} @ ${current_price:.2f}")
    
    async def manage_positions(self):
        """Manage open positions with stops and targets"""
        
        import yfinance as yf
        
        for symbol, position in list(self.positions.items()):
            ticker = yf.Ticker(symbol)
            current_price = ticker.history(period='1d')['Close'].iloc[-1]
            
            entry_price = position['entry_price']
            
            # Calculate P&L based on signal type
            if position['signal'] == 'CALLS':
                pnl = (current_price - entry_price) / entry_price
            else:  # PUTS
                pnl = (entry_price - current_price) / entry_price
            
            # Time in trade
            time_held = (datetime.now() - position['entry_time']).seconds / 60
            
            # Exit conditions
            should_exit = False
            exit_reason = ""
            
            if pnl <= self.thresholds['stop_loss']:
                should_exit = True
                exit_reason = "Stop loss"
            elif pnl >= self.thresholds['take_profit']:
                should_exit = True
                exit_reason = "Profit target"
            elif time_held >= self.thresholds['max_hold_minutes']:
                should_exit = True
                exit_reason = "Time stop"
            
            if should_exit:
                # Close position
                self._close_position(symbol, current_price, pnl, exit_reason)
                del self.positions[symbol]
            else:
                print(f"  {symbol}: {pnl:+.1%} P&L, {time_held:.0f} min held")
    
    def _close_position(self, symbol: str, exit_price: float, pnl: float, reason: str):
        """Record position close"""
        
        conn = sqlite3.connect(self.memory_db)
        cursor = conn.cursor()
        
        cursor.execute('''
            UPDATE trades 
            SET exit_price = ?, pnl = ?
            WHERE symbol = ? AND exit_price IS NULL
            ORDER BY timestamp DESC
            LIMIT 1
        ''', (exit_price, pnl, symbol))
        
        conn.commit()
        conn.close()
        
        emoji = "🟢" if pnl > 0 else "🔴"
        print(f"{emoji} CLOSED: {symbol} {pnl:+.1%} - {reason}")
    
    async def learn_from_trades(self):
        """Analyze trades and adjust thresholds"""
        
        conn = sqlite3.connect(self.memory_db)
        
        # Get recent trades
        df = pd.read_sql_query('''
            SELECT * FROM trades 
            WHERE exit_price IS NOT NULL 
            AND timestamp > datetime('now', '-7 days')
        ''', conn)
        
        if len(df) > 5:
            win_rate = len(df[df['pnl'] > 0]) / len(df)
            avg_win = df[df['pnl'] > 0]['pnl'].mean() if len(df[df['pnl'] > 0]) > 0 else 0
            avg_loss = df[df['pnl'] < 0]['pnl'].mean() if len(df[df['pnl'] < 0]) > 0 else 0
            
            print(f"📚 Learning: Win rate {win_rate:.0%}, Avg win {avg_win:.1%}, Avg loss {avg_loss:.1%}")
            
            # Adjust thresholds based on performance
            if win_rate < 0.5:
                # Increase selectivity
                self.thresholds['entry_confidence'] = min(85, self.thresholds['entry_confidence'] + 5)
                print(f"  → Raising confidence threshold to {self.thresholds['entry_confidence']}")
            elif win_rate > 0.7 and avg_win > abs(avg_loss) * 1.5:
                # Can be less selective
                self.thresholds['entry_confidence'] = max(70, self.thresholds['entry_confidence'] - 2)
                print(f"  → Lowering confidence threshold to {self.thresholds['entry_confidence']}")
        
        conn.close()
    
    async def compare_strategies(self, symbol: str):
        """Use your strategy comparison endpoint"""
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.api_url}/strategy_comparison/",
                params={"symbol": symbol, "timeframe": "5m"}
            )
            return response.json()
    
    
    
    
    
    
    def get_stats(self) -> Dict:
        """Get performance statistics"""
        conn = sqlite3.connect(self.memory_db)
        
        df = pd.read_sql_query('''
            SELECT * FROM trades WHERE exit_price IS NOT NULL
        ''', conn)
        
        conn.close()
        
        if df.empty:
            return {'total_trades': 0}
        
        total_pnl = df['pnl'].sum()
        
        return {
            'total_trades': len(df),
            'open_positions': len(self.positions),
            'win_rate': len(df[df['pnl'] > 0]) / len(df) * 100,
            'total_pnl': total_pnl * 100,  # As percentage
            'best_trade': df['pnl'].max() * 100,
            'worst_trade': df['pnl'].min() * 100,
            'avg_pnl': df['pnl'].mean() * 100
        }

# Standalone runner
async def run_agent_with_api():
    """Run the agent that uses your API"""
    
    print("Starting agent that uses Enhanced API...")
    print("Make sure your API is running:")
    print("  python -m uvicorn trade_analysis.enhanced_api:app --host 0.0.0.0 --port 8000")
    print("")
    
    agent = TradingAgent(api_url="http://localhost:8000")
    
    # Start agent
    agent_task = asyncio.create_task(agent.run())
    
    # Print stats periodically
    while True:
        await asyncio.sleep(900)  # Every 5 minutes
        stats = agent.get_stats()
        print(f"\n📊 Agent Stats: {json.dumps(stats, indent=2)}\n")

# Utility function to analyze agent's learning
def analyze_agent_performance():
    """Analyze what the agent has learned"""
    
    conn = sqlite3.connect("agent_memory.db")
    
    # Get all trades
    df = pd.read_sql_query("SELECT * FROM trades", conn)
    
    if df.empty:
        print("No trades yet")
        return
    
    # Analyze by signal type
    print("\n=== Performance by Signal Type ===")
    for signal in df['signal'].unique():
        signal_df = df[df['signal'] == signal]
        closed = signal_df[signal_df['exit_price'].notna()]
        
        if len(closed) > 0:
            win_rate = len(closed[closed['pnl'] > 0]) / len(closed) * 100
            avg_pnl = closed['pnl'].mean() * 100
            
            print(f"{signal}: {len(closed)} trades, {win_rate:.0f}% win rate, {avg_pnl:+.1f}% avg")
    
    # Analyze by confidence level
    print("\n=== Performance by Confidence ===")
    df['conf_bucket'] = (df['confidence'] // 10) * 10
    
    for conf in sorted(df['conf_bucket'].unique()):
        conf_df = df[df['conf_bucket'] == conf]
        closed = conf_df[conf_df['exit_price'].notna()]
        
        if len(closed) > 0:
            win_rate = len(closed[closed['pnl'] > 0]) / len(closed) * 100
            print(f"{conf}-{conf+10}% confidence: {win_rate:.0f}% win rate")
    
    # Best and worst trades
    print("\n=== Notable Trades ===")
    if len(df[df['exit_price'].notna()]) > 0:
        best = df.loc[df['pnl'].idxmax()]
        worst = df.loc[df['pnl'].idxmin()]
        
        print(f"Best: {best['symbol']} {best['signal']} +{best['pnl']*100:.1f}%")
        print(f"Worst: {worst['symbol']} {worst['signal']} {worst['pnl']*100:.1f}%")
    
    conn.close()

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "analyze":
        analyze_agent_performance()
    else:
        asyncio.run(run_agent_with_api())