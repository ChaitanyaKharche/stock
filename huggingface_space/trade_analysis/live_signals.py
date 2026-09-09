# trade_analysis/live_signals.py
"""
Live trading signal generator with prop-style data sources
Combines public alternatives to proprietary data
"""

import asyncio
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import httpx
from typing import Dict, List

class LiveTradingSignalGenerator:
    """
    Generates live trading signals using advanced public data
    Mimics prop trading strategies with accessible data
    """
    
    def __init__(self):
        self.signal_threshold = 0.7  # 70% confidence minimum
        self.position_limits = {
            'scalp': 0.25,    # 25% max position for scalps
            'momentum': 0.5,  # 50% for momentum plays
            'swing': 0.3      # 30% for swings
        }
    
    async def generate_live_signal(self, symbol: str, timeframe: str = '5m') -> Dict:
        """
        Main function - generates actionable trading signals
        """
        
        # 1. Get unusual options activity (smart money indicator)
        options_signal = await self.analyze_options_flow(symbol)
        
        # 2. Get institutional flow indicators
        institutional_flow = await self.get_institutional_indicators(symbol)
        
        # 3. Get real-time momentum
        momentum = self.calculate_live_momentum(symbol, timeframe)
        
        # 4. Get market maker levels
        mm_levels = await self.get_market_maker_levels(symbol)
        
        # 5. Combine all signals
        master_signal = self.generate_master_signal(
            options_signal, institutional_flow, momentum, mm_levels
        )
        
        return master_signal
    
    async def analyze_options_flow(self, symbol: str) -> Dict:
        """
        Analyze unusual options activity - this is what prop firms watch
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Get next 3 expiration dates
            exp_dates = ticker.options[:3] if len(ticker.options) >= 3 else ticker.options
            
            unusual_activity = {
                'call_volume': 0,
                'put_volume': 0,
                'call_oi': 0,
                'put_oi': 0,
                'unusual_strikes': []
            }
            
            for exp_date in exp_dates:
                opt_chain = ticker.option_chain(exp_date)
                calls = opt_chain.calls
                puts = opt_chain.puts
                
                # Find unusual volume (volume > 2x open interest)
                unusual_calls = calls[calls['volume'] > calls['openInterest'] * 2]
                unusual_puts = puts[puts['volume'] > puts['openInterest'] * 2]
                
                # Track total unusual volume
                unusual_activity['call_volume'] += unusual_calls['volume'].sum()
                unusual_activity['put_volume'] += unusual_puts['volume'].sum()
                unusual_activity['call_oi'] += unusual_calls['openInterest'].sum()
                unusual_activity['put_oi'] += unusual_puts['openInterest'].sum()
                
                # Find specific strikes with heavy activity
                if not unusual_calls.empty:
                    top_call = unusual_calls.nlargest(1, 'volume')
                    if not top_call.empty:
                        unusual_activity['unusual_strikes'].append({
                            'type': 'CALL',
                            'strike': float(top_call['strike'].iloc[0]),
                            'volume': int(top_call['volume'].iloc[0]),
                            'expiry': exp_date
                        })
                
                if not unusual_puts.empty:
                    top_put = unusual_puts.nlargest(1, 'volume')
                    if not top_put.empty:
                        unusual_activity['unusual_strikes'].append({
                            'type': 'PUT',
                            'strike': float(top_put['strike'].iloc[0]),
                            'volume': int(top_put['volume'].iloc[0]),
                            'expiry': exp_date
                        })
            
            # Calculate put/call ratio
            total_calls = unusual_activity['call_volume']
            total_puts = unusual_activity['put_volume']
            
            if total_calls > 0:
                pc_ratio = total_puts / total_calls
            else:
                pc_ratio = 2.0  # High puts if no calls
            
            # Generate signal based on unusual activity
            if pc_ratio < 0.5 and total_calls > 10000:
                signal = 'STRONG_BULLISH'
                confidence = min(90, 60 + (total_calls / 1000))
            elif pc_ratio > 1.5 and total_puts > 10000:
                signal = 'STRONG_BEARISH'
                confidence = min(90, 60 + (total_puts / 1000))
            elif pc_ratio < 0.7:
                signal = 'BULLISH'
                confidence = 60
            elif pc_ratio > 1.3:
                signal = 'BEARISH'
                confidence = 60
            else:
                signal = 'NEUTRAL'
                confidence = 40
            
            return {
                'signal': signal,
                'confidence': confidence,
                'pc_ratio': pc_ratio,
                'unusual_strikes': unusual_activity['unusual_strikes'][:3],  # Top 3
                'call_volume': total_calls,
                'put_volume': total_puts
            }
            
        except Exception as e:
            print(f"Options flow analysis failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}
    
    async def get_institutional_indicators(self, symbol: str) -> Dict:
        """
        Get institutional flow indicators (DIX, GEX alternatives)
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            # Get institutional ownership changes
            inst_ownership = info.get('heldPercentInstitutions', 0)
            
            # Get recent insider transactions (proxy for smart money)
            # In production, you'd use SEC Edgar API for real insider data
            
            # Get 13F filing changes (quarterly institutional holdings)
            # This would need fintel.io or whalewisdom API in production
            
            # For now, use volume profile as proxy
            hist = ticker.history(period='5d', interval='1h')
            if not hist.empty:
                avg_volume = hist['Volume'].mean()
                recent_volume = hist['Volume'].iloc[-1]
                
                # Volume surge often indicates institutional activity
                volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
                
                if volume_ratio > 2:
                    inst_signal = 'ACCUMULATION'
                elif volume_ratio < 0.5:
                    inst_signal = 'DISTRIBUTION'
                else:
                    inst_signal = 'NEUTRAL'
                
                return {
                    'signal': inst_signal,
                    'volume_ratio': volume_ratio,
                    'inst_ownership': inst_ownership,
                    'confidence': min(80, 40 + volume_ratio * 10)
                }
            
            return {'signal': 'NEUTRAL', 'confidence': 40}
            
        except Exception as e:
            print(f"Institutional indicators failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}
    
    def calculate_live_momentum(self, symbol: str, timeframe: str) -> Dict:
        """
        Calculate real-time momentum indicators
        """
        try:
            ticker = yf.Ticker(symbol)
            
            # Map timeframe to yfinance interval
            interval_map = {
                '1m': '1m',
                '5m': '5m', 
                '15m': '15m',
                '1h': '60m'
            }
            
            interval = interval_map.get(timeframe, '5m')
            period = '1d' if interval in ['1m', '5m'] else '5d'
            
            df = ticker.history(period=period, interval=interval)
            
            if len(df) < 20:
                return {'signal': 'NEUTRAL', 'confidence': 0}
            
            # Calculate momentum indicators
            close = df['Close']
            volume = df['Volume']
            
            # Price momentum
            returns_5 = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) > 5 else 0
            returns_10 = (close.iloc[-1] / close.iloc[-11] - 1) if len(close) > 10 else 0
            
            # Volume confirmation
            vol_ma = volume.rolling(20).mean()
            vol_spike = volume.iloc[-1] > vol_ma.iloc[-1] * 1.5 if not vol_ma.empty else False
            
            # RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50
            
            # Generate momentum signal
            if returns_5 > 0.01 and returns_10 > 0.015 and vol_spike and current_rsi > 55:
                momentum_signal = 'STRONG_UP'
                confidence = 80
            elif returns_5 < -0.01 and returns_10 < -0.015 and vol_spike and current_rsi < 45:
                momentum_signal = 'STRONG_DOWN'
                confidence = 80
            elif returns_5 > 0.005 and current_rsi > 50:
                momentum_signal = 'UP'
                confidence = 60
            elif returns_5 < -0.005 and current_rsi < 50:
                momentum_signal = 'DOWN'
                confidence = 60
            else:
                momentum_signal = 'NEUTRAL'
                confidence = 40
            
            return {
                'signal': momentum_signal,
                'confidence': confidence,
                'returns_5bar': returns_5,
                'returns_10bar': returns_10,
                'rsi': current_rsi,
                'volume_spike': vol_spike
            }
            
        except Exception as e:
            print(f"Momentum calculation failed: {e}")
            return {'signal': 'NEUTRAL', 'confidence': 0}
    
    async def get_market_maker_levels(self, symbol: str) -> Dict:
        """
        Calculate key levels where market makers hedge (gamma levels)
        """
        try:
            ticker = yf.Ticker(symbol)
            current_price = ticker.info.get('currentPrice', ticker.history(period='1d')['Close'].iloc[-1])
            
            # Get nearest expiration options
            if ticker.options:
                exp_date = ticker.options[0]
                opt_chain = ticker.option_chain(exp_date)
                
                all_options = pd.concat([opt_chain.calls, opt_chain.puts])
                
                # Calculate gamma exposure at each strike
                # Simplified version - real calculation needs Black-Scholes
                all_options['gamma_exposure'] = all_options['openInterest'] * all_options.get('gamma', 0.01) * 100
                
                # Find key levels
                key_levels = all_options.nlargest(5, 'openInterest')[['strike', 'openInterest']]
                
                # Determine support/resistance
                support_levels = key_levels[key_levels['strike'] < current_price]['strike'].tolist()
                resistance_levels = key_levels[key_levels['strike'] > current_price]['strike'].tolist()
                
                return {
                    'current_price': current_price,
                    'support': support_levels[:2],  # Top 2 support levels
                    'resistance': resistance_levels[:2],  # Top 2 resistance levels
                    'max_pain': float(key_levels['strike'].iloc[0]) if not key_levels.empty else current_price
                }
            
            return {'current_price': current_price, 'support': [], 'resistance': []}
            
        except Exception as e:
            print(f"Market maker levels failed: {e}")
            return {'support': [], 'resistance': []}
    
    def generate_master_signal(self, options: Dict, institutional: Dict, 
                              momentum: Dict, mm_levels: Dict) -> Dict:
        """
        Combine all signals into actionable trading signal
        """
        
        # Weight each signal source
        weights = {
            'options': 0.35,      # Unusual options = smart money
            'institutional': 0.25, # Volume/institutional flow
            'momentum': 0.30,      # Price action
            'levels': 0.10        # Support/resistance
        }
        
        # Convert signals to scores
        signal_map = {
            'STRONG_BULLISH': 1.0, 'BULLISH': 0.7, 'STRONG_UP': 1.0, 'UP': 0.7,
            'ACCUMULATION': 0.6,
            'NEUTRAL': 0.0,
            'DISTRIBUTION': -0.6,
            'STRONG_BEARISH': -1.0, 'BEARISH': -0.7, 'STRONG_DOWN': -1.0, 'DOWN': -0.7
        }
        
        options_score = signal_map.get(options['signal'], 0)
        inst_score = signal_map.get(institutional['signal'], 0)
        momentum_score = signal_map.get(momentum['signal'], 0)
        
        # Calculate weighted score
        total_score = (
            options_score * weights['options'] +
            inst_score * weights['institutional'] +
            momentum_score * weights['momentum']
        )
        
        # Calculate confidence
        weighted_confidence = (
            options.get('confidence', 0) * weights['options'] +
            institutional.get('confidence', 0) * weights['institutional'] +
            momentum.get('confidence', 0) * weights['momentum']
        )
        
        # Level analysis
        current_price = mm_levels.get('current_price', 0)
        support = mm_levels.get('support', [])
        resistance = mm_levels.get('resistance', [])
        
        # Adjust confidence based on levels
        if support and current_price > 0:
            distance_to_support = (current_price - max(support)) / current_price if support else 0.05
            if distance_to_support < 0.01:  # Very close to support
                if total_score > 0:
                    weighted_confidence += 10  # Boost bullish signals near support
        
        # Generate final signal
        if total_score > 0.5 and weighted_confidence > 70:
            action = 'BUY_CALLS'
            position_size = min(0.5, weighted_confidence / 100 * 0.6)
        elif total_score < -0.5 and weighted_confidence > 70:
            action = 'BUY_PUTS'
            position_size = min(0.5, weighted_confidence / 100 * 0.6)
        else:
            action = 'HOLD'
            position_size = 0
        
        # Build detailed reasoning
        reasoning = []
        
        if options['signal'] != 'NEUTRAL':
            reasoning.append(f"Options: {options['signal']} (P/C: {options.get('pc_ratio', 1):.2f})")
        
        if options.get('unusual_strikes'):
            for strike_info in options['unusual_strikes'][:2]:
                reasoning.append(f"Unusual {strike_info['type']} activity at ${strike_info['strike']}")
        
        if momentum['signal'] != 'NEUTRAL':
            reasoning.append(f"Momentum: {momentum['signal']} (RSI: {momentum.get('rsi', 50):.0f})")
        
        if institutional['signal'] != 'NEUTRAL':
            reasoning.append(f"Institutional: {institutional['signal']}")
        
        if support:
            reasoning.append(f"Support: ${max(support):.2f}")
        if resistance:
            reasoning.append(f"Resistance: ${min(resistance):.2f}")
        
        return {
            'action': action,
            'confidence': int(weighted_confidence),
            'position_size': position_size,
            'reasoning': ' | '.join(reasoning),
            'details': {
                'options_flow': options,
                'institutional': institutional,
                'momentum': momentum,
                'key_levels': mm_levels,
                'composite_score': total_score
            },
            'timestamp': datetime.now().isoformat()
        }

# Usage example
async def main():
    generator = LiveTradingSignalGenerator()
    
    # Generate live signal
    signal = await generator.generate_live_signal('QQQ', '5m')
    
    print(f"\n{'='*60}")
    print(f"LIVE TRADING SIGNAL - {signal['timestamp']}")
    print(f"{'='*60}")
    print(f"ACTION: {signal['action']}")
    print(f"CONFIDENCE: {signal['confidence']}%")
    print(f"POSITION SIZE: {signal['position_size']*100:.1f}% of capital")
    print(f"\nREASONING:\n{signal['reasoning']}")
    print(f"{'='*60}")
    
    # Show unusual options if found
    if signal['details']['options_flow'].get('unusual_strikes'):
        print("\nUNUSUAL OPTIONS ACTIVITY:")
        for strike in signal['details']['options_flow']['unusual_strikes']:
            print(f"  - {strike['type']} ${strike['strike']} ({strike['volume']:,} volume)")
    
    return signal

if __name__ == "__main__":
    asyncio.run(main())