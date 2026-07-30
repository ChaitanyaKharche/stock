# trade_analysis/momentum_trading_engine.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import asyncio
import warnings
warnings.filterwarnings("ignore")

class MomentumTradingEngine:
    """
    High-frequency momentum trading engine for options
    Optimized for 2-5 minute scalping and intraday momentum
    """
    
    def __init__(self):
        self.momentum_thresholds = {
            # Scalping thresholds (2-5 minutes)
            'scalp': {
                'entry_momentum': 0.25,
                'exit_momentum': 0.10,
                'volume_threshold': 1.5,  # 150% of average volume
                'volatility_min': 0.15,   # Minimum IV for entry
                'max_hold_minutes': 5
            },
            
            # Intraday momentum (15-60 minutes)
            'intraday': {
                'entry_momentum': 0.35,
                'exit_momentum': 0.15,
                'volume_threshold': 2.0,   # 200% of average volume
                'volatility_min': 0.20,
                'max_hold_minutes': 60
            },
            
            # Gap trading (overnight to opening)
            'gap': {
                'gap_threshold': 0.02,     # 2% gap minimum
                'volume_confirmation': 3.0, # 300% volume spike
                'momentum_sustainability': 0.40,
                'max_hold_minutes': 30
            }
        }
        
        # Options-specific parameters
        self.options_params = {
            'min_open_interest': 100,      # Minimum OI for liquidity
            'max_bid_ask_spread': 0.15,    # Maximum 15% spread
            'iv_percentile_min': 30,       # Min IV percentile
            'iv_percentile_max': 80,       # Max IV percentile
            'delta_range': (0.15, 0.85),   # Delta range for entries
            'theta_max': -0.05,            # Maximum theta decay per day
            'gamma_min': 0.01              # Minimum gamma for momentum
        }
    
    def analyze_momentum_setup(self, ohlcv_data: Dict, sentiment_data: Dict, 
                             alternative_data: Dict) -> Dict:
        """
        Analyze current momentum setup across multiple timeframes
        """
        setups = {}
        
        for timeframe, df in ohlcv_data.items():
            if df.empty:
                continue
                
            # Calculate momentum indicators
            momentum_indicators = self._calculate_momentum_indicators(df)
            
            # Assess volume profile
            volume_profile = self._analyze_volume_profile(df)
            
            # Check volatility conditions
            volatility_conditions = self._check_volatility_conditions(df)
            
            # Generate momentum score
            momentum_score = self._calculate_momentum_score(
                momentum_indicators, volume_profile, volatility_conditions, sentiment_data
            )
            
            setups[timeframe] = {
                'momentum_score': momentum_score,
                'indicators': momentum_indicators,
                'volume_profile': volume_profile,
                'volatility': volatility_conditions,
                'trade_signal': self._generate_trade_signal(momentum_score, timeframe)
            }
        
        # Generate master signal from all timeframes
        master_signal = self._generate_master_signal(setups, sentiment_data, alternative_data)
        
        return {
            'timeframe_setups': setups,
            'master_signal': master_signal,
            'confidence': self._calculate_signal_confidence(setups, sentiment_data)
        }
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> Dict:
        """Calculate comprehensive momentum indicators"""
        if len(df) < 20:
            return self._default_momentum_indicators()
        
        close = df['Close']
        high = df['High']
        low = df['Low']
        volume = df['Volume']
        
        # Price momentum
        price_change_1 = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
        price_change_5 = (close.iloc[-1] - close.iloc[-6]) / close.iloc[-6] if len(close) > 5 else 0
        price_change_10 = (close.iloc[-1] - close.iloc[-11]) / close.iloc[-11] if len(close) > 10 else 0
        
        # Velocity indicators
        velocity_1min = abs(price_change_1) * 60  # Annualized 1-minute velocity
        velocity_5min = abs(price_change_5) * 12  # Annualized 5-minute velocity
        
        # Momentum strength
        momentum_strength = (velocity_1min + velocity_5min) / 2
        
        # Directional momentum
        up_moves = sum(1 for i in range(1, min(10, len(close))) if close.iloc[-i] > close.iloc[-i-1])
        down_moves = sum(1 for i in range(1, min(10, len(close))) if close.iloc[-i] < close.iloc[-i-1])
        directional_bias = (up_moves - down_moves) / max(up_moves + down_moves, 1)
        
        # Acceleration
        if len(close) >= 3:
            recent_momentum = (close.iloc[-1] - close.iloc[-2]) / close.iloc[-2]
            previous_momentum = (close.iloc[-2] - close.iloc[-3]) / close.iloc[-3]
            acceleration = recent_momentum - previous_momentum
        else:
            acceleration = 0
        
        # Range expansion
        recent_range = high.iloc[-1] - low.iloc[-1]
        avg_range = (high - low).tail(10).mean()
        range_expansion = recent_range / avg_range if avg_range > 0 else 1
        
        return {
            'price_change_1min': price_change_1,
            'price_change_5min': price_change_5,
            'velocity_1min': velocity_1min,
            'velocity_5min': velocity_5min,
            'momentum_strength': momentum_strength,
            'directional_bias': directional_bias,
            'acceleration': acceleration,
            'range_expansion': range_expansion
        }
    
    def _analyze_volume_profile(self, df: pd.DataFrame) -> Dict:
        """Analyze volume patterns for momentum confirmation"""
        if len(df) < 10:
            return {'volume_ratio': 1.0, 'volume_trend': 'neutral', 'volume_spike': False}
        
        volume = df['Volume']
        
        # Current vs average volume
        current_volume = volume.iloc[-1]
        avg_volume = volume.tail(20).mean()
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1
        
        # Volume trend
        recent_volume = volume.tail(5).mean()
        previous_volume = volume.tail(10).head(5).mean()
        
        if recent_volume > previous_volume * 1.2:
            volume_trend = 'increasing'
        elif recent_volume < previous_volume * 0.8:
            volume_trend = 'decreasing'
        else:
            volume_trend = 'neutral'
        
        # Volume spike detection
        volume_spike = volume_ratio > 2.0
        
        # Volume-price correlation
        price_changes = df['Close'].pct_change().tail(10)
        volume_changes = volume.pct_change().tail(10)
        
        correlation = price_changes.corr(volume_changes)
        if pd.isna(correlation):
            correlation = 0
        
        return {
            'volume_ratio': volume_ratio,
            'volume_trend': volume_trend,
            'volume_spike': volume_spike,
            'volume_price_correlation': correlation
        }
    
    def _check_volatility_conditions(self, df: pd.DataFrame) -> Dict:
        """Check volatility conditions for momentum trading"""
        if len(df) < 20:
            return {'current_volatility': 0.2, 'volatility_regime': 'normal', 'volatility_trend': 'stable'}
        
        close = df['Close']
        
        # Calculate realized volatility
        returns = close.pct_change().dropna()
        current_vol = returns.tail(10).std() * np.sqrt(252)  # Annualized
        
        # Historical volatility comparison
        hist_vol = returns.std() * np.sqrt(252)
        vol_ratio = current_vol / hist_vol if hist_vol > 0 else 1
        
        # Volatility regime
        if vol_ratio > 1.5:
            vol_regime = 'high'
        elif vol_ratio < 0.7:
            vol_regime = 'low'
        else:
            vol_regime = 'normal'
        
        # Volatility trend
        recent_vol = returns.tail(5).std() * np.sqrt(252)
        previous_vol = returns.tail(15).head(10).std() * np.sqrt(252)
        
        if recent_vol > previous_vol * 1.2:
            vol_trend = 'increasing'
        elif recent_vol < previous_vol * 0.8:
            vol_trend = 'decreasing'
        else:
            vol_trend = 'stable'
        
        return {
            'current_volatility': current_vol,
            'volatility_ratio': vol_ratio,
            'volatility_regime': vol_regime,
            'volatility_trend': vol_trend
        }
    
    def _calculate_momentum_score(self, momentum_indicators: Dict, volume_profile: Dict, 
                                volatility_conditions: Dict, sentiment_data: Dict) -> float:
        """Calculate composite momentum score"""
        
        score = 0.0
        
        # Price momentum component (40% weight)
        momentum_strength = momentum_indicators.get('momentum_strength', 0)
        directional_bias = momentum_indicators.get('directional_bias', 0)
        acceleration = momentum_indicators.get('acceleration', 0)
        
        price_score = (momentum_strength * 0.5 + 
                      abs(directional_bias) * 0.3 + 
                      abs(acceleration) * 0.2)
        score += price_score * 0.4
        
        # Volume confirmation (25% weight)
        volume_ratio = volume_profile.get('volume_ratio', 1.0)
        volume_spike = volume_profile.get('volume_spike', False)
        volume_correlation = abs(volume_profile.get('volume_price_correlation', 0))
        
        volume_score = (min(volume_ratio / 2, 1.0) * 0.5 +
                       (1.0 if volume_spike else 0.0) * 0.3 +
                       volume_correlation * 0.2)
        score += volume_score * 0.25
        
        # Volatility conditions (20% weight)
        vol_ratio = volatility_conditions.get('volatility_ratio', 1.0)
        vol_trend = volatility_conditions.get('volatility_trend', 'stable')
        
        vol_score = (min(vol_ratio, 2.0) / 2.0 * 0.7 +
                    (0.3 if vol_trend == 'increasing' else 0.0) * 0.3)
        score += vol_score * 0.20
        
        # Sentiment boost (15% weight) 
        sentiment_composite = sentiment_data.get('composite_score', 0)
        sentiment_confidence = sentiment_data.get('confidence', 'LOW')
        
        sentiment_multiplier = 1.2 if sentiment_confidence == 'HIGH' else 1.0
        sentiment_score = abs(sentiment_composite) * sentiment_multiplier
        score += sentiment_score * 0.15
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_trade_signal(self, momentum_score: float, timeframe: str) -> Dict:
        """Generate trade signal based on momentum score and timeframe"""
        
        if timeframe == '15m':
            strategy_type = 'scalp'
        elif timeframe == 'hourly':
            strategy_type = 'intraday'
        else:
            strategy_type = 'scalp'
        
        thresholds = self.momentum_thresholds[strategy_type]
        
        if momentum_score > thresholds['entry_momentum']:
            signal_strength = 'STRONG' if momentum_score > 0.6 else 'MODERATE'
            signal = f"{signal_strength}_MOMENTUM"
            conviction = min(momentum_score, 1.0)
            
            # Position sizing based on conviction
            if conviction > 0.8:
                position_size = 0.5  # 50% of available capital
            elif conviction > 0.6:
                position_size = 0.3  # 30% of available capital
            else:
                position_size = 0.2  # 20% of available capital
                
        else:
            signal = 'NO_SIGNAL'
            conviction = 0.0
            position_size = 0.0
        
        return {
            'signal': signal,
            'conviction': conviction,
            'position_size': position_size,
            'strategy_type': strategy_type,
            'hold_time_minutes': thresholds['max_hold_minutes']
        }
    
    def _generate_master_signal(self, setups: Dict, sentiment_data: Dict, 
                              alternative_data: Dict) -> Dict:
        """Generate master trading signal from all timeframes"""
        
        if not setups:
            return {'signal': 'NO_SIGNAL', 'conviction': 0, 'strategy': 'WAIT'}
        
        # Weight timeframes by importance for momentum trading
        timeframe_weights = {
            '15m': 0.5,    # Primary for scalping
            'hourly': 0.3, # Secondary confirmation
            'daily': 0.2   # Trend context
        }
        
        weighted_momentum = 0
        total_weight = 0
        
        for timeframe, setup in setups.items():
            weight = timeframe_weights.get(timeframe, 0.1)
            momentum_score = setup.get('momentum_score', 0)
            
            weighted_momentum += momentum_score * weight
            total_weight += weight
        
        avg_momentum = weighted_momentum / total_weight if total_weight > 0 else 0
        
        # Apply sentiment and alternative data modifiers
        sentiment_boost = abs(sentiment_data.get('composite_score', 0)) * 0.2
        vix_penalty = max(0, (alternative_data.get('vix_level', 20) - 25) * 0.01)
        
        final_momentum = avg_momentum + sentiment_boost - vix_penalty
        
        # Generate final signal
        if final_momentum > 0.6:
            return {
                'signal': 'STRONG_MOMENTUM',
                'conviction': min(final_momentum, 1.0),
                'strategy': 'AGGRESSIVE_SCALP',
                'timeframe': '5m',
                'expected_hold': '2-5 minutes'
            }
        elif final_momentum > 0.4:
            return {
                'signal': 'MODERATE_MOMENTUM', 
                'conviction': final_momentum,
                'strategy': 'STANDARD_MOMENTUM',
                'timeframe': '15m',
                'expected_hold': '10-30 minutes'
            }
        elif final_momentum > 0.25:
            return {
                'signal': 'WEAK_MOMENTUM',
                'conviction': final_momentum,
                'strategy': 'CAUTIOUS_ENTRY',
                'timeframe': '30m',
                'expected_hold': '30-60 minutes'
            }
        else:
            return {
                'signal': 'NO_MOMENTUM',
                'conviction': 0,
                'strategy': 'WAIT',
                'timeframe': 'N/A',
                'expected_hold': 'N/A'
            }
    
    def _calculate_signal_confidence(self, setups: Dict, sentiment_data: Dict) -> str:
        """Calculate overall signal confidence"""
        
        # Check timeframe alignment
        signals = [setup.get('trade_signal', {}).get('signal', 'NO_SIGNAL') 
                  for setup in setups.values()]
        
        momentum_signals = [s for s in signals if 'MOMENTUM' in s]
        
        # High confidence: multiple timeframes agree + high sentiment confidence
        if (len(momentum_signals) >= 2 and 
            sentiment_data.get('confidence') == 'HIGH'):
            return 'HIGH'
        
        # Medium confidence: some agreement or high momentum on one timeframe
        elif (len(momentum_signals) >= 1 and 
              sentiment_data.get('confidence') in ['HIGH', 'MEDIUM']):
            return 'MEDIUM'
        
        else:
            return 'LOW'
    
    def generate_options_strategy(self, master_signal: Dict, current_price: float, 
                                implied_volatility: float) -> Dict:
        """Generate specific options strategy based on momentum signal"""
        
        signal = master_signal.get('signal', 'NO_SIGNAL')
        conviction = master_signal.get('conviction', 0)
        strategy_type = master_signal.get('strategy', 'WAIT')
        
        if signal == 'NO_MOMENTUM':
            return {'strategy': 'WAIT', 'contracts': [], 'risk_reward': None}
        
        # Strategy selection based on momentum and volatility
        if strategy_type == 'AGGRESSIVE_SCALP':
            return self._generate_scalp_strategy(current_price, implied_volatility, conviction)
        elif strategy_type == 'STANDARD_MOMENTUM':
            return self._generate_momentum_strategy(current_price, implied_volatility, conviction)
        else:
            return self._generate_conservative_strategy(current_price, implied_volatility, conviction)
    
    def _generate_scalp_strategy(self, current_price: float, iv: float, conviction: float) -> Dict:
        """Generate high-frequency scalping strategy"""
        
        # Use ATM or slightly OTM options for maximum gamma
        strike_offset = 0.005 * current_price  # 0.5% OTM
        call_strike = current_price + strike_offset
        put_strike = current_price - strike_offset
        
        # Position sizing based on conviction
        base_contracts = int(conviction * 10)  # Scale contracts with conviction
        
        return {
            'strategy': 'LONG_STRADDLE_SCALP',
            'contracts': [
                {
                    'type': 'CALL',
                    'strike': call_strike,
                    'quantity': base_contracts,
                    'dte': 1,  # Same day expiration for scalping
                    'target_profit': 0.25,  # 25% profit target
                    'stop_loss': 0.15       # 15% stop loss
                },
                {
                    'type': 'PUT',
                    'strike': put_strike,
                    'quantity': base_contracts,
                    'dte': 1,
                    'target_profit': 0.25,
                    'stop_loss': 0.15
                }
            ],
            'max_hold_time': '5 minutes',
            'risk_reward': 1.67,  # 25% target / 15% stop
            'iv_requirement': 'IV > 25th percentile'
        }
    
    def _generate_momentum_strategy(self, current_price: float, iv: float, conviction: float) -> Dict:
        """Generate standard momentum strategy"""
        
        # Use ITM options for momentum plays
        strike_offset = 0.02 * current_price  # 2% ITM
        call_strike = current_price - strike_offset
        put_strike = current_price + strike_offset
        
        base_contracts = int(conviction * 5)
        
        return {
            'strategy': 'DIRECTIONAL_MOMENTUM',
            'contracts': [
                {
                    'type': 'CALL',
                    'strike': call_strike,
                    'quantity': base_contracts,
                    'dte': 2,  # 2 DTE for momentum
                    'target_profit': 0.50,  # 50% profit target
                    'stop_loss': 0.25       # 25% stop loss
                }
            ],
            'max_hold_time': '30 minutes',
            'risk_reward': 2.0,
            'iv_requirement': 'IV 30th-70th percentile'
        }
    
    def _generate_conservative_strategy(self, current_price: float, iv: float, conviction: float) -> Dict:
        """Generate conservative momentum strategy"""
        
        # Use spreads to limit risk
        strike_width = 0.01 * current_price  # 1% spread width
        
        return {
            'strategy': 'BULL_CALL_SPREAD',
            'contracts': [
                {
                    'type': 'CALL_SPREAD',
                    'long_strike': current_price,
                    'short_strike': current_price + strike_width,
                    'quantity': int(conviction * 3),
                    'dte': 7,  # 1 week for conservative plays
                    'target_profit': 0.30,
                    'stop_loss': 0.20
                }
            ],
            'max_hold_time': '2 hours',
            'risk_reward': 1.5,
            'iv_requirement': 'Any IV level'
        }
    
    def _default_momentum_indicators(self) -> Dict:
        """Default momentum indicators when insufficient data"""
        return {
            'price_change_1min': 0,
            'price_change_5min': 0,
            'velocity_1min': 0,
            'velocity_5min': 0,
            'momentum_strength': 0,
            'directional_bias': 0,
            'acceleration': 0,
            'range_expansion': 1.0
        }

# Integration class for existing system
class IntegratedMomentumEngine:
    """
    Integration layer between momentum engine and existing trading system
    """
    
    def __init__(self):
        self.momentum_engine = MomentumTradingEngine()
        
    def generate_enhanced_signal(self, market_data: Dict, sentiment_data: Dict, 
                                alternative_data: Dict) -> Dict:
        """
        Generate enhanced trading signal using momentum analysis
        """
        
        # Extract OHLCV data
        ohlcv_data = {}
        for timeframe in ['15m', 'hourly', 'daily']:
            if timeframe in market_data:
                ohlcv_data[timeframe] = market_data[timeframe]
        
        # Run momentum analysis
        momentum_analysis = self.momentum_engine.analyze_momentum_setup(
            ohlcv_data, sentiment_data, alternative_data
        )
        
        # Generate options strategy
        current_price = self._get_current_price(ohlcv_data)
        implied_vol = alternative_data.get('iv_rank', 50) / 100.0
        
        options_strategy = self.momentum_engine.generate_options_strategy(
            momentum_analysis['master_signal'], current_price, implied_vol
        )
        
        # Convert to format compatible with existing system
        master_signal = momentum_analysis['master_signal']
        
        return {
            'signal': self._convert_signal_format(master_signal['signal']),
            'confidence': int(master_signal['conviction'] * 100),
            'reasoning': self._generate_reasoning(momentum_analysis),
            'position_size': master_signal['conviction'] * 0.5,  # Max 50% allocation
            'momentum_analysis': momentum_analysis,
            'options_strategy': options_strategy,
            'timeframe_recommendation': master_signal.get('timeframe', '15m'),
            'expected_hold_time': master_signal.get('expected_hold', 'Unknown')
        }
    
    def _get_current_price(self, ohlcv_data: Dict) -> float:
        """Extract current price from market data"""
        for timeframe in ['15m', 'hourly', 'daily']:
            if timeframe in ohlcv_data and not ohlcv_data[timeframe].empty:
                return float(ohlcv_data[timeframe]['Close'].iloc[-1])
        return 400.0  # Default for QQQ
    
    def _convert_signal_format(self, momentum_signal: str) -> str:
        """Convert momentum signal to existing system format"""
        if 'MOMENTUM' in momentum_signal:
            return 'CALLS'  # Momentum typically bullish for options
        else:
            return 'HOLD'
    
    def _generate_reasoning(self, momentum_analysis: Dict) -> str:
        """Generate human-readable reasoning"""
        master_signal = momentum_analysis['master_signal']
        confidence = momentum_analysis['confidence']
        
        signal_type = master_signal['signal']
        strategy = master_signal['strategy']
        conviction = master_signal['conviction']
        
        reasoning_parts = []
        
        if signal_type != 'NO_MOMENTUM':
            reasoning_parts.append(f"{signal_type.replace('_', ' ').title()} detected")
            reasoning_parts.append(f"Strategy: {strategy.replace('_', ' ').title()}")
            reasoning_parts.append(f"Conviction: {conviction:.0%}")
            reasoning_parts.append(f"Confidence: {confidence}")
        else:
            reasoning_parts.append("No clear momentum detected")
            reasoning_parts.append("Waiting for better setup")
        
        return ". ".join(reasoning_parts) + "."