# trade_analysis/prop_integration.py
"""
Main integration module for prop trading features
Connects new components with existing system
"""

from .market_microstructure import MarketMicrostructure, OrderBookSnapshot
from .statistical_arbitrage import StatisticalArbitrage
from .advanced_risk import AdvancedRiskManagement
from .execution_engine import SmartExecutionEngine
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from datetime import datetime

class PropTradingIntegration:
    """
    Integrates institutional features with existing momentum system
    """
    
    def __init__(self):
        self.microstructure = MarketMicrostructure()
        self.stat_arb = StatisticalArbitrage()
        self.risk_mgmt = AdvancedRiskManagement()
        self.execution = SmartExecutionEngine()
        
    def enhance_signal_with_microstructure(self, existing_signal: Dict, 
                                          order_book: Optional[OrderBookSnapshot] = None,
                                          recent_trades: Optional[pd.DataFrame] = None) -> Dict:
        """
        Enhance existing momentum signal with microstructure analysis
        """
        enhanced = existing_signal.copy()
        
        if order_book and recent_trades is not None:
            micro_signal = self.microstructure.microstructure_alpha(
                [order_book], recent_trades
            )
            
            # Combine signals
            if micro_signal['confidence'] > 70 and micro_signal['signal'] != 'NEUTRAL':
                # Boost confidence if microstructure agrees
                if (micro_signal['signal'] == 'BUY' and existing_signal['signal'] == 'CALLS') or \
                   (micro_signal['signal'] == 'SELL' and existing_signal['signal'] == 'PUTS'):
                    enhanced['confidence'] = min(100, existing_signal['confidence'] + 10)
                    enhanced['reasoning'] += f". Microstructure confirms: imbalance={micro_signal['imbalance']:.2f}"
                    
            # Add toxicity warning
            if micro_signal.get('toxicity', 0) > 0.7:
                enhanced['confidence'] *= 0.7
                enhanced['reasoning'] += ". WARNING: Toxic flow detected"
                
        return enhanced
    
    def add_statistical_arbitrage_signals(self, price_data: pd.DataFrame) -> Dict:
        """
        Generate statistical arbitrage opportunities
        """
        if price_data.shape[1] < 2:
            return {'pairs': [], 'regime': 'UNKNOWN'}
        
        # Find cointegrated pairs
        pairs = self.stat_arb.find_cointegrated_pairs(price_data)
        
        signals = []
        for symbol1, symbol2, p_value in pairs[:5]:  # Top 5 pairs
            signal = self.stat_arb.generate_pairs_signal(
                symbol1, symbol2, price_data
            )
            if signal['signal'] != 'NO_SIGNAL':
                signals.append({
                    'pair': f"{symbol1}/{symbol2}",
                    'signal': signal['signal'],
                    'z_score': signal['z_score'],
                    'half_life': signal['half_life'],
                    'confidence': signal['confidence']
                })
        
        # Detect market regime
        returns = price_data.pct_change().mean(axis=1)
        regime = self.stat_arb.hidden_markov_regime_detection(returns)
        
        return {
            'pairs_signals': signals,
            'market_regime': regime
        }
    
    def calculate_position_sizes(self, signals: Dict[str, Dict], 
                                capital: float, 
                                historical_returns: pd.DataFrame) -> Dict:
        """
        Professional position sizing with risk management
        """
        # Calculate portfolio metrics
        var_metrics = self.risk_mgmt.calculate_var(historical_returns)
        
        # Position sizing matrix
        position_sizes = self.risk_mgmt.position_sizing_matrix(signals, capital)
        
        # Risk parity overlay
        risk_parity_weights = self.risk_mgmt.risk_parity_allocation(historical_returns)
        
        # Combine approaches
        final_positions = {}
        for symbol in position_sizes:
            base_size = position_sizes[symbol]
            
            # Apply risk parity adjustment
            if symbol in risk_parity_weights:
                rp_adjustment = risk_parity_weights[symbol]
                final_positions[symbol] = base_size * (1 + rp_adjustment) / 2
            else:
                final_positions[symbol] = base_size
            
            # Apply VaR constraint
            if var_metrics['historical_var'] < -0.02:  # 2% VaR limit
                final_positions[symbol] *= 0.5  # Reduce all positions
        
        return {
            'positions': final_positions,
            'risk_metrics': var_metrics,
            'total_exposure': sum(final_positions.values()),
            'max_position': max(final_positions.values()) if final_positions else 0
        }
    
    def generate_execution_plan(self, symbol: str, size: float, 
                               urgency: str, market_data: pd.DataFrame) -> Dict:
        """
        Create institutional-grade execution plan
        """
        # Calculate market metrics
        vwap = self.execution.calculate_vwap(market_data)
        avg_volume = market_data['Volume'].mean()
        volatility = market_data['Close'].pct_change().std()
        
        market_conditions = {
            'volatility': volatility,
            'avg_volume': avg_volume,
            'spread': 0.01  # Approximate for now
        }
        
        # Get execution strategy
        exec_strategy = self.execution.adaptive_execution(
            urgency, market_conditions, size
        )
        
        # Generate schedule based on algorithm
        if exec_strategy['algorithm'] == 'TWAP':
            schedule = self.execution.twap_schedule(
                size, 
                exec_strategy['recommended_duration']
            )
        elif exec_strategy['algorithm'] == 'VWAP':
            schedule = self.execution.vwap_schedule(
                size,
                market_data.set_index(market_data.index.hour)['Volume'].groupby(level=0).sum()
            )
        else:
            schedule = [(datetime.now(), size)]  # Immediate execution
        
        # Iceberg order for large sizes
        if size > avg_volume * 0.01:
            iceberg = self.execution.iceberg_order(size)
        else:
            iceberg = None
        
        return {
            'symbol': symbol,
            'total_size': size,
            'algorithm': exec_strategy['algorithm'],
            'urgency': urgency,
            'schedule': schedule[:5],  # First 5 slices
            'participation_rate': exec_strategy['participation_rate'],
            'expected_impact': exec_strategy['expected_impact'],
            'vwap_target': vwap,
            'iceberg_params': iceberg
        }