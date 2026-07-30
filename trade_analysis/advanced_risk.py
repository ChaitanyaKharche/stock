# trade_analysis/advanced_risk.py

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional

class AdvancedRiskManagement:
    """
    Institutional-grade risk management system
    Implements VaR, portfolio optimization, and position limits
    """
    
    def __init__(self):
        self.var_confidence = 0.95
        self.max_leverage = 2.0  # Conservative for retail
        self.max_position_pct = 0.1  # 10% max per position
        self.correlation_window = 60
        
    def calculate_var(self, returns: pd.DataFrame, confidence: float = 0.95) -> Dict:
        """
        Calculate Value at Risk using multiple methods
        """
        results = {}
        
        # Historical VaR
        var_percentile = (1 - confidence) * 100
        historical_var = returns.quantile(var_percentile / 100)
        results['historical_var'] = float(historical_var.mean())
        
        # Parametric VaR (assumes normal distribution)
        mean_return = returns.mean()
        std_return = returns.std()
        z_score = stats.norm.ppf(1 - confidence)
        parametric_var = mean_return + z_score * std_return
        results['parametric_var'] = float(parametric_var.mean())
        
        # Monte Carlo VaR
        n_simulations = 1000
        simulated_returns = np.random.normal(
            mean_return.mean(), 
            std_return.mean(), 
            n_simulations
        )
        monte_carlo_var = np.percentile(simulated_returns, var_percentile)
        results['monte_carlo_var'] = float(monte_carlo_var)
        
        # Conditional VaR (Expected Shortfall)
        threshold = returns.quantile(var_percentile / 100)
        cvar = returns[returns <= threshold].mean()
        results['cvar'] = float(cvar.mean())
        
        return results
    
    def calculate_portfolio_metrics(self, positions: Dict[str, float], 
                                  returns: pd.DataFrame) -> Dict:
        """
        Calculate portfolio-level risk metrics
        """
        if not positions or returns.empty:
            return {'sharpe': 0, 'sortino': 0, 'max_drawdown': 0}
        
        # Filter for positions we have
        symbols = [s for s in positions.keys() if s in returns.columns]
        if not symbols:
            return {'sharpe': 0, 'sortino': 0, 'max_drawdown': 0}
        
        weights = np.array([positions[s] for s in symbols])
        weights = weights / weights.sum()  # Normalize
        
        # Portfolio returns
        portfolio_returns = (returns[symbols] * weights).sum(axis=1)
        
        # Sharpe Ratio (assuming 0 risk-free rate)
        sharpe = portfolio_returns.mean() / portfolio_returns.std() * np.sqrt(252)
        
        # Sortino Ratio (downside deviation)
        downside_returns = portfolio_returns[portfolio_returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 1
        sortino = portfolio_returns.mean() / downside_std * np.sqrt(252)
        
        # Maximum Drawdown
        cumulative = (1 + portfolio_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'max_drawdown': float(max_drawdown),
            'daily_vol': float(portfolio_returns.std()),
            'annual_vol': float(portfolio_returns.std() * np.sqrt(252))
        }
    
    def kelly_criterion(self, win_rate: float, avg_win: float, 
                       avg_loss: float) -> float:
        """
        Calculate optimal position size using Kelly Criterion
        Used by Renaissance Tech for position sizing
        """
        if avg_loss == 0:
            return 0
        
        # Kelly formula: f = (p*b - q) / b
        # where p = win rate, q = loss rate, b = win/loss ratio
        b = abs(avg_win / avg_loss)
        q = 1 - win_rate
        
        kelly = (win_rate * b - q) / b
        
        # Apply Kelly fraction (usually 0.25 for safety)
        conservative_kelly = kelly * 0.25
        
        return max(0, min(conservative_kelly, self.max_position_pct))
    
    def position_sizing_matrix(self, signals: Dict[str, Dict], 
                              capital: float) -> Dict[str, float]:
        """
        Determine position sizes based on multiple factors
        """
        position_sizes = {}
        
        for symbol, signal_data in signals.items():
            confidence = signal_data.get('confidence', 0) / 100
            volatility = signal_data.get('volatility', 0.02)
            
            # Base size from confidence
            base_size = confidence * self.max_position_pct * capital
            
            # Adjust for volatility (inverse relationship)
            target_vol = 0.02  # 2% daily vol target
            vol_adjustment = min(target_vol / volatility, 2.0) if volatility > 0 else 1
            
            # Apply limits
            position_size = base_size * vol_adjustment
            position_size = min(position_size, self.max_position_pct * capital)
            
            position_sizes[symbol] = position_size
        
        return position_sizes
    
    def risk_parity_allocation(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Risk parity allocation (equal risk contribution)
        Used by Bridgewater and other systematic funds
        """
        if returns.empty:
            return {}
        
        # Calculate volatilities
        volatilities = returns.std()
        
        # Inverse volatility weighting
        inv_vols = 1 / volatilities
        weights = inv_vols / inv_vols.sum()
        
        return weights.to_dict()