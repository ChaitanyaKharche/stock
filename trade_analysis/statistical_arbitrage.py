# trade_analysis/statistical_arbitrage.py

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller
from sklearn.linear_model import LinearRegression
from typing import Dict, List, Tuple, Optional

class StatisticalArbitrage:
    """
    Statistical arbitrage engine implementing pairs trading and cointegration
    Based on techniques used by Renaissance Technologies and Two Sigma
    """
    
    def __init__(self):
        self.lookback_period = 60  # days for cointegration test
        self.z_score_threshold = 2.0
        self.half_life_threshold = 20  # Maximum half-life in periods
        
    def find_cointegrated_pairs(self, price_data: pd.DataFrame, 
                               p_value_threshold: float = 0.05) -> List[Tuple[str, str, float]]:
        """
        Find cointegrated pairs using Engle-Granger test
        Core of statistical arbitrage strategies
        """
        pairs = []
        columns = price_data.columns
        
        for i in range(len(columns)):
            for j in range(i + 1, len(columns)):
                series1 = price_data[columns[i]].dropna()
                series2 = price_data[columns[j]].dropna()
                
                # Ensure same length
                min_len = min(len(series1), len(series2))
                if min_len < self.lookback_period:
                    continue
                    
                series1 = series1.iloc[-min_len:]
                series2 = series2.iloc[-min_len:]
                
                # Test for cointegration
                score, p_value, _ = coint(series1, series2)
                
                if p_value < p_value_threshold:
                    pairs.append((columns[i], columns[j], p_value))
        
        return sorted(pairs, key=lambda x: x[2])
    
    def calculate_spread(self, series1: pd.Series, series2: pd.Series) -> Tuple[pd.Series, float]:
        """
        Calculate spread using hedge ratio from linear regression
        """
        # Calculate hedge ratio
        X = series1.values.reshape(-1, 1)
        y = series2.values
        
        model = LinearRegression()
        model.fit(X, y)
        hedge_ratio = model.coef_[0]
        
        # Calculate spread
        spread = series2 - hedge_ratio * series1
        
        return spread, hedge_ratio
    
    def calculate_half_life(self, spread: pd.Series) -> float:
        """
        Calculate mean reversion half-life using Ornstein-Uhlenbeck process
        """
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Remove first element to align
        spread_lag = spread_lag.iloc[1:]
        
        # Regression: dy = -theta * (y - mu) * dt + sigma * dW
        X = spread_lag.values.reshape(-1, 1)
        y = spread_diff.values
        
        model = LinearRegression()
        model.fit(X, y)
        
        theta = -model.coef_[0]
        half_life = np.log(2) / theta if theta > 0 else np.inf
        
        return half_life
    
    def generate_pairs_signal(self, symbol1: str, symbol2: str, 
                             price_data: pd.DataFrame) -> Dict:
        """
        Generate trading signal for a pair
        """
        if symbol1 not in price_data.columns or symbol2 not in price_data.columns:
            return {'signal': 'NO_DATA', 'confidence': 0}
        
        series1 = price_data[symbol1].dropna()
        series2 = price_data[symbol2].dropna()
        
        # Calculate spread and metrics
        spread, hedge_ratio = self.calculate_spread(series1, series2)
        half_life = self.calculate_half_life(spread)
        
        # Check if half-life is reasonable
        if half_life > self.half_life_threshold or half_life <= 0:
            return {
                'signal': 'NO_SIGNAL',
                'confidence': 0,
                'half_life': half_life
            }
        
        # Calculate z-score
        spread_mean = spread.mean()
        spread_std = spread.std()
        current_z_score = (spread.iloc[-1] - spread_mean) / spread_std
        
        # Generate signal
        signal = 'NEUTRAL'
        confidence = 0
        
        if current_z_score > self.z_score_threshold:
            signal = 'SHORT_SPREAD'  # Sell symbol2, buy symbol1
            confidence = min((current_z_score - 2) * 30 + 60, 90)
        elif current_z_score < -self.z_score_threshold:
            signal = 'LONG_SPREAD'  # Buy symbol2, sell symbol1
            confidence = min((abs(current_z_score) - 2) * 30 + 60, 90)
        
        return {
            'signal': signal,
            'confidence': confidence,
            'z_score': current_z_score,
            'half_life': half_life,
            'hedge_ratio': hedge_ratio,
            'spread_mean': spread_mean,
            'spread_std': spread_std
        }
    
    def hidden_markov_regime_detection(self, returns: pd.Series, n_states: int = 3) -> Dict:
        """
        Simplified HMM for regime detection (Renaissance Tech style)
        Identifies market regimes: trending, mean-reverting, volatile
        """
        if len(returns) < 100:
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        # Calculate rolling statistics
        vol_20 = returns.rolling(20).std()
        vol_60 = returns.rolling(60).std()
        mean_20 = returns.rolling(20).mean()
        
        # Simple regime classification
        current_vol = vol_20.iloc[-1]
        long_vol = vol_60.iloc[-1]
        current_mean = mean_20.iloc[-1]
        
        if pd.isna(current_vol) or pd.isna(long_vol):
            return {'regime': 'UNKNOWN', 'confidence': 0}
        
        # Classify regime
        vol_ratio = current_vol / long_vol if long_vol > 0 else 1
        
        if vol_ratio > 1.5:
            regime = 'VOLATILE'
            confidence = min(vol_ratio * 30, 90)
        elif abs(current_mean) > current_vol * 2:
            regime = 'TRENDING'
            confidence = 80
        else:
            regime = 'MEAN_REVERTING'
            confidence = 70
        
        return {
            'regime': regime,
            'confidence': confidence,
            'volatility_ratio': vol_ratio,
            'trend_strength': abs(current_mean) / current_vol if current_vol > 0 else 0
        }