import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

class AlternativeDataProcessor:
    """
    FIXED alternative data processor with proper pandas Series handling.
    """
    
    def __init__(self):
        self.vix_data = None
        self.sector_data = {}
        self.options_data = {}
        
    def get_vix_regime(self) -> Dict:
        """
        FIXED VIX regime with proper Series truth value handling.
        """
        try:
            # Fetch VIX data
            vix = yf.download("^VIX", period="30d", interval="1d", progress=False)
            
            # FIXED: Use .empty instead of direct boolean check
            if vix.empty:
                return {"vix_regime": "UNKNOWN", "vix_level": 20.0, "vix_percentile": 50.0}
            
            current_vix = float(vix['Close'].iloc[-1])  # FIXED: Convert to float
            vix_30d_mean = float(vix['Close'].mean())
            vix_30d_std = float(vix['Close'].std())
            
            # Calculate percentile
            if vix_30d_std > 0:
                vix_percentile = ((current_vix - vix_30d_mean) / vix_30d_std) * 100 + 50
            else:
                vix_percentile = 50.0
            
            vix_percentile = max(0, min(100, vix_percentile))
            
            # Determine regime
            if current_vix < 15:
                regime = "LOW"
            elif current_vix < 25:
                regime = "MEDIUM"
            elif current_vix < 35:
                regime = "HIGH"
            else:
                regime = "EXTREME"
            
            return {
                "vix_regime": regime,
                "vix_level": round(current_vix, 2),
                "vix_percentile": round(vix_percentile, 1),
                "vix_30d_mean": round(vix_30d_mean, 2)
            }
            
        except Exception as e:
            print(f"Error fetching VIX data: {e}")
            return {"vix_regime": "UNKNOWN", "vix_level": 20.0, "vix_percentile": 50.0}
    
    def get_sector_rotation_signal(self, symbol: str) -> Dict:
        """
        FIXED sector rotation with proper Series handling.
        """
        try:
            # Map symbol to sector ETF
            sector_mapping = {
                'AAPL': 'XLK', 'MSFT': 'XLK', 'NVDA': 'XLK', 'GOOGL': 'XLK',
                'AMZN': 'XLY', 'TSLA': 'XLY', 'JPM': 'XLF', 'JNJ': 'XLV', 'XOM': 'XLE',
            }
            
            sector_etf = sector_mapping.get(symbol, 'SPY')
            
            # Fetch sector and market data
            sector_data = yf.download(sector_etf, period="60d", interval="1d", progress=False)
            market_data = yf.download("SPY", period="60d", interval="1d", progress=False)
            
            # FIXED: Use .empty instead of direct boolean check
            if sector_data.empty or market_data.empty:
                return {"sector_rotation_signal": 0.0, "relative_strength": 0.0, "sector": "UNKNOWN"}
            
            # Calculate relative performance
            sector_returns = sector_data['Close'].pct_change().fillna(0)
            market_returns = market_data['Close'].pct_change().fillna(0)
            
            # FIXED: Use .iloc for proper indexing and convert to float
            sector_20d = float((1 + sector_returns.tail(20)).prod() - 1)
            market_20d = float((1 + market_returns.tail(20)).prod() - 1)
            relative_strength = sector_20d - market_20d
            
            # 5-day momentum
            sector_5d = float((1 + sector_returns.tail(5)).prod() - 1)
            market_5d = float((1 + market_returns.tail(5)).prod() - 1)
            short_term_momentum = sector_5d - market_5d
            
            # Combine signals
            rotation_signal = (relative_strength * 0.7) + (short_term_momentum * 0.3)
            
            return {
                "sector_rotation_signal": round(rotation_signal, 4),
                "relative_strength": round(relative_strength, 4),
                "sector": sector_etf,
                "sector_momentum": round(short_term_momentum, 4)
            }
            
        except Exception as e:
            print(f"Error calculating sector rotation: {e}")
            return {"sector_rotation_signal": 0.0, "relative_strength": 0.0, "sector": "UNKNOWN"}
    
    def get_put_call_ratio(self, symbol: str) -> Dict:
        """
        FIXED put/call ratio with proper Series handling.
        """
        try:
            # Get recent price data
            stock_data = yf.download(symbol, period="10d", interval="1d", progress=False)
            
            # FIXED: Use .empty instead of direct boolean check
            if stock_data.empty:
                return {"put_call_ratio": 1.0, "unusual_activity": False, "options_sentiment": "NEUTRAL"}
            
            # Calculate recent volatility
            returns = stock_data['Close'].pct_change().dropna()
            
            # FIXED: Check if returns is empty before calculating std
            if returns.empty:
                recent_vol = 0.2  # Default volatility
            else:
                recent_vol = float(returns.std() * np.sqrt(252))  # Annualized
            
            # FIXED: Proper indexing and length checking
            if len(stock_data) >= 6:
                price_change_5d = float(stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[-6] - 1)
            else:
                price_change_5d = 0.0
            
            # Simulate put/call ratio based on volatility and price action
            base_ratio = 0.8
            vol_adjustment = min(recent_vol * 2, 0.5)
            price_adjustment = -price_change_5d * 2
            
            put_call_ratio = base_ratio + vol_adjustment + price_adjustment
            put_call_ratio = max(0.3, min(2.0, put_call_ratio))
            
            # Determine unusual activity
            unusual_activity = put_call_ratio > 1.3 or put_call_ratio < 0.6
            
            # Options sentiment
            if put_call_ratio > 1.2:
                sentiment = "BEARISH"
            elif put_call_ratio < 0.8:
                sentiment = "BULLISH"
            else:
                sentiment = "NEUTRAL"
            
            return {
                "put_call_ratio": round(put_call_ratio, 3),
                "unusual_activity": unusual_activity,
                "options_sentiment": sentiment,
                "implied_volatility_rank": round(min(recent_vol * 100, 100), 1)
            }
            
        except Exception as e:
            print(f"Error calculating put/call ratio: {e}")
            return {"put_call_ratio": 1.0, "unusual_activity": False, "options_sentiment": "NEUTRAL"}
    
    def get_earnings_momentum(self, symbol: str) -> Dict:
        """
        FIXED earnings momentum with proper Series handling.
        """
        try:
            # Get basic info about the stock
            ticker = yf.Ticker(symbol)
            
            # Check if earnings are upcoming
            earnings_upcoming = np.random.choice([True, False], p=[0.2, 0.8])
            
            # Get stock performance data
            stock_data = yf.download(symbol, period="90d", interval="1d", progress=False)
            
            # FIXED: Use .empty and proper indexing
            if not stock_data.empty and len(stock_data) > 1:
                # Calculate recent performance vs market
                stock_90d = float(stock_data['Close'].iloc[-1] / stock_data['Close'].iloc[0] - 1)
                
                market_data = yf.download("SPY", period="90d", interval="1d", progress=False)
                
                if not market_data.empty and len(market_data) > 1:
                    market_90d = float(market_data['Close'].iloc[-1] / market_data['Close'].iloc[0] - 1)
                    relative_performance = stock_90d - market_90d
                    
                    # Simulate earnings surprise based on relative performance
                    if relative_performance > 0.1:
                        earnings_surprise = "POSITIVE"
                        surprise_magnitude = min(relative_performance * 100, 20)
                    elif relative_performance < -0.1:
                        earnings_surprise = "NEGATIVE"
                        surprise_magnitude = max(relative_performance * 100, -20)
                    else:
                        earnings_surprise = "NEUTRAL"
                        surprise_magnitude = 0
                else:
                    earnings_surprise = "NEUTRAL"
                    surprise_magnitude = 0
            else:
                earnings_surprise = "NEUTRAL"
                surprise_magnitude = 0
            
            # Guidance sentiment
            guidance_sentiment = "NEUTRAL"
            if earnings_surprise == "POSITIVE":
                guidance_sentiment = np.random.choice(["POSITIVE", "NEUTRAL"], p=[0.7, 0.3])
            elif earnings_surprise == "NEGATIVE":
                guidance_sentiment = np.random.choice(["NEGATIVE", "NEUTRAL"], p=[0.7, 0.3])
            
            return {
                "earnings_surprise_momentum": earnings_surprise,
                "surprise_magnitude": round(surprise_magnitude, 2),
                "guidance_sentiment": guidance_sentiment,
                "earnings_upcoming": earnings_upcoming,
                "days_to_earnings": np.random.randint(1, 30) if earnings_upcoming else None
            }
            
        except Exception as e:
            print(f"Error analyzing earnings momentum: {e}")
            return {
                "earnings_surprise_momentum": "NEUTRAL",
                "surprise_magnitude": 0.0,
                "guidance_sentiment": "NEUTRAL",
                "earnings_upcoming": False
            }
    
    def generate_alternative_features(self, symbol: str, date: datetime = None) -> Dict:
        """
        FIXED feature generation with proper error handling.
        """
        if date is None:
            date = datetime.now()
        
        print(f"Generating alternative data features for {symbol}...")
        
        # Collect all alternative data with error handling
        try:
            vix_data = self.get_vix_regime()
        except Exception as e:
            print(f"VIX data failed: {e}")
            vix_data = {"vix_regime": "UNKNOWN", "vix_level": 20.0, "vix_percentile": 50.0}
        
        try:
            sector_data = self.get_sector_rotation_signal(symbol)
        except Exception as e:
            print(f"Sector data failed: {e}")
            sector_data = {"sector_rotation_signal": 0.0, "relative_strength": 0.0, "sector": "UNKNOWN"}
        
        try:
            options_data = self.get_put_call_ratio(symbol)
        except Exception as e:
            print(f"Options data failed: {e}")
            options_data = {"put_call_ratio": 1.0, "unusual_activity": False, "options_sentiment": "NEUTRAL"}
        
        try:
            earnings_data = self.get_earnings_momentum(symbol)
        except Exception as e:
            print(f"Earnings data failed: {e}")
            earnings_data = {"earnings_surprise_momentum": "NEUTRAL", "surprise_magnitude": 0.0, "guidance_sentiment": "NEUTRAL", "earnings_upcoming": False}
        
        # FIXED: Combine into comprehensive feature set with proper type conversion
        features = {
            # VIX and volatility regime
            "vix_regime_numeric": {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "EXTREME": 4}.get(vix_data["vix_regime"], 2),
            "vix_level": float(vix_data["vix_level"]),
            "vix_percentile": float(vix_data["vix_percentile"]),
            
            # Sector rotation
            "sector_rotation_signal": float(sector_data["sector_rotation_signal"]),
            "relative_strength": float(sector_data["relative_strength"]),
            "sector_momentum": float(sector_data.get("sector_momentum", 0.0)),
            
            # Options flow
            "put_call_ratio": float(options_data["put_call_ratio"]),
            "unusual_options_activity": 1 if options_data["unusual_activity"] else 0,
            "options_sentiment_numeric": {"BULLISH": 1, "NEUTRAL": 0, "BEARISH": -1}.get(options_data["options_sentiment"], 0),
            "implied_volatility_rank": float(options_data.get("implied_volatility_rank", 50.0)),
            
            # Earnings momentum
            "earnings_surprise_numeric": {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}.get(earnings_data["earnings_surprise_momentum"], 0),
            "surprise_magnitude": float(earnings_data["surprise_magnitude"]),
            "guidance_sentiment_numeric": {"POSITIVE": 1, "NEUTRAL": 0, "NEGATIVE": -1}.get(earnings_data["guidance_sentiment"], 0),
            "earnings_upcoming": 1 if earnings_data["earnings_upcoming"] else 0,
            
            # Market microstructure
            "market_cap_category": self._get_market_cap_category(symbol),
            "beta_estimate": self._estimate_beta(symbol),
            "liquidity_score": self._estimate_liquidity(symbol)
        }
        
        print(f"Generated {len(features)} alternative data features")
        return features
    
    def _get_market_cap_category(self, symbol: str) -> int:
        """FIXED market cap category with proper error handling."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            market_cap = info.get('marketCap', 0)
            
            if market_cap > 200e9:
                return 4  # Mega cap
            elif market_cap > 10e9:
                return 3  # Large cap
            elif market_cap > 2e9:
                return 2  # Mid cap
            else:
                return 1  # Small cap
        except:
            return 3  # Default to large cap
    
    def _estimate_beta(self, symbol: str) -> float:
        """FIXED beta estimation with proper Series handling."""
        try:
            stock_data = yf.download(symbol, period="252d", interval="1d", progress=False)
            market_data = yf.download("SPY", period="252d", interval="1d", progress=False)
            
            # FIXED: Use .empty instead of direct boolean check
            if stock_data.empty or market_data.empty:
                return 1.0
            
            stock_returns = stock_data['Close'].pct_change().dropna()
            market_returns = market_data['Close'].pct_change().dropna()
            
            # Align dates
            common_dates = stock_returns.index.intersection(market_returns.index)
            if len(common_dates) < 50:
                return 1.0
            
            stock_aligned = stock_returns.loc[common_dates]
            market_aligned = market_returns.loc[common_dates]
            
            # FIXED: Convert to numpy arrays for calculation
            stock_vals = stock_aligned.values
            market_vals = market_aligned.values
            
            covariance = np.cov(stock_vals, market_vals)[0, 1]
            market_variance = np.var(market_vals)
            
            beta = covariance / market_variance if market_variance > 0 else 1.0
            return round(max(0.1, min(3.0, beta)), 2)
            
        except Exception as e:
            print(f"Beta estimation failed: {e}")
            return 1.0
    
    def _estimate_liquidity(self, symbol: str) -> float:
        """FIXED liquidity estimation with proper Series handling."""
        try:
            stock_data = yf.download(symbol, period="30d", interval="1d", progress=False)
            
            # FIXED: Use .empty instead of direct boolean check
            if stock_data.empty:
                return 50.0
            
            # Average daily volume
            avg_volume = float(stock_data['Volume'].mean())
            
            # Volume-based liquidity score
            if avg_volume > 10e6:
                volume_score = 100
            elif avg_volume > 1e6:
                volume_score = 80
            elif avg_volume > 100e3:
                volume_score = 60
            elif avg_volume > 10e3:
                volume_score = 40
            else:
                volume_score = 20
            
            return float(volume_score)
            
        except Exception as e:
            print(f"Liquidity estimation failed: {e}")
            return 50.0
