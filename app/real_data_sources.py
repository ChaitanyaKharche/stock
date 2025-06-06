import pandas as pd
import numpy as np
import requests
from typing import Dict, List, Optional
from datetime import datetime, timedelta
import yfinance as yf
import warnings
warnings.filterwarnings("ignore")

class RealDataProvider:
    """
    Complete real data provider using your actual API keys.
    """
    
    def __init__(self):
        # Your actual API keys
        self.fred_api_key = "4d5ee6809817383ae261f77b8ad99f6f"
        self.alpha_vantage_api_key = "COBUX7ZN3KK2WG9H"
        
        # API endpoints
        self.fred_base_url = "https://api.stlouisfed.org/fred"
        self.alpha_vantage_base_url = "https://www.alphavantage.co/query"
        
    def get_economic_indicators(self) -> Dict:
        """
        Get comprehensive economic indicators from FRED API.
        """
        indicators = {
            'GDP': 'GDP',  # Gross Domestic Product
            'UNRATE': 'UNRATE',  # Unemployment Rate
            'FEDFUNDS': 'FEDFUNDS',  # Federal Funds Rate
            'CPIAUCSL': 'CPIAUCSL',  # Consumer Price Index
            'DEXUSEU': 'DEXUSEU',  # USD/EUR Exchange Rate
            'DGS10': 'DGS10',  # 10-Year Treasury Rate
            'VIXCLS': 'VIXCLS',  # VIX Volatility Index
            'NASDAQCOM': 'NASDAQCOM'  # NASDAQ Composite Index
        }
        
        economic_data = {}
        
        for name, series_id in indicators.items():
            try:
                # FRED API call
                url = f"{self.fred_base_url}/series/observations"
                params = {
                    'series_id': series_id,
                    'api_key': self.fred_api_key,
                    'file_type': 'json',
                    'limit': 10,  # Get last 10 observations
                    'sort_order': 'desc'
                }
                
                response = requests.get(url, params=params, timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    observations = data.get('observations', [])
                    
                    if observations:
                        # Get most recent valid observation
                        for obs in observations:
                            if obs['value'] != '.':
                                economic_data[name] = {
                                    'value': float(obs['value']),
                                    'date': obs['date'],
                                    'series_id': series_id
                                }
                                break
                
                # Rate limiting - FRED allows 120 requests per minute
                import time
                time.sleep(0.5)  # 0.5 second delay between requests
                
            except Exception as e:
                print(f"Error fetching {name} from FRED: {e}")
                economic_data[name] = {'value': 0.0, 'date': 'N/A', 'series_id': series_id}
        
        return economic_data
    
    def get_alpha_vantage_fundamentals(self, symbol: str) -> Dict:
        """
        Get fundamental data from Alpha Vantage (free tier).
        """
        try:
            # Company Overview (Free tier)
            url = self.alpha_vantage_base_url
            params = {
                'function': 'OVERVIEW',
                'symbol': symbol,
                'apikey': self.alpha_vantage_api_key
            }
            
            response = requests.get(url, params=params, timeout=15)
            
            if response.status_code == 200:
                data = response.json()
                
                # Extract key fundamental metrics
                fundamentals = {
                    'market_cap': self._safe_float(data.get('MarketCapitalization', 0)),
                    'pe_ratio': self._safe_float(data.get('PERatio', 0)),
                    'peg_ratio': self._safe_float(data.get('PEGRatio', 0)),
                    'book_value': self._safe_float(data.get('BookValue', 0)),
                    'dividend_yield': self._safe_float(data.get('DividendYield', 0)),
                    'eps': self._safe_float(data.get('EPS', 0)),
                    'revenue_ttm': self._safe_float(data.get('RevenueTTM', 0)),
                    'profit_margin': self._safe_float(data.get('ProfitMargin', 0)),
                    'operating_margin': self._safe_float(data.get('OperatingMarginTTM', 0)),
                    'return_on_equity': self._safe_float(data.get('ReturnOnEquityTTM', 0)),
                    'return_on_assets': self._safe_float(data.get('ReturnOnAssetsTTM', 0)),
                    'debt_to_equity': self._safe_float(data.get('DebtToEquityRatio', 0)),
                    'beta': self._safe_float(data.get('Beta', 1.0)),
                    'sector': data.get('Sector', 'Unknown'),
                    'industry': data.get('Industry', 'Unknown')
                }
                
                return fundamentals
            
        except Exception as e:
            print(f"Error fetching Alpha Vantage fundamentals for {symbol}: {e}")
        
        # Return default values if API fails
        return {
            'market_cap': 0, 'pe_ratio': 0, 'peg_ratio': 0, 'book_value': 0,
            'dividend_yield': 0, 'eps': 0, 'revenue_ttm': 0, 'profit_margin': 0,
            'operating_margin': 0, 'return_on_equity': 0, 'return_on_assets': 0,
            'debt_to_equity': 0, 'beta': 1.0, 'sector': 'Unknown', 'industry': 'Unknown'
        }
    
    def get_alpha_vantage_technical_indicators(self, symbol: str) -> Dict:
        """
        Get technical indicators from Alpha Vantage (free tier).
        """
        indicators = {}
        
        # List of free technical indicators
        free_indicators = [
            ('RSI', 'RSI'),
            ('MACD', 'MACD'),
            ('STOCH', 'STOCH'),
            ('ADX', 'ADX')
        ]
        
        for indicator_name, function in free_indicators:
            try:
                url = self.alpha_vantage_base_url
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': 'daily',
                    'time_period': 14,
                    'series_type': 'close',
                    'apikey': self.alpha_vantage_api_key
                }
                
                response = requests.get(url, params=params, timeout=15)
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Extract most recent value
                    if f'Technical Analysis: {function}' in data:
                        tech_data = data[f'Technical Analysis: {function}']
                        if tech_data:
                            latest_date = max(tech_data.keys())
                            latest_values = tech_data[latest_date]
                            
                            if indicator_name == 'RSI':
                                indicators['RSI'] = self._safe_float(latest_values.get('RSI', 50))
                            elif indicator_name == 'MACD':
                                indicators['MACD'] = self._safe_float(latest_values.get('MACD', 0))
                                indicators['MACD_Signal'] = self._safe_float(latest_values.get('MACD_Signal', 0))
                                indicators['MACD_Hist'] = self._safe_float(latest_values.get('MACD_Hist', 0))
                            elif indicator_name == 'STOCH':
                                indicators['SlowK'] = self._safe_float(latest_values.get('SlowK', 50))
                                indicators['SlowD'] = self._safe_float(latest_values.get('SlowD', 50))
                            elif indicator_name == 'ADX':
                                indicators['ADX'] = self._safe_float(latest_values.get('ADX', 25))
                
                # Rate limiting for Alpha Vantage free tier (25 requests per day)
                import time
                time.sleep(12)  # 12 seconds between requests to stay under daily limit
                
            except Exception as e:
                print(f"Error fetching {indicator_name} from Alpha Vantage: {e}")
        
        return indicators
    
    def get_enhanced_market_data(self, symbol: str) -> Dict:
        """
        Get comprehensive market data combining multiple sources.
        """
        print(f"Fetching enhanced market data for {symbol}...")
        
        # Get data from multiple sources
        economic_data = self.get_economic_indicators()
        fundamentals = self.get_alpha_vantage_fundamentals(symbol)
        technical_indicators = self.get_alpha_vantage_technical_indicators(symbol)
        
        # Get additional data from yfinance (free and reliable)
        yf_data = self._get_yfinance_data(symbol)
        
        # Combine all data sources
        enhanced_data = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
            'economic_indicators': economic_data,
            'fundamentals': fundamentals,
            'technical_indicators': technical_indicators,
            'market_data': yf_data,
            'data_sources': ['FRED', 'Alpha Vantage', 'Yahoo Finance']
        }
        
        return enhanced_data
    
    def _get_yfinance_data(self, symbol: str) -> Dict:
        """Get supplementary data from yfinance."""
        try:
            ticker = yf.Ticker(symbol)
            
            # Get recent price data
            hist = ticker.history(period="5d")
            if not hist.empty:
                latest = hist.iloc[-1]
                
                # Get additional info
                info = ticker.info
                
                return {
                    'current_price': float(latest['Close']),
                    'volume': int(latest['Volume']),
                    'high_52w': info.get('fiftyTwoWeekHigh', 0),
                    'low_52w': info.get('fiftyTwoWeekLow', 0),
                    'avg_volume': info.get('averageVolume', 0),
                    'shares_outstanding': info.get('sharesOutstanding', 0),
                    'float_shares': info.get('floatShares', 0)
                }
        except Exception as e:
            print(f"Error fetching yfinance data: {e}")
        
        return {}
    
    def _safe_float(self, value) -> float:
        """Safely convert value to float."""
        try:
            if value in [None, '', 'None', 'N/A']:
                return 0.0
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def get_market_regime_analysis(self) -> Dict:
        """
        Analyze current market regime using economic indicators.
        """
        economic_data = self.get_economic_indicators()
        
        # Extract key indicators
        vix = economic_data.get('VIXCLS', {}).get('value', 20)
        fed_funds = economic_data.get('FEDFUNDS', {}).get('value', 2.5)
        unemployment = economic_data.get('UNRATE', {}).get('value', 5.0)
        
        # Determine market regime
        regime = "NEUTRAL"
        risk_level = "MEDIUM"
        
        if vix < 15 and unemployment < 4.5:
            regime = "BULL_MARKET"
            risk_level = "LOW"
        elif vix > 30 or unemployment > 7:
            regime = "BEAR_MARKET"
            risk_level = "HIGH"
        elif vix > 25:
            regime = "HIGH_VOLATILITY"
            risk_level = "HIGH"
        
        return {
            'regime': regime,
            'risk_level': risk_level,
            'vix_level': vix,
            'fed_funds_rate': fed_funds,
            'unemployment_rate': unemployment,
            'analysis_date': datetime.now().isoformat()
        }
    

    # Add this method to your AlternativeDataProcessor class
    def get_real_economic_data(self, symbol: str) -> Dict:
        """
        Get real economic data using FRED and Alpha Vantage APIs.
        """
        from .real_data_sources import RealDataProvider
        
        data_provider = RealDataProvider()
        
        # Get economic indicators
        economic_indicators = data_provider.get_economic_indicators()
        
        # Get fundamental data
        fundamentals = data_provider.get_alpha_vantage_fundamentals(symbol)
        
        # Get market regime
        market_regime = data_provider.get_market_regime_analysis()
        
        # Combine into enhanced features
        enhanced_features = {
            # Economic indicators from FRED
            "gdp_growth": economic_indicators.get('GDP', {}).get('value', 0),
            "unemployment_rate": economic_indicators.get('UNRATE', {}).get('value', 5.0),
            "fed_funds_rate": economic_indicators.get('FEDFUNDS', {}).get('value', 2.5),
            "inflation_rate": economic_indicators.get('CPIAUCSL', {}).get('value', 2.0),
            "vix_level": economic_indicators.get('VIXCLS', {}).get('value', 20.0),
            
            # Fundamental data from Alpha Vantage
            "pe_ratio": fundamentals.get('pe_ratio', 15),
            "market_cap": fundamentals.get('market_cap', 0),
            "beta": fundamentals.get('beta', 1.0),
            "profit_margin": fundamentals.get('profit_margin', 0),
            "debt_to_equity": fundamentals.get('debt_to_equity', 0),
            
            # Market regime analysis
            "market_regime": market_regime.get('regime', 'NEUTRAL'),
            "risk_level": market_regime.get('risk_level', 'MEDIUM')
        }
        
        return enhanced_features

