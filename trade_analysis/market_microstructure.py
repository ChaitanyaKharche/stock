# trade_analysis/market_microstructure.py

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
import asyncio
from concurrent.futures import ThreadPoolExecutor

@dataclass
class OrderBookSnapshot:
    """Level 2 order book data structure"""
    timestamp: datetime
    bids: List[Tuple[float, float]]  # [(price, size), ...]
    asks: List[Tuple[float, float]]
    
    @property
    def bid_ask_spread(self) -> float:
        if self.bids and self.asks:
            return self.asks[0][0] - self.bids[0][0]
        return 0
    
    @property
    def mid_price(self) -> float:
        if self.bids and self.asks:
            return (self.asks[0][0] + self.bids[0][0]) / 2
        return 0

class MarketMicrostructure:
    """
    Market microstructure analysis inspired by prop trading firms
    Implements order flow toxicity, imbalance detection, and spread analysis
    """
    
    def __init__(self):
        self.toxic_flow_threshold = 0.7
        self.imbalance_threshold = 0.6
        self.spread_percentile_window = 100
        
    def calculate_order_flow_imbalance(self, order_book: OrderBookSnapshot, 
                                      levels: int = 5) -> float:
        """
        Calculate order book imbalance like Jane Street
        Positive = buy pressure, Negative = sell pressure
        """
        bid_volume = sum(size for _, size in order_book.bids[:levels])
        ask_volume = sum(size for _, size in order_book.asks[:levels])
        
        if bid_volume + ask_volume == 0:
            return 0
            
        imbalance = (bid_volume - ask_volume) / (bid_volume + ask_volume)
        return imbalance
    
    def detect_toxic_flow(self, trades: pd.DataFrame, window: int = 100) -> float:
        """
        Detect toxic flow using VPIN (Volume-Synchronized Probability of Informed Trading)
        Used by HFT firms to avoid adverse selection
        """
        if len(trades) < window:
            return 0
        
        # Calculate buy/sell volume imbalance
        trades['signed_volume'] = trades['volume'] * np.where(
            trades['price'].diff() > 0, 1, -1
        )
        
        # Rolling window for toxicity
        buy_volume = trades['signed_volume'].rolling(window).apply(
            lambda x: x[x > 0].sum()
        )
        sell_volume = trades['signed_volume'].rolling(window).apply(
            lambda x: abs(x[x < 0].sum())
        )
        
        # VPIN approximation
        total_volume = buy_volume + sell_volume
        vpin = abs(buy_volume - sell_volume) / total_volume
        
        return vpin.iloc[-1] if not vpin.empty else 0
    
    def calculate_effective_spread(self, trades: pd.DataFrame, 
                                 quotes: pd.DataFrame) -> float:
        """
        Calculate realized spread vs quoted spread
        Key metric for market makers
        """
        if trades.empty or quotes.empty:
            return 0
        
        # Merge trades with prevailing quotes
        merged = pd.merge_asof(
            trades.sort_values('timestamp'),
            quotes.sort_values('timestamp'),
            on='timestamp',
            direction='backward'
        )
        
        # Effective spread = 2 * |trade_price - mid_price|
        merged['effective_spread'] = 2 * abs(
            merged['price'] - merged['mid_price']
        )
        
        return merged['effective_spread'].mean()
    
    def microstructure_alpha(self, order_book_history: List[OrderBookSnapshot],
                           trades: pd.DataFrame) -> Dict:
        """
        Generate microstructure-based trading signals
        """
        if len(order_book_history) < 2:
            return {'signal': 'NEUTRAL', 'confidence': 0}
        
        # Current order book metrics
        current_book = order_book_history[-1]
        imbalance = self.calculate_order_flow_imbalance(current_book)
        spread = current_book.bid_ask_spread
        
        # Historical metrics
        spreads = [book.bid_ask_spread for book in order_book_history]
        avg_spread = np.mean(spreads)
        spread_percentile = np.percentile(spreads, 75)
        
        # Toxicity check
        toxicity = self.detect_toxic_flow(trades) if not trades.empty else 0
        
        # Signal generation
        signal = 'NEUTRAL'
        confidence = 0
        
        # High imbalance + tight spread = momentum opportunity
        if imbalance > self.imbalance_threshold and spread < avg_spread:
            signal = 'BUY'
            confidence = min(imbalance * 100, 90)
        elif imbalance < -self.imbalance_threshold and spread < avg_spread:
            signal = 'SELL'
            confidence = min(abs(imbalance) * 100, 90)
        
        # Avoid toxic flow
        if toxicity > self.toxic_flow_threshold:
            confidence *= 0.5
            
        return {
            'signal': signal,
            'confidence': confidence,
            'imbalance': imbalance,
            'spread_ratio': spread / avg_spread if avg_spread > 0 else 1,
            'toxicity': toxicity
        }
