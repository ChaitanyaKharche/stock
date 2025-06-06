import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import yfinance as yf
from dataclasses import dataclass
import json

@dataclass
class BacktestResult:
    """Complete backtest results structure."""
    symbol: str
    start_date: str
    end_date: str
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    total_return: float
    sharpe_ratio: float
    max_drawdown: float
    avg_trade_return: float
    best_trade: float
    worst_trade: float
    trades_detail: List[Dict]

class StrategyBacktester:
    """
    Complete backtesting framework for gap prediction strategy.
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.trades = []
        self.daily_returns = []
        self.positions = []
        
    def run_historical_backtest(
        self, 
        symbol: str, 
        start_date: str, 
        end_date: str,
        strategy_params: Dict = None
    ) -> BacktestResult:
        """
        Run complete historical backtest of gap prediction strategy.
        """
        print(f"🚀 Starting backtest for {symbol} from {start_date} to {end_date}")
        
        # Default strategy parameters
        if strategy_params is None:
            strategy_params = {
                "min_confidence": 50,
                "max_position_size": 0.25,
                "stop_loss": 0.03,  # 3% stop loss
                "take_profit": 0.06,  # 6% take profit
                "hold_days": 1  # Gap trading - hold for 1 day
            }
        
        # Fetch historical data
        try:
            stock_data = yf.download(symbol, start=start_date, end=end_date, interval="1d", progress=False)
            if stock_data.empty:
                raise ValueError(f"No data found for {symbol}")
        except Exception as e:
            print(f"Error fetching data: {e}")
            return self._empty_backtest_result(symbol, start_date, end_date)
        
        # Reset backtest state
        self.current_capital = self.initial_capital
        self.trades = []
        self.daily_returns = []
        self.positions = []
        
        # Simulate trading day by day
        for i in range(len(stock_data) - strategy_params["hold_days"]):
            current_date = stock_data.index[i]
            current_price = stock_data['Close'].iloc[i]
            
            # Simulate gap prediction signal
            signal_data = self._simulate_gap_signal(stock_data.iloc[:i+1], current_date)
            
            # Check if we should enter a position
            if self._should_enter_position(signal_data, strategy_params):
                trade_result = self._execute_trade(
                    symbol=symbol,
                    entry_date=current_date,
                    entry_price=current_price,
                    signal_data=signal_data,
                    stock_data=stock_data.iloc[i:i+strategy_params["hold_days"]+1],
                    strategy_params=strategy_params
                )
                
                if trade_result:
                    self.trades.append(trade_result)
            
            # Calculate daily portfolio value
            portfolio_value = self._calculate_portfolio_value(current_date, current_price)
            daily_return = (portfolio_value - self.initial_capital) / self.initial_capital
            self.daily_returns.append(daily_return)
        
        # Calculate final metrics
        return self._calculate_backtest_metrics(symbol, start_date, end_date)
    
    def _simulate_gap_signal(self, historical_data: pd.DataFrame, current_date: datetime) -> Dict:
        """
        Simulate gap prediction signal based on historical patterns.
        """
        if len(historical_data) < 20:
            return {"signal": "HOLD", "confidence": 0, "gap_probability": 50}
        
        # Calculate technical indicators
        close_prices = historical_data['Close']
        
        # RSI calculation
        delta = close_prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        current_rsi = rsi.iloc[-1] if not rsi.empty else 50
        
        # Simple moving averages
        sma_10 = close_prices.rolling(10).mean().iloc[-1]
        sma_20 = close_prices.rolling(20).mean().iloc[-1]
        current_price = close_prices.iloc[-1]
        
        # Volatility
        returns = close_prices.pct_change().dropna()
        volatility = returns.rolling(20).std().iloc[-1] * np.sqrt(252)
        
        # Generate signal based on technical conditions
        signal = "HOLD"
        confidence = 30
        gap_probability = 50
        
        # Bullish conditions
        if (current_price > sma_10 > sma_20 and 
            30 < current_rsi < 70 and 
            volatility > 0.15):
            signal = "LONG"
            confidence = min(60 + (current_rsi - 50), 85)
            gap_probability = 65
        
        # Bearish conditions
        elif (current_price < sma_10 < sma_20 and 
              (current_rsi > 70 or current_rsi < 30) and 
              volatility > 0.15):
            signal = "SHORT"
            confidence = min(60 + abs(50 - current_rsi), 85)
            gap_probability = 65
        
        return {
            "signal": signal,
            "confidence": confidence,
            "gap_probability": gap_probability,
            "rsi": current_rsi,
            "price_vs_sma10": (current_price / sma_10 - 1) * 100,
            "volatility": volatility
        }
    
    def _should_enter_position(self, signal_data: Dict, strategy_params: Dict) -> bool:
        """Determine if we should enter a position."""
        return (signal_data["signal"] in ["LONG", "SHORT"] and 
                signal_data["confidence"] >= strategy_params["min_confidence"])
    
    def _execute_trade(
        self, 
        symbol: str, 
        entry_date: datetime, 
        entry_price: float,
        signal_data: Dict, 
        stock_data: pd.DataFrame, 
        strategy_params: Dict
    ) -> Optional[Dict]:
        """
        Execute a complete trade with entry, exit, and P&L calculation.
        """
        if len(stock_data) < 2:
            return None
        
        # Calculate position size based on confidence
        confidence_multiplier = signal_data["confidence"] / 100
        position_size = min(
            strategy_params["max_position_size"] * confidence_multiplier,
            strategy_params["max_position_size"]
        )
        
        # Calculate shares to buy/sell
        position_value = self.current_capital * position_size
        shares = int(position_value / entry_price)
        
        if shares == 0:
            return None
        
        # Simulate holding period
        exit_date = stock_data.index[-1]
        exit_price = stock_data['Close'].iloc[-1]
        
        # Check for stop loss or take profit during holding period
        for i in range(1, len(stock_data)):
            current_price = stock_data['Close'].iloc[i]
            current_date = stock_data.index[i]
            
            if signal_data["signal"] == "LONG":
                # Long position - check stop loss and take profit
                price_change = (current_price - entry_price) / entry_price
                
                if price_change <= -strategy_params["stop_loss"]:
                    exit_date = current_date
                    exit_price = current_price
                    break
                elif price_change >= strategy_params["take_profit"]:
                    exit_date = current_date
                    exit_price = current_price
                    break
            
            elif signal_data["signal"] == "SHORT":
                # Short position - check stop loss and take profit
                price_change = (entry_price - current_price) / entry_price
                
                if price_change <= -strategy_params["stop_loss"]:
                    exit_date = current_date
                    exit_price = current_price
                    break
                elif price_change >= strategy_params["take_profit"]:
                    exit_date = current_date
                    exit_price = current_price
                    break
        
        # Calculate P&L
        if signal_data["signal"] == "LONG":
            trade_return = (exit_price - entry_price) / entry_price
        else:  # SHORT
            trade_return = (entry_price - exit_price) / entry_price
        
        trade_pnl = position_value * trade_return
        
        # Update capital
        self.current_capital += trade_pnl
        
        # Record trade
        trade_record = {
            "symbol": symbol,
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": exit_date.strftime("%Y-%m-%d"),
            "signal": signal_data["signal"],
            "confidence": signal_data["confidence"],
            "entry_price": round(entry_price, 2),
            "exit_price": round(exit_price, 2),
            "shares": shares,
            "position_value": round(position_value, 2),
            "trade_return": round(trade_return * 100, 2),  # Percentage
            "trade_pnl": round(trade_pnl, 2),
            "total_capital": round(self.current_capital, 2)
        }
        
        return trade_record
    
    def _calculate_portfolio_value(self, current_date: datetime, current_price: float) -> float:
        """Calculate current portfolio value."""
        # For simplicity, assume all positions are closed daily (gap trading)
        return self.current_capital
    
    def _calculate_backtest_metrics(self, symbol: str, start_date: str, end_date: str) -> BacktestResult:
        """Calculate comprehensive backtest metrics."""
        if not self.trades:
            return self._empty_backtest_result(symbol, start_date, end_date)
        
        # Basic trade statistics
        total_trades = len(self.trades)
        winning_trades = len([t for t in self.trades if t["trade_pnl"] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0
        
        # Return calculations
        total_return = ((self.current_capital - self.initial_capital) / self.initial_capital) * 100
        
        # Trade return statistics
        trade_returns = [t["trade_return"] for t in self.trades]
        avg_trade_return = np.mean(trade_returns) if trade_returns else 0
        best_trade = max(trade_returns) if trade_returns else 0
        worst_trade = min(trade_returns) if trade_returns else 0
        
        # Sharpe ratio calculation
        if len(self.daily_returns) > 1:
            daily_return_std = np.std(self.daily_returns)
            avg_daily_return = np.mean(self.daily_returns)
            sharpe_ratio = (avg_daily_return / daily_return_std * np.sqrt(252)) if daily_return_std > 0 else 0
        else:
            sharpe_ratio = 0
        
        # Maximum drawdown calculation
        cumulative_returns = np.cumprod([1 + r for r in self.daily_returns])
        running_max = np.maximum.accumulate(cumulative_returns)
        drawdowns = (cumulative_returns - running_max) / running_max
        max_drawdown = abs(min(drawdowns)) * 100 if len(drawdowns) > 0 else 0
        
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_trades=total_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate, 2),
            total_return=round(total_return, 2),
            sharpe_ratio=round(sharpe_ratio, 2),
            max_drawdown=round(max_drawdown, 2),
            avg_trade_return=round(avg_trade_return, 2),
            best_trade=round(best_trade, 2),
            worst_trade=round(worst_trade, 2),
            trades_detail=self.trades
        )
    
    def _empty_backtest_result(self, symbol: str, start_date: str, end_date: str) -> BacktestResult:
        """Return empty backtest result for error cases."""
        return BacktestResult(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date,
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            total_return=0.0,
            sharpe_ratio=0.0,
            max_drawdown=0.0,
            avg_trade_return=0.0,
            best_trade=0.0,
            worst_trade=0.0,
            trades_detail=[]
        )

def run_strategy_comparison(symbols: List[str], start_date: str, end_date: str) -> Dict:
    """
    Run backtests across multiple symbols and compare performance.
    """
    backtester = StrategyBacktester()
    results = {}
    
    for symbol in symbols:
        print(f"Backtesting {symbol}...")
        result = backtester.run_historical_backtest(symbol, start_date, end_date)
        results[symbol] = result
    
    # Create comparison summary
    summary = {
        "best_performer": max(results.keys(), key=lambda s: results[s].total_return),
        "highest_win_rate": max(results.keys(), key=lambda s: results[s].win_rate),
        "best_sharpe": max(results.keys(), key=lambda s: results[s].sharpe_ratio),
        "lowest_drawdown": min(results.keys(), key=lambda s: results[s].max_drawdown),
        "results": results
    }
    
    return summary
