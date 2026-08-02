
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from pathlib import Path
import json

from ..paths import LOGS_DIR

class BreakoutTrader:
    """Production-grade breakout strategy with realistic execution costs"""
    
    def __init__(self, symbol, csv_path, walk_start='2014-01-01', 
                 target_pct=5.0, stop_pct=2.0, entry_slippage=0.10, 
                 exit_slippage=0.05, commission=0.001, use_realistic_fills=True):
        """
        Init trader with all cost parameters
        
        Parameters:
          target_pct: Profit target %
          stop_pct: Stop loss %
          entry_slippage: Slippage at entry (% against you)
          exit_slippage: Slippage at exit (% against you)
          commission: Per-trade commission %
          use_realistic_fills: Use High/Low for realistic entry
        """
        
        self.symbol = symbol
        self.target_pct = target_pct
        self.stop_pct = stop_pct
        self.entry_slippage = entry_slippage
        self.exit_slippage = exit_slippage
        self.commission = commission
        self.use_realistic_fills = use_realistic_fills
        
        self.df = self._load_data(csv_path)
        self.walk_idx = self._find_walk_start(walk_start)
        
    def _load_data(self, csv_path):
        """Load and clean data"""
        df = pd.read_csv(csv_path)
        
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'timestamp']:
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]
        
        df['date'] = pd.to_datetime(df[date_col], utc=True)
        df['date'] = df['date'].dt.tz_localize(None)
        
        cols_lower = {col: col.lower() for col in df.columns}
        for old_col, new_col in cols_lower.items():
            if 'close' in new_col:
                df['close'] = pd.to_numeric(df[old_col], errors='coerce')
            elif 'high' in new_col:
                df['high'] = pd.to_numeric(df[old_col], errors='coerce')
            elif 'low' in new_col:
                df['low'] = pd.to_numeric(df[old_col], errors='coerce')
            elif 'open' in new_col:
                df['open'] = pd.to_numeric(df[old_col], errors='coerce')
        
        df = df.dropna(subset=['date', 'close', 'high', 'low', 'open'])
        df = df.sort_values('date').reset_index(drop=True)
        
        return df
    
    def _find_walk_start(self, walk_start_date):
        """Find index where walk-forward starts"""
        walk_start = pd.to_datetime(walk_start_date)
        walk_idx = self.df[self.df['date'] >= walk_start].index[0] if len(self.df[self.df['date'] >= walk_start]) > 0 else len(self.df) // 2
        return walk_idx
    
    def _get_realistic_entry(self, yesterday_high, today_open, today_close, today_high):
        """Get realistic entry price with slippage"""
        
        if self.use_realistic_fills:
            breakout_level = yesterday_high
            
            if today_close > breakout_level:
                if today_open > breakout_level:
                    entry_price = today_open
                else:
                    entry_price = breakout_level
            else:
                return None
        else:
            entry_price = today_close
        
        entry_with_slippage = entry_price * (1 + self.entry_slippage / 100)
        
        return entry_with_slippage
    
    def _get_realistic_exit(self, entry_price, today_high, today_low, today_close):
        """Get realistic exit with slippage and costs"""
        
        target_price = entry_price * (1 + self.target_pct / 100)
        stop_price = entry_price * (1 - self.stop_pct / 100)
        
        if today_high >= target_price:
            exit_price = target_price * (1 - self.exit_slippage / 100)
            exit_reason = 'TARGET'
            gross_pnl_pct = self.target_pct
        elif today_low <= stop_price:
            exit_price = stop_price * (1 + self.exit_slippage / 100)
            exit_reason = 'STOP'
            gross_pnl_pct = -self.stop_pct
        else:
            exit_price = today_close * (1 - self.exit_slippage / 100)
            exit_reason = 'CLOSE'
            gross_pnl_pct = (exit_price / entry_price - 1) * 100
        
        net_pnl_pct = gross_pnl_pct - (self.commission * 2 * 100)
        
        return exit_price, net_pnl_pct, exit_reason
    
    def backtest_walkforward(self):
        """Run walk-forward backtest with all costs"""
        
        trades = []
        position = None
        entry_price = 0
        entry_date = None
        
        for i in range(self.walk_idx + 1, len(self.df)):
            current_date = self.df.iloc[i]['date'].date()
            current_close = self.df.iloc[i]['close']
            current_high = self.df.iloc[i]['high']
            current_low = self.df.iloc[i]['low']
            current_open = self.df.iloc[i]['open']
            
            historical_data = self.df.iloc[:i]
            yesterday_date = (self.df.iloc[i]['date'] - timedelta(days=1)).date()
            yesterday_data = historical_data[historical_data['date'].dt.date == yesterday_date]
            
            if yesterday_data.empty:
                continue
            
            yesterday_high = yesterday_data['high'].max()
            
            if position == 'LONG':
                exit_price, net_pnl_pct, exit_reason = self._get_realistic_exit(
                    entry_price, current_high, current_low, current_close
                )
                
                trades.append({
                    'entry_date': entry_date,
                    'exit_date': current_date,
                    'symbol': self.symbol,
                    'entry_price': round(entry_price, 2),
                    'exit_price': round(exit_price, 2),
                    'pnl_pct': round(net_pnl_pct, 3),
                    'days_held': (current_date - entry_date).days,
                    'exit_reason': exit_reason,
                    'gross_pnl_pct': round(net_pnl_pct + (self.commission * 2 * 100), 3)
                })
                
                position = None
            
            if position is None:
                entry_price_raw = self._get_realistic_entry(
                    yesterday_high, current_open, current_close, current_high
                )
                
                if entry_price_raw is not None:
                    position = 'LONG'
                    entry_price = entry_price_raw
                    entry_date = current_date
        
        return pd.DataFrame(trades) if trades else pd.DataFrame()
    
    def calculate_metrics(self, trades_df):
        """Calculate comprehensive metrics"""
        
        if trades_df.empty:
            return None
        
        total_trades = len(trades_df)
        winning = (trades_df['pnl_pct'] > 0).sum()
        losing = (trades_df['pnl_pct'] < 0).sum()
        
        win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
        
        total_return = trades_df['pnl_pct'].sum()
        avg_return = trades_df['pnl_pct'].mean()
        std_dev = trades_df['pnl_pct'].std()
        
        sharpe = (avg_return / std_dev * np.sqrt(252)) if std_dev > 0 else 0
        
        best_trade = trades_df['pnl_pct'].max()
        worst_trade = trades_df['pnl_pct'].min()
        avg_win = trades_df[trades_df['pnl_pct'] > 0]['pnl_pct'].mean() if winning > 0 else 0
        avg_loss = trades_df[trades_df['pnl_pct'] < 0]['pnl_pct'].mean() if losing > 0 else 0
        
        profit_factor = abs((winning * avg_win) / (losing * avg_loss)) if losing > 0 and avg_loss != 0 else 0
        
        avg_days = trades_df['days_held'].mean()
        
        max_dd = self._calculate_max_drawdown(trades_df)
        
        return {
            'total_trades': total_trades,
            'win_rate': win_rate,
            'winning': winning,
            'losing': losing,
            'total_return': total_return,
            'avg_return': avg_return,
            'sharpe': sharpe,
            'best_trade': best_trade,
            'worst_trade': worst_trade,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'std_dev': std_dev,
            'max_dd': max_dd,
            'avg_days': avg_days
        }
    
    def _calculate_max_drawdown(self, trades_df):
        """Calculate maximum drawdown"""
        cumsum = trades_df['pnl_pct'].cumsum()
        return (cumsum - cumsum.expanding().max()).min()
    
    def print_results(self, metrics):
        """Print formatted results"""
        
        print(f"\n{'=' * 70}")
        print(f"WALK-FORWARD + SLIPPAGE: {self.symbol}")
        print(f"{'=' * 70}")
        print(f"Training: 2005 - 2013 ({self.walk_idx} days)")
        print(f"Testing: 2014 - 2025 ({len(self.df) - self.walk_idx} days)")
        print(f"\nExecution costs:")
        print(f"  Entry slippage: {self.entry_slippage}%")
        print(f"  Exit slippage: {self.exit_slippage}%")
        print(f"  Commission: {self.commission}% per trade")
        
        print(f"\n📊 Results:")
        print(f"  Trades:         {metrics['total_trades']:>8.0f}")
        print(f"  Win rate:       {metrics['win_rate']:>8.1f}%")
        print(f"  Wins/Losses:    {metrics['winning']}/{metrics['losing']}")
        print(f"  Total return:   {metrics['total_return']:>8.2f}%")
        print(f"  Avg/trade:      {metrics['avg_return']:>8.3f}%")
        print(f"  Best trade:     {metrics['best_trade']:>8.2f}%")
        print(f"  Worst trade:    {metrics['worst_trade']:>8.2f}%")
        print(f"  Avg win:        {metrics['avg_win']:>8.2f}%")
        print(f"  Avg loss:       {metrics['avg_loss']:>8.2f}%")
        print(f"  Profit factor:  {metrics['profit_factor']:>8.2f}")
        print(f"  Sharpe:         {metrics['sharpe']:>8.2f}")
        print(f"  Max DD:         {metrics['max_dd']:>8.2f}%")
        print(f"  Std dev:        {metrics['std_dev']:>8.2f}%")
        print(f"  Avg days/trade: {metrics['avg_days']:>8.1f}")


def main():
    
    INPUT_DIR = "historical_data"
    SYMBOLS = ["TSLA", "NVDA", "AAPL", "QQQ", "AMZN", "SPY", "MSFT"]
    
    print(f"\n{'=' * 70}")
    print("WALK-FORWARD BACKTEST WITH REALISTIC EXECUTION COSTS")
    print(f"{'=' * 70}")
    print(f"Includes: Entry slippage, exit slippage, commissions")
    
    all_results = {}
    
    for symbol in SYMBOLS:
        csv_path = f"{INPUT_DIR}/{symbol}_20y.csv"
        
        if not Path(csv_path).exists():
            print(f"\n❌ {symbol}: File not found")
            continue
        
        try:
            trader = BreakoutTrader(
                symbol,
                csv_path,
                target_pct=5.0,
                stop_pct=2.0,
                entry_slippage=0.10,
                exit_slippage=0.05,
                commission=0.001,
                use_realistic_fills=True
            )
            
            trades_df = trader.backtest_walkforward()
            
            if not trades_df.empty:
                metrics = trader.calculate_metrics(trades_df)
                trader.print_results(metrics)
                
                all_results[symbol] = {
                    'metrics': metrics,
                    'trades': trades_df
                }
                
                trades_df.to_csv(str(LOGS_DIR / f'trades_{symbol}_slippage.csv'), index=False)
        
        except Exception as e:
            print(f"\n❌ {symbol}: {e}")
    
    if all_results:
        print(f"\n\n{'=' * 70}")
        print("SUMMARY - ALL SYMBOLS")
        print(f"{'=' * 70}\n")
        
        summary_data = []
        for symbol, data in all_results.items():
            m = data['metrics']
            summary_data.append({
                'Symbol': symbol,
                'Trades': f"{m['total_trades']:.0f}",
                'Win%': f"{m['win_rate']:.1f}%",
                'Return%': f"{m['total_return']:.1f}%",
                'Sharpe': f"{m['sharpe']:.2f}",
                'Profit Factor': f"{m['profit_factor']:.2f}",
                'Max DD%': f"{m['max_dd']:.1f}%"
            })
        
        summary_df = pd.DataFrame(summary_data)
        print(summary_df.to_string(index=False))
        
        best_symbol = max(all_results.items(), key=lambda x: x[1]['metrics']['sharpe'])
        print(f"\n🏆 Best: {best_symbol[0]} ({best_symbol[1]['metrics']['sharpe']:.2f} Sharpe)")
        
        print(f"\nDEPLOYMENT RECOMMENDATION:")
        for symbol, data in sorted(all_results.items(), key=lambda x: x[1]['metrics']['sharpe'], reverse=True):
            m = data['metrics']
            if m['sharpe'] > 2.0:
                print(f"  ✓ {symbol}: Deploy (Sharpe {m['sharpe']:.2f})")
            elif m['sharpe'] > 1.0:
                print(f"  ⚠️  {symbol}: Maybe deploy (Sharpe {m['sharpe']:.2f})")
            else:
                print(f"  ❌ {symbol}: Skip (Sharpe {m['sharpe']:.2f})")
        
        print(f"\nNext: Deploy on TSLA, NVDA, AAPL at $250-500 per trade")
        print(f"Position sizing: Risk no more than 2% per trade")


if __name__ == "__main__":
    main()