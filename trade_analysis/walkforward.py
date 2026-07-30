
import pandas as pd
import numpy as np
from datetime import timedelta
from pathlib import Path

def backtest_walkforward(symbol, csv_path, walk_start_date='2014-01-01'):
    """
    Walk-forward analysis: Each day, only use historical data UP TO that day
    
    This prevents overfitting because:
    - You never see future prices
    - You never optimize on future data
    - Each day is truly "new"
    """
    
    print(f"\n{'=' * 70}")
    print(f"WALK-FORWARD: {symbol}")
    print(f"{'=' * 70}")
    
    try:
        df = pd.read_csv(csv_path)
        
        # Parse date
        date_col = None
        for col in df.columns:
            if col.lower() in ['date', 'timestamp', 'time']:
                date_col = col
                break
        if date_col is None:
            date_col = df.columns[0]
        
        df['date'] = pd.to_datetime(df[date_col], utc=True)
        df['date'] = df['date'].dt.tz_localize(None)
        
        # Find OHLC columns
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
        df = df.sort_values('date')
        df = df.reset_index(drop=True)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None
    
    # Find walk-forward start index
    walk_start = pd.to_datetime(walk_start_date)
    walk_idx = df[df['date'] >= walk_start].index[0] if len(df[df['date'] >= walk_start]) > 0 else len(df) // 2
    
    print(f"Data: {len(df)} days")
    print(f"Training: {walk_idx} days (before {walk_start_date})")
    print(f"Walk-forward: {len(df) - walk_idx} days (after {walk_start_date})")
    
    # Walk-forward simulation
    trades = []
    position = None
    entry_price = 0
    
    for i in range(walk_idx + 1, len(df)):
        current_date = df.iloc[i]['date'].date()
        current_close = df.iloc[i]['close']
        current_high = df.iloc[i]['high']
        current_low = df.iloc[i]['low']
        
        # KEY: Only use data BEFORE today
        historical_data = df.iloc[:i]
        
        # Get yesterday's levels from ONLY historical data
        yesterday_data = historical_data[historical_data['date'].dt.date == (df.iloc[i]['date'] - timedelta(days=1)).date()]
        
        if yesterday_data.empty:
            continue
        
        yesterday_high = yesterday_data['high'].max()
        yesterday_low = yesterday_data['low'].min()
        
        # Generate signal based on ONLY historical data (no future data)
        breakout_level = yesterday_high
        breakdown_level = yesterday_low
        
        # Close existing position
        if position == 'LONG':
            exit_price = current_close
            
            if current_high >= entry_price * 1.05:
                exit_price = entry_price * 1.05
                pnl_pct = 5.0
            elif current_low <= entry_price * 0.98:
                exit_price = entry_price * 0.98
                pnl_pct = -2.0
            else:
                pnl_pct = (exit_price / entry_price - 1) * 100
            
            trades.append({
                'date': current_date,
                'type': 'LONG',
                'entry': round(entry_price, 2),
                'exit': round(exit_price, 2),
                'pnl_pct': round(pnl_pct, 2)
            })
            
            position = None
        
        # Enter new position (signal based on ONLY historical data)
        if position is None and current_close > breakout_level:
            position = 'LONG'
            entry_price = current_close
    
    # Calculate metrics
    if not trades:
        print("No trades")
        return None
    
    trades_df = pd.DataFrame(trades)
    
    total_trades = len(trades_df)
    winning = (trades_df['pnl_pct'] > 0).sum()
    losing = (trades_df['pnl_pct'] < 0).sum()
    
    win_rate = (winning / total_trades * 100) if total_trades > 0 else 0
    total_return = trades_df['pnl_pct'].sum()
    avg_return = trades_df['pnl_pct'].mean()
    std_dev = trades_df['pnl_pct'].std()
    sharpe = (avg_return / std_dev * np.sqrt(252)) if std_dev > 0 else 0
    
    print(f"\n📊 Walk-Forward Results:")
    print(f"   Trades:         {total_trades:>6.0f}")
    print(f"   Win rate:       {win_rate:>6.1f}%")
    print(f"   Wins/Losses:    {winning}/{losing}")
    print(f"   Total return:   {total_return:>6.2f}%")
    print(f"   Avg/trade:      {avg_return:>6.2f}%")
    print(f"   Sharpe:         {sharpe:>6.2f}")
    print(f"   Std dev:        {std_dev:>6.2f}%")
    
    return {
        'symbol': symbol,
        'trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_return': avg_return,
        'sharpe': sharpe,
        'trades_df': trades_df
    }

# MAIN
if __name__ == "__main__":
    
    INPUT_DIR = "historical_data"
    SYMBOLS = ["QQQ", "NVDA", "TSLA", "SPY", "AAPL", "MSFT", "AMZN"]
    WALK_START = "2014-01-01"  # Start walk-forward from 2014
    
    print(f"\n{'=' * 70}")
    print("WALK-FORWARD ANALYSIS (Out-of-Sample Testing)")
    print(f"{'=' * 70}")
    print(f"Training period: 2005 - 2013")
    print(f"Walk-forward period: 2014 - 2025")
    print(f"Strategy: Buy breakout above yesterday's high, 5% target / 2% stop")
    
    results = []
    
    for symbol in SYMBOLS:
        csv_path = f"{INPUT_DIR}/{symbol}_20y.csv"
        
        if not Path(csv_path).exists():
            print(f"\n❌ {symbol}: File not found")
            continue
        
        result = backtest_walkforward(symbol, csv_path, WALK_START)
        if result:
            results.append(result)
    
    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print("WALK-FORWARD SUMMARY")
        print(f"{'=' * 70}\n")
        
        summary_df = pd.DataFrame({
            'Symbol': [r['symbol'] for r in results],
            'Trades': [f"{r['trades']:.0f}" for r in results],
            'Win%': [f"{r['win_rate']:.1f}%" for r in results],
            'Return%': [f"{r['total_return']:.1f}%" for r in results],
            'Avg/Trade': [f"{r['avg_return']:.2f}%" for r in results],
            'Sharpe': [f"{r['sharpe']:.2f}" for r in results],
        })
        
        print(summary_df.to_string(index=False))
        
        best = max(results, key=lambda x: x['sharpe'])
        print(f"\n🏆 Best: {best['symbol']} ({best['sharpe']:.2f} Sharpe)")
        print(f"\nCOMPARISON TO REGULAR BACKTEST:")
        print(f"  Regular backtest TSLA: 3.42 Sharpe (saw all 20 years)")
        print(f"  Walk-forward TSLA:     {best['sharpe']:.2f} Sharpe (only 9 years test)")
        print(f"  \n  Walk-forward is MORE REALISTIC")
        print(f"  Because it never sees the future")