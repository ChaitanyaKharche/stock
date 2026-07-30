import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
from datetime import datetime, timedelta
import os
from pathlib import Path

# PUT YOUR API KEY AND SECRET HERE
API_KEY = "PKJXXJGKHTLUJ27E63Q2CB73DH"  # Replace with your actual API key API: 
SECRET_KEY = "3NK9D3AETRxrCTwzKkGk8npjSof8jfoAycpLUfCzTTwp"  # Replace with your actual secret key

# Symbols to download
SYMBOLS = ["QQQ", "NVDA", "TSLA", "SPY", "AAPL", "MSFT", "AMZN"]

# Download parameters
START_DATE = "2024-01-01"
END_DATE = datetime.now().strftime("%Y-%m-%d")
OUTPUT_DIR = "historical_data_5min"

def download_5min_data(symbols, start_date, end_date, output_dir):
    """Download 5-minute bars from Alpaca for multiple symbols"""
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Create Alpaca client
    client = StockHistoricalDataClient(
        api_key=API_KEY,
        secret_key=SECRET_KEY
    )
    
    print(f"Downloading 5-min data for {len(symbols)} symbols...")
    print(f"Date range: {start_date} to {end_date}")
    print(f"Output directory: {output_dir}\n")
    
    # Create TimeFrame for 5 minutes
    timeframe_5min = TimeFrame(5, TimeFrameUnit.Minute)
    
    for symbol in symbols:
        try:
            print(f"Downloading {symbol}...", end=" ", flush=True)
            
            # Create request for single symbol
            request = StockBarsRequest(
                symbol_or_symbols=[symbol],
                timeframe=timeframe_5min,
                start=start_date,
                end=end_date,
                feed="sip",
                adjustment="split"
            )
            
            # Download data
            bars = client.get_stock_bars(request)
            
            # CRITICAL FIX: Handle Alpaca MultiIndex DataFrame correctly
            # Alpaca returns DataFrame with MultiIndex (symbol, timestamp)
            # We need to flatten it and extract just this symbol's data
            
            if bars.df.empty:
                print(f"❌ No data available")
                continue
            
            # Get the DataFrame
            df = bars.df
            
            # Reset index to convert MultiIndex to columns
            # This converts (symbol, timestamp) index to regular columns
            df = df.reset_index()
            
            # Now df has columns: ['symbol', 'timestamp', 'open', 'high', 'low', 'close', 'volume']
            # We can safely rename them
            
            # Check what columns we actually have
            print(f"\n    DEBUG: Columns after reset_index: {list(df.columns)}", end="")
            
            # Rename to standard format
            df = df.rename(columns={
                'timestamp': 'date',
                'open': 'open',
                'high': 'high',
                'low': 'low',
                'close': 'close',
                'volume': 'volume'
            })
            
            # Keep only the columns we need
            cols_to_keep = ['date', 'open', 'high', 'low', 'close', 'volume']
            available_cols = [c for c in cols_to_keep if c in df.columns]
            df = df[available_cols]
            
            # Ensure proper types
            df["date"] = pd.to_datetime(df["date"])
            for col in ["open", "high", "low", "close", "volume"]:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors="coerce")
            
            # Drop NaN rows
            df = df.dropna()
            
            if df.empty:
                print(f"❌ No valid data after processing")
                continue
            
            # Save to CSV
            output_path = f"{output_dir}/{symbol}_5min.csv"
            df.to_csv(output_path, index=False)
            
            print(f" ✓ {len(df):>6d} bars saved")
            
        except Exception as e:
            print(f"❌ Error: {str(e)[:70]}")
    
    print(f"\n✓ All downloads complete!")
    print(f"Files saved to: {output_dir}/")

if __name__ == "__main__":
    print("=" * 70)
    print("ALPACA 5-MINUTE DATA DOWNLOADER (FINAL VERSION)")
    print("=" * 70)
    print(f"\nUsing API Key: {API_KEY[:10]}...{API_KEY[-4:]}")
    print(f"Symbols: {', '.join(SYMBOLS)}")
    print(f"Date range: {START_DATE} to {END_DATE}")
    print()
    
    download_5min_data(SYMBOLS, START_DATE, END_DATE, OUTPUT_DIR)
    
    # Verify downloads
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    for symbol in SYMBOLS:
        file_path = f"{OUTPUT_DIR}/{symbol}_5min.csv"
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✓ {symbol}: {len(df):>6d} rows")
        else:
            print(f"❌ {symbol}: File not found")