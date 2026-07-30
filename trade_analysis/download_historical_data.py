import yfinance as yf
import pandas as pd
import requests
import os
import time
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import List, Tuple

# Setup logging with UTF-8 encoding
import sys
log_file = logging.FileHandler('download_log.txt', encoding='utf-8')
console = logging.StreamHandler(sys.stdout)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[log_file, console]
)
logger = logging.getLogger(__name__)

def download_ticker(ticker: str, output_dir: str = "historical_data", 
                    years: int = 20, max_retries: int = 3) -> Tuple[str, bool]:
    """
    Download historical data for SINGLE ticker
    IMPORTANT: Download one ticker at a time to avoid MultiIndex
    
    Returns: (status_message, success_bool)
    """
    start_date = datetime.now().replace(year=datetime.now().year - years).strftime('%Y-%m-%d')
    
    for attempt in range(max_retries):
        try:
            logger.info(f"Downloading {ticker}... (attempt {attempt + 1}/{max_retries})")
            
            # CRITICAL: Download SINGLE ticker, not multiple (avoids MultiIndex)
            # Use .Ticker API instead of .download() for more control
            ticker_obj = yf.Ticker(ticker)
            data = ticker_obj.history(start=start_date, end=None, interval='1d')
            
            if data.empty:
                msg = f"No data for {ticker}"
                logger.warning(msg)
                return f"Failed: {ticker} - {msg}", False
            
            # Ensure Date is index
            if data.index.name != 'Date':
                data.index.name = 'Date'
            
            # Reset index to make Date a column (NO .str accessor needed!)
            data_with_date = data.reset_index()
            
            # Check columns (simple check - don't use .str)
            logger.info(f"  Columns: {list(data_with_date.columns)}")
            
            # Required columns (basic check)
            required = ['Date', 'Close']  # At minimum need these
            if not all(col in data_with_date.columns for col in required):
                missing = [col for col in required if col not in data_with_date.columns]
                logger.error(f"  Missing columns: {missing}")
                return f"Failed: {ticker} - Missing {missing}", False
            
            # Ensure we have OHLCV
            if not all(col in data_with_date.columns for col in ['Open', 'High', 'Low', 'Volume']):
                logger.warning(f"  {ticker} missing some OHLCV columns, continuing anyway")
            
            # Save to CSV
            output_path = f"{output_dir}/{ticker}_20y.csv"
            data_with_date.to_csv(output_path, index=False)
            
            msg = f"Success: {ticker} - {len(data)} days"
            logger.info(msg)
            return msg, True
            
        except Exception as e:
            error_msg = str(e)[:100]
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                logger.warning(f"  {ticker} failed: {error_msg}")
                logger.info(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"  {ticker} max retries: {error_msg}")
                return f"Failed: {ticker} - {error_msg}", False
    
    return f"Failed: {ticker} - Unknown", False

def get_manual_tickers() -> List[str]:
    """Get manual ticker list"""
    return ['MSFT', 'TSLA', 'QQQ', 'SPY', 'NVDA', 'AAPL', 'AMZN']

def get_github_tickers() -> List[str]:
    """Download ticker list from GitHub"""
    try:
        url = "https://raw.githubusercontent.com/rreichel3/US-Stock-Symbols/main/all/all_tickers.txt"
        response = requests.get(url, timeout=10)
        all_tickers = response.text.strip().split('\n')
        logger.info(f"Downloaded {len(all_tickers)} tickers from GitHub")
        return all_tickers
    except Exception as e:
        logger.error(f"Failed to download from GitHub: {e}, falling back to manual list")
        return get_manual_tickers()

def main(tickers_source: str = 'manual', 
         max_workers: int = 5, 
         output_dir: str = 'historical_data',
         custom_tickers: str = None):
    """
    Download historical data for multiple tickers
    
    Args:
        tickers_source: 'manual' or 'github'
        max_workers: Number of parallel workers
        output_dir: Output directory for CSVs
        custom_tickers: Comma-separated ticker list (overrides source)
    """
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Get ticker list
    if custom_tickers:
        all_tickers = [t.strip().upper() for t in custom_tickers.split(',')]
        logger.info(f"Using custom ticker list: {all_tickers}")
    elif tickers_source == 'github':
        all_tickers = get_github_tickers()
    else:
        all_tickers = get_manual_tickers()
        logger.info(f"Using manual ticker list: {all_tickers}")
    
    # Deduplicate
    all_tickers = list(set(all_tickers))
    logger.info(f"Downloading {len(all_tickers)} unique tickers")
    
    # Download in parallel
    logger.info(f"Starting download with {max_workers} workers...")
    start_time = time.time()
    
    successful = 0
    failed = []
    
    # IMPORTANT: Each worker downloads ONE ticker independently (avoids MultiIndex)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        results = executor.map(
            lambda t: download_ticker(t, output_dir),
            all_tickers
        )
        
        for msg, success in results:
            if success:
                print(msg)
                successful += 1
            else:
                print(msg)
                ticker = msg.split(':')[1].strip() if ':' in msg else '?'
                failed.append(ticker)
    
    elapsed = time.time() - start_time
    
    # Summary
    print(f"\n{'='*60}")
    print(f"Downloaded {successful}/{len(all_tickers)} tickers to {output_dir}/")
    print(f"Time elapsed: {elapsed:.1f} seconds")
    
    if failed:
        print(f"Failed: {', '.join(f for f in failed if f != '?')}")
    
    logger.info(f"Download complete: {successful}/{len(all_tickers)} successful in {elapsed:.1f}s")
    
    return successful, failed

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download historical stock data")
    parser.add_argument("--source", default="manual", choices=["github", "manual"], 
                       help="Ticker source (default: manual)")
    parser.add_argument("--workers", type=int, default=5, 
                       help="Number of parallel workers (default: 5)")
    parser.add_argument("--output", default="historical_data", 
                       help="Output directory (default: historical_data)")
    parser.add_argument("--tickers", type=str, default=None,
                       help="Comma-separated ticker list (overrides source), e.g., 'NVDA,TSLA,SPY,QQQ'")
    
    args = parser.parse_args()
    
    main(
        tickers_source=args.source,
        max_workers=args.workers,
        output_dir=args.output,
        custom_tickers=args.tickers
    )