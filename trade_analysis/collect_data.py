# collect_data.py
import asyncio
import argparse
import json
import httpx
import os

# We import the provider from your existing structure
from trade_analysis.data import UnifiedDataProvider

async def main(symbol: str):
    """
    Fetches data from external APIs that might be blocked on the HPC
    and saves it to a local JSON file.
    """
    print(f"--- Starting data collection for {symbol} ---")
    
    # Ensure the directory for saving the data exists
    output_dir = "local_data"
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{symbol.upper()}_external_data.json")

    provider = UnifiedDataProvider()
    all_data = {}

    # 1. Fetch Finnhub News Data
    async with httpx.AsyncClient() as client:
        print("Fetching news data from Finnhub...")
        news_data, source = await provider.fetch_news(symbol, client)
        if source != "error":
            all_data['news_data'] = news_data
            print(f"✅ Successfully fetched {len(news_data)} news articles.")
        else:
            print("❌ Failed to fetch news data.")
            all_data['news_data'] = [] # Save empty list on failure

    # 2. Fetch Reddit Data
    print("Fetching social sentiment data from Reddit...")
    reddit_data, source = await provider.fetch_reddit_data(symbol)
    if source != "error": # fetch_reddit_data doesn't return 'error', but good practice
        all_data['reddit_data'] = reddit_data
        print(f"✅ Successfully fetched {len(reddit_data)} Reddit posts.")
    else:
        print("❌ Failed to fetch Reddit data.")
        all_data['reddit_data'] = []

    # 3. Save to JSON file
    with open(output_path, 'w') as f:
        json.dump(all_data, f, indent=4)
    
    print(f"\n--- Data collection complete. ---")
    print(f"All data saved to: {output_path}")
    
    await provider.close()
    print(f"\n--- Data collection complete. ---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect external financial data for a given stock symbol.")
    parser.add_argument("--symbol", type=str, required=True, help="Stock symbol to collect data for (e.g., QQQ).")
    args = parser.parse_args()
    
    # You will need to have your environment variables (FINNHUB_API_KEY, REDDIT_...)
    # set in your local terminal for this to work.
    asyncio.run(main(args.symbol))