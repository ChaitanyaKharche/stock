import os
import pandas as pd
import numpy as np
import yfinance as yf
import finnhub
import praw
import pandas_ta as ta
from transformers import pipeline
from openai import OpenAI
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from datetime import datetime, timedelta, date
from dotenv import load_dotenv

# Load environment variables from.env file
load_dotenv()

# --- 1. API Key Configuration ---
FINNHUB_API_KEY = os.getenv('FINNHUB_API_KEY')
REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# --- 2. Data Acquisition Functions ---

def get_stock_data(symbol: str, period: str = "1y") -> pd.DataFrame | None:
    """ Fetches historical OHLCV data for a given stock symbol. """
    print(f"Fetching stock data for {symbol}...")
    try:
        stock = yf.Ticker(symbol)
        hist_data = stock.history(period=period)
        if hist_data.empty:
            print(f"Warning: No data found for symbol {symbol} for period {period}.")
            return None
        hist_data.columns = [col.capitalize() for col in hist_data.columns] # Standardize column names
        # Convert index to Date (yfinance often returns datetime index)
        hist_data.index = pd.to_datetime(hist_data.index).date
        print(f"Successfully fetched {len(hist_data)} data points for {symbol}.")
        # Ensure required columns exist
        required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
        if not all(col in hist_data.columns for col in required_cols):
             print(f"Warning: Missing required columns in yfinance data for {symbol}. Available: {hist_data.columns}")
             # Attempt to return available columns, TA might fail later
             return hist_data[[col for col in required_cols if col in hist_data.columns]]
        return hist_data[required_cols]
    except Exception as e:
        print(f"Error fetching stock data for {symbol}: {e}")
        return None

def get_news_data(symbol: str, days_back: int = 1) -> pd.DataFrame | None:
    """ Fetches recent news headlines for a given stock symbol using Finnhub. """
    if not FINNHUB_API_KEY:
        print("Error: Finnhub API key not configured.")
        return None
    print(f"Fetching news data for {symbol}...")
    finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    try:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
        news = finnhub_client.company_news(symbol, _from=start_date, to=end_date)
        if not news:
            print(f"No news found for {symbol} in the last {days_back} day(s).")
            return pd.DataFrame(columns=['datetime', 'headline', 'source', 'summary'])

        # FIX: Initialize empty list
        news_list = []
        for item in news:
            news_list.append({
                'datetime': datetime.fromtimestamp(item['datetime']),
                'headline': item['headline'],
                'source': item['source'],
                'summary': item.get('summary', '')
            })

        news_df = pd.DataFrame(news_list).sort_values(by='datetime', ascending=False)
        print(f"Successfully fetched {len(news_df)} news items for {symbol}.")
        return news_df
    except Exception as e:
        print(f"Error fetching news data for {symbol} from Finnhub: {e}")
        if "API limit reached" in str(e):
            print("Consider upgrading your Finnhub plan or reducing request frequency.")
        return None

def get_reddit_data(symbol: str, subreddits: list = ['stocks', 'wallstreetbets'], limit: int = 50) -> pd.DataFrame | None:
    """ Fetches recent Reddit posts mentioning a stock symbol. """
    # FIX: Correct check for API credentials
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT or \
       (REDDIT_CLIENT_ID and 'YOUR_REDDIT' in REDDIT_CLIENT_ID): # Check if placeholder is still there
        print("Error: Reddit API credentials not configured properly.")
        return None

    print(f"Fetching Reddit data for {symbol}...")
    try:
        reddit = praw.Reddit(client_id=REDDIT_CLIENT_ID,
                             client_secret=REDDIT_CLIENT_SECRET,
                             user_agent=REDDIT_USER_AGENT)

        # FIX: Initialize empty list
        all_posts = []
        search_query = f'"{symbol}"' # Search for exact symbol
        print(f"Searching Reddit for '{search_query}' in subreddits: {subreddits}...")
        for sub_name in subreddits:
            try:
                subreddit = reddit.subreddit(sub_name)
                posts = subreddit.search(search_query, limit=limit, sort='new', time_filter='day') # Limit search scope
                count = 0
                for post in posts:
                    # Check if symbol is in title/body (search can be broad)
                    if symbol.lower() in post.title.lower() or symbol.lower() in post.selftext.lower():
                         all_posts.append({
                             'datetime': datetime.fromtimestamp(post.created_utc),
                             'title': post.title,
                             'subreddit': sub_name,
                             'score': post.score,
                             'num_comments': post.num_comments
                         })
                         count += 1
                print(f" Found {count} relevant posts in r/{sub_name}.")
            except praw.exceptions.PRAWException as praw_e:
                 print(f" PRAW error accessing r/{sub_name}: {praw_e}")
            except Exception as e:
                 print(f" Error processing subreddit r/{sub_name}: {e}")
        if not all_posts:
            print(f"No recent Reddit posts found mentioning '{symbol}' in specified subreddits.")
            return pd.DataFrame(columns=['datetime', 'title', 'subreddit', 'score', 'num_comments'])
        reddit_df = pd.DataFrame(all_posts).sort_values(by='datetime', ascending=False)
        print(f"Successfully fetched {len(reddit_df)} total relevant Reddit posts.")
        return reddit_df
    except praw.exceptions.PRAWException as e:
        print(f"Error initializing PRAW or during Reddit API call: {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred fetching Reddit data: {e}")
        return None

# --- 3. Technical Analysis Function ---

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """ Calculates ADX, +DMI, -DMI, 9EMA on Close, and 9EMA on Volume. """
    if df is None or df.empty:
        print("Error: Input DataFrame for TA is None or empty.")
        return pd.DataFrame() # Return empty df
    required_cols = ['High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        print(f"Error: DataFrame must contain {required_cols}. Found: {df.columns}")
        return df # Return original df, subsequent steps might fail

    print("Calculating technical indicators...")
    df_ta = df.copy() # Work on a copy
    try:
        # Use pandas_ta to calculate indicators
        df_ta.ta.adx(append=True) # Calculates ADX_14, DMP_14, DMN_14
        df_ta.ta.ema(close='Close', length=9, append=True, col_names=('EMA_9_Close',))
        df_ta.ta.ema(close='Volume', length=9, append=True, col_names=('EMA_9_Volume',))

        # Rename columns for consistency
        rename_map = {'ADX_14': 'ADX', 'DMP_14': 'DMI_Plus', 'DMN_14': 'DMI_Minus'}
        df_ta.rename(columns={k: v for k, v in rename_map.items() if k in df_ta.columns}, inplace=True)

        print("Technical indicators calculated.")
        return df_ta

    except Exception as e:
        print(f"Error calculating technical indicators: {e}")
        # Return original df without new indicators in case of error
        return df

# --- 4. Sentiment Analysis Function ---

# Load sentiment analysis model globally ONCE on startup
try:
    print("Loading FinBERT sentiment analysis model...")
    sentiment_analyzer = pipeline("sentiment-analysis", model="ProsusAI/finbert")
    print("FinBERT model loaded successfully.")
except Exception as e:
    print(f"CRITICAL ERROR: Failed to load sentiment analysis model: {e}")
    print("Sentiment analysis will not be available.")
    sentiment_analyzer = None

def analyze_sentiment(text_data: pd.DataFrame | None, text_column: str) -> pd.DataFrame | None:
    """ Performs sentiment analysis on a DataFrame column using the pre-loaded model. """
    if sentiment_analyzer is None:
        print("Sentiment analyzer not available.")
        # Return input df but add empty columns so downstream code doesn't break
        if text_data is not None:
             text_data['sentiment_label'] = 'unavailable'
             text_data['sentiment_score'] = 0.0
        return text_data

    if text_data is None or text_column not in text_data.columns:
        print(f"Error: Input DataFrame for sentiment is None or missing column '{text_column}'.")
        return None

    if text_data.empty:
        print("Input DataFrame is empty, no sentiment to analyze.")
        text_data['sentiment_label'] = None
        text_data['sentiment_score'] = None
        return text_data

    texts = text_data[text_column].tolist()
    print(f"Analyzing sentiment for {len(texts)} items from column '{text_column}'...")
    # FIX: Initialize empty list
    results = []
    try:
        # Process texts, handling potential errors and long inputs
        for text in texts:
             max_length = 512 # FinBERT's typical max sequence length
             truncated_text = text[:max_length] if isinstance(text, str) and len(text) > max_length else text

             if not isinstance(truncated_text, str) or not truncated_text.strip():
                 results.append({'label': 'neutral', 'score': 0.0}) # Handle invalid input
                 continue

             try:
                 # The pipeline returns a list containing a dict, e.g., [{'label': 'positive', 'score': 0.9...}]
                 sentiment_result = sentiment_analyzer(truncated_text)
                 if sentiment_result and isinstance(sentiment_result, list):
                      results.append(sentiment_result[0]) 
                 else:
                      results.append({'label': 'error', 'score': 0.0})
             except Exception as analysis_err:
                 print(f" Error analyzing text chunk: '{str(truncated_text)[:50]}...': {analysis_err}")
                 results.append({'label': 'error', 'score': 0.0}) # Mark errors

        # Extract labels and scores
        sentiment_labels = [r.get('label', 'error') for r in results]
        # FIX: Initialize empty list
        sentiment_scores = []
        for r in results:
            score = r.get('score', 0.0)
            label = r.get('label', 'error').lower()
            if label == 'negative':
                sentiment_scores.append(-score)
            elif label == 'positive':
                sentiment_scores.append(score)
            else: # neutral or error
                sentiment_scores.append(0.0)

        text_data['sentiment_label'] = sentiment_labels
        text_data['sentiment_score'] = sentiment_scores
        print("Sentiment analysis complete.")
        return text_data

    except Exception as e:
        print(f"An unexpected error occurred during sentiment analysis batch processing: {e}")
        # Attempt to return original data with error columns
        text_data['sentiment_label'] = 'batch_error'
        text_data['sentiment_score'] = 0.0
        return text_data


def aggregate_sentiment(sentiment_df: pd.DataFrame | None) -> tuple[float, int]:
    """ Helper to safely aggregate sentiment scores from a DataFrame. """
    if sentiment_df is None or sentiment_df.empty or 'sentiment_score' not in sentiment_df.columns:
        return 0.0, 0
    # Ensure scores are numeric, replace NaN with 0 before averaging
    valid_scores = pd.to_numeric(sentiment_df['sentiment_score'], errors='coerce').fillna(0.0)
    if valid_scores.empty:
         return 0.0, 0
    avg_score = valid_scores.mean()
    count = len(sentiment_df) # Count includes items processed
    # Handle potential numpy types if avg_score is nan/inf
    if pd.isna(avg_score) or not np.isfinite(avg_score):
        avg_score = 0.0
    return float(avg_score), count

# --- 5. LLM Prediction Function ---

def predict_gap_with_llm(symbol: str, last_close: float, indicators: dict,
                         news_sentiment: float, reddit_sentiment: float,
                         news_count: int, reddit_count: int) -> str | None:
    """ Uses an LLM (via OpenAI API) to predict the next day opening gap direction. """
    if not OPENAI_API_KEY:
        print("Error: OPENAI_API_KEY environment variable not set.")
        return None

    print("Preparing prompt and querying LLM...")
    client = OpenAI(api_key=OPENAI_API_KEY)

    # Format indicators nicely, handling potential missing values ('N/A' or 0.0 from TA step)
    def format_indicator(k, v): # Pass key 'k' to check for Volume
        if isinstance(v, (int, float)):
            if 'Volume' in k: # Format volume differently
                 return f"{v:,.0f}"
            else:
                 return f"{v:.2f}"
        return "N/A" # If it's still 'N/A' or other non-numeric

    indicator_text = "\n".join([f"- {k}: {format_indicator(k, v)}" for k, v in indicators.items()])

    prompt = f"""
    Analyze the following financial data for stock symbol {symbol} to predict the direction of the opening gap for the next trading day (compared to yesterday's close of ${last_close:.2f}).

    Latest Technical Indicators:
    {indicator_text}

    Recent Sentiment Analysis (higher score is more positive, lower is more negative, range ~[-1, 1]):
    - Average News Sentiment (last 24h, {news_count} items): {news_sentiment:.2f}
    - Average Reddit Sentiment (last 24h, {reddit_count} posts): {reddit_sentiment:.2f}

    Based *only* on the provided technical and sentiment data, will the stock price likely gap UP or gap DOWN at the next market open compared to the last close of ${last_close:.2f}?

    Provide your prediction as a single word: UP, DOWN, or FLAT.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a financial analyst assistant providing stock gap predictions based on given data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1, 
            max_tokens=10,
            timeout=20
        )
        prediction = response.choices[0].message.content.strip().upper() 
        print(f"LLM raw response: {prediction}")

        valid_predictions = {"UP", "DOWN", "FLAT"}
        if prediction in valid_predictions:
            print(f"LLM Prediction for {symbol}: {prediction}")
            return prediction
        else:
            print(f"Warning: LLM returned unexpected response: '{prediction}'.")
            if "UP" in prediction: return "UP"
            if "DOWN" in prediction: return "DOWN"
            if "FLAT" in prediction or "NEUTRAL" in prediction: return "FLAT"
            return None # Could not interpret

    except Exception as e:
        print(f"Error querying LLM: {e}")
        return None


app = FastAPI(
    title="Stock Gap Predictor API",
    description="Predicts next-day stock opening gap direction using TA, Sentiment, and LLM.",
    version="0.1.1" 
)

class PredictionResponse(BaseModel):
    symbol: str
    prediction_input: dict 
    predicted_direction: str | None
    status: str
    error_message: str | None = None

@app.post("/predict/", response_model=PredictionResponse)
async def run_prediction_pipeline_endpoint(symbol: str = Query(..., description="Stock ticker symbol", example="AAPL")):
    """
    Endpoint to fetch data, run analysis, and predict the opening gap direction.
    """
    print(f"\n--- Received prediction request for {symbol} ---")
    status = "Processing"
    error_msg = None
    llm_prediction_result = None
    prediction_input_data = {"symbol": symbol} 

    try:

        stock_df = get_stock_data(symbol, period="45d")
        if stock_df is None or stock_df.empty:
            status = "Error"
            error_msg = f"Could not fetch stock data for {symbol}"
            raise HTTPException(status_code=404, detail=error_msg)

        if 'Close' not in stock_df.columns or stock_df['Close'].empty:
             status = "Error"
             error_msg = f"Missing 'Close' price data for {symbol}"
             raise HTTPException(status_code=500, detail=error_msg)

        last_close = stock_df['Close'].iloc[-1]
        prediction_input_data["last_close"] = last_close

        stock_df_ta = calculate_technical_indicators(stock_df) 
        required_indicators = ['ADX', 'DMI_Plus', 'DMI_Minus', 'EMA_9_Close', 'EMA_9_Volume']
        latest_indicators = {}
        missing_indicators = False
        for k in required_indicators:
            if k not in stock_df_ta.columns or stock_df_ta[k].isnull().all():
                latest_indicators[k] = 0.0 
                missing_indicators = True
                print(f"Warning: Indicator '{k}' is missing or all NaN for {symbol}. Defaulting to 0.0.")
            else:
                last_valid_idx = stock_df_ta[k].last_valid_index()
                if last_valid_idx is not None:
                    value = stock_df_ta.loc[last_valid_idx, k]
                    latest_indicators[k] = value.item() if isinstance(value, (np.generic)) else float(value)
                else:
                    latest_indicators[k] = 0.0 # Default if somehow still no valid index found
                    missing_indicators = True
                    print(f"Warning: Could not find last valid index for '{k}' for {symbol}. Defaulting to 0.0.")

        if missing_indicators:
            print(f"Proceeding with some indicators defaulted to 0.0 for {symbol}.")
        prediction_input_data["indicators"] = latest_indicators


        # 3. Fetch and Analyze News Sentiment (last 1 day)
        news_df = get_news_data(symbol, days_back=1)
        news_df_sentiment = analyze_sentiment(news_df, 'headline') # analyze_sentiment handles None input
        avg_news_sentiment, news_count = aggregate_sentiment(news_df_sentiment)
        prediction_input_data["news_sentiment_avg"] = avg_news_sentiment
        prediction_input_data["news_count"] = news_count

        # 4. Fetch and Analyze Reddit Sentiment (last 1 day, smaller limit for API)
        reddit_df = get_reddit_data(symbol, limit=25) # Reduced limit for real-time endpoint
        reddit_df_sentiment = analyze_sentiment(reddit_df, 'title') # analyze_sentiment handles None input
        avg_reddit_sentiment, reddit_count = aggregate_sentiment(reddit_df_sentiment)
        prediction_input_data["reddit_sentiment_avg"] = avg_reddit_sentiment
        prediction_input_data["reddit_count"] = reddit_count

        # 5. Get LLM Prediction
        llm_prediction_result = predict_gap_with_llm(
            symbol=symbol,
            last_close=last_close,
            indicators=latest_indicators,
            news_sentiment=avg_news_sentiment,
            reddit_sentiment=avg_reddit_sentiment,
            news_count=news_count,
            reddit_count=reddit_count
        )

        if llm_prediction_result:
            status = "Success"
        else:
            status = "Completed with errors"
            error_msg = "LLM prediction failed or returned invalid format."
            print(f"Warning: {error_msg} for {symbol}")


    except HTTPException as http_exc:
        # Re-raise HTTP exceptions from earlier stages
        raise http_exc
    except Exception as e:
        status = "Error"
        error_msg = f"An unexpected error occurred in the pipeline: {e}"
        print(f"CRITICAL ERROR for {symbol}: {error_msg}")
        # Return a 500 error for internal issues
        raise HTTPException(status_code=500, detail=error_msg)

    # Return the final response
    return PredictionResponse(
        symbol=symbol,
        prediction_input=prediction_input_data, # Include the data used for the prediction
        predicted_direction=llm_prediction_result,
        status=status,
        error_message=error_msg
    )

# --- 7. Running the Application ---

if __name__ == "__main__":
    import uvicorn
    print("\n--- Starting Stock Gap Predictor API (Corrected) ---")
    # Basic checks for essential API keys
    if not FINNHUB_API_KEY: print("Warning: FINNHUB_API_KEY not set.")
    if not REDDIT_CLIENT_ID or not REDDIT_CLIENT_SECRET or not REDDIT_USER_AGENT: print("Warning: Reddit API credentials not fully set.")
    if not OPENAI_API_KEY: print("Warning: OPENAI_API_KEY not set.")
    if sentiment_analyzer is None: print("Warning: Sentiment analysis model failed to load.")
    print("API will be available at http://127.0.0.1:8000")
    print("Access interactive documentation at http://127.0.0.1:8000/docs")
    print("------------------------------------------------------\n")


    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)

    