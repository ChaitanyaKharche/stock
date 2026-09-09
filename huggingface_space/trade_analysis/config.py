import os
from dotenv import load_dotenv

load_dotenv()

# API Keys
TWELVE_KEY = os.getenv("TWELVE_KEY")
FINNHUB_KEY = os.getenv("FINNHUB_KEY") 
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET")
REDDIT_USER_AGENT = os.getenv("REDDIT_USER_AGENT")
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "local")

if not FINNHUB_KEY or not REDDIT_CLIENT_ID:
    raise ImportError("CRITICAL: Required API keys (FINNHUB_KEY, REDDIT_CLIENT_ID) are not set in the environment.")

print("✅ Config module loaded. API keys and settings are ready.")
