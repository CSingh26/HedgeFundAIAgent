import requests
import datetime
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

if not POLYGON_API_KEY:
    print("⚠️ POLYGON_API_KEY missing. Polygon API calls will fail.")
if not NEWS_API_KEY:
    print("⚠️ NEWS_API_KEY missing. NewsAPI calls will fail.")


def get_stock_data(symbol):
    """Fetch historical stock data (year-to-date daily) from Polygon."""
    try:
        today = datetime.date.today()
        start_date = f"{today.year}-01-01"
        end_date = f"{today.year}-12-31"

        url = (
            f"https://api.polygon.io/v2/aggs/ticker/{symbol.upper()}/range/1/day/"
            f"{start_date}/{end_date}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
        )
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "results" not in data:
            print(f"[ERROR] No data found for {symbol}: {data}")
            return {"error": f"No data found for {symbol}"}

        print(f"✅ Polygon data fetched for {symbol}")
        return data

    except Exception as e:
        print(f"[ERROR] Request failed for {symbol}: {e}")
        return {"error": str(e)}


def get_news_data(symbol):
    """Fetch recent news for a given symbol via NewsAPI."""
    try:
        today = datetime.date.today()
        from_date = today - datetime.timedelta(days=7)

        url = (
            f"https://newsapi.org/v2/everything?q={symbol}&from={from_date}"
            f"&sortBy=publishedAt&apiKey={NEWS_API_KEY}"
        )
        response = requests.get(url, timeout=20)
        response.raise_for_status()
        data = response.json()

        if "articles" not in data:
            print(f"[ERROR] No news found for {symbol}: {data}")
            return {"error": f"No news found for {symbol}"}

        print(f"📰 News data fetched for {symbol}")
        return data

    except Exception as e:
        print(f"[ERROR] News request failed for {symbol}: {e}")
        return {"error": str(e)}
