# dataFetcher.py
import os
import datetime as dt
import requests
from dotenv import load_dotenv

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

def _safe_get_json(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[dataFetcher] GET failed: {url} -> {e}")
        return {"error": str(e)}

def get_stock_data(symbol: str):
    sym = symbol.upper()
    if not POLYGON_API_KEY:
        return {"error": "Missing POLYGON_API_KEY"}

    end = dt.date.today()
    start = end - dt.timedelta(days=365)
    url = (
        f"https://api.polygon.io/v2/aggs/ticker/{sym}/range/1/day/"
        f"{start.isoformat()}/{end.isoformat()}?adjusted=true&sort=asc&apiKey={POLYGON_API_KEY}"
    )
    data = _safe_get_json(url)
    if "error" in data:
        return data
    try:
        results = data.get("results", [])
        if not results:
            return {"note": "no polygon results", "raw": data}
        closes = [r.get("c") for r in results if r.get("c") is not None]
        highs  = [r.get("h") for r in results if r.get("h") is not None]
        lows   = [r.get("l") for r in results if r.get("l") is not None]
        volume = [r.get("v") for r in results if r.get("v") is not None]
        snap = dict(
            last_close=closes[-1],
            ytd_return=round((closes[-1] / closes[0] - 1.0) * 100, 2) if closes else None,
            high_52w = max(highs) if highs else None,
            low_52w  = min(lows) if lows else None,
            avg_vol_30d = int(sum(volume[-30:]) / max(1, len(volume[-30:]))) if volume else None,
            count=len(results)
        )
        return snap
    except Exception as e:
        return {"error": f"parse error: {e}", "raw": data}

def get_news_data(symbol: str):
    if not NEWS_API_KEY:
        return {"error": "Missing NEWS_API_KEY"}

    today = dt.datetime.utcnow()
    from_date = (today - dt.timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    q = symbol.upper()
    url = (
        "https://newsapi.org/v2/everything?"
        f"q={q}&from={from_date}&sortBy=publishedAt&language=en&pageSize=6&apiKey={NEWS_API_KEY}"
    )
    data = _safe_get_json(url)
    if "error" in data:
        return data
    try:
        arts = data.get("articles", [])[:6]
        clean = []
        for a in arts:
            clean.append({
                "title": a.get("title"),
                "source": a.get("source", {}).get("name"),
                "publishedAt": a.get("publishedAt"),
                "desc": a.get("description")
            })
        return {"sample": clean, "count": len(clean)}
    except Exception as e:
        return {"error": f"news parse error: {e}", "raw": data}
