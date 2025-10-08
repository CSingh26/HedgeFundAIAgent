# advisor.py
import os
import time
from collections import defaultdict
from math import sqrt

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from scipy.stats import norm  # pip install scipy

load_dotenv()

POLYGON_API_KEY = os.getenv("POLYGON_API_KEY")
FMP_API_KEY     = os.getenv("FMP_API_KEY")

# =========================
# Helpers
# =========================
def _safe_get_json(url):
    try:
        r = requests.get(url, timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[advisor] GET failed: {url} -> {e}")
        return {"error": str(e)}

def chunked(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

# =========================
# Universe providers
# =========================
def get_sp500_list():
    # Try FMP
    if FMP_API_KEY:
        url = f"https://financialmodelingprep.com/api/v3/sp500_constituent?apikey={FMP_API_KEY}"
        data = _safe_get_json(url)
        if isinstance(data, list) and data and "error" not in data:
            t = [x.get("symbol") for x in data if x.get("symbol")]
            if t:
                return t
    # Fallback Slickcharts
    try:
        html = requests.get("https://www.slickcharts.com/sp500", timeout=20).text
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table", {"class": "table"})
        tickers = []
        for row in table.tbody.find_all("tr"):
            t = row.find_all("td")[2].get_text(strip=True)
            tickers.append(t)
        tickers = list(dict.fromkeys(tickers))[:300]
        return tickers
    except Exception as e:
        print(f"[advisor] Slickcharts fallback failed: {e}")
        return ["AAPL","MSFT","NVDA","GOOGL","AMZN","META","BRK.B","UNH","XOM","JPM","V","HD","LLY","PG"]

# =========================
# Fundamentals (optional)
# =========================
def get_profile_fmp(symbol):
    if not FMP_API_KEY:
        return {}
    url = f"https://financialmodelingprep.com/api/v3/profile/{symbol.upper()}?apikey={FMP_API_KEY}"
    data = _safe_get_json(url)
    return data[0] if isinstance(data, list) and data else {}

def get_ratios_fmp(symbol):
    if not FMP_API_KEY:
        return {}
    url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{symbol.upper()}?apikey={FMP_API_KEY}"
    data = _safe_get_json(url)
    return data[0] if isinstance(data, list) and data else {}

def get_profiles_fmp_batch(symbols):
    if not FMP_API_KEY or not symbols:
        return {}
    out = {}
    try:
        for batch in chunked(symbols, 50):
            url = f"https://financialmodelingprep.com/api/v3/profile/{','.join(batch)}?apikey={FMP_API_KEY}"
            data = _safe_get_json(url)
            if isinstance(data, list):
                for row in data:
                    sym = row.get("symbol")
                    if sym:
                        out[sym] = row
            time.sleep(0.2)
    except Exception as e:
        print(f"[advisor] FMP profiles batch failed: {e}")
    return out

def get_ratios_fmp_batch(symbols):
    if not FMP_API_KEY or not symbols:
        return {}
    out = {}
    try:
        for batch in chunked(symbols, 50):
            url = f"https://financialmodelingprep.com/api/v3/ratios-ttm/{','.join(batch)}?apikey={FMP_API_KEY}"
            data = _safe_get_json(url)
            if isinstance(data, list):
                for row in data:
                    sym = row.get("symbol")
                    if sym and sym not in out:
                        out[sym] = row
            time.sleep(0.2)
    except Exception as e:
        print(f"[advisor] FMP ratios batch failed: {e}")
    return out

# =========================
# Prices & OHLCV
# =========================
def get_last_price(symbol):
    """
    Robust last price in USD terms where possible:
    1) Polygon previous close
    2) FMP quote-short
    3) yfinance last close
    """
    sym = symbol.upper()

    # Polygon prev
    try:
        if POLYGON_API_KEY:
            url_prev = f"https://api.polygon.io/v2/aggs/ticker/{sym}/prev?adjusted=true&apiKey={POLYGON_API_KEY}"
            data2 = _safe_get_json(url_prev)
            if isinstance(data2, dict) and data2.get("results"):
                c = data2["results"][0].get("c")
                if c:
                    price = float(c)
                    print(f"[advisor] Price for {sym} from Polygon prev: {price}")
                    return price
    except Exception as e:
        print(f"[advisor] Polygon prev failed for {sym}: {e}")

    # FMP quote-short
    try:
        if FMP_API_KEY:
            url_fmp = f"https://financialmodelingprep.com/api/v3/quote-short/{sym}?apikey={FMP_API_KEY}"
            d = _safe_get_json(url_fmp)
            if isinstance(d, list) and d and d[0].get("price"):
                price = float(d[0]["price"])
                print(f"[advisor] Price for {sym} from FMP: {price}")
                return price
    except Exception as e:
        print(f"[advisor] FMP quote failed for {sym}: {e}")

    # yfinance last close
    try:
        y = yf.Ticker(sym).history(period="2d", auto_adjust=False)
        if not y.empty:
            price = float(y["Close"].iloc[-1])
            print(f"[advisor] Price for {sym} from yfinance: {price}")
            return price
    except Exception as e:
        print(f"[advisor] yfinance last close failed for {sym}: {e}")

    return None

def fetch_hist_ohlcv(tickers, period="1y"):
    if isinstance(tickers, str):
        tickers = [tickers]

    df = yf.download(
        tickers,
        period=period,
        interval="1d",
        progress=False,
        auto_adjust=False,
        threads=True
    )
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        close = df["Adj Close"] if "Adj Close" in df.columns.levels[0] else df["Close"]
        vol   = df["Volume"]
    else:
        close = df[["Adj Close"]] if "Adj Close" in df.columns else df[["Close"]]
        vol   = df[["Volume"]]
        close.columns = [tickers[0]]
        vol.columns   = [tickers[0]]

    return close.dropna(how="all"), vol.dropna(how="all")

# =========================
# Risk / Signals
# =========================
def rolling_vol(series: pd.Series):
    r = series.pct_change().dropna()
    return (r.rolling(21).std() * (252**0.5)).iloc[-1]

def inverse_vol_weights(df: pd.DataFrame):
    vols = df.apply(rolling_vol)
    inv = 1.0 / vols.replace(0, np.nan)
    w = inv / inv.sum()
    return w.fillna(0.0)

def risk_budget(target_risk: str):
    target_risk = (target_risk or "balanced").lower()
    if target_risk in ["conservative", "low"]:
        return dict(max_pos=0.10, hedge_ratio=0.25)
    if target_risk in ["aggressive", "high"]:
        return dict(max_pos=0.25, hedge_ratio=0.05)
    return dict(max_pos=0.15, hedge_ratio=0.15)

def compute_momentum_vol(prices_df: pd.DataFrame):
    stats = {}
    if prices_df is None or prices_df.empty:
        return stats
    rets = prices_df.pct_change().dropna()
    ann_vol = rets.rolling(21).std().iloc[-1] * (252**0.5)
    for col in prices_df.columns:
        s = prices_df[col].dropna()
        if s.empty:
            continue
        def window_ret(days):
            if len(s) <= days:
                return None
            return float(s.iloc[-1] / s.iloc[-days-1] - 1.0)
        stats[col] = dict(
            mom_3m = window_ret(63),
            mom_6m = window_ret(126),
            mom_12m= window_ret(252) if len(s) > 252 else None,
            vol = float(ann_vol.get(col, np.nan))
        )
    return stats

def decide_action(pnl_pct, momentum_3m, drawdown_recent):
    if pnl_pct >= 0.25 and (momentum_3m < 0.0 or drawdown_recent < -0.10):
        return "DIVERSIFY"
    if pnl_pct <= -0.20 or (momentum_3m < -0.05 and drawdown_recent < -0.12):
        return "EXIT"
    return "HOLD"

# =========================
# Forecasting & Ratings
# =========================
def compute_return_stats(price_series: pd.Series):
    s = price_series.dropna()
    rets = np.log(s / s.shift(1)).dropna()
    if len(rets) < 30:
        mu_d = rets.mean() if len(rets) else 0.0003
        sig_d = rets.std(ddof=1) if len(rets) else 0.015
    else:
        mu_d = rets.mean()
        sig_d = rets.std(ddof=1)
    mu_ann = float(mu_d * 252)
    sigma_ann = float(sig_d * sqrt(252))
    mu_63d = float(mu_d * 63)
    sigma_63d = float(sig_d * sqrt(63))
    return dict(mu_ann=mu_ann, sigma_ann=sigma_ann, mu_63d=mu_63d, sigma_63d=sigma_63d)

def forecast_range(last_price: float, mu_ann: float, sigma_ann: float, days: int, p_low=0.10, p_high=0.90):
    T = days / 252.0
    mu = mu_ann
    sigma = sigma_ann
    mid = last_price * float(np.exp(mu * T))
    z_low = norm.ppf(p_low); z_high = norm.ppf(p_high)
    drift = (mu - 0.5 * sigma * sigma) * T
    volterm = sigma * sqrt(T)
    low = last_price * float(np.exp(drift + volterm * z_low))
    high = last_price * float(np.exp(drift + volterm * z_high))
    return dict(low=round(low, 2), mid=round(mid, 2), high=round(high, 2))

def projections_bull_bear(last_price: float, stats: dict):
    mu = stats["mu_ann"]; sig = stats["sigma_ann"]
    mu_bull = mu + 0.5 * sig; sig_bull = max(0.0001, sig * 0.9)
    mu_bear = mu - 0.5 * sig; sig_bear = sig * 1.1
    return {
        "3m": {
            "bull": forecast_range(last_price, mu_bull, sig_bull, 63),
            "bear": forecast_range(last_price, mu_bear, sig_bear, 63),
        },
        "6m": {
            "bull": forecast_range(last_price, mu_bull, sig_bull, 126),
            "bear": forecast_range(last_price, mu_bear, sig_bear, 126),
        }
    }

def horizon_rating(stats: dict, mom_3m: float, drawdown_min: float, risk_level: str):
    mu = stats["mu_ann"]
    sig = max(1e-6, stats["sigma_ann"])
    sharpe_like = mu / sig
    mom = mom_3m
    dd_penalty = drawdown_min
    score = 0.6 * sharpe_like + 0.3 * (mom * 5) + 0.1 * (dd_penalty * 5)
    r = (risk_level or "balanced").lower()
    if r == "conservative":
        score -= 0.15 * max(0, sig - 0.25)
    elif r == "aggressive":
        score += 0.10 * (sig)

    def bucket(x):
        if x >= 1.0: return ("Optimal", "High conviction; favorable reward vs. risk.")
        if x >= 0.3: return ("Reasonable", "Acceptable; be selective on entries.")
        return ("Avoid", "Unfavorable risk-adjusted outlook.")

    q_score = score + 0.1 * mom
    s_score = score + 0.05 * sharpe_like
    q_label, q_note = bucket(q_score)
    s_label, s_note = bucket(s_score)
    return dict(
        quarterly=dict(label=q_label, note=q_note, score=round(q_score, 2)),
        six_month=dict(label=s_label, note=s_note, score=round(s_score, 2)),
        metrics=dict(sharpe_like=round(sharpe_like, 2), ann_vol=round(sig, 2), ann_mu=round(mu, 2))
    )

# =========================
# Dynamic screening
# =========================
def screen_candidates(current_symbol, risk_level="balanced", max_names=10, per_sector_cap=2):
    universe = get_sp500_list()
    universe = [u for u in universe if u.upper() != current_symbol.upper()]
    if not universe:
        return []

    prof = get_profiles_fmp_batch(universe)   # may be {}
    ratios = get_ratios_fmp_batch(universe)   # may be {}

    universe = universe[:250]
    close, vol = fetch_hist_ohlcv(universe, period="1y")
    if close.empty or vol.empty:
        return []

    last_px = close.ffill().iloc[-1]
    avg_vol = vol.rolling(30).mean().iloc[-1]
    adv = (avg_vol * last_px).dropna()

    min_adv = 50e6 if risk_level != "aggressive" else 20e6
    liquid = adv[adv >= min_adv].index.tolist()
    if not liquid:
        liquid = adv.sort_values(ascending=False).head(100).index.tolist()

    prices = close[liquid].dropna(axis=1, how="all")
    if prices.empty:
        return []

    stats = compute_momentum_vol(prices)

    keep = []
    for sym in prices.columns:
        s = stats.get(sym, {})
        m6 = s.get("mom_6m"); m12 = s.get("mom_12m")
        if m6 is None and m12 is None:
            continue
        if risk_level == "conservative":
            if (m6 is not None and m6 > 0.02) or (m12 is not None and m12 > 0.04):
                keep.append(sym)
        elif risk_level == "aggressive":
            if (m6 is not None and m6 > -0.05) or (m12 is not None and m12 > 0.0):
                keep.append(sym)
        else:
            if (m6 is not None and m6 > 0.0) or (m12 is not None and m12 > 0.02):
                keep.append(sym)

    if not keep:
        keep = list(prices.columns)[:80]

    rows = []
    for sym in keep:
        s = stats.get(sym, {})
        p = prof.get(sym, {})
        r = ratios.get(sym, {})
        mcap = p.get("mktCap") or p.get("marketCap") or 0
        m6 = s.get("mom_6m") or 0.0
        m12 = s.get("mom_12m") or 0.0
        vol_ = s.get("vol"); vol_pen = 0.0 if vol_ is None or np.isnan(vol_) else float(vol_)
        pe = r.get("priceEarningsRatioTTM"); margin = r.get("netProfitMarginTTM")
        q_bonus = 0.0
        if pe and pe > 0 and pe < 40: q_bonus += 0.1
        if margin and margin > 0.10:  q_bonus += 0.1
        mom = 0.6*m6 + 0.4*m12
        size = np.log1p(mcap) if mcap else 0.0
        score = 2.0*mom + 0.35*size - 0.75*vol_pen + q_bonus
        sector = p.get("sector", "N/A")
        rows.append((sym, score, sector))

    rows.sort(key=lambda x: x[1], reverse=True)

    sector_count = defaultdict(int)
    selected = []
    for sym, score, sector in rows:
        if sector_count[sector] >= per_sector_cap:
            continue
        selected.append(sym)
        sector_count[sector] += 1
        if len(selected) >= max_names:
            break

    return selected

# =========================
# Main entrypoint for /advise
# =========================
def advise_position(symbol, entry_price, shares, risk_level="balanced"):
    symbol = symbol.upper()
    last = get_last_price(symbol)
    if last is None:
        raise ValueError(f"Could not fetch a last price for {symbol}. Check APIs/keys.")

    position_value = last * shares
    pnl = (last - entry_price) * shares
    pnl_pct = (last / entry_price - 1.0)

    close, _ = fetch_hist_ohlcv([symbol], period="1y")
    s = close[symbol].dropna() if symbol in close.columns else pd.Series(dtype=float)
    if s.empty:
        s = yf.Ticker(symbol).history(period="1y", auto_adjust=False)["Close"]

    mom_3m = s.iloc[-1] / s.iloc[-63] - 1.0 if len(s) > 63 else 0.0
    rolling_max = s.cummax()
    dd = (s / rolling_max - 1.0).iloc[-63:].min() if len(s) > 63 else (s / s.cummax() - 1.0).min()

    profile = get_profile_fmp(symbol)
    ratios  = get_ratios_fmp(symbol)
    sector  = profile.get("sector", "N/A") if profile else "N/A"

    stats = compute_return_stats(s)
    projections = projections_bull_bear(last, stats)
    rating = horizon_rating(stats, mom_3m, dd, risk_level)

    action = decide_action(pnl_pct, mom_3m, dd)
    rb = risk_budget(risk_level)

    recommendation = {
        "symbol": symbol,
        "last_price": round(last, 4),
        "entry_price": round(entry_price, 4),
        "shares": shares,
        "position_value": round(position_value, 2),
        "pnl": round(pnl, 2),
        "pnl_pct": round(pnl_pct * 100, 2),
        "risk_level": risk_level,
        "sector": sector,
        "action": action,
        "notes": {},
        "projections": projections,
        "rating": rating,
    }

    if action in ["DIVERSIFY", "EXIT"]:
        take_out_value = position_value * (0.30 if action == "DIVERSIFY" else 1.00)
        candidates = screen_candidates(symbol, risk_level=risk_level, max_names=10, per_sector_cap=2)
        if not candidates:
            candidates = ["AAPL","MSFT","NVDA","GOOGL","AMZN"]
        candidates = [c for c in candidates if c.upper() != symbol.upper()]

        close_cand, _ = fetch_hist_ohlcv(candidates, period="1y")
        prices = close_cand.dropna(axis=1, how="all")
        if prices.empty:
            close_cand, _ = fetch_hist_ohlcv(["AAPL","MSFT","NVDA","GOOGL","AMZN"], period="1y")
            prices = close_cand

        w = inverse_vol_weights(prices)
        w = w.clip(upper=rb["max_pos"]).replace([np.inf, -np.inf], 0.0)
        w = w / w.sum()

        allocs = (w * take_out_value).round(2).sort_values(ascending=False)
        recommendation["reallocation"] = {
            "amount_to_redeploy": float(take_out_value),
            "weights": {k: float(v) for k, v in w.sort_values(ascending=False).to_dict().items()},
            "dollar_allocations": {k: float(v) for k, v in allocs.to_dict().items()},
        }
        recommendation["notes"]["hedge"] = (
            f"Consider {int(rb['hedge_ratio']*100)}% notional index hedge "
            f"(e.g., inverse ETF or short index futures) or vol targeting."
        )
    else:
        rets = s.pct_change().dropna()
        vol_now = rets.rolling(14).std().iloc[-1] if len(rets) > 14 else (rets.std() if not rets.empty else 0.02)
        trail = max(0.08, min(0.20, float(vol_now) * 3.0))
        recommendation["notes"]["hold_params"] = {
            "suggested_trailing_stop_pct": round(trail * 100, 2),
            "rationale": "Momentum steady; hold with a disciplined trailing stop."
        }

    recommendation["rationale"] = {
        "momentum_3m": round(mom_3m * 100, 2),
        "recent_drawdown_min": round(dd * 100, 2),
        "sector": sector,
        "quality_hint": {
            "pe": ratios.get("priceEarningsRatioTTM") if ratios else None,
            "profit_margin": ratios.get("netProfitMarginTTM") if ratios else None,
        },
        "risk_metrics": rating.get("metrics", {}),
    }

    return recommendation
