# main.py
import os
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from telegram import Bot

from agents import hedge_fund_agents, risk_manager
from dataFetcher import get_stock_data, get_news_data
from reportFormatter import format_report, format_advice_output
from advisor import advise_position

load_dotenv()

print("\n🔍 Loaded Keys Summary:")
for key in ["TELEGRAM_TOKEN", "POLYGON_API_KEY", "NEWS_API_KEY", "OPENAI_API_KEY", "FMP_API_KEY"]:
    print(f"{key}: {'✅ Loaded' if os.getenv(key) else '❌ Missing'}")

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN missing in .env file")

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

def _clean_command(text: str):
    t = (text or "").strip()
    if not t:
        return None, None, {}
    if t.startswith("/"):
        parts = t[1:].split()
        cmd = parts[0].upper()
        sym = parts[1].upper() if len(parts) > 1 else None
        args = {}
        for tok in parts[2:]:
            if "=" in tok:
                k, v = tok.split("=", 1)
                args[k.strip().upper()] = v.strip()
        return cmd, sym, args
    return None, t.upper(), {}

async def _send_chunks(chat_id, chunks):
    # Only send up to 2 chunks, per requirement
    for c in chunks[:2]:
        await bot.send_message(chat_id=chat_id, text=c, parse_mode="Markdown")

@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "")

    cmd, symbol, args = _clean_command(text)

    if not symbol:
        await bot.send_message(
            chat_id=chat_id,
            text="⚠️ Send a ticker.\n/analyze AAPL\n/advise AAPL ENTRY=168.5 SHARES=120 RISK=balanced",
        )
        return {"status": "invalid"}

    try:
        if cmd in (None, "ANALYZE"):
            await bot.send_message(chat_id=chat_id, text=f"⏳ Analyzing {symbol}…")
            market = get_stock_data(symbol)
            news = get_news_data(symbol)
            if "error" in market or "error" in news:
                await bot.send_message(chat_id=chat_id, text="⚠️ Some upstream data failed; summarizing with what's available.")
            panel = hedge_fund_agents(symbol, market, news)
            panel_concat = " ".join([f"{k}: {v}" for k, v in panel.items()])
            risk = risk_manager(panel_concat)
            chunks = format_report(panel, risk, symbol)
            await _send_chunks(chat_id, chunks)
            return {"status": "ok"}

        if cmd == "ADVISE":
            entry = float(args.get("ENTRY")) if args.get("ENTRY") else None
            shares = int(float(args.get("SHARES"))) if args.get("SHARES") else None
            risk = (args.get("RISK") or "balanced").lower()
            if entry is None or shares is None:
                await bot.send_message(chat_id=chat_id, text="⚠️ Usage: /advise TICKER ENTRY=123.45 SHARES=100 RISK=balanced")
                return {"status": "invalid"}
            await bot.send_message(chat_id=chat_id, text=f"🤖 Evaluating {symbol} position…")
            rec = advise_position(symbol, entry, shares, risk)
            chunks = format_advice_output(rec)
            await _send_chunks(chat_id, chunks)
            return {"status": "ok"}

        # Unknown command -> default to analyze
        await bot.send_message(chat_id=chat_id, text=f"ℹ️ Unknown command `{cmd}`. Running /analyze {symbol}.")
        market = get_stock_data(symbol)
        news = get_news_data(symbol)
        panel = hedge_fund_agents(symbol, market, news)
        panel_concat = " ".join([f"{k}: {v}" for k, v in panel.items()])
        risk = risk_manager(panel_concat)
        chunks = format_report(panel, risk, symbol)
        await _send_chunks(chat_id, chunks)
        return {"status": "ok"}

    except Exception as e:
        print(f"[ERROR] {e}")
        await bot.send_message(chat_id=chat_id, text=f"❌ Error: {e}")
        return {"status": "error"}

@app.get("/")
async def home():
    return {"status": "running", "message": "AI Hedge Fund Agent is online"}
