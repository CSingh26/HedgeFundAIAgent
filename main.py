from fastapi import FastAPI, Request
from telegram import Bot
import os
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ One-time startup log
print("\n🔍 Loaded Keys Summary:")
for key in ["TELEGRAM_TOKEN", "POLYGON_API_KEY", "NEWS_API_KEY", "OPENAI_API_KEY", "FMP_API_KEY"]:
    print(f"{key}: {'✅ Loaded' if os.getenv(key) else '❌ Missing'}")

# ✅ Initialize Telegram bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN missing in .env file")

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

from agents import hedge_fund_agents, risk_manager
from dataFetcher import get_stock_data, get_news_data
from advisor import advise_position
from reportFormatter import format_report, format_advice_output


@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip()

    if not text:
        await bot.send_message(
            chat_id=chat_id,
            text="⚠️ Please send a command.\nTry: `/analyze AAPL` or `/advise AAPL ENTRY=168.50 SHARES=120 RISK=balanced`",
            parse_mode="Markdown"
        )
        return {"status": "invalid"}

    upper = text.upper()

    # --- New: /ADVISE command ---
    if upper.startswith("/ADVISE"):
        try:
            parts = text.split()
            symbol = parts[1].upper()
            kv = {p.split("=", 1)[0].upper(): p.split("=", 1)[1] for p in parts[2:] if "=" in p}
            entry = float(kv.get("ENTRY"))
            shares = int(float(kv.get("SHARES")))
            risk = (kv.get("RISK", "balanced")).lower()

            await bot.send_message(chat_id=chat_id,
                                   text=f"🧮 Evaluating position {symbol} (entry ${entry}, shares {shares}, risk {risk})...")

            advice = advise_position(symbol, entry, shares, risk)
            chunks = format_advice_output(advice)
            for i, part in enumerate(chunks, start=1):
                await bot.send_message(chat_id=chat_id, text=f"({i}/{len(chunks)})\n{part}", parse_mode="Markdown")

            return {"status": "ok"}

        except Exception as e:
            await bot.send_message(
                chat_id=chat_id,
                text=("⚠️ Usage: `/advise TICKER ENTRY=123.45 SHARES=100 RISK=conservative|balanced|aggressive`\n"
                      f"Error: {e}"),
                parse_mode="Markdown"
            )
            return {"status": "bad_params"}

    # --- Existing: /ANALYZE command ---
    if upper.startswith("/ANALYZE"):
        parts = text.split(" ", 1)
        symbol = parts[1].strip().upper() if len(parts) > 1 else None
        if not symbol:
            await bot.send_message(chat_id=chat_id, text="⚠️ Please provide a ticker. Example: `/analyze AAPL`",
                                   parse_mode="Markdown")
            return {"status": "invalid"}

        await bot.send_message(chat_id=chat_id, text=f"⏳ Running quarterly analysis for {symbol}...")

        market = get_stock_data(symbol)
        news = get_news_data(symbol)
        if "error" in market or "error" in news:
            await bot.send_message(chat_id=chat_id, text="⚠️ Data fetch failed. Try another ticker or check API keys.")
            return {"status": "fetch_failed"}

        analysis = hedge_fund_agents(symbol, market, news)
        riskrpt = risk_manager(analysis)
        report_chunks = format_report(analysis, riskrpt)

        if isinstance(report_chunks, list):
            for i, part in enumerate(report_chunks, start=1):
                await bot.send_message(chat_id=chat_id, text=f"({i}/{len(report_chunks)})\n{part}", parse_mode="Markdown")
        else:
            await bot.send_message(chat_id=chat_id, text=str(report_chunks)[:4000], parse_mode="Markdown")

        print(f"✅ Completed analysis for {symbol}")
        return {"status": "ok"}

    # Fallback: help text
    await bot.send_message(
        chat_id=chat_id,
        text=(
            "👋 Try:\n"
            "• `/analyze AAPL` — quarterly research report\n"
            "• `/advise AAPL ENTRY=168.50 SHARES=120 RISK=balanced` — Diversify/Hold/Exit with dollar allocations"
        ),
        parse_mode="Markdown"
    )
    return {"status": "help"}


@app.get("/")
async def home():
    return {"status": "running", "message": "AI Hedge Fund Agent is online"}
