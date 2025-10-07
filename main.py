from fastapi import FastAPI, Request
from telegram import Bot
import os
import asyncio
from dotenv import load_dotenv

# ✅ Load environment variables
load_dotenv()

# ✅ One-time startup log
print("\n🔍 Loaded Keys Summary:")
for key in ["TELEGRAM_TOKEN", "POLYGON_API_KEY", "NEWS_API_KEY", "OPENAI_API_KEY"]:
    print(f"{key}: {'✅ Loaded' if os.getenv(key) else '❌ Missing'}")

# ✅ Initialize Telegram bot
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
if not TELEGRAM_TOKEN:
    raise ValueError("❌ TELEGRAM_TOKEN missing in .env file")

bot = Bot(token=TELEGRAM_TOKEN)
app = FastAPI()

from agents import hedge_fund_agents, risk_manager
from dataFetcher import get_stock_data, get_news_data
from reportFormatter import format_report


@app.post("/webhook")
async def telegram_webhook(request: Request):
    data = await request.json()
    message = data.get("message", {})
    chat_id = message.get("chat", {}).get("id")
    text = message.get("text", "").strip().upper()

    # 🧹 Clean Telegram command prefix (e.g., "/analyze AAPL" -> "AAPL")
    if text.startswith("/ANALYZE"):
        parts = text.split(" ", 1)
        text = parts[1].strip().upper() if len(parts) > 1 else None

    if not text:
        await bot.send_message(
            chat_id=chat_id,
            text="⚠️ Please provide a valid ticker symbol (e.g. AAPL, TSLA)."
        )
        return {"status": "invalid"}

    await bot.send_message(chat_id=chat_id, text=f"⏳ Running quarterly analysis for {text}...")

    try:
        market = get_stock_data(text)
        news = get_news_data(text)
        if "error" in market or "error" in news:
            raise Exception("API call failed")

        # 🧠 Run agents
        analysis = hedge_fund_agents(text, market, news)
        risk = risk_manager(analysis)

        # 🧾 Generate and send formatted report (handles long messages automatically)
        report = format_report(analysis, risk)

        if isinstance(report, list):
            for i, part in enumerate(report, start=1):
                await bot.send_message(
                    chat_id=chat_id,
                    text=f"({i}/{len(report)})\n{part[:4000]}",
                    parse_mode="Markdown"
                )
        else:
            await bot.send_message(
                chat_id=chat_id,
                text=report[:4000],
                parse_mode="Markdown"
            )

        print(f"✅ Completed analysis for {text}")
        return {"status": "ok"}

    except Exception as e:
        print(f"[ERROR] {e}")
        await bot.send_message(chat_id=chat_id, text=f"❌ Error occurred: {e}")
        return {"status": "error"}


@app.get("/")
async def home():
    return {"status": "running", "message": "AI Hedge Fund Agent is online"}
