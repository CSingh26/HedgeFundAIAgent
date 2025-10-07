from openai import OpenAI
import os
from dotenv import load_dotenv

# ✅ Load .env file
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    print("⚠️ OPENAI_API_KEY missing — OpenAI requests will fail.")

client = OpenAI(api_key=OPENAI_API_KEY)

def hedge_fund_agents(symbol, market_data, news_data):
    """Combine insights from multiple legendary investor personas"""
    try:
        print(f"🤖 Running AI hedge fund analysis for {symbol}")

        context = {
            "symbol": symbol,
            "market_summary": str(market_data)[:5000],
            "news_summary": str(news_data)[:3000]
        }

        prompt = f"""
        You are a panel of hedge fund managers (Warren Buffett, Ray Dalio, Bill Ackman, Steve Cohen, and Charlie Munger).
        Evaluate the stock {symbol}.
        1. Discuss market trends and risk exposure.
        2. Assess recent news impact.
        3. Give a short-term and long-term outlook.
        4. Conclude with an investment decision and risk level.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial analyst and hedge fund manager."},
                {"role": "user", "content": prompt}
            ]
        )

        result = response.choices[0].message.content
        print(f"✅ Hedge fund analysis complete for {symbol}")
        return {"symbol": symbol, "analysis": result}

    except Exception as e:
        print(f"[ERROR] Hedge fund agent failed: {e}")
        return {"error": str(e)}

def risk_manager(analysis):
    """Run final risk assessment"""
    try:
        symbol = analysis.get("symbol")
        text = analysis.get("analysis", "")

        prompt = f"""
        Based on this analysis of {symbol}, estimate:
        - Risk rating (1–10)
        - Suggested portfolio allocation (%)
        - Recommended hedging strategy
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a risk manager analyzing portfolio exposure."},
                {"role": "user", "content": prompt + "\n" + text}
            ]
        )

        result = response.choices[0].message.content
        print(f"✅ Risk manager report complete for {symbol}")
        return {"symbol": symbol, "risk_report": result}

    except Exception as e:
        print(f"[ERROR] Risk manager failed: {e}")
        return {"error": str(e)}
