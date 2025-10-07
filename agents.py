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
    """Combine insights from multiple legendary investor personas."""
    try:
        print(f"🤖 Running AI hedge fund analysis for {symbol}")

        prompt = f"""
        You are a panel of hedge fund managers (Warren Buffett, Ray Dalio, Bill Ackman, Steve Cohen, Charlie Munger).
        Evaluate the stock {symbol}.
        1) Market trends & risk exposure
        2) Recent news impact
        3) Short-term vs long-term outlook
        4) Final recommendation with a risk level (1-10) and brief rationale
        Use numbered structure and be concise.
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
    """Run final risk assessment (LLM)."""
    try:
        symbol = analysis.get("symbol")
        text = analysis.get("analysis", "")

        prompt = f"""
        Based on the following analysis for {symbol}, output:
        - Risk rating (1–10)
        - Suggested portfolio allocation (%) by conviction
        - Hedging idea (index hedge or options)
        Keep it short and structured.
        """

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a risk manager analyzing portfolio exposure."},
                {"role": "user", "content": prompt + "\n\n" + text}
            ]
        )

        result = response.choices[0].message.content
        print(f"✅ Risk manager report complete for {symbol}")
        return {"symbol": symbol, "risk_report": result}

    except Exception as e:
        print(f"[ERROR] Risk manager failed: {e}")
        return {"error": str(e)}
