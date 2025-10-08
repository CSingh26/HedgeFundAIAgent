# agents.py
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

SYSTEMS = {
    "Cohen":   "You are Steve Cohen. Multi-strategy, info edge, rapid execution, advanced risk management.",
    "Munger":  "You are Charlie Munger. Mental models, skepticism, moats, intrinsic value.",
    "Buffett": "You are Warren Buffett. Long-term value, quality businesses, moats, intrinsic value vs price.",
    "Ackman":  "You are Bill Ackman. High-conviction, activist catalysts, risk control.",
    "Dalio":   "You are Ray Dalio. Macro, systematic, risk parity, regimes.",
}

def _call_model(system, user):
    # Prompt shortened to encourage concise output
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user + "\n\nReturn 4–6 bullets total. Keep it under ~80 words."},
        ],
        temperature=0.35,
        max_tokens=300,
    )
    return resp.choices[0].message.content

def hedge_fund_agents(symbol, market, news):
    base = (
        f"Ticker: {symbol}\n"
        f"MARKET snapshot (sanitized): {market}\n"
        f"NEWS sample (24–72h): {news}\n"
        "Give only the MOST material points: near-term drivers, key risks, valuation/quality hints."
    )
    out = {}
    for name, sys in SYSTEMS.items():
        try:
            out[name] = _call_model(sys, base)
        except Exception as e:
            out[name] = f"[{name} agent error] {e}"
    return out

def risk_manager(panel_summary_text):
    sys = "You are a Chief Risk Officer. Be numeric, crisp, and brief."
    prompt = (
        "Synthesize into: Position size % range, Risk (1-10), Exp. return range, "
        "Stop/TP guides, Simple hedge idea. Max ~80 words.\n\n"
        f"PANEL:\n{panel_summary_text}"
    )
    try:
        return _call_model(sys, prompt)
    except Exception as e:
        return f"[Risk manager error] {e}"
