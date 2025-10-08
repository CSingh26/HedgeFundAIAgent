# reportFormatter.py
from advisor import (
    fetch_hist_ohlcv, compute_return_stats, projections_bull_bear,
    horizon_rating, get_last_price
)

# Hard cap: max 2 messages. Keep each under ~1900 chars to be safe in Markdown.
TELEGRAM_LIMIT = 1900
MAX_MESSAGES = 2

def _fmt_usd(x):
    try:
        return f"${float(x):,.2f} USD"
    except Exception:
        return f"${x} USD"

def _truncate(s, n):
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"

def _split_for_telegram(s: str):
    # Try to split on paragraph or sentence, but cap to MAX_MESSAGES
    if len(s) <= TELEGRAM_LIMIT:
        return [s]
    # Two-part split
    cut = s.rfind("\n\n", 0, TELEGRAM_LIMIT)
    if cut == -1:
        cut = max(s.rfind(". ", 0, TELEGRAM_LIMIT), s.rfind("! ", 0, TELEGRAM_LIMIT), s.rfind("? ", 0, TELEGRAM_LIMIT))
        if cut == -1:
            cut = TELEGRAM_LIMIT
    part1 = s[:cut].strip()
    part2 = s[cut:].lstrip()
    # Ensure second fits
    if len(part2) > TELEGRAM_LIMIT:
        part2 = _truncate(part2, TELEGRAM_LIMIT)
    return [part1, part2][:MAX_MESSAGES]

# ============= Short /analyze =============
def _shorten_agent_text(txt, max_len=220):
    # keep only first ~220 chars to compress
    return _truncate(txt.replace("\n", " ").strip(), max_len)

def format_report(agents_output: dict, risk_summary: str, symbol: str):
    # --- Agents, but very short ---
    lines = []
    lines.append(f"📊 **Research Snapshot — {symbol}**")
    for name in ["Buffett", "Munger", "Ackman", "Dalio", "Cohen"]:
        t = agents_output.get(name)
        if t:
            lines.append(f"- **{name}:** {_shorten_agent_text(t)}")

    # --- Risk synthesis (short) ---
    if risk_summary:
        lines.append(f"\n🛡️ **Risk Synthesis:** {_truncate(risk_summary.replace(chr(10), ' '), 300)}")

    # --- Projections & ratings (short) ---
    try:
        last = get_last_price(symbol)
        if last:
            close, _ = fetch_hist_ohlcv([symbol], period="1y")
            s = close[symbol].dropna() if symbol in close.columns else None
            if s is not None and not s.empty:
                stats = compute_return_stats(s)
                proj = projections_bull_bear(last, stats)
                mom_3m = s.iloc[-1] / s.iloc[-63] - 1.0 if len(s) > 63 else 0.0
                rolling_max = s.cummax()
                dd = (s / rolling_max - 1.0).iloc[-63:].min() if len(s) > 63 else (s / s.cummax() - 1.0).min()
                rating = horizon_rating(stats, mom_3m, dd, "balanced")
                p3b, p3r = proj["3m"]["bull"], proj["3m"]["bear"]
                lines.append(
                    f"\n📈 **3m Bull/Bear:** "
                    f"**Bull** L {_fmt_usd(p3b['low'])} / M {_fmt_usd(p3b['mid'])} / H {_fmt_usd(p3b['high'])} | "
                    f"**Bear** L {_fmt_usd(p3r['low'])} / M {_fmt_usd(p3r['mid'])} / H {_fmt_usd(p3r['high'])}"
                )
                q = rating.get("quarterly", {}); s6 = rating.get("six_month", {})
                lines.append(f"🧮 **Ratings:** 3m: {q.get('label','?')} (score {q.get('score','?')}) | 6m: {s6.get('label','?')} (score {s6.get('score','?')})")
    except Exception as e:
        lines.append(f"_Projection error: {e}_")

    body = "\n".join(lines)
    chunks = _split_for_telegram(body)
    # Prepend part counters only if 2 chunks
    if len(chunks) == 2:
        chunks[0] = "(1/2)\n" + chunks[0]
        chunks[1] = "(2/2)\n" + chunks[1]
    return chunks

# ============= Short /advise =============
def format_advice_output(advice: dict):
    if not advice or "symbol" not in advice:
        return ["⚠️ Could not generate advice."]

    # Header with forced USD formatting everywhere
    lines = []
    lines.append(f"🧭 **Position Advice — {advice['symbol']}**")
    lines.append(
        f"Entry: {_fmt_usd(advice['entry_price'])} | Last: {_fmt_usd(advice['last_price'])} | "
        f"Shares: {advice['shares']} | Value: {_fmt_usd(advice['position_value'])}"
    )
    lines.append(f"P/L: {_fmt_usd(advice['pnl'])} ({advice['pnl_pct']}%) | Risk: {advice['risk_level'].title()} | Sector: {advice.get('sector','N/A')}")
    lines.append(f"Decision: **{advice['action']}**")

    # Projections (compact 3m + 6m, bull/bear)
    proj = advice.get("projections", {})
    if proj:
        p3b = proj.get("3m", {}).get("bull", {})
        p3r = proj.get("3m", {}).get("bear", {})
        p6b = proj.get("6m", {}).get("bull", {})
        p6r = proj.get("6m", {}).get("bear", {})
        lines.append(
            "\n📈 **Projections**"
            f"\n3m — Bull L {_fmt_usd(p3b.get('low'))} / M {_fmt_usd(p3b.get('mid'))} / H {_fmt_usd(p3b.get('high'))}"
            f" | Bear L {_fmt_usd(p3r.get('low'))} / M {_fmt_usd(p3r.get('mid'))} / H {_fmt_usd(p3r.get('high'))}"
        )
        lines.append(
            f"6m — Bull L {_fmt_usd(p6b.get('low'))} / M {_fmt_usd(p6b.get('mid'))} / H {_fmt_usd(p6b.get('high'))}"
            f" | Bear L {_fmt_usd(p6r.get('low'))} / M {_fmt_usd(p6r.get('mid'))} / H {_fmt_usd(p6r.get('high'))}"
        )

    # Ratings compact
    rating = advice.get("rating", {})
    if rating:
        q = rating.get("quarterly", {}); s6 = rating.get("six_month", {})
        lines.append(
            f"\n🧮 **Ratings** — 3m: {q.get('label','?')} (score {q.get('score','?')}); "
            f"6m: {s6.get('label','?')} (score {s6.get('score','?')})"
        )

    # Action-specific short block
    if advice["action"] in ["DIVERSIFY", "EXIT"]:
        rea = advice.get("reallocation", {})
        amt = rea.get("amount_to_redeploy", 0)
        lines.append(f"\n💡 Redeploy: {_fmt_usd(amt)} across top picks (inverse-vol).")
        if rea.get("dollar_allocations"):
            # show only top 6 names
            pairs = sorted(rea["dollar_allocations"].items(), key=lambda x: -x[1])[:6]
            short = ", ".join([f"{k} {_fmt_usd(v)}" for k, v in pairs if v > 0])
            lines.append(f"Alloc: {short}")
        hedge = advice.get("notes", {}).get("hedge")
        if hedge:
            lines.append(f"🛡️ Hedge: {hedge}")
    else:
        hp = advice.get("notes", {}).get("hold_params", {})
        if hp:
            lines.append(f"\n🧷 Hold: Trailing stop ~{hp.get('suggested_trailing_stop_pct','?')}% — {hp.get('rationale','')}")

    # Rationale super short
    rat = advice.get("rationale", {})
    lines.append(
        f"\n🔎 Why: 3m Mom {rat.get('momentum_3m','?')}% | "
        f"Min DD {rat.get('recent_drawdown_min','?')}% | "
        f"PE {rat.get('quality_hint',{}).get('pe','?')} | "
        f"Margin {rat.get('quality_hint',{}).get('profit_margin','?')}"
    )

    body = "\n".join(lines)
    chunks = _split_for_telegram(body)
    if len(chunks) == 2:
        chunks[0] = "(1/2)\n" + chunks[0]
        chunks[1] = "(2/2)\n" + chunks[1]
    return chunks
