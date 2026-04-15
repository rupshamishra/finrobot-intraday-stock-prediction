"""
NSE 30-MIN FORWARD TEST — RUNNER
====================================
Wraps the REAL LLM agent pipeline (AutoGen + all 14 tools) for forward testing.

The agent predicts the NEXT 30 MINUTES from the time you run it.
Entry price = latest 15m candle close (live intraday price).
Predicted close = entry × (1 ± scaled_atr_30min_pct).
Actual close    = the 15m candle that closes ~30 min after run_timestamp.

Run this DURING market hours (09:15–14:45 IST). Each run produces one
prediction per stock. Run it multiple times a day to build a dataset.

USAGE:
  Step 1 — Run during market hours (09:15–14:45 IST):
      python forward_test_runner.py --run

  Step 2 — Fetch actuals ~35 min after --run completes:
      python forward_test_runner.py --fetch

  Step 3 — Evaluate + plot:
      python forward_test_runner.py --evaluate

REQUIREMENTS:
  - yfinance_utils_nse.py  (or patched into finrobot)
  - autogen, finrobot installed
  - OAI_CONFIG_LIST.json in parent folder (../OAI_CONFIG_LIST.json)
"""

import argparse
import json
import re
import sys
import os
import time
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

warnings.filterwarnings("ignore")

# ─── CONFIG ──────────────────────────────────────────────────────────────────

WATCHLIST = [
    "HDFCBANK.NS",
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "ICICIBANK.NS",
]

def _predictions_file(window: int) -> Path:
    return Path(f"ft_predictions_{window}min.jsonl")

def _chart_file(window: int) -> Path:
    return Path(f"ft_results_{window}min.png")


# ─── STEP 1: RUN THE REAL LLM AGENT ─────────────────────────────────────────

def run_agent_for_stock(symbol: str, llm_config: dict, YFinanceUtils, window: int = 30) -> dict:
    """
    Run the full AutoGen + 14-tool agent for one stock.
    window: prediction horizon in minutes (30 or 60).
    Returns a prediction dict with parsed fields + raw LLM output.
    """
    import autogen
    from autogen.cache import Cache
    from finrobot.toolkits import register_toolkits
    from finrobot.utils import get_current_date

    today = get_current_date()
    start_date = (datetime.today() - timedelta(days=45)).strftime("%Y-%m-%d")

    # Current IST time — used in prompt so agent knows "now"
    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    ist_now_str = ist_now.strftime("%H:%M IST")

    analyst = autogen.AssistantAgent(
        name="NSE_Market_Analyst",
        system_message=_build_system_prompt(),
        llm_config=llm_config,
    )

    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        is_termination_msg=lambda x: (
            isinstance(x.get("content"), str) and "TERMINATE" in x.get("content", "")
        ) or (
            isinstance(x.get("content"), list) and
            any("TERMINATE" in (b.get("text","") if isinstance(b,dict) else "") for b in x.get("content",[]))
        ),
        human_input_mode="NEVER",
        max_consecutive_auto_reply=20,
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )

    tools = _build_tools(YFinanceUtils)
    register_toolkits(tools, analyst, user_proxy)

    prompt = _build_prompt(symbol, today, start_date, ist_now_str, window)

    raw_output = ""
    run_timestamp = datetime.now(timezone.utc).isoformat()

    try:
        with Cache.disk(cache_seed=None) as cache:  # cache_seed=None = no caching (fresh run)
            result = user_proxy.initiate_chat(
                analyst,
                message=prompt,
                cache=cache,
            )

        # Capture ALL messages — handle both str and list content types.
        # AutoGen v0.2+ may store the LLM's final reply as a list of content
        # blocks: [{"type": "text", "text": "..."}]. We must handle both forms.
        for msg in user_proxy.chat_messages.get(analyst, []):
            content = msg.get("content") or ""
            if isinstance(content, str) and content:
                raw_output += content + "\n\n"
            elif isinstance(content, list):
                # Extract text from each block in the list
                for block in content:
                    if isinstance(block, dict):
                        text = block.get("text") or block.get("content") or ""
                        if text and isinstance(text, str):
                            raw_output += text + "\n\n"

    except Exception as e:
        print(f"    ⚠ Agent error for {symbol}: {e}")
        raw_output = f"AGENT_ERROR: {str(e)}"

    # Save raw agent output for debugging
    AGENT_LOG_DIR = Path("ft_agent_logs")
    AGENT_LOG_DIR.mkdir(exist_ok=True)
    log_file = AGENT_LOG_DIR / f"{symbol.replace('.', '_')}_{today}.txt"
    log_file.write_text(raw_output, encoding="utf-8")

    # Parse the structured output
    parsed = parse_agent_output(raw_output, symbol, window)
    parsed["run_timestamp"] = run_timestamp
    parsed["raw_output_file"] = str(log_file)

    return parsed


def _build_system_prompt() -> str:
    return """
You are an expert NSE (National Stock Exchange of India) intraday trader and analyst.
You have deep knowledge of:
  - Indian market microstructure (NSE/BSE, 9:15-15:30 IST session)
  - F&O dynamics: weekly expiry every Thursday, monthly = last Thursday of month
  - FII/DII flows and their impact on index-heavy stocks
  - India VIX (>20 = high fear / wide swings, <15 = low fear / tight range)
  - USD/INR: rupee weakness means FII outflows, bearish for indices
  - Crude oil: Brent >$80 bearish for Indian oil importers
  - Pivot points and VWAP as key NSE intraday reference levels

RULES FOR READING TOOL OUTPUT — follow exactly, never override
==============================================================

RSI:
  Copy rsi_signal from get_technical_indicators verbatim into your output.
  If rsi_signal = 'neutral', write 'neutral'. NEVER write 'oversold' unless
  rsi_signal explicitly equals 'oversold'. RSI=30.75 is NEUTRAL, not oversold.
  CRITICAL — RSI IS NOT A DIRECTIONAL SIGNAL FOR 30-MIN PREDICTION:
  RSI oversold does NOT mean Bullish. RSI overbought does NOT mean Bearish.
  Oversold/overbought means exhaustion — the trend may CONTINUE, not reverse.
  For 30-min direction use MACD trend + EMA trend as primary signals.
  Only use RSI to note exhaustion risk, never to set direction.

MACD:
  Two separate fields are returned: macd_crossover and macd_trend.
  - macd_crossover = 'none'              → write 'macd_trend (no crossover this session)'
  - macd_crossover = 'bearish_crossover' → write 'bearish crossover confirmed'
  - macd_crossover = 'bullish_crossover' → write 'bullish crossover confirmed'
  NEVER write 'crossover' if macd_crossover = 'none'.

PCR and Max Pain:
  If put_call_ratio or max_pain_level contain the word UNAVAILABLE,
  write exactly 'N/A (data unavailable)'. NEVER estimate or guess these values.

Crude Oil:
  If brent_crude_proxy.data_anomaly = True, write 'Data anomaly — excluded from analysis'.
  Do NOT use change_pct from that field under any circumstances.

PRICE TARGETS — USE PRE-COMPUTED SETUPS ONLY
=============================================
Use bearish_setup for Bearish, bullish_setup for Bullish predictions.
NEVER manually select S1 or R1 as a target.
NEGATIVE R:R RULE: After selecting a setup, check R:R.
  If R:R < 0 or target is on the wrong side of entry (target < entry for Bullish,
  or target > entry for Bearish), you used the wrong setup.
  Switch to the other setup, or set Direction=Sideways. NEVER output R:R < 0.

EXPECTED MOVE:
Use ONLY the expected_intraday_range_pct value from THIS stock's get_technical_indicators.
Format: '±X.XX% (bearish bias)' or '±X.XX% (bullish bias)'.

CONFIDENCE SCORING — USE compute_confidence() TOOL, NEVER COMPUTE YOURSELF
===========================================================================
After finalising direction AND price targets, call compute_confidence() with ALL fields:
  compute_confidence(
    symbol=..., direction=..., macd_trend=..., ema_trend=...,
    nifty_change_pct=..., nifty_5d_trend=..., obv_trend=...,
    news_conflict=..., is_expiry_day=...,
    vix=...,          ← India VIX from get_nse_market_context
    rr_ratio=...,     ← R:R from selected setup (must be > 0)
    target_price=..., ← target from selected setup
    entry_price=...,  ← latest 15m close from get_intraday_data
  )
Copy score, confidence VERBATIM. If macro_override=True, add macro_override_reason
to DATA QUALITY FLAGS. If contradiction=True, set Direction=Sideways + tradeable=False
and add contradiction_reason to DATA QUALITY FLAGS.

Always call ALL tools before synthesizing. Reply TERMINATE when done.
"""


def _build_prompt(symbol: str, today: str, start_date: str, ist_now_str: str, window: int = 30) -> str:
    import math
    scaled = round(math.sqrt(window / 375), 4)
    intraday_interval = "15m" if window <= 30 else "30m"
    return f"""
You are an NSE intraday analyst. Today is {today}. Current time: {ist_now_str}.
NSE session: 09:15–15:30 IST.
Target stock: {symbol}

YOUR TASK: Predict the price direction and move for THIS STOCK over the
NEXT {window} MINUTES from right now ({ist_now_str}).

ENTRY PRICE RULE: Use the 'latest_close' from get_intraday_data() — this is
the most recent {intraday_interval} candle close, i.e. the current live price.

EXPECTED MOVE RULE: The expected_intraday_range_pct from get_technical_indicators
is a FULL SESSION range. Scale it to {window} minutes:
  {window}min_move_pct = expected_intraday_range_pct × sqrt({window} / 375)
                       = expected_intraday_range_pct × {scaled}
  (375 = total NSE session minutes)
Round to 2 decimal places. Use this scaled value in the report.

STEP 1 — MARKET BACKDROP:
  1a. get_nse_market_context()
  1b. get_fno_data(symbol="{symbol}")

STEP 2 — STOCK ANALYSIS:
  2a. get_stock_data(symbol="{symbol}", start_date="{start_date}", end_date="{today}")
  2b. get_technical_indicators(symbol="{symbol}")
  2c. get_support_resistance(symbol="{symbol}")
  2d. get_intraday_data(symbol="{symbol}", interval="{intraday_interval}", period="5d")
  2e. get_extended_company_info(symbol="{symbol}")
  2f. get_company_news(symbol="{symbol}")
  2g. get_sector_peers(symbol="{symbol}")
  2h. get_data_sanity_check(symbol="{symbol}")

STEP 3 — COMPUTE CONFIDENCE:
  Call compute_confidence() with raw values from tools. Copy output verbatim.

STEP 4 — OUTPUT (fill every field from tool data):

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE {window}-MIN FORECAST — {symbol}
Date: {today} | As of: {ist_now_str} | Horizon: next {window} minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREDICTION SUMMARY:
  Direction       : [Bullish / Bearish / Sideways]
  Expected Move   : [±X.XX% (bearish/bullish bias — {window}min scaled)]
  Price Targets   : Entry=____, Target=____, Stop=____, R:R=____
  Confidence      : [High/Medium/Low]
  Max Pain Level  : [value OR 'N/A (data unavailable)']

INTRADAY CONTEXT (from get_intraday_data):
  Current Price   : ____ (latest {intraday_interval} close) [above/below] VWAP ____
  Session High    : ____ | Session Low: ____
  Last 2 candles  : [brief description of recent {intraday_interval} price action]
  Morning Bias    : [describe whether price is trending up/down/flat since open]

TECHNICAL SIGNALS:
  RSI(14)         : ____ → [rsi_signal VERBATIM]
  MACD            : macd_trend=____ | macd_crossover=____
  EMA Trend       : [ema_trend verbatim]
  BB Position     : [bb_position verbatim]
  VWAP            : Price ____ is [above/below] VWAP ____
  ATR(14)         : ____ pts (full session) → ±____% scaled to {window}min
  ADX(14)         : ____ → [strong_trend/weak_trend]
  OBV Trend       : [obv_trend] → [confirms/contradicts direction]
  Volume          : [volume_vs_avg] vs 20d average

MARKET CONTEXT:
  NIFTY50         : ____ [____% change] | 5d trend: ____
  BANKNIFTY       : ____ [____% change]
  India VIX       : ____ → [level]
  USD/INR         : ____
  Crude Oil       : [value or 'Data anomaly — excluded']
  F&O Context     : PCR=____, Max Pain=____, [days]d to expiry
  Sector Peers    : [outperforming/underperforming] by ____% vs sector avg

KEY CATALYSTS:
  Positives:
    1. ...
  Risks:
    1. ...

CONFIDENCE JUSTIFICATION:
  - Technical aligned (MACD+EMA confirm direction): [Yes/No]
  - NIFTY trend aligned:                            [Yes/No]
  - OBV volume confirming:                          [Yes/No]
  - News clear (no major conflict):                 [Yes/No]
  - Non-expiry day:                                 [Yes/No]
  → Score: [X/5] → [High/Medium/Low]

DATA QUALITY FLAGS:
  [flags or 'None — all checks passed']

FINAL REASONING:
[3-4 sentences focusing on the NEXT 30 MINUTES specifically — 
reference current price vs VWAP, recent candle direction, and momentum.]

IMPORTANT — You must now call log_forecast() as a tool call (not as text).
Do NOT write 'TERMINATE' yet. First make the tool call:
  log_forecast(symbol="{symbol}", date="{today}", direction=<Direction>,
               entry_price=<Entry>, target_price=<Target>, stop_price=<Stop>,
               rr_ratio=<R:R>, confidence=<Confidence>, score=<Score>,
               tradeable=<True/False>, factors=<factors dict from compute_confidence>)
After log_forecast() returns, reply with ONLY the word: TERMINATE
"""


def _build_tools(YFinanceUtils) -> list:
    return [
        {"function": YFinanceUtils.get_stock_data,          "name": "get_stock_data",          "description": "Daily OHLCV price history for an NSE stock."},
        {"function": YFinanceUtils.get_company_news,         "name": "get_company_news",         "description": "Latest company news headlines."},
        {"function": YFinanceUtils.get_extended_company_info,"name": "get_extended_company_info","description": "NSE fundamentals: P/E, beta, 52w range, analyst target."},
        {"function": YFinanceUtils.get_intraday_data,        "name": "get_intraday_data",        "description": "15m intraday OHLCV candles with VWAP."},
        {"function": YFinanceUtils.get_technical_indicators, "name": "get_technical_indicators", "description": "RSI, MACD, BB, EMA, ATR, ADX, OBV. Copy rsi_signal verbatim. Use macd_crossover field."},
        {"function": YFinanceUtils.get_support_resistance,   "name": "get_support_resistance",   "description": "Classic + Camarilla pivots. Use bearish_setup/bullish_setup for targets."},
        {"function": YFinanceUtils.get_nse_market_context,   "name": "get_nse_market_context",   "description": "NIFTY50, BANKNIFTY, VIX, USD/INR, Crude Oil."},
        {"function": YFinanceUtils.get_fno_data,             "name": "get_fno_data",             "description": "F&O expiry flag, PCR, Max Pain. If UNAVAILABLE, report N/A."},
        {"function": YFinanceUtils.get_sector_peers,         "name": "get_sector_peers",         "description": "Sector peer performance comparison."},
        {"function": YFinanceUtils.get_data_sanity_check,    "name": "get_data_sanity_check",    "description": "Cross-validates data. Report any flags in Final Reasoning."},
        {"function": YFinanceUtils.compute_confidence,       "name": "compute_confidence",       "description": "MANDATORY. Deterministic confidence score. Copy output verbatim."},
        {"function": YFinanceUtils.log_forecast,             "name": "log_forecast",             "description": "MANDATORY final step. Log prediction to forecasts.jsonl."},
    ]


# ─── PARSE LLM OUTPUT ────────────────────────────────────────────────────────

def parse_agent_output(text: str, symbol: str, window: int = 30) -> dict:
    """
    Extract structured fields from the LLM's formatted report.
    Uses regex against the known output template. Falls back to None on miss.
    """
    def extract(pattern, default=None, cast=None):
        m = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if not m:
            return default
        val = m.group(1).strip()
        try:
            return cast(val) if cast else val
        except Exception:
            return default

    # Direction
    direction = extract(r"Direction\s*:\s*(Bullish|Bearish|Sideways)")

    # Expected move % — grab the numeric part e.g. ±1.95% → 1.95
    exp_move_pct = extract(r"Expected Move\s*:.*?±(\d+\.?\d*)\s*%", cast=float)

    # Price targets
    entry  = extract(r"Entry\s*=\s*([0-9]+\.?[0-9]*)", cast=float)
    target = extract(r"Target\s*=\s*([0-9]+\.?[0-9]*)", cast=float)
    stop   = extract(r"Stop\s*=\s*([0-9]+\.?[0-9]*)", cast=float)
    rr     = extract(r"R:R\s*=\s*([0-9]+\.?[0-9]*)", cast=float)

    # Confidence
    confidence = extract(r"Confidence\s*:\s*(High|Medium|Low)")

    # Score
    score = extract(r"Score:\s*(\d)/5", cast=int)

    # ATR % — now the 30-min scaled value is printed in report
    atr_pct = extract(r"ATR\(14\)\s*:.*?±(\d+\.?\d*)\s*%\s*scaled to 30min", cast=float)
    if atr_pct is None:  # fallback to any ATR% in line
        atr_pct = extract(r"ATR\(14\)\s*:.*?±(\d+\.?\d*)\s*%", cast=float)

    # Latest close — from "Current Price : XXXX (latest 15m close)" line (new template)
    latest_close = extract(r"Current Price\s*:\s*([0-9]+\.?[0-9]*)", cast=float)
    if latest_close is None:
        # fallback: old VWAP line
        latest_close = extract(r"Price\s+([0-9]+\.?[0-9]*)\s+is\s+(?:above|below)\s+VWAP", cast=float)
    if latest_close is None:
        latest_close = entry  # last resort

    # Tradeable — if "untradeable" appears in price targets section
    tradeable_text = extract(r"(Setup untradeable|viable setup|tradeable=True|tradeable=False)", default="")
    tradeable = False if tradeable_text and "untradeable" in tradeable_text.lower() else True

    # Compute predicted close from direction + expected move
    predicted_close = None
    if latest_close and exp_move_pct and direction:
        mult = {"bullish": 1.0, "bearish": -1.0, "sideways": 0.0}.get(direction.lower(), 0.0)
        predicted_close = round(latest_close * (1 + mult * exp_move_pct / 100), 2)

    return {
        "symbol":          symbol,
        "date":            datetime.now().strftime("%Y-%m-%d"),
        "window_min":      window,
        "run_timestamp":   None,           # filled by caller
        "direction":       direction,
        "exp_move_pct":    exp_move_pct,
        "entry_price":     entry or latest_close,
        "target_price":    target,
        "stop_price":      stop,
        "rr_ratio":        rr,
        "confidence":      confidence,
        "score":           score,
        "atr_pct":         atr_pct,
        "latest_close":    latest_close,
        "predicted_close": predicted_close,
        "tradeable":       tradeable,
        # actuals — filled later
        "actual_close":    None,
        "actual_high":     None,
        "actual_low":      None,
        "actual_open":     None,
        "actual_candle_ts": None,
        "error_pct":       None,
        "direction_correct": None,
        "actual_direction":  None,
        "parse_ok":        all(x is not None for x in [direction, exp_move_pct, entry, target, stop, confidence]),
        "raw_output_file": None,           # filled by caller
    }


# ─── STEP 2: FETCH ACTUALS ────────────────────────────────────────────────────

def fetch_actuals(window: int = 30):
    """
    For each prediction without actuals, fetch the candle that closes
    ~window minutes after run_timestamp.
    """
    PREDICTIONS_FILE = _predictions_file(window)
    if not PREDICTIONS_FILE.exists():
        print("No predictions file found. Run --run first.")
        return

    records = []
    for l in PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l:
            continue
        try:
            records.append(json.loads(l))
        except json.JSONDecodeError:
            print(f"  ⚠ skipping corrupt line: {l[:60]}...")
    pending = [r for r in records if r.get("actual_close") is None]

    # ── Recompute direction_correct for ALL existing complete records ─────────
    recomputed = 0
    for r in records:
        if r.get("actual_close") is None:
            continue
        entry = r.get("entry_price") or r.get("latest_close")
        act   = r.get("actual_close")
        if entry and act:
            move_pct = abs((act - entry) / entry * 100)
            atr_pct  = r.get("atr_pct") or 0.5
            actual_dir = "Bullish" if act > entry else ("Bearish" if act < entry else "Sideways")
            r["actual_direction"] = actual_dir
            if r.get("direction") == "Sideways":
                r["direction_correct"] = move_pct <= atr_pct
            else:
                r["direction_correct"] = actual_dir == r.get("direction")
            recomputed += 1
    if recomputed:
        print(f"  ℹ Recomputed direction_correct for {recomputed} existing records (Sideways = within ±ATR%).")
    # ─────────────────────────────────────────────────────────────────────────

    ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
    print(f"\n{'='*60}")
    print(f"FETCH {window}-MIN ACTUALS — {ist_now.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"Pending: {len(pending)} / {len(records)}")
    print(f"{'='*60}\n")

    wait_secs = window * 60
    filled = 0
    for r in records:
        if r.get("actual_close") is not None:
            continue

        symbol = r["symbol"]
        run_ts = r.get("run_timestamp")
        print(f"  → {symbol} ", end="", flush=True)

        # Check if window minutes have passed since run_timestamp
        if run_ts:
            run_dt = datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
            if run_dt.tzinfo is None:
                run_dt = run_dt.replace(tzinfo=timezone.utc)
            now_utc = datetime.now(timezone.utc)
            elapsed = now_utc - run_dt
            if elapsed.total_seconds() < wait_secs:
                mins_left = int((wait_secs - elapsed.total_seconds()) / 60)
                print(f"⏳ only {int(elapsed.total_seconds()/60)}min elapsed — wait {mins_left} more min")
                continue

        actual = _fetch_next_candle(symbol, run_ts, window)
        if actual is None:
            print("⏳ candle not yet available in yfinance — try again in a few minutes")
            continue

        r.update(actual)

        # Compute errors
        pred  = r.get("predicted_close")
        act   = r["actual_close"]
        entry = r.get("entry_price") or r.get("latest_close")  # direction judged FROM entry

        if pred and act:
            r["error_pct"] = round((act - pred) / pred * 100, 4)

        # direction_correct = did actual move from ENTRY in the predicted direction?
        # Special case: Sideways = correct if actual move is within ±ATR% of entry
        if entry and act:
            move_pct = abs((act - entry) / entry * 100)
            atr_pct  = r.get("atr_pct") or 0.5   # fallback 0.5% if missing
            if act > entry:
                actual_dir = "Bullish"
            elif act < entry:
                actual_dir = "Bearish"
            else:
                actual_dir = "Sideways"
            r["actual_direction"] = actual_dir
            predicted_dir = r.get("direction")
            if predicted_dir == "Sideways":
                # Sideways correct if move stayed within one scaled ATR
                r["direction_correct"] = move_pct <= atr_pct
            else:
                r["direction_correct"] = actual_dir == predicted_dir

        filled += 1
        dir_icon = "✓" if r.get("direction_correct") else "✗"
        err_str  = f"err={r['error_pct']:+.3f}%" if r.get("error_pct") is not None else ""
        print(f"✅  actual={act}  predicted={pred}  {err_str}  dir={dir_icon}")

    # Save
    PREDICTIONS_FILE.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"\n{'='*60}")
    print(f"✅ {filled} actuals saved. {len(pending)-filled} still pending.")

    if filled > 0:
        _print_quick_stats(records)


def _fetch_next_candle(symbol: str, run_ts: str, window: int = 30) -> dict | None:
    """
    Fetch the first candle closing >= window minutes after run_ts (UTC).
    Uses 15m candles for <=30min window, 30m candles for 60min window.
    """
    try:
        if run_ts:
            run_dt_utc = datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
            if run_dt_utc.tzinfo is None:
                run_dt_utc = run_dt_utc.replace(tzinfo=timezone.utc)
        else:
            run_dt_utc = datetime.now(timezone.utc)

        target_utc = run_dt_utc + timedelta(minutes=window)
        interval   = "15m" if window <= 30 else "30m"

        df = yf.Ticker(symbol).history(interval=interval, period="2d")
        if df.empty:
            return None

        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        # First candle closing AFTER target time
        later = df[df.index >= target_utc]
        if later.empty:
            return None

        c = later.iloc[0]
        return {
            "actual_open":      round(float(c["Open"]),  2),
            "actual_high":      round(float(c["High"]),  2),
            "actual_low":       round(float(c["Low"]),   2),
            "actual_close":     round(float(c["Close"]), 2),
            "actual_candle_ts": str(c.name),
        }
    except Exception as e:
        print(f"\n    ⚠ fetch error: {e}")
        return None


def _print_quick_stats(records):
    complete = [r for r in records if r.get("actual_close") is not None]
    if not complete:
        return
    scored   = [r for r in complete if r.get("direction_correct") is not None]
    dir_acc  = sum(1 for r in scored if r["direction_correct"]) / len(scored) if scored else 0
    errors   = [abs(r["error_pct"]) for r in complete if r.get("error_pct") is not None]
    mape     = sum(errors) / len(errors) if errors else 0
    print(f"\nQUICK STATS ({len(complete)} predictions):")
    print(f"  Directional accuracy : {dir_acc:.1%}  (actual vs entry price)")
    print(f"  MAPE                 : {mape:.4f}%")
    print(f"\n  {'Symbol':<16} {'PredDir':<8} {'Entry':>8} {'Pred':>8} {'Actual':>8} {'Err%':>8} {'ActDir':<7} {'✓/✗'}")
    print("  " + "-"*72)
    for r in complete:
        icon    = "✓" if r.get("direction_correct") else "✗"
        pred_s  = f"{r['predicted_close']:.2f}" if r.get("predicted_close") is not None else "N/A"
        entry_s = f"{r['entry_price']:.2f}"     if r.get("entry_price")     is not None else "N/A"
        act_s   = f"{r['actual_close']:.2f}"    if r.get("actual_close")    is not None else "N/A"
        err_s   = f"{r['error_pct']:>+8.3f}%"  if r.get("error_pct")       is not None else "    N/A%"
        act_dir = str(r.get("actual_direction", "?"))[:4]
        line = (f"  {r['symbol']:<16} {str(r.get('direction','?')):<8} {entry_s:>8} "
                f"{pred_s:>8} {act_s:>8} "
                f"{err_s} {act_dir:<7} {icon}")
        print(line)


# ─── STEP 3: EVALUATE + PLOT ─────────────────────────────────────────────────

def evaluate_and_plot(window: int = 30):
    """Compute full error metrics and generate 4-chart visualisation."""
    PREDICTIONS_FILE = _predictions_file(window)
    CHART_FILE       = _chart_file(window)
    if not PREDICTIONS_FILE.exists():
        print("No predictions file. Run --run and --fetch first.")
        return

    records = []
    for l in PREDICTIONS_FILE.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l:
            continue
        try:
            records.append(json.loads(l))
        except json.JSONDecodeError:
            print(f"  ⚠ skipping corrupt line: {l[:60]}...")
    complete = [r for r in records if r.get("actual_close") is not None and r.get("predicted_close") is not None]

    if not complete:
        print("No complete records yet (need both prediction + actual).")
        print("Run --fetch first.")
        return

    df = pd.DataFrame(complete)
    df["signed_err"]   = ((df["actual_close"] - df["predicted_close"]) / df["predicted_close"] * 100).round(4)
    df["abs_err"]      = df["signed_err"].abs()
    df["range_hit"]    = df.apply(lambda r: _range_hit(r), axis=1)
    df["sym_short"]    = df["symbol"].str.replace(".NS", "", regex=False)

    _print_full_stats(df)
    _make_charts(df, CHART_FILE)

    # Save CSV
    csv_file = Path(f"ft_results_{window}min.csv")
    df.to_csv(csv_file, index=False)
    print(f"\n✅ Results CSV  : {csv_file.resolve()}")
    print(f"✅ Chart PNG    : {CHART_FILE.resolve()}")


def _range_hit(r):
    """Did actual EOD close land within predicted_close ± atr_pct band?
    Tests whether the agent's expected intraday range contained the final price."""
    if r.get("atr_pct") and r.get("predicted_close") and r.get("actual_close"):
        band = r["predicted_close"] * r["atr_pct"] / 100
        return (r["predicted_close"] - band) <= r["actual_close"] <= (r["predicted_close"] + band)
    return False


def _print_full_stats(df: pd.DataFrame):
    print(f"\n{'='*70}")
    print(f"NSE 30-MIN FORWARD TEST — FULL EVALUATION ({len(df)} predictions)")
    print(f"{'='*70}")

    # Overall
    dir_acc  = df["direction_correct"].mean()
    mape     = df["abs_err"].mean()
    rmse     = np.sqrt((df["signed_err"]**2).mean())
    bias     = df["signed_err"].mean()
    rhit     = df["range_hit"].mean()
    print(f"\nOVERALL METRICS (all {len(df)} predictions):")
    print(f"  Directional Accuracy  : {dir_acc:.1%}")
    print(f"  MAPE                  : {mape:.4f}%")
    print(f"  RMSE                  : {rmse:.4f}%")
    print(f"  Prediction Bias       : {bias:+.4f}%  ({'over-bullish' if bias>0 else 'over-bearish'})")
    print(f"  ATR Range Hit Rate    : {rhit:.1%}  (actual inside \u00b1ATR% of predicted)")

    # Filtered: High+Medium, non-Sideways only
    tradeable_df = df[
        (df["confidence"].isin(["High", "Medium"])) &
        (df["direction"] != "Sideways")
    ]
    if len(tradeable_df) > 0:
        t_acc  = tradeable_df["direction_correct"].mean()
        t_mape = tradeable_df["abs_err"].mean()
        skipped = len(df) - len(tradeable_df)
        print(f"\nFILTERED METRICS (High+Medium, non-Sideways — {len(tradeable_df)} predictions, {skipped} skipped):")
        acc_icon = "✅" if t_acc >= 0.6 else "⚠️"
        print(f"  Directional Accuracy  : {t_acc:.1%}  {acc_icon}")
        print(f"  MAPE                  : {t_mape:.4f}%")
        print(f"  Skipped (Low/Sideways): {skipped} ({skipped/len(df):.0%} of total)")

    # Per-stock detail
    print(f"\nPER-STOCK DETAIL:")
    print(f"  {'Symbol':<16} {'PredDir':<10} {'ActDir':<10} {'Dir':>4} "
          f"{'Entry':>8} {'Pred':>8} {'Actual':>8} {'Err%':>8} {'Conf':>6} {'Use?':>5}")
    print("  " + "-"*92)
    for _, r in df.iterrows():
        icon        = "✓" if r["direction_correct"] else "✗"
        is_sideways = r.get("direction") == "Sideways"
        is_low      = r.get("confidence") == "Low"
        use         = "—" if (is_sideways or is_low) else icon
        flag        = " <skip" if (is_sideways or is_low) else ""
        pred = f"{r['predicted_close']:.2f}" if pd.notna(r.get("predicted_close")) else "N/A"
        ent  = f"{r['entry_price']:.2f}"     if pd.notna(r.get("entry_price"))     else "N/A"
        print(f"  {r['sym_short']:<16} {str(r.get('direction','?')):<10} "
              f"{str(r.get('actual_direction','?')):<10} {icon:>4} "
              f"{ent:>8} {pred:>8} {r['actual_close']:>8.2f} "
              f"{r['signed_err']:>+8.3f}% {str(r.get('confidence','?')):>6} {use:>5}{flag}")

    # By confidence level
    print(f"\nBY CONFIDENCE LEVEL:")
    for level in ["High", "Medium", "Low"]:
        sub = df[df["confidence"] == level]
        if len(sub) == 0:
            continue
        acc = sub["direction_correct"].mean()
        flag = "  ✅ use" if (level in ("High","Medium") and acc >= 0.55) else                ("  ⚠️  borderline" if level in ("High","Medium") else "  ❌ skip")
        print(f"  {level:<8}: n={len(sub):>2}  dir_acc={acc:.1%}  "
              f"mape={sub['abs_err'].mean():.4f}%{flag}")

    # By direction
    print(f"\nBY PREDICTED DIRECTION:")
    for d in ["Bullish", "Bearish", "Sideways"]:
        sub = df[df["direction"] == d]
        if len(sub) == 0:
            continue
        acc = sub["direction_correct"].mean()
        flag = "  ❌ skip (no edge)" if d == "Sideways" else ""
        print(f"  {d:<10}: n={len(sub):>2}  acc={acc:.1%}  "
              f"mape={sub['abs_err'].mean():.4f}%{flag}")

    # Recommendation
    print(f"\nRECOMMENDATION:")
    if len(tradeable_df) > 0:
        if t_acc >= 0.65:
            print(f"  ✅ High+Medium non-Sideways predictions are reliable ({t_acc:.0%} acc).")
            print(f"     Filter out Low confidence and Sideways calls.")
        elif t_acc >= 0.50:
            print(f"  ⚠️  High+Medium accuracy is {t_acc:.0%} — marginal. Accumulate more data.")
            print(f"     Definitely skip Low confidence and Sideways.")
        else:
            print(f"  ❌ Even filtered accuracy is low ({t_acc:.0%}). Check macro alignment.")
    print(f"  📊 Need ~50+ filtered predictions for statistically stable conclusions.")


def _make_charts(df: pd.DataFrame, chart_file: Path = None):
    """Generate 4-panel chart."""
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.patch.set_facecolor("#0d0d1a")
    date_str = df["date"].iloc[0] if "date" in df.columns else datetime.now().strftime("%Y-%m-%d")
    fig.suptitle(f"NSE LLM Pipeline — Forward Test Results  ·  {date_str}",
                 fontsize=15, fontweight="bold", color="white", y=0.98)

    syms   = df["sym_short"].tolist()
    n      = len(df)
    x      = np.arange(n)
    w      = 0.32
    c_ok   = "#00c896"
    c_err  = "#ff4d6d"
    c_pred = "#4a9eff"
    c_ent  = "#ffd700"
    c_act  = "#00c896"

    def _ax_style(ax, title):
        ax.set_facecolor("#12122a")
        ax.set_title(title, color="white", pad=8, fontsize=10)
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_color("#333355")
        ax.grid(axis="y", alpha=0.15, color="white")

    # ── Chart 1: Entry vs Predicted vs Actual EOD close ──────────────────────
    ax1 = axes[0, 0]
    _ax_style(ax1, "Entry / Predicted / Actual (30-min close)")

    b1 = ax1.bar(x - w,    df["entry_price"],    w, label="Entry",     color=c_ent,  alpha=0.8, zorder=3)
    b2 = ax1.bar(x,        df["predicted_close"],w, label="Predicted", color=c_pred, alpha=0.8, zorder=3)
    b3 = ax1.bar(x + w,    df["actual_close"],   w, label="Actual",    color=c_act,  alpha=0.8, zorder=3)

    ax1.set_xticks(x)
    ax1.set_xticklabels(syms, rotation=30, ha="right", color="white", fontsize=9)
    ax1.set_ylabel("Price (₹)", color="white")
    ax1.legend(facecolor="#12122a", labelcolor="white", fontsize=8)

    # ── Chart 2: Directional accuracy per stock ───────────────────────────────
    ax2 = axes[0, 1]
    _ax_style(ax2, f"Directional Accuracy — {df['direction_correct'].mean():.1%} overall")

    bar_colors = [c_ok if v else c_err for v in df["direction_correct"]]
    ax2.bar(syms, [1]*n, color=bar_colors, alpha=0.85, zorder=3)

    conf_labels = df.get("confidence", pd.Series(["?"]*n))
    for i, (_, r) in enumerate(df.iterrows()):
        pd_   = str(r.get("direction", "?"))[:4]
        ad_   = str(r.get("actual_direction", "?"))[:4]
        conf_ = str(r.get("confidence", "?"))
        ax2.text(i, 0.5, f"{pd_}→{ad_}\n{conf_}", ha="center", va="center",
                 fontsize=8, color="white", fontweight="bold")

    ax2.set_yticks([])
    ax2.set_xticklabels(syms, rotation=30, ha="right", color="white", fontsize=9)
    patch_ok  = mpatches.Patch(color=c_ok,  label="Correct ✓")
    patch_bad = mpatches.Patch(color=c_err, label="Wrong ✗")
    ax2.legend(handles=[patch_ok, patch_bad], facecolor="#12122a", labelcolor="white", fontsize=8)

    # ── Chart 3: Signed error % per stock ────────────────────────────────────
    ax3 = axes[1, 0]
    _ax_style(ax3, f"Signed Prediction Error %  (MAPE={df['abs_err'].mean():.4f}%)")

    err_colors = [c_ok if e >= 0 else c_err for e in df["signed_err"]]
    ax3.bar(syms, df["signed_err"], color=err_colors, alpha=0.85, zorder=3)
    ax3.axhline(0,                  color="white",  linewidth=0.8, linestyle="--", alpha=0.5)
    ax3.axhline(df["signed_err"].mean(), color="#ffd700", linewidth=1.5, linestyle=":",
                label=f"Bias {df['signed_err'].mean():+.3f}%")

    for i, (v, s) in enumerate(zip(df["signed_err"], syms)):
        ax3.text(i, v + (0.005 * (1 if v >= 0 else -1)),
                 f"{v:+.2f}%", ha="center",
                 va="bottom" if v >= 0 else "top",
                 fontsize=8, color="white")

    ax3.set_ylabel("(Actual − Predicted) / Predicted × 100", color="white")
    ax3.set_xticklabels(syms, rotation=30, ha="right", color="white", fontsize=9)
    ax3.legend(facecolor="#12122a", labelcolor="white", fontsize=8)

    # ── Chart 4: Confidence vs actual direction correctness (bubble chart) ───
    ax4 = axes[1, 1]
    _ax_style(ax4, "Confidence Level vs Direction Outcome")

    conf_map = {"High": 3, "Medium": 2, "Low": 1}
    x4 = [conf_map.get(str(r.get("confidence")), 0) + np.random.uniform(-0.15, 0.15)
           for _, r in df.iterrows()]
    y4 = [1 if r.get("direction_correct") else 0 for _, r in df.iterrows()]
    colors4 = [c_ok if v else c_err for v in df["direction_correct"]]

    ax4.scatter(x4, y4, c=colors4, s=200, alpha=0.85, zorder=3, edgecolors="white", linewidths=0.5)
    for xi, yi, sym in zip(x4, y4, syms):
        ax4.text(xi, yi + 0.05, sym, ha="center", fontsize=7, color="white")

    ax4.set_xticks([1, 2, 3])
    ax4.set_xticklabels(["Low", "Medium", "High"], color="white")
    ax4.set_yticks([0, 1])
    ax4.set_yticklabels(["Wrong", "Correct"], color="white")
    ax4.set_xlim(0.5, 3.5)
    ax4.set_ylim(-0.4, 1.4)
    ax4.set_xlabel("LLM Confidence Level", color="white")
    ax4.set_ylabel("Direction Outcome", color="white")
    ax4.axhline(0.5, color="#555577", linewidth=0.7, linestyle="--")

    # ── Summary footer ────────────────────────────────────────────────────────
    dir_acc  = df["direction_correct"].mean()
    mape_val = df["abs_err"].mean()
    bias_val = df["signed_err"].mean()
    rhit_val = df["range_hit"].mean()
    summary = (f"Dir Accuracy: {dir_acc:.1%}   MAPE: {mape_val:.4f}%   "
               f"Bias: {bias_val:+.4f}%   ATR Range Hit: {rhit_val:.1%}")
    fig.text(0.5, 0.01, summary, ha="center", fontsize=10, color="#ffd700",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#12122a",
                       edgecolor="#ffd700", alpha=0.9))

    if chart_file is None:
        chart_file = Path("ft_results.png")
    plt.tight_layout(rect=[0, 0.04, 1, 0.96])
    plt.savefig(chart_file, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"\n✅ Chart saved  : {chart_file.resolve()}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Forward Test Pipeline")
    parser.add_argument("--run",      action="store_true", help="Run LLM agent predictions")
    parser.add_argument("--fetch",    action="store_true", help="Fetch actual prices")
    parser.add_argument("--evaluate", action="store_true", help="Evaluate + plot results")
    parser.add_argument("--stocks",   nargs="+", default=WATCHLIST, help="Override watchlist")
    parser.add_argument("--all",      action="store_true", help="Run all three steps")
    parser.add_argument("--window",   type=int, default=30, choices=[30, 60],
                        help="Prediction horizon in minutes: 30 (default) or 60")
    args = parser.parse_args()

    if args.all:
        args.run = args.fetch = args.evaluate = True

    if not any([args.run, args.fetch, args.evaluate]):
        parser.print_help()
        return

    window = args.window
    PREDICTIONS_FILE = _predictions_file(window)

    if args.run:
        # ── Market hours guard ───────────────────────────────────────────────
        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        latest_start = ist_now.replace(hour=15, minute=30, second=0, microsecond=0) - timedelta(minutes=window)
        market_open  = ist_now.replace(hour=9,  minute=15, second=0, microsecond=0)
        if not (market_open <= ist_now <= latest_start):
            print(f"\n⚠  WARNING: Current IST time is {ist_now.strftime('%H:%M')}.")
            print(f"   {window}-min predictions need live intraday data.")
            print(f"   Best window: 09:15–{latest_start.strftime('%H:%M')} IST.")
            ans = input("   Continue anyway? [y/N]: ").strip().lower()
            if ans != "y":
                print("   Aborted. Run again during market hours.")
                sys.exit(0)
        # ────────────────────────────────────────────────────────────────────

        print(f"\n{'='*60}")
        print(f"NSE {window}-MIN FORWARD TEST — PREDICTIONS")
        print(f"Time   : {ist_now.strftime('%Y-%m-%d %H:%M IST')}")
        print(f"File   : {PREDICTIONS_FILE}")
        print(f"Stocks : {args.stocks}")
        print(f"{'='*60}")

        # Import dependencies (only needed for --run)
        try:
            import autogen
            from autogen import LLMConfig
            from finrobot.toolkits import register_toolkits
            from finrobot.data_source.yfinance_utils import YFinanceUtils
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Make sure autogen and finrobot are installed.")
            sys.exit(1)

        config_list = LLMConfig.from_json(path="../OAI_CONFIG_LIST.json").config_list
        llm_config  = {"config_list": config_list, "timeout": 180, "temperature": 0.1}

        records = []
        for i, symbol in enumerate(args.stocks, 1):
            print(f"\n[{i}/{len(args.stocks)}] Running agent for {symbol}...")
            pred = run_agent_for_stock(symbol, llm_config, YFinanceUtils, window)
            records.append(pred)

            status = "✅ parsed OK" if pred.get("parse_ok") else "⚠ parse incomplete"
            print(f"    {status} — dir={pred.get('direction')}  "
                  f"conf={pred.get('confidence')}  "
                  f"predicted={pred.get('predicted_close')}")

            if i < len(args.stocks):
                print(f"    Waiting 10s before next stock...")
                time.sleep(10)

        # Append to predictions file
        with open(PREDICTIONS_FILE, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"\n{'='*60}")
        print(f"✅ {len(records)} predictions saved to {PREDICTIONS_FILE}")
        print(f"\nPREDICTION SUMMARY:")
        print(f"{'Symbol':<16} {'Direction':<10} {'Entry':>8} {'Predicted':>10} {'Confidence':>10}")
        print("-"*58)
        for r in records:
            ent  = f"{r['entry_price']:.2f}"     if r.get("entry_price")     else "N/A"
            pred = f"{r['predicted_close']:.2f}" if r.get("predicted_close") else "N/A"
            print(f"{r['symbol']:<16} {str(r.get('direction','?')):<10} "
                  f"{ent:>8} {pred:>10} {str(r.get('confidence','?')):>10}")
        print(f"\n→ Run  --fetch --window {window}  in ~{window+5} minutes to capture actuals.")

    if args.fetch:
        fetch_actuals(window)

    if args.evaluate:
        evaluate_and_plot(window)


if __name__ == "__main__":
    main()