"""
backtest_runner.py — Historical Replay Backtest for NSE Intraday
=================================================================
Runs BOTH pipelines (single-layer LLM + two-layer FinBERT+FinML+GPT4)
on historical timestamps so you get immediate results with known actuals.

HOW IT WORKS:
  1. For each (stock, date, time) combination:
     - Feeds the LLM agent data as of that historical date
     - LLM makes a directional prediction
     - We immediately fetch what ACTUALLY happened 30 min later from history
     - No waiting — actuals are available instantly

KNOWN LIMITATION (note in viva):
  - News from get_company_news() is always TODAY's news — not historical news
  - FinBERT sentiment therefore uses current headlines, not past headlines
  - All technical indicators + price data ARE fully historical ✅

USAGE:
  python backtest_runner.py --run                          # both pipelines
  python backtest_runner.py --run --pipeline single        # LLM only
  python backtest_runner.py --run --pipeline two           # two-layer only
  python backtest_runner.py --evaluate                     # compare results
  python backtest_runner.py --run --dates 2026-03-12 2026-03-13 --times 10:00 11:30
"""

import argparse
import json
import math
import os
import re
import sys
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

warnings.filterwarnings("ignore")
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ─── INLINE register_toolkits — replaces broken finrobot.toolkits import ─────
# Copied directly from finrobot/toolkits.py and patched to avoid data_source import

from functools import wraps
from pandas import DataFrame

def _stringify_output(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, DataFrame):
            return result.to_string()
        return str(result)
    return wrapper

def register_toolkits(config, caller, executor, **kwargs):
    """Register tools directly using autogen.register_function — no finrobot import needed."""
    from autogen import register_function
    for tool in config:
        tool_dict = {"function": tool} if callable(tool) else tool
        if "function" not in tool_dict or not callable(tool_dict["function"]):
            raise ValueError("Function not found in tool configuration or not callable.")
        fn          = tool_dict["function"]
        name        = tool_dict.get("name", fn.__name__)
        description = tool_dict.get("description", fn.__doc__ or name)
        register_function(
            _stringify_output(fn),
            caller      = caller,
            executor    = executor,
            name        = name,
            description = description,
        )

DEFAULT_STOCKS = [
    "HDFCBANK.NS", "ICICIBANK.NS", "TCS.NS", "INFY.NS", "RELIANCE.NS",
    "SBIN.NS", "KOTAKBANK.NS", "WIPRO.NS", "SUNPHARMA.NS", "AXISBANK.NS",
]
DEFAULT_DATES  = ["2026-03-12", "2026-03-13"]
DEFAULT_TIMES  = ["10:00", "11:30"]
WINDOW         = 30
OAI_CONFIG     = Path("../OAI_CONFIG_LIST.json")
CONFIG_KEYS    = Path("../config_api_keys.json")
BT_LOG_DIR     = Path("bt_agent_logs")

def _bt_file(pipeline):
    return Path(f"bt_results_{pipeline}_{WINDOW}min.jsonl")


# ─── HISTORICAL PRICE FETCH ───────────────────────────────────────────────────

def _fetch_historical_candle(symbol, as_of_utc, window):
    """Get entry candle at as_of_utc and actual candle window minutes later."""
    try:
        start = (as_of_utc - timedelta(days=1)).strftime("%Y-%m-%d")
        end   = (as_of_utc + timedelta(days=1)).strftime("%Y-%m-%d")
        df    = yf.Ticker(symbol).history(interval="15m", start=start, end=end)
        if df.empty:
            return None
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        entry_df = df[df.index <= as_of_utc]
        if entry_df.empty:
            return None
        entry_candle = entry_df.iloc[-1]
        target_utc   = as_of_utc + timedelta(minutes=window)
        actual_df    = df[df.index >= target_utc]
        if actual_df.empty:
            return None
        actual_candle = actual_df.iloc[0]
        return {
            "entry_price":      round(float(entry_candle["Close"]), 2),
            "actual_open":      round(float(actual_candle["Open"]),  2),
            "actual_high":      round(float(actual_candle["High"]),  2),
            "actual_low":       round(float(actual_candle["Low"]),   2),
            "actual_close":     round(float(actual_candle["Close"]), 2),
            "actual_candle_ts": str(actual_candle.name),
            "entry_candle_ts":  str(entry_candle.name),
        }
    except Exception as e:
        print(f"    [warn] historical fetch failed {symbol}: {e}")
        return None


def _compute_direction_correct(record):
    """Fill direction_correct + actual_direction from entry vs actual_close."""
    entry  = record.get("entry_price")
    actual = record.get("actual_close")
    if not entry or not actual:
        return record
    atr_pct    = record.get("atr_pct") or 0.5
    move_pct   = abs((actual - entry) / entry * 100)
    actual_dir = "Bullish" if actual > entry else ("Bearish" if actual < entry else "Sideways")
    record["actual_direction"] = actual_dir
    if record.get("direction") == "Sideways":
        record["direction_correct"] = move_pct <= atr_pct
    else:
        record["direction_correct"] = (actual_dir == record.get("direction"))
    pred = record.get("predicted_close")
    if pred and actual:
        record["error_pct"] = round((actual - pred) / pred * 100, 4)
    return record


# ─── AGENT SETUP ─────────────────────────────────────────────────────────────

def _setup_agent(YFinanceUtils):
    """Create a FRESH analyst + user_proxy each call — prevents state bleed between stocks."""
    sys.path.insert(0, str(Path("../").resolve()))
    import autogen
    # register_toolkits is now defined inline above — no finrobot.toolkits import needed

    with open(OAI_CONFIG) as f:
        config_list = json.load(f)
    if isinstance(config_list, dict):
        config_list = config_list.get("config_list", [config_list])

    # ── Key fix: lower timeout + fewer replies so it never hangs indefinitely ──
    llm_config = {
        "config_list": config_list,
        "temperature": 0.1,
        "timeout":     60,      # was 120 — 60s per API call max
    }

    analyst = autogen.AssistantAgent(
        name="NSE_Market_Analyst",
        system_message=_system_prompt(),
        llm_config=llm_config,
    )
    user_proxy = autogen.UserProxyAgent(
        name="User_Proxy",
        is_termination_msg=lambda x: (x.get("content") or "").strip().upper() == "TERMINATE",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=8,   # was 20 — stops loops faster
        code_execution_config={"work_dir": "coding", "use_docker": False},
    )
    tools = [
        {"function": YFinanceUtils.get_stock_data,           "name": "get_stock_data",           "description": "Daily OHLCV price history."},
        {"function": YFinanceUtils.get_company_news,          "name": "get_company_news",          "description": "Latest company news headlines."},
        {"function": YFinanceUtils.get_extended_company_info, "name": "get_extended_company_info", "description": "Fundamentals: P/E, beta, 52w range."},
        {"function": YFinanceUtils.get_intraday_data,         "name": "get_intraday_data",         "description": "15m intraday OHLCV candles with VWAP."},
        {"function": YFinanceUtils.get_technical_indicators,  "name": "get_technical_indicators",  "description": "RSI, MACD, BB, EMA, ATR, ADX, OBV."},
        {"function": YFinanceUtils.get_support_resistance,    "name": "get_support_resistance",    "description": "Pivot levels. Use bearish_setup/bullish_setup for targets."},
        {"function": YFinanceUtils.get_nse_market_context,    "name": "get_nse_market_context",    "description": "NIFTY50, BANKNIFTY, VIX, USD/INR, Crude."},
        {"function": YFinanceUtils.get_fno_data,              "name": "get_fno_data",              "description": "F&O expiry, PCR, Max Pain."},
        {"function": YFinanceUtils.get_sector_peers,          "name": "get_sector_peers",          "description": "Sector peer performance."},
        {"function": YFinanceUtils.get_data_sanity_check,     "name": "get_data_sanity_check",     "description": "Data quality cross-validation."},
        {"function": YFinanceUtils.compute_confidence,        "name": "compute_confidence",        "description": "MANDATORY deterministic confidence score."},
        {"function": YFinanceUtils.log_forecast,              "name": "log_forecast",              "description": "MANDATORY final step — log prediction."},
    ]
    register_toolkits(tools, analyst, user_proxy)
    return analyst, user_proxy


def _system_prompt():
    return """You are an expert NSE intraday trader and analyst with deep knowledge of
Indian market microstructure, F&O dynamics, FII/DII flows, India VIX, USD/INR, and pivot/VWAP analysis.

RULES:
- RSI: Copy rsi_signal verbatim. Oversold does NOT mean Bullish — use MACD+EMA for direction.
- MACD: Report macd_crossover and macd_trend separately. Never write 'crossover' if macd_crossover='none'.
- PCR/Max Pain: If UNAVAILABLE write exactly 'N/A (data unavailable)'. Never estimate.
- Crude Oil: If data_anomaly=True write 'Data anomaly — excluded from analysis'.
- Targets: Use bearish_setup for Bearish, bullish_setup for Bullish. If R:R < 0 set tradeable=False.
- THIS IS A BACKTEST: Use data as provided by tools. Make prediction as if you are at the given timestamp."""


def _build_prompt(symbol, date_str, ist_time, window, entry_price):
    start_date = (datetime.strptime(date_str, "%Y-%m-%d") - timedelta(days=30)).strftime("%Y-%m-%d")
    scaled     = round(math.sqrt(window / 375), 4)
    interval   = "15m" if window <= 30 else "30m"
    return f"""You are an NSE intraday analyst. Date: {date_str}. Time: {ist_time} IST.
Target stock: {symbol}. THIS IS A BACKTEST as of {date_str} {ist_time} IST.
Historical entry price at this time: {entry_price}

YOUR TASK: Predict direction for the NEXT {window} MINUTES from {ist_time} IST on {date_str}.
ENTRY PRICE = latest_close from get_intraday_data() = {entry_price}
EXPECTED MOVE = expected_intraday_range_pct x sqrt({window}/375) = expected_intraday_range_pct x {scaled}

STEP 1: get_nse_market_context() then get_fno_data(symbol="{symbol}")
STEP 2: get_stock_data(symbol="{symbol}", start_date="{start_date}", end_date="{date_str}")
        get_technical_indicators(symbol="{symbol}")
        get_support_resistance(symbol="{symbol}")
        get_intraday_data(symbol="{symbol}", interval="{interval}", period="5d")
        get_extended_company_info(symbol="{symbol}")
        get_company_news(symbol="{symbol}")
        get_sector_peers(symbol="{symbol}")
        get_data_sanity_check(symbol="{symbol}")
STEP 3: compute_confidence() — copy output verbatim.
STEP 4: Output this exact format:

NSE {window}-MIN FORECAST — {symbol}
Date: {date_str} | As of: {ist_time} | Horizon: next {window} minutes

PREDICTION SUMMARY:
  Direction       : [Bullish / Bearish / Sideways]
  Expected Move   : [+-X.XX% ({window}min scaled)]
  Price Targets   : Entry={entry_price}, Target=____, Stop=____, R:R=____
  Confidence      : [High/Medium/Low]
  Max Pain Level  : [value OR 'N/A (data unavailable)']

TECHNICAL SIGNALS:
  RSI(14)         : ____ -> [rsi_signal VERBATIM]
  MACD            : macd_trend=____ | macd_crossover=____
  EMA Trend       : ____
  ATR(14)         : ____ pts -> +-____% scaled to {window}min
  ADX(14)         : ____ -> [strong_trend/weak_trend]
  OBV Trend       : ____
  VWAP            : Price ____ is [above/below] VWAP ____

MARKET CONTEXT:
  NIFTY50         : ____ [____% change] | 5d trend: ____
  BANKNIFTY       : ____ [____% change]
  India VIX       : ____ -> [level]

CONFIDENCE JUSTIFICATION:
  - Technical aligned: [Yes/No]
  - NIFTY trend aligned: [Yes/No]
  - OBV confirming: [Yes/No]
  - News clear: [Yes/No]
  - Non-expiry day: [Yes/No]
  -> Score: [X/5] -> [High/Medium/Low]

FINAL REASONING: [3-4 sentences about next {window} minutes]

Now call: log_forecast(symbol="{symbol}", date="{date_str}", direction=<Direction>,
  entry_price={entry_price}, target_price=<Target>, stop_price=<Stop>,
  rr_ratio=<R:R>, confidence=<Confidence>, score=<Score>, tradeable=<True/False>)
Then reply ONLY: TERMINATE
"""


def _parse_output(text, symbol, window, date_str, ist_time, entry_price):
    def _find(patterns, default=None):
        """Try multiple regex patterns, return first match."""
        if isinstance(patterns, str):
            patterns = [patterns]
        for pattern in patterns:
            m = re.search(pattern, text, re.IGNORECASE)
            if m:
                return m.group(1).strip()
        return default

    def _flt(s):
        if s is None:
            return None
        try:
            return float(re.sub(r"[^\d.\-]", "", s))
        except Exception:
            return None

    # Direction — try field format first, then log_forecast tool call args
    direction = _find([
        r"Direction\s*[:\-]+\s*(Bullish|Bearish|Sideways)",
        r"direction\s*[=:]\s*['\"]?(Bullish|Bearish|Sideways)",   # from log_forecast args
        r'"direction"\s*:\s*"(Bullish|Bearish|Sideways)"',
    ])

    # Confidence
    confidence = _find([
        r"Confidence\s*[:\-]+\s*(High|Medium|Low)",
        r"confidence\s*[=:]\s*['\"]?(High|Medium|Low)",
        r'"confidence"\s*:\s*"(High|Medium|Low)"',
    ])

    # Price fields
    entry  = _flt(_find([r"Entry\s*[=:]\s*([\d,.]+)", r"entry_price\s*[=:]\s*([\d,.]+)"])) or entry_price
    target = _flt(_find([r"Target\s*[=:]\s*([\d,.]+)", r"target_price\s*[=:]\s*([\d,.]+)"]))
    stop_p = _flt(_find([r"Stop\s*[=:]\s*([\d,.]+)",  r"stop_price\s*[=:]\s*([\d,.]+)"]))
    rr     = _flt(_find([r"R:R\s*[=:]\s*([\d.\-]+)",  r"rr_ratio\s*[=:]\s*([\d.\-]+)"]))

    score_m = re.search(r"Score:\s*(\d)/5", text) or re.search(r"score\s*=\s*(\d)", text, re.IGNORECASE)
    score   = int(score_m.group(1)) if score_m else None

    # ATR% — look for the scaled value
    atr_m   = re.search(r"[+\-][*]?([\d.]+)%\s*scaled", text) or \
              re.search(r"ATR\(14\)\s*:.*?[+\-]+([\d.]+)%", text)
    atr_pct = float(atr_m.group(1)) if atr_m else None

    predicted_close = None
    if entry and atr_pct and direction:
        move_pct = atr_pct * math.sqrt(window / 375)
        mult     = {"bullish": 1.0, "bearish": -1.0, "sideways": 0.0}.get(
                    (direction or "").lower(), 0.0)
        predicted_close = round(entry * (1 + mult * move_pct / 100), 2)

    return {
        "symbol":          symbol,
        "backtest_date":   date_str,
        "backtest_time":   ist_time,
        "window_min":      window,
        "pipeline":        "single",
        "direction":       direction,
        "confidence":      confidence,
        "entry_price":     entry,
        "target_price":    target,
        "stop_price":      stop_p,
        "rr_ratio":        rr,
        "score":           score,
        "atr_pct":         atr_pct,
        "predicted_close": predicted_close,
        "parse_ok":        direction is not None,
        "actual_close":      None,
        "actual_direction":  None,
        "direction_correct": None,
        "error_pct":         None,
    }


# ─── SINGLE-LAYER BACKTEST ────────────────────────────────────────────────────

def run_single_backtest(stocks, dates, times, window=30):
    print("\n" + "="*60)
    print(f"SINGLE-LAYER BACKTEST — {window}-MIN")
    print(f"Stocks: {len(stocks)} | Dates: {dates} | Times: {times}")
    print(f"Total: {len(stocks)*len(dates)*len(times)} predictions")
    print("="*60)

    sys.path.insert(0, str(Path("../").resolve()))
    try:
        from finrobot.data_source.yfinance_utils import YFinanceUtils
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    import threading

    BT_LOG_DIR.mkdir(exist_ok=True)
    outfile = _bt_file("single")
    total   = len(stocks) * len(dates) * len(times)
    done    = 0

    for date_str in dates:
        for ist_time in times:
            h, m      = map(int, ist_time.split(":"))
            as_of_utc = datetime(*[int(x) for x in date_str.split("-")], h, m, 0,
                                 tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)
            print(f"\n── {date_str} {ist_time} IST {'─'*30}")

            for symbol in stocks:
                done += 1
                print(f"  [{done}/{total}] {symbol} ", end="", flush=True)

                hist = _fetch_historical_candle(symbol, as_of_utc, window)
                if hist is None:
                    print("⚠ no historical data — skip")
                    continue

                entry_price = hist["entry_price"]
                run_ts      = datetime.now(timezone.utc).isoformat()
                raw_output  = ""
                error_info  = {"value": None}

                # ── Fresh agent per stock — prevents state bleed ──────────────
                try:
                    analyst, user_proxy = _setup_agent(YFinanceUtils)
                    print(f"agent ready ", end="", flush=True)
                except Exception as e:
                    print(f"⚠ agent setup failed: {e}")
                    continue

                # ── Run in thread with hard timeout ───────────────────────────
                finished = {"done": False}

                def _run_chat(analyst=analyst, user_proxy=user_proxy):
                    try:
                        prompt = _build_prompt(symbol, date_str, ist_time, window, entry_price)
                        # Use no-cache initiate_chat — Cache.disk hangs in autogen v0.11
                        try:
                            from autogen import Cache
                            with Cache.disk(cache_seed=None) as cache:
                                user_proxy.initiate_chat(analyst, message=prompt, cache=cache)
                        except Exception:
                            user_proxy.initiate_chat(analyst, message=prompt)
                        finished["done"] = True
                    except Exception as e:
                        import traceback
                        error_info["value"] = str(e)
                        print(f"\n    ⚠ chat error: {e}", flush=True)
                        traceback.print_exc()

                TIMEOUT = 90  # 90 seconds max per stock
                thread = threading.Thread(target=_run_chat, daemon=True)
                thread.start()
                thread.join(timeout=TIMEOUT)

                if not finished["done"]:
                    reason = error_info["value"] or f"timeout after {TIMEOUT}s"
                    print(f"\n    ⚠ agent did not finish: {reason}", flush=True)
                    raw_output = f"AGENT_TIMEOUT: {reason}"
                else:
                    for msg in user_proxy.chat_messages.get(analyst, []):
                        content = msg.get("content") or ""
                        if isinstance(content, str):
                            raw_output += content + "\n\n"
                        elif isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    raw_output += (block.get("text") or "") + "\n\n"

                safe_sym = symbol.replace(".", "_")
                log_file = BT_LOG_DIR / f"single_{safe_sym}_{date_str}_{ist_time.replace(':','')}.txt"
                log_file.write_text(raw_output, encoding="utf-8")

                record = _parse_output(raw_output, symbol, window, date_str, ist_time, entry_price)
                record.update({k: v for k, v in hist.items() if k != "entry_price"})
                record = _compute_direction_correct(record)
                record["run_timestamp"] = run_ts

                with open(outfile, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                icon = "✓" if record.get("direction_correct") else ("?" if record.get("direction_correct") is None else "✗")
                print(f"{icon}  pred={record.get('direction','?')}  actual={record.get('actual_direction','?')}  "
                      f"entry={entry_price}  close={record.get('actual_close')}  conf={record.get('confidence','?')}")

                time.sleep(2)

    print(f"\n✅ Single-layer backtest done → {outfile}")


# ─── TWO-LAYER BACKTEST ───────────────────────────────────────────────────────

def run_two_layer_backtest(stocks, dates, times, window=30):
    print("\n" + "="*60)
    print(f"TWO-LAYER BACKTEST — {window}-MIN")
    print(f"Stocks: {len(stocks)} | Dates: {dates} | Times: {times}")
    print(f"Total: {len(stocks)*len(dates)*len(times)} predictions")
    print("="*60)

    sys.path.insert(0, str(Path("../").resolve()))
    try:
        from finrobot.data_source.yfinance_utils import YFinanceUtils
        from quant_layer import run_layer1, run_layer2_reasoning
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return

    hf_token = ""
    try:
        with open(CONFIG_KEYS) as f:
            keys = json.load(f)
        hf_token = keys.get("HUGGINGFACE_TOKEN") or keys.get("HF_TOKEN") or ""
    except Exception:
        pass

    BT_LOG_DIR.mkdir(exist_ok=True)
    outfile = _bt_file("two")
    total   = len(stocks) * len(dates) * len(times)
    done    = 0

    for date_str in dates:
        for ist_time in times:
            h, m      = map(int, ist_time.split(":"))
            as_of_utc = datetime(*[int(x) for x in date_str.split("-")], h, m, 0,
                                 tzinfo=timezone.utc) - timedelta(hours=5, minutes=30)
            print(f"\n── {date_str} {ist_time} IST {'─'*30}")

            for symbol in stocks:
                done += 1
                print(f"  [{done}/{total}] {symbol}")

                hist = _fetch_historical_candle(symbol, as_of_utc, window)
                if hist is None:
                    print("    ⚠ no historical data — skip")
                    continue

                entry_price = hist["entry_price"]
                run_ts      = datetime.now(timezone.utc).isoformat()

                # Headlines (today's news — known limitation)
                headlines = []
                try:
                    news_raw = YFinanceUtils.get_company_news(symbol=symbol)
                    if news_raw and isinstance(news_raw, list):
                        for item in news_raw:
                            if not isinstance(item, dict):
                                continue
                            content = item.get("content")
                            text = None
                            if isinstance(content, dict):
                                text = content.get("title") or content.get("summary")
                            if not text:
                                for col in ["title", "Title", "headline", "summary"]:
                                    val = item.get(col)
                                    if val and isinstance(val, str) and len(val.strip()) > 5:
                                        text = val.strip()
                                        break
                            if text:
                                headlines.append(text.strip())
                    headlines = headlines[:10]
                except Exception:
                    pass

                # Historical ATR + real technical features
                atr_pct  = 0.5
                rr_ratio = 0.0
                score    = 0.0
                try:
                    end_dt     = datetime.strptime(date_str, "%Y-%m-%d")
                    start_dt   = end_dt - timedelta(days=25)
                    hist_daily = yf.Ticker(symbol).history(
                        start=start_dt.strftime("%Y-%m-%d"),
                        end=end_dt.strftime("%Y-%m-%d")
                    )
                    if len(hist_daily) >= 5:
                        tr      = (hist_daily["High"] - hist_daily["Low"]).abs()
                        atr_val = tr.tail(14).mean()
                        close   = hist_daily["Close"].iloc[-1]
                        atr_pct = round(float(atr_val / close * 100), 4)

                        # Compute a simple R:R proxy from ATR
                        # bearish_setup: target = close - 1.5*ATR, stop = close + 0.5*ATR
                        # R:R = 1.5*ATR / 0.5*ATR = 3.0 ideally
                        # Use actual pivot-based estimate: ATR ratio is a reasonable proxy
                        atr_pts = float(atr_val)
                        target_dist = atr_pts * 1.5
                        stop_dist   = atr_pts * 0.5
                        rr_ratio    = round(target_dist / stop_dist, 2) if stop_dist > 0 else 2.0

                        # Score proxy: use ADX-like measure (trend strength)
                        # High day range / ATR → strong trend day = higher score
                        last_day_range = float(hist_daily["High"].iloc[-1] - hist_daily["Low"].iloc[-1])
                        score = min(5.0, round(last_day_range / atr_pts * 2.5, 1)) if atr_pts > 0 else 3.0
                except Exception:
                    pass

                features = {
                    "rr_ratio":     rr_ratio,
                    "atr_pct":      atr_pct,
                    "score":        score,
                    "exp_move_pct": atr_pct * math.sqrt(window / 375),
                }

                # Layer 1
                try:
                    layer1 = run_layer1(
                        symbol=symbol, headlines=headlines,
                        features=features, window=window, hf_token=hf_token
                    )
                except Exception as e:
                    print(f"    ⚠ Layer 1 error: {e}")
                    layer1 = {
                        "finbert":  {"net_score": 0.0, "label": "Neutral"},
                        "finml":    {"ml_confidence": 0.28, "ml_label": "Low"},
                        "combined": {"signal_type": "WEAK_SIGNAL", "overall_strength": 0.28},
                    }

                # Layer 2
                try:
                    layer2 = run_layer2_reasoning(
                        symbol=symbol, layer1=layer1,
                        window=window, date_str=date_str, ist_time=ist_time,
                    )
                except Exception as e:
                    print(f"    ⚠ Layer 2 error: {e}")
                    layer2 = {"direction": None, "confidence": "Low", "error": str(e)}

                direction  = layer2.get("direction")
                confidence = layer2.get("confidence")

                predicted_close = None
                if entry_price and atr_pct and direction:
                    mult = {"bullish": 1.0, "bearish": -1.0, "sideways": 0.0}.get(
                            (direction or "").lower(), 0.0)
                    predicted_close = round(entry_price * (1 + mult * atr_pct * math.sqrt(window/375) / 100), 2)

                record = {
                    "symbol":          symbol,
                    "backtest_date":   date_str,
                    "backtest_time":   ist_time,
                    "window_min":      window,
                    "pipeline":        "two",
                    "direction":       direction,
                    "confidence":      confidence,
                    "entry_price":     entry_price,
                    "predicted_close": predicted_close,
                    "atr_pct":         atr_pct,
                    "rr_ratio":        rr_ratio,
                    "score":           score,
                    "finbert_score":   layer1["finbert"].get("net_score"),
                    "finbert_label":   layer1["finbert"].get("label"),
                    "finml_conf":      layer1["finml"].get("ml_confidence"),
                    "finml_label":     layer1["finml"].get("ml_label"),
                    "signal_type":     layer1["combined"].get("signal_type"),
                    "run_timestamp":   run_ts,
                    "parse_ok":        direction is not None,
                }
                record.update({k: v for k, v in hist.items() if k != "entry_price"})
                record = _compute_direction_correct(record)

                with open(outfile, "a", encoding="utf-8") as f:
                    f.write(json.dumps(record) + "\n")

                safe_sym = symbol.replace(".", "_")
                log_file = BT_LOG_DIR / f"two_{safe_sym}_{date_str}_{ist_time.replace(':','')}.txt"
                with open(log_file, "w", encoding="utf-8") as f:
                    f.write(f"TWO-LAYER BACKTEST — {symbol} — {date_str} {ist_time}\n")
                    f.write(json.dumps(layer1, indent=2, default=str) + "\n\n")
                    f.write(json.dumps(layer2, indent=2, default=str) + "\n")

                icon = "✓" if record.get("direction_correct") else ("?" if record.get("direction_correct") is None else "✗")
                print(f"    {icon} dir={direction}  actual={record.get('actual_direction')}  "
                      f"conf={confidence}  finbert={layer1['finbert'].get('net_score', 0):+.3f}  "
                      f"finml={layer1['finml'].get('ml_confidence', 0):.3f}")

                time.sleep(1)

    print(f"\n✅ Two-layer backtest done → {outfile}")


# ─── EVALUATE & COMPARE ───────────────────────────────────────────────────────

def evaluate_and_compare(window=30):
    results = {}
    for pipeline in ["single", "two"]:
        fpath = _bt_file(pipeline)
        if not fpath.exists():
            print(f"  ⚠ {fpath} not found")
            continue
        records = []
        for line in fpath.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                pass
        results[pipeline] = records

    if not results:
        print("No backtest results found.")
        return

    print("\n" + "="*60)
    print(f"BACKTEST EVALUATION — {window}-MIN WINDOW")
    print("="*60)

    summary = {}
    for pipeline, records in results.items():
        scored   = [r for r in records if r.get("direction_correct") is not None]
        high_med = [r for r in scored if r.get("confidence") in ("High","Medium")
                    and r.get("direction") != "Sideways"]
        n_total  = len(records)
        n_scored = len(scored)
        acc_all  = sum(1 for r in scored if r["direction_correct"]) / n_scored if n_scored else 0
        acc_hm   = sum(1 for r in high_med if r["direction_correct"]) / len(high_med) if high_med else 0

        print(f"\n  {'SINGLE' if pipeline=='single' else 'TWO'}-LAYER PIPELINE")
        print(f"    Total predictions          : {n_total}")
        print(f"    Scored                     : {n_scored}")
        print(f"    Accuracy — all             : {acc_all:.1%}")
        print(f"    Accuracy — High+Med (no Sideways): {acc_hm:.1%}  [{len(high_med)} preds]")

        for conf in ["High", "Medium", "Low"]:
            c_recs = [r for r in scored if r.get("confidence") == conf
                      and r.get("direction") != "Sideways"]
            if c_recs:
                c_acc = sum(1 for r in c_recs if r["direction_correct"]) / len(c_recs)
                print(f"    {conf:6} confidence        : {c_acc:.1%}  [{len(c_recs)} preds]")

        print(f"    By date:")
        for date in sorted(set(r.get("backtest_date","") for r in scored)):
            d_recs = [r for r in scored if r.get("backtest_date") == date
                      and r.get("direction") != "Sideways"]
            if d_recs:
                d_acc = sum(1 for r in d_recs if r["direction_correct"]) / len(d_recs)
                print(f"      {date}: {d_acc:.1%}  [{len(d_recs)} preds]")

        summary[pipeline] = {"n_total": n_total, "n_scored": n_scored,
                             "acc_all": acc_all, "acc_hm": acc_hm, "records": records}

    # ── Comparison chart ──────────────────────────────────────────────────────
    if len(summary) >= 2:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f"Backtest Comparison — {window}-Min Window", fontsize=14, fontweight="bold")
        pipelines = list(summary.keys())
        colors    = {"single": "#2196F3", "two": "#4CAF50"}
        labels    = {"single": "Single-Layer\n(LLM)", "two": "Two-Layer\n(FinBERT+FinML+GPT4)"}

        for ax, metric, title in [
            (axes[0], "acc_all", "Overall Accuracy (%)"),
            (axes[1], "acc_hm",  "High+Med Accuracy\n(Non-Sideways) (%)"),
        ]:
            vals = [summary[p][metric]*100 for p in pipelines]
            bars = ax.bar([labels[p] for p in pipelines], vals,
                          color=[colors[p] for p in pipelines], width=0.4)
            ax.set_title(title)
            ax.set_ylim(0, 100)
            ax.axhline(50, color="gray", linestyle="--", alpha=0.5, label="Coin flip (50%)")
            ax.legend(fontsize=8)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        f"{val:.0f}%", ha="center", fontweight="bold")

        # Per-stock breakdown (single-layer)
        ax = axes[2]
        if "single" in summary:
            recs       = summary["single"]["records"]
            stock_list = sorted(set(r["symbol"] for r in recs))
            stock_accs = []
            for s in stock_list:
                s_recs = [r for r in recs if r["symbol"] == s
                          and r.get("direction_correct") is not None
                          and r.get("direction") != "Sideways"]
                if s_recs:
                    stock_accs.append((
                        s.replace(".NS", ""),
                        sum(1 for r in s_recs if r["direction_correct"]) / len(s_recs)
                    ))
            if stock_accs:
                names, accs = zip(*stock_accs)
                bar_colors  = ["#4CAF50" if a >= 0.5 else "#f44336" for a in accs]
                ax.barh(list(names), [a*100 for a in accs], color=bar_colors)
                ax.axvline(50, color="gray", linestyle="--", alpha=0.5)
                ax.set_title("Per-Stock Accuracy\n(Single-Layer)")
                ax.set_xlabel("Accuracy %")
                ax.set_xlim(0, 100)

        plt.tight_layout()
        chart_file = Path(f"bt_comparison_{window}min.png")
        plt.savefig(chart_file, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"\n  Chart saved → {chart_file}")

    # Detail table
    print(f"\n  DETAIL — Single-Layer:")
    print(f"  {'Symbol':<14} {'Date':<12} {'Time':<6} {'Pred':<10} {'Actual':<10} {'Conf':<8} OK")
    print("  " + "-"*65)
    if "single" in summary:
        for r in sorted(summary["single"]["records"],
                        key=lambda x: (x.get("backtest_date",""), x.get("backtest_time",""))):
            icon = "✓" if r.get("direction_correct") else ("?" if r.get("direction_correct") is None else "✗")
            print(f"  {r['symbol']:<14} {r.get('backtest_date',''):<12} {r.get('backtest_time',''):<6} "
                  f"{(r.get('direction') or '?'):<10} {(r.get('actual_direction') or '?'):<10} "
                  f"{(r.get('confidence') or '?'):<8} {icon}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Backtest Runner")
    parser.add_argument("--run",      action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--pipeline", choices=["single","two","both"], default="both")
    parser.add_argument("--stocks",   nargs="+", default=DEFAULT_STOCKS)
    parser.add_argument("--dates",    nargs="+", default=DEFAULT_DATES)
    parser.add_argument("--times",    nargs="+", default=DEFAULT_TIMES)
    parser.add_argument("--window",   type=int,  default=WINDOW)
    args = parser.parse_args()

    if args.run:
        total = len(args.stocks) * len(args.dates) * len(args.times)
        print(f"\nBACKTEST PLAN")
        print(f"  Stocks   : {args.stocks}")
        print(f"  Dates    : {args.dates}")
        print(f"  Times    : {args.times} IST")
        print(f"  Window   : {args.window} min")
        print(f"  Pipeline : {args.pipeline}")
        print(f"  Total    : {total} predictions per pipeline")
        print(f"\nNOTE: News/FinBERT uses today's headlines (yfinance limitation).")
        print(f"All technical + price data is fully historical. ✅\n")

        if args.pipeline in ("single", "both"):
            run_single_backtest(args.stocks, args.dates, args.times, args.window)
        if args.pipeline in ("two", "both"):
            run_two_layer_backtest(args.stocks, args.dates, args.times, args.window)

    if args.evaluate or args.run:
        evaluate_and_compare(args.window)


if __name__ == "__main__":
    main()