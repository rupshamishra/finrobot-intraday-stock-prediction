"""
two_layer_agent.py — True Tool-Based Two-Layer NSE Forecasting Pipeline
========================================================================
Architecture: FinBERT + XGBoost registered as AutoGen tools alongside
the existing 14 FinRobot data tools. The GPT-4 agent calls ALL tools
in a genuine Chain-of-Thought sequence — it decides when to call FinBERT
and FinML based on what it has learned from prior tool calls.

This is the proper FinRobot design philosophy:
    LLM Agent + Registered Tools = Agent-driven CoT reasoning

PIPELINE (agent-driven, not pre-computed):
    STEP 1  get_nse_market_context() + get_fno_data()
    STEP 2  get_stock_data() + get_technical_indicators() +
            get_support_resistance() + get_intraday_data() +
            get_extended_company_info() + get_sector_peers() +
            get_data_sanity_check()
    STEP 3  get_company_news() → run_finbert_sentiment(headlines)
    STEP 4  run_finml_predict(rr_ratio, atr_pct, score, exp_move_pct)
    STEP 5  compute_confidence() + full CoT synthesis
    STEP 6  log_forecast()

VS OLD ARCHITECTURE:
    Old: pre-compute FinBERT+FinML outside agent → dump into one prompt
    New: agent calls FinBERT+FinML as tools → real multi-step reasoning

USAGE:
    python two_layer_agent.py --run --window 30
    python two_layer_agent.py --run --window 30 --stocks HDFCBANK.NS TCS.NS
    python two_layer_agent.py --fetch --window 30
    python two_layer_agent.py --evaluate --window 30
    python two_layer_agent.py --compare --window 30
"""

import sys
import json
import math
import time
import argparse
import warnings
import re
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Annotated

import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─── CONFIG ──────────────────────────────────────────────────────────────────

WATCHLIST = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "RELIANCE.NS", "ONGC.NS",      "NTPC.NS",      "POWERGRID.NS","COALINDIA.NS",
    "TCS.NS",      "INFY.NS",      "WIPRO.NS",     "HCLTECH.NS",  "TECHM.NS",
    "SUNPHARMA.NS","DRREDDY.NS",   "CIPLA.NS",     "APOLLOHOSP.NS","LUPIN.NS",
]

OAI_CONFIG  = Path("../OAI_CONFIG_LIST.json")
CONFIG_KEYS = Path("../config_api_keys.json")
LOG_DIR     = Path("ft_agent_logs_2layer_v2")


def _pred_file(window):
    return Path(f"ft_predictions_2layer_v2_{window}min.jsonl")


def _res_file(window):
    return Path(f"ft_results_2layer_v2_{window}min.csv")


# ─── TOOL 1: FINBERT SENTIMENT ───────────────────────────────────────────────

def _load_hf_token() -> str:
    try:
        with open(CONFIG_KEYS) as f:
            keys = json.load(f)
        return keys.get("HUGGINGFACE_TOKEN") or keys.get("HF_TOKEN") or ""
    except Exception:
        return ""


def run_finbert_sentiment(
    headlines: Annotated[list, "List of news headline strings to score"]
) -> dict:
    """
    TOOL: Run FinBERT sentiment analysis on news headlines.
    Uses ProsusAI/finbert via HuggingFace Inference API.
    Returns net_score (-1 to +1), label (Bullish/Neutral/Bearish),
    and per-headline breakdown. Call this AFTER get_company_news().
    """
    import requests
    import time as _time

    HF_URL   = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
    hf_token = _load_hf_token()

    if not headlines:
        return {
            "net_score":        0.0,
            "label":            "Neutral",
            "headlines_scored": 0,
            "error":            "no headlines provided",
            "bullish":          0,
            "bearish":          0,
            "neutral":          0,
        }

    if not hf_token:
        return {
            "net_score":        0.0,
            "label":            "Neutral",
            "headlines_scored": 0,
            "error":            "no HuggingFace token in config_api_keys.json",
            "bullish":          0,
            "bearish":          0,
            "neutral":          0,
        }

    clean   = [h.strip()[:256] for h in headlines if isinstance(h, str) and h.strip()][:10]
    headers = {"Authorization": f"Bearer {hf_token}", "Content-Type": "application/json"}

    try:
        resp = requests.post(HF_URL, headers=headers, json={"inputs": clean}, timeout=30)
        if resp.status_code == 503:
            _time.sleep(20)
            resp = requests.post(HF_URL, headers=headers, json={"inputs": clean}, timeout=30)

        if resp.status_code != 200:
            return {
                "net_score":        0.0,
                "label":            "Neutral",
                "headlines_scored": 0,
                "error":            f"HF API {resp.status_code}",
                "bullish":          0,
                "bearish":          0,
                "neutral":          0,
            }

        results = resp.json()
        pos_scores, neg_scores = [], []
        per_hl, bull, bear, neut = [], 0, 0, 0

        for i, hr in enumerate(results):
            if isinstance(hr, dict):
                hr = [hr]
            scores = {item["label"].lower(): item["score"] for item in hr}
            pos    = scores.get("positive", 0.0)
            neg    = scores.get("negative", 0.0)
            pos_scores.append(pos)
            neg_scores.append(neg)
            lbl = max(scores, key=scores.get)
            if   lbl == "positive": bull += 1
            elif lbl == "negative": bear += 1
            else:                   neut += 1
            per_hl.append({
                "headline": clean[i][:80],
                "label":    lbl,
                "positive": round(pos, 3),
                "negative": round(neg, 3),
            })

        net = round(float(np.mean(pos_scores)) - float(np.mean(neg_scores)), 4)
        lbl = "Bullish" if net >= 0.15 else ("Bearish" if net <= -0.15 else "Neutral")

        return {
            "net_score":        net,
            "label":            lbl,
            "headlines_scored": len(per_hl),
            "bullish":          bull,
            "bearish":          bear,
            "neutral":          neut,
            "per_headline":     per_hl,
            "error":            None,
        }

    except Exception as e:
        return {
            "net_score":        0.0,
            "label":            "Neutral",
            "headlines_scored": 0,
            "error":            str(e)[:100],
            "bullish":          0,
            "bearish":          0,
            "neutral":          0,
        }


# ─── TOOL 2: FINML (XGBOOST) PREDICT ────────────────────────────────────────

_xgb_model_cache = {}


def _get_training_files(window):
    candidates = [
        Path(f"bt_results_single_{window}min.jsonl"),
        Path(f"ft_predictions_{window}min.jsonl"),
        Path(f"ft_predictions_2layer_{window}min.jsonl"),
        Path(f"ft_predictions_2layer_v2_{window}min.jsonl"),
    ]
    return [p for p in candidates if p.exists()]


def _extract_feat(r):
    rr = r.get("rr_ratio") or 0.0
    if abs(rr) > 50:
        rr = 0.0
    atr   = r.get("atr_pct") or 0.5
    score = r.get("score") or 0.0
    if score == 0:
        score = {"High": 4, "Medium": 3, "Low": 2}.get(r.get("confidence"), 0)
    exp  = r.get("exp_move_pct") or (atr * 0.283)
    hour = 11.0
    ts   = r.get("run_timestamp") or r.get("logged_at")
    if ts:
        try:
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ist  = dt + timedelta(hours=5, minutes=30)
            hour = ist.hour + ist.minute / 60
        except Exception:
            pass
    return [
        float(rr),
        float(atr),
        float(score),
        float(exp),
        float(hour),
        1.0 if hour < 11.5 else 0.0,
    ]


def _train_xgb(window):
    try:
        from xgboost import XGBClassifier
    except ImportError:
        return None, 0, "xgboost not installed — pip install xgboost"

    files = _get_training_files(window)
    if not files:
        return None, 0, "no training files found"

    records = []
    for fp in files:
        try:
            for line in fp.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r       = json.loads(line)
                    dir_cap = (r.get("direction") or "").capitalize()
                    if (
                        r.get("actual_close") is not None
                        and r.get("direction_correct") is not None
                        and dir_cap not in ("", "Sideways")
                        and r.get("confidence") in ("High", "Medium")
                    ):
                        records.append(r)
                except Exception:
                    pass
        except Exception:
            pass

    if len(records) < 10:
        return None, len(records), f"only {len(records)} training samples (need 10)"

    X        = np.array([_extract_feat(r) for r in records])
    y        = np.array([1 if r["direction_correct"] else 0 for r in records])
    pos_rate = y.mean()
    scale    = (1 - pos_rate) / pos_rate if pos_rate > 0 else 1.0

    model = XGBClassifier(
        n_estimators     = 100,
        max_depth        = 3,
        learning_rate    = 0.1,
        subsample        = 0.8,
        colsample_bytree = 0.8,
        scale_pos_weight = scale,
        use_label_encoder= False,
        eval_metric      = "logloss",
        random_state     = 42,
        verbosity        = 0,
    )
    model.fit(X, y)

    try:
        from sklearn.model_selection import cross_val_score
        n_splits = min(5, len(records) // 8)
        if n_splits >= 3:
            cv_acc = round(float(cross_val_score(model, X, y, cv=n_splits).mean()), 4)
        else:
            cv_acc = round(float((y == model.predict(X)).mean()), 4)
    except Exception:
        cv_acc = None

    feat_names = ["rr_ratio", "atr_pct", "score", "exp_move_pct", "hour_ist", "is_morning"]
    importance = {
        k: round(float(v), 4)
        for k, v in zip(feat_names, model.feature_importances_)
    }

    print(
        f"    ✅ XGBoost trained: {len(records)} samples, cv_acc={cv_acc}, "
        f"pos_rate={pos_rate:.0%}, top_feat={max(importance, key=importance.get)}"
    )

    return model, len(records), None, cv_acc, importance


def run_finml_predict(
    rr_ratio:     Annotated[float, "Risk-reward ratio from get_support_resistance()"],
    atr_pct:      Annotated[float, "ATR as % of price from get_technical_indicators()"],
    score:        Annotated[float, "Confidence score 1-5 from compute_confidence()"],
    exp_move_pct: Annotated[float, "Expected move % scaled to prediction window"],
    window:       Annotated[int,   "Prediction window in minutes (30 or 60)"] = 30,
) -> dict:
    """
    TOOL: Run XGBoost FinML classifier to estimate probability this prediction
    will be direction_correct. Trained on historical NSE predictions.
    Call this AFTER get_technical_indicators() and get_support_resistance()
    so you have real rr_ratio, atr_pct, and score values.
    Returns ml_probability (0-1), ml_label (High/Medium/Low),
    feature_importance dict, and training_samples count.
    """
    cache_key = f"{window}min_xgb_v1"

    if cache_key not in _xgb_model_cache:
        result = _train_xgb(window)
        if result[0] is None:
            _xgb_model_cache[cache_key] = {
                "model":  None,
                "n":      result[1],
                "reason": result[2],
            }
        else:
            model, n, _, cv_acc, importance = result
            _xgb_model_cache[cache_key] = {
                "model":      model,
                "n":          n,
                "cv_acc":     cv_acc,
                "importance": importance,
            }

    cached = _xgb_model_cache[cache_key]

    if cached["model"] is None:
        # Rule-based fallback
        prob = 0.5
        if abs(rr_ratio) <= 50 and rr_ratio >= 2.0:
            prob += 0.10
        elif rr_ratio >= 1.0:
            prob += 0.05
        elif rr_ratio <= 0.3:
            prob -= 0.10
        if   score >= 4:         prob += 0.08
        elif score <= 2:         prob -= 0.08
        if   exp_move_pct > 1.5: prob -= 0.06
        prob = round(max(0.15, min(0.85, prob)), 4)
        lbl  = "High" if prob >= 0.65 else ("Medium" if prob >= 0.45 else "Low")
        return {
            "ml_probability":   prob,
            "ml_label":         lbl,
            "model_type":       f"RuleBased ({cached.get('reason', 'insufficient data')})",
            "training_samples": cached.get("n", 0),
            "feature_importance": {},
            "cv_accuracy":      None,
            "error":            cached.get("reason"),
        }

    model      = cached["model"]
    ist_now    = datetime.utcnow() + timedelta(hours=5, minutes=30)
    hour_ist   = ist_now.hour + ist_now.minute / 60
    is_morning = 1.0 if hour_ist < 11.5 else 0.0
    rr_safe    = 0.0 if abs(rr_ratio) > 50 else rr_ratio

    X    = np.array([[rr_safe, atr_pct, score, exp_move_pct, hour_ist, is_morning]])
    prob = round(float(model.predict_proba(X)[0][1]), 4)
    lbl  = "High" if prob >= 0.65 else ("Medium" if prob >= 0.45 else "Low")

    return {
        "ml_probability":   prob,
        "ml_label":         lbl,
        "model_type":       "XGBoost (xgboost)",
        "training_samples": cached["n"],
        "feature_importance": cached.get("importance", {}),
        "cv_accuracy":      cached.get("cv_acc"),
        "error":            None,
    }


# ─── AGENT SETUP ─────────────────────────────────────────────────────────────

def _build_system_prompt():
    return """You are an expert NSE (National Stock Exchange of India) intraday quantitative
analyst with deep knowledge of Indian market microstructure, F&O dynamics, FII/DII flows,
India VIX, USD/INR impact, crude oil, pivot levels and VWAP.

You have access to 16 tools — 14 market data tools plus FinBERT sentiment and XGBoost FinML.
You MUST call ALL relevant tools and reason step-by-step across their outputs.

TOOL CALLING RULES:
- Always call get_nse_market_context() first for macro backdrop
- Always call get_technical_indicators() before run_finml_predict()
  so you have real rr_ratio, atr_pct, and score values
- Always call get_company_news() before run_finbert_sentiment()
  and pass the actual headlines list to it
- Always call compute_confidence() with raw values — copy output verbatim
- Always end with log_forecast() then reply TERMINATE

RSI RULE: Copy rsi_signal verbatim. Oversold ≠ Bullish. Use MACD+EMA for direction.
MACD RULE: Report macd_crossover and macd_trend separately.
PCR/Max Pain: Write 'N/A (data unavailable)' if UNAVAILABLE — never estimate.
Crude Oil: Write 'Data anomaly — excluded' if data_anomaly=True.
R:R RULE: If R:R < 0 set tradeable=False.
"""


def _build_agent_prompt(symbol: str, window: int) -> str:
    today      = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=30)).strftime("%Y-%m-%d")
    ist_now    = (datetime.utcnow() + timedelta(hours=5, minutes=30)).strftime("%H:%M IST")
    scaled     = round(math.sqrt(window / 375), 4)
    interval   = "15m" if window <= 30 else "30m"

    return f"""You are an NSE intraday analyst. Today: {today}. Time: {ist_now}.
Target: {symbol}. Predict direction for the NEXT {window} MINUTES.

EXPECTED MOVE = expected_intraday_range_pct × √({window}/375) = × {scaled}

━━━ STEP 1 — MACRO BACKDROP ━━━
Call: get_nse_market_context()
Call: get_fno_data(symbol="{symbol}")

━━━ STEP 2 — STOCK ANALYSIS ━━━
Call: get_stock_data(symbol="{symbol}", start_date="{start_date}", end_date="{today}")
Call: get_technical_indicators(symbol="{symbol}")
Call: get_support_resistance(symbol="{symbol}")
Call: get_intraday_data(symbol="{symbol}", interval="{interval}", period="5d")
Call: get_extended_company_info(symbol="{symbol}")
Call: get_sector_peers(symbol="{symbol}")
Call: get_data_sanity_check(symbol="{symbol}")

━━━ STEP 3 — NEWS SENTIMENT (FinBERT) ━━━
Call: get_company_news(symbol="{symbol}")
Then call: run_finbert_sentiment(headlines=[<list of headline strings from above>])
Interpret: What does news sentiment say vs technical picture? Agreement or conflict?

━━━ STEP 4 — ML SIGNAL (XGBoost FinML) ━━━
Call: run_finml_predict(
    rr_ratio=<from get_support_resistance bearish_setup or bullish_setup>,
    atr_pct=<atr_pct from get_technical_indicators>,
    score=<score from compute_confidence or estimate 1-5>,
    exp_move_pct=<expected_intraday_range_pct × {scaled}>,
    window={window}
)
Interpret: What does the historical ML model say about this setup quality?

━━━ STEP 5 — CONFIDENCE ━━━
Call: compute_confidence() with raw values — copy output verbatim.

━━━ STEP 6 — OUTPUT ━━━
Fill this exact format:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE {window}-MIN FORECAST — {symbol}
Date: {today} | As of: {ist_now} | Horizon: next {window} minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

PREDICTION SUMMARY:
  Direction       : [Bullish / Bearish / Sideways]
  Expected Move   : [±X.XX% ({window}min scaled)]
  Price Targets   : Entry=____, Target=____, Stop=____, R:R=____
  Confidence      : [High/Medium/Low]
  Max Pain Level  : [value OR 'N/A (data unavailable)']

QUANTITATIVE SIGNALS:
  FinBERT Sentiment : [net_score] → [label] ([N] headlines scored)
  FinBERT Insight   : [1 sentence — does news support or contradict technicals?]
  XGBoost FinML     : prob=[probability] → [label] (trained on [N] samples)
  FinML Insight     : [1 sentence — what does historical accuracy say about this setup?]
  Signal Agreement  : [STRONG_BUY / STRONG_SELL / TECHNICAL_ONLY / SENTIMENT_ONLY / CONFLICT / WEAK]

TECHNICAL SIGNALS:
  RSI(14)   : ____ → [rsi_signal VERBATIM]
  MACD      : macd_trend=____ | macd_crossover=____
  EMA Trend : ____
  ATR(14)   : ____ pts → ±____% scaled to {window}min
  ADX(14)   : ____ → [strong/weak trend]
  OBV       : ____
  VWAP      : Price ____ is [above/below] VWAP ____

MARKET CONTEXT:
  NIFTY50   : ____ [____% change] | 5d trend: ____
  BANKNIFTY : ____ [____% change]
  India VIX : ____ → [level]

CONFIDENCE JUSTIFICATION:
  - Technical aligned (MACD+EMA): [Yes/No]
  - NIFTY trend aligned:          [Yes/No]
  - OBV confirming:               [Yes/No]
  - News/FinBERT aligned:         [Yes/No]
  - Non-expiry day:               [Yes/No]
  → Score: [X/5] → [High/Medium/Low]

FINAL REASONING:
[4-5 sentences synthesizing technical signals, FinBERT sentiment, XGBoost probability,
and market context. Explain how ALL signals combine to support this prediction.]

Call: log_forecast(symbol="{symbol}", date="{today}", direction=<Direction>,
    entry_price=<Entry>, target_price=<Target>, stop_price=<Stop>,
    rr_ratio=<R:R>, confidence=<Confidence>, score=<Score>, tradeable=<True/False>)
Then reply ONLY: TERMINATE
"""


# ─── RUN AGENT FOR ONE STOCK ─────────────────────────────────────────────────

def run_agent(symbol: str, window: int, YFinanceUtils) -> dict:
    import threading

    print(f"🚀 Starting agent for {symbol}", flush=True)

    # ── Step 1: import AutoGen ────────────────────────────────────────────────
    try:
        import autogen
        print(f"    ✅ autogen imported (version: {getattr(autogen, '__version__', 'unknown')})",
              flush=True)
    except ImportError as e:
        print(f"    ❌ FATAL: cannot import autogen → {e}", flush=True)
        print("    → Fix: pip install pyautogen", flush=True)
        return {"symbol": symbol, "direction": "UNKNOWN",
                "confidence": "Low", "error": f"autogen import failed: {e}"}

    # ── Step 2: import register_toolkits ─────────────────────────────────────
    try:
        from finrobot.toolkits import register_toolkits
        print("    ✅ finrobot.toolkits imported", flush=True)
    except ImportError as e:
        print(f"    ❌ FATAL: cannot import finrobot.toolkits → {e}", flush=True)
        print("    → Fix: make sure finrobot is installed and you ran from tutorials_advanced/",
              flush=True)
        return {"symbol": symbol, "direction": "UNKNOWN",
                "confidence": "Low", "error": f"finrobot import failed: {e}"}

    # ── Step 3: load OAI config ───────────────────────────────────────────────
    oai_path = OAI_CONFIG
    # Also try current directory as fallback
    if not oai_path.exists():
        oai_path = Path("OAI_CONFIG_LIST.json")
    if not oai_path.exists():
        oai_path = Path("../OAI_CONFIG_LIST.json")

    if not oai_path.exists():
        print(f"    ❌ FATAL: OAI_CONFIG_LIST.json not found!", flush=True)
        print(f"    → Searched: {OAI_CONFIG}, OAI_CONFIG_LIST.json, ../OAI_CONFIG_LIST.json",
              flush=True)
        return {"symbol": symbol, "direction": "UNKNOWN",
                "confidence": "Low", "error": "OAI_CONFIG_LIST.json not found"}

    print(f"    ✅ Loading config from: {oai_path.resolve()}", flush=True)
    try:
        with open(oai_path) as f:
            cfg = json.load(f)
        if isinstance(cfg, dict):
            cfg = cfg.get("config_list", [cfg])
        print(f"    ✅ LLM config loaded: {len(cfg)} model(s) — "
              f"first model: {cfg[0].get('model', '?') if cfg else 'none'}", flush=True)
    except Exception as e:
        print(f"    ❌ FATAL: failed to parse {oai_path} → {e}", flush=True)
        return {"symbol": symbol, "direction": "UNKNOWN",
                "confidence": "Low", "error": f"config parse failed: {e}"}

    # ── Step 4: build agents ──────────────────────────────────────────────────
    llm_config = {
        "config_list": cfg,
        "temperature": 0.1,
        "timeout":     120,      # raised from 60 → 120s for slow API
    }

    try:
        analyst = autogen.AssistantAgent(
            name           = "NSE_Quant_Analyst",
            system_message = _build_system_prompt(),
            llm_config     = llm_config,
        )
        user_proxy = autogen.UserProxyAgent(
            name                       = "User_Proxy",
            is_termination_msg         = lambda x: (x.get("content") or "").strip().upper() == "TERMINATE",
            human_input_mode           = "NEVER",
            max_consecutive_auto_reply = 20,
            code_execution_config      = {"work_dir": "coding", "use_docker": False},
        )
        print("    ✅ Agents created", flush=True)
    except Exception as e:
        print(f"    ❌ FATAL: agent creation failed → {e}", flush=True)
        return {"symbol": symbol, "direction": "UNKNOWN",
                "confidence": "Low", "error": f"agent creation failed: {e}"}

    # ── Step 5: register tools ────────────────────────────────────────────────
    tools = [
        # ── 14 existing FinRobot data tools ──────────────────────────────────
        {
            "function":    YFinanceUtils.get_stock_data,
            "name":        "get_stock_data",
            "description": "Daily OHLCV price history.",
        },
        {
            "function":    YFinanceUtils.get_company_news,
            "name":        "get_company_news",
            "description": "Latest company news headlines. Returns list of strings.",
        },
        {
            "function":    YFinanceUtils.get_extended_company_info,
            "name":        "get_extended_company_info",
            "description": "NSE fundamentals: P/E, beta, 52w range.",
        },
        {
            "function":    YFinanceUtils.get_intraday_data,
            "name":        "get_intraday_data",
            "description": "15m intraday OHLCV candles with VWAP.",
        },
        {
            "function":    YFinanceUtils.get_technical_indicators,
            "name":        "get_technical_indicators",
            "description": "RSI, MACD, BB, EMA, ATR, ADX, OBV. Includes rr_ratio and atr_pct.",
        },
        {
            "function":    YFinanceUtils.get_support_resistance,
            "name":        "get_support_resistance",
            "description": "Pivot levels. bearish_setup and bullish_setup contain rr_ratio.",
        },
        {
            "function":    YFinanceUtils.get_nse_market_context,
            "name":        "get_nse_market_context",
            "description": "NIFTY50, BANKNIFTY, VIX, USD/INR, Crude Oil.",
        },
        {
            "function":    YFinanceUtils.get_fno_data,
            "name":        "get_fno_data",
            "description": "F&O expiry, PCR, Max Pain.",
        },
        {
            "function":    YFinanceUtils.get_sector_peers,
            "name":        "get_sector_peers",
            "description": "Sector peer performance comparison.",
        },
        {
            "function":    YFinanceUtils.get_data_sanity_check,
            "name":        "get_data_sanity_check",
            "description": "Data quality cross-validation.",
        },
        {
            "function":    YFinanceUtils.compute_confidence,
            "name":        "compute_confidence",
            "description": "MANDATORY: deterministic 5-factor confidence score. Copy output verbatim.",
        },
        {
            "function":    YFinanceUtils.log_forecast,
            "name":        "log_forecast",
            "description": "MANDATORY final step: log prediction to forecasts.jsonl.",
        },
        # ── 2 NEW quantitative tools ─────────────────────────────────────────
        {
            "function":    run_finbert_sentiment,
            "name":        "run_finbert_sentiment",
            "description": (
                "MANDATORY STEP 3: Run FinBERT sentiment on news headlines. "
                "Pass the list of headline strings from get_company_news(). "
                "Returns net_score (-1 to +1) and label (Bullish/Neutral/Bearish)."
            ),
        },
        {
            "function":    run_finml_predict,
            "name":        "run_finml_predict",
            "description": (
                "MANDATORY STEP 4: Run XGBoost ML classifier on technical features. "
                "Pass rr_ratio from get_support_resistance(), atr_pct and score from "
                "get_technical_indicators()/compute_confidence(). "
                "Returns ml_probability (0-1) indicating historical accuracy of this setup."
            ),
        },
    ]

    try:
        register_toolkits(tools, analyst, user_proxy)
        print(f"    ✅ {len(tools)} tools registered", flush=True)
    except Exception as e:
        print(f"    ❌ FATAL: register_toolkits failed → {e}", flush=True)
        return {"symbol": symbol, "direction": "UNKNOWN",
                "confidence": "Low", "error": f"register_toolkits failed: {e}"}

    # ── Step 6: run the chat ──────────────────────────────────────────────────
    prompt     = _build_agent_prompt(symbol, window)
    run_ts     = datetime.now(timezone.utc).isoformat()
    raw_output = ""
    error_msg  = {"value": None}

    print("🧠 Calling agent...", flush=True)

    finished = {"done": False}

    def run_chat():
        try:
            # Try with Cache first; fall back if Cache.disk is unavailable
            try:
                from autogen import Cache
                with Cache.disk(cache_seed=None) as cache:
                    user_proxy.initiate_chat(analyst, message=prompt, cache=cache)
            except (ImportError, AttributeError, TypeError):
                print("    ⚠ Cache unavailable — running without cache", flush=True)
                user_proxy.initiate_chat(analyst, message=prompt)
            finished["done"] = True
        except Exception as e:
            import traceback
            error_msg["value"] = str(e)
            # Print full traceback so the user can see exactly what broke
            print(f"\n    ❌ Agent chat error: {e}", flush=True)
            print("    ── Traceback ──────────────────────────────────────", flush=True)
            traceback.print_exc()
            print("    ───────────────────────────────────────────────────", flush=True)

    CHAT_TIMEOUT = window * 60 + 120    # window minutes + 2 min buffer
    thread = threading.Thread(target=run_chat, daemon=True)
    thread.start()
    thread.join(timeout=CHAT_TIMEOUT)

    if not finished["done"]:
        reason = error_msg["value"] or "timeout"
        print(f"    ❌ Agent did not finish in {CHAT_TIMEOUT}s → {reason}", flush=True)
        if error_msg["value"] is None:
            print("    ⚠ Possible causes:", flush=True)
            print("      1. API key invalid or quota exceeded", flush=True)
            print("      2. Network timeout reaching OpenAI/Azure", flush=True)
            print("      3. Model name mismatch in OAI_CONFIG_LIST.json", flush=True)
        return {
            "symbol":    symbol,
            "direction": "UNKNOWN",
            "confidence": "Low",
            "error":     reason,
        }

    print("    ✅ Agent finished", flush=True)

    msgs = user_proxy.chat_messages.get(analyst, [])
    print(f"    📨 Chat messages collected: {len(msgs)}", flush=True)

    for msg in msgs:
        content = msg.get("content") or ""
        if isinstance(content, str):
            raw_output += content + "\n\n"
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict):
                    raw_output += (block.get("text") or "") + "\n\n"

    LOG_DIR.mkdir(exist_ok=True)
    today    = datetime.today().strftime("%Y-%m-%d")
    log_file = LOG_DIR / f"{symbol.replace('.', '_')}_{today}.txt"
    log_file.write_text(raw_output, encoding="utf-8")

    return _parse(raw_output, symbol, window, run_ts, str(log_file))


# ─── PARSE AGENT OUTPUT ──────────────────────────────────────────────────────

def _parse(text: str, symbol: str, window: int, run_ts: str, log_file: str) -> dict:

    def _find(patterns, default=None):
        if isinstance(patterns, str):
            patterns = [patterns]
        for p in patterns:
            m = re.search(p, text, re.IGNORECASE)
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

    direction  = _find([
        r"Direction\s*[:\-]+\s*(Bullish|Bearish|Sideways)",
        r"direction\s*[=:]\s*['\"]?(Bullish|Bearish|Sideways)",
    ])
    confidence = _find([
        r"Confidence\s*[:\-]+\s*(High|Medium|Low)",
        r"confidence\s*[=:]\s*['\"]?(High|Medium|Low)",
    ])
    entry  = _flt(_find([r"Entry\s*[=:]\s*([\d,.]+)",   r"entry_price\s*[=:]\s*([\d,.]+)"]))
    target = _flt(_find([r"Target\s*[=:]\s*([\d,.]+)",  r"target_price\s*[=:]\s*([\d,.]+)"]))
    stop_p = _flt(_find([r"Stop\s*[=:]\s*([\d,.]+)",    r"stop_price\s*[=:]\s*([\d,.]+)"]))
    rr     = _flt(_find([r"R:R\s*[=:]\s*([\d.\-]+)",    r"rr_ratio\s*[=:]\s*([\d.\-]+)"]))

    score_m = (
        re.search(r"Score:\s*(\d)/5", text)
        or re.search(r"score\s*=\s*(\d)", text, re.IGNORECASE)
    )
    score = int(score_m.group(1)) if score_m else None

    atr_m = (
        re.search(r"[±+-]+([\d.]+)%\s*scaled", text)
        or re.search(r"ATR\(14\).*?[±+-]+([\d.]+)%", text)
    )
    atr_pct = float(atr_m.group(1)) if atr_m else None

    fb_score = _flt(_find([
        r"net_score\s*[=:]\s*([+-]?[\d.]+)",
        r"FinBERT Sentiment\s*:.*?([+-]?[\d.]+)",
    ]))
    fb_label = _find([r"FinBERT Sentiment\s*:.*?→\s*(Bullish|Bearish|Neutral)"])
    ml_prob  = _flt(_find([r"prob\s*=\s*([\d.]+)", r"ml_probability\s*[=:]\s*([\d.]+)"]))
    ml_label = _find([r"XGBoost FinML\s*:.*?→\s*(High|Medium|Low)"])
    sig_agr  = _find([r"Signal Agreement\s*:\s*(\S+)"])

    predicted_close = None
    if entry and atr_pct and direction:
        move_pct = atr_pct * math.sqrt(window / 375)
        mult     = {"bullish": 1.0, "bearish": -1.0, "sideways": 0.0}.get(
            (direction or "").lower(), 0.0
        )
        predicted_close = round(entry * (1 + mult * move_pct / 100), 2)

    return {
        "symbol":           symbol,
        "date":             datetime.today().strftime("%Y-%m-%d"),
        "window_min":       window,
        "run_timestamp":    run_ts,
        "direction":        direction,
        "confidence":       confidence,
        "entry_price":      entry,
        "target_price":     target,
        "stop_price":       stop_p,
        "rr_ratio":         rr,
        "score":            score,
        "atr_pct":          atr_pct,
        "predicted_close":  predicted_close,
        "finbert_score":    fb_score,
        "finbert_label":    fb_label,
        "ml_probability":   ml_prob,
        "ml_label":         ml_label,
        "signal_agreement": sig_agr,
        "pipeline":         "two_layer_v2_tool_based",
        "parse_ok":         direction is not None,
        "raw_output_file":  log_file,
        "actual_close":     None,
        "actual_direction": None,
        "direction_correct":None,
        "error_pct":        None,
    }


# ─── FETCH ACTUALS ────────────────────────────────────────────────────────────

def fetch_actuals(window: int = 30):
    import yfinance as yf

    pfile = _pred_file(window)
    if not pfile.exists():
        print(f"No predictions file: {pfile}")
        return

    records = []
    for line in pfile.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            records.append(json.loads(line))
        except Exception:
            pass

    pending    = [r for r in records if r.get("actual_close") is None]
    wait_secs  = window * 60
    ist_now    = datetime.utcnow() + timedelta(hours=5, minutes=30)
    interval   = "15m" if window <= 30 else "30m"

    print(f"\n{'='*60}")
    print(f"FETCH {window}-MIN ACTUALS — {ist_now.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"Pending: {len(pending)} / {len(records)}")
    print(f"{'='*60}\n")

    filled = 0
    for r in records:
        if r.get("actual_close") is not None:
            continue

        symbol = r["symbol"]
        run_ts = r.get("run_timestamp")
        print(f"  → {symbol} ", end="", flush=True)

        if run_ts:
            run_dt = datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
            if run_dt.tzinfo is None:
                run_dt = run_dt.replace(tzinfo=timezone.utc)
            elapsed = (datetime.now(timezone.utc) - run_dt).total_seconds()
            if elapsed < wait_secs:
                print(f"⏳ wait {int((wait_secs - elapsed) / 60)} more min")
                continue

        try:
            target_utc = (
                datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
                + timedelta(minutes=window)
            )
            df = yf.Ticker(symbol).history(interval=interval, period="2d")
            if df.empty:
                print("⏳ no data")
                continue
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            later = df[df.index >= target_utc]
            if later.empty:
                print("⏳ candle not yet available")
                continue

            c = later.iloc[0]
            r.update({
                "actual_open":      round(float(c["Open"]),  2),
                "actual_high":      round(float(c["High"]),  2),
                "actual_low":       round(float(c["Low"]),   2),
                "actual_close":     round(float(c["Close"]), 2),
                "actual_candle_ts": str(c.name),
            })

            entry  = r.get("entry_price")
            actual = r["actual_close"]
            if entry and actual:
                atr_pct    = r.get("atr_pct") or 0.5
                move_pct   = abs((actual - entry) / entry * 100)
                actual_dir = "Bullish" if actual > entry else "Bearish"
                r["actual_direction"]  = actual_dir
                r["direction_correct"] = (
                    move_pct <= atr_pct
                    if r.get("direction") == "Sideways"
                    else actual_dir == r.get("direction")
                )

            if r.get("predicted_close") and actual:
                r["error_pct"] = round(
                    (actual - r["predicted_close"]) / r["predicted_close"] * 100, 4
                )

            filled += 1
            icon = "✓" if r.get("direction_correct") else "✗"
            print(
                f"✅ actual={actual}  entry={entry}  "
                f"pred={r.get('direction')}  actual_dir={r.get('actual_direction')}  {icon}"
            )

        except Exception as e:
            print(f"⚠ {e}")

    if filled > 0:
        pfile.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )
        print(f"\n✅ Saved {pfile}  ({filled} actuals filled)")


# ─── EVALUATE ────────────────────────────────────────────────────────────────

def evaluate(window: int = 30):
    pfile = _pred_file(window)
    if not pfile.exists():
        print(f"No file: {pfile}")
        return

    records = [
        json.loads(l)
        for l in pfile.read_text(encoding="utf-8").splitlines()
        if l.strip()
    ]
    scored = [r for r in records if r.get("direction_correct") is not None]
    hm     = [
        r for r in scored
        if r.get("confidence") in ("High", "Medium")
        and r.get("direction") != "Sideways"
    ]

    print(f"\n{'='*60}")
    print(f"TWO-LAYER V2 (Tool-Based) — {window}-MIN EVALUATION")
    print(f"{'='*60}")
    print(f"  Total predictions     : {len(records)}")
    print(f"  Scored                : {len(scored)}")

    if scored:
        acc_all = sum(1 for r in scored if r["direction_correct"]) / len(scored)
        print(f"  Accuracy (all)        : {acc_all:.1%}")
    else:
        print("  Accuracy (all)        : N/A")

    if hm:
        acc_hm = sum(1 for r in hm if r["direction_correct"]) / len(hm)
        print(f"  Accuracy (H+M, no SW) : {acc_hm:.1%}  [{len(hm)} preds]")
    else:
        print("  Accuracy (H+M, no SW) : N/A")

    if records:
        sw_count = sum(1 for r in records if r.get("direction") == "Sideways")
        print(f"  Sideways              : {sw_count} ({sw_count / len(records):.0%})")

    print(f"\n  {'Symbol':<16} {'Dir':<10} {'Conf':<8} {'FB':<8} {'ML':<6} {'Actual':<10} ✓")
    print("  " + "-" * 65)
    for r in records:
        icon = (
            "✓" if r.get("direction_correct")
            else ("?" if r.get("direction_correct") is None else "✗")
        )
        print(
            f"  {r['symbol']:<16} {r.get('direction', '?'):<10} "
            f"{r.get('confidence', '?'):<8} "
            f"{(r.get('finbert_score') or 0):+.3f}   "
            f"{(r.get('ml_probability') or 0):.3f}  "
            f"{r.get('actual_direction', '?'):<10} {icon}"
        )


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="NSE Two-Layer V2 — Tool-Based Agent Pipeline"
    )
    parser.add_argument("--run",      action="store_true")
    parser.add_argument("--fetch",    action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--stocks",   nargs="+", default=WATCHLIST)
    parser.add_argument("--window",   type=int,  default=30)
    args = parser.parse_args()

    if args.run:
        sys.path.insert(0, str(Path("../").resolve()))

        # ── Pre-flight checks ─────────────────────────────────────────────────
        print("\n🔍 PRE-FLIGHT CHECKS")

        # 1. finrobot
        try:
            from finrobot.data_source.yfinance_utils import YFinanceUtils
            print("  ✅ finrobot.data_source.yfinance_utils imported OK")
        except ImportError as e:
            print(f"  ❌ FATAL: {e}")
            print("  → Fix: pip install -e ../ (from Finrobot_old/)")
            return

        # 2. autogen
        try:
            import autogen
            print(f"  ✅ autogen OK (v{getattr(autogen, '__version__', '?')})")
        except ImportError:
            print("  ❌ FATAL: autogen not installed → pip install pyautogen")
            return

        # 3. OAI config
        oai_ok = False
        for candidate in [Path("../OAI_CONFIG_LIST.json"), Path("OAI_CONFIG_LIST.json")]:
            if candidate.exists():
                print(f"  ✅ OAI config found: {candidate.resolve()}")
                try:
                    cfg_test = json.loads(candidate.read_text())
                    models   = [c.get("model", "?") for c in (cfg_test if isinstance(cfg_test, list) else cfg_test.get("config_list", [cfg_test]))]
                    print(f"     Models in config: {models}")
                    oai_ok = True
                except Exception as e:
                    print(f"  ⚠ OAI config parse warning: {e}")
                    oai_ok = True   # file exists; let run_agent handle it
                break
        if not oai_ok:
            print("  ❌ FATAL: OAI_CONFIG_LIST.json not found in ./ or ../")
            print("  → Create it with your OpenAI/Azure API key and model name")
            return

        # 4. numpy / xgboost
        try:
            import numpy   # noqa: F401
            print("  ✅ numpy OK")
        except ImportError:
            print("  ⚠ numpy missing → pip install numpy")
        try:
            import xgboost  # noqa: F401
            print("  ✅ xgboost OK")
        except ImportError:
            print("  ⚠ xgboost missing (rule-based fallback will be used) → pip install xgboost")

        print()

        ist_now = datetime.utcnow() + timedelta(hours=5, minutes=30)
        print(f"\n{'='*60}")
        print(f"TWO-LAYER V2 — TOOL-BASED AGENT PIPELINE")
        print(f"Time: {ist_now.strftime('%Y-%m-%d %H:%M IST')}")
        print(f"Stocks: {len(args.stocks)} | Window: {args.window}min")
        print(f"Architecture: 14 data tools + FinBERT tool + XGBoost tool")
        print(f"{'='*60}\n")

        pfile = _pred_file(args.window)
        for i, symbol in enumerate(args.stocks, 1):
            print(f"\n[{i}/{len(args.stocks)}] {symbol}")
            try:
                pred = run_agent(symbol, args.window, YFinanceUtils)
                with open(pfile, "a", encoding="utf-8") as f:
                    f.write(json.dumps(pred) + "\n")
                print(
                    f"  ✅ dir={pred.get('direction')}  conf={pred.get('confidence')}  "
                    f"finbert={pred.get('finbert_score')}  ml={pred.get('ml_probability')}"
                )
            except Exception as e:
                print(f"  ❌ {e}")
            time.sleep(2)

        print(f"\n✅ {len(args.stocks)} predictions → {pfile}")
        print(f"→ Run --fetch --window {args.window} in ~{args.window + 5} minutes")

    if args.fetch:
        fetch_actuals(args.window)

    if args.evaluate:
        evaluate(args.window)


if __name__ == "__main__":
    main()