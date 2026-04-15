"""
two_layer_runner.py — NSE Intraday Two-Layer Pipeline Runner
=============================================================

Integrates quant_layer.py (Layer 1: FinBERT + FinML) with
the existing FinRobot agent pipeline (Layer 2: GPT-4 reasoning).

HOW IT WORKS:
    Step 1: Fetch raw data using yfinance_utils_nse.py tools directly
    Step 2: Run FinBERT on news headlines (Layer 1A)
    Step 3: Run FinML on technical features (Layer 1B)
    Step 4: GPT-4 reasons ONLY on quant signal outputs (Layer 2)
    Step 5: Save predictions to ft_predictions_2layer_{window}min.jsonl
    Step 6: Fetch actuals + evaluate (same as forward_test_runner.py)

USAGE:
    # Run predictions
    python two_layer_runner.py --run --window 30
    python two_layer_runner.py --run --window 60

    # Fetch actuals (same timing as forward_test_runner)
    python two_layer_runner.py --fetch --window 30

    # Evaluate
    python two_layer_runner.py --evaluate --window 30

    # Compare two-layer vs single-layer
    python two_layer_runner.py --compare --window 30

    # Full pipeline at once
    python two_layer_runner.py --run --window 30 --stocks HDFCBANK.NS TCS.NS

SETUP:
    1. Copy quant_layer.py to same directory as this file
    2. Add HuggingFace token to config_api_keys.json:
       {"HUGGINGFACE_TOKEN": "hf_your_token_here", ...}
    3. Get free token at: https://huggingface.co/settings/tokens
       (just needs "Make calls to Inference Providers" permission)
"""

import sys
import json
import time
import argparse
import math
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone

# ─── CONFIG ──────────────────────────────────────────────────────────────────

WATCHLIST = [
    "HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS",
    "RELIANCE.NS", "ONGC.NS", "NTPC.NS", "POWERGRID.NS", "COALINDIA.NS",
    "TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS", "LTIMINDTEC.NS",
    "SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "APOLLOHOSP.NS", "LUPIN.NS",
]

AGENT_LOG_DIR = Path("ft_agent_logs_2layer")

def _predictions_file(window: int) -> Path:
    return Path(f"ft_predictions_2layer_{window}min.jsonl")

def _results_file(window: int) -> Path:
    return Path(f"ft_results_2layer_{window}min.csv")


# ─── HELPER: flatten yfinance MultiIndex columns ──────────────────────────────

def _flatten_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    yfinance >= 0.2.x returns MultiIndex columns like ('Close', 'TCS.NS').
    Flatten to single-level ('Close') so standard subscripting works.
    """
    if df is not None and isinstance(df.columns, pd.MultiIndex):
        df = df.copy()
        df.columns = df.columns.get_level_values(0)
    return df


# ─── DATA FETCHER (calls yfinance tools directly) ─────────────────────────────

def fetch_stock_data(symbol: str, YFinanceUtils) -> dict:
    """
    Fetch all raw data needed for Layer 1 pre-processing.
    Returns dict with headlines, tech_data, market_data, entry_price.
    """
    today      = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=45)).strftime("%Y-%m-%d")

    data = {
        "symbol":      symbol,
        "headlines":   [],
        "tech_data":   {},
        "market_data": {},
        "entry_price": None,
        "atr_pct":     None,
        "errors":      [],
    }

    # ── News headlines ────────────────────────────────────────────────────────
    try:
        news_raw = YFinanceUtils.get_company_news(symbol=symbol)
        print(f"    [debug] news type: {type(news_raw)}, count: {len(news_raw) if news_raw else 0}")

        headlines = []
        if news_raw and isinstance(news_raw, list):
            for item in news_raw:
                text = None
                if isinstance(item, str):
                    text = item.strip()
                elif isinstance(item, dict):
                    # yfinance >= 0.2.x: {"content": {"title": "...", ...}}
                    content = item.get("content")
                    if isinstance(content, dict):
                        text = content.get("title") or content.get("summary") or content.get("description")
                    # yfinance older: {"title": "...", "summary": "..."}
                    if not text:
                        for col in ["title", "Title", "headline", "Headline",
                                    "summary", "Summary", "description", "Description"]:
                            val = item.get(col)
                            if val and isinstance(val, str) and len(val.strip()) > 5:
                                text = val.strip()
                                break
                if text and len(text.strip()) > 5:
                    headlines.append(text.strip())

        data["headlines"] = headlines[:10]
        print(f"    [debug] extracted {len(data['headlines'])} headlines")
        if data["headlines"]:
            print(f"    [debug] first: {data['headlines'][0][:80]}")
        else:
            if news_raw and len(news_raw) > 0:
                print(f"    [debug] sample raw item: {str(news_raw[0])[:200]}")
    except Exception as e:
        data["errors"].append(f"news: {e}")
        print(f"    [debug] news fetch error: {e}")

    # ── Technical indicators ──────────────────────────────────────────────────
    try:
        tech = YFinanceUtils.get_technical_indicators(symbol=symbol)
        if tech is None:
            data["errors"].append("technicals: returned None")
        elif isinstance(tech, dict):
            data["tech_data"].update(tech)
        elif hasattr(tech, "__dict__"):
            data["tech_data"].update(vars(tech))
        elif isinstance(tech, str):
            import re
            def _extract(pattern, text, cast=float):
                m = re.search(pattern, text, re.IGNORECASE)
                if m:
                    try:
                        return cast(m.group(1).replace(",", ""))
                    except Exception:
                        return None
                return None
            data["tech_data"]["rsi"]       = _extract(r"RSI[^:]*:\s*([\d.]+)", tech)
            data["tech_data"]["atr_pct"]   = _extract(r"atr_pct[^:]*:\s*([\d.]+)", tech) or \
                                             _extract(r"ATR.*?([\d.]+)%", tech)
            data["tech_data"]["adx"]       = _extract(r"ADX[^:]*:\s*([\d.]+)", tech)
            data["tech_data"]["score"]     = _extract(r"score[^:]*:\s*([\d.]+)", tech, int)
            data["tech_data"]["rr_ratio"]  = _extract(r"rr_ratio[^:]*:\s*([\d.\-]+)", tech)
            data["tech_data"] = {k: v for k, v in data["tech_data"].items() if v is not None}
    except Exception as e:
        data["errors"].append(f"technicals: {e}")

    # ── Support & resistance (for rr_ratio) ───────────────────────────────────
    try:
        sr = YFinanceUtils.get_support_resistance(symbol=symbol)
        if sr and isinstance(sr, dict):
            for setup_key in ["bearish_setup", "bullish_setup"]:
                setup = sr.get(setup_key, {})
                if isinstance(setup, dict) and setup.get("rr_ratio") is not None:
                    rr = setup["rr_ratio"]
                    if isinstance(rr, (int, float)) and abs(rr) <= 50:
                        data["tech_data"]["rr_ratio"] = round(float(rr), 3)
                    break
        elif isinstance(sr, str):
            import re
            m = re.search(r"rr_ratio[^:]*:\s*([\d.\-]+)", sr, re.IGNORECASE)
            if m:
                try:
                    rr = float(m.group(1))
                    if abs(rr) <= 50:
                        data["tech_data"]["rr_ratio"] = rr
                except Exception:
                    pass
    except Exception as e:
        data["errors"].append(f"support_resistance: {e}")

    # ── Intraday data (entry price) ───────────────────────────────────────────
    try:
        intraday = YFinanceUtils.get_intraday_data(symbol=symbol, interval="15m", period="5d")
        if intraday is not None and hasattr(intraday, "iloc") and len(intraday) > 0:
            # FIX: flatten MultiIndex columns before subscripting
            intraday = _flatten_df(intraday)
            latest = intraday.iloc[-1]
            try:
                close_val = latest.get("Close") if hasattr(latest, "get") else latest["Close"]
                if close_val is not None:
                    data["entry_price"] = round(float(close_val), 2)
                    data["tech_data"]["latest_close"] = data["entry_price"]
            except Exception:
                # fallback: last numeric value in the row
                data["entry_price"] = round(float(latest.iloc[-1]), 2)
    except Exception as e:
        data["errors"].append(f"intraday: {e}")

    # ── Market context ────────────────────────────────────────────────────────
    try:
        mkt = YFinanceUtils.get_nse_market_context()
        if mkt is not None and isinstance(mkt, dict):
            data["market_data"] = mkt
    except Exception as e:
        data["errors"].append(f"market_context: {e}")

    # ── ATR pct ──────────────────────────────────────────────────────────────
    data["atr_pct"] = data["tech_data"].get("atr_pct") or data["tech_data"].get("atr_percent")

    if data["errors"]:
        print(f"    ⚠ Data fetch warnings for {symbol}: {'; '.join(data['errors'])}")

    return data


# ─── SINGLE STOCK PIPELINE ───────────────────────────────────────────────────

def run_two_layer_for_stock(
    symbol:     str,
    window:     int,
    YFinanceUtils,
    hf_token:   str,
    llm_config: dict,
) -> dict:
    """
    Full two-layer pipeline for one stock.
    Returns prediction dict compatible with forward_test_runner.py format.
    """
    from quant_layer import run_layer1, run_layer2_reasoning

    run_timestamp = datetime.now(timezone.utc).isoformat()

    # ── Fetch raw data ────────────────────────────────────────────────────────
    print(f"  Fetching raw data...")
    raw = fetch_stock_data(symbol, YFinanceUtils)

    # ── Fallback: get entry_price directly from yfinance if tools failed ──────
    if raw["entry_price"] is None:
        try:
            import yfinance as yf
            hist = yf.Ticker(symbol).history(interval="15m", period="1d")
            hist = _flatten_df(hist)
            if not hist.empty and "Close" in hist.columns:
                raw["entry_price"] = round(float(hist["Close"].iloc[-1]), 2)
                print(f"  [fallback] entry_price={raw['entry_price']} (direct yfinance)")
        except Exception as e:
            print(f"  [fallback] entry_price fetch failed: {e}")

    # ── Fallback: atr_pct from daily history if missing ───────────────────────
    if not raw["atr_pct"]:
        try:
            import yfinance as yf
            hist = yf.Ticker(symbol).history(period="20d")
            hist = _flatten_df(hist)
            if len(hist) >= 2 and "High" in hist.columns and "Low" in hist.columns:
                tr = (hist["High"] - hist["Low"]).abs()
                atr_val = tr.tail(14).mean()
                close   = hist["Close"].iloc[-1]
                raw["atr_pct"] = round(float(atr_val / close * 100), 4)
                print(f"  [fallback] atr_pct={raw['atr_pct']}%")
        except Exception:
            pass

    # ── Build FinML features ──────────────────────────────────────────────────
    tech    = raw["tech_data"]
    atr_pct = raw["atr_pct"] or tech.get("atr_pct") or tech.get("atr_percent") or 0.5

    rr_ratio = (tech.get("rr_ratio") or
                tech.get("risk_reward") or
                tech.get("risk_reward_ratio") or 0.0)
    if abs(rr_ratio) > 50:
        rr_ratio = 0.0

    score = (tech.get("score") or
             tech.get("confidence_score") or
             tech.get("overall_score") or 0.0)

    exp_move = (tech.get("expected_intraday_range_pct") or
                tech.get("exp_move_pct") or
                atr_pct) * math.sqrt(window / 375)

    features = {
        "rr_ratio":     round(float(rr_ratio), 3),
        "atr_pct":      round(float(atr_pct),  4),
        "score":        round(float(score),     1),
        "exp_move_pct": round(float(exp_move),  4),
    }
    print(f"  [debug] FinML features: rr={features['rr_ratio']}  atr={features['atr_pct']}%  "
          f"score={features['score']}  exp_move={features['exp_move_pct']}%")

    # ── Layer 1: FinBERT + FinML ──────────────────────────────────────────────
    layer1 = run_layer1(
        symbol    = symbol,
        headlines = raw["headlines"],
        features  = features,
        window    = window,
        hf_token  = hf_token,
    )

    # ── Layer 2: GPT-4 reasoning ──────────────────────────────────────────────
    print(f"  [Layer 2] GPT-4 synthesizing quant signals...")
    layer2 = run_layer2_reasoning(
        symbol      = symbol,
        window      = window,
        layer1      = layer1,
        raw_tech    = tech,
        raw_market  = raw["market_data"],
        llm_config  = llm_config,
    )

    # ── Save agent log ─────────────────────────────────────────────────────────
    AGENT_LOG_DIR.mkdir(exist_ok=True)
    safe_sym  = symbol.replace(".", "_")
    date_str  = datetime.now().strftime("%Y-%m-%d")
    log_file  = AGENT_LOG_DIR / f"{safe_sym}_{date_str}.txt"
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"TWO-LAYER ANALYSIS — {symbol} — {run_timestamp}\n")
        f.write("="*60 + "\n\n")
        f.write("LAYER 1A — FINBERT SENTIMENT\n")
        f.write(json.dumps(layer1["finbert"], indent=2) + "\n\n")
        f.write("LAYER 1B — FINML CLASSIFIER\n")
        f.write(json.dumps(layer1["finml"], indent=2, default=str) + "\n\n")
        f.write("LAYER 1 COMBINED\n")
        f.write(json.dumps(layer1["combined"], indent=2) + "\n\n")
        f.write("LAYER 2 — GPT-4 REASONING\n")
        f.write(layer2.get("raw_output", "N/A") + "\n")

    # ── Build prediction record ───────────────────────────────────────────────
    direction  = layer2.get("direction")
    confidence = layer2.get("confidence")
    entry      = raw["entry_price"]

    predicted_close = None
    if entry and atr_pct and direction:
        move_pct = atr_pct * math.sqrt(window / 375)
        mult     = {"bullish": 1.0, "bearish": -1.0, "sideways": 0.0}.get(
                    (direction or "").lower(), 0.0)
        predicted_close = round(entry * (1 + mult * move_pct / 100), 2)

    tech_score = features.get("score") or 0.0
    if tech_score == 0.0:
        score_map = {"High": 4, "Medium": 3, "Low": 2, None: 0}
        tech_score = score_map.get(confidence, 0)

    record = {
        # Core prediction fields (compatible with forward_test_runner.py)
        "symbol":            symbol,
        "date":              datetime.now().strftime("%Y-%m-%d"),
        "window_min":        window,
        "pipeline":          "two_layer",
        "run_timestamp":     run_timestamp,
        "direction":         direction,
        "exp_move_pct":      round(atr_pct * math.sqrt(window / 375), 4) if atr_pct else None,
        "entry_price":       entry,
        "predicted_close":   predicted_close,
        "confidence":        confidence,
        "score":             tech_score,
        "rr_ratio":          features.get("rr_ratio", 0.0),
        "atr_pct":           atr_pct,
        "latest_close":      entry,
        "tradeable":         confidence in ("High", "Medium") and direction != "Sideways",

        # Two-layer specific fields
        "finbert_score":     layer1["finbert"]["net_score"],
        "finbert_sentiment": layer1["finbert"]["sentiment_label"],
        "finbert_headlines": layer1["finbert"]["headlines_scored"],
        "finml_prob":        layer1["finml"]["ml_confidence"],
        "finml_label":       layer1["finml"]["ml_label"],
        "finml_model_type":  layer1["finml"]["model_type"],
        "signal_agreement":  layer1["combined"]["agreement"],
        "overall_strength":  layer1["combined"]["overall_strength"],
        "signal_conflict":   layer2.get("signal_conflict"),
        "key_reason":        layer2.get("key_reason"),
        "risk_note":         layer2.get("risk_note"),
        "finbert_interp":    layer2.get("finbert_interp"),
        "finml_interp":      layer2.get("finml_interp"),

        # Actuals — filled by --fetch
        "actual_close":      None,
        "actual_high":       None,
        "actual_low":        None,
        "actual_open":       None,
        "actual_candle_ts":  None,
        "error_pct":         None,
        "direction_correct": None,
        "actual_direction":  None,

        "parse_ok":          layer2.get("parse_ok", False),
        "raw_output_file":   str(log_file),
        "layer2_error":      layer2.get("error"),
    }

    return record


# ─── FETCH ACTUALS ───────────────────────────────────────────────────────────

def fetch_actuals(window: int = 30):
    """Same logic as forward_test_runner.py but for two-layer predictions file."""
    import yfinance as yf

    pfile = _predictions_file(window)
    if not pfile.exists():
        print(f"No predictions file found: {pfile}")
        return

    records = []
    for l in pfile.read_text(encoding="utf-8").splitlines():
        l = l.strip()
        if not l:
            continue
        try:
            records.append(json.loads(l))
        except json.JSONDecodeError:
            print(f"  ⚠ skipping corrupt line: {l[:60]}...")

    pending   = [r for r in records if r.get("actual_close") is None]
    wait_secs = window * 60
    ist_now   = datetime.utcnow() + timedelta(hours=5, minutes=30)

    print(f"\n{'='*60}")
    print(f"FETCH {window}-MIN ACTUALS (Two-Layer) — {ist_now.strftime('%Y-%m-%d %H:%M IST')}")
    print(f"Pending: {len(pending)} / {len(records)}")
    print(f"{'='*60}\n")

    # Recompute direction_correct for already-filled records
    for r in records:
        if r.get("actual_close") is None:
            continue
        entry = r.get("entry_price")
        act   = r.get("actual_close")
        if entry and act:
            move_pct   = abs((act - entry) / entry * 100)
            atr_pct    = r.get("atr_pct") or 0.5
            actual_dir = "Bullish" if act > entry else ("Bearish" if act < entry else "Sideways")
            r["actual_direction"] = actual_dir
            if r.get("direction") == "Sideways":
                r["direction_correct"] = move_pct <= atr_pct
            else:
                r["direction_correct"] = actual_dir == r.get("direction")

    interval = "15m" if window <= 30 else "30m"
    filled   = 0

    for r in records:
        if r.get("actual_close") is not None:
            continue

        symbol = r["symbol"]
        run_ts = r.get("run_timestamp")
        print(f"  → {symbol} ", end="", flush=True)

        # ── FIX: guard against missing run_timestamp ──────────────────────────
        if not run_ts:
            print("⚠ no run_timestamp, skipping")
            continue

        try:
            run_dt = datetime.fromisoformat(run_ts.replace("Z", "+00:00"))
            if run_dt.tzinfo is None:
                run_dt = run_dt.replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError) as e:
            print(f"⚠ bad run_timestamp ({run_ts!r}): {e}")
            continue

        elapsed = (datetime.now(timezone.utc) - run_dt).total_seconds()
        if elapsed < wait_secs:
            mins_left = int((wait_secs - elapsed) / 60)
            print(f"⏳ only {int(elapsed/60)}min elapsed — wait {mins_left} more min")
            continue

        target_utc = run_dt + timedelta(minutes=window)

        # ── Fetch candle ──────────────────────────────────────────────────────
        try:
            df = yf.Ticker(symbol).history(interval=interval, period="2d")

            if df is None or df.empty:
                print("⏳ no data")
                continue

            # FIX: flatten MultiIndex columns (yfinance >= 0.2.x)
            df = _flatten_df(df)

            # FIX: verify required OHLC columns are present
            required = {"Open", "High", "Low", "Close"}
            if not required.issubset(set(df.columns)):
                print(f"⚠ unexpected columns: {df.columns.tolist()}")
                continue

            # Normalise timezone
            if df.index.tz is None:
                df.index = df.index.tz_localize("UTC")
            else:
                df.index = df.index.tz_convert("UTC")

            later = df[df.index >= target_utc]
            if later.empty:
                print("⏳ candle not yet available")
                continue

            c = later.iloc[0]

            # FIX: guard each OHLC value before float() conversion
            def _safe_float(val, field):
                if val is None:
                    raise ValueError(f"Column '{field}' is None after flatten")
                return round(float(val), 2)

            r.update({
                "actual_open":      _safe_float(c["Open"],  "Open"),
                "actual_high":      _safe_float(c["High"],  "High"),
                "actual_low":       _safe_float(c["Low"],   "Low"),
                "actual_close":     _safe_float(c["Close"], "Close"),
                "actual_candle_ts": str(c.name),
            })

            act   = r["actual_close"]
            pred  = r.get("predicted_close")
            entry = r.get("entry_price") or r.get("latest_close")

            if pred is not None and act is not None:
                r["error_pct"] = round((act - pred) / pred * 100, 4)

            if entry and act:
                move_pct   = abs((act - entry) / entry * 100)
                atr_pct    = r.get("atr_pct") or 0.5
                actual_dir = "Bullish" if act > entry else ("Bearish" if act < entry else "Sideways")
                r["actual_direction"] = actual_dir
                if r.get("direction") == "Sideways":
                    r["direction_correct"] = move_pct <= atr_pct
                else:
                    r["direction_correct"] = actual_dir == r.get("direction")
            else:
                r["direction_correct"] = None
                r["actual_direction"]  = None

            filled += 1
            icon      = "✓" if r.get("direction_correct") else ("?" if r.get("direction_correct") is None else "✗")
            err_str   = f"err={r['error_pct']:+.3f}%" if r.get("error_pct") is not None else "no_pred"
            adir      = r.get("actual_direction", "?")
            entry_str = f"entry={entry:.2f}" if entry else "entry=None"
            print(f"✅  actual={act}  {entry_str}  predicted_dir={r.get('direction')}  actual_dir={adir}  {icon}")

        except Exception as e:
            print(f"⚠ {e}")

    pfile.write_text("\n".join(json.dumps(r) for r in records) + "\n", encoding="utf-8")
    print(f"\n✅ Saved {pfile} ({filled} new actuals filled)")


# ─── COMPARE TWO-LAYER VS SINGLE-LAYER ───────────────────────────────────────

def compare_pipelines(window: int = 30):
    """
    Compare two-layer vs single-layer (forward_test_runner) accuracy.
    Prints a side-by-side comparison table.
    """
    tl_file = _predictions_file(window)
    sl_file = Path(f"ft_predictions_{window}min.jsonl")

    def load_complete(fpath):
        if not fpath.exists():
            return []
        records = []
        for l in fpath.read_text(encoding="utf-8").splitlines():
            l = l.strip()
            if l:
                try:
                    r = json.loads(l)
                    if r.get("actual_close") is not None and r.get("direction_correct") is not None:
                        records.append(r)
                except Exception:
                    pass
        return records

    tl_records = load_complete(tl_file)
    sl_records = load_complete(sl_file)

    def stats(records, label):
        if not records:
            print(f"\n  {label}: no complete records found")
            return
        total    = len(records)
        scored   = [r for r in records if r.get("direction") not in (None, "Sideways")
                    and r.get("confidence") in ("High", "Medium")]
        dir_acc  = sum(1 for r in scored if r["direction_correct"]) / len(scored) if scored else 0
        all_acc  = sum(1 for r in records if r["direction_correct"]) / total
        errors   = [abs(r["error_pct"]) for r in records if r.get("error_pct") is not None]
        mape     = sum(errors) / len(errors) if errors else 0
        sideways = sum(1 for r in records if r.get("direction") == "Sideways")
        low_conf = sum(1 for r in records if r.get("confidence") == "Low")

        print(f"\n  {'─'*50}")
        print(f"  {label}")
        print(f"  {'─'*50}")
        print(f"  Total predictions       : {total}")
        print(f"  Filtered (H+M, non-SW)  : {len(scored)}")
        print(f"  Dir accuracy (filtered) : {dir_acc:.1%}")
        print(f"  Dir accuracy (all)      : {all_acc:.1%}")
        print(f"  MAPE                    : {mape:.4f}%")
        print(f"  Sideways predictions    : {sideways} ({sideways/total:.0%})")
        print(f"  Low confidence          : {low_conf} ({low_conf/total:.0%})")

    print(f"\n{'='*60}")
    print(f"PIPELINE COMPARISON — {window}-MIN WINDOW")
    print(f"{'='*60}")
    stats(sl_records, "SINGLE-LAYER (FinRobot agent, 14 tools)")
    stats(tl_records, "TWO-LAYER (FinBERT + FinML + GPT-4 reasoning)")

    if tl_records and sl_records:
        print(f"\n  {'─'*50}")
        print(f"  PER-STOCK COMPARISON (filtered accuracy)")
        print(f"  {'─'*50}")
        print(f"  {'Symbol':<16} {'Single-Layer':>14} {'Two-Layer':>12} {'Winner':>8}")
        print(f"  {'─'*54}")

        symbols = sorted(set(r["symbol"] for r in tl_records + sl_records))
        for sym in symbols:
            sl_sym = [r for r in sl_records if r["symbol"] == sym and
                      r.get("direction") not in (None, "Sideways") and
                      r.get("confidence") in ("High", "Medium")]
            tl_sym = [r for r in tl_records if r["symbol"] == sym and
                      r.get("direction") not in (None, "Sideways") and
                      r.get("confidence") in ("High", "Medium")]

            if not sl_sym and not tl_sym:
                continue

            sl_acc = f"{sum(1 for r in sl_sym if r['direction_correct'])/len(sl_sym):.0%} (n={len(sl_sym)})" if sl_sym else "N/A"
            tl_acc = f"{sum(1 for r in tl_sym if r['direction_correct'])/len(tl_sym):.0%} (n={len(tl_sym)})" if tl_sym else "N/A"

            sl_val = sum(1 for r in sl_sym if r['direction_correct'])/len(sl_sym) if sl_sym else 0
            tl_val = sum(1 for r in tl_sym if r['direction_correct'])/len(tl_sym) if tl_sym else 0
            winner = "TwoLayer" if tl_val > sl_val else ("SingleLayer" if sl_val > tl_val else "Tie")

            print(f"  {sym:<16} {sl_acc:>14} {tl_acc:>12} {winner:>8}")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NSE Two-Layer Intraday Pipeline")
    parser.add_argument("--run",      action="store_true", help="Run two-layer predictions")
    parser.add_argument("--fetch",    action="store_true", help="Fetch actual prices")
    parser.add_argument("--evaluate", action="store_true", help="Print evaluation stats")
    parser.add_argument("--compare",  action="store_true", help="Compare vs single-layer")
    parser.add_argument("--stocks",   nargs="+", default=WATCHLIST)
    parser.add_argument("--window",   type=int, default=30, choices=[30, 60])
    args = parser.parse_args()

    if not any([args.run, args.fetch, args.evaluate, args.compare]):
        parser.print_help()
        return

    window = args.window

    if args.run:
        # ── Market hours guard ─────────────────────────────────────────────
        ist_now      = datetime.utcnow() + timedelta(hours=5, minutes=30)
        latest_start = ist_now.replace(hour=15, minute=30, second=0) - timedelta(minutes=window)
        market_open  = ist_now.replace(hour=9,  minute=15, second=0)
        if not (market_open <= ist_now <= latest_start):
            print(f"\n⚠  Outside optimal window ({ist_now.strftime('%H:%M IST')}). Best: 09:15–{latest_start.strftime('%H:%M')} IST")
            ans = input("   Continue anyway? [y/N]: ").strip().lower()
            if ans != "y":
                sys.exit(0)

        # ── Imports ────────────────────────────────────────────────────────
        sys.path.insert(0, str(Path("../").resolve()))
        try:
            from quant_layer import _load_hf_token, _load_openai_config
            from finrobot.data_source.yfinance_utils import YFinanceUtils
        except ImportError as e:
            print(f"❌ Import error: {e}")
            print("   Ensure quant_layer.py and finrobot are available.")
            sys.exit(1)

        hf_token   = _load_hf_token()
        llm_config = _load_openai_config()

        if not hf_token:
            print("⚠  No HuggingFace token found. FinBERT will be skipped.")
            print("   Add 'HUGGINGFACE_TOKEN': 'hf_...' to config_api_keys.json")
            print("   Get free token at: https://huggingface.co/settings/tokens")

        pfile   = _predictions_file(window)
        records = []

        print(f"\n{'='*60}")
        print(f"TWO-LAYER NSE PIPELINE — {window}-MIN PREDICTIONS")
        print(f"Time   : {ist_now.strftime('%Y-%m-%d %H:%M IST')}")
        print(f"File   : {pfile}")
        print(f"Stocks : {len(args.stocks)}")
        print(f"{'='*60}")

        for i, symbol in enumerate(args.stocks, 1):
            print(f"\n[{i}/{len(args.stocks)}] {symbol}")
            try:
                record = run_two_layer_for_stock(
                    symbol, window, YFinanceUtils, hf_token, llm_config
                )
                records.append(record)
                status = "✅" if record.get("parse_ok") else "⚠"
                print(f"  {status} dir={record.get('direction')}  "
                      f"conf={record.get('confidence')}  "
                      f"finbert={record.get('finbert_score', 0):+.3f}  "
                      f"finml={record.get('finml_prob', 0):.3f}")
            except Exception as e:
                print(f"  ❌ Error: {e}")
                records.append({
                    "symbol": symbol, "window_min": window, "pipeline": "two_layer",
                    "run_timestamp": datetime.now(timezone.utc).isoformat(),
                    "direction": None, "confidence": None, "parse_ok": False,
                    "layer2_error": str(e),
                })

            if i < len(args.stocks):
                time.sleep(5)

        # Save
        with open(pfile, "a", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        print(f"\n{'='*60}")
        print(f"✅ {len(records)} predictions saved to {pfile}")
        print(f"\nSUMMARY:")
        print(f"{'Symbol':<16} {'Direction':<10} {'Conf':<8} {'FinBERT':>8} {'FinML':>7} {'Agreement'}")
        print("─"*65)
        for r in records:
            fb  = f"{r.get('finbert_score', 0):+.3f}" if r.get('finbert_score') is not None else "N/A"
            fm  = f"{r.get('finml_prob', 0):.3f}"     if r.get('finml_prob')    is not None else "N/A"
            agr = (r.get("signal_agreement") or "N/A")[:15]
            print(f"{r['symbol']:<16} {str(r.get('direction','?')):<10} "
                  f"{str(r.get('confidence','?')):<8} {fb:>8} {fm:>7} {agr}")
        print(f"\n→ Run  --fetch --window {window}  in ~{window+5} minutes")

    if args.fetch:
        fetch_actuals(window)

    if args.evaluate or args.compare:
        compare_pipelines(window)


if __name__ == "__main__":
    main()