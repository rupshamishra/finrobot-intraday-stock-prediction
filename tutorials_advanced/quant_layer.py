"""
quant_layer.py — Two-Layer Quantitative Pre-Analysis for NSE Intraday
======================================================================

LAYER 1A — FinBERT Sentiment (HuggingFace free API)
    Uses ProsusAI/finbert to score each news headline as
    positive / neutral / negative with a confidence score.
    Aggregates into a single net_sentiment score (-1 to +1).

LAYER 1B — FinML Technical Classifier (scikit-learn)
    RandomForest trained exclusively on bt_results_single_{window}min.jsonl (backtest data).
    This ensures clean training data with real rr_ratio + score values and diverse market regimes.
    Predicts probability of direction_correct given technical features.
    Falls back to rule-based scoring if < 30 training samples exist.

LAYER 2 — LLM Reasoning (OpenAI GPT-4 via FinRobot)
    Receives pre-computed quant scores from Layer 1.
    Reasons ONLY on the signal outputs — not raw data.
    Produces final directional call with explanation.

USAGE:
    from quant_layer import run_two_layer_analysis
    result = run_two_layer_analysis(symbol="HDFCBANK.NS", window=30)

SETUP:
    pip install transformers huggingface_hub scikit-learn pandas numpy
    Set HF_TOKEN in config_api_keys.json (free HuggingFace account token)
    Set OPENAI key in OAI_CONFIG_LIST (same as forward_test_runner.py)
"""

import os
import json
import time
import math
import requests
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta, timezone


# ─── CONFIG ──────────────────────────────────────────────────────────────────

HF_API_URL   = "https://router.huggingface.co/hf-inference/models/ProsusAI/finbert"
CONFIG_KEYS  = Path("../config_api_keys.json")
OAI_CONFIG   = Path("../OAI_CONFIG_LIST.json")
MIN_TRAIN_SAMPLES = 10          # minimum completed predictions before FinML trains
SENTIMENT_MAX_HEADLINES = 10    # max headlines to send to FinBERT per stock

# All possible prediction files to draw training data from (jsonl format)
def _get_training_files(window: int) -> list:
    """Return training files for FinML, ordered by data quality.

    Priority:
    1. ft_predictions_2layer_v2  — new tool-based pipeline (best: real rr_ratio,
                                   score, FinBERT, XGBoost all from agent CoT)
    2. bt_results_single         — backtest data (real rr_ratio + score from agent,
                                   diverse market regimes, clean actuals)
    3. ft_predictions_30min      — single-layer forward test (real agent values,
                                   but some missed fetches / duplicates)
    4. ft_predictions_2layer     — old two-layer (rr_ratio often 0, lower quality)

    We exclude files with known quality issues:
    - ft_predictions files with duplicate run entries are still included
      because _train_finml_model filters on direction_correct not None
    """
    candidates = [
        Path(f"ft_predictions_2layer_v2_{window}min.jsonl"),  # best quality — tool-based
        Path(f"bt_results_single_{window}min.jsonl"),          # backtest — diverse regimes
        Path(f"ft_predictions_{window}min.jsonl"),             # single-layer forward test
        Path(f"ft_predictions_2layer_{window}min.jsonl"),      # old two-layer (lower quality)
    ]
    existing = [p for p in candidates if p.exists()]
    print(f"    [FinML] Training sources ({len(existing)} files): "
          f"{[f.name for f in existing]}")
    return existing


# ─── HELPERS ─────────────────────────────────────────────────────────────────

def _load_hf_token() -> str:
    """Load HuggingFace token from config_api_keys.json."""
    if CONFIG_KEYS.exists():
        try:
            with open(CONFIG_KEYS) as f:
                keys = json.load(f)
            token = keys.get("HUGGINGFACE_TOKEN") or keys.get("HF_TOKEN") or ""
            if token:
                return token
        except Exception:
            pass
    # Fallback: environment variable
    return os.environ.get("HF_TOKEN", "")


def _load_openai_config() -> dict:
    """Load OpenAI LLM config from OAI_CONFIG_LIST."""
    try:
        with open(OAI_CONFIG, encoding="utf-8") as f:
            config_list = json.load(f)
        # Handle both array format and {"config_list": [...]} format
        if isinstance(config_list, dict):
            config_list = config_list.get("config_list", config_list)
        return {"config_list": config_list, "timeout": 120, "temperature": 0.1}
    except Exception as e:
        raise RuntimeError(f"Cannot load OAI_CONFIG_LIST: {e}")


# ─── LAYER 1A: FINBERT SENTIMENT ─────────────────────────────────────────────

def run_finbert_sentiment(headlines: list[str], hf_token: str) -> dict:
    """
    Send headlines to ProsusAI/finbert via HuggingFace free Inference API.
    Returns aggregated sentiment dict.

    FinBERT outputs 3 labels per headline:
        positive / neutral / negative  with softmax probabilities

    Aggregation:
        net_score = mean(positive_prob) - mean(negative_prob)
        Range: -1.0 (fully negative) to +1.0 (fully positive)
        0.0 = neutral
    """
    if not headlines:
        return _neutral_sentiment("no headlines provided")

    if not hf_token:
        return _neutral_sentiment("no HuggingFace token — add HF_TOKEN to config_api_keys.json")

    # Truncate to max headlines, clean text
    headlines = [h.strip()[:256] for h in headlines if h.strip()][:SENTIMENT_MAX_HEADLINES]

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type":  "application/json",
    }
    payload = {"inputs": headlines}

    try:
        resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

        # Handle model loading (HF cold start)
        if resp.status_code == 503:
            print("    ⏳ FinBERT model loading on HF servers, waiting 20s...")
            time.sleep(20)
            resp = requests.post(HF_API_URL, headers=headers, json=payload, timeout=30)

        if resp.status_code == 402:
            return _neutral_sentiment("HF free tier monthly credits exhausted")

        if resp.status_code != 200:
            return _neutral_sentiment(f"HF API error {resp.status_code}: {resp.text[:100]}")

        results = resp.json()

        # results is list of lists: [[{label, score}, ...], ...]
        pos_scores, neg_scores, neu_scores = [], [], []
        per_headline = []

        for i, headline_result in enumerate(results):
            # headline_result can be a list of dicts or a single dict
            if isinstance(headline_result, dict):
                headline_result = [headline_result]
            scores = {item["label"].lower(): item["score"] for item in headline_result}
            pos = scores.get("positive", 0.0)
            neg = scores.get("negative", 0.0)
            neu = scores.get("neutral",  0.0)
            pos_scores.append(pos)
            neg_scores.append(neg)
            neu_scores.append(neu)
            label = max(scores, key=scores.get)
            per_headline.append({
                "headline": headlines[i][:80],
                "label":    label,
                "positive": round(pos, 3),
                "negative": round(neg, 3),
                "neutral":  round(neu, 3),
            })

        if not per_headline:
            return _neutral_sentiment("FinBERT returned empty results for all headlines")

        net_score      = round(float(np.mean(pos_scores)) - float(np.mean(neg_scores)), 4)
        avg_positive   = round(float(np.mean(pos_scores)), 4)
        avg_negative   = round(float(np.mean(neg_scores)), 4)
        avg_neutral    = round(float(np.mean(neu_scores)), 4)
        bullish_count  = sum(1 for h in per_headline if h["label"] == "positive")
        bearish_count  = sum(1 for h in per_headline if h["label"] == "negative")
        neutral_count  = sum(1 for h in per_headline if h["label"] == "neutral")

        # Derive sentiment label
        if net_score >= 0.15:
            sentiment_label = "Bullish"
        elif net_score <= -0.15:
            sentiment_label = "Bearish"
        else:
            sentiment_label = "Neutral"

        return {
            "source":          "FinBERT (ProsusAI/finbert)",
            "model":           "ProsusAI/finbert",
            "headlines_scored": len(per_headline),
            "net_score":       net_score,        # -1 to +1
            "avg_positive":    avg_positive,
            "avg_negative":    avg_negative,
            "avg_neutral":     avg_neutral,
            "bullish_count":   bullish_count,
            "bearish_count":   bearish_count,
            "neutral_count":   neutral_count,
            "sentiment_label": sentiment_label,
            "per_headline":    per_headline,
            "error":           None,
        }

    except requests.exceptions.Timeout:
        return _neutral_sentiment("HF API timeout — FinBERT unavailable")
    except Exception as e:
        return _neutral_sentiment(f"FinBERT error: {str(e)[:100]}")


def _neutral_sentiment(reason: str) -> dict:
    return {
        "source":           "FinBERT (ProsusAI/finbert)",
        "model":            "ProsusAI/finbert",
        "headlines_scored": 0,
        "net_score":        0.0,
        "avg_positive":     0.0,
        "avg_negative":     0.0,
        "avg_neutral":      1.0,
        "bullish_count":    0,
        "bearish_count":    0,
        "neutral_count":    0,
        "sentiment_label":  "Neutral",
        "per_headline":     [],
        "error":            reason,
    }


# ─── LAYER 1B: FINML TECHNICAL CLASSIFIER ────────────────────────────────────

# Features used for FinML model
# These must match what's available in the prediction jsonl records
FINML_FEATURES = [
    "rr_ratio",       # Risk-reward ratio (from agent)
    "atr_pct",        # ATR as % of price (volatility)
    "score",          # LLM confidence score 1-5
    "exp_move_pct",   # Expected move scaled to window
]

# Extra features extracted from record fields during training
FINML_EXTRA_FEATURES = [
    "hour_ist",       # Hour of prediction (market timing)
    "is_morning",     # 1 if before 11:30 IST (morning session)
]

_finml_model_cache = {}


def run_finml_classifier(
    symbol: str,
    features: dict,
    window: int = 30,
) -> dict:
    """
    Train (or load cached) RandomForest on historical ft_results data.
    Predict probability that this prediction will be direction_correct.

    features: dict with keys matching FINML_FEATURES
              e.g. {"rr_ratio": 2.1, "atr_pct": 0.85, "exp_move_pct": 0.24, "score": 4}

    Returns:
        ml_confidence:    float 0.0–1.0 (probability of being correct)
        ml_label:         "High" / "Medium" / "Low"
        model_type:       "RandomForest" or "RuleBased"
        training_samples: int
        feature_importance: dict
    """
    model_key = f"{window}min_v3"   # v3 = includes two_layer_v2 tool-based predictions

    # ── Try to train/load RandomForest ──────────────────────────────────────
    if model_key not in _finml_model_cache:
        model_result = _train_finml_model(None, window)
        _finml_model_cache[model_key] = model_result

    cached = _finml_model_cache[model_key]

    if cached["model"] is None:
        # Fallback to rule-based scoring
        return _rule_based_ml_score(features, cached["n_samples"])

    # ── Run prediction ───────────────────────────────────────────────────────
    try:
        model     = cached["model"]
        feat_cols = cached["feature_cols"]

        # Build input using same feature extraction as training
        fv      = {**features}   # caller-supplied features
        # Add time features if not supplied
        if "hour_ist" not in fv:
            from datetime import datetime, timedelta, timezone
            ist = datetime.utcnow() + timedelta(hours=5, minutes=30)
            fv["hour_ist"]   = ist.hour + ist.minute / 60
            fv["is_morning"] = 1.0 if fv["hour_ist"] < 11.5 else 0.0
        # Sanitise rr_ratio
        if abs(fv.get("rr_ratio", 0) or 0) > 50:
            fv["rr_ratio"] = 0.0

        X_input = np.array([[fv.get(f, 0.0) or 0.0 for f in feat_cols]])
        prob    = model.predict_proba(X_input)[0][1]   # probability of direction_correct=True

        if prob >= 0.65:
            label = "High"
        elif prob >= 0.45:
            label = "Medium"
        else:
            label = "Low"

        # Feature importance
        importance = {
            f: round(float(imp), 4)
            for f, imp in zip(feat_cols, model.feature_importances_)
        }
        top_feature = max(importance, key=importance.get)

        return {
            "ml_confidence":     round(float(prob), 4),
            "ml_label":          label,
            "model_type":        "RandomForest (scikit-learn)",
            "training_samples":  cached["n_samples"],
            "feature_cols":      feat_cols,
            "feature_importance": importance,
            "top_feature":       top_feature,
            "train_accuracy":    cached.get("train_accuracy"),
            "error":             None,
        }

    except Exception as e:
        return _rule_based_ml_score(features, cached.get("n_samples", 0))


def _extract_features(r: dict) -> dict | None:
    """Extract feature vector from a prediction record. Returns None if unusable."""
    # rr_ratio — direct field
    rr = r.get("rr_ratio")
    if rr is None or abs(rr) > 50:   # filter absurd R:R values (>50 are data errors)
        rr = 0.0

    # atr_pct
    atr = r.get("atr_pct") or 0.5

    # score — LLM confidence 1-5
    score = r.get("score") or 0
    if score == 0:
        # derive from confidence label
        conf_map = {"High": 4, "Medium": 3, "Low": 2}
        score = conf_map.get(r.get("confidence"), 0)

    # exp_move_pct
    exp_move = r.get("exp_move_pct") or (atr * 0.28)   # fallback: atr * sqrt(30/375)

    # Time features from run_timestamp
    hour_ist = 11   # default mid-morning
    is_morning = 1
    ts = r.get("run_timestamp") or r.get("logged_at")
    if ts:
        try:
            from datetime import datetime, timedelta, timezone
            dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            ist = dt + timedelta(hours=5, minutes=30)
            hour_ist   = ist.hour + ist.minute / 60
            is_morning = 1 if hour_ist < 11.5 else 0
        except Exception:
            pass

    return {
        "rr_ratio":     float(rr),
        "atr_pct":      float(atr),
        "score":        float(score),
        "exp_move_pct": float(exp_move),
        "hour_ist":     float(hour_ist),
        "is_morning":   float(is_morning),
    }


def _train_finml_model(results_file, window: int) -> dict:
    """Load completed predictions from jsonl files, train RandomForest."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score

    training_files = _get_training_files(window)
    if not training_files:
        return {"model": None, "n_samples": 0, "reason": "no prediction files found yet"}

    records = []
    for fpath in training_files:
        try:
            for line in fpath.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if (r.get("actual_close") is not None and
                        r.get("direction_correct") is not None and
                        (r.get("direction") or "").capitalize() not in ("", "Sideways", "None") and
                        r.get("confidence") in ("High", "Medium")):
                        records.append(r)
                except json.JSONDecodeError:
                    pass
        except Exception as e:
            print(f"    ⚠ Could not read {fpath}: {e}")

    if len(records) < MIN_TRAIN_SAMPLES:
        return {
            "model":     None,
            "n_samples": len(records),
            "reason":    f"only {len(records)} usable training samples (need {MIN_TRAIN_SAMPLES})"
        }

    try:
        # Extract features per record
        feat_names = ["rr_ratio", "atr_pct", "score", "exp_move_pct", "hour_ist", "is_morning"]
        rows, labels = [], []
        for r in records:
            fv = _extract_features(r)
            if fv is None:
                continue
            rows.append([fv[f] for f in feat_names])
            labels.append(1 if r["direction_correct"] else 0)

        if len(rows) < MIN_TRAIN_SAMPLES:
            return {"model": None, "n_samples": len(rows), "reason": "insufficient rows after feature extraction"}

        X = np.array(rows)
        y = np.array(labels)

        # Check class balance — warn if heavily imbalanced
        pos_rate = y.mean()
        if pos_rate > 0.85 or pos_rate < 0.15:
            print(f"    ⚠ FinML training data imbalanced: {pos_rate:.0%} correct — model may overfit")

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=3,           # shallow = less overfit on small data
            min_samples_leaf=5,    # require 5+ samples per leaf
            max_features="sqrt",
            random_state=42,
            class_weight="balanced",  # handle imbalance
        )
        model.fit(X, y)

        # Cross-val accuracy
        try:
            n_splits = min(5, len(rows) // 10)
            if n_splits >= 3:
                cv_scores = cross_val_score(model, X, y, cv=n_splits, scoring="accuracy")
                train_accuracy = round(float(cv_scores.mean()), 4)
            else:
                train_accuracy = round(float((y == model.predict(X)).mean()), 4)
        except Exception:
            train_accuracy = None

        print(f"    ✅ FinML trained EXCLUSIVELY on backtest data: {len(rows)} samples "
              f"from {[f.name for f in training_files]}, "
              f"cv_acc={train_accuracy}, pos_rate={pos_rate:.0%}")

        return {
            "model":          model,
            "feature_cols":   feat_names,
            "n_samples":      len(rows),
            "train_accuracy": train_accuracy,
            "pos_rate":       round(float(pos_rate), 3),
            "reason":         None,
        }

    except Exception as e:
        return {"model": None, "n_samples": len(records), "reason": str(e)}


def _rule_based_ml_score(features: dict, n_samples: int) -> dict:
    """
    Fallback when not enough data to train RandomForest.
    Uses simple rules based on known indicators of good predictions.
    """
    score = 0.5   # start neutral

    rr    = features.get("rr_ratio") or 0
    atr   = features.get("atr_pct")  or 0
    conf  = features.get("score")    or 0
    move  = features.get("exp_move_pct") or 0

    # Good R:R is a strong predictor
    if rr >= 2.0:   score += 0.12
    elif rr >= 1.0: score += 0.06
    elif rr <= 0.3: score -= 0.12

    # High LLM confidence score
    if conf >= 4:   score += 0.10
    elif conf <= 2: score -= 0.10

    # Very large expected moves are harder to get right
    if move > 1.5:  score -= 0.08
    elif move < 0.1: score -= 0.05

    # High ATR = volatile = harder
    if atr > 1.5:   score -= 0.06

    prob  = max(0.1, min(0.9, score))
    label = "High" if prob >= 0.65 else ("Medium" if prob >= 0.45 else "Low")

    return {
        "ml_confidence":     round(prob, 4),
        "ml_label":          label,
        "model_type":        f"RuleBased (only {n_samples} samples, need {MIN_TRAIN_SAMPLES})",
        "training_samples":  n_samples,
        "feature_cols":      FINML_FEATURES,
        "feature_importance": {},
        "top_feature":       "score",
        "train_accuracy":    None,
        "error":             f"insufficient data ({n_samples} samples)",
    }


# ─── LAYER 1 AGGREGATOR ──────────────────────────────────────────────────────

def run_layer1(
    symbol:    str,
    headlines: list[str],
    features:  dict,
    window:    int = 30,
    hf_token:  str = "",
) -> dict:
    """
    Run both Layer 1A (FinBERT) and 1B (FinML) and return combined dict
    ready to be injected into the Layer 2 LLM prompt.
    """
    print(f"\n  [Layer 1A] Running FinBERT sentiment on {len(headlines)} headlines...")
    sentiment = run_finbert_sentiment(headlines, hf_token)
    if sentiment["error"]:
        print(f"    ⚠ FinBERT: {sentiment['error']}")
    else:
        print(f"    ✅ FinBERT: net_score={sentiment['net_score']:+.3f} "
              f"→ {sentiment['sentiment_label']} "
              f"({sentiment['bullish_count']}↑ {sentiment['bearish_count']}↓ {sentiment['neutral_count']}→)")

    print(f"  [Layer 1B] Running FinML classifier...")
    ml = run_finml_classifier(symbol, features, window)
    if ml["error"]:
        print(f"    ⚠ FinML: {ml['error']}")
    else:
        print(f"    ✅ FinML: prob={ml['ml_confidence']:.3f} → {ml['ml_label']} "
              f"[{ml['model_type']}]")

    return {
        "finbert":  sentiment,
        "finml":    ml,
        "combined": _combine_signals(sentiment, ml),
    }


def _combine_signals(sentiment: dict, ml: dict) -> dict:
    """
    Combine FinBERT + FinML into a single agreement/conflict assessment.
    This is what gets passed to Layer 2 LLM as the summary.
    """
    sent_label = sentiment["sentiment_label"]   # Bullish / Neutral / Bearish
    ml_label   = ml["ml_label"]                 # High / Medium / Low
    net        = sentiment["net_score"]
    prob       = ml["ml_confidence"]

    # Agreement check
    sent_bullish = sent_label == "Bullish"
    sent_bearish = sent_label == "Bearish"
    ml_confident = ml_label in ("High", "Medium")

    if sent_bullish and ml_confident:
        agreement = "STRONG_BUY_SIGNAL"
        note = "FinBERT bullish + FinML high confidence → strong setup"
    elif sent_bearish and ml_confident:
        agreement = "STRONG_SELL_SIGNAL"
        note = "FinBERT bearish + FinML high confidence → strong short setup"
    elif sent_bullish and ml_label == "Low":
        agreement = "SENTIMENT_ONLY"
        note = "FinBERT bullish but FinML low confidence → weaker signal"
    elif sent_bearish and ml_label == "Low":
        agreement = "SENTIMENT_ONLY"
        note = "FinBERT bearish but FinML low confidence → weaker signal"
    elif sent_label == "Neutral" and ml_confident:
        agreement = "TECHNICAL_ONLY"
        note = "Neutral news but FinML confident → pure technical play"
    else:
        agreement = "WEAK_SIGNAL"
        note = "Both FinBERT neutral and FinML low confidence → proceed with caution"

    return {
        "agreement":        agreement,
        "note":             note,
        "finbert_label":    sent_label,
        "finbert_score":    net,
        "finml_label":      ml_label,
        "finml_prob":       prob,
        "overall_strength": _overall_strength(net, prob),
    }


def _overall_strength(net_sentiment: float, ml_prob: float) -> float:
    """Weighted combination: 40% sentiment + 60% ML probability."""
    # Normalize sentiment to 0-1
    sent_normalized = (net_sentiment + 1) / 2
    return round(0.4 * sent_normalized + 0.6 * ml_prob, 4)


# ─── LAYER 2: LLM REASONING ──────────────────────────────────────────────────

def run_layer2_reasoning(
    symbol:     str,
    window:     int,
    layer1:     dict,
    raw_tech:   dict,   # technical indicators from get_technical_indicators()
    raw_market: dict,   # market context from get_nse_market_context()
    llm_config: dict,
) -> dict:
    """
    Layer 2: GPT-4 reasons ONLY on pre-computed quant outputs.
    It does NOT call any tools — all data is pre-fetched and structured.

    raw_tech:   dict with keys like rsi, macd_trend, ema_trend, atr_pct etc.
    raw_market: dict with keys like nifty_change_pct, vix, banknifty_change_pct etc.
    """
    import openai

    ist_now     = datetime.utcnow() + timedelta(hours=5, minutes=30)
    ist_now_str = ist_now.strftime("%Y-%m-%d %H:%M IST")
    prompt      = _build_layer2_prompt(symbol, window, ist_now_str, layer1, raw_tech, raw_market)

    try:
        # Load API key from config
        config    = llm_config["config_list"][0]
        api_key   = config.get("api_key") or config.get("OPENAI_API_KEY")
        api_base  = config.get("base_url") or config.get("api_base") or "https://api.openai.com/v1"
        model     = config.get("model", "gpt-4")

        client    = openai.OpenAI(api_key=api_key, base_url=api_base)
        response  = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are an NSE intraday quantitative analyst. "
                        "You receive pre-computed quantitative signals from FinBERT (sentiment) "
                        "and FinML (technical ML classifier). "
                        "Your job is to synthesize these signals and make a final trading decision. "
                        "You do NOT have access to raw data — reason only from the signals provided. "
                        "Be concise. Reply TERMINATE when done."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=800,
        )

        raw_output = response.choices[0].message.content
        parsed     = _parse_layer2_output(raw_output, symbol, window)
        parsed["raw_output"] = raw_output
        parsed["prompt_used"] = prompt
        return parsed

    except Exception as e:
        return {
            "error":         str(e),
            "direction":     None,
            "confidence":    None,
            "reasoning":     None,
            "raw_output":    None,
        }


def _build_layer2_prompt(
    symbol:     str,
    window:     int,
    ist_now_str: str,
    layer1:     dict,
    raw_tech:   dict,
    raw_market: dict,
) -> str:
    """Build the Layer 2 reasoning prompt from pre-computed quant signals."""

    finbert  = layer1["finbert"]
    finml    = layer1["finml"]
    combined = layer1["combined"]

    # Format per-headline sentiment table
    headlines_str = ""
    for h in finbert.get("per_headline", [])[:5]:
        icon = "📈" if h["label"] == "positive" else ("📉" if h["label"] == "negative" else "➡")
        headlines_str += f"    {icon} [{h['label'].upper():<8} {h['positive']:.2f}pos/{h['negative']:.2f}neg] {h['headline']}\n"
    if not headlines_str:
        headlines_str = "    (no headlines available or FinBERT unavailable)\n"

    # Format feature importance
    fi = finml.get("feature_importance", {})
    fi_str = "  ".join(f"{k}={v:.3f}" for k, v in sorted(fi.items(), key=lambda x: -x[1])[:3])
    if not fi_str:
        fi_str = "rule-based (insufficient training data)"

    # Technical signals summary
    tech_str = ""
    for key in ["rsi", "macd_trend", "ema_trend", "bb_position", "obv_trend",
                "adx", "vwap_position", "volume_vs_avg"]:
        val = raw_tech.get(key)
        if val is not None:
            tech_str += f"  {key:<20}: {val}\n"
    if not tech_str:
        tech_str = "  (technical data not provided to Layer 2)\n"

    # Market context summary
    mkt_str = ""
    for key in ["nifty_change_pct", "banknifty_change_pct", "vix", "usd_inr"]:
        val = raw_market.get(key)
        if val is not None:
            mkt_str += f"  {key:<24}: {val}\n"
    if not mkt_str:
        mkt_str = "  (market data not provided to Layer 2)\n"

    return f"""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
TWO-LAYER QUANTITATIVE ANALYSIS — {symbol}
As of: {ist_now_str} | Horizon: next {window} minutes
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

══ LAYER 1A: FINBERT NEWS SENTIMENT (ProsusAI/finbert) ══
  Net Sentiment Score : {finbert['net_score']:+.4f}  (range: -1.0 bearish to +1.0 bullish)
  Sentiment Label     : {finbert['sentiment_label']}
  Headlines Scored    : {finbert['headlines_scored']}
  Distribution        : {finbert['bullish_count']} bullish / {finbert['bearish_count']} bearish / {finbert['neutral_count']} neutral
  FinBERT Error       : {finbert['error'] or 'None'}

  Top headline scores:
{headlines_str}

══ LAYER 1B: FINML TECHNICAL CLASSIFIER (RandomForest) ══
  ML Probability      : {finml['ml_confidence']:.4f}  (probability direction_correct = True)
  ML Label            : {finml['ml_label']}
  Model Type          : {finml['model_type']}
  Training Samples    : {finml['training_samples']}
  Feature Importance  : {fi_str}
  CV Accuracy         : {finml.get('train_accuracy') or 'N/A'}
  FinML Error         : {finml['error'] or 'None'}

══ LAYER 1 COMBINED SIGNAL ══
  Agreement Type      : {combined['agreement']}
  Overall Strength    : {combined['overall_strength']:.4f}  (0=weak, 1=strong)
  Note                : {combined['note']}

══ SUPPORTING TECHNICAL INDICATORS ══
{tech_str}
══ MARKET CONTEXT ══
{mkt_str}

══ YOUR TASK (Layer 2 Reasoning) ══
Based on the above pre-computed quantitative signals:

1. INTERPRET the FinBERT sentiment — does the news flow support or oppose the technical picture?
2. INTERPRET the FinML probability — is the technical setup historically reliable?
3. IDENTIFY any conflicts between sentiment and technical signals
4. MAKE A FINAL CALL for the next {window} minutes

Output EXACTLY this format:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NSE {window}-MIN TWO-LAYER FORECAST — {symbol}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

QUANTITATIVE SYNTHESIS:
  FinBERT Interpretation : [1-2 sentences on what the sentiment means]
  FinML Interpretation   : [1-2 sentences on what the ML probability means]
  Signal Conflict        : [Yes/No — explain if yes]

FINAL DECISION:
  Direction    : [Bullish / Bearish / Sideways]
  Confidence   : [High / Medium / Low]
  Key Reason   : [1 sentence — the single most important factor]
  Risk Note    : [1 sentence — main risk to this call]

TERMINATE
"""


def _parse_layer2_output(text: str, symbol: str, window: int) -> dict:
    """Parse structured output from Layer 2 LLM."""
    import re

    def extract(pattern, default=None):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(1).strip() if m else default

    direction  = extract(r"Direction\s*:\s*(Bullish|Bearish|Sideways)")
    confidence = extract(r"Confidence\s*:\s*(High|Medium|Low)")
    key_reason = extract(r"Key Reason\s*:\s*(.+)")
    risk_note  = extract(r"Risk Note\s*:\s*(.+)")
    fb_interp  = extract(r"FinBERT Interpretation\s*:\s*(.+)")
    ml_interp  = extract(r"FinML Interpretation\s*:\s*(.+)")
    conflict   = extract(r"Signal Conflict\s*:\s*(Yes|No[^,\n]*)")

    return {
        "symbol":             symbol,
        "window_min":         window,
        "direction":          direction,
        "confidence":         confidence,
        "key_reason":         key_reason,
        "risk_note":          risk_note,
        "finbert_interp":     fb_interp,
        "finml_interp":       ml_interp,
        "signal_conflict":    conflict,
        "parse_ok":           all(x is not None for x in [direction, confidence]),
        "timestamp":          datetime.now(timezone.utc).isoformat(),
    }


# ─── MAIN ENTRY POINT ────────────────────────────────────────────────────────

def run_two_layer_analysis(
    symbol:     str,
    window:     int = 30,
    headlines:  list[str] | None = None,
    tech_data:  dict | None = None,
    market_data: dict | None = None,
) -> dict:
    """
    Full two-layer pipeline for one stock.

    Parameters
    ----------
    symbol      : NSE ticker e.g. "HDFCBANK.NS"
    window      : prediction horizon in minutes (30 or 60)
    headlines   : list of news headline strings (from get_company_news)
    tech_data   : dict from get_technical_indicators (rsi, macd_trend etc.)
    market_data : dict from get_nse_market_context (nifty_change_pct, vix etc.)

    Returns
    -------
    dict with keys: layer1, layer2, summary
    """
    print(f"\n{'='*60}")
    print(f"TWO-LAYER QUANT ANALYSIS — {symbol} ({window}-min)")
    print(f"{'='*60}")

    headlines   = headlines   or []
    tech_data   = tech_data   or {}
    market_data = market_data or {}

    # Extract features for FinML
    features = {
        "rr_ratio":      tech_data.get("rr_ratio", 0),
        "atr_pct":       tech_data.get("atr_pct", 0),
        "exp_move_pct":  tech_data.get("expected_intraday_range_pct", 0) * math.sqrt(window / 375)
                         if tech_data.get("expected_intraday_range_pct") else 0,
        "score":         tech_data.get("confidence_score", 0),
    }

    # ── Layer 1 ──────────────────────────────────────────────────────────────
    hf_token = _load_hf_token()
    layer1   = run_layer1(symbol, headlines, features, window, hf_token)

    # ── Layer 2 ──────────────────────────────────────────────────────────────
    print(f"\n  [Layer 2] Running GPT-4 reasoning on quant signals...")
    try:
        llm_config = _load_openai_config()
        layer2     = run_layer2_reasoning(
            symbol, window, layer1, tech_data, market_data, llm_config
        )
        if layer2.get("error"):
            print(f"    ⚠ Layer 2 error: {layer2['error']}")
        else:
            print(f"    ✅ Layer 2: {layer2.get('direction')} / {layer2.get('confidence')}")
            print(f"       Reason: {layer2.get('key_reason', 'N/A')}")
    except Exception as e:
        layer2 = {"error": str(e), "direction": None, "confidence": None}
        print(f"    ⚠ Layer 2 failed: {e}")

    # ── Summary ──────────────────────────────────────────────────────────────
    summary = {
        "symbol":            symbol,
        "window_min":        window,
        "finbert_score":     layer1["finbert"]["net_score"],
        "finbert_sentiment": layer1["finbert"]["sentiment_label"],
        "finml_prob":        layer1["finml"]["ml_confidence"],
        "finml_label":       layer1["finml"]["ml_label"],
        "signal_agreement":  layer1["combined"]["agreement"],
        "overall_strength":  layer1["combined"]["overall_strength"],
        "final_direction":   layer2.get("direction"),
        "final_confidence":  layer2.get("confidence"),
        "key_reason":        layer2.get("key_reason"),
        "risk_note":         layer2.get("risk_note"),
        "signal_conflict":   layer2.get("signal_conflict"),
    }

    print(f"\n{'─'*60}")
    print(f"SUMMARY — {symbol}")
    print(f"  FinBERT  : {summary['finbert_score']:+.3f} → {summary['finbert_sentiment']}")
    print(f"  FinML    : {summary['finml_prob']:.3f} → {summary['finml_label']}")
    print(f"  Agreement: {summary['signal_agreement']}")
    print(f"  FINAL    : {summary['final_direction']} / {summary['final_confidence']}")
    print(f"{'─'*60}")

    return {
        "layer1":  layer1,
        "layer2":  layer2,
        "summary": summary,
    }


# ─── STANDALONE TEST ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    """
    Quick test with dummy data.
    Replace with real data from yfinance_utils_nse.py tools.
    """
    test_headlines = [
        "HDFC Bank reports record Q3 net profit, beats analyst estimates",
        "RBI raises concerns about rising retail loan NPAs in banking sector",
        "HDFC Bank launches new digital lending platform for SME customers",
        "Foreign institutional investors increase stake in HDFC Bank",
        "Banking sector faces headwinds from rising interest rate environment",
    ]

    test_tech = {
        "rsi":                        58.3,
        "macd_trend":                 "bullish",
        "ema_trend":                  "price above 20 EMA and 50 EMA",
        "bb_position":                "middle band",
        "obv_trend":                  "rising",
        "adx":                        28.5,
        "vwap_position":              "above",
        "volume_vs_avg":              "1.2x average",
        "atr_pct":                    0.82,
        "rr_ratio":                   2.1,
        "expected_intraday_range_pct": 0.82,
        "confidence_score":           4,
    }

    test_market = {
        "nifty_change_pct":     0.45,
        "banknifty_change_pct": 0.62,
        "vix":                  14.2,
        "usd_inr":              83.45,
    }

    result = run_two_layer_analysis(
        symbol      = "HDFCBANK.NS",
        window      = 30,
        headlines   = test_headlines,
        tech_data   = test_tech,
        market_data = test_market,
    )

    print("\n\nFULL LAYER 2 OUTPUT:")
    print(result["layer2"].get("raw_output", "N/A"))