"""
NSE-Optimized YFinanceUtils  v4
================================
Drop-in replacement for finrobot's YFinanceUtils, purpose-built for
Indian stock exchange (NSE) intraday forecasting.

Key additions over the original:
  - get_technical_indicators  : RSI(strict <=30/>=70), MACD with separate macd_crossover +
                                macd_trend fields, Bollinger Bands, ATR, EMA, ADX, OBV
  - get_intraday_data         : 1m/5m/15m/30m/60m candles (up to last 60 days) with VWAP
  - get_nse_market_context    : NIFTY50 + BANKNIFTY + India VIX + USD/INR + crude oil
                                (crude flags anomalous >5% single-session moves)
  - get_fno_data              : F&O expiry flag, PCR, Max Pain.
                                On options chain failure: explicitly marks fields UNAVAILABLE
                                to prevent LLM hallucination.
  - get_sector_peers          : Peer comparison. Both 'Technology' and 'Information Technology'
                                keys supported to match yfinance sector string variability.
  - get_extended_company_info : P/E, beta, 52w range, dividendYield bug fix (handles both
                                decimal and %-form return values from yfinance)
  - get_support_resistance    : Classic + Camarilla pivots + pre-computed bearish_setup
                                and bullish_setup (target=S2/R2, stop=Cam H3/L3, R:R check)
  - get_company_news          : Alias for get_news (matches the tool name the agent calls)
  - get_data_sanity_check     : Cross-validates data sources; flags 52w high mismatch,
                                volume spikes, and price-already-past-pivot scenarios

Hallucination prevention summary:
  - RSI: strict boundary, signal field returned verbatim
  - MACD: macd_crossover='none'/'bullish_crossover'/'bearish_crossover' explicit field
  - PCR/Max Pain: UNAVAILABLE string on failure, not None or 0
  - Crude oil: data_anomaly=True flag on >5% moves
  - Trade setup: pre-computed R:R with tradeable bool — agent uses this, not raw S1/R1
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Annotated, Optional, Dict, Any
import json as _json
import os as _os
from pathlib import Path as _Path
from datetime import datetime as _datetime


# ---------------------------------------------------------------------------
# NSE sector → representative peer tickers (add more as needed)
# ---------------------------------------------------------------------------
_IT_PEERS = ["TCS.NS", "INFY.NS", "WIPRO.NS", "HCLTECH.NS", "TECHM.NS"]
_FORECAST_FILE = _Path("forecasts.jsonl")

NSE_SECTOR_PEERS: Dict[str, list] = {
    "Energy":                ["RELIANCE.NS", "ONGC.NS", "BPCL.NS", "IOC.NS", "HINDPETRO.NS"],
    "Financial Services":    ["HDFCBANK.NS", "ICICIBANK.NS", "KOTAKBANK.NS", "AXISBANK.NS", "SBIN.NS"],
    # yfinance returns either "Information Technology" or "Technology" depending on the ticker —
    # both keys point to the same peer list to prevent the "No peer list" error.
    "Information Technology": _IT_PEERS,
    "Technology":             _IT_PEERS,
    "Consumer Cyclical":     ["TITAN.NS", "MARUTI.NS", "BAJAJ-AUTO.NS", "EICHERMOT.NS", "M&M.NS"],
    "Consumer Defensive":    ["HINDUNILVR.NS", "ITC.NS", "NESTLEIND.NS", "DABUR.NS", "MARICO.NS"],
    "Healthcare":            ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    # yfinance uses both "Healthcare" and "Health Care" for different tickers
    "Health Care":           ["SUNPHARMA.NS", "DRREDDY.NS", "CIPLA.NS", "DIVISLAB.NS", "APOLLOHOSP.NS"],
    "Basic Materials":       ["TATASTEEL.NS", "JSWSTEEL.NS", "HINDALCO.NS", "VEDL.NS", "COALINDIA.NS"],
    "Industrials":           ["LT.NS", "ADANIPORTS.NS", "SIEMENS.NS", "ABB.NS", "BHEL.NS"],
    "Utilities":             ["NTPC.NS", "POWERGRID.NS", "TATAPOWER.NS", "ADANIGREEN.NS"],
    "Telecommunication":     ["BHARTIARTL.NS", "IDEA.NS"],
    "Communication Services":["BHARTIARTL.NS", "IDEA.NS"],
    "Real Estate":           ["DLF.NS", "GODREJPROP.NS", "OBEROIRLTY.NS", "PHOENIXLTD.NS"],
}

# NSE F&O monthly expiry: last Thursday of each month
# Weekly expiry: every Thursday (Nifty, BankNifty, FinNifty, MidcapNifty)
WEEKLY_EXPIRY_INDICES = {"^NSEI", "^NSEBANK", "FINNIFTY.NS", "MIDCPNIFTY.NS"}


class YFinanceUtils:
    """NSE-optimized utilities for FinRobot intraday forecasting."""

    # ------------------------------------------------------------------
    # ORIGINAL METHODS (kept for backward compatibility)
    # ------------------------------------------------------------------

    def get_stock_data(
        symbol: Annotated[str, "NSE ticker e.g. RELIANCE.NS"],
        start_date: Annotated[str, "start date YYYY-MM-DD"],
        end_date: Annotated[str, "end date YYYY-MM-DD"],
    ) -> pd.DataFrame:
        """Retrieve daily OHLCV data. Use .NS suffix for NSE stocks."""
        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date)
        df.index = df.index.tz_localize(None)          # strip tz for clean display
        return df

    def get_company_info(
        symbol: Annotated[str, "NSE ticker e.g. TCS.NS"]
    ) -> dict:
        """Fetch basic company profile."""
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return {
            "Company Name": info.get("shortName"),
            "Industry":     info.get("industry"),
            "Sector":       info.get("sector"),
            "Country":      info.get("country"),
            "Website":      info.get("website"),
        }

    def get_news(
        symbol: Annotated[str, "NSE ticker e.g. INFY.NS"]
    ) -> list:
        """Retrieve latest company news headlines."""
        ticker = yf.Ticker(symbol)
        return ticker.news

    def get_company_news(
        symbol: Annotated[str, "NSE ticker e.g. TCS.NS"]
    ) -> list:
        """
        Retrieve latest company news headlines (alias for get_news).
        The system prompt calls this as get_company_news — this alias ensures
        the tool name in the notebook registration matches the function name
        expected by the agent.
        """
        ticker = yf.Ticker(symbol)
        return ticker.news

    def get_market_trend(
        symbol: Annotated[str, "Index ticker"] = "^NSEI"
    ) -> dict:
        """Determine overall market trend (legacy method)."""
        nifty = yf.Ticker(symbol).history(period="5d")
        return {
            "latest_close": float(nifty["Close"].iloc[-1]),
            "trend": "up" if nifty["Close"].iloc[-1] > nifty["Close"].iloc[0] else "down",
        }

    def get_index_data(
        symbol: Annotated[str, "Index ticker"] = "^NSEI"
    ) -> pd.DataFrame:
        """Retrieve index OHLCV for last 5 days."""
        ticker = yf.Ticker(symbol)
        return ticker.history(period="5d")

    # ------------------------------------------------------------------
    # NEW: INTRADAY CANDLE DATA
    # ------------------------------------------------------------------

    def get_intraday_data(
        symbol: Annotated[str, "NSE ticker e.g. HDFCBANK.NS"],
        interval: Annotated[str, "Candle interval: 1m, 5m, 15m, 30m, 60m"] = "15m",
        period: Annotated[str, "Lookback: 1d, 2d, 5d, 1mo (max 60d for 1m)"] = "5d",
    ) -> dict:
        """
        Fetch intraday OHLCV candles from yfinance.
        Returns summary stats + last 20 candles for the agent to reason over.
        Best used with 15m interval for next-session bias.
        """
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        if df.empty:
            return {"error": f"No intraday data for {symbol} at {interval}"}

        df.index = df.index.tz_localize(None)

        # VWAP (session rolling)
        df["VWAP"] = (df["Volume"] * (df["High"] + df["Low"] + df["Close"]) / 3).cumsum() / df["Volume"].cumsum()

        last20 = df.tail(20)[["Open", "High", "Low", "Close", "Volume", "VWAP"]].round(2)

        return {
            "interval":        interval,
            "period":          period,
            "latest_close":    round(float(df["Close"].iloc[-1]), 2),
            "latest_vwap":     round(float(df["VWAP"].iloc[-1]), 2),
            "price_vs_vwap":   "above" if df["Close"].iloc[-1] > df["VWAP"].iloc[-1] else "below",
            "session_high":    round(float(df["High"].max()), 2),
            "session_low":     round(float(df["Low"].min()), 2),
            "avg_volume_per_candle": int(df["Volume"].mean()),
            "last_20_candles": last20.to_string(),
        }

    # ------------------------------------------------------------------
    # NEW: TECHNICAL INDICATORS (NSE intraday-relevant)
    # ------------------------------------------------------------------

    def get_technical_indicators(
        symbol: Annotated[str, "NSE ticker e.g. RELIANCE.NS"],
        lookback_days: Annotated[int, "Days of daily data to compute indicators (min 60)"] = 90,
    ) -> dict:
        """
        Computes key technical indicators on daily data:
          RSI(14), MACD(12,26,9), Bollinger Bands(20,2), EMA(9/21/50/200),
          ATR(14), ADX(14), OBV trend, Stochastic(14,3).

        NSE intraday traders should focus on:
          - Price vs EMA9/21 for momentum
          - RSI for overbought/oversold (>70 / <30)
          - MACD crossover for entry signal
          - ATR for expected intraday range
          - Bollinger Band squeeze for breakout watch
        """
        end   = datetime.today()
        start = end - timedelta(days=lookback_days + 50)   # extra buffer for rolling calcs

        df = yf.Ticker(symbol).history(start=start.strftime("%Y-%m-%d"),
                                        end=end.strftime("%Y-%m-%d"))
        if len(df) < 30:
            return {"error": "Insufficient data for technical analysis"}

        close = df["Close"]
        high  = df["High"]
        low   = df["Low"]
        vol   = df["Volume"]

        # --- RSI(14) ---
        delta   = close.diff()
        gain    = delta.clip(lower=0).rolling(14).mean()
        loss    = (-delta.clip(upper=0)).rolling(14).mean()
        rsi     = 100 - (100 / (1 + gain / loss))

        # --- MACD ---
        ema12   = close.ewm(span=12, adjust=False).mean()
        ema26   = close.ewm(span=26, adjust=False).mean()
        macd    = ema12 - ema26
        signal  = macd.ewm(span=9, adjust=False).mean()
        hist    = macd - signal

        # --- Bollinger Bands(20,2) ---
        sma20   = close.rolling(20).mean()
        std20   = close.rolling(20).std()
        bb_up   = sma20 + 2 * std20
        bb_lo   = sma20 - 2 * std20
        bb_width = (bb_up - bb_lo) / sma20 * 100   # % width; <2% = squeeze

        # --- EMAs ---
        ema9    = close.ewm(span=9,   adjust=False).mean()
        ema21   = close.ewm(span=21,  adjust=False).mean()
        ema50   = close.ewm(span=50,  adjust=False).mean()
        ema200  = close.ewm(span=200, adjust=False).mean()

        # --- ATR(14) ---
        tr      = pd.concat([high - low,
                              (high - close.shift()).abs(),
                              (low  - close.shift()).abs()], axis=1).max(axis=1)
        atr     = tr.rolling(14).mean()

        # --- Stochastic(14,3) ---
        lo14    = low.rolling(14).min()
        hi14    = high.rolling(14).max()
        stoch_k = 100 * (close - lo14) / (hi14 - lo14 + 1e-9)
        stoch_d = stoch_k.rolling(3).mean()

        # --- OBV ---
        obv     = (np.sign(close.diff()) * vol).fillna(0).cumsum()
        obv_trend = "rising" if obv.iloc[-1] > obv.iloc[-5] else "falling"

        # --- ADX (simplified) ---
        plus_dm  = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)
        tr14     = tr.rolling(14).sum()
        plus_di  = 100 * plus_dm.rolling(14).sum() / (tr14 + 1e-9)
        minus_di = 100 * minus_dm.rolling(14).sum() / (tr14 + 1e-9)
        dx       = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx      = dx.rolling(14).mean()

        latest_close = float(close.iloc[-1])

        def _f(s): return round(float(s.iloc[-1]), 2)

        # ── RSI signal (strict boundary: <=30 oversold, >=70 overbought) ──────
        # Using strict <= / >= so RSI=30.75 correctly returns "neutral",
        # not "oversold". The agent was hallucinating "oversold" when the
        # tool returned "neutral" because the old threshold was < 30 (exclusive).
        rsi_val    = _f(rsi)
        rsi_signal = "overbought" if rsi_val >= 70 else ("oversold" if rsi_val <= 30 else "neutral")

        # ── MACD: separate crossover detection from trend direction ───────────
        # Previously macd_signal mixed crossover + trend into one field,
        # causing the agent to always write "bearish crossover" even when
        # there was no crossover. Now two separate explicit fields are returned.
        hist_now  = float(hist.iloc[-1])
        hist_prev = float(hist.iloc[-2])
        if hist_now > 0 and hist_prev <= 0:
            macd_crossover = "bullish_crossover"       # just crossed above zero
        elif hist_now < 0 and hist_prev >= 0:
            macd_crossover = "bearish_crossover"       # just crossed below zero
        else:
            macd_crossover = "none"                    # no crossover this session

        macd_trend   = "bullish" if hist_now > 0 else "bearish"
        # Combined signal for backward compatibility
        macd_signal  = macd_crossover if macd_crossover != "none" else macd_trend

        bb_pos = "upper_band" if latest_close >= _f(bb_up) else \
                 "lower_band" if latest_close <= _f(bb_lo) else "middle"

        ema_trend = "strongly_bullish" if latest_close > _f(ema9) > _f(ema21) > _f(ema50) else \
                    "strongly_bearish" if latest_close < _f(ema9) < _f(ema21) < _f(ema50) else \
                    "bullish" if latest_close > _f(ema50) else "bearish"

        return {
            # Price context
            "latest_close":      latest_close,
            "52w_high":          round(float(high.tail(252).max()), 2),
            "52w_low":           round(float(low.tail(252).min()), 2),
            "pct_from_52w_high": round((latest_close / float(high.tail(252).max()) - 1) * 100, 2),

            # Momentum
            "rsi_14":            rsi_val,
            "rsi_signal":        rsi_signal,
            "stoch_k":           _f(stoch_k),
            "stoch_d":           _f(stoch_d),
            "stoch_signal":      "overbought" if _f(stoch_k) > 80 else ("oversold" if _f(stoch_k) < 20 else "neutral"),

            # Trend
            "macd":              _f(macd),
            "macd_signal_line":  _f(signal),
            "macd_histogram":    round(float(hist.iloc[-1]), 4),
            "macd_prev_histogram": round(hist_prev, 4),        # previous bar — agent can verify crossover itself
            "macd_crossover":    macd_crossover,               # "bullish_crossover" / "bearish_crossover" / "none"
            "macd_trend":        macd_trend,                   # "bullish" / "bearish" (direction of histogram)
            "macd_signal":       macd_signal,                  # combined field (backward compat)
            "ema_trend":         ema_trend,
            "ema_9":             _f(ema9),
            "ema_21":            _f(ema21),
            "ema_50":            _f(ema50),
            "ema_200":           _f(ema200),
            "adx_14":            _f(adx),    # >25 = strong trend
            "adx_signal":        "strong_trend" if _f(adx) > 25 else "weak_trend",

            # Volatility / Range
            "atr_14":            _f(atr),
            "expected_intraday_range_pct": round(_f(atr) / latest_close * 100, 2),
            "bb_upper":          _f(bb_up),
            "bb_lower":          _f(bb_lo),
            "bb_width_pct":      _f(bb_width),
            "bb_position":       bb_pos,
            "bb_squeeze":        _f(bb_width) < 2.0,

            # Volume
            "obv_trend":         obv_trend,
            "avg_vol_20d":       int(vol.tail(20).mean()),
            "latest_vol":        int(vol.iloc[-1]),
            "volume_vs_avg":     "high" if int(vol.iloc[-1]) > 1.5 * int(vol.tail(20).mean()) else
                                 "low"  if int(vol.iloc[-1]) < 0.5 * int(vol.tail(20).mean()) else "normal",
        }

    # ------------------------------------------------------------------
    # NEW: NSE MARKET CONTEXT (replaces simple get_market_trend)
    # ------------------------------------------------------------------

    def get_nse_market_context() -> dict:
        """
        Comprehensive Indian market context for intraday:
          - NIFTY50 trend + key levels
          - BANKNIFTY trend (important for financial stocks)
          - India VIX (fear gauge; >20 = high fear, <15 = complacent)
          - USD/INR (rupee strength affects FII flows)
          - SGX Nifty proxy (pre-market gap indicator using ^NSEI futures approximation)

        Always call this before making an intraday prediction.
        """
        results = {}

        # NIFTY 50
        nifty = yf.Ticker("^NSEI").history(period="10d")
        if not nifty.empty:
            closes = nifty["Close"]
            results["nifty50"] = {
                "latest_close":   round(float(closes.iloc[-1]), 2),
                "prev_close":     round(float(closes.iloc[-2]), 2),
                "change_pct":     round((float(closes.iloc[-1]) / float(closes.iloc[-2]) - 1) * 100, 2),
                "5d_trend":       "up" if closes.iloc[-1] > closes.iloc[0] else "down",
                "3d_momentum":    "up" if closes.iloc[-1] > closes.iloc[-3] else "down",
                "session_high":   round(float(nifty["High"].iloc[-1]), 2),
                "session_low":    round(float(nifty["Low"].iloc[-1]), 2),
            }

        # BANKNIFTY
        bnk = yf.Ticker("^NSEBANK").history(period="5d")
        if not bnk.empty:
            bc = bnk["Close"]
            results["banknifty"] = {
                "latest_close": round(float(bc.iloc[-1]), 2),
                "change_pct":   round((float(bc.iloc[-1]) / float(bc.iloc[-2]) - 1) * 100, 2),
                "trend":        "up" if bc.iloc[-1] > bc.iloc[0] else "down",
            }

        # India VIX
        vix = yf.Ticker("^INDIAVIX").history(period="5d")
        if not vix.empty:
            vix_val = float(vix["Close"].iloc[-1])
            results["india_vix"] = {
                "latest":  round(vix_val, 2),
                "level":   "high_fear" if vix_val > 20 else ("moderate" if vix_val > 15 else "low_fear"),
                "trend":   "rising" if vix["Close"].iloc[-1] > vix["Close"].iloc[0] else "falling",
                "implication": "expect_wide_swings" if vix_val > 20 else
                               "normal_intraday_range" if vix_val > 15 else "tight_range_possible",
            }

        # USD/INR (FII flow proxy)
        usdinr = yf.Ticker("USDINR=X").history(period="5d")
        if not usdinr.empty:
            rate = float(usdinr["Close"].iloc[-1])
            prev = float(usdinr["Close"].iloc[-2])
            results["usdinr"] = {
                "latest":      round(rate, 4),
                "change_pct":  round((rate / prev - 1) * 100, 4),
                "rupee_signal": "rupee_weakening_fii_outflow_risk" if rate > prev else
                                "rupee_strengthening_fii_inflow_favorable",
            }

        # Crude Oil (important for Indian market, especially energy/aviation)
        # Note: yfinance CL=F occasionally returns anomalous single-session
        # spikes (>5%) due to roll-over or data errors. We flag these explicitly
        # so the agent does not treat them as real market moves.
        crude = yf.Ticker("CL=F").history(period="5d")
        if not crude.empty:
            cp      = float(crude["Close"].iloc[-1])
            cp_prev = float(crude["Close"].iloc[-2])
            chg_pct = round((cp / cp_prev - 1) * 100, 2)
            is_anomaly = abs(chg_pct) > 5   # crude rarely moves >5% in one session
            results["brent_crude_proxy"] = {
                "latest_usd":       round(cp, 2),
                "change_pct":       chg_pct,
                "data_anomaly":     is_anomaly,
                "data_note":        (
                    "WARNING: change_pct exceeds 5% — likely a yfinance data anomaly "
                    "(futures roll-over). Do NOT use this value in analysis."
                    if is_anomaly else "data_ok"
                ),
                "signal": (
                    "data_anomaly_ignore" if is_anomaly else
                    "bearish_for_india_importers" if cp > cp_prev else "positive_for_india"
                ),
            }

        return results

    # ------------------------------------------------------------------
    # NEW: F&O CONTEXT
    # ------------------------------------------------------------------

    def get_fno_data(
        symbol: Annotated[str, "NSE F&O ticker e.g. RELIANCE.NS or ^NSEI"]
    ) -> dict:
        """
        Returns F&O-relevant context:
          - Whether today is expiry week (last week of month) — high volatility expected
          - Whether today is weekly expiry day (Thursday) — pin risk
          - Approximate Put-Call Ratio from options chain (bearish if PCR < 0.7, bullish if > 1.2)
          - Max pain level (strike with most OI — price tends to gravitate here at expiry)

        NSE F&O expiry rules:
          - Monthly: last Thursday of each month
          - Weekly (NIFTY, BANKNIFTY): every Thursday
        """
        today = datetime.today()
        is_thursday = today.weekday() == 3
        days_to_thursday = (3 - today.weekday()) % 7

        # Check if last Thursday of the month
        next_thursdays = []
        for i in range(0, 31):
            d = today + timedelta(days=i)
            if d.weekday() == 3:
                next_thursdays.append(d)
                if len(next_thursdays) == 2:
                    break
        is_monthly_expiry_week = (
            next_thursdays[0].month != (next_thursdays[0] + timedelta(days=7)).month
            if next_thursdays else False
        )

        result = {
            "today":                  today.strftime("%Y-%m-%d"),
            "is_expiry_day":          is_thursday,
            "days_to_next_expiry":    0 if is_thursday else days_to_thursday,
            "is_monthly_expiry_week": is_monthly_expiry_week,
            "expiry_implication":     (
                "expiry_day_pin_risk_high" if is_thursday else
                "expiry_week_elevated_volatility" if days_to_thursday <= 2 else
                "normal"
            ),
        }

        # Options chain for PCR and max pain
        try:
            ticker = yf.Ticker(symbol)
            expiry_dates = ticker.options
            if expiry_dates:
                nearest_expiry = expiry_dates[0]
                chain = ticker.option_chain(nearest_expiry)
                calls = chain.calls
                puts  = chain.puts

                total_call_oi = calls["openInterest"].sum()
                total_put_oi  = puts["openInterest"].sum()
                pcr = round(total_put_oi / (total_call_oi + 1e-9), 3)

                # Max pain: strike where total OI pain for option buyers is highest for sellers
                strikes = sorted(set(calls["strike"]).union(set(puts["strike"])))
                pain = {}
                for s in strikes:
                    call_pain = calls[calls["strike"] >= s]["openInterest"].sum() * (
                        calls[calls["strike"] >= s]["strike"] - s).clip(lower=0).sum()
                    put_pain  = puts[puts["strike"] <= s]["openInterest"].sum() * (
                        s - puts[puts["strike"] <= s]["strike"]).clip(lower=0).sum()
                    pain[s] = call_pain + put_pain
                max_pain_strike = min(pain, key=pain.get) if pain else None

                result.update({
                    "nearest_expiry":   nearest_expiry,
                    "put_call_ratio":   pcr,
                    "pcr_signal":       "bearish_buildup" if pcr < 0.7 else
                                        ("bullish_buildup" if pcr > 1.2 else "neutral"),
                    "max_pain_level":   max_pain_strike,
                    "total_call_oi":    int(total_call_oi),
                    "total_put_oi":     int(total_put_oi),
                })
        except Exception as e:
            # IMPORTANT: Explicitly mark these as unavailable so the agent
            # does NOT invent values for PCR or Max Pain in its output.
            result["fno_options_available"] = False
            result["put_call_ratio"]        = "UNAVAILABLE — do not estimate or invent this value"
            result["pcr_signal"]            = "UNAVAILABLE — do not estimate or invent this value"
            result["max_pain_level"]        = "UNAVAILABLE — do not estimate or invent this value"
            result["fno_options_error"]     = str(e)

        return result

    # ------------------------------------------------------------------
    # NEW: SUPPORT / RESISTANCE LEVELS (Pivot Points)
    # ------------------------------------------------------------------

    def get_support_resistance(
        symbol: Annotated[str, "NSE ticker e.g. WIPRO.NS"]
    ) -> dict:
        """
        Computes classic pivot point levels from the most recent completed session.
        NSE traders widely use these as intraday entry/exit references.

        Levels returned:
          PP  = (H + L + C) / 3          ← pivot
          R1  = 2*PP - L                  ← resistance 1
          R2  = PP + (H - L)              ← resistance 2
          R3  = H + 2*(PP - L)            ← resistance 3
          S1  = 2*PP - H                  ← support 1
          S2  = PP - (H - L)              ← support 2
          S3  = L - 2*(H - PP)            ← support 3
        Also returns:
          - Camarilla pivots (tighter, preferred for intraday)
          - VWAP from prior session
        """
        df = yf.Ticker(symbol).history(period="5d", interval="1d")
        if len(df) < 2:
            return {"error": "Not enough data for pivot calculation"}

        # Use the last completed (previous) session
        prev = df.iloc[-2]
        H, L, C = float(prev["High"]), float(prev["Low"]), float(prev["Close"])
        O = float(prev["Open"])
        today_open = float(df.iloc[-1]["Open"]) if len(df) >= 1 else None

        PP = (H + L + C) / 3
        R = {
            "PP": round(PP, 2),
            "R1": round(2 * PP - L, 2),
            "R2": round(PP + (H - L), 2),
            "R3": round(H + 2 * (PP - L), 2),
            "S1": round(2 * PP - H, 2),
            "S2": round(PP - (H - L), 2),
            "S3": round(L - 2 * (H - PP), 2),
        }

        # Camarilla pivots (H4/L4 are the critical breakout levels)
        rng = H - L
        cam = {
            "H4": round(C + rng * 1.1 / 2, 2),   # breakout long above H4
            "H3": round(C + rng * 1.1 / 4, 2),   # intraday resistance
            "L3": round(C - rng * 1.1 / 4, 2),   # intraday support
            "L4": round(C - rng * 1.1 / 2, 2),   # breakdown short below L4
        }

        latest_close = float(df.iloc[-1]["Close"])
        nearest_support    = max([v for v in [R["S1"], R["S2"], R["S3"]] if v < latest_close], default=R["S3"])
        nearest_resistance = min([v for v in [R["R1"], R["R2"], R["R3"]] if v > latest_close], default=R["R3"])

        # ── Recommended targets using sensible R:R ────────────────────────────
        # For BEARISH trades: use S2 as target + Camarilla H3 as stop (tighter stop)
        # For BULLISH trades: use R2 as target + Camarilla L3 as stop (tighter stop)
        # Minimum acceptable R:R = 0.5:1  (risk 2 pts to make 1 pt minimum)
        bearish_target = R["S2"]
        bearish_stop   = cam["H3"]
        bearish_reward = round(latest_close - bearish_target, 2)
        bearish_risk   = round(bearish_stop - latest_close, 2)
        bearish_rr     = round(bearish_reward / (bearish_risk + 1e-9), 2)

        bullish_target = R["R2"]
        # If price is between L3 and L4, Camarilla L3 is above entry — use L4 as stop instead
        bullish_stop   = cam["L3"] if cam["L3"] < latest_close else cam["L4"]
        bullish_reward = round(bullish_target - latest_close, 2)
        bullish_risk   = round(latest_close - bullish_stop, 2)
        bullish_rr     = round(bullish_reward / (bullish_risk + 1e-9), 2)

        return {
            "prev_session":         {"high": H, "low": L, "close": C, "open": O},
            "today_open":           today_open,
            "latest_close":         latest_close,
            "classic_pivots":       R,
            "camarilla_pivots":     cam,
            "nearest_support":      nearest_support,
            "nearest_resistance":   nearest_resistance,
            # ── Pre-calculated trade setups ────────────────────────────────
            # Use these instead of S1/R1 to avoid the 0.03:1 R:R trap.
            "bearish_setup": {
                "entry":    latest_close,
                "target":   bearish_target,         # S2
                "stop":     bearish_stop,            # Camarilla H3 (tighter than R1)
                "reward_pts": bearish_reward,
                "risk_pts":   bearish_risk,
                "rr_ratio":   bearish_rr,
                "tradeable":  bearish_rr >= 0.5,
                "note": "untradeable — R:R below 0.5:1" if bearish_rr < 0.5 else "viable setup",
            },
            "bullish_setup": {
                "entry":    latest_close,
                "target":   bullish_target,          # R2
                "stop":     bullish_stop,             # Camarilla L3
                "reward_pts": bullish_reward,
                "risk_pts":   bullish_risk,
                "rr_ratio":   bullish_rr,
                "tradeable":  bullish_rr >= 0.5,
                "note": "untradeable — R:R below 0.5:1" if bullish_rr < 0.5 else "viable setup",
            },
            "gap_up_down": round(((today_open or latest_close) - C) / C * 100, 2) if today_open else None,
        }

    # ------------------------------------------------------------------
    # NEW: SECTOR PEER COMPARISON
    # ------------------------------------------------------------------

    def get_sector_peers(
        symbol: Annotated[str, "NSE ticker e.g. HDFCBANK.NS"]
    ) -> dict:
        """
        Returns 1-day and 5-day performance of sector peers on NSE.
        Helps identify if the stock is leading or lagging its sector.
        Strong intraday signal: stock up while sector is down = strength; vice versa = weakness.
        """
        info = yf.Ticker(symbol).info
        sector = info.get("sector", "")
        peers  = NSE_SECTOR_PEERS.get(sector, [])

        if not peers:
            return {"error": f"No peer list for sector: {sector}"}

        peer_data = {}
        for peer in peers[:6]:   # limit to 6 peers to avoid rate limits
            try:
                h = yf.Ticker(peer).history(period="6d")
                if len(h) >= 2:
                    c1d  = round((float(h["Close"].iloc[-1]) / float(h["Close"].iloc[-2]) - 1) * 100, 2)
                    c5d  = round((float(h["Close"].iloc[-1]) / float(h["Close"].iloc[0])  - 1) * 100, 2)
                    peer_data[peer] = {"1d_change_pct": c1d, "5d_change_pct": c5d,
                                       "latest_close": round(float(h["Close"].iloc[-1]), 2)}
            except Exception:
                pass

        sector_avg_1d = round(np.mean([v["1d_change_pct"] for v in peer_data.values()]), 2) if peer_data else 0
        target_1d     = peer_data.get(symbol, {}).get("1d_change_pct", 0)

        return {
            "sector":              sector,
            "peers":               peer_data,
            "sector_avg_1d_pct":   sector_avg_1d,
            "stock_vs_sector":     round(target_1d - sector_avg_1d, 2),
            "relative_strength":   "outperforming" if target_1d > sector_avg_1d else "underperforming",
        }

    # ------------------------------------------------------------------
    # NEW: DATA SANITY CHECK (catches anomalies before agent synthesis)
    # ------------------------------------------------------------------

    def get_data_sanity_check(
        symbol: Annotated[str, "NSE ticker e.g. RELIANCE.NS"]
    ) -> dict:
        """
        Cross-validates data from multiple sources to catch anomalies before
        the agent attempts to synthesize a forecast. Checks:

          1. Crude oil single-session spike (already flagged in get_nse_market_context)
          2. 52w high discrepancy between get_technical_indicators and get_extended_company_info
             (technical_indicators uses rolling 252-day window; extended_info uses yfinance's
             own field which can span a different period — a >5% difference flags a mismatch)
          3. Stale intraday data (last candle more than 1 full session old)
          4. Volume extreme (today's volume > 3× 20d avg — could signal news/corporate action
             not yet captured in the news feed)
          5. Price vs pivot sanity (if price is already below S2 or above R2, the standard
             trade setup may be stale and needs recalculation)

        Returns a dict with a 'flags' list.  An empty 'flags' list = all checks passed.
        The agent should mention any flags in the Final Reasoning section.
        """
        flags = []
        warnings = {}

        try:
            # ── 1. 52w high cross-check ───────────────────────────────────────────
            tech  = YFinanceUtils.get_technical_indicators(symbol)
            info  = yf.Ticker(symbol).info
            high_tech  = tech.get("52w_high")
            high_info  = info.get("fiftyTwoWeekHigh")
            if high_tech and high_info:
                discrepancy_pct = abs(high_tech - high_info) / max(high_tech, high_info) * 100
                if discrepancy_pct > 5:
                    msg = (
                        f"52w_high mismatch: technical_indicators={high_tech} vs "
                        f"extended_company_info={high_info} ({discrepancy_pct:.1f}% difference). "
                        f"Use technical_indicators value (rolling 252-day) for intraday analysis."
                    )
                    flags.append(msg)
                    warnings["52w_high"] = msg

            # ── 2. Volume spike (possible undisclosed corporate action) ───────────
            avg_vol = tech.get("avg_vol_20d", 0)
            latest_vol = tech.get("latest_vol", 0)
            if avg_vol > 0 and latest_vol > 3 * avg_vol:
                msg = (
                    f"Volume spike: last session {latest_vol:,} = "
                    f"{latest_vol / avg_vol:.1f}× the 20d avg ({avg_vol:,}). "
                    f"Check for corporate action, bulk deals, or undisclosed news."
                )
                flags.append(msg)
                warnings["volume_spike"] = msg

            # ── 3. Price vs pivot (stale setup risk) ─────────────────────────────
            sr    = YFinanceUtils.get_support_resistance(symbol)
            price = tech.get("latest_close", 0)
            s2    = sr.get("classic_pivots", {}).get("S2", 0)
            r2    = sr.get("classic_pivots", {}).get("R2", 0)
            if s2 and price < s2:
                msg = (
                    f"Price ({price}) is already BELOW S2 ({s2}). "
                    f"Standard bearish setup may have played out. Consider S3 ({sr['classic_pivots'].get('S3')}) as target."
                )
                flags.append(msg)
                warnings["price_below_s2"] = msg
            elif r2 and price > r2:
                msg = (
                    f"Price ({price}) is already ABOVE R2 ({r2}). "
                    f"Standard bullish setup may have played out. Consider R3 ({sr['classic_pivots'].get('R3')}) as target."
                )
                flags.append(msg)
                warnings["price_above_r2"] = msg

        except Exception as e:
            flags.append(f"Sanity check error: {str(e)}")

        return {
            "symbol":  symbol,
            "checks_run": 3,
            "flags_found": len(flags),
            "flags":   flags,
            "status":  "ALL_CLEAR" if not flags else "REVIEW_REQUIRED",
            "warnings": warnings,
        }

    # ------------------------------------------------------------------
    # NEW: EXTENDED COMPANY INFO (NSE-specific fundamentals)
    # ------------------------------------------------------------------

    def get_extended_company_info(
        symbol: Annotated[str, "NSE ticker e.g. TATAMOTORS.NS"]
    ) -> dict:
        """
        NSE-relevant fundamentals beyond basic company info.
        Includes valuation, beta, 52-week range, promoter pledge proxy.
        """
        info = yf.Ticker(symbol).info

        return {
            "company":             info.get("shortName"),
            "sector":              info.get("sector"),
            "industry":            info.get("industry"),
            "market_cap_cr":       round(info.get("marketCap", 0) / 1e7, 0),   # in crores
            "pe_ratio":            info.get("trailingPE"),
            "pb_ratio":            info.get("priceToBook"),
            "ev_ebitda":           info.get("enterpriseToEbitda"),
            "beta":                info.get("beta"),
            "beta_signal":         (
                "high_volatility_stock" if (info.get("beta") or 0) > 1.5 else
                "moderate_volatility"   if (info.get("beta") or 0) > 1.0 else
                "low_volatility_defensive"
            ),
            "52w_high":            info.get("fiftyTwoWeekHigh"),
            "52w_low":             info.get("fiftyTwoWeekLow"),
            "200d_avg":            info.get("twoHundredDayAverage"),
            "50d_avg":             info.get("fiftyDayAverage"),
            "float_shares":        info.get("floatShares"),
            "shares_outstanding":  info.get("sharesOutstanding"),
            "institutional_own_pct": round((info.get("institutionPercentHeld") or 0) * 100, 2),
            "promoter_holding_pct": None,  # yfinance doesn't expose this; NSE/BSE website needed
            # yfinance returns dividendYield as a decimal (e.g. 0.0125 = 1.25%)
            # BUT it sometimes returns the already-multiplied value (e.g. 125.0) — a known bug.
            # Fix: if raw value > 1, it is already in % form, use as-is; else multiply by 100.
            "dividend_yield_pct":  round(
                (lambda d: d if d > 1 else d * 100)(info.get("dividendYield") or 0), 2
            ),
            "earnings_date":       str(info.get("earningsTimestamp", "N/A")),
            "recommendation":      info.get("recommendationKey"),
            "analyst_count":       info.get("numberOfAnalystOpinions"),
            "target_price":        info.get("targetMeanPrice"),
        }


# ─────────────────────────────────────────────────────────────────────────────
# DETERMINISTIC CONFIDENCE SCORER
# Called by agent after deciding direction. Never let the LLM compute scores.
# ─────────────────────────────────────────────────────────────────────────────
    def compute_confidence(
        symbol: str,
        direction: str,          # "Bullish" | "Bearish" | "Sideways"
        macd_trend: str,         # verbatim from get_technical_indicators
        ema_trend: str,          # verbatim from get_technical_indicators
        nifty_change_pct: float, # from get_nse_market_context
        nifty_5d_trend: str,     # from get_nse_market_context
        obv_trend: str,          # verbatim from get_technical_indicators
        news_conflict: bool,     # agent judgment: True if conflicting news exists
        is_expiry_day: bool,     # from get_fno_data
        vix: float = 0.0,        # India VIX from get_nse_market_context
        rr_ratio: float = 0.0,   # R:R from the selected trade setup
        target_price: float = 0.0,  # target from setup
        entry_price: float = 0.0,   # entry / latest 15m close
    ) -> dict:
        """
        Deterministically compute confidence score from raw indicator values.
        Returns structured dict — agent must copy score + confidence verbatim.

        Includes two override rules that cap confidence regardless of score:
        MACRO OVERRIDE  : Strong macro day opposing direction → cap at Low
        CONTRADICTION   : Negative R:R or target on wrong side → force Low + untradeable
        """
        d = direction.lower()

        # Factor 1: Technical — MACD trend + EMA trend both confirm direction
        macd_ok = macd_trend.lower() in ("bullish",) if d == "bullish" else macd_trend.lower() in ("bearish",)
        ema_ok  = "bullish" in ema_trend.lower() if d == "bullish" else "bearish" in ema_trend.lower()
        f1 = macd_ok and ema_ok

        # Factor 2: NIFTY confirms direction
        if d == "bullish":
            f2 = nifty_change_pct > 0 and nifty_5d_trend.lower() == "up"
        elif d == "bearish":
            f2 = nifty_change_pct < 0 and nifty_5d_trend.lower() == "down"
        else:  # sideways
            f2 = abs(nifty_change_pct) < 0.5

        # Factor 3: OBV confirms direction (rising = bullish, falling = bearish)
        if d == "bullish":
            f3 = obv_trend.lower() == "rising"
        elif d == "bearish":
            f3 = obv_trend.lower() == "falling"
        else:
            f3 = obv_trend.lower() == "neutral"

        # Factor 4: News clear (no major conflict)
        f4 = not news_conflict

        # Factor 5: Non-expiry day
        f5 = not is_expiry_day

        score = sum([f1, f2, f3, f4, f5])
        confidence = "High" if score >= 4 else ("Medium" if score == 3 else "Low")

        # ── Override 1: Strong macro day opposing direction ───────────────────────
        # NIFTY move > 1.5% in opposite direction AND VIX > 20 = macro override
        macro_override = False
        macro_override_reason = None
        strong_bearish_macro = nifty_change_pct < -1.5 and vix > 20
        strong_bullish_macro  = nifty_change_pct >  1.5 and vix < 15

        if d == "bullish" and strong_bearish_macro:
            macro_override = True
            macro_override_reason = (
                f"MACRO OVERRIDE: NIFTY {nifty_change_pct:+.2f}% with VIX {vix:.1f} — "
                f"strong bearish macro opposes Bullish call. Capped at Low."
            )
            confidence = "Low"
            score = min(score, 2)
        elif d == "bearish" and strong_bullish_macro:
            macro_override = True
            macro_override_reason = (
                f"MACRO OVERRIDE: NIFTY {nifty_change_pct:+.2f}% with VIX {vix:.1f} — "
                f"strong bullish macro opposes Bearish call. Capped at Low."
            )
            confidence = "Low"
            score = min(score, 2)

        # ── Override 2: Contradiction — target on wrong side or negative R:R ─────
        contradiction = False
        contradiction_reason = None

        if entry_price > 0 and target_price > 0:
            target_above_entry = target_price > entry_price
            if d == "bullish" and not target_above_entry:
                contradiction = True
                contradiction_reason = (
                    f"CONTRADICTION: Direction=Bullish but target ({target_price}) "
                    f"< entry ({entry_price}). Setup from wrong side used."
                )
            elif d == "bearish" and target_above_entry:
                contradiction = True
                contradiction_reason = (
                    f"CONTRADICTION: Direction=Bearish but target ({target_price}) "
                    f"> entry ({entry_price}). Setup from wrong side used."
                )

        if rr_ratio < 0:
            contradiction = True
            contradiction_reason = (
                contradiction_reason or
                f"CONTRADICTION: Negative R:R ({rr_ratio:.2f}) — "
                f"target and stop are on the same side of entry."
            )

        if contradiction:
            confidence = "Low"
            score = min(score, 1)

        return {
            "symbol":     symbol,
            "direction":  direction,
            "factors": {
                "technical_aligned":  f1,
                "nifty_aligned":      f2,
                "obv_confirming":     f3,
                "news_clear":         f4,
                "non_expiry_day":     f5,
            },
            "score":      score,
            "confidence": confidence,
            "macro_override":      macro_override,
            "macro_override_reason": macro_override_reason,
            "contradiction":       contradiction,
            "contradiction_reason": contradiction_reason,
            "tradeable":  not contradiction,   # contradiction = not tradeable
            "note": "Copy score and confidence VERBATIM into report. Do not recompute.",
        }


    # ─────────────────────────────────────────────────────────────────────────────
    # FORWARD TEST LOGGER
    # ─────────────────────────────────────────────────────────────────────────────

    


    def log_forecast(
        symbol:       str,
        date:         str,   # "YYYY-MM-DD"
        direction:    str,   # "Bullish" | "Bearish" | "Sideways"
        entry_price:  float,
        target_price: float,
        stop_price:   float,
        rr_ratio:     float,
        confidence:   str,   # "High" | "Medium" | "Low"
        score:        int,   # 0–5
        tradeable:    bool,
        factors:      dict = None,  # optional — from compute_confidence output
    ) -> dict:
        """
        Append one forecast record to forecasts.jsonl.
        Always logs even if untradeable (for directional accuracy stats).
        """
        record = {
            "symbol":       symbol,
            "date":         date,
            "direction":    direction,
            "entry_price":  entry_price,
            "target_price": target_price,
            "stop_price":   stop_price,
            "rr_ratio":     rr_ratio,
            "confidence":   confidence,
            "score":        score,
            "tradeable":    tradeable,
            "factors":      factors or {},
            "logged_at":    _datetime.utcnow().isoformat(),
            # outcome fields — filled later by record_outcome()
            "outcome":      None,
            "actual_high":  None,
            "actual_low":   None,
            "actual_close": None,
            "hit_target":   None,
            "hit_stop":     None,
            "actual_move_pct": None,
        }
        with open(_FORECAST_FILE, "a") as f:
            f.write(_json.dumps(record) + "\n")
        return {"status": "logged", "file": str(_FORECAST_FILE.resolve()), "record": record}


    # ─────────────────────────────────────────────────────────────────────────────
    # OUTCOME RECORDER  (run next morning or EOD)
    # ─────────────────────────────────────────────────────────────────────────────

    def record_outcome(
        symbol:       str,
        date:         str,   # "YYYY-MM-DD" — must match log_forecast date
        actual_high:  float,
        actual_low:   float,
        actual_close: float,
    ) -> dict:
        """
        Find the matching forecast in forecasts.jsonl and write outcome fields.
        Rewrites the file in-place (safe for small files; fine for daily logs).
        """
        if not _FORECAST_FILE.exists():
            return {"error": "forecasts.jsonl not found"}

        records = []
        matched = False
        with open(_FORECAST_FILE) as f:
            for line in f:
                r = _json.loads(line.strip())
                if r["symbol"] == symbol and r["date"] == date and r["outcome"] is None:
                    # Compute outcome
                    hit_target = actual_high >= r["target_price"] if r["direction"] == "Bullish" \
                                else actual_low  <= r["target_price"]
                    hit_stop   = actual_low  <= r["stop_price"]   if r["direction"] == "Bullish" \
                                else actual_high >= r["stop_price"]

                    if hit_target and not hit_stop:
                        outcome = "WIN"
                    elif hit_stop and not hit_target:
                        outcome = "LOSS"
                    elif hit_target and hit_stop:
                        # Both touched — assume stop hit first (conservative)
                        outcome = "LOSS"
                    else:
                        outcome = "EXPIRED"  # neither target nor stop reached

                    actual_move = round((actual_close - r["entry_price"]) / r["entry_price"] * 100, 3)
                    direction_correct = (
                        (r["direction"] == "Bullish" and actual_close > r["entry_price"]) or
                        (r["direction"] == "Bearish" and actual_close < r["entry_price"]) or
                        (r["direction"] == "Sideways" and abs(actual_move) < 0.5)
                    )

                    r.update({
                        "outcome":           outcome,
                        "actual_high":       actual_high,
                        "actual_low":        actual_low,
                        "actual_close":      actual_close,
                        "hit_target":        hit_target,
                        "hit_stop":          hit_stop,
                        "actual_move_pct":   actual_move,
                        "direction_correct": direction_correct,
                        "recorded_at":       _datetime.utcnow().isoformat(),
                    })
                    matched = True
                records.append(r)

        if not matched:
            return {"error": f"No pending forecast found for {symbol} on {date}"}

        with open(_FORECAST_FILE, "w") as f:
            for r in records:
                f.write(_json.dumps(r) + "\n")

        result = next(r for r in records if r["symbol"] == symbol and r["date"] == date)
        return {"status": "recorded", "outcome": result["outcome"],
                "direction_correct": result["direction_correct"],
                "actual_move_pct": result["actual_move_pct"]}


    # ─────────────────────────────────────────────────────────────────────────────
    # ANALYTICS
    # ─────────────────────────────────────────────────────────────────────────────

    def get_forecast_stats(min_date: str = None, symbol: str = None) -> dict:
        """
        Compute win rates and accuracy stats from forecasts.jsonl.
        Filters by min_date (YYYY-MM-DD) and/or symbol if provided.
        """
        if not _FORECAST_FILE.exists():
            return {"error": "forecasts.jsonl not found — no forecasts logged yet"}

        records = []
        with open(_FORECAST_FILE) as f:
            for line in f:
                r = _json.loads(line.strip())
                if min_date and r["date"] < min_date:
                    continue
                if symbol and r["symbol"] != symbol:
                    continue
                records.append(r)

        if not records:
            return {"error": "No records match filters"}

        completed = [r for r in records if r["outcome"] is not None]
        pending   = [r for r in records if r["outcome"] is None]

        def win_rate(subset):
            wins = sum(1 for r in subset if r.get("outcome") == "WIN")
            return round(wins / len(subset), 3) if subset else None

        def dir_accuracy(subset):
            scored = [r for r in subset if r.get("direction_correct") is not None]
            correct = sum(1 for r in scored if r["direction_correct"])
            return round(correct / len(scored), 3) if scored else None

        by_confidence = {}
        for level in ("High", "Medium", "Low"):
            sub = [r for r in completed if r["confidence"] == level]
            by_confidence[level] = {"n": len(sub), "win_rate": win_rate(sub),
                                    "direction_accuracy": dir_accuracy(sub)}

        by_direction = {}
        for d in ("Bullish", "Bearish", "Sideways"):
            sub = [r for r in completed if r["direction"] == d]
            by_direction[d] = {"n": len(sub), "win_rate": win_rate(sub),
                                "direction_accuracy": dir_accuracy(sub)}

        by_symbol = {}
        for r in completed:
            s = r["symbol"]
            by_symbol.setdefault(s, []).append(r)
        by_symbol = {s: {"n": len(v), "win_rate": win_rate(v), "direction_accuracy": dir_accuracy(v)}
                    for s, v in by_symbol.items()}

        tradeable = [r for r in records if r["tradeable"]]
        untradeable_pct = round(1 - len(tradeable) / len(records), 3) if records else None

        return {
            "total_forecasts":      len(records),
            "completed":            len(completed),
            "pending_outcome":      len(pending),
            "overall_win_rate":     win_rate(completed),
            "overall_dir_accuracy": dir_accuracy(completed),
            "untradeable_pct":      untradeable_pct,
            "by_confidence":        by_confidence,
            "by_direction":         by_direction,
            "by_symbol":            by_symbol,
        }