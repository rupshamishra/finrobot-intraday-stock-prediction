import yfinance as yf
import pandas as pd

TICKERS = ["MSFT", "AAPL", "GOOGL", "NVDA", "TSLA"]

results = []

for ticker in TICKERS:
    df = yf.download(
        ticker,
        interval="1h",
        period="7d",
        progress=False,
    )

    if df.empty or len(df) < 3:
        print(f"{ticker}: insufficient data")
        continue

    df = df.sort_index()

    # Use second-last candle as prediction point
    pred_idx = df.index[-2]
    next_idx = df.index[-1]

    # FORCE scalar floats (THIS IS THE KEY FIX)
    price_before = float(df.loc[pred_idx, "Close"])
    price_after = float(df.loc[next_idx, "Close"])

    return_pct = (price_after - price_before) / price_before

    if return_pct > 0.001:
        actual_move = "UP"
    elif return_pct < -0.001:
        actual_move = "DOWN"
    else:
        actual_move = "NEUTRAL"

    results.append({
        "ticker": ticker,
        "prediction_time": pred_idx.strftime("%Y-%m-%d %H:%M"),
        "price_before": round(price_before, 2),
        "price_after": round(price_after, 2),
        "return_pct_%": round(return_pct * 100, 3),
        "actual_move": actual_move,
    })

df_results = pd.DataFrame(results)

print("\nBACKTEST RESULTS")
print(df_results)

df_results.to_csv("hourly_backtest_results.csv", index=False)
print("\nSaved to hourly_backtest_results.csv")
