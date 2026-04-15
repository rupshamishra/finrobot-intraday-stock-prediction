import autogen

TICKERS = ["MSFT", "AAPL", "GOOGL", "NVDA", "TSLA"]

llm_config = {
    "config_list": autogen.config_list_from_json("OAI_CONFIG_LIST"),
    "temperature": 0,
}

agent = autogen.AssistantAgent(
    name="SignalAgent",
    llm_config=llm_config,
    system_message=(
        "You are a financial forecasting agent.\n"
        "You must NOT use real-time data, browsing, prices, or news.\n"
        "You must rely ONLY on general market knowledge and reasoning.\n"
        "You must NOT write explanations or code.\n"
        "You must ONLY output in this exact format:\n\n"
        "Signal: UP/DOWN/NEUTRAL\n"
        "Confidence: <number between 0 and 1>"
    ),
)

PREDICTION_TIME = "2026-01-02 10:00"

for ticker in TICKERS:
    reply = agent.generate_reply(
        messages=[
            {
                "role": "user",
                "content": (
                    f"Assume the date is {PREDICTION_TIME}. "
                    f"Without using any real market data, "
                    f"predict the next 1-hour movement direction for {ticker}."
                ),
            }
        ]
    )

    print(f"\n{ticker} @ {PREDICTION_TIME}")
    print(reply)
