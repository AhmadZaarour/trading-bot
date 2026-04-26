# trading-bot

Crypto trading bot with:
- **Live engines** for Binance futures and spot trading.
- **Backtesting** for futures and spot strategies.
- Technical-indicator feature generation (EMA/RSI/MACD/ATR/ADX/Bollinger/volume).
- Multiple strategies (`SimpleStrategy`, `AdaptiveFuturesStrategy`, `AdaptiveSpotStrategy`).

## Run the CLI bot

```bash
cd /workspace/trading-bot
python app/main.py
```

You will be prompted to choose futures/spot and live/backtest mode.

## Run the local UI sandbox

A small Streamlit interface is available to test signals and run a lightweight local backtest against the CSV history files.

```bash
cd /workspace/trading-bot
streamlit run app/interface.py
```

The UI lets you:
- choose futures/spot CSV data,
- switch between built-in strategies,
- inspect a specific bar and its generated signal,
- run a toy risk-based backtest and inspect recent trades.
