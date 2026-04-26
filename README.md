# trading-bot

A Python crypto trading bot that supports:
- **Futures live trading loop** (Binance futures)
- **Spot live trading loop**
- **Futures backtesting**
- **Spot backtesting**

Main entrypoint: `app/main.py`.

## What the code currently does

- Pulls OHLCV candles from Binance via `python-binance` data providers.
- Computes indicators (RSI/EMA/ATR and pattern helpers) in `indicators/features.py`.
- Uses strategy classes (default in `main.py` is `SimpleStrategy` for futures and `AdaptiveSpotStrategy` for spot) to generate trade signals.
- Sizes positions with `RiskManager` using SL distance, leverage cap, and fee/slippage buffers.
- Places market orders and exit orders through broker wrappers.
- Runs backtests with next-bar execution, slippage/fee modeling, and equity curve statistics.

## Minimal web sandbox UI

A tiny Flask page is provided in `app/app.py` for quick manual testing.

### Run it

```bash
cd app
python app.py
```

Then open `http://localhost:5000`.

### UI actions

- **Check Latest Signal**: fetches latest candles and displays the current strategy decision.
- **Run Backtest Summary**: runs a short backtest simulation and displays a compact JSON summary.

## Install

```bash
pip install -e .
```

## Notes

- Configure API credentials with environment variables (`API_KEY`, `API_SECRET`, and testnet keys).
- `main.py` live modes run infinite loops by design.
- There is duplicated source under both `app/` and `crypto_trading_agent/`; keep them synchronized or consolidate into one package.
