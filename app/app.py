from __future__ import annotations

from typing import Any

import pandas as pd
import yaml
from flask import Flask, render_template_string, request

from backtest.runner import Backtester
from engine.broker import BinanceFuturesBroker
from engine.data import BinanceDataProvider
from engine.risk import RiskManager
from indicators.features import add_indicators
from strategy.simple_strategy import SimpleStrategy


app = Flask(__name__)

PAGE = """
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Trading Bot Sandbox</title>
    <style>
      body { font-family: Arial, sans-serif; margin: 2rem auto; max-width: 960px; line-height: 1.4; }
      .card { border: 1px solid #ddd; border-radius: 8px; padding: 1rem; margin-bottom: 1rem; }
      button { padding: 0.5rem 0.9rem; margin-right: 0.5rem; }
      pre { background: #111; color: #eee; padding: 0.8rem; overflow-x: auto; border-radius: 6px; }
      table { border-collapse: collapse; width: 100%; }
      th, td { border: 1px solid #ccc; padding: 0.4rem; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
      .muted { color: #555; font-size: 0.95rem; }
    </style>
  </head>
  <body>
    <h1>Trading Bot Sandbox</h1>
    <p class="muted">Quickly inspect the latest strategy signal or run a lightweight backtest summary without launching the infinite live loop.</p>

    <div class="card">
      <h2>Config</h2>
      <pre>{{ config_text }}</pre>
      <form method="post">
        <button name="action" value="signal">Check Latest Signal</button>
        <button name="action" value="backtest">Run Backtest Summary</button>
      </form>
    </div>

    {% if signal %}
    <div class="card">
      <h2>Latest Signal</h2>
      <pre>{{ signal }}</pre>
      {% if candles_html %}
      <h3>Recent Candles (tail)</h3>
      {{ candles_html|safe }}
      {% endif %}
    </div>
    {% endif %}

    {% if report %}
    <div class="card">
      <h2>Backtest Summary</h2>
      <pre>{{ report }}</pre>
    </div>
    {% endif %}

    {% if error %}
    <div class="card">
      <h2>Error</h2>
      <pre>{{ error }}</pre>
    </div>
    {% endif %}
  </body>
</html>
"""


def _load_config() -> dict[str, Any]:
    with open("config/default.yaml", "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def _create_runtime(config: dict[str, Any]):
    symbol = config["SYMBOL"][0] if isinstance(config["SYMBOL"], list) else config["SYMBOL"]
    strategy = SimpleStrategy()
    broker = BinanceFuturesBroker(testnet=True)
    data = BinanceDataProvider(testnet=True)
    risk = RiskManager(broker)
    return symbol, strategy, broker, data, risk


def _signal_snapshot(config: dict[str, Any]) -> tuple[dict[str, Any], str]:
    symbol, strategy, _, data, _ = _create_runtime(config)
    df = data.latest_df(symbol=symbol, interval=config["INTERVAL"], limit=config["LOOKBACK"])
    df = add_indicators(df)
    signal = strategy.evaluate(df)

    tail_cols = ["open", "high", "low", "close", "volume", "rsi", "ema_20", "atr"]
    tail = df[tail_cols].tail(6).round(4)
    return signal, tail.to_html(classes="candles", border=0)


def _run_backtest_summary(config: dict[str, Any]) -> dict[str, Any]:
    symbol, strategy, broker, data, risk = _create_runtime(config)
    config_copy = dict(config)
    config_copy["SYMBOL"] = symbol

    backtest = Backtester(risk, broker, strategy, data, config_copy, initial_balance=1000)
    trades, equity_curve = backtest.simulate()

    if not trades:
        return {"trades": 0, "message": "No trades were produced for the selected lookback."}

    trades_df = pd.DataFrame(trades)
    wins = int((trades_df["pnl"] > 0).sum())
    total = len(trades_df)
    final_balance = float(equity_curve[-1]) if equity_curve else 1000.0

    return {
        "trades": total,
        "wins": wins,
        "win_rate_pct": round((wins / total) * 100, 2),
        "total_pnl": round(float(trades_df["pnl"].sum()), 2),
        "avg_pnl": round(float(trades_df["pnl"].mean()), 2),
        "final_balance": round(final_balance, 2),
        "sample_trade": trades[0],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    config = _load_config()
    payload = {
        "config_text": yaml.safe_dump(config, sort_keys=False),
        "signal": None,
        "candles_html": None,
        "report": None,
        "error": None,
    }

    if request.method == "POST":
        action = request.form.get("action")
        try:
            if action == "signal":
                signal, candles_html = _signal_snapshot(config)
                payload["signal"] = signal
                payload["candles_html"] = candles_html
            elif action == "backtest":
                payload["report"] = _run_backtest_summary(config)
        except Exception as exc:
            payload["error"] = str(exc)

    return render_template_string(PAGE, **payload)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
