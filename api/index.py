from __future__ import annotations

from dataclasses import dataclass
from html import escape
from http.server import BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional
from urllib.parse import parse_qs, urlparse
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "app"
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

from indicators.features import add_indicators  # noqa: E402
from strategy.adaptive_strategy import AdaptiveFuturesStrategy, AdaptiveSpotStrategy  # noqa: E402
from strategy.simple_strategy import SimpleStrategy  # noqa: E402


@dataclass
class SimTrade:
    side: str
    entry: float
    exit: float
    sl: float
    tp: float
    qty: float
    pnl: float


def load_ohlcv(dataset: str) -> pd.DataFrame:
    path = ROOT / ("app/past_data/futures_3y.csv" if dataset == "futures" else "app/past_data/spot_3y.csv")
    df = pd.read_csv(path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.set_index("open_time")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    return add_indicators(df)


def build_strategy(name: str):
    if name == "simple":
        return SimpleStrategy()
    if name == "adaptive_futures":
        return AdaptiveFuturesStrategy()
    return AdaptiveSpotStrategy()


def resolve_exit(row: pd.Series, side: str, sl: float, tp: float) -> Optional[float]:
    low, high = float(row["low"]), float(row["high"])
    if side == "long":
        if low <= sl and high >= tp:
            return sl
        if low <= sl:
            return sl
        if high >= tp:
            return tp
        return None
    if high >= sl and low <= tp:
        return sl
    if high >= sl:
        return sl
    if low <= tp:
        return tp
    return None


def toy_backtest(df: pd.DataFrame, strategy, lookback: int, risk: float, max_bars: int, balance: float):
    trades: list[SimTrade] = []
    i = max(lookback, 1)
    while i < len(df) - 1:
        signal = strategy.test_evaluate(df, i)
        if not signal or signal.get("signal") not in ("long", "short"):
            i += 1
            continue

        side = signal["signal"]
        sl = float(signal["sl"])
        tp = float(signal["tp"])
        entry_i = i + 1
        entry = float(df.iloc[entry_i]["open"])

        if (side == "long" and not (sl < entry < tp)) or (side == "short" and not (tp < entry < sl)):
            i += 1
            continue

        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            i += 1
            continue

        qty = (balance * risk) / risk_per_unit
        exit_price = float(df.iloc[entry_i]["close"])
        exit_i = entry_i

        for j in range(entry_i + 1, len(df)):
            maybe_exit = resolve_exit(df.iloc[j], side, sl, tp)
            if maybe_exit is not None:
                exit_price, exit_i = maybe_exit, j
                break
            if (j - entry_i) >= max_bars:
                exit_price, exit_i = float(df.iloc[j]["close"]), j
                break

        pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
        balance += pnl
        trades.append(SimTrade(side, entry, exit_price, sl, tp, qty, pnl))
        i = exit_i + 1

    return trades, balance


def render_page(params: dict[str, str]) -> str:
    dataset = params.get("dataset", "futures")
    strategy_name = params.get("strategy", "simple")
    lookback = int(params.get("lookback", "250"))
    risk = float(params.get("risk", "0.02"))
    max_bars = int(params.get("max_bars", "20"))
    start_balance = float(params.get("start_balance", "1000"))
    inspect_index = int(params.get("inspect_index", "205"))
    action = params.get("action", "signal")

    result_html = ""
    try:
        df = load_ohlcv(dataset)
        strategy = build_strategy(strategy_name)
        inspect_index = max(205, min(inspect_index, len(df) - 1))

        if action == "backtest":
            trades, final_balance = toy_backtest(df, strategy, lookback, risk, max_bars, start_balance)
            wins = sum(1 for t in trades if t.pnl > 0)
            win_rate = (wins / len(trades) * 100) if trades else 0.0
            result_html = (
                f"<h3>Backtest Result</h3>"
                f"<p>Trades: {len(trades)} | Final Balance: {final_balance:.2f} | "
                f"PnL: {final_balance - start_balance:.2f} | Win Rate: {win_rate:.2f}%</p>"
            )
        else:
            signal = strategy.test_evaluate(df, inspect_index)
            result_html = f"<h3>Signal @ bar {inspect_index}</h3><pre>{escape(str(signal))}</pre>"
    except Exception as exc:
        result_html = f"<h3>Error</h3><pre>{escape(str(exc))}</pre>"

    selected = lambda x, y: "selected" if x == y else ""
    return f"""
<html><body style='font-family: sans-serif; max-width: 900px; margin: 2rem auto;'>
<h2>Trading Bot Web Tester</h2>
<form method='get' action='/api'>
  <label>Dataset:</label>
  <select name='dataset'>
    <option value='futures' {selected(dataset, 'futures')}>futures</option>
    <option value='spot' {selected(dataset, 'spot')}>spot</option>
  </select>
  <label>Strategy:</label>
  <select name='strategy'>
    <option value='simple' {selected(strategy_name, 'simple')}>simple</option>
    <option value='adaptive_futures' {selected(strategy_name, 'adaptive_futures')}>adaptive_futures</option>
    <option value='adaptive_spot' {selected(strategy_name, 'adaptive_spot')}>adaptive_spot</option>
  </select><br/><br/>
  <label>Lookback:</label><input name='lookback' value='{lookback}' />
  <label>Risk:</label><input name='risk' value='{risk}' />
  <label>Max bars:</label><input name='max_bars' value='{max_bars}' />
  <label>Start balance:</label><input name='start_balance' value='{start_balance}' />
  <label>Inspect index:</label><input name='inspect_index' value='{inspect_index}' /><br/><br/>
  <button type='submit' name='action' value='signal'>Run Signal</button>
  <button type='submit' name='action' value='backtest'>Run Backtest</button>
</form>
{result_html}
<p><a href='/health'>health</a></p>
</body></html>
"""


class handler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path in ("/health", "/api/health"):
            self.send_response(200)
            self.send_header("Content-type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"ok")
            return

        if parsed.path in ("/", "/api", "/api/"):
            query = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            page = render_page(query).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(page)
            return

        self.send_response(404)
        self.send_header("Content-type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write(b"not found")
