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
            pnl = final_balance - start_balance
            pnl_color = "#67f9b3" if pnl >= 0 else "#ff6b9a"
            result_html = (
                "<div class='card'>"
                "<h3>Backtest Result</h3>"
                "<div class='metrics'>"
                f"<div class='metric'><span>Trades</span><strong>{len(trades)}</strong></div>"
                f"<div class='metric'><span>Final Balance</span><strong>{final_balance:.2f}</strong></div>"
                f"<div class='metric'><span>PnL</span><strong style='color:{pnl_color}'>{pnl:.2f}</strong></div>"
                f"<div class='metric'><span>Win Rate</span><strong>{win_rate:.2f}%</strong></div>"
                "</div>"
                "</div>"
            )
        else:
            signal = strategy.test_evaluate(df, inspect_index)
            result_html = (
                "<div class='card'>"
                f"<h3>Signal @ bar {inspect_index}</h3>"
                f"<pre class='signal-box'>{escape(str(signal))}</pre>"
                "</div>"
            )
    except Exception as exc:
        result_html = (
            "<div class='card error'>"
            "<h3>Error</h3>"
            f"<pre class='signal-box'>{escape(str(exc))}</pre>"
            "</div>"
        )

    selected = lambda x, y: "selected" if x == y else ""
    return f"""
<html>
<head>
  <meta name='viewport' content='width=device-width, initial-scale=1' />
  <style>
    :root {{
      --bg:#070b17;
      --panel:#11182b;
      --panel2:#0f1424;
      --text:#e8eeff;
      --muted:#9db0d8;
      --accent:#57d5ff;
      --accent2:#8a7dff;
      --border:#1f2d4a;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin:0; font-family: Inter, system-ui, -apple-system, Segoe UI, sans-serif;
      color:var(--text);
      background: radial-gradient(1200px 600px at 10% -10%, #1a2f63 0%, transparent 55%),
                  radial-gradient(1000px 600px at 100% 0%, #2b175b 0%, transparent 50%),
                  var(--bg);
    }}
    .wrap {{ max-width: 1100px; margin: 2rem auto; padding: 0 1rem; }}
    .hero {{
      background: linear-gradient(120deg, rgba(87,213,255,.14), rgba(138,125,255,.12));
      border:1px solid var(--border); border-radius:16px; padding:1rem 1.25rem; margin-bottom:1rem;
      box-shadow: 0 10px 30px rgba(0,0,0,.35);
    }}
    h2 {{ margin:.2rem 0; }}
    .sub {{ color: var(--muted); font-size:.95rem; }}
    .grid {{ display:grid; grid-template-columns: repeat(12, 1fr); gap: 12px; }}
    .card {{
      background: linear-gradient(180deg, var(--panel), var(--panel2));
      border: 1px solid var(--border); border-radius: 14px; padding: 1rem;
      box-shadow: 0 8px 24px rgba(0,0,0,.28);
    }}
    .form-card {{ grid-column: span 12; }}
    .result-card {{ grid-column: span 12; }}
    @media (min-width: 900px) {{
      .form-card {{ grid-column: span 7; }}
      .result-card {{ grid-column: span 5; }}
    }}
    label {{ font-size: .85rem; color: var(--muted); display:block; margin-bottom:.35rem; }}
    input, select {{
      width:100%; background:#0a1020; color:var(--text); border:1px solid var(--border);
      border-radius:10px; padding:.62rem .7rem; outline:none;
    }}
    .fields {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; }}
    .fields-3 {{ display:grid; grid-template-columns: repeat(3, 1fr); gap:10px; margin-top:10px; }}
    .btns {{ margin-top:12px; display:flex; gap:10px; }}
    button {{
      border:0; border-radius:10px; padding:.7rem 1rem; font-weight:600; cursor:pointer;
      color:#05101f; background:linear-gradient(90deg, var(--accent), #8cf8ff);
    }}
    button.secondary {{
      color:#fff; background:linear-gradient(90deg, var(--accent2), #a289ff);
    }}
    .metrics {{ display:grid; grid-template-columns: repeat(2, 1fr); gap:10px; }}
    .metric {{ background:#0a1020; border:1px solid var(--border); border-radius:10px; padding:.65rem; }}
    .metric span {{ display:block; color:var(--muted); font-size:.8rem; }}
    .metric strong {{ font-size:1.05rem; }}
    .signal-box {{
      margin:0; white-space:pre-wrap; background:#0a1020; border:1px solid var(--border);
      border-radius:10px; padding:.8rem; color:#c2d8ff;
    }}
    .footer {{ margin-top: 1rem; color:var(--muted); font-size:.85rem; }}
    .footer a {{ color: var(--accent); text-decoration:none; }}
    .error {{ border-color:#5a2644; }}
  </style>
</head>
<body>
  <div class='wrap'>
    <div class='hero'>
      <h2>Trading Bot — Quantum Console</h2>
      <div class='sub'>Futuristic web tester for signal inspection and quick strategy sanity checks.</div>
    </div>
    <div class='grid'>
      <div class='card form-card'>
        <form method='get' action='/api'>
          <div class='fields'>
            <div>
              <label>Dataset</label>
              <select name='dataset'>
                <option value='futures' {selected(dataset, 'futures')}>futures</option>
                <option value='spot' {selected(dataset, 'spot')}>spot</option>
              </select>
            </div>
            <div>
              <label>Strategy</label>
              <select name='strategy'>
                <option value='simple' {selected(strategy_name, 'simple')}>simple</option>
                <option value='adaptive_futures' {selected(strategy_name, 'adaptive_futures')}>adaptive_futures</option>
                <option value='adaptive_spot' {selected(strategy_name, 'adaptive_spot')}>adaptive_spot</option>
              </select>
            </div>
          </div>
          <div class='fields-3'>
            <div><label>Lookback</label><input name='lookback' value='{lookback}' /></div>
            <div><label>Risk</label><input name='risk' value='{risk}' /></div>
            <div><label>Max bars</label><input name='max_bars' value='{max_bars}' /></div>
          </div>
          <div class='fields-3'>
            <div><label>Start balance</label><input name='start_balance' value='{start_balance}' /></div>
            <div><label>Inspect index</label><input name='inspect_index' value='{inspect_index}' /></div>
            <div class='btns'>
              <button type='submit' name='action' value='signal'>Run Signal</button>
              <button class='secondary' type='submit' name='action' value='backtest'>Run Backtest</button>
            </div>
          </div>
        </form>
      </div>
      <div class='result-card'>{result_html}</div>
    </div>
    <div class='footer'>Health check: <a href='/health'>/health</a></div>
  </div>
</body>
</html>
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
