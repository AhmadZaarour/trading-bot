from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import streamlit as st

from indicators.features import add_indicators
from strategy.adaptive_strategy import AdaptiveFuturesStrategy, AdaptiveSpotStrategy
from strategy.simple_strategy import SimpleStrategy


@dataclass
class SimTrade:
    side: str
    entry_i: int
    exit_i: int
    entry: float
    exit: float
    sl: float
    tp: float
    qty: float
    pnl: float


def load_ohlcv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "open_time" in df.columns:
        df["open_time"] = pd.to_datetime(df["open_time"])
        df = df.set_index("open_time")

    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)

    return add_indicators(df)


def build_strategy(name: str):
    if name == "SimpleStrategy":
        return SimpleStrategy()
    if name == "AdaptiveFuturesStrategy":
        return AdaptiveFuturesStrategy()
    return AdaptiveSpotStrategy()


def resolve_exit(row: pd.Series, side: str, sl: float, tp: float) -> Optional[float]:
    low = float(row["low"])
    high = float(row["high"])

    if side == "long":
        if low <= sl and high >= tp:
            return sl  # conservative
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


def run_toy_backtest(
    df: pd.DataFrame,
    strategy,
    lookback: int,
    risk_per_trade: float,
    max_bars: int,
    initial_balance: float,
) -> Dict[str, object]:
    balance = float(initial_balance)
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

        if side == "long" and not (sl < entry < tp):
            i += 1
            continue
        if side == "short" and not (tp < entry < sl):
            i += 1
            continue

        risk_per_unit = abs(entry - sl)
        if risk_per_unit <= 0:
            i += 1
            continue

        qty = (balance * risk_per_trade) / risk_per_unit
        bars_open = 0
        exit_price = float(df.iloc[entry_i]["close"])
        exit_i = entry_i

        j = entry_i + 1
        while j < len(df):
            bars_open += 1
            row = df.iloc[j]
            maybe_exit = resolve_exit(row, side, sl, tp)
            if maybe_exit is not None:
                exit_price = maybe_exit
                exit_i = j
                break
            if bars_open >= max_bars:
                exit_price = float(row["close"])
                exit_i = j
                break
            j += 1

        pnl = (exit_price - entry) * qty if side == "long" else (entry - exit_price) * qty
        balance += pnl
        trades.append(
            SimTrade(
                side=side,
                entry_i=entry_i,
                exit_i=exit_i,
                entry=entry,
                exit=exit_price,
                sl=sl,
                tp=tp,
                qty=qty,
                pnl=pnl,
            )
        )

        i = exit_i + 1

    win_rate = 0.0
    if trades:
        win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) * 100

    return {
        "trades": trades,
        "final_balance": balance,
        "pnl": balance - initial_balance,
        "win_rate": win_rate,
    }


def main() -> None:
    st.set_page_config(page_title="Trading Bot Sandbox", layout="wide")
    st.title("Trading Bot Sandbox")
    st.caption("Quick UI to inspect signals and run a lightweight local backtest.")

    data_map = {
        "Futures CSV": Path("app/past_data/futures_3y.csv"),
        "Spot CSV": Path("app/past_data/spot_3y.csv"),
    }

    with st.sidebar:
        st.header("Controls")
        data_choice = st.selectbox("Dataset", list(data_map.keys()))
        strategy_name = st.selectbox(
            "Strategy",
            ["SimpleStrategy", "AdaptiveFuturesStrategy", "AdaptiveSpotStrategy"],
        )
        lookback = st.number_input("Lookback bars", min_value=50, max_value=400, value=250)
        risk = st.slider("Risk per trade", min_value=0.001, max_value=0.05, value=0.02, step=0.001)
        max_bars = st.number_input("Max bars per trade", min_value=1, max_value=100, value=20)
        initial_balance = st.number_input("Initial balance (USDT)", min_value=100.0, value=1000.0)

    df = load_ohlcv(data_map[data_choice])
    strategy = build_strategy(strategy_name)

    st.subheader("Price Snapshot")
    st.line_chart(df[["close", "ema_20", "ema_50", "ema_200"]].tail(300))

    idx = st.slider("Inspect bar index", min_value=205, max_value=len(df) - 1, value=len(df) - 1)
    signal = strategy.test_evaluate(df, idx)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Signal Output")
        st.json(signal)
    with c2:
        st.markdown("### Candle")
        st.dataframe(df.iloc[[idx]][["open", "high", "low", "close", "volume", "rsi", "atr", "adx"]])

    if st.button("Run toy backtest"):
        result = run_toy_backtest(df, strategy, int(lookback), float(risk), int(max_bars), float(initial_balance))
        st.markdown("### Backtest Summary")
        st.write(
            {
                "trades": len(result["trades"]),
                "final_balance": round(result["final_balance"], 2),
                "pnl": round(result["pnl"], 2),
                "win_rate": round(result["win_rate"], 2),
            }
        )

        if result["trades"]:
            trades_df = pd.DataFrame([t.__dict__ for t in result["trades"]])
            st.dataframe(trades_df.tail(25))


if __name__ == "__main__":
    main()
