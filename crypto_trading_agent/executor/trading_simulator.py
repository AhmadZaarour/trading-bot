import pandas as pd
from typing import Callable, List, Dict, Any
import numpy as np

def bars_needed_to_reach(entry, tp, atr, pad=1.5):
    """
    Roughly how many bars you need for price to traverse the target distance,
    assuming ~ATR per bar on average; pad>1 adds slack.
    """
    if atr is None or atr <= 0 or tp is None:
        return 0
    dist = abs(tp - entry)
    return int(np.ceil(pad * (dist / atr)))

def simulate_trades(
    df: pd.DataFrame,
    strategy_func: Callable,
    volume_ma_period: int = 20,
    slippage: float = 0.0005,   # 0.05% slippage
    allow_overlap: bool = False
) -> pd.DataFrame:
    """
    Backtest trade simulation with realistic rules.
    
    Parameters:
        df (pd.DataFrame): Must contain ['open','high','low','close','volume'].
        strategy_func (Callable): function(prev2, prev1, curr, volume_ma, idx, df) -> dict or None
        volume_ma_period (int): Lookback for volume filter.
        lookahead (int): Candles to look ahead for SL/TP hit.
        slippage (float): Fractional slippage per trade (e.g. 0.001 = 0.1%).
        allow_overlap (bool): If False, skips candles until trade is resolved.
        
    Returns:
        pd.DataFrame: All simulated trades.
    """
    trades: List[Dict[str, Any]] = []
    i = 2
    tolerance = 0.001  # ~0.1% tolerance for breakeven detection
    max_look = 50

    while i < len(df) - max_look:
        if i - volume_ma_period < 0:
            i += 1
            continue

        prev2, prev1, curr = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        volume_ma = df["volume"].iloc[i - volume_ma_period:i].mean()

        # get signal
        signal = strategy_func(prev2, prev1, curr, volume_ma, i, df)
        if not signal or not signal.get("signal"):
            i += 1
            continue
        if signal:

            order = signal["signal"]
            entry = float(signal["entry"])
            sl = float(signal["sl"])
            tp = float(signal["tp"])
            atr = float(signal["atr"])

            # apply slippage on entry
            if order == "long":
                entry *= (1 + slippage)
                needed_bars = bars_needed_to_reach(entry, tp, atr)
                lookahead   = max(5, min(48, needed_bars))
            elif order == "short":
                entry *= (1 - slippage)
                needed_bars = bars_needed_to_reach(entry, tp, atr)
                lookahead   = max(5, min(48, needed_bars))
            else:
                continue

            future_df = df.iloc[i + 1:i + lookahead + 1]

            result, exit_price, jumped = "no_result", entry, False

            # Check for immediate exit conditions
            counter = 0
            for _,row in future_df.iterrows():
                counter += 1
                if order == "long":
                    if row["low"] <= sl:
                        result, exit_price, jumped = "loss", sl, True
                        skip = counter
                        break
                    elif row["high"] >= tp:
                        result, exit_price, jumped = "win", tp, True
                        skip = counter
                        break
                    elif abs(row["close"] - entry) <= tolerance:
                        result, exit_price, jumped = "breakeven", row["close"], True
                        skip = counter
                        break
                    else:
                        if row["close"] < entry:
                            result, exit_price, jumped = "loss", row["close"], True
                            skip = counter
                            break
                        elif row["close"] > entry:
                            result, exit_price, jumped = "win", row["close"], True
                            skip = counter
                            break
                elif order == "short":
                    if row["high"] >= sl:
                        result, exit_price, jumped = "loss", sl, True
                        skip = counter
                        break
                    elif row["low"] <= tp:
                        result, exit_price, jumped = "win", tp, True
                        skip = counter
                        break
                    elif abs(row["close"] - entry) <= tolerance:
                        result, exit_price, jumped = "breakeven", row["close"], True
                        skip = counter
                        break
                    else:
                        if row["close"] > entry:
                            result, exit_price, jumped = "loss", row["close"], True
                            skip = counter
                            break
                        elif row["close"] < entry:
                            result, exit_price, jumped = "win", row["close"], True
                            skip = counter
                            break
                else:
                    continue

            trade_metadata = {k: signal.get(k) for k in [
                "ema_trend", "macd_momentum", "rsi_divergence",
                "has_pattern", "sr_confluence", "fib", "triangle",
                "wedge", "flag_pattern", "double", "score", "rr", "volatility_ok"
            ]}

            trades.append({
                "entry_time": curr.name,
                "order": order,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit": exit_price,
                "result": result,
                **trade_metadata
            })

        if jumped:
            i += skip + 1
        else:
            i += 1

    return pd.DataFrame(trades)

