import os
import time
from datetime import datetime, timezone
from typing import List, Tuple, Dict, Any, Optional
from strategy.patterns_short import *
from strategy.patterns_long import *
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from binance.client import Client
from binance.enums import *
import ta  # pip install ta

# ===============================
# Env / Client (Futures Testnet)
# ===============================
load_dotenv()
API_KEY = os.getenv("API_KEY_testnet")
API_SECRET = os.getenv("API_SECRET_testnet")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing API_KEY_testnet / API_SECRET_testnet in .env")

client = Client(API_KEY, API_SECRET, testnet=True)
# Ensure futures testnet endpoints (python-binance sometimes needs this explicit override)
client.FUTURES_URL = 'https://testnet.binancefuture.com/fapi'

# ===============================
# Parameters
# ===============================
SYMBOL = "XRPUSDT"
INTERVAL = Client.KLINE_INTERVAL_1HOUR  # 1h
LOOKBACK = 120            # candles to feed strategy
RISK_PER_TRADE = 0.02
MAX_LEVERAGE = 5
MAX_BARS_PER_TRADE = 20   # exit if exceeded
POLL_SECONDS = 30         # check every 30s for a new closed candle

# ===============================
# Helpers: OHLCV / Indicators
# ===============================
def fetch_ohlcv(symbol: str, interval: str, limit: int) -> pd.DataFrame:
    """Closed klines only -> DataFrame indexed by timestamp with ohlcv floats."""
    kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("close_time", inplace=True)  # align to candle close

    num_cols = ["open","high","low","close","volume"]
    df[num_cols] = df[num_cols].astype(float)
    return df[num_cols]

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMA
    df["ema_50"] = ta.trend.EMAIndicator(df["close"], window=50).ema_indicator()
    df["ema_200"] = ta.trend.EMAIndicator(df["close"], window=200).ema_indicator()
    # RSI
    df["rsi"] = ta.momentum.RSIIndicator(df["close"], window=14).rsi()
    # MACD
    macd = ta.trend.MACD(df["close"])
    df["macd_line"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["macd_hist"] = macd.macd_diff()
    # ATR
    atr = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["atr"] = atr.average_true_range()

    df.dropna(inplace=True)
    return df

# ===============================
# Your S/R (real-time friendly)
# ===============================
def get_support_levels(df: pd.DataFrame, i: int, lookback=10, min_touches=3, tolerance=0.01) -> List[float]:
    """Cluster lows in the last `lookback` bars; return strongest support."""
    if i < lookback:
        return []
    lows = df["low"].iloc[i - lookback : i].values
    candidates = []
    for lv in lows:
        cluster = [x for x in lows if abs(x - lv) / lv <= tolerance]
        if len(cluster) >= min_touches:
            candidates.append(float(np.mean(cluster)))
    if candidates:
        # choose representative near median (stable)
        med = float(np.median(candidates))
        best = min(candidates, key=lambda x: abs(x - med))
        return [best]
    return []

def get_resistance_levels(df: pd.DataFrame, i: int, lookback=10, min_touches=3, tolerance=0.01) -> List[float]:
    """Cluster highs in the last `lookback` bars; return strongest resistance."""
    if i < lookback:
        return []
    highs = df["high"].iloc[i - lookback : i].values
    candidates = []
    for hv in highs:
        cluster = [x for x in highs if abs(x - hv) / hv <= tolerance]
        if len(cluster) >= min_touches:
            candidates.append(float(np.mean(cluster)))
    if candidates:
        med = float(np.median(candidates))
        best = min(candidates, key=lambda x: abs(x - med))
        return [best]
    return []

# ==== Utility ====
def has_resistance(r_level):
    if r_level:
        return True
    return False

# ==== Core Logic ====
def get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma):
    r_levels = get_resistance_levels(df, i)

    # Candle patterns
    has_bearish_pattern = any([
        is_bearish_engulfing(prev1, curr),
        is_hanging_man(curr),
        #is_shooting_star(curr),
        #is_dark_cloud_cover(prev1, curr),
        is_three_black_crows(prev2, prev1, curr),
        is_bearish_harami(prev1, curr),
        #is_bearish_marubozu(curr),
        #is_bearish_belt_hold(curr),
        is_tweezer_top(prev1, curr),
        is_bearish_three_inside_down(prev2, prev1, curr)
    ])

    # Indicators
    indicators = {
        "ema_trend": curr["ema_50"] < curr["ema_200"],
        "healthy_rsi": curr["rsi"] < 50 or curr["rsi"] > 70,
        "macd_momentum": curr["macd_line"] < curr["macd_signal"],
        "volume_confirmation": curr["volume"] > volume_ma,
        "rsi_divergence": is_rsi_bearish_divergence(df, i),
        "fib_bounce": is_fib_bounce_bearish(df, i),
        "sr_confluence": has_resistance(r_levels),
        "triangle_confluence": is_bearish_triangle_pattern(df, i),
        "flag_pattern": is_bearish_flag_pattern(df, i),
        "double_top": is_double_top(df, i),
        "falling_wedge": is_falling_wedge(df, i),
        "falling_triangle": is_falling_triangle(df, i)
    }

    return has_bearish_pattern, indicators, r_levels

def has_support(s_level):
    if s_level:
        return True
    return False

def get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma):
    s_levels = get_support_levels(df, i)

    # Candle patterns
    has_bullish_pattern = any([
        is_bullish_engulfing(prev1, curr),
        is_hammer(curr),
        #is_morning_star(prev2, prev1, curr),
        #is_piercing_line(prev1, curr),
        is_three_white_soldiers(prev2, prev1, curr),
        is_bullish_harami(prev1, curr),
        #is_bullish_marubozu(curr),
        #is_bullish_belt_hold(curr),
        is_tweezer_bottom(prev1, curr),
        is_bullish_three_inside_up(prev2, prev1, curr)
    ])

    indicators = {
        "ema_trend": curr["ema_50"] > curr["ema_200"],
        "healthy_rsi": curr["rsi"] > 50 or curr["rsi"] < 30,
        "macd_momentum": curr["macd_line"] > curr["macd_signal"],
        "volume_confirmation": curr["volume"] > volume_ma,
        "rsi_divergence": is_rsi_bullish_divergence(df, i),
        "fib_bounce": is_fib_bounce(df, i),
        "sr_confluence": has_support(s_levels),
        "triangle_confluence": is_triangle_pattern(df, i),
        "flag_pattern": is_flag_pattern(df, i),
        "double_bottom": is_double_bottom(df, i),
        "rising_wedge": is_rising_wedge(df, i),
        "rising_triangle": is_rising_triangle(df, i)
    }

    return has_bullish_pattern, indicators, s_levels

# ===============================
# Dynamic TP/SL (ATR + S/R)
# ===============================
def dynamic_tp_sl(
    entry: float,
    r_levels: List[float],
    s_levels: List[float],
    atr: float,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: float = 2.0,
    tolerance_atr_mult: float = 2.0,  # only accept S/R within 2*ATR of entry
) -> Tuple[float, float, float, float]:
    sl_long_raw = entry - atr_mult_sl * atr
    tp_long_raw = entry + atr_mult_tp * atr
    sl_short_raw = entry + atr_mult_sl * atr
    tp_short_raw = entry - atr_mult_tp * atr

    nearest_res_above = min([r for r in (r_levels or []) if r > entry], default=None)
    nearest_sup_below = max([s for s in (s_levels or []) if s < entry], default=None)

    # Long
    sl_long = sl_long_raw
    if nearest_sup_below and (entry - nearest_sup_below) <= tolerance_atr_mult * atr:
        sl_long = nearest_sup_below
    tp_long = tp_long_raw
    if nearest_res_above and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        tp_long = nearest_res_above

    # Short
    sl_short = sl_short_raw
    if nearest_res_above and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        sl_short = nearest_res_above
    tp_short = tp_short_raw
    if nearest_sup_below and (entry - nearest_sup_below) <= tolerance_atr_mult * atr:
        tp_short = nearest_sup_below

    return float(tp_long), float(sl_long), float(tp_short), float(sl_short)

def rr_for(direction: str, entry: float, tp: float, sl: float) -> float:
    if direction == "long" and sl < entry < tp:
        return (tp - entry) / (entry - sl)
    if direction == "short" and tp < entry < sl:
        return (entry - tp) / (sl - entry)
    return 0.0

# =======================================
# Confluence Filter (trend/momentum/vol)
# =======================================
def evaluate_trade_confluence(df: pd.DataFrame, i: int) -> Dict[str, Any]:
    """Returns {'signal': 'long'|'short'|None, 'entry','sl','tp', ...}"""
    curr = df.iloc[i]
    prev1 = df.iloc[i - 1]
    prev2 = df.iloc[i - 2]
    volume_ma = df["volume"].iloc[i-5:i].mean() if i >= 5 else df["volume"].iloc[:i].mean()

    entry = curr["close"]
    r_levels = get_resistance_levels(df, i)
    s_levels = get_support_levels(df, i)

    tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
        entry=entry, r_levels=r_levels, s_levels=s_levels, atr=curr["atr"]
    )

    # --- Get bullish + bearish pattern signals
    bull_pattern, bull_ind, bull_sr = get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma)
    bear_pattern, bear_ind, bear_sr = get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma)

    # Trend
    trend_long = curr["ema_50"] > curr["ema_200"]
    trend_short = curr["ema_50"] < curr["ema_200"]

    # Momentum
    momentum_long = (curr["macd_line"] > curr["macd_signal"] and curr["rsi"] > 50) or (
        curr["rsi"] < 30 and curr["macd_line"] > curr["macd_signal"]
    )
    momentum_short = (curr["macd_line"] < curr["macd_signal"] and curr["rsi"] < 50) or (
        curr["rsi"] > 70 and curr["macd_line"] < curr["macd_signal"]
    )

    # Volatility regime (no lookahead)
    curr_atr_pct = curr["atr"] / curr["close"]
    atr_ma_pct = (df["atr"].iloc[: i + 1] / df["close"].iloc[: i + 1]).rolling(50).mean().iloc[-1]
    atr_ok = curr_atr_pct >= atr_ma_pct if np.isfinite(atr_ma_pct) else True

    # RR
    rr_long = rr_for("long", entry, tp_long, sl_long)
    rr_short = rr_for("short", entry, tp_short, sl_short)
    rr_long_ok = rr_long >= 2.0
    rr_short_ok = rr_short >= 2.0

    # Decide
    if trend_long and momentum_long and atr_ok and rr_long_ok:
        score = (
            #bull_pattern * 0.5
            #+ bull_ind["fib_bounce"] * 0.5
            #bull_ind["sr_confluence"] * 2
            bull_ind["flag_pattern"] * 0.75
            + bull_ind["triangle_confluence"] * 0.5
            + bull_ind["rising_triangle"] * 0.75
            + bull_ind["rising_wedge"] * 0.75
            + bull_ind["double_bottom"] * 0.5
            + bull_ind["rsi_divergence"] * 0.5
        )
        if score >= 1:
            return {
                "signal": "long",
                "entry": float(entry),
                "sl": float(sl_long),
                "tp": float(tp_long),
                "rr": float(rr_long),
            }
    if trend_short and momentum_short and atr_ok and rr_short_ok:
        score = (
            #bear_pattern * 0.5
            #+ bear_ind["fib_bounce"] * 0.5
            #bear_ind["sr_confluence"] * 2
            bear_ind["flag_pattern"] * 0.75
            + bear_ind["triangle_confluence"] * 0.5
            + bear_ind["falling_triangle"] * 0.75
            + bear_ind["falling_wedge"] * 0.75
            + bear_ind["double_top"] * 0.5
            + bear_ind["rsi_divergence"] * 0.5
        )
        if score >= 1:
            return {
                "signal": "short",
                "entry": float(entry),
                "sl": float(sl_short),
                "tp": float(tp_short),
                "rr": float(rr_short),
            }
    return {"signal": None}

# ===============================
# Binance utilities (futures)
# ===============================
def get_symbol_filters(symbol: str):
    info = client.futures_exchange_info()
    sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
    lot = next(f for f in sym["filters"] if f["filterType"] == "LOT_SIZE")
    price = next(f for f in sym["filters"] if f["filterType"] == "PRICE_FILTER")
    return {
        "stepSize": float(lot["stepSize"]),
        "minQty": float(lot["minQty"]),
        "tickSize": float(price["tickSize"]),
    }

def round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return float(np.floor(qty / step) * step)

def round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    return float(np.round(np.round(price / tick) * tick, 10))

def get_usdt_balance() -> float:
    bals = client.futures_account_balance()
    usdt = next(b for b in bals if b["asset"] == "USDT")
    return float(usdt["balance"])

def get_position_amt(symbol: str) -> float:
    pos = client.futures_position_information(symbol=symbol)[0]
    return float(pos["positionAmt"])  # >0 long, <0 short, 0 flat

def ensure_leverage(symbol: str, leverage: int = MAX_LEVERAGE):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print("Leverage change warning:", e)

# ===============================
# Main loop (acts on candle close)
# ===============================
def main():
    print("Starting Futures Testnet bot...")
    filters = get_symbol_filters(SYMBOL)
    step = filters["stepSize"]
    tick = filters["tickSize"]

    ensure_leverage(SYMBOL, MAX_LEVERAGE)

    open_trade: Optional[Dict[str, Any]] = None
    last_seen_close_time: Optional[pd.Timestamp] = None

    while True:
        try:
            df = fetch_ohlcv(SYMBOL, INTERVAL, LOOKBACK + 5)
            df = add_indicators(df)
            if df.empty:
                time.sleep(POLL_SECONDS)
                continue

            latest_close_time = df.index[-1]
            if last_seen_close_time is not None and latest_close_time == last_seen_close_time:
                # no new closed candle yet
                time.sleep(POLL_SECONDS)
                continue

            # New closed candle available
            last_seen_close_time = latest_close_time
            i = len(df) - 1
            entry_price = float(df["close"].iloc[-1])

            # Manage open trade timeout
            if open_trade:
                open_trade["bars_open"] += 1
                if open_trade["bars_open"] >= MAX_BARS_PER_TRADE:
                    # market exit
                    amt = abs(get_position_amt(SYMBOL))
                    if amt > 0:
                        side = SIDE_SELL if open_trade["side"] == SIDE_BUY else SIDE_BUY
                        qty = round_step(amt, step)
                        if qty > 0:
                            client.futures_create_order(
                                symbol=SYMBOL,
                                side=side,
                                type=ORDER_TYPE_MARKET,
                                quantity=str(qty),
                                reduceOnly=True,
                            )
                    open_trade = None
                    print(f"[{datetime.now(timezone.utc)}] Exited due to max bars.")
                    time.sleep(POLL_SECONDS)
                    continue

            # If no position open, look for new signal
            current_pos = get_position_amt(SYMBOL)
            if abs(current_pos) < 1e-12 and not open_trade:
                trade = evaluate_trade_confluence(df, i)
                if trade.get("signal") in ("long", "short"):
                    # Sizing
                    balance = get_usdt_balance()
                    risk_amount = balance * RISK_PER_TRADE
                    sl = float(trade["sl"])
                    tp = float(trade["tp"])
                    sl_pct = abs(entry_price - sl) / entry_price
                    if sl_pct <= 0:
                        time.sleep(POLL_SECONDS)
                        continue

                    raw_position = risk_amount / sl_pct
                    max_position = (balance * MAX_LEVERAGE) / entry_price
                    qty = min(raw_position, max_position)

                    qty = round_step(qty, step)
                    if qty <= 0:
                        time.sleep(POLL_SECONDS)
                        continue

                    # Market entry
                    side = SIDE_BUY if trade["signal"] == "long" else SIDE_SELL
                    order = client.futures_create_order(
                        symbol=SYMBOL,
                        side=side,
                        type=ORDER_TYPE_MARKET,
                        quantity=str(qty),
                    )

                    # Exit orders (reduceOnly)
                    opp_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                    stop_price = round_tick(sl, tick)
                    tp_price = round_tick(tp, tick)

                    # Stop Loss
                    client.futures_create_order(
                        symbol=SYMBOL,
                        side=opp_side,
                        type=FUTURE_ORDER_TYPE_STOP_MARKET,
                        stopPrice=str(stop_price),
                        closePosition=True,   # close entire position
                        reduceOnly=True,
                        workingType="CONTRACT_PRICE",
                    )

                    # Take Profit
                    client.futures_create_order(
                        symbol=SYMBOL,
                        side=opp_side,
                        type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                        stopPrice=str(tp_price),
                        closePosition=True,
                        reduceOnly=True,
                        workingType="CONTRACT_PRICE",
                    )

                    open_trade = {
                        "side": side,
                        "entry_time": datetime.now(timezone.utc),
                        "bars_open": 0,
                    }

                    print(f"[{datetime.now(timezone.utc)}] Entered {trade['signal']} qty={qty} entry≈{entry_price} SL={stop_price} TP={tp_price}")

            time.sleep(POLL_SECONDS)

        except Exception as e:
            print("Error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()

