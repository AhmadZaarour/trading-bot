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
import ta
from config import *
from data.fetcher import fetch_ohlcv
from strategy.adaptive_trade_logic import dynamic_tp_sl
from strategy.long_short_adaptive_logic import get_bearish_indicators, get_bullish_indicators, rr_for
from indicators.calculator import add_indicators
from risk.manager import *
from utils.helpers import *

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
    _, bull_ind, _ = get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma)
    _, bear_ind, _ = get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma)

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
            bull_ind["flag_pattern"] * 0.75
            + bull_ind["fib_bounce"] * 0.25
            + bull_ind["triangle_confluence"] * 0.5
            + bull_ind["rising_triangle"] * 0.75
            + bull_ind["rising_wedge"] * 0.75
            + bull_ind["rsi_divergence"] * 0.5
        )
        if score >= 1.25:
            return {
                "signal": "long",
                "entry": float(entry),
                "sl": float(sl_long),
                "tp": float(tp_long),
                "rr": float(rr_long),
            }
    if trend_short and momentum_short and atr_ok and rr_short_ok:
        score = (
            bear_ind["flag_pattern"] * 0.75
            + bear_ind["fib_bounce"] * 0.25
            + bear_ind["triangle_confluence"] * 0.5
            + bear_ind["falling_triangle"] * 0.75
            + bear_ind["falling_wedge"] * 0.75
            + bear_ind["rsi_divergence"] * 0.5
        )
        if score >= 1.25:
            return {
                "signal": "short",
                "entry": float(entry),
                "sl": float(sl_short),
                "tp": float(tp_short),
                "rr": float(rr_short),
            }
    return {"signal": None}

# ===============================
# Main loop (acts on candle close)
# ===============================
def main():
    print("Starting Futures Testnet bot...")
    filters = get_symbol_filters(SYMBOL, client)
    step = filters["stepSize"]
    tick = filters["tickSize"]

    ensure_leverage(SYMBOL, MAX_LEVERAGE, client)

    open_trade: Optional[Dict[str, Any]] = None
    last_seen_close_time: Optional[pd.Timestamp] = None

    while True:
        try:
            df = fetch_ohlcv(SYMBOL, INTERVAL, LOOKBACK + 1, client)
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

            # Manage open trade state
            current_pos = get_position_amt(SYMBOL, client)
            
            if open_trade:
                print("managing trade ...")
                open_trade["bars_open"] += 1
                # Case 1: Position was force-closed (SL/TP hit)
                if abs(current_pos) < 1e-12:
                    print(f"[{datetime.now(timezone.utc)}] Trade closed (SL/TP triggered).")
                    # Clean up leftover orders to avoid ghost TP/SL
                    try:
                        client.futures_cancel_all_open_orders(symbol=SYMBOL)
                    except Exception as e:
                        print("Cancel leftover orders failed:", e)
                    open_trade = None

                # Case 2: Timeout exit
                elif open_trade["bars_open"] >= MAX_BARS_PER_TRADE:
                    amt = abs(current_pos)
                    if amt > 0:
                        side = SIDE_SELL if open_trade["signal"] == "long" else SIDE_BUY
                        client.futures_create_order(
                            symbol=SYMBOL,
                            side=side,
                            type=ORDER_TYPE_MARKET,
                            quantity=str(amt),
                            reduceOnly=True,
                        )
                        print(f"[{datetime.now(timezone.utc)}] Exited trade due to max bars.")
                    open_trade = None


            # If no position open, look for new signal
            current_pos = get_position_amt(SYMBOL, client)
            if abs(current_pos) < 1e-12 and not open_trade:
                trade = evaluate_trade_confluence(df, i)
                print("Evaluating trade confluence...")
                if trade.get("signal") in ("long", "short"):
                    # Sizing
                    balance = get_usdt_available(client)  # available, not total
                    sl = float(trade["sl"])
                    tp = float(trade["tp"])

                    qty = safe_position_size(entry_price, sl, balance, filters, MAX_LEVERAGE, RISK_PER_TRADE)

                    print(f"Balance(avail)={balance}, entry={entry_price}, sl={sl}, "
                        f"sl_dist={abs(entry_price-sl)}, calc_qty={qty}, "
                        f"notional≈{qty*entry_price if qty else 0}")

                    if qty <= 0:
                        print("Skipped trade: qty too small or margin-capped.")
                        time.sleep(POLL_SECONDS)
                        continue

                    # Market entry (with backoff if margin tight)
                    side = SIDE_BUY if trade["signal"] == "long" else SIDE_SELL
                    qty = round_step(qty, step)
                    order = try_market_order_with_backoff(client, SYMBOL, side, qty)
                    if order is None:
                        print("Aborted: could not place market order within margin limits.")
                        time.sleep(POLL_SECONDS)
                        continue

                    # Exit orders (closePosition)
                    opp_side = SIDE_SELL if side == SIDE_BUY else SIDE_BUY
                    if order:
                        stop_price = round_tick(sl, tick)
                        # SL
                        client.futures_create_order(
                            symbol=SYMBOL,
                            side=opp_side,
                            type=FUTURE_ORDER_TYPE_STOP_MARKET,
                            stopPrice=stop_price,
                            quantity=str(qty),
                            reduceOnly=True,
                        )
                        tp_price = round_tick(tp, tick)
                        #tp
                        client.futures_create_order(
                            symbol=SYMBOL,
                            side=opp_side,
                            type=FUTURE_ORDER_TYPE_TAKE_PROFIT_MARKET,
                            stopPrice=tp_price,
                            quantity=str(qty),
                            reduceOnly=True,
                        )
                        print("set sl/tp ...")
                    else:
                        print("no order yet ...")
                        continue

                    open_trade = {
                    "signal": trade["signal"],  # long/short
                    "side": side,               # buy/sell
                    "entry_time": datetime.now(timezone.utc),
                    "bars_open": 0,
                    "sl": stop_price,
                    "tp": tp_price,
                    "qty": qty                  # position size
                }


                    print(f" opened trade: [{datetime.now(timezone.utc)}] Entered {trade['signal']} | qty={qty} amount USDT={qty*entry_price} | entry≈{entry_price} | SL={stop_price} | TP={tp_price}")
                    print("trade ordered ...")

        except Exception as e:
            print("Error:", e)
            time.sleep(5)

if __name__ == "__main__":
    main()

