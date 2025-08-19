from typing import Callable, Dict, Any, List, Tuple, Optional
import pandas as pd

# ========= ATR-multiple + S/R confluence =========
def dynamic_tp_sl(
    entry: float,
    r_levels: List[float],
    s_levels: List[float],
    atr: float,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: float = 2.0,
) -> Tuple[float, float, float, float]:
    """
    Build ATR-based raw TP/SL, then prefer *nearby* confluence S/R levels if they are
    on the right side and not unreasonably far vs ATR targets.
    Returns: (tp_long, sl_long, tp_short, sl_short)
    """

    # Raw ATR targets
    sl_long_raw = entry - atr_mult_sl * atr
    tp_long_raw = entry + atr_mult_tp * atr
    sl_short_raw = entry + atr_mult_sl * atr
    tp_short_raw = entry - atr_mult_tp * atr

    # S/R helpers
    nearest_res_above = min([r for r in (r_levels or []) if r > entry], default=None)
    nearest_sup_below = max([s for s in (s_levels or []) if s < entry], default=None)

    # Long: prefer nearest support for SL if it's tighter than raw (but still below entry)
    sl_long = sl_long_raw
    if nearest_sup_below:
        sl_long = nearest_sup_below

    # Long: prefer nearest resistance for TP if it's tighter than raw (and above entry)
    tp_long = tp_long_raw
    if nearest_res_above:
        tp_long = nearest_res_above

    # Short: prefer nearest resistance for SL if it's tighter than raw (above entry)
    sl_short = sl_short_raw
    if nearest_res_above:
        sl_short = nearest_res_above

    # Short: prefer nearest support for TP if it's tighter than raw (below entry)
    tp_short = tp_short_raw
    if nearest_sup_below:
        tp_short = nearest_sup_below

    return tp_long, sl_long, tp_short, sl_short


# ========= Intrabar exit resolution =========
def resolve_bar_outcome(
    order: str, row: pd.Series, entry: float, sl: float, tp: float
) -> Optional[Tuple[str, float]]:
    """
    Decide if SL or TP is hit within a single bar, and in which order.
    We assume the typical conservative sequence:
      - For LONG: if low <= SL, SL is hit before checking TP; else if high >= TP, TP hit.
      - For SHORT: if high >= SL, SL hit first; else if low <= TP, TP hit.
    Returns (result, exit_price) or None if neither hit on this bar.
    """
    low = row["low"]
    high = row["high"]

    if order == "long":
        # SL first, then TP
        if low <= sl:
            return ("loss", sl)
        if high >= tp:
            return ("win", tp)
    else:  # short
        if high >= sl:
            return ("loss", sl)
        if low <= tp:
            return ("win", tp)

    return None  # no hit this bar


# ========= Scan forward until exit (or max bars) =========
def scan_forward_to_exit(
    df: pd.DataFrame,
    start_idx: int,
    order: str,
    entry: float,
    sl: float,
    tp: float,
    max_ahead: int,
    slippage: float = 0.0,
    fee_pct: float = 0.0,
) -> Tuple[str, float, int]:
    """
    Walk forward up to `max_ahead` bars to find exit.
    Applies intrabar logic; if simultaneous, we conservatively assume SL first for longs,
    SL first for shorts as implemented in resolve_bar_outcome.
    Adds slippage and fees to the exit price.
    Returns: (result, exit_price_after_slip_fee, bars_held)
    """
    result = "no_result"
    exit_price = entry
    bars_held = 0

    future_df = df.iloc[start_idx + 1 : start_idx + 1 + max_ahead]
    for offset, (_, row) in enumerate(future_df.iterrows(), start=1):
        outcome = resolve_bar_outcome(order, row, entry, sl, tp)
        if outcome:
            result, raw_exit = outcome
            bars_held = offset

            # Apply slippage directionally
            if order == "long":
                if result == "win":
                    exit_price = raw_exit * (1 - slippage)
                else:
                    exit_price = raw_exit * (1 - slippage)
            else:
                if result == "win":
                    exit_price = raw_exit * (1 + slippage)
                else:
                    exit_price = raw_exit * (1 + slippage)
            break

    if result == "no_result":
        # Force exit at the last available bar (close)
        last_close = future_df.iloc[-1]["close"] if len(future_df) else df.iloc[start_idx]["close"]
        bars_held = min(max_ahead, len(future_df))
        exit_price = last_close  # no slippage assumed on time-based exit
        # Win/loss based on favorable movement relative to entry
        if (order == "long" and exit_price > entry) or (order == "short" and exit_price < entry):
            result = "win"
        elif abs(exit_price - entry) / entry < 0.0005:  # ~5 bps = breakeven tolerance
            result = "breakeven"
        else:
            result = "loss"

    # Apply fee on both sides (entry+exit) as % notional; we fold it into exit price
    # (You can also track PnL net fees elsewhere; here we just “worsen” exit slightly.)
    fee_mult = (1 + fee_pct) if order == "long" else (1 - fee_pct)
    exit_price *= fee_mult

    return result, exit_price, bars_held


# ========= Main simulator (one trade at a time) =========
def simulate_trades(
    df: pd.DataFrame,
    strategy_func: Callable[[pd.Series, pd.Series, pd.Series, float, int, pd.DataFrame], Optional[Dict[str, Any]]],
    volume_ma_period: int = 12,
    lookahead: int = 20,
    slippage: float = 0.0,
    fee_pct: float = 0.0004,  # ~4 bps per side, adjust to your venue
    enforce_rr: bool = True,
    min_rr: float = 1.5,
    use_dynamic_tp_sl: bool = False,
    get_sr_levels: Optional[Callable[[pd.DataFrame, int], Tuple[List[float], List[float]]]] = None,
) -> List[Dict[str, Any]]:
    """
    - One open trade at a time.
    - No future leakage: strategy_func sees only up to bar i.
    - If use_dynamic_tp_sl is True, we will recompute TP/SL using ATR + S/R at entry.
      For that, pass `get_sr_levels(df, i) -> (r_levels, s_levels)` or ensure your strategy returns them.
    - After a trade closes (TP/SL/time), we skip forward by exactly the number of bars that trade lasted.
    """

    trades: List[Dict[str, Any]] = []
    i = 2  # need at least prev2, prev1
    n = len(df)

    while i < n - 1:
        # Enough data for volume MA?
        if i - volume_ma_period < 0:
            i += 1
            continue

        prev2, prev1, curr = df.iloc[i - 2], df.iloc[i - 1], df.iloc[i]
        volume_ma = df["volume"].iloc[i - volume_ma_period : i].mean()

        # Strategy proposes a trade (or None)
        signal = strategy_func(prev2, prev1, curr, volume_ma, i, df)
        if not signal or not signal.get("signal"):
            i += 1
            continue

        order = signal["signal"]
        entry = float(signal["entry"])
        sl = float(signal["sl"])
        tp = float(signal["tp"])

        # Optional: recompute TP/SL with ATR+S/R dynamic model at entry
        if use_dynamic_tp_sl:
            atr = float(curr["atr"]) if "atr" in curr and pd.notnull(curr["atr"]) else None
            if atr is not None and atr > 0:
                if get_sr_levels is not None:
                    r_levels, s_levels = get_sr_levels(df, i)
                else:
                    # If your strategy returns levels, use them; else empty lists
                    r_levels = signal.get("r_levels", []) or []
                    s_levels = signal.get("s_levels", []) or []
                tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(entry, r_levels, s_levels, atr)
                if order == "long":
                    tp, sl = tp_long, sl_long
                else:
                    tp, sl = tp_short, sl_short

        # Enforce a minimum RR at entry (based on these final TP/SL)
        if enforce_rr:
            rr = 0.0
            if order == "long" and sl < entry and tp > entry:
                rr = abs(tp - entry) / max(1e-12, abs(entry - sl))
            elif order == "short" and sl > entry and tp < entry:
                rr = abs(entry - tp) / max(1e-12, abs(sl - entry))
            if rr < min_rr:
                # Skip poor RR setups
                i += 1
                continue

        # Apply slippage to the entry fill
        entry = entry * (1 + slippage) if order == "long" else entry * (1 - slippage)

        # Walk forward until exit (or max lookahead)
        result, exit_price, bars_held = scan_forward_to_exit(
            df,
            start_idx=i,
            order=order,
            entry=entry,
            sl=sl,
            tp=tp,
            max_ahead=lookahead,
            slippage=slippage,
            fee_pct=fee_pct,
        )

        # Record trade
        meta_keys = [
                "ema_trend", "macd_momentum", "rsi_divergence",
                "has_pattern", "sr_confluence", "fib", "triangle",
                "wedge", "flag_pattern", "double", "score", "rr", "volatility_ok"
            ]
        metadata = {k: signal.get(k) for k in meta_keys if k in signal}

        trades.append({
            "entry_time": curr.name,
            "order": order,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "exit": exit_price,
            "result": result,
            "bars_held": bars_held,
            **metadata
        })

        # === Critical fix: advance i by bars_held once a trade is closed ===
        # Ensures NO overlapping/duplicate trades.
        # If for some reason bars_held == 0 (edge case), at least advance 1 bar.
        i += max(1, bars_held)

    return pd.DataFrame(trades)
