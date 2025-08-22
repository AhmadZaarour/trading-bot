from .long_short_adaptive_logic import *
import numpy as np

def analyze_row_dynamic(prev2, prev1, curr, volume_ma, i, df):
    
    entry = curr["close"]
    _,_,r_levels = get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma)
    _,_,s_levels = get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma)
    #window = 14
    #start = max(0, i - window)
    #recent_closes = df["close"].iloc[start:i+1]
    #volatility = np.std(recent_closes)
    atr = curr["atr"]  # already computed in add_indicators
    tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
        entry=curr["close"],
        r_levels=r_levels,     # list of resistances above
        s_levels=s_levels,     # list of supports below
        atr=atr
    )

    trade = evaluate_trade_confluence(df, i, prev2, prev1, curr, volume_ma, entry, tp_long, sl_long, tp_short, sl_short)
    return trade

from typing import List, Tuple

def dynamic_tp_sl(
    entry: float,
    r_levels: List[float],
    s_levels: List[float],
    atr: float,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: float = 2.0,
    tolerance: float = 2,  # max multiple of ATR allowed for S/R override
) -> Tuple[float, float, float, float]:
    """
    ATR-based TP/SL with S/R confluence preference.
    Returns: (tp_long, sl_long, tp_short, sl_short)
    """

    # Raw ATR targets
    sl_long_raw = entry - atr_mult_sl * atr
    tp_long_raw = entry + atr_mult_tp * atr
    sl_short_raw = entry + atr_mult_sl * atr
    tp_short_raw = entry - atr_mult_tp * atr

    # Nearest S/R
    nearest_res_above = min([r for r in (r_levels or []) if r > entry], default=None)
    nearest_sup_below = max([s for s in (s_levels or []) if s < entry], default=None)

    # --- LONG ---
    sl_long = sl_long_raw
    if nearest_sup_below and (entry - nearest_sup_below) <= tolerance * atr:
        sl_long = nearest_sup_below  # use support if reasonably close

    tp_long = tp_long_raw
    if nearest_res_above and (nearest_res_above - entry) <= tolerance * atr:
        tp_long = nearest_res_above  # use resistance if reasonably close

    # --- SHORT ---
    sl_short = sl_short_raw
    if nearest_res_above and (nearest_res_above - entry) <= tolerance * atr:
        sl_short = nearest_res_above  # use resistance if close

    tp_short = tp_short_raw
    if nearest_sup_below and (entry - nearest_sup_below) <= tolerance * atr:
        tp_short = nearest_sup_below  # use support if close

    return tp_long, sl_long, tp_short, sl_short
