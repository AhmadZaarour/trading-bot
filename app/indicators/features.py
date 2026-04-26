import pandas as pd
import ta
import numpy as np
from typing import List, Tuple, Optional
from .patterns_short import *
from .patterns_long import *

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # EMA
    df["ema_20"] = ta.trend.EMAIndicator(df["close"], window=20).ema_indicator()
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
    # ADX
    adx = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=14
    )
    df["adx"] = adx.adx()
    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_high"] = bb.bollinger_hband()
    df["bb_mid"] = bb.bollinger_mavg()
    df["bb_low"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_high"] - df["bb_low"]) / df["bb_mid"]
    # Volume
    df["volume_sma_20"] = df["volume"].rolling(20).mean()

    df.dropna(inplace=True)
    return df

def dynamic_tp_sl_long_oco(
    entry: float,
    r_levels: List[float],
    s_levels: List[float],
    atr: float,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: float = 2.0,
    tolerance_atr_mult: float = 2.0,
    # “buffers” around levels to improve fill/trigger behavior
    tp_buffer: float = 0.002,          # 0.1% below resistance
    sl_buffer: float = 0.002,          # 0.1% below support
    stop_limit_offset: float = 0.002,  # stopLimit below stopPrice by 0.1%
) -> Tuple[float, float, float]:
    if atr <= 0:
        # fallback: just use small % if ATR unavailable
        atr = entry * 0.002  # 0.2% fallback (tune this)

    # raw ATR-based levels
    sl_raw = entry - atr_mult_sl * atr
    tp_raw = entry + atr_mult_tp * atr

    # nearest SR
    nearest_res_above: Optional[float] = min((r for r in (r_levels or []) if r > entry), default=None)
    nearest_sup_below: Optional[float] = max((s for s in (s_levels or []) if s < entry), default=None)

    # start with raw
    sl = sl_raw
    tp = tp_raw

    # snap SL to support if close
    if nearest_sup_below is not None and (entry - nearest_sup_below) <= tolerance_atr_mult * atr:
        # slightly below support so it actually protects
        sl = nearest_sup_below * (1.0 - sl_buffer)

    # snap TP to resistance if close
    if nearest_res_above is not None and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        # slightly below resistance so it’s more likely to fill
        tp = nearest_res_above * (1.0 - tp_buffer)

    # sanity clamps for a long position
    if tp <= entry:
        tp = entry + atr_mult_tp * atr
    if sl >= entry:
        sl = entry - atr_mult_sl * atr

    stop_price = sl
    stop_limit_price = sl * (1.0 - stop_limit_offset)  # required for OCO sell

    return float(tp), float(stop_price), float(stop_limit_price)


def dynamic_tp_sl(
    entry: float,
    r_levels: List[float],
    s_levels: List[float],
    atr: float,
    atr_mult_sl: float = 1.0,
    atr_mult_tp: float = 2.0,
    tolerance_atr_mult: float = 2.0,
    tp_buffer: float = 0.002,  # 0.2%
    sl_buffer: float = 0.002,  # 0.2%
) -> Tuple[float, float, float, float]:
    if atr <= 0:
        atr = entry * 0.002

    sl_long = entry - atr_mult_sl * atr
    tp_long = entry + atr_mult_tp * atr
    sl_short = entry + atr_mult_sl * atr
    tp_short = entry - atr_mult_tp * atr

    nearest_res_above = min((r for r in (r_levels or []) if r > entry), default=None)
    nearest_sup_below = max((s for s in (s_levels or []) if s < entry), default=None)

    # Long: snap with buffers
    if nearest_sup_below is not None and (entry - nearest_sup_below) <= tolerance_atr_mult * atr:
        sl_long = nearest_sup_below * (1.0 - sl_buffer)
    if nearest_res_above is not None and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        tp_long = nearest_res_above * (1.0 - tp_buffer)

    # Short: snap with buffers
    if nearest_res_above is not None and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        sl_short = nearest_res_above * (1.0 + sl_buffer)
    if nearest_sup_below is not None and (entry - nearest_sup_below) <= tolerance_atr_mult * atr:
        tp_short = nearest_sup_below * (1.0 + tp_buffer)

    # sanity
    if tp_long <= entry:
        tp_long = entry + atr_mult_tp * atr
    if sl_long >= entry:
        sl_long = entry - atr_mult_sl * atr

    if tp_short >= entry:
        tp_short = entry - atr_mult_tp * atr
    if sl_short <= entry:
        sl_short = entry + atr_mult_sl * atr

    return float(tp_long), float(sl_long), float(tp_short), float(sl_short)

def simple_tp_sl(entry: float, signal: str, s_levels: List[float], r_levels: List[float]):
    if signal == "long":
        tp = r_levels if r_levels else entry * 1.02
        sl = entry * 0.98
    elif signal == "short":
        tp = s_levels if s_levels else entry * 0.98
        sl = entry * 1.02
    else:
        raise ValueError("Signal must be 'long' or 'short'")

    return float(tp), float(sl)

# ==== Core Logic ====
def get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma):
    r_levels = get_resistance_levels(df, i, curr)

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
        "triangle_confluence": is_bearish_triangle_pattern(df, i),
        "flag_pattern": is_bearish_flag_pattern(df, i),
        "double_top": is_double_top(df, i),
        "falling_wedge": is_falling_wedge(df, i),
        "falling_triangle": is_falling_triangle(df, i)
    }

    return has_bearish_pattern, indicators, r_levels

def get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma):
    s_levels = get_support_levels(df, i, curr)

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
        "triangle_confluence": is_triangle_pattern(df, i),
        "flag_pattern": is_flag_pattern(df, i),
        "double_bottom": is_double_bottom(df, i),
        "rising_wedge": is_rising_wedge(df, i),
        "rising_triangle": is_rising_triangle(df, i)
    }

    return has_bullish_pattern, indicators, s_levels

def rr_for(direction, entry, tp, sl):
    if direction == "long" and sl is not None and tp is not None and sl < entry < tp:
        return (tp - entry) / (entry - sl)
    if direction == "short" and sl is not None and tp is not None and tp < entry < sl:
        return (entry - tp) / (sl - entry)
    return 0.0

def _cluster_levels(values: np.ndarray, tol_pct: float, min_touches: int) -> List[Tuple[float, int]]:
    """
    Sort + cluster consecutive values that are within tol_pct of the running cluster mean.
    Returns [(cluster_mean, touches), ...] sorted by cluster_mean.
    """
    if len(values) == 0:
        return []

    vals = np.sort(values.astype(float))
    clusters: List[Tuple[float, int]] = []

    cur_sum = float(vals[0])
    cur_count = 1
    cur_mean = cur_sum

    for v in vals[1:]:
        # Compare to current cluster mean (more stable than comparing to v itself)
        if abs(v - cur_mean) / max(cur_mean, 1e-12) <= tol_pct:
            cur_sum += float(v)
            cur_count += 1
            cur_mean = cur_sum / cur_count
        else:
            if cur_count >= min_touches:
                clusters.append((cur_mean, cur_count))
            cur_sum = float(v)
            cur_count = 1
            cur_mean = cur_sum

    if cur_count >= min_touches:
        clusters.append((cur_mean, cur_count))

    return clusters

def get_support_levels(
    df: pd.DataFrame,
    i: int,
    curr,
    lookback: int = 15,
    min_touches: int = 3,
    top_n: int = 1,
) -> List[float]:
    if i < lookback:
        return []

    lows = df["low"].iloc[i - lookback : i].to_numpy()
    ref_price = float(df["close"].iloc[i - 1])
    atr_pct = curr["atr"] / curr["close"]  # e.g. 0.008 = 0.8%
    tolerance = min(0.015, max(0.004, 2.0 * atr_pct))
    # clamp to 0.4% .. 1.5%, centered around 2*ATR%


    clusters = _cluster_levels(lows, tolerance, min_touches)
    # supports = clusters below price, pick closest below (highest level)
    supports = [(lvl, touches) for (lvl, touches) in clusters if lvl < ref_price]
    supports.sort(key=lambda x: x[0], reverse=True)  # closest below first

    return [lvl for (lvl, _) in supports[:top_n]]

def get_resistance_levels(
    df: pd.DataFrame,
    i: int,
    curr,
    lookback: int = 15,
    min_touches: int = 3,
    top_n: int = 1,
) -> List[float]:
    if i < lookback:
        return []

    highs = df["high"].iloc[i - lookback : i].to_numpy()
    ref_price = float(df["close"].iloc[i - 1])
    atr_pct = curr["atr"] / curr["close"]  # e.g. 0.008 = 0.8%
    tolerance = min(0.015, max(0.004, 2.0 * atr_pct))

    clusters = _cluster_levels(highs, tolerance, min_touches)
    # resistances = clusters above price, pick closest above (lowest level)
    resistances = [(lvl, touches) for (lvl, touches) in clusters if lvl > ref_price]
    resistances.sort(key=lambda x: x[0])  # closest above first

    return [lvl for (lvl, _) in resistances[:top_n]]