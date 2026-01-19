import pandas as pd
import ta
from typing import List, Tuple
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
        sl_long = nearest_sup_below * 0.99
    tp_long = tp_long_raw
    if nearest_res_above and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        tp_long = nearest_res_above * 1.02

    # Short
    sl_short = sl_short_raw
    if nearest_res_above and (nearest_res_above - entry) <= tolerance_atr_mult * atr:
        sl_short = nearest_res_above * 1.01
    tp_short = tp_short_raw
    if nearest_sup_below and (entry - nearest_sup_below) <= tolerance_atr_mult * atr:
        tp_short = nearest_sup_below * 0.98

    return float(tp_long), float(sl_long), float(tp_short), float(sl_short)

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

def rr_for(direction, entry, tp, sl):
    """Risk/Reward with sanity checks."""
    if direction == "long" and sl and tp and sl < entry < tp:
        return (tp - entry) / (entry - sl)
    if direction == "short" and sl and tp and tp < entry < sl:
        return (entry - tp) / (sl - entry)
    return 0.0
