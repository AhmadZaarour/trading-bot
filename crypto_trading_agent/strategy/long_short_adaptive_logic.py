# short_trade_logic.py
from .patterns_short import *
from .patterns_long import *

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

def evaluate_trade_confluence(df, i, prev2, prev1, curr, volume_ma, entry, tp_long, sl_long, tp_short, sl_short):
    """
    Evaluates trade setups using a realistic confluence model:
    1. Trend filter (EMA50 vs EMA200)
    2. Momentum confirmation (MACD + RSI)
    3. Volatility filter (ATR regime)
    4. Risk/Reward filter (RR >= 2.0)
    5. Secondary confluence (patterns, divergences, S/R, etc.)
    """

    # --- Get bullish + bearish pattern signals
    bull_pattern, bull_ind, bull_sr = get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma)
    bear_pattern, bear_ind, bear_sr = get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma)


    trade = None
    # --- Long/short trade ---
    rr_long  = rr_for("long",  curr["close"], tp_long,  sl_long)
    rr_short = rr_for("short", curr["close"], tp_short, sl_short)


    # === 1. Trend filter ===
    trend_long = abs(curr["ema_50"] - curr["ema_200"]) < 0.1
    trend_short = abs(curr["ema_50"] - curr["ema_200"]) < 0.1

    # === 2. Momentum filter ===
    momentum_long = (curr["macd_line"] > curr["macd_signal"] and curr["rsi"] >= 48) or (curr["rsi"] < 30 and curr["macd_line"] > curr["macd_signal"])
    momentum_short = (curr["macd_line"] < curr["macd_signal"] and curr["rsi"] < 48) or (curr["rsi"] > 70 and curr["macd_line"] < curr["macd_signal"])


    # === 3. Volatility filter ===
    # Normalized ATR (% of price)
    curr_atr_pct = curr["atr"] / curr["close"]

    # Lookback volatility baseline (last 50 candles up to current index)
    idx = curr.name  # current row index in df
    atr_ma_pct = (df.loc[:idx, "atr"] / df.loc[:idx, "close"]).rolling(50).mean().iloc[-1]

    # Check if current volatility is acceptable
    atr_ok = curr_atr_pct >= 1.0 * atr_ma_pct if "atr" in curr else True

    # === 4. Risk/Reward filter ===
    rr_long_ok = rr_long >= 2.0
    rr_short_ok = rr_short >= 2.0

    # --- Long setup
    if trend_long and momentum_long and atr_ok and rr_long_ok:
        score = (
            #bull_pattern * 0.5
            #+ bull_ind["fib_bounce"] * 0.5
            #bull_ind["sr_confluence"] * 2
            bull_ind["flag_pattern"] * 0.75
            + bull_ind["triangle_confluence"] * 0.5
            + bull_ind["rising_triangle"] * 0.75
            + bull_ind["rising_wedge"] * 0.75
            #+ bull_ind["double_bottom"] * 0.5
            + bull_ind["rsi_divergence"] * 0.5
        )
        if score >= 1:
            
            trade = {
                "signal": "long",
                "entry": entry,
                "sl": sl_long,
                "tp": tp_long,
                "rr": rr_long,
                "atr": curr["atr"],
                "ema_trend": True,
                "macd_momentum": True,
                "volatility_ok": atr_ok,
                "score": score,
                "has_pattern": bull_pattern,
                "fib": bull_ind["fib_bounce"],
                "sr_confluence": bull_ind["sr_confluence"],
                "flag_pattern": bull_ind["flag_pattern"],
                "triangle": bull_ind["rising_triangle"],
                "wedge": bull_ind["rising_wedge"],
                "double": bull_ind["double_bottom"],
                "rsi_divergence": bull_ind["rsi_divergence"],
                "sr_confluence": has_support(bull_sr)
            }
        else:
            trade = {
                "signal": None
            }

    # --- Short setup
    elif trend_short and momentum_short and atr_ok and rr_short_ok:
        score = (
            #bear_pattern * 0.5
            #+ bear_ind["fib_bounce"] * 0.5
            #bear_ind["sr_confluence"] * 2
            bear_ind["flag_pattern"] * 0.75
            + bear_ind["triangle_confluence"] * 0.5
            + bear_ind["falling_triangle"] * 0.75
            + bear_ind["falling_wedge"] * 0.75
            #+ bear_ind["double_top"] * 0.5
            + bear_ind["rsi_divergence"] * 0.5
        )
        if score >= 1:
            trade = {
                "signal": "short",
                "entry": entry,
                "sl": sl_short,
                "tp": tp_short,
                "rr": rr_short,
                "atr": curr["atr"],
                "ema_trend": True,
                "macd_momentum": True,
                "volatility_ok": atr_ok,
                "score": score,
                "has_pattern": bear_pattern,
                "fib": bear_ind["fib_bounce"],
                "sr_confluence": bear_ind["sr_confluence"],
                "flag_pattern": bear_ind["flag_pattern"],
                "triangle": bear_ind["falling_triangle"],
                "wedge": bear_ind["falling_wedge"],
                "double": bear_ind["double_top"],
                "rsi_divergence": bear_ind["rsi_divergence"],
                "sr_confluence": has_resistance(bear_sr)
            }
        else:
            trade = {
                "signal": None
            }

    return trade
