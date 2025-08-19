from .patterns_short import *

def is_near_r(price, r_levels):
    if r_levels >= price * 1.02 and r_levels <= price * 1.05:
        return True
    return False

def bearish_signal(prev2, prev1, curr, volume_ma, i, df):

    r_levels = get_resistance_levels(df, i)

    has_bearish_pattern = (
        is_bearish_engulfing(prev1, curr)
        or is_hanging_man(curr)
        or is_shooting_star(curr)
        or is_dark_cloud_cover(prev1, curr)
        or is_three_black_crows(prev2, prev1, curr)
        or is_bearish_harami(prev1, curr)
        or is_bearish_marubozu(curr)
        or is_bearish_belt_hold(curr)
        or is_tweezer_top(prev1, curr)
        or is_bearish_three_inside_down(prev2, prev1, curr)
    )

    bearish_indicators = {
        "ema_trend": curr["ema_50"] < curr["ema_200"],
        "healthy_rsi": 35 <= curr["rsi"] <= 55,
        "macd_momentum": curr["macd_line"] < curr["macd_signal"],
        "volume_confirmation": curr["volume"] > volume_ma,
        "rsi_divergence": is_rsi_bearish_divergence(df, i),
        "fib_bounce_bearish": is_fib_bounce_bearish(df, i),
        "sr_confluence": is_near_r(curr["high"], r_levels),
        "triangle_confluence": is_bearish_triangle_pattern(df, i),
        "flag_pattern": is_bearish_flag_pattern(df, i),
        "double_top": is_double_top(df, i),
        "falling_wedge": is_falling_wedge(df, i),
        "falling_triangle": is_falling_triangle(df, i)
    }

    return has_bearish_pattern, bearish_indicators, r_levels

def bearish_score(bearish_indicators, has_bearish_pattern):

    score = 0

    if bearish_indicators:
        ema_trend = bearish_indicators["ema_trend"]
        healthy_rsi = bearish_indicators["healthy_rsi"]
        macd_momentum = bearish_indicators["macd_momentum"]
        volume_confirmation = bearish_indicators["volume_confirmation"]
        rsi_divergence = bearish_indicators["rsi_divergence"]
        fib_bounce = bearish_indicators["fib_bounce_bearish"]
        sr_confluence = bearish_indicators["sr_confluence"]
        triangle_confluence = bearish_indicators["triangle_confluence"]
        flag_pattern = bearish_indicators["flag_pattern"]
        double_top = bearish_indicators["double_top"]
        falling_triangle = bearish_indicators["falling_triangle"]
        falling_wedge = bearish_indicators["falling_wedge"]

        # Score calculation
        score = (
            ema_trend * 3 +
            macd_momentum * 3 +
            #volume_confirmation * 3 +
            has_bearish_pattern +
            rsi_divergence * 3 +
            fib_bounce +
            sr_confluence +
            triangle_confluence +
            flag_pattern +
            double_top +
            falling_triangle * 3 +
            falling_wedge
        )
    return score

def short_trade(signal, entry, tp, sl, bearish_indicators, has_bearish_pattern, score):

    return {
        "signal": signal,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "ema_trend": bearish_indicators["ema_trend"],
        "healthy_rsi": bearish_indicators["healthy_rsi"],
        "macd_momentum": bearish_indicators["macd_momentum"],
        #"volume_confirmation": bearish_indicators["volume_confirmation"],
        "rsi_divergence": bearish_indicators["rsi_divergence"],
        "has_pattern": has_bearish_pattern,
        "sr_confluence": bearish_indicators["sr_confluence"],
        "fib_bounce": bearish_indicators["fib_bounce_bearish"],
        "triangle": bearish_indicators["falling_triangle"],
        "wedge": bearish_indicators["falling_wedge"],
        "triangle_confluence": bearish_indicators["triangle_confluence"],
        "flag_pattern": bearish_indicators["flag_pattern"],
        "double": bearish_indicators["double_top"],
        "score": score
    }