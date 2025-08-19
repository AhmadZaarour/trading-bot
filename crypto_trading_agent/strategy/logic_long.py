from .patterns_long import *

def is_near_s(price, s_levels):
    if s_levels <= price * 0.98 and s_levels >= price * 0.95:
        return True
    return False

def bullish_signal(prev2, prev1, curr, volume_ma, i, df):

    s_levels = get_support_levels(df, i)

    has_bullish_pattern = (
        is_bullish_engulfing(prev1, curr)
        or is_hammer(curr)
        or is_morning_star(prev2, prev1, curr)
        or is_piercing_line(prev1, curr)
        or is_three_white_soldiers(prev2, prev1, curr)
        or is_inverted_hammer(curr)
        or is_bullish_harami(prev1, curr)
        or is_bullish_marubozu(curr)
        or is_bullish_belt_hold(curr)
        or is_tweezer_bottom(prev1, curr)
        or is_bullish_three_inside_up(prev2, prev1, curr)
    )

    bullish_indicators = {
        "ema_trend": curr["ema_50"] > curr["ema_200"],
        "healthy_rsi": 45 <= curr["rsi"] <= 65,
        "macd_momentum": curr["macd_line"] > curr["macd_signal"],
        "volume_confirmation": curr["volume"] > volume_ma,
        "rsi_divergence": is_rsi_bullish_divergence(df, i),
        "fib_bounce": is_fib_bounce(df, i),
        "sr_confluence": is_near_s(curr["low"], s_levels),
        "triangle_confluence": is_triangle_pattern(df, i),
        "flag_pattern": is_flag_pattern(df, i),
        "double_bottom": is_double_bottom(df, i),
        "rising_triangle": is_rising_triangle(df, i),
        "rising_wedge": is_rising_wedge(df, i)
    }

    return has_bullish_pattern, bullish_indicators, s_levels

def bullish_score(bullish_indicators, has_bullish_pattern):

    score = 0

    if bullish_indicators:
        ema_trend = bullish_indicators["ema_trend"]
        healthy_rsi = bullish_indicators["healthy_rsi"]
        macd_momentum = bullish_indicators["macd_momentum"]
        volume_confirmation = bullish_indicators["volume_confirmation"]
        rsi_divergence = bullish_indicators["rsi_divergence"]
        fib_bounce = bullish_indicators["fib_bounce"]
        sr_confluence = bullish_indicators["sr_confluence"]
        triangle_confluence = bullish_indicators["triangle_confluence"]
        flag_pattern = bullish_indicators["flag_pattern"]
        double_bottom = bullish_indicators["double_bottom"]
        rising_triangle = bullish_indicators["rising_triangle"]
        rising_wedge = bullish_indicators["rising_wedge"]

        # Score calculation
        score = (
            ema_trend * 3 +
            macd_momentum * 3 +
            #volume_confirmation +
            has_bullish_pattern +
            rsi_divergence * 3 +
            fib_bounce +
            sr_confluence +
            triangle_confluence +
            flag_pattern +
            double_bottom +
            rising_triangle * 3 +
            rising_wedge
            )
    return score

def long_trade(signal, entry, tp, sl, bullish_indicators, has_bullish_pattern, score):

    return {
        "signal": signal,
        "entry": entry,
        "sl": sl,
        "tp": tp,
        "ema_trend": bullish_indicators["ema_trend"],
        "healthy_rsi": bullish_indicators["healthy_rsi"],
        "macd_momentum": bullish_indicators["macd_momentum"],
        #"volume_confirmation": bullish_indicators["volume_confirmation"],
        "rsi_divergence": bullish_indicators["rsi_divergence"],
        "has_pattern": has_bullish_pattern,
        "sr_confluence": bullish_indicators["sr_confluence"],
        "fib_bounce": bullish_indicators["fib_bounce"],
        "triangle": bullish_indicators["rising_triangle"],
        "wedge": bullish_indicators["rising_wedge"],
        "triangle_confluence": bullish_indicators["triangle_confluence"],
        "flag_pattern": bullish_indicators["flag_pattern"],
        "double": bullish_indicators["double_bottom"],
        "score": score
    }
