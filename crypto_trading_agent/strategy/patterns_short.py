def is_bearish_engulfing(prev, curr):
    required_keys = ["open", "close"]
    if not all(k in prev for k in required_keys) or not all(k in curr for k in required_keys):
        return False
    return (curr["close"] < curr["open"] and
            prev["close"] > prev["open"] and
            curr["open"] >= prev["close"] and
            curr["close"] <= prev["open"])

def is_hanging_man(curr):
    # Hanging man: small real body near the top, long lower shadow, little or no upper shadow
    body_top = max(curr["open"], curr["close"])
    body_bottom = min(curr["open"], curr["close"])
    lower_shadow = body_bottom - curr["low"]
    upper_shadow = curr["high"] - body_top
    body_size = abs(curr["close"] - curr["open"])
    candle_range = curr["high"] - curr["low"]
    return (
        curr["close"] < curr["open"] and
        lower_shadow > 2 * body_size and
        upper_shadow <= 0.1 * candle_range and
        body_size <= 0.3 * candle_range
    )

def is_evening_star(prev2, prev1, curr):
    # Evening Star: bullish candle, small-bodied candle (star), bearish candle closing well into the body of the first
    body_prev2 = abs(prev2["close"] - prev2["open"])
    body_prev1 = abs(prev1["close"] - prev1["open"])
    body_curr = abs(curr["close"] - curr["open"])
    candle_range_prev2 = prev2["high"] - prev2["low"]
    candle_range_prev1 = prev1["high"] - prev1["low"]
    candle_range_curr = curr["high"] - curr["low"]

    # Star should have a small body relative to prev2
    small_star = body_prev1 < 0.5 * body_prev2

    # Gaps: prev1 opens above prev2 close, curr opens below prev1 close (not always required in crypto, but more reliable)
    gap_up = prev1["open"] > prev2["close"]
    gap_down = curr["open"] < prev1["close"]

    return (
        prev2["close"] > prev2["open"] and  # first bullish
        small_star and
        curr["close"] < curr["open"] and  # third bearish
        curr["close"] < (prev2["open"] + prev2["close"]) / 2 and  # closes well into first body
        (gap_up or prev1["open"] >= prev2["close"]) and
        (gap_down or curr["open"] <= prev1["close"])
    )

def is_shooting_star(curr):
    # Shooting star: small real body near the low, long upper shadow, little or no lower shadow
    body_top = max(curr["open"], curr["close"])
    body_bottom = min(curr["open"], curr["close"])
    upper_shadow = curr["high"] - body_top
    lower_shadow = body_bottom - curr["low"]
    body_size = abs(curr["close"] - curr["open"])
    candle_range = curr["high"] - curr["low"]
    return (
        upper_shadow > 2 * body_size and
        lower_shadow <= 0.1 * candle_range and
        body_size <= 0.3 * candle_range
    )

def is_bearish_harami(prev, curr):
    # Bearish Harami: previous candle bullish, current candle bearish and contained within previous candle's body
    return (
        prev["close"] > prev["open"] and
        curr["close"] < curr["open"] and
        curr["open"] < prev["close"] and curr["open"] > prev["open"] and
        curr["close"] < prev["close"] and curr["close"] > prev["open"]
    )

def is_three_black_crows(prev2, prev1, curr):
    return (
        prev2["close"] < prev2["open"] and
        prev1["close"] < prev1["open"] and
        curr["close"] < curr["open"] and
        prev1["open"] < prev2["open"] and prev1["open"] > prev2["close"] and
        curr["open"] < prev1["open"] and curr["open"] > prev1["close"] and
        curr["close"] < prev1["close"] and prev1["close"] < prev2["close"]
    )

def is_dark_cloud_cover(prev, curr):
    required_keys = ["close", "open"]
    if not all(k in prev for k in required_keys) or not all(k in curr for k in required_keys):
        return False
    return (
        prev["close"] > prev["open"] and
        curr["open"] > prev["close"] and
        curr["close"] < (prev["open"] + prev["close"]) / 2 and
        curr["close"] > prev["open"] and
        curr["close"] < prev["close"]
    )

def is_bearish_marubozu(curr):
    body_size = abs(curr["close"] - curr["open"])
    candle_range = curr["high"] - curr["low"]
    upper_shadow = max(0, curr["high"] - max(curr["open"], curr["close"]))
    lower_shadow = max(0, min(curr["open"], curr["close"]) - curr["low"])
    return (
        curr["close"] < curr["open"] and
        upper_shadow <= 0.05 * candle_range and
        lower_shadow <= 0.05 * candle_range and
        body_size >= 0.9 * candle_range
        )

def is_tweezer_top(prev1, curr):
    required_keys = ["close", "open", "high", "low"]
    if not all(k in curr for k in required_keys) or not all(k in prev1 for k in required_keys):
        return False
    return (curr["close"] < curr["open"] and
            prev1["close"] > prev1["open"] and
            curr["high"] == prev1["high"] and
            curr["low"] > prev1["low"])

def is_bearish_three_inside_down(prev2, prev1, curr):
    # prev2: first candle (bullish), prev1: second candle (bearish, inside prev2), curr: third candle (bearish, closes below prev2 open)
    return (
        prev2["close"] > prev2["open"] and  # first candle bullish
        prev1["close"] < prev1["open"] and  # second candle bearish
        prev1["open"] < prev2["close"] and prev1["close"] > prev2["open"] and  # second candle inside first
        curr["close"] < curr["open"] and  # third candle bearish
        curr["close"] < prev2["open"]  # third closes below first open
    )

def is_bearish_belt_hold(curr):
    # Bearish belt hold: open at/near high, close much lower, little/no upper shadow
    upper_shadow = curr["high"] - curr["open"]
    body = curr["open"] - curr["close"]
    candle_range = curr["high"] - curr["low"]
    return (
        curr["close"] < curr["open"] and
        upper_shadow <= 0.05 * candle_range and  # very small upper shadow
        body >= 0.7 * candle_range  # large bearish body
    )

def is_rsi_bearish_divergence(df, i, rsi_window=14):
    if i < 2 or i - rsi_window < 0:
        return False

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    # Look for higher price high
    if curr["high"] <= prev["high"]:
        return False

    # Look for lower RSI high
    if curr["rsi"] >= prev["rsi"]:
        return False

    return True

def is_fib_bounce_bearish(df, i, lookback=7):
    if i < lookback or i >= len(df):
        return False
    # For bearish: look for price retracing upward to a Fibonacci resistance zone after a down move
    # Define swing high and low in the last `lookback` candles (bearish: high first, then low)
    recent_high = df["high"].iloc[i - lookback:i].max()
    recent_low = df["low"].iloc[i - lookback:i].min()

    # Calculate key Fibonacci retracement levels from high to low
    fib_382 = recent_high - (recent_high - recent_low) * 0.382
    fib_618 = recent_high - (recent_high - recent_low) * 0.618

    # Check if current price is bouncing down from that zone (bearish rejection)
    current_price = df.iloc[i]["close"]
    prev_close = df.iloc[i - 1]["close"] if i > 0 else current_price

    # Bearish rejection: price enters the zone and then closes below it
    in_zone_prev = fib_618 <= prev_close <= fib_382
    out_zone_now = current_price < fib_618

    return in_zone_prev and out_zone_now

def is_double_top(df, i, window=5):
    if i < window:
        return False

    highs = df["high"].iloc[i - window:i].values

    # Find the first peak (left), trough (middle), and second peak (right)
    peak1_idx = highs.argmax()
    if peak1_idx == 0 or peak1_idx == len(highs) - 1:
        return False  # Peak can't be at the edge

    # Find the second peak after the trough
    trough_idx = highs[peak1_idx + 1:].argmin() + (peak1_idx + 1)
    if trough_idx == len(highs) - 1:
        return False  # Trough can't be at the end

    peak2_idx = highs[trough_idx + 1:].argmax() + (trough_idx + 1)
    if peak2_idx >= len(highs):
        return False

    peak1 = highs[peak1_idx]
    peak2 = highs[peak2_idx]
    trough = highs[trough_idx]

    # Peaks should be similar and higher than trough
    peaks_similar = abs(peak1 - peak2) / peak1 < 0.05
    peaks_above_trough = peak1 > trough and peak2 > trough

    return peaks_similar and peaks_above_trough

def is_bearish_triangle_pattern(df, i, window=8, threshold=0.05):
    if i < window:
        return False
    # For a flat bottom and falling top (descending triangle, bearish)
    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if lows are approximately equal (flat bottom)
    lows_flat = all(abs(lows[j] - lows[j + 1]) <= threshold * ((lows[j] + lows[j + 1]) / 2) for j in range(len(lows) - 1))

    # Check if highs are descending (lower highs)
    highs_descending = all(highs[j] > highs[j + 1] for j in range(len(highs) - 1))

    return lows_flat and highs_descending

def is_bearish_flag_pattern(df, i, window=5):
    if i < window:
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs are descending and lows are flat or slightly ascending (bearish flag)
    highs_descending = all(highs[j] > highs[j + 1] for j in range(len(highs) - 1))
    lows_flat_or_ascending = all(lows[j] <= lows[j + 1] * 1.02 for j in range(len(lows) - 1))  # allow up to 2% rise

    return highs_descending and lows_flat_or_ascending

def is_falling_wedge(df, i, window=10):
    if i < window:
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs are descending and lows are descending but lows descend less steeply (converging)
    highs_descending = all(highs[j] > highs[j + 1] for j in range(len(highs) - 1))
    lows_descending = all(lows[j] > lows[j + 1] for j in range(len(lows) - 1))

    # Calculate the slopes (rate of descent)
    high_slope = (highs[-1] - highs[0]) / window
    low_slope = (lows[-1] - lows[0]) / window

    # For a falling wedge, the lows should descend less steeply than the highs (low_slope > high_slope)
    converging = abs(low_slope) < abs(high_slope)

    return highs_descending and lows_descending and converging

def is_falling_triangle(df, i, window=10):
    if i < window:
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs are flat or slightly descending and lows are descending
    highs_flat_or_descending = all(highs[j] >= highs[j + 1] for j in range(len(highs) - 1))
    lows_descending = all(lows[j] > lows[j + 1] for j in range(len(lows) - 1))

    return highs_flat_or_descending and lows_descending

def resistance_levels(df, i, lookback=20):
    resistance = []

    if i < lookback or i >= len(df):
        return resistance
    
    high = df["high"].iloc[i]

    resistance_lvl = df["high"].iloc[i - lookback:i].max()

    if resistance_lvl >= high * 1.05:
        resistance.append(resistance_lvl)
    return resistance

def levels(df, i, lookback=10):
    resistance = []
    if i < lookback or i >= len(df)-lookback:
        return resistance

    window = df["high"].iloc[i-lookback:i]
    if df["high"].iloc[i] < window.min():
        resistance.append(window.mean())
    return resistance

import numpy as np

def get_resistance_levels(df, i, lookback=10, min_touches=3, tolerance=0.01):
    """
    Real-time friendly resistance detection:
    - Finds clusters of highs in the last `lookback` bars
    - Requires at least `min_touches` within tolerance
    - Returns strongest resistance (avg of cluster with most touches)
    """
    if i < lookback:
        return []

    highs = df["high"].iloc[i-lookback:i].values
    resistance_candidates = []

    for high in highs:
        # count how many highs are within tolerance of this one
        cluster = [x for x in highs if abs(x - high) / high <= tolerance]
        if len(cluster) >= min_touches:
            # store cluster avg
            resistance_candidates.append(np.mean(cluster))

    if resistance_candidates:
        # return most frequent (mode-like)
        return [min(resistance_candidates, key=lambda x: abs(x - np.median(resistance_candidates)))]
    
    return []
