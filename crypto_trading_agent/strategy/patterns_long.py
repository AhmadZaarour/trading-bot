def is_bullish_engulfing(prev, curr):
    return (
        curr.get("close", 0) > curr.get("open", 0) and
        prev.get("close", 0) < prev.get("open", 0) and
        curr.get("close", 0) > prev.get("open", 0) and
        curr.get("open", 0) < prev.get("close", 0)
    )

def is_hammer(curr):
    open_ = curr.get("open", 0)
    close = curr.get("close", 0)
    low = curr.get("low", 0)
    high = curr.get("high", 0)
    body = abs(close - open_)
    lower_wick = min(open_, close) - low
    upper_wick = high - max(open_, close)

    return (
        lower_wick > 2 * body and 
        upper_wick < body
    )

def is_morning_star(prev2, prev1, curr):
    # Three candle reversal pattern
    return (
        prev2["close"] < prev2["open"] and  # red
        abs(prev1["close"] - prev1["open"]) < (prev2["open"] - prev2["close"]) * 0.5 and  # small doji/star
        curr["close"] > curr["open"] and
        curr["close"] > ((prev2["open"] + prev2["close"]) / 2)
    )

def is_piercing_line(prev, curr):
    return (
        prev["close"] < prev["open"] and  # red candle
        curr["open"] < prev["low"] and    # opens lower
        curr["close"] > ((prev["open"] + prev["close"]) / 2) and  # closes > midpoint
        curr["close"] < prev["open"]      # doesn't fully engulf
    )

def is_three_white_soldiers(prev2, prev1, curr):
    return (
        prev2["close"] > prev2["open"] and
        prev1["close"] > prev1["open"] and
        curr["close"] > curr["open"] and
        prev1["open"] > prev2["open"] and
        curr["open"] > prev1["open"] and
        prev1["close"] > prev2["close"] and
        curr["close"] > prev1["close"]
    )

def is_inverted_hammer(curr):
    body = abs(curr["close"] - curr["open"])
    upper_wick = curr["high"] - max(curr["close"], curr["open"])
    lower_wick = min(curr["close"], curr["open"]) - curr["low"]
    
    return (
        upper_wick > 2 * body and
        lower_wick < body
    )

def is_bullish_harami(prev, curr):
    return (
        prev["close"] < prev["open"] and  # red
        curr["close"] > curr["open"] and  # green
        curr["open"] > prev["close"] and
        curr["close"] < prev["open"]
    )

def is_tweezer_bottom(prev, curr):
    return (
        abs(prev["low"] - curr["low"]) < 0.2 * (prev["high"] - prev["low"]) and
        prev["close"] < prev["open"] and  # red
        curr["close"] > curr["open"]      # green
    )

def is_bullish_belt_hold(curr):
    # Opens at/near low, closes strong green
    body = curr["close"] - curr["open"]
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    return (
        body > 0 and
        lower_wick < 0.1 * body and
        upper_wick < 0.3 * body
    )

def is_bullish_marubozu(curr):
    # No wicks, strong green candle
    body = curr["close"] - curr["open"]
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    return (
        body > 0 and
        lower_wick < 1e-6 and
        upper_wick < 1e-6
    )

def is_bullish_three_inside_up(prev, curr, next_):
    # Harami followed by confirmation
    return (
        prev["close"] < prev["open"] and
        curr["open"] > prev["close"] and
        curr["close"] < prev["open"] and
        next_["close"] > curr["close"] and
        next_["close"] > prev["open"]
    )

def is_rsi_bullish_divergence(df, i, rsi_window=14):
    if i < 2 or i - rsi_window < 0:
        return False

    if "rsi" not in df.columns:
        return False

    prev = df.iloc[i - 1]
    curr = df.iloc[i]

    # Look for lower price low
    if curr.get("low", None) is None or prev.get("low", None) is None:
        return False
    if curr["low"] >= prev["low"]:
        return False

    # Look for higher RSI low
    if curr.get("rsi", None) is None or prev.get("rsi", None) is None:
        return False
    if curr["rsi"] <= prev["rsi"]:
        return False

    return True

def is_fib_bounce(df, i, lookback=7, tolerance=0.001):
    if i < lookback or i >= len(df):
        return False

    recent_low = df["low"].iloc[i - lookback:i].min()
    recent_high = df["high"].iloc[i - lookback:i].max()

    # Ensure it's an uptrend swing (low before high)
    if recent_high <= recent_low:
        return False

    fib_382 = recent_low + (recent_high - recent_low) * 0.382
    fib_618 = recent_low + (recent_high - recent_low) * 0.618

    curr = df.iloc[i]
    prev = df.iloc[i - 1]

    # Check if price recently entered fib zone (wick or close)
    in_zone_prev = fib_382 <= prev["low"] <= fib_618 or fib_382 <= prev["close"] <= fib_618
    # Rejection if now we’re back above fib_618 (close confirmation)
    rejection = curr["close"] > fib_618 * (1 + tolerance)

    return in_zone_prev and rejection


import numpy as np

def get_support_levels(df, i, lookback=10, min_touches=3, tolerance=0.01):
    """
    Real-time friendly support detection:
    - Finds clusters of lows in the last `lookback` bars
    - Requires at least `min_touches` within tolerance
    - Returns strongest support (avg of cluster with most touches)
    """
    if i < lookback:
        return []

    lows = df["low"].iloc[i-lookback:i].values
    support_candidates = []

    for low in lows:
        # count how many lows are within tolerance of this one
        cluster = [x for x in lows if abs(x - low) / low <= tolerance]
        if len(cluster) >= min_touches:
            # store cluster avg
            support_candidates.append(np.mean(cluster))

    if support_candidates:
        # return most frequent (mode-like)
        return [min(support_candidates, key=lambda x: abs(x - np.median(support_candidates)))]
    
    return []

def get_oblique_support(df, i, window=10, min_points=3, tolerance=0.01):
    """
    Finds an oblique (trendline) support using linear regression on lows.
    Returns the slope and intercept if a valid trendline is found, else None.
    """
    os = []
    if i < window or i > len(df):
        return os

    lows = df["low"].iloc[i - window:i].values
    x = range(window)

    # Fit a line to the lows
    coeffs = np.polyfit(x, lows, 1)
    slope, intercept = coeffs

    # Count how many points are within tolerance of the line (support touches)
    support_points = sum(
        abs((slope * xi + intercept) - low) / low < tolerance for xi, low in zip(x, lows)
    )

    if support_points >= min_points and slope > 0:
        os.append(intercept)
        return os
    return os

def get_oblique_resistance(df, i, window=10, min_points=3, tolerance=0.01):
    """
    Finds an oblique (trendline) resistance using linear regression on highs.
    Returns the slope and intercept if a valid trendline is found, else None.
    """
    if i < window or i > len(df):
        return None

    highs = df["high"].iloc[i - window:i].values
    x = range(window)

    # Fit a line to the highs
    coeffs = np.polyfit(x, highs, 1)
    slope, intercept = coeffs

    # Count how many points are within tolerance of the line (resistance touches)
    resistance_points = sum(
        abs((slope * xi + intercept) - high) / high < tolerance for xi, high in zip(x, highs)
    )

    if resistance_points >= min_points and slope < 0:
        return slope, intercept
    return None

def is_triangle_pattern(df, i, window=8, threshold=0.05):
    if i < window or i > len(df):
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs are approximately equal (flat top)
    highs_flat = all(abs(highs[j] - highs[j + 1]) <= threshold * highs[j] for j in range(len(highs) - 1))

    # Check if lows are ascending (higher lows)
    lows_ascending = all(lows[j] < lows[j + 1] for j in range(len(lows) - 1))

    return highs_flat and lows_ascending

def is_flag_pattern(df, i, window=5):
    if i < window or i > len(df):
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs are ascending and lows are ascending
    highs_ascending = all(highs[j] < highs[j + 1] for j in range(len(highs) - 1))
    lows_ascending = all(lows[j] < lows[j + 1] for j in range(len(lows) - 1))

    return highs_ascending and lows_ascending

def is_double_bottom(df, i, window=10):
    if i < window or window < 3:
        return False

    lows = df["low"].iloc[i - window:i].values

    # Find indices of the two lowest points
    sorted_indices = lows.argsort()
    low1_idx, low2_idx = sorted(sorted_indices[:2])

    # Ensure there is a peak between the two lows
    if abs(low1_idx - low2_idx) < 2:
        return False

    first_low = lows[low1_idx]
    second_low = lows[low2_idx]
    peak = max(lows[min(low1_idx, low2_idx)+1:max(low1_idx, low2_idx)])

    # Check if lows are similar and separated by a peak
    return (
        abs(first_low - second_low) / first_low < 0.05 and
        peak > first_low and peak > second_low
    )

def is_rising_wedge(df, i, window=10):
    if i < window:
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs and lows are both ascending
    highs_ascending = all(highs[j] < highs[j + 1] for j in range(len(highs) - 1))
    lows_ascending = all(lows[j] < lows[j + 1] for j in range(len(lows) - 1))

    # In a rising wedge, highs and lows both rise, but highs rise less steeply (converging)
    if not (highs_ascending and lows_ascending):
        return False

    highs_slope = (highs[-1] - highs[0]) / window
    lows_slope = (lows[-1] - lows[0]) / window

    return highs_slope < lows_slope

def is_rising_triangle(df, i, window=10):
    if i < window or i > len(df):
        return False

    highs = df["high"].iloc[i - window:i].values
    lows = df["low"].iloc[i - window:i].values

    # Check if highs are ascending and lows are flat or slightly ascending
    highs_ascending = all(highs[j] < highs[j + 1] for j in range(len(highs) - 1))
    lows_flat_or_ascending = all(lows[j] <= lows[j + 1] for j in range(len(lows) - 1))

    return highs_ascending and lows_flat_or_ascending