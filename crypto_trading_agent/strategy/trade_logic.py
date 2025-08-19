from .logic_long import *
from .logic_short import *

def analyze_row(prev2, prev1, curr, volume_ma, i, df):
    has_bullish_pattern, bullish_indicators, s_levels = bullish_signal(prev2, prev1, curr, volume_ma, i, df)
    has_bearish_pattern, bearish_indicators, r_levels = bearish_signal(prev2, prev1, curr, volume_ma, i, df)
    threshold = 9.5

    # Both indicators present
    if bullish_indicators and bearish_indicators:
        bullish_score_value = bullish_score(bullish_indicators, has_bullish_pattern)
        bearish_score_value = bearish_score(bearish_indicators, has_bearish_pattern)

        # Both patterns present: go with higher score
        if has_bullish_pattern and has_bearish_pattern:
            if bullish_score_value >= bearish_score_value and bullish_score_value >= threshold:
                entry = curr["close"]
                signal = "long"

                tp = r_levels[0] if r_levels and entry * 1.04 > r_levels[0] > entry * 1.02 else entry * 1.02
                sl = s_levels[0] if s_levels and entry * 0.98 < s_levels[0] < entry * 0.99 else entry * 0.99

                return long_trade(signal, entry, tp, sl, bullish_indicators, has_bullish_pattern, bullish_score_value)
            
            elif bearish_score_value > bullish_score_value and bearish_score_value >= threshold:
                entry = curr["close"]
                signal = "short"

                sl = r_levels[0] if r_levels and entry * 1.01 < r_levels[0] < entry * 1.04 else entry * 1.01
                tp = s_levels[0] if s_levels and entry * 0.98 > s_levels[0] > entry * 0.96 else entry * 0.98

                return short_trade(signal, entry, tp, sl, bearish_indicators, has_bearish_pattern, bearish_score_value)
            
            else:
                return {"signal": None}
        # Only bearish pattern present, but both indicators: prefer short if bearish score above threshold
        elif has_bearish_pattern and bearish_score_value >= threshold:
            entry = curr["close"]
            signal = "short"

            sl = r_levels[0] if r_levels and entry * 1.01 < r_levels[0] < entry * 1.04 else entry * 1.01
            tp = s_levels[0] if s_levels and entry * 0.98 > s_levels[0] > entry * 0.96 else entry * 0.98

            return short_trade(signal, entry, tp, sl, bearish_indicators, has_bearish_pattern, bearish_score_value)
        
        # Only bullish pattern present, but both indicators: prefer long if bullish score above threshold
        elif has_bullish_pattern and bullish_score_value >= threshold:
            entry = curr["close"]
            signal = "long"

            tp = r_levels[0] if r_levels and entry * 1.04 > r_levels[0] > entry * 1.02 else entry * 1.02
            sl = s_levels[0] if s_levels and entry * 0.98 < s_levels[0] < entry * 0.99 else entry * 0.99

            return long_trade(signal, entry, tp, sl, bullish_indicators, has_bullish_pattern, bullish_score_value)
        
        # No patterns, just indicators: go with higher score as before
        elif bullish_score_value >= bearish_score_value and bullish_score_value >= threshold:
            entry = curr["close"]
            signal = "long"

            tp = r_levels[0] if r_levels and entry * 1.04 > r_levels[0] > entry * 1.02 else entry * 1.02
            sl = s_levels[0] if s_levels and entry * 0.98 < s_levels[0] < entry * 0.99 else entry * 0.99

            return long_trade(signal, entry, tp, sl, bullish_indicators, has_bullish_pattern, bullish_score_value)
        
        elif bearish_score_value > bullish_score_value and bearish_score_value >= threshold:
            entry = curr["close"]
            signal = "short"

            sl = r_levels[0] if r_levels and entry * 1.01 < r_levels[0] < entry * 1.04 else entry * 1.01
            tp = s_levels[0] if s_levels and entry * 0.98 > s_levels[0] > entry * 0.96 else entry * 0.98

            return short_trade(signal, entry, tp, sl, bearish_indicators, has_bearish_pattern, bearish_score_value)
        
        else:
            return {"signal": None}

    # Only bullish indicators
    elif bullish_indicators:
        score = bullish_score(bullish_indicators, has_bullish_pattern)
        if score >= threshold:
            entry = curr["close"]
            signal = "long"

            tp = r_levels[0] if r_levels and entry * 1.04 > r_levels[0] > entry * 1.02 else entry * 1.02
            sl = s_levels[0] if s_levels and entry * 0.98 < s_levels[0] < entry * 0.99 else entry * 0.99

            return long_trade(signal, entry, tp, sl, bullish_indicators, has_bullish_pattern, score)
        
        else:
            return {"signal": None}

    # Only bearish indicators
    elif bearish_indicators:
        score = bearish_score(bearish_indicators, has_bearish_pattern)
        if score >= threshold:
            entry = curr["close"]
            signal = "short"

            sl = r_levels[0] if r_levels and entry * 1.01 < r_levels[0] < entry * 1.04 else entry * 1.01
            tp = s_levels[0] if s_levels and entry * 0.98 > s_levels[0] > entry * 0.96 else entry * 0.98

            return short_trade(signal, entry, tp, sl, bearish_indicators, has_bearish_pattern, score)
        
        else:
            return {"signal": None}

    # No indicators
    else:
        return {"signal": None}