from .base import Strategy
from indicators.features import (
    rr_for,
    get_bearish_indicators,
    get_bullish_indicators,
    dynamic_tp_sl,
)


class MyStrategy(Strategy):
    def evaluate(self, df):
        i = len(df) - 1
        return self._evaluate(df, i)

    def test_evaluate(self, df, i):
        return self._evaluate(df, i)
    
    def test_live(self, df):
        i = len(df) - 1
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i]) if "atr" in df.columns else 0.01
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1] if i - 1 >= 0 else None
        prev2 = df.iloc[i - 2] if i - 2 >= 0 else None
        volume_ma = (
            curr.get("volume_sma_20")
            if "volume_sma_20" in df.columns
            else df["volume"].rolling(20).mean().iloc[i]
        )

        # --- Get bullish + bearish pattern signals
        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(
            df, i, prev2, prev1, curr, volume_ma
        )
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(
            df, i, prev2, prev1, curr, volume_ma
        )

        # --- Regime detection (trend vs range)
        trend_strength = (
            curr["adx"] >= 20
            and abs(curr["ema_50"] - curr["ema_200"]) / entry >= 0.001
        )
        regime = "trend" if trend_strength else "range"

        # === Volatility filter ===
        curr_atr_pct = curr["atr"] / curr["close"]
        idx = curr.name
        atr_ma_pct = (df.loc[:idx, "atr"] / df.loc[:idx, "close"]).rolling(50).mean().iloc[-1]
        atr_ok = True
        if atr_ma_pct and atr_ma_pct > 0:
            atr_ok = 0.8 * atr_ma_pct <= curr_atr_pct <= 2.5 * atr_ma_pct

        # === Trade parameters ===
        if regime == "trend":
            atr_mult_tp = 3.0
            atr_mult_sl = 1.5
            min_rr = 1.6
        else:
            atr_mult_tp = 1.5
            atr_mult_sl = 1.0
            min_rr = 1.2

        tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
            entry,
            bear_sr,
            bull_sr,
            atr,
            atr_mult_sl=atr_mult_sl,
            atr_mult_tp=atr_mult_tp,
        )

        rr_long = rr_for("long", curr["close"], tp_long, sl_long)
        rr_short = rr_for("short", curr["close"], tp_short, sl_short)

        # === Trend setups ===
        trend_long = (
            curr["ema_20"] > curr["ema_50"] > curr["ema_200"]
            and curr["close"] > curr["ema_20"]
            and curr["macd_line"] > curr["macd_signal"]
            and 60 <= curr["rsi"]
            and curr["volume"] > volume_ma
        )
        trend_short = (
            curr["ema_20"] < curr["ema_50"] < curr["ema_200"]
            and curr["close"] < curr["ema_20"]
            and curr["macd_line"] < curr["macd_signal"]
            and curr["rsi"] <= 30
            and curr["volume"] > volume_ma
        )

        # === Range setups ===
        range_long = (
            curr["rsi"] <= 30
            and curr["close"] <= curr["bb_low"] * 1.01
            and curr["macd_line"] > curr["macd_signal"]
            and curr["ema_20"] < curr["close"]
        )
        range_short = (
            curr["rsi"] >= 60
            and curr["close"] >= curr["bb_high"] * 0.99
            and curr["macd_line"] < curr["macd_signal"]
            and curr["ema_20"] > curr["close"]
        )

        return {
                    "signal": "long",
                    "entry": entry,
                    "sl": sl_long,
                    "tp": tp_long,
                    "rr": rr_long,
                    "atr": curr["atr"],
                    "regime": regime,
                    "volatility_ok": atr_ok,
                    "has_pattern": bull_pattern,
                }

    def _evaluate(self, df, i):
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i]) if "atr" in df.columns else 0.01
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1] if i - 1 >= 0 else None
        prev2 = df.iloc[i - 2] if i - 2 >= 0 else None
        volume_ma = (
            curr.get("volume_sma_20")
            if "volume_sma_20" in df.columns
            else df["volume"].rolling(20).mean().iloc[i]
        )

        # --- Get bullish + bearish pattern signals
        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(
            df, i, prev2, prev1, curr, volume_ma
        )
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(
            df, i, prev2, prev1, curr, volume_ma
        )

        # --- Regime detection (trend vs range)
        trend_strength = (
            curr["adx"] >= 20
            and abs(curr["ema_50"] - curr["ema_200"]) / entry >= 0.001
        )
        regime = "trend" if trend_strength else "range"

        # === Volatility filter ===
        curr_atr_pct = curr["atr"] / curr["close"]
        idx = curr.name
        atr_ma_pct = (df.loc[:idx, "atr"] / df.loc[:idx, "close"]).rolling(50).mean().iloc[-1]
        atr_ok = True
        if atr_ma_pct and atr_ma_pct > 0:
            atr_ok = 0.8 * atr_ma_pct <= curr_atr_pct <= 2.5 * atr_ma_pct

        # === Trade parameters ===
        if regime == "trend":
            atr_mult_tp = 3.0
            atr_mult_sl = 1.5
            min_rr = 1.6
        else:
            atr_mult_tp = 1.5
            atr_mult_sl = 1.0
            min_rr = 1.2

        tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
            entry,
            bear_sr,
            bull_sr,
            atr,
            atr_mult_sl=atr_mult_sl,
            atr_mult_tp=atr_mult_tp,
        )

        rr_long = rr_for("long", curr["close"], tp_long, sl_long)
        rr_short = rr_for("short", curr["close"], tp_short, sl_short)

        # === Trend setups ===
        trend_long = (
            curr["ema_20"] > curr["ema_50"] > curr["ema_200"]
            and curr["close"] > curr["ema_20"]
            and curr["macd_line"] > curr["macd_signal"]
            and 60 <= curr["rsi"]
            and curr["volume"] > volume_ma
        )
        trend_short = (
            curr["ema_20"] < curr["ema_50"] < curr["ema_200"]
            and curr["close"] < curr["ema_20"]
            and curr["macd_line"] < curr["macd_signal"]
            and curr["rsi"] <= 30
            and curr["volume"] > volume_ma
        )

        # === Range setups ===
        range_long = (
            curr["rsi"] <= 30
            and curr["close"] <= curr["bb_low"] * 1.01
            and curr["macd_line"] > curr["macd_signal"]
            and curr["ema_20"] < curr["close"]
        )
        range_short = (
            curr["rsi"] >= 60
            and curr["close"] >= curr["bb_high"] * 0.99
            and curr["macd_line"] < curr["macd_signal"]
            and curr["ema_20"] > curr["close"]
        )

        # === Scoring ===
        def score_setup(has_pattern, indicators):
            score = 0.0
            score += 0.3 if atr_ok else 0.0
            score += 0.2 if curr["volume"] > volume_ma else 0.0
            score += 0.2 if has_pattern else 0.0
            score += 0.2 if indicators["rsi_divergence"] else 0.0
            score += 0.1 if indicators["flag_pattern"] or indicators["triangle_confluence"] else 0.0
            score += 0.1 if indicators["fib_bounce"] else 0.0
            return score

        if regime == "trend" and trend_long and rr_long >= min_rr:
            score = score_setup(bull_pattern, bull_ind)
            if score >= 0.7:
                return {
                    "signal": "long",
                    "entry": entry,
                    "sl": sl_long,
                    "tp": tp_long,
                    "rr": rr_long,
                    "atr": curr["atr"],
                    "regime": regime,
                    "score": score,
                    "volatility_ok": atr_ok,
                    "has_pattern": bull_pattern,
                }

        if regime == "trend" and trend_short and rr_short >= min_rr:
            score = score_setup(bear_pattern, bear_ind)
            if score >= 0.7:
                return {
                    "signal": "short",
                    "entry": entry,
                    "sl": sl_short,
                    "tp": tp_short,
                    "rr": rr_short,
                    "atr": curr["atr"],
                    "regime": regime,
                    "score": score,
                    "volatility_ok": atr_ok,
                    "has_pattern": bear_pattern,
                }

        if regime == "range" and range_long and rr_long >= min_rr:
            score = score_setup(bull_pattern, bull_ind)
            if score >= 0.7:
                return {
                    "signal": "long",
                    "entry": entry,
                    "sl": sl_long,
                    "tp": tp_long,
                    "rr": rr_long,
                    "atr": curr["atr"],
                    "regime": regime,
                    "score": score,
                    "volatility_ok": atr_ok,
                    "has_pattern": bull_pattern,
                }

        if regime == "range" and range_short and rr_short >= min_rr:
            score = score_setup(bear_pattern, bear_ind)
            if score >= 0.7:
                return {
                    "signal": "short",
                    "entry": entry,
                    "sl": sl_short,
                    "tp": tp_short,
                    "rr": rr_short,
                    "atr": curr["atr"],
                    "regime": regime,
                    "score": score,
                    "volatility_ok": atr_ok,
                    "has_pattern": bear_pattern,
                }

        return {"signal": None}