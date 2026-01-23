from .base import Strategy
from indicators.features import (
    rr_for,
    get_bearish_indicators,
    get_bullish_indicators,
    dynamic_tp_sl,
    dynamic_tp_sl_long_oco,
)


class AdaptiveFuturesStrategy(Strategy):
    def __init__(self, min_rr: float = 1.5):
        self.min_rr = min_rr

    def evaluate(self, df):
        i = len(df) - 1
        return self._evaluate(df, i)

    def test_evaluate(self, df, i):
        return self._evaluate(df, i)

    def _evaluate(self, df, i):
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i]) if "atr" in df.columns else entry * 0.002
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1] if i - 1 >= 0 else None
        prev5 = df.iloc[i - 5] if i - 5 >= 0 else None

        volume_ma = (
            curr.get("volume_sma_20")
            if "volume_sma_20" in df.columns
            else df["volume"].rolling(20).mean().iloc[i]
        )

        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(
            df, i, df.iloc[i - 2] if i - 2 >= 0 else None, prev1, curr, volume_ma
        )
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(
            df, i, df.iloc[i - 2] if i - 2 >= 0 else None, prev1, curr, volume_ma
        )

        ema_gap = abs(curr["ema_50"] - curr["ema_200"]) / entry
        ema_slope = 0.0
        if prev5 is not None:
            ema_slope = (curr["ema_50"] - prev5["ema_50"]) / entry

        trend_strength = curr["adx"] >= 18 and ema_gap >= 0.001
        regime = "trend" if trend_strength else "range"

        bb_width = curr.get("bb_width", 0.0)
        idx = curr.name
        bb_width_ma = df.loc[:idx, "bb_width"].rolling(50).mean().iloc[-1]

        curr_atr_pct = curr["atr"] / curr["close"]
        atr_ma_pct = (df.loc[:idx, "atr"] / df.loc[:idx, "close"]).rolling(50).mean().iloc[-1]
        atr_ok = True
        if atr_ma_pct and atr_ma_pct > 0:
            atr_ok = 0.7 * atr_ma_pct <= curr_atr_pct <= 2.8 * atr_ma_pct

        volatility_regime = "high" if atr_ma_pct and curr_atr_pct > 1.2 * atr_ma_pct else "normal"

        if regime == "trend":
            atr_mult_tp = 3.2 if volatility_regime == "normal" else 2.6
            atr_mult_sl = 1.6 if volatility_regime == "normal" else 1.9
            min_rr = max(self.min_rr, 1.7)
        else:
            atr_mult_tp = 1.6
            atr_mult_sl = 1.1
            min_rr = max(self.min_rr, 1.3)

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

        macd_hist = curr["macd_hist"]
        macd_hist_prev = prev1["macd_hist"] if prev1 is not None else macd_hist

        trend_long = (
            curr["ema_20"] > curr["ema_50"] > curr["ema_200"]
            and curr["close"] > curr["ema_20"]
            and curr["macd_line"] > curr["macd_signal"]
            and macd_hist > 0
            and curr["rsi"] >= 55
            and curr["volume"] > volume_ma
            and ema_slope > 0
        )
        trend_short = (
            curr["ema_20"] < curr["ema_50"] < curr["ema_200"]
            and curr["close"] < curr["ema_20"]
            and curr["macd_line"] < curr["macd_signal"]
            and macd_hist < 0
            and curr["rsi"] <= 45
            and curr["volume"] > volume_ma
            and ema_slope < 0
        )

        range_long = (
            curr["rsi"] <= 35
            and curr["close"] <= curr["bb_low"] * 1.005
            and macd_hist > macd_hist_prev
            and curr["ema_20"] < curr["close"]
        )
        range_short = (
            curr["rsi"] >= 65
            and curr["close"] >= curr["bb_high"] * 0.995
            and macd_hist < macd_hist_prev
            and curr["ema_20"] > curr["close"]
        )

        breakout_long = (
            bb_width_ma
            and bb_width > bb_width_ma * 1.2
            and curr["close"] > curr["bb_high"]
            and curr["rsi"] >= 60
            and curr["volume"] > volume_ma
        )
        breakout_short = (
            bb_width_ma
            and bb_width > bb_width_ma * 1.2
            and curr["close"] < curr["bb_low"]
            and curr["rsi"] <= 40
            and curr["volume"] > volume_ma
        )

        def score_setup(has_pattern, indicators):
            score = 0.0
            score += 0.25 if atr_ok else 0.0
            score += 0.2 if curr["volume"] > volume_ma else 0.0
            score += 0.2 if has_pattern else 0.0
            score += 0.15 if indicators["rsi_divergence"] else 0.0
            score += 0.1 if indicators["flag_pattern"] or indicators["triangle_confluence"] else 0.0
            score += 0.1 if indicators["fib_bounce"] else 0.0
            score += 0.1 if trend_strength else 0.0
            return score

        candidates = []

        if (regime == "trend" and trend_long) or breakout_long:
            if rr_long >= min_rr:
                score = score_setup(bull_pattern, bull_ind)
                if score >= 0.65:
                    candidates.append(
                        {
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
                    )

        if (regime == "trend" and trend_short) or breakout_short:
            if rr_short >= min_rr:
                score = score_setup(bear_pattern, bear_ind)
                if score >= 0.65:
                    candidates.append(
                        {
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
                    )

        if regime == "range" and range_long and rr_long >= min_rr:
            score = score_setup(bull_pattern, bull_ind)
            if score >= 0.65 and bb_width_ma and bb_width < bb_width_ma * 1.1:
                candidates.append(
                    {
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
                )

        if regime == "range" and range_short and rr_short >= min_rr:
            score = score_setup(bear_pattern, bear_ind)
            if score >= 0.65 and bb_width_ma and bb_width < bb_width_ma * 1.1:
                candidates.append(
                    {
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
                )

        if not candidates:
            return {"signal": None}

        return max(candidates, key=lambda trade: (trade["score"], trade.get("rr", 0)))


class AdaptiveSpotStrategy(Strategy):
    def __init__(self, min_rr: float = 1.4, require_trend: bool = True):
        self.min_rr = min_rr
        self.require_trend = require_trend

    def evaluate(self, df):
        i = len(df) - 1
        return self._evaluate(df, i)

    def test_evaluate(self, df, i):
        return self._evaluate(df, i)

    def _evaluate(self, df, i):
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i]) if "atr" in df.columns else entry * 0.002
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1] if i - 1 >= 0 else None
        prev5 = df.iloc[i - 5] if i - 5 >= 0 else None

        volume_ma = (
            curr.get("volume_sma_20")
            if "volume_sma_20" in df.columns
            else df["volume"].rolling(20).mean().iloc[i]
        )

        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(
            df, i, df.iloc[i - 2] if i - 2 >= 0 else None, prev1, curr, volume_ma
        )
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(
            df, i, df.iloc[i - 2] if i - 2 >= 0 else None, prev1, curr, volume_ma
        )

        ema_gap = abs(curr["ema_50"] - curr["ema_200"]) / entry
        ema_slope = 0.0
        if prev5 is not None:
            ema_slope = (curr["ema_50"] - prev5["ema_50"]) / entry

        trend_strength = curr["adx"] >= 18 and ema_gap >= 0.001
        regime = "trend" if trend_strength else "range"

        bb_width = curr.get("bb_width", 0.0)
        idx = curr.name
        bb_width_ma = df.loc[:idx, "bb_width"].rolling(50).mean().iloc[-1]

        curr_atr_pct = curr["atr"] / curr["close"]
        atr_ma_pct = (df.loc[:idx, "atr"] / df.loc[:idx, "close"]).rolling(50).mean().iloc[-1]
        atr_ok = True
        if atr_ma_pct and atr_ma_pct > 0:
            atr_ok = 0.7 * atr_ma_pct <= curr_atr_pct <= 2.5 * atr_ma_pct

        if regime == "trend":
            atr_mult_tp = 2.8
            atr_mult_sl = 1.4
            min_rr = max(self.min_rr, 1.6)
        else:
            atr_mult_tp = 1.5
            atr_mult_sl = 1.0
            min_rr = max(self.min_rr, 1.3)

        tp, stop, stop_limit = dynamic_tp_sl_long_oco(
            entry,
            bear_sr,
            bull_sr,
            atr,
            atr_mult_sl=atr_mult_sl,
            atr_mult_tp=atr_mult_tp,
        )

        rr_long = rr_for("long", curr["close"], tp, stop)

        macd_hist = curr["macd_hist"]
        macd_hist_prev = prev1["macd_hist"] if prev1 is not None else macd_hist

        trend_long = (
            curr["ema_20"] > curr["ema_50"] > curr["ema_200"]
            and curr["close"] > curr["ema_20"]
            and curr["macd_line"] > curr["macd_signal"]
            and macd_hist > 0
            and curr["rsi"] >= 55
            and curr["volume"] > volume_ma
            and ema_slope > 0
        )

        range_long = (
            curr["rsi"] <= 35
            and curr["close"] <= curr["bb_low"] * 1.005
            and macd_hist > macd_hist_prev
            and curr["ema_20"] < curr["close"]
        )

        breakout_long = (
            bb_width_ma
            and bb_width > bb_width_ma * 1.2
            and curr["close"] > curr["bb_high"]
            and curr["rsi"] >= 60
            and curr["volume"] > volume_ma
        )

        def score_setup(has_pattern, indicators):
            score = 0.0
            score += 0.25 if atr_ok else 0.0
            score += 0.2 if curr["volume"] > volume_ma else 0.0
            score += 0.2 if has_pattern else 0.0
            score += 0.15 if indicators["rsi_divergence"] else 0.0
            score += 0.1 if indicators["flag_pattern"] or indicators["triangle_confluence"] else 0.0
            score += 0.1 if indicators["fib_bounce"] else 0.0
            score += 0.1 if trend_strength else 0.0
            return score

        candidates = []

        if ((regime == "trend" and trend_long) or breakout_long) and rr_long >= min_rr:
            score = score_setup(bull_pattern, bull_ind)
            if score >= 0.65:
                candidates.append(
                    {
                        "signal": "long",
                        "entry": entry,
                        "sl": stop,
                        "tp": tp,
                        "stop_limit": stop_limit,
                        "rr": rr_long,
                        "atr": curr["atr"],
                        "regime": regime,
                        "score": score,
                        "volatility_ok": atr_ok,
                        "has_pattern": bull_pattern,
                    }
                )

        if not self.require_trend and regime == "range" and range_long and rr_long >= min_rr:
            score = score_setup(bull_pattern, bull_ind)
            if score >= 0.65 and bb_width_ma and bb_width < bb_width_ma * 1.1:
                candidates.append(
                    {
                        "signal": "long",
                        "entry": entry,
                        "sl": stop,
                        "tp": tp,
                        "stop_limit": stop_limit,
                        "rr": rr_long,
                        "atr": curr["atr"],
                        "regime": regime,
                        "score": score,
                        "volatility_ok": atr_ok,
                        "has_pattern": bull_pattern,
                    }
                )

        if trend_strength and bear_pattern and bear_ind.get("rsi_divergence"):
            candidates.append(
                {
                    "signal": "short",
                    "entry": entry,
                    "atr": curr["atr"],
                    "regime": regime,
                    "score": 0.55,
                    "volatility_ok": atr_ok,
                    "has_pattern": bear_pattern,
                }
            )

        if not candidates:
            return {"signal": None}

        return max(candidates, key=lambda trade: (trade["score"], trade.get("rr", 0)))
