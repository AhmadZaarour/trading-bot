from .base import Strategy
from indicators.features import (
    rr_for,
    get_bullish_indicators,
    get_bearish_indicators,
    dynamic_tp_sl_long_oco,
    dynamic_tp_sl,
)

class AdaptiveSpotStrategy(Strategy):
    def __init__(
        self,
        min_rr: float = 1.6,
        adx_min: float = 23.0,
        vol_mult: float = 1.2,
        require_above_ema200: bool = True,
        require_trend: bool = True,
    ):
        self.min_rr = min_rr
        self.adx_min = adx_min
        self.vol_mult = vol_mult
        self.require_above_ema200 = require_above_ema200
        self.require_trend = require_trend

    def evaluate(self, df):
        i = len(df) - 1
        return self._evaluate(df, i)

    def test_evaluate(self, df, i):
        return self._evaluate(df, i)

    def _evaluate(self, df, i):
        if i < 205:  # needs ema_200 + slope window
            return {"signal": None}

        entry = float(df["close"].iloc[i])
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1]
        prev5 = df.iloc[i - 5]

        atr = float(curr["atr"]) if "atr" in df.columns else entry * 0.002

        volume_ma = curr.get("volume_sma_20", df["volume"].rolling(20).mean().iloc[i])
        vol_ok = curr["volume"] >= self.vol_mult * volume_ma

        # Trend / regime
        ema_gap = abs(curr["ema_50"] - curr["ema_200"]) / max(entry, 1e-12)
        ema_slope = (curr["ema_50"] - prev5["ema_50"]) / max(entry, 1e-12)

        trend_strength = (curr["adx"] >= self.adx_min) and (ema_gap >= 0.0015) and (ema_slope > 0)
        regime = "trend" if trend_strength else "range"

        if self.require_trend and regime != "trend":
            return {"signal": None}

        if self.require_above_ema200 and curr["close"] <= curr["ema_200"]:
            return {"signal": None}

        # Indicators + SR
        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(
            df, i, df.iloc[i - 2], prev1, curr, volume_ma
        )
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(
            df, i, df.iloc[i - 2], prev1, curr, volume_ma
        )

        # ATR multipliers (slightly wider for wicky 1h alts)
        atr_mult_tp = 3.0
        atr_mult_sl = 1.7

        tp, stop, stop_limit = dynamic_tp_sl_long_oco(
            entry,
            bear_sr,   # resistances
            bull_sr,   # supports
            atr,
            atr_mult_sl=atr_mult_sl,
            atr_mult_tp=atr_mult_tp,
            # if your helper supports these, set them wider for $5–$15 coins:
            tp_buffer=0.002,          # 0.2% below resistance
            sl_buffer=0.002,          # 0.2% below support
            stop_limit_offset=0.002,  # 0.2% below stop
        )

        # sanity + RR
        if stop >= entry or tp <= entry:
            return {"signal": None}

        rr_long = rr_for("long", curr["close"], tp, stop)
        if rr_long < self.min_rr:
            return {"signal": None}

        macd_hist = curr["macd_hist"]
        macd_hist_prev = prev1["macd_hist"]

        # Trend continuation (tighter)
        trend_long = (
            curr["ema_20"] > curr["ema_50"] > curr["ema_200"]
            and curr["close"] > curr["ema_20"]
            and curr["macd_line"] > curr["macd_signal"]
            and macd_hist > 0
            and curr["rsi"] >= 55
            and ema_slope > 0
            and vol_ok
        )

        # Breakout: require a real cross + volume
        bb_width = curr.get("bb_width", 0.0)
        idx = curr.name
        bb_width_ma = df.loc[:idx, "bb_width"].rolling(50).mean().iloc[-1] if "bb_width" in df.columns else None

        breakout_long = (
            bb_width_ma
            and bb_width > bb_width_ma * 1.3
            and prev1["close"] <= prev1["bb_high"]
            and curr["close"] > curr["bb_high"]
            and curr["rsi"] >= 60
            and vol_ok
        )

        # Score (keep yours but make volume meaningful)
        def score_setup(has_pattern, indicators):
            score = 0.0
            score += 0.25 if vol_ok else 0.0
            score += 0.20 if has_pattern else 0.0
            score += 0.15 if indicators.get("rsi_divergence") else 0.0
            score += 0.10 if indicators.get("flag_pattern") or indicators.get("triangle_confluence") else 0.0
            score += 0.10 if indicators.get("fib_bounce") else 0.0
            score += 0.20 if trend_strength else 0.0
            return score

        if trend_long or breakout_long:
            score = score_setup(bull_pattern, bull_ind)
            if score >= 0.70:
                return {
                    "signal": "long",
                    "entry": entry,
                    "sl": stop,
                    "tp": tp,
                    "stop_limit": stop_limit,
                    "rr": rr_long,
                    "atr": curr["atr"],
                    "regime": regime,
                    "score": score,
                }

        return {"signal": None}

class AdaptiveFuturesStrategy(Strategy):
    def __init__(
        self,
        min_rr: float = 1.5,
        adx_min: float = 25.0,
        vol_mult: float = 1.3,          # require volume >= vol_mult * volume_ma
        score_min: float = 0.7,        # stricter than 0.65
        allow_range: bool = True,       # set False to trend/breakout only
    ):
        self.min_rr = min_rr
        self.adx_min = adx_min
        self.vol_mult = vol_mult
        self.score_min = score_min
        self.allow_range = allow_range

    def evaluate(self, df):
        i = len(df) - 1
        return self._evaluate(df, i)

    def test_evaluate(self, df, i):
        return self._evaluate(df, i)

    def _evaluate(self, df, i):
        # Need enough history for ema_200 and slope windows
        if i < 205:
            return {"signal": None}

        entry = float(df["close"].iloc[i])
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1]
        prev5 = df.iloc[i - 5]

        atr = float(curr["atr"]) if "atr" in df.columns else entry * 0.002

        volume_ma = (
            curr.get("volume_sma_20")
            if "volume_sma_20" in df.columns
            else df["volume"].rolling(20).mean().iloc[i]
        )
        vol_ok = curr["volume"] >= self.vol_mult * volume_ma

        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(
            df, i, df.iloc[i - 2], prev1, curr, volume_ma
        )
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(
            df, i, df.iloc[i - 2], prev1, curr, volume_ma
        )

        # Trend strength (stricter)
        ema_gap = abs(curr["ema_50"] - curr["ema_200"]) / max(entry, 1e-12)
        ema_slope = (curr["ema_50"] - prev5["ema_50"]) / max(entry, 1e-12)

        trend_strength = (
            curr["adx"] >= self.adx_min
            and ema_gap >= 0.0015
            and abs(ema_slope) >= 0.0004
        )
        regime = "trend" if trend_strength else "range"

        # Volatility sanity (keep your idea, slightly tighter bounds)
        idx = curr.name
        curr_atr_pct = curr["atr"] / max(curr["close"], 1e-12)
        atr_ma_pct = (df.loc[:idx, "atr"] / df.loc[:idx, "close"]).rolling(50).mean().iloc[-1]
        atr_ok = True
        if atr_ma_pct and atr_ma_pct > 0:
            atr_ok = 0.7 * atr_ma_pct <= curr_atr_pct <= 2.5 * atr_ma_pct

        # Dynamic ATR multipliers
        # (slightly wider SL in high-vol regimes)
        volatility_regime = "high" if atr_ma_pct and curr_atr_pct > 1.25 * atr_ma_pct else "normal"
        if regime == "trend":
            atr_mult_tp = 3.0 if volatility_regime == "normal" else 2.6
            atr_mult_sl = 1.7 if volatility_regime == "normal" else 2.0
            min_rr = max(self.min_rr, 1.7)
        else:
            atr_mult_tp = 1.7
            atr_mult_sl = 1.2
            min_rr = max(self.min_rr, 1.35)

        # TP/SL (long & short)
        tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
            entry,
            bear_sr,  # resistances
            bull_sr,  # supports
            atr,
            atr_mult_sl=atr_mult_sl,
            atr_mult_tp=atr_mult_tp,
        )

        # basic sanity
        if sl_long >= entry or tp_long <= entry:
            rr_long = -1e9
        else:
            rr_long = rr_for("long", curr["close"], tp_long, sl_long)

        if sl_short <= entry or tp_short >= entry:
            rr_short = -1e9
        else:
            rr_short = rr_for("short", curr["close"], tp_short, sl_short)

        macd_hist = curr["macd_hist"]
        macd_hist_prev = prev1["macd_hist"]

        # Directional bias: helps prevent fighting the bigger trend
        long_bias_ok = curr["close"] > curr["ema_200"] and ema_slope > 0
        short_bias_ok = curr["close"] < curr["ema_200"] and ema_slope < 0

        # Trend continuation setups (tighter)
        trend_long = (
            long_bias_ok
            and curr["ema_20"] > curr["ema_50"] > curr["ema_200"]
            and curr["close"] > curr["ema_20"]
            and curr["macd_line"] > curr["macd_signal"]
            and macd_hist > 0
            and curr["rsi"] >= 65
            and vol_ok
        )
        trend_short = (
            short_bias_ok
            and curr["ema_20"] < curr["ema_50"] < curr["ema_200"]
            and curr["close"] < curr["ema_20"]
            and curr["macd_line"] < curr["macd_signal"]
            and macd_hist < 0
            and curr["rsi"] <= 30
            and vol_ok
        )

        # Breakouts: require real cross + squeeze expansion + volume
        bb_width = curr.get("bb_width", 0.0)
        bb_width_ma = df.loc[:idx, "bb_width"].rolling(50).mean().iloc[-1] if "bb_width" in df.columns else None

        breakout_long = (
            bb_width_ma
            and bb_width > bb_width_ma * 1.3
            and prev1["close"] <= prev1["bb_high"]
            and curr["close"] > curr["ema_20"]
            and curr["close"] > curr["bb_high"]
            and curr["rsi"] >= 70
            and vol_ok
            and long_bias_ok
        )
        breakout_short = (
            bb_width_ma
            and bb_width > bb_width_ma * 1.3
            and prev1["close"] >= prev1["bb_low"]
            and curr["close"] < curr["ema_20"]
            and curr["close"] < curr["bb_low"]
            and curr["rsi"] <= 30
            and vol_ok
            and short_bias_ok
        )

        # Range mean-reversion (optional, tighter)
        range_long = (
            self.allow_range
            and regime == "range"
            and curr["rsi"] <= 33
            and curr["close"] <= curr["bb_low"] * 1.003
            and macd_hist > macd_hist_prev
            and curr["close"] > curr["ema_20"]  # reduces catching falling knives
        )
        range_short = (
            self.allow_range
            and regime == "range"
            and curr["rsi"] >= 67
            and curr["close"] >= curr["bb_high"] * 0.997
            and macd_hist < macd_hist_prev
            and curr["close"] < curr["ema_20"]
        )

        def score_setup(has_pattern, indicators, is_trend: bool):
            score = 0.0
            score += 0.25 if atr_ok else 0.0
            score += 0.25 if vol_ok else 0.0
            score += 0.20 if has_pattern else 0.0
            #score += 0.15 if indicators.get("rsi_divergence") else 0.0
            score += 0.10 if indicators.get("flag_pattern") or indicators.get("triangle_confluence") else 0.0
            score += 0.05 if indicators.get("fib_bounce") else 0.0
            score += 0.10 if is_trend else 0.0
            return score

        candidates = []

        # LONG candidates
        if (trend_long or breakout_long) and rr_long >= min_rr:
            score = score_setup(bull_pattern, bull_ind, is_trend=True)
            if score >= self.score_min:
                candidates.append(
                    self._trade_filler("long", curr["rsi"], curr["ema_20"],
                      volume_ma, curr["atr"], ema_gap, 
                      ema_slope, entry, curr["close"], 
                      sl_long, tp_long, rr_long, 
                      regime, score, atr_ok, bull_pattern)
                )

        # SHORT candidates
        if (trend_short or breakout_short) and rr_short >= min_rr:
            score = score_setup(bear_pattern, bear_ind, is_trend=True)
            if score >= self.score_min:
                candidates.append(
                    self._trade_filler("short", curr["rsi"], curr["ema_20"],
                      volume_ma, curr["atr"], ema_gap, 
                      ema_slope, entry, curr["close"], 
                      sl_short, tp_short, rr_short, 
                      regime, score, atr_ok, bear_pattern)
                )

        # Range (optional)
        if range_long and rr_long >= min_rr and bb_width_ma and bb_width < bb_width_ma * 1.05:
            score = score_setup(bull_pattern, bull_ind, is_trend=False)
            if score >= self.score_min:
                candidates.append(
                    self._trade_filler("long", curr["rsi"], curr["ema_20"],
                      volume_ma, curr["atr"], ema_gap, 
                      ema_slope, entry, curr["close"], 
                      sl_long, tp_long, rr_long, 
                      regime, score, atr_ok, bull_pattern)
                )

        if range_short and rr_short >= min_rr and bb_width_ma and bb_width < bb_width_ma * 1.05:
            score = score_setup(bear_pattern, bear_ind, is_trend=False)
            if score >= self.score_min:
                candidates.append(
                    self._trade_filler("short", curr["rsi"], curr["ema_20"],
                      volume_ma, curr["atr"], ema_gap, 
                      ema_slope, entry, curr["close"], 
                      sl_short, tp_short, rr_short, 
                      regime, score, atr_ok, bear_pattern)
                )

        if not candidates:
            return {"signal": None}

        return max(candidates, key=lambda t: (t["score"], t.get("rr", 0.0)))
    
    def _trade_filler(self, signal, rsi, ema20,
                      volume_ma, atr, ema_gap, 
                      ema_slope, entry, close, 
                      sl, tp, rr, regime, score, 
                      atr_ok, pattern) -> dict:
        trade = {
            "signal": signal,
            "rsi": rsi,
            "ema20": ema20,
            "volume_ma": volume_ma,
            "atr": atr,
            "ema_gap": ema_gap,
            "ema_slope": ema_slope,
            "entry": entry,
            "close": close,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "regime": regime,
            "score": score,
            "atr_ok": atr_ok,
            "pattern": pattern,
        }
        return trade