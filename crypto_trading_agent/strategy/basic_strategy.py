from .base import Strategy
from indicators.features import rr_for, get_bearish_indicators, get_bullish_indicators, has_resistance, has_support, dynamic_tp_sl

class MyStrategy(Strategy):
    def evaluate(self, df):
        i = len(df) - 1
        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i]) if "atr" in df.columns else 0.01
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1] if i - 1 >= 0 else None
        prev2 = df.iloc[i - 2] if i - 2 >= 0 else None
        volume_ma = df["volume"].rolling(20).mean().iloc[i] if "volume" in df.columns else 0

        # --- Get bullish + bearish pattern signals
        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma)
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma)

        tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
            entry,
            bear_sr,
            bull_sr,
            atr
        )

        trade = None
        # --- Long/short trade ---
        rr_long  = rr_for("long",  curr["close"], tp_long,  sl_long)
        rr_short = rr_for("short", curr["close"], tp_short, sl_short)


        # === 1. Trend filter ===
        trend_long = curr["ema_50"] > curr["ema_200"] #and (curr["ema_50"] - curr["ema_200"]) > 0.1
        trend_short = curr["ema_50"] < curr["ema_200"] #and (curr["ema_50"] - curr["ema_200"]) < -0.1

        # === 2. Momentum filter ===
        momentum_long = (curr["macd_line"] > curr["macd_signal"] and curr["rsi"] >= 50) or (curr["rsi"] < 30 and curr["macd_line"] > curr["macd_signal"])
        momentum_short = (curr["macd_line"] < curr["macd_signal"] and curr["rsi"] < 50) or (curr["rsi"] > 70 and curr["macd_line"] < curr["macd_signal"])


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
        if trend_long and momentum_long and rr_long_ok:
            score = (
                bull_ind["flag_pattern"] * 0.75
                + bull_ind["fib_bounce"] * 0.25
                + bull_ind["triangle_confluence"] * 0.5
                + bull_ind["rising_triangle"] * 0.75
                + bull_ind["rising_wedge"] * 0.75
                + bull_ind["rsi_divergence"] * 0.5
                + atr_ok * 0.5
            )
            if score >= 0.5:
                
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
        elif trend_short and momentum_short and rr_short_ok:
            score = (
                bear_ind["flag_pattern"] * 0.75
                + bear_ind["fib_bounce"] * 0.25
                + bear_ind["triangle_confluence"] * 0.5
                + bear_ind["falling_triangle"] * 0.75
                + bear_ind["falling_wedge"] * 0.75
                + bear_ind["rsi_divergence"] * 0.5
                + atr_ok * 0.5
            )
            if score >= 0.5:
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
        else:
            trade = {
                "signal": None
            }

        return trade

    def test_evaluate(self, df, i):

        entry = float(df["close"].iloc[i])
        atr = float(df["atr"].iloc[i]) if "atr" in df.columns else 0.01
        curr = df.iloc[i]
        prev1 = df.iloc[i - 1] if i - 1 >= 0 else None
        prev2 = df.iloc[i - 2] if i - 2 >= 0 else None
        volume_ma = df["volume"].rolling(20).mean().iloc[i] if "volume" in df.columns else 0

        # --- Get bullish + bearish pattern signals
        bull_pattern, bull_ind, bull_sr = get_bullish_indicators(df, i, prev2, prev1, curr, volume_ma)
        bear_pattern, bear_ind, bear_sr = get_bearish_indicators(df, i, prev2, prev1, curr, volume_ma)

        tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(
            entry,
            bear_sr,
            bull_sr,
            atr
        )

        trade = None
        # --- Long/short trade ---
        rr_long  = rr_for("long",  curr["close"], tp_long,  sl_long)
        rr_short = rr_for("short", curr["close"], tp_short, sl_short)


        # === 1. Trend filter ===
        trend_long = curr["ema_50"] > curr["ema_200"] #and (curr["ema_50"] - curr["ema_200"]) > 0.1
        trend_short = curr["ema_50"] < curr["ema_200"] #and (curr["ema_50"] - curr["ema_200"]) < -0.1

        # === 2. Momentum filter ===
        momentum_long = (curr["macd_line"] > curr["macd_signal"] and curr["rsi"] >= 50) or (curr["rsi"] < 30 and curr["macd_line"] > curr["macd_signal"])
        momentum_short = (curr["macd_line"] < curr["macd_signal"] and curr["rsi"] < 50) or (curr["rsi"] > 70 and curr["macd_line"] < curr["macd_signal"])


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
        if trend_long and momentum_long and rr_long_ok:
            score = (
                bull_ind["flag_pattern"] * 0.75
                + bull_ind["fib_bounce"] * 0.25
                + bull_ind["triangle_confluence"] * 0.5
                + bull_ind["rising_triangle"] * 0.75
                + bull_ind["rising_wedge"] * 0.75
                + bull_ind["rsi_divergence"] * 0.5
                + atr_ok * 0.5
            )
            if score >= 0.5:
                
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
        elif trend_short and momentum_short and rr_short_ok:
            score = (
                bear_ind["flag_pattern"] * 0.75
                + bear_ind["fib_bounce"] * 0.25
                + bear_ind["triangle_confluence"] * 0.5
                + bear_ind["falling_triangle"] * 0.75
                + bear_ind["falling_wedge"] * 0.75
                + bear_ind["rsi_divergence"] * 0.5
                + atr_ok * 0.5
            )
            if score >= 0.5:
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
        else:
            trade = {
                "signal": None
            }

        return trade
