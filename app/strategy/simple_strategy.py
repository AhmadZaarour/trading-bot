from .base import Strategy
from indicators.features import (
    rr_for,
    get_bearish_indicators,
    get_bullish_indicators,
    dynamic_tp_sl,
)


class SimpleStrategy(Strategy):
    def evaluate(self, df):
        i = len(df) - 1
        return self._evaluate(df, i)

    def test_evaluate(self, df, i):
        return self._evaluate(df, i)
    

    def _evaluate(self, df, i):
        curr = df.iloc[i]
        entry = float(curr["close"])
        close = float(curr["close"])
        atr = float(curr["atr"])
        prev1 = df.iloc[i-1] if i-1 >= 0 else None
        prev2 = df.iloc[i-2] if i-2 >= 0 else None
        volume_ma = df["volume_sma_20"].iloc[i]

        bullish = get_bullish_indicators(df, i, prev1, prev2, curr, volume_ma)
        bearish = get_bearish_indicators(df, i, prev1, prev2, curr, volume_ma)

        pattern_bull, ind_bull, r_lvls = bullish
        pattern_bear, ind_bear, s_lvls = bearish

        tp_long, sl_long, tp_short, sl_short = dynamic_tp_sl(entry, r_lvls, s_lvls, atr)

        rsi = curr["rsi"]
        ema20 = curr["ema_20"]
        fib_bounce = ind_bull.get("fib_bounce")
        fib_reject = ind_bear.get("fib_bounce")

        long_momentum = rsi > 50 and rsi < 70
        short_momentum = rsi < 50 and rsi > 30

        if close > ema20 and fib_bounce and long_momentum and pattern_bull:
            return {"signal": "long", "tp": tp_long, "sl": sl_long, "entry": entry}
        elif close < ema20 and fib_reject and short_momentum and pattern_bear:
            return {"signal": "short", "tp": tp_short, "sl": sl_short, "entry": entry}
        else:
            return {"signal": None}
