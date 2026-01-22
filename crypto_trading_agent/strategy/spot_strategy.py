from .basic_strategy import MyStrategy


class SpotStrategy(MyStrategy):
    def __init__(self, min_rr: float = 1.4, require_trend: bool = True):
        super().__init__()
        self.min_rr = min_rr
        self.require_trend = require_trend

    def evaluate(self, df):
        trade = super().evaluate(df)
        return self._filter_trade(trade)

    def test_evaluate(self, df, i):
        trade = super().test_evaluate(df, i)
        return self._filter_trade(trade)

    def _filter_trade(self, trade):
        if not trade or trade.get("signal") != "long":
            return {"signal": None}

        rr = trade.get("rr")
        if rr is not None and rr < self.min_rr:
            return {"signal": None}

        if self.require_trend and trade.get("regime") not in (None, "trend"):
            return {"signal": None}

        return trade
