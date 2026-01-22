import time
from datetime import datetime, timezone
from decimal import Decimal
from typing import Optional

import pandas as pd

from indicators.features import add_indicators


class SpotEngine:
    def __init__(
        self,
        data,
        broker,
        strategy,
        symbol: str,
        interval: str,
        lookback: int,
        risk_per_trade: float,
        max_bars_per_trade: int,
        poll_seconds: int,
    ):
        self.data = data
        self.broker = broker
        self.strategy = strategy
        self.symbol = symbol
        self.interval = interval
        self.lookback = lookback
        self.risk_per_trade = risk_per_trade
        self.max_bars_per_trade = max_bars_per_trade
        self.poll_seconds = poll_seconds
        self.last_seen_close: Optional[pd.Timestamp] = None
        self.open_trade: Optional[dict] = None

    def run(self) -> None:
        print("Starting spot engine...")
        while True:
            try:
                df = self.data.latest_df(self.symbol, self.interval, self.lookback)
                if df.empty:
                    time.sleep(self.poll_seconds)
                    continue

                df = add_indicators(df)
                latest_close_time = df.index[-1]
                if self.last_seen_close is not None and latest_close_time == self.last_seen_close:
                    time.sleep(self.poll_seconds)
                    continue

                self.last_seen_close = latest_close_time

                current_position = self.broker.get_position(self.symbol)

                if self.open_trade:
                    self.open_trade["bars_open"] += 1
                    if abs(current_position["amt"]) < 1e-12:
                        print("Spot trade closed via TP/SL.")
                        self.open_trade = None
                    elif self.open_trade["bars_open"] >= self.max_bars_per_trade:
                        print("Max bars open reached, exiting spot trade.")
                        self._force_exit()

                if abs(current_position["amt"]) < 1e-12 and not self.open_trade:
                    trade = self.strategy.evaluate(df)
                    if trade.get("signal") != "long":
                        continue

                    entry = Decimal(str(trade["entry"]))
                    sl = Decimal(str(trade["sl"]))
                    tp = Decimal(str(trade["tp"]))

                    if sl >= entry or tp <= entry:
                        print("Skipped trade: invalid TP/SL levels for spot long.")
                        continue

                    qty, quote = self._size_position(entry, sl)
                    if qty <= 0 or quote <= 0:
                        print("Skipped trade: qty too small or below exchange minimums.")
                        continue

                    buy = self.broker.market_buy_by_quote(self.symbol, float(quote))
                    filled_qty, avg_price = self.broker.parse_filled_buy(buy)
                    if filled_qty <= 0:
                        print("Buy did not fill, skipping.")
                        continue

                    oco = self.broker.place_sell_oco(
                        symbol=self.symbol,
                        qty_base=filled_qty,
                        take_profit_price=float(tp),
                        stop_price=float(sl),
                    )

                    self.open_trade = {
                        "symbol": self.symbol,
                        "side": "BUY",
                        "qty": filled_qty,
                        "entry": avg_price,
                        "bars_open": 0,
                        "sl": float(sl),
                        "tp": float(tp),
                        "entry_time": datetime.now(timezone.utc),
                        "oco_order": oco,
                    }
                    print(f"Opened spot trade: {self.open_trade}")

            except Exception as exc:
                print("Spot engine error:", exc)
                time.sleep(self.poll_seconds)

    def _size_position(self, entry: Decimal, sl: Decimal) -> tuple[Decimal, Decimal]:
        balance = Decimal(str(self.broker.get_usdt_free()))
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return Decimal("0"), Decimal("0")

        risk_amount = balance * Decimal(str(self.risk_per_trade))
        qty = risk_amount / sl_dist

        qty, _ = self.broker.round_qty_price(self.symbol, qty, None)
        filters = self.broker.get_filters(self.symbol)

        if qty < filters["minQty"]:
            return Decimal("0"), Decimal("0")

        quote = qty * entry
        if filters["minNotional"] is not None and quote < filters["minNotional"]:
            return Decimal("0"), Decimal("0")

        return qty, quote

    def _force_exit(self) -> None:
        try:
            self.broker.cancel_all_orders(self.symbol)
        except Exception as exc:
            print("Error cancelling spot orders:", exc)

        position = self.broker.get_position(self.symbol)
        if position["amt"] > 0:
            try:
                self.broker.market_sell(self.symbol, position["amt"])
            except Exception as exc:
                print("Error market-selling spot position:", exc)

        self.open_trade = None
