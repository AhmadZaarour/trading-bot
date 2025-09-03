import time
from typing import Optional
import pandas as pd
from datetime import datetime, timezone
from indicators.features import add_indicators
import yaml

# Open and load the YAML file
with open("config/default.yaml", "r") as file:
    config = yaml.safe_load(file)  # safe_load prevents running arbitrary code

class Engine:
    def __init__(self, data, broker, risk, strategy):
        self.data = data
        self.broker = broker
        self.risk = risk
        self.strategy = strategy
        self.symbol = config["SYMBOL"]
        self.interval = config["INTERVAL"]
        self.lookback = config["LOOKBACK"]
        self.max_leverage = config["RISK"]["MAX_LEVERAGE"]
        self.risk_per_trade = config["RISK"]["RISK_PER_TRADE"]
        self.max_bars_per_trade = config["RISK"]["MAX_BARS_PER_TRADE"]
        self.poll_seconds = config["POLL_SECONDS"]
        self.last_seen_close: Optional[pd.Timestamp] = None
        self.open_trade: Optional[dict] = None

    def run(self):
        print("Starting engine...")
        while True:
            try:
                print(self.interval)
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
                i = len(df) - 1

                current_position = self.broker.get_position(self.symbol)

                # handling open trade
                if self.open_trade:
                    print("Monitoring open trade...")
                    self.open_trade["bars_open"] += 1
                    if abs(current_position["amt"]) < 1e-12:
                        try:
                            self.broker.close_all_positions(self.symbol)
                            print("Trade closed SL/TP hit.")
                        except Exception as e:
                            print("Error closing positions:", e)
                        self.open_trade = None
                    elif self.open_trade["bars_open"] >= self.max_bars_per_trade:
                        print("Max bars open reached.")
                        try:
                            self.broker.close_all_positions(self.symbol)
                        except Exception as e:
                            print("Error closing positions:", e)
                        self.open_trade = None

                # if no open trade, evaluate strategy
                if abs(current_position["amt"]) < 1e-12 and not self.open_trade:

                    trade = self.strategy.test_evaluate(df)
                    if trade["signal"] in ["long", "short"]:

                        balance = self.broker.get_balance()
                        sl = self.broker.round_tick(trade["sl"], self.broker.get_filters(self.symbol)["tickSize"])
                        tp = self.broker.round_tick(trade["tp"], self.broker.get_filters(self.symbol)["tickSize"])
                        entry_price = trade["entry"]
                        qty = self.risk.safe_position_size(entry_price, sl, balance, self.broker.get_filters(self.symbol), self.max_leverage, self.risk_per_trade)

                        print(f"Balance(avail)={balance}, entry={entry_price}, sl={sl}, "
                        f"sl_dist={abs(entry_price-sl)}, calc_qty={qty}, "
                        f"notional≈{qty*entry_price if qty else 0}")

                        if qty <= 0:
                            print("Skipped trade: qty too small or margin-capped.")
                            time.sleep(self.poll_seconds)
                            continue

                        if abs(current_position["amt"]) < 1e-12:

                            side = "BUY" if trade["signal"] == "long" else "SELL"
                            qty = self.broker.round_step(qty, self.broker.get_filters(self.symbol)["stepSize"])
                            order = self.risk.try_market_order_with_backoff(self.symbol, side, qty)

                            if order:
                                self.broker.place_exit_orders(self.symbol, side, qty, sl, tp)
                                self.open_trade = trade
                                self.open_trade["entry_time"] = datetime.now(timezone.utc)
                                print("SL/TP orders placed.")

                                self.open_trade = {
                                    "symbol": self.symbol,
                                    "side": side,
                                    "qty": qty,
                                    "entry": entry_price,
                                    "bars_open": 0,
                                    "sl": sl,
                                    "tp": tp,
                                    "entry_time": self.open_trade["entry_time"]
                                }

                                print(f"Opened {side} trade: {self.open_trade}")
                                print("Waiting for next candle...")

            except Exception as e:
                print("Engine error:", e)
                time.sleep(self.poll_seconds)
