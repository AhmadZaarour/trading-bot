from decimal import Decimal
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from indicators.features import add_indicators


class SpotBacktester:
    def __init__(self, broker, strategy, data, config, initial_balance=1000):
        self.broker = broker
        self.strategy = strategy
        self.data = data
        self.initial_balance = Decimal(str(initial_balance))
        self.risk_per_trade = config["RISK"]["RISK_PER_TRADE"]
        self.max_bars = config["RISK"]["MAX_BARS_PER_TRADE"]
        self.symbol = config["SYMBOL"]
        self.interval = config["INTERVAL"]
        self.limit = 1000
        self.lookback = config["LOOKBACK"]
        self.balance = self.initial_balance
        self.equity_curve = []
        self.trades = []
        backtest_cfg = config.get("BACKTEST", {})
        self.taker_fee = Decimal(str(backtest_cfg.get("TAKER_FEE", 0.0004)))
        self.slippage_bps = Decimal(str(backtest_cfg.get("SLIPPAGE_BPS", 5)))
        self.slippage_pct = self.slippage_bps / Decimal("10000")

    def simulate(self):
        open_trade = None
        pending_signal = None
        df = self.data.latest_df(self.symbol, self.interval, self.limit)
        if df.empty:
            print("No data for spot backtest.")
            return self.trades, self.equity_curve

        df = add_indicators(df)
        print(f"Data loaded: {len(df)} rows.")

        i = self.lookback
        while i < len(df):
            if pending_signal:
                entry_row = df.iloc[i]
                entry_price = Decimal(str(entry_row["open"]))
                sl = pending_signal["sl"]
                tp = pending_signal["tp"]
                if not self._valid_trade_levels(entry_price, sl, tp):
                    pending_signal = None
                    i += 1
                    continue

                qty = self._size_position(entry_price, sl)
                if qty > 0:
                    execution_price = self._apply_slippage(entry_price, is_entry=True)
                    entry_fee = self._calc_fee(execution_price, qty)
                    self.balance -= entry_fee
                    open_trade = {
                        "side": "long",
                        "entry": float(execution_price),
                        "sl": float(sl),
                        "tp": float(tp),
                        "bars_open": 0,
                        "qty": float(qty),
                        "entry_time": df.index[i],
                        "entry_fee": float(entry_fee),
                    }
                    self.equity_curve.append(float(self.balance))
                pending_signal = None

            if open_trade:
                open_trade["bars_open"] += 1
                row = df.iloc[i]
                exit_price = self._resolve_exit_price(open_trade, row)
                if exit_price is not None:
                    self._close_trade(open_trade, exit_price, df, i)
                    open_trade = None
                i += 1
                continue

            signal = self.strategy.test_evaluate(df, i)
            if not signal or signal.get("signal") != "long":
                i += 1
                continue

            entry = signal.get("entry")
            sl = signal.get("sl")
            tp = signal.get("tp")
            if entry is None or sl is None or tp is None:
                i += 1
                continue

            entry = Decimal(str(entry))
            sl = Decimal(str(sl))
            tp = Decimal(str(tp))

            if not self._valid_trade_levels(entry, sl, tp):
                i += 1
                continue
            if i + 1 >= len(df):
                break

            pending_signal = {
                "sl": sl,
                "tp": tp,
            }
            i += 1

        return self.trades, self.equity_curve

    def run(self):
        trades, equity_curve = self.simulate()

        if not trades:
            print("No spot trades executed.")
            return

        df = pd.DataFrame(trades)

        wins = df[df["pnl"] > 0].shape[0]
        losses = df[df["pnl"] < 0].shape[0]
        breakevens = df[df["pnl"] == 0].shape[0]
        total = len(df)

        win_rate = (wins / total) * 100 if total > 0 else 0

        final_balance = equity_curve[-1] if equity_curve else float(self.initial_balance)
        total_profit = final_balance - float(self.initial_balance)
        roi = (total_profit / float(self.initial_balance)) * 100 if self.initial_balance > 0 else 0

        eq = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq) if eq.size else np.array([])
        drawdowns = peaks - eq if eq.size else np.array([])
        max_dd = drawdowns.max() if drawdowns.size else 0
        max_dd_pct = (max_dd / peaks.max()) * 100 if peaks.size and peaks.max() > 0 else 0

        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(len(returns))) if np.std(returns) > 0 else 0

        r_mults = []
        for t in trades:
            if t.get("sl") is not None and t.get("entry") is not None and t.get("pnl") is not None:
                risk = abs(t["entry"] - t["sl"]) * t["qty"]
                if risk > 0:
                    r_mults.append(t["pnl"] / risk)
        avg_r = np.mean(r_mults) if r_mults else 0

        print("\n=== Spot Backtest Report ===")
        print(f"Total trades: {total}")
        print(f"Wins: {wins} | Losses: {losses} | Breakevens: {breakevens}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Initial balance: ${float(self.initial_balance):,.2f}")
        print(f"Final balance:   ${final_balance:,.2f}")
        print(f"Total profit:    ${total_profit:,.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Max Drawdown: ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Average R-multiple: {avg_r:.2f}")

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.plot(eq, label="Equity", color="blue")
        ax1.set_ylabel("Balance (USDT)")
        ax1.set_title("Equity Curve")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(drawdowns, label="Drawdown", color="red")
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown (USDT)")
        ax2.set_title("Drawdowns")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

        return df

    def _size_position(self, entry: Decimal, sl: Decimal) -> Decimal:
        sl_dist = abs(entry - sl)
        if sl_dist <= 0:
            return Decimal("0")

        risk_amount = self.balance * Decimal(str(self.risk_per_trade))
        qty = risk_amount / sl_dist

        qty, _ = self.broker.round_qty_price(self.symbol, qty, None)
        filters = self.broker.get_filters(self.symbol)

        if qty < filters["minQty"]:
            return Decimal("0")

        quote = qty * entry
        if filters["minNotional"] is not None and quote < filters["minNotional"]:
            return Decimal("0")

        return qty

    def _close_trade(self, trade, exit_price, df, i):
        execution_price = self._apply_slippage(Decimal(str(exit_price)), is_entry=False)
        gross_pnl = (execution_price - Decimal(str(trade["entry"]))) * Decimal(str(trade["qty"]))
        exit_fee = self._calc_fee(execution_price, Decimal(str(trade["qty"])))
        pnl = gross_pnl - exit_fee
        self.balance += pnl
        trade["exit_price"] = float(execution_price)
        trade["exit_time"] = df.index[i]
        trade["gross_pnl"] = float(gross_pnl)
        trade["exit_fee"] = float(exit_fee)
        trade["pnl"] = float(pnl - Decimal(str(trade.get("entry_fee", 0.0))))
        self.trades.append(trade)
        self.equity_curve.append(float(self.balance))

    def _valid_trade_levels(self, entry: Decimal, sl: Decimal, tp: Decimal) -> bool:
        return sl < entry < tp

    def _resolve_exit_price(self, trade, row) -> Optional[float]:
        if trade["bars_open"] >= self.max_bars:
            return float(row["close"])

        low = float(row["low"])
        high = float(row["high"])
        sl = trade["sl"]
        tp = trade["tp"]

        if low <= sl and high >= tp:
            return sl
        if low <= sl:
            return sl
        if high >= tp:
            return tp
        return None

    def _apply_slippage(self, price: Decimal, is_entry: bool) -> Decimal:
        if self.slippage_pct <= 0:
            return price
        return price * (Decimal("1") + self.slippage_pct) if is_entry else price * (Decimal("1") - self.slippage_pct)

    def _calc_fee(self, price: Decimal, qty: Decimal) -> Decimal:
        if self.taker_fee <= 0:
            return Decimal("0")
        return price * qty * self.taker_fee
