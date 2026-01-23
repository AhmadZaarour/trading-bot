import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from indicators.features import add_indicators
import yaml

class Backtester:
    def __init__(self, risk, broker, strategy, data, config, initial_balance=1000):
        self.risk = risk
        self.broker = broker
        self.strategy = strategy
        self.data = data
        self.initial_balance = initial_balance
        self.leverage = config["RISK"]["MAX_LEVERAGE"]
        self.risk_per_trade = config["RISK"]["RISK_PER_TRADE"]
        self.max_bars = config["RISK"]["MAX_BARS_PER_TRADE"]
        self.symbol = config["SYMBOL"]
        self.interval = config["INTERVAL"]
        self.limit = 1000
        self.lookback = config["LOOKBACK"]
        self.balance = initial_balance
        self.equity_curve = []
        self.trades = []
        backtest_cfg = config.get("BACKTEST", {})
        self.taker_fee = float(backtest_cfg.get("TAKER_FEE", 0.0004))
        self.slippage_bps = float(backtest_cfg.get("SLIPPAGE_BPS", 5))
        self.slippage_pct = self.slippage_bps / 10000.0

    def simulate(self):
        open_trade = None
        pending_signal = None
        df = pd.read_csv("past_data/futures_3y.csv")
        if df.empty:
            print("No data for backtest.")
            return self.trades, self.equity_curve
        df = add_indicators(df)
        print(f"Data loaded: {len(df)} rows.")

        i = self.lookback
        while i < len(df):
            if pending_signal:
                entry_row = df.iloc[i]
                entry_price = float(entry_row["open"])
                side = pending_signal["side"]
                sl = pending_signal["sl"]
                tp = pending_signal["tp"]
                if not self._valid_trade_levels(side, entry_price, sl, tp):
                    pending_signal = None
                    i += 1
                    continue

                qty = self.risk.safe_position_size(
                    entry_price,
                    sl,
                    self.balance,
                    self.broker.get_filters(self.symbol),
                    self.leverage,
                    self.risk_per_trade,
                )
                if qty > 0:
                    execution_price = self._apply_slippage(entry_price, side, is_entry=True)
                    entry_fee = self._calc_fee(execution_price, qty)
                    self.balance -= entry_fee
                    open_trade = {
                        "side": side,
                        "entry": execution_price,
                        "sl": sl,
                        "tp": tp,
                        "bars_open": 0,
                        "qty": qty,
                        "entry_time": df.index[i],
                        "entry_fee": entry_fee,
                    }
                    self.equity_curve.append(self.balance)
                pending_signal = None

            if open_trade:
                # manage trade exits
                open_trade["bars_open"] += 1
                row = df.iloc[i]
                exit_price = self._resolve_exit_price(open_trade, row)
                if exit_price is not None:
                    self._close_trade(open_trade, exit_price, df, i)
                    open_trade = None

                i += 1  # Move to next bar after managing trade
            else:
                signal = self.strategy.test_evaluate(df, i)
                if not signal or signal.get("signal") not in ("long", "short"):
                    i += 1
                    continue

                entry = signal.get("entry")
                sl = signal.get("sl")
                tp = signal.get("tp")
                if entry is None or sl is None or tp is None:
                    i += 1
                    continue

                entry = float(entry)
                sl = float(sl)
                tp = float(tp)
                if not self._valid_trade_levels(signal["signal"], entry, sl, tp):
                    i += 1
                    continue
                if i + 1 >= len(df):
                    break

                pending_signal = {
                    "side": signal["signal"],
                    "sl": sl,
                    "tp": tp,
                }
                i += 1  # Move to next bar for execution

        return self.trades, self.equity_curve
    
    def run(self):

        trades, equity_curve = self.simulate()

        if not trades:
            print("No trades executed.")
            return

        df = pd.DataFrame(trades)

        # Basic counts
        wins = df[df["pnl"] > 0].shape[0]
        losses = df[df["pnl"] < 0].shape[0]
        breakevens = df[df["pnl"] == 0].shape[0]
        total = len(df)

        # Win rate
        win_rate = (wins / total) * 100 if total > 0 else 0

        # Final balance
        final_balance = equity_curve[-1] if equity_curve else self.initial_balance
        total_profit = final_balance - self.initial_balance
        roi = (total_profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        # Max Drawdown
        eq = np.array(equity_curve)
        peaks = np.maximum.accumulate(eq)
        drawdowns = peaks - eq
        max_dd = drawdowns.max() if len(drawdowns) > 0 else 0
        max_dd_pct = (max_dd / peaks.max()) * 100 if peaks.max() > 0 else 0

        # Sharpe ratio (simple, assuming each trade ~1 period, rf=0)
        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(len(returns))) if np.std(returns) > 0 else 0

        # R-multiples
        r_mults = []
        for t in trades:
            if "sl" in t and t["sl"] and t["entry"] and t.get("pnl") is not None:
                risk = abs(t["entry"] - t["sl"]) * t["qty"]
                if risk > 0:
                    r_mults.append(t["pnl"] / risk)
        avg_r = np.mean(r_mults) if r_mults else 0

        # --- Print report ---
        print("\n=== Backtest Report ===")
        print(f"Total trades: {total}")
        print(f"Wins: {wins} | Losses: {losses} | Breakevens: {breakevens}")
        print(f"Win rate: {win_rate:.2f}%")
        print(f"Initial balance: ${self.initial_balance:,.2f}")
        print(f"Final balance:   ${final_balance:,.2f}")
        print(f"Total profit:    ${total_profit:,.2f}")
        print(f"ROI: {roi:.2f}%")
        print(f"Max Drawdown: ${max_dd:,.2f} ({max_dd_pct:.2f}%)")
        print(f"Sharpe Ratio: {sharpe:.2f}")
        print(f"Average R-multiple: {avg_r:.2f}")

        # --- Plot equity curve and drawdown ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        # Equity curve
        ax1.plot(eq, label="Equity", color="blue")
        ax1.set_ylabel("Balance (USDT)")
        ax1.set_title("Equity Curve")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        # Drawdown
        ax2.plot(drawdowns, label="Drawdown", color="red")
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, color="red", alpha=0.3)
        ax2.set_ylabel("Drawdown (USDT)")
        ax2.set_title("Drawdowns")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

        return df

    def _close_trade(self, trade, exit_price, df, i):
        execution_price = self._apply_slippage(exit_price, trade["side"], is_entry=False)
        if trade["side"] == "long":
            gross_pnl = (execution_price - trade["entry"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry"] - execution_price) * trade["qty"]

        exit_fee = self._calc_fee(execution_price, trade["qty"])
        pnl = gross_pnl - exit_fee
        self.balance += pnl
        trade["exit_price"] = execution_price
        trade["exit_time"] = df.index[i]
        trade["gross_pnl"] = gross_pnl
        trade["exit_fee"] = exit_fee
        trade["pnl"] = pnl - trade.get("entry_fee", 0.0)
        self.trades.append(trade)
        self.equity_curve.append(self.balance)

    def _valid_trade_levels(self, side: str, entry: float, sl: float, tp: float) -> bool:
        if side == "long":
            return sl < entry < tp
        return tp < entry < sl

    def _resolve_exit_price(self, trade, row):
        if trade["bars_open"] >= self.max_bars:
            return float(row["close"])

        low = float(row["low"])
        high = float(row["high"])
        sl = trade["sl"]
        tp = trade["tp"]

        if trade["side"] == "long":
            if low <= sl and high >= tp:
                return sl
            if low <= sl:
                return sl
            if high >= tp:
                return tp
        else:
            if high >= sl and low <= tp:
                return sl
            if high >= sl:
                return sl
            if low <= tp:
                return tp
        return None

    def _apply_slippage(self, price: float, side: str, is_entry: bool) -> float:
        if self.slippage_pct <= 0:
            return price

        if side == "long":
            return price * (1 + self.slippage_pct) if is_entry else price * (1 - self.slippage_pct)
        return price * (1 - self.slippage_pct) if is_entry else price * (1 + self.slippage_pct)

    def _calc_fee(self, price: float, qty: float) -> float:
        if self.taker_fee <= 0:
            return 0.0
        return price * qty * self.taker_fee
