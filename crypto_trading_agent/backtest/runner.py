import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from indicators.features import add_indicators
import yaml

# Open and load the YAML file
with open("config/default.yaml", "r") as file:
    config = yaml.safe_load(file)

class Backtester:
    def __init__(self, strategy, data, initial_balance=1000, leverage=5, risk_per_trade=0.02, max_bars=20):
        self.strategy = strategy
        self.data = data
        self.initial_balance = initial_balance
        self.leverage = leverage
        self.risk_per_trade = risk_per_trade
        self.max_bars = max_bars
        self.symbol = config["SYMBOL"]
        self.interval = config["INTERVAL"]
        self.lookback = 1000
        self.balance = initial_balance
        self.equity_curve = []
        self.trades = []

    def simulate(self):
        open_trade = None

        df = self.data.latest_df(self.symbol, self.interval, self.lookback)
        if df.empty:
            print("No data for backtest.")
        df = add_indicators(df)

        for i in range(len(df)):  # skip warmup
            if open_trade:
                # manage trade exits
                open_trade["bars_open"] += 1
                price = df["close"].iloc[i]

                if open_trade["side"] == "long":
                    if price <= open_trade["sl"] or price >= open_trade["tp"] or open_trade["bars_open"] >= self.max_bars:
                        exit_price = price
                        self._close_trade(open_trade, exit_price, df, i)
                        open_trade = None

                elif open_trade["side"] == "short":
                    if price >= open_trade["sl"] or price <= open_trade["tp"] or open_trade["bars_open"] >= self.max_bars:
                        exit_price = price
                        self._close_trade(open_trade, exit_price, df, i)
                        open_trade = None

            else:
                # look for new trade
                signal = self.strategy.test_evaluate(df)
                if signal["signal"] in ("long", "short"):
                    qty = self._position_size(signal["entry"], signal["sl"])
                    if qty > 0:
                        open_trade = {
                            "side": signal["signal"],
                            "entry": signal["entry"],
                            "sl": signal["sl"],
                            "tp": signal["tp"],
                            "bars_open": 0,
                            "qty": qty,
                            "entry_time": df.index[i],
                        }

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

    def _position_size(self, entry, sl):
        sl_pct = abs(entry - sl) / entry
        if sl_pct <= 0:
            return 0
        risk_amount = self.balance * self.risk_per_trade
        qty = min(risk_amount / sl_pct, (self.balance * self.leverage) / entry)
        return qty

    def _close_trade(self, trade, exit_price, df, i):

        if trade["side"] == "long":
            pnl = (exit_price - trade["entry"]) * trade["qty"]
        else:
            pnl = (trade["entry"] - exit_price) * trade["qty"]

        self.balance += pnl
        trade["exit_price"] = exit_price
        trade["exit_time"] = df.index[i]
        trade["pnl"] = pnl
        self.trades.append(trade)
        self.equity_curve.append(self.balance)
