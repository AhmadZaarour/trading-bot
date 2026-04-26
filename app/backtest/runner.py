import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml

from indicators.features import add_indicators


class Backtester:
    def __init__(self, risk, broker, strategy, data, config, initial_balance=1000):
        self.risk = risk
        self.broker = broker
        self.strategy = strategy
        self.data = data

        self.initial_balance = float(initial_balance)
        self.balance = float(initial_balance)

        self.leverage = config["RISK"]["MAX_LEVERAGE"]
        self.risk_per_trade = config["RISK"]["RISK_PER_TRADE"]
        self.max_bars = config["RISK"]["MAX_BARS_PER_TRADE"]

        self.symbol = config["SYMBOL"]
        self.interval = config["INTERVAL"]
        self.limit = 1000
        self.lookback = config["LOOKBACK"]

        self.equity_curve = []
        self.trades = []

        #backtest_cfg = config.get("BACKTEST", {})
        #self.taker_fee = float(backtest_cfg.get("TAKER_FEE", 0.0004))
        #self.slippage_bps = float(backtest_cfg.get("SLIPPAGE_BPS", 5))
        self.slippage_bps = 5
        self.taker_fee = 0.0004
        self.slippage_pct = self.slippage_bps / 10000.0

    # ------------------------- Core Simulation -------------------------

    def simulate(self):
        open_trade = None
        pending_signal = None

        #df = pd.read_csv("past_data/futures_3y.csv")
        df = self.data.latest_df(symbol=self.symbol, interval=self.interval, limit=self.limit)
        if df.empty:
            print("No data for backtest.")
            return self.trades, self.equity_curve

        df = add_indicators(df)
        print(f"Data loaded: {len(df)} rows.")

        # Start equity curve at initial balance
        self.equity_curve = [self.balance]

        i = int(self.lookback)
        if i < 0:
            i = 0

        while i < len(df):
            row = df.iloc[i]

            # 1) Execute pending signal at THIS bar open (next-bar execution)
            if pending_signal is not None:
                entry_price = float(row["open"])
                side = pending_signal["side"]
                sl = float(pending_signal["sl"])
                tp = float(pending_signal["tp"])

                # Validate levels using the actual entry (open) price
                if self._valid_trade_levels(side, entry_price, sl, tp):
                    qty = self.risk.safe_position_size(
                        entry_price=entry_price,
                        sl=sl,
                        balance=self.balance,
                        filters=self.broker.get_filters(self.symbol),
                        leverage=self.leverage,
                        risk_per_trade=self.risk_per_trade,
                    )

                    if qty > 0:
                        exec_entry = self._apply_slippage(entry_price, side=side, is_entry=True)
                        entry_fee = self._calc_fee(exec_entry, qty)
                        self.balance -= entry_fee

                        open_trade = {
                            "side": side,
                            "entry": exec_entry,
                            "sl": sl,
                            "tp": tp,
                            "bars_open": 0,
                            "qty": float(qty),
                            "entry_i": i,
                            "entry_fee": float(entry_fee),
                            "entry_time": df.index[i],
                            "risk_per_unit": abs(exec_entry - sl),  # store initial risk per unit for R-multiple
                        }

                pending_signal = None

            # 2) Manage open trade (conservative candle logic)
            if open_trade is not None:
                # Skip management on entry bar (best practice)
                if i > open_trade["entry_i"]:
                    open_trade["bars_open"] += 1

                    # First resolve exit with CURRENT SL/TP (conservative: don't assume intrabar stop-move)
                    exit_price = self._resolve_exit_price(open_trade, row)

                    if exit_price is not None:
                        self._close_trade(open_trade, exit_price, df, i)
                        open_trade = None
                    else:
                        continue

            # 3) If flat and no pending signal, evaluate signal at bar close -> schedule for next bar
            if open_trade is None and pending_signal is None:
                signal = self.strategy.test_evaluate(df, i)
                if signal and signal.get("signal") in ("long", "short"):
                    entry = signal.get("entry")
                    sl = signal.get("sl")
                    tp = signal.get("tp")

                    if entry is not None and sl is not None and tp is not None:
                        entry = float(entry)
                        sl = float(sl)
                        tp = float(tp)

                        # Validate using the strategy's entry estimate (close) to reduce junk,
                        # actual execution will be validated again at next bar open.
                        if self._valid_trade_levels(signal["signal"], entry, sl, tp):
                            if i + 1 < len(df):
                                pending_signal = {"side": signal["signal"], "sl": sl, "tp": tp}

            # 4) Mark-to-market equity each bar (best practice)
            close_price = float(row["close"])
            equity = self.balance
            if open_trade is not None:
                if open_trade["side"] == "long":
                    equity += (close_price - open_trade["entry"]) * open_trade["qty"]
                else:
                    equity += (open_trade["entry"] - close_price) * open_trade["qty"]

            self.equity_curve.append(equity)
            i += 1

        # 5) Force close any still-open trade at end of dataset (realize PnL)
        if open_trade is not None:
            last_i = len(df) - 1
            last_row = df.iloc[last_i]
            self._close_trade(open_trade, float(last_row["close"]), df, last_i)
            open_trade = None
            self.equity_curve.append(self.balance)

        return self.trades, self.equity_curve

    # ------------------------- Reporting -------------------------

    def run(self):
        trades, equity_curve = self.simulate()

        if not trades:
            print("No trades executed.")
            return

        df = pd.DataFrame(trades)

        wins = df[df["pnl"] > 0].shape[0]
        losses = df[df["pnl"] < 0].shape[0]
        breakevens = df[df["pnl"] == 0].shape[0]
        total = len(df)

        win_rate = (wins / total) * 100 if total > 0 else 0

        final_balance = equity_curve[-1] if equity_curve else self.initial_balance
        total_profit = final_balance - self.initial_balance
        roi = (total_profit / self.initial_balance) * 100 if self.initial_balance > 0 else 0

        eq = np.array(equity_curve, dtype=float)
        peaks = np.maximum.accumulate(eq)
        drawdowns = peaks - eq
        max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0
        max_dd_pct = (max_dd / float(peaks.max())) * 100 if float(peaks.max()) > 0 else 0.0

        returns = np.diff(eq) / eq[:-1] if len(eq) > 1 else np.array([])
        sharpe = (np.mean(returns) / np.std(returns) * np.sqrt(len(returns))) if np.std(returns) > 0 else 0.0

        # R-multiples using INITIAL risk (not moved SL)
        r_mults = []
        for t in trades:
            risk_per_unit = float(t.get("risk_per_unit", 0.0))
            if risk_per_unit > 0:
                risk = risk_per_unit * float(t["qty"])
                r_mults.append(float(t["pnl"]) / risk)

        avg_r = float(np.mean(r_mults)) if r_mults else 0.0

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

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

        ax1.plot(eq, label="Equity")
        ax1.set_ylabel("Equity (USDT)")
        ax1.set_title("Equity Curve")
        ax1.legend()
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2.plot(drawdowns, label="Drawdown")
        ax2.fill_between(range(len(drawdowns)), drawdowns, 0, alpha=0.3)
        ax2.set_ylabel("Drawdown (USDT)")
        ax2.set_title("Drawdowns")
        ax2.legend()
        ax2.grid(True, linestyle="--", alpha=0.5)

        plt.tight_layout()
        plt.show()

        return df

    # ------------------------- Trade Mechanics -------------------------

    def _close_trade(self, trade, exit_price, df, i):
        # Exit action depends on side: long exit = SELL, short exit = BUY
        exec_exit = self._apply_slippage(exit_price, side=trade["side"], is_entry=False)

        if trade["side"] == "long":
            gross_pnl = (exec_exit - trade["entry"]) * trade["qty"]
        else:
            gross_pnl = (trade["entry"] - exec_exit) * trade["qty"]

        exit_fee = self._calc_fee(exec_exit, trade["qty"])

        # Balance already paid entry_fee on entry; add (gross - exit_fee) now
        self.balance += (gross_pnl - exit_fee)

        net_pnl = (gross_pnl - exit_fee) - float(trade.get("entry_fee", 0.0))

        trade["exit_price"] = exec_exit
        trade["exit_time"] = df.index[i]
        trade["gross_pnl"] = gross_pnl
        trade["exit_fee"] = exit_fee
        trade["pnl"] = net_pnl
        trade["result"] = "win" if net_pnl > 0 else "loss" if net_pnl < 0 else "breakeven"

        self.trades.append(trade)

    def _valid_trade_levels(self, side: str, entry: float, sl: float, tp: float) -> bool:
        if side == "long":
            return sl < entry < tp
        return tp < entry < sl

    def _resolve_exit_price(self, trade, row):
        # Time-based exit
        if trade["bars_open"] >= self.max_bars:
            return float(row["close"])

        low = float(row["low"])
        high = float(row["high"])
        sl = float(trade["sl"])
        tp = float(trade["tp"])

        # Conservative handling if both hit: assume SL first
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
        """
        Adverse slippage based on the actual action:
        - Long entry = BUY, long exit = SELL
        - Short entry = SELL, short exit = BUY
        BUY slips up, SELL slips down.
        """
        if self.slippage_pct <= 0:
            return price

        is_buy = (side == "long" and is_entry) or (side == "short" and not is_entry)
        return price * (1 + self.slippage_pct) if is_buy else price * (1 - self.slippage_pct)

    def _calc_fee(self, price: float, qty: float) -> float:
        if self.taker_fee <= 0:
            return 0.0
        return price * qty * self.taker_fee

    # ------------------------- Stop Management -------------------------

    def _move_sl_best_practice(
        self,
        side: str,
        entry: float,
        sl_initial: float,
        sl_current: float,
        bar_high: float,
        bar_low: float,
        r_trigger_be: float = 1.0,     # breakeven at +1R
        r_trigger_trail: float = 2.0,  # start trailing at +2R
        be_buffer: float = 0.0,
    ) -> float:
        """
        Candle-safe stop management using high/low.
        Conservative: this is applied only after confirming no TP/SL hit this bar.
        - Moves SL to breakeven at +1R
        - Optionally trails after +2R (locks ~1R behind extreme)
        """
        risk = abs(entry - sl_initial)
        if risk <= 0:
            return sl_current

        if side == "long":
            be_trigger = entry + r_trigger_be * risk
            trail_trigger = entry + r_trigger_trail * risk

            if bar_high >= be_trigger:
                sl_current = max(sl_current, entry + be_buffer)

            if bar_high >= trail_trigger:
                trail_sl = bar_high - risk
                sl_current = max(sl_current, trail_sl)

            return sl_current

        # short
        be_trigger = entry - r_trigger_be * risk
        trail_trigger = entry - r_trigger_trail * risk

        if bar_low <= be_trigger:
            sl_current = min(sl_current, entry - be_buffer)

        if bar_low <= trail_trigger:
            trail_sl = bar_low + risk
            sl_current = min(sl_current, trail_sl)

        return sl_current

    # ------------------------- Export -------------------------

    def _save_to_json(self, filepath: str):
        with open(filepath, "w") as f:
            yaml.dump({"trades": self.trades}, f)
