from data.fetcher import fetch_ohlcv
from indicators.calculator import add_indicators
from strategy.adaptive_trade_logic import analyze_row_dynamic
from executor.simulator import simulate_trades
import pandas as pd
import numpy as np

# Fetch and prepare data
coin = input("Enter the coin symbol (e.g., BTCUSDT): ")
df = fetch_ohlcv(coin, "1h", 1000)
df = add_indicators(df)

# Run simulation
results_df = simulate_trades(df, analyze_row_dynamic)

starting_balance = 500  # initial USD


if results_df.empty:
    print("No trades executed.")
else:
    win_count = (results_df["result"] == "win").sum()
    loss_count = (results_df["result"] == "loss").sum()
    breakeven_count = (results_df["result"] == "breakeven").sum()
    no_result_count = (results_df["result"] == "no_result").sum()
    total = len(results_df)
    error_count = 0

        # --- Settings ---
    max_leverage = 5        # Cap position size to avoid unrealistic risk
    risk_per_trade = 0.02   # Keep your existing risk setting

    # --- Initialize ---
    balance = starting_balance
    equity_curve = [balance]
    trade_results, r_multiples = [], []

    for _, trade in results_df.iterrows():
        entry, sl, tp, outcome = trade['entry'], trade['sl'], trade['tp'], trade['result']
        if trade["order"] == "long":
            if trade["result"] == "win" and entry > trade["exit"]:
                error_count += 1
            if trade["result"] == "loss" and entry < trade["exit"]:
                error_count += 1
        elif trade["order"] == "short":
            if trade["result"] == "win" and entry < trade["exit"]:
                error_count += 1
            if trade["result"] == "loss" and entry > trade["exit"]:
                error_count += 1
        else:
            continue

        # --- Skip invalid trades ---
        if sl is None or tp is None or sl == entry or entry <= 0:
            equity_curve.append(balance)
            continue

        # --- Risk & Position sizing ---
        risk_amount = balance * risk_per_trade
        sl_pct = abs(entry - sl) / entry
        if sl_pct <= 0:  # invalid SL distance
            equity_curve.append(balance)
            continue

        position_size = risk_amount / sl_pct
        max_position_size = (balance * max_leverage) / entry
        position_size = min(position_size, max_position_size)

        # --- Trade outcome ---
        if trade["order"] == "long":
            if outcome == "win":
                profit = position_size * (tp - entry)
                balance += profit
                r_multiple = profit / risk_amount
            elif outcome == "loss":
                loss = position_size * (sl - entry)
                balance += loss
                r_multiple = loss / risk_amount
            else:  # breakeven or no result
                r_multiple = 0
        elif trade["order"] == "short":
            if outcome == "win":
                profit = position_size * (entry - tp)
                balance += profit
                r_multiple = profit / risk_amount
            elif outcome == "loss":
                loss = position_size * (entry - sl)
                balance += loss
                r_multiple = loss / risk_amount
            else:  # breakeven or no result
                r_multiple = 0

        trade_results.append(outcome)
        r_multiples.append(r_multiple)
        equity_curve.append(balance)

        print(f"time: {trade['entry_time']} Trade outcome: {outcome}, entry: {entry}, SL: {sl}, TP: {tp}, close: {trade['exit']}, order: {trade['order']}, bars held: {trade['bars_held']}")

    # --- Convert equity curve ---
    equity_curve = np.array(equity_curve)

    # --- Metrics ---
    wins = trade_results.count("win")
    losses = trade_results.count("loss")
    breakevens = trade_results.count("breakeven")
    total_trades = len(trade_results)
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    # Max Drawdown (based on equity, not starting balance)
    peak = np.maximum.accumulate(equity_curve)
    drawdown = (peak - equity_curve)
    max_drawdown = drawdown.max()
    max_drawdown_pct = (max_drawdown / peak.max() * 100) if peak.max() > 0 else 0

    # Sharpe Ratio (annualized approx using trade-level returns)
    returns = np.diff(equity_curve) / equity_curve[:-1]
    returns = returns[np.isfinite(returns)]
    sharpe_ratio = (np.mean(returns) / np.std(returns) * np.sqrt(len(returns))) if np.std(returns) > 0 else 0

    # Avg R-multiple
    avg_r_multiple = np.mean(r_multiples) if r_multiples else 0

    # --- Results ---
    metrics = {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "breakevens": breakevens,
        "win_rate": win_rate,
        "initial_balance": starting_balance,
        "final_balance": balance,
        "total_profit": balance - starting_balance,
        "roi_pct": ((balance - starting_balance) / starting_balance) * 100,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "sharpe_ratio": sharpe_ratio,
        "avg_r_multiple": avg_r_multiple
    }

    # --- Print nicely ---
    print(f"Total trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses} | Breakevens: {breakevens} | No result: {no_result_count} | Errors: {error_count}")
    print(f"Win rate: {win_rate:.2f}%")
    print(f"Initial balance: ${starting_balance:,.2f}")
    print(f"Final balance: ${balance:,.2f}")
    print(f"Total profit: ${balance - starting_balance:,.2f}")
    print(f"ROI: {metrics['roi_pct']:.2f}%")
    print(f"Max Drawdown: ${max_drawdown:,.2f} ({max_drawdown_pct:.2f}%)")
    print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
    print(f"Average R-multiple: {avg_r_multiple:.2f}")

    import matplotlib.pyplot as plt

    # Filter only wins and losses
    wins = results_df[results_df["result"] == "win"]
    losses = results_df[results_df["result"] == "loss"]

    # Indicator columns
    indicator_cols = [
                "ema_trend", "macd_momentum", "rsi_divergence",
                "has_pattern", "sr_confluence", "fib", "triangle",
                "wedge", "flag_pattern", "double", "rr", "volatility_ok"
            ]

    # Calculate percentage occurrence of each condition in wins/losses
    win_freq = wins[indicator_cols].mean(numeric_only=True) * 100
    loss_freq = losses[indicator_cols].mean(numeric_only=True) * 100

    # Combine into one DataFrame for plotting
    compare_df = pd.DataFrame({
        "Wins %": win_freq,
        "Losses %": loss_freq
    })

    # Plot side-by-side bar chart
    ax = compare_df.plot(kind="bar", figsize=(12, 6))
    plt.title("Indicator Presence in Winning vs Losing Trades")
    plt.ylabel("Percentage of Trades (%)")
    plt.xticks(rotation=45)
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # Print sorted by difference (loss-heavy)
    print("\nIndicators more common in losses than wins:")
    loss_heavy = (loss_freq - win_freq).sort_values(ascending=False)
    print(loss_heavy)

    # Plot the equity curve as a line chart
    plt.figure(figsize=(12, 6))
    plt.plot(equity_curve, label="Equity Curve")
    plt.title("Equity Curve Over Time")
    plt.xlabel("Trade Number")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    # Plot the equity curve as a line chart
    plt.figure(figsize=(12, 6))
    plt.plot(r_multiples, label="R_Multiples Curve")
    plt.title("R_Multiples Curve Over Time")
    plt.xlabel("Trade Number")
    plt.ylabel("R_Multiple")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
