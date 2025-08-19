def simulate_trades(df, strategy_func, volume_ma_period=5, lookahead=5):
    trades = []
    i = 2
    while i < len(df) - lookahead:
        if i - volume_ma_period < 0:
            i += 1
            continue
        prev2 = df.iloc[i - 2]
        prev1 = df.iloc[i - 1]
        curr = df.iloc[i]
        volume_ma = df["volume"].iloc[i - volume_ma_period:i].mean()

        signal = strategy_func(prev2, prev1, curr, volume_ma, i, df)

        if signal["signal"] == "long":
            entry = signal["entry"]
            sl = signal["sl"]
            tp = signal["tp"]

            # Look ahead N candles
            if i + lookahead + 1 > len(df):
                break
            future_df = df.iloc[i + 1:i + lookahead + 1]

            final_close = future_df.iloc[-1]["close"]
            tolerance = 0.001

            result = "no_result"
            jumped = False
            for _, row in future_df.iterrows():
                if i + lookahead < len(df):
                    if row["low"] <= sl:
                        result = "loss"
                        final_close = row["low"]
                        i += 5
                        jumped = True
                        break

                    elif row["high"] >= tp:
                        result = "win"
                        final_close = row["high"]
                        i += 5
                        jumped = True
                        break

                    else:
                        if i + lookahead + 5 < len(df):
                            future_df_extended = df.iloc[i + 5 + 1:i + lookahead + 5 + 1]
                            final_close = future_df_extended.iloc[-1]["close"]
                            for _, row in future_df_extended.iterrows():
                                if row["low"] <= sl:
                                    final_close = row["low"]
                                    result = "loss"
                                    i += 10
                                    jumped = True
                                    break

                                elif row["high"] >= tp:
                                    final_close = row["high"]
                                    result = "win"
                                    i += 10
                                    jumped = True
                                    break

                                elif abs(final_close - entry) / entry < tolerance:
                                    result = "breakeven"
                                    i += 10
                                    jumped = True
                                    break

                                elif final_close > entry:
                                    result = "win"
                                    i += 10
                                    jumped = True
                                    break
                                elif final_close < entry:
                                    result = "loss"
                                    i += 10
                                    jumped = True
                                    break
                            if jumped:
                                break
                        else:
                            if row["low"] <= sl:
                                final_close = row["low"]
                                result = "loss"
                                i += 5
                                jumped = True
                                break

                            elif row["high"] >= tp:
                                final_close = row["high"]
                                result = "win"
                                i += 5
                                jumped = True
                                break

                            elif abs(final_close - entry) / entry < tolerance:
                                result = "breakeven"
                                i += 5
                                jumped = True
                                break

                            elif final_close > entry:
                                result = "win"
                                i += 5
                                jumped = True
                                break
                            elif final_close < entry:
                                result = "loss"
                                i += 5
                                jumped = True
                                break
            trades.append({
                "entry_time": curr.name,
                "order": signal["signal"],
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit": final_close,
                "result": result,
                "ema_trend": signal["ema_trend"],
                "healthy_rsi": signal["healthy_rsi"],
                "macd_momentum": signal["macd_momentum"],
                #"volume_confirmation": signal["volume_confirmation"],
                "rsi_divergence": signal["rsi_divergence"],
                "has_pattern": signal["has_pattern"],
                "sr_confluence": signal["sr_confluence"],
                "fib_bounce": signal["fib_bounce"],
                "triangle": signal["triangle"],
                "wedge": signal["wedge"],
                "flag_pattern": signal["flag_pattern"],
                "double": signal["double"],
                "score": signal["score"]
            })
            if not jumped:
                i += 1
        

        elif not signal["signal"]:
            i += 1

    return trades