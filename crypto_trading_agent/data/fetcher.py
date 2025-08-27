from binance.client import Client
import pandas as pd

# ===============================
# Helpers: OHLCV / Indicators
# ===============================
def fetch_ohlcv(symbol: str, interval: str, limit: int, client) -> pd.DataFrame:
    """Closed klines only -> DataFrame indexed by timestamp with ohlcv floats."""
    kl = client.futures_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(kl, columns=[
        "open_time","open","high","low","close","volume",
        "close_time","quote_asset_volume","num_trades",
        "taker_buy_base","taker_buy_quote","ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    df.set_index("close_time", inplace=True)  # align to candle close

    num_cols = ["open","high","low","close","volume"]
    df[num_cols] = df[num_cols].astype(float)
    return df[num_cols]
