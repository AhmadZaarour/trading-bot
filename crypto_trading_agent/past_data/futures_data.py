import argparse
import os
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
from binance.client import Client

COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "qav",
    "num_trades",
    "taker_base_vol",
    "taker_quote_vol",
    "ignore",
]


def build_client(testnet: bool) -> Client:
    if testnet:
        api_key = os.getenv("API_KEY_TEST")
        api_secret = os.getenv("API_SECRET_TEST")
        return Client(api_key, api_secret, testnet=True)

    api_key = os.getenv("API_KEY")
    api_secret = os.getenv("API_SECRET")
    return Client(api_key, api_secret)


def fetch_futures_klines(
    client: Client,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
) -> list:
    return client.futures_klines(
        symbol=symbol,
        interval=interval,
        startTime=start_ms,
        endTime=end_ms,
        limit=limit,
    )


def collect_last_three_years(
    client: Client,
    symbol: str,
    interval: str,
    limit: int,
    sleep_s: float,
) -> pd.DataFrame:
    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=365 * 3)
    start_ms = int(start_dt.timestamp() * 1000)
    end_ms = int(end_dt.timestamp() * 1000)

    klines = []
    next_start = start_ms
    while next_start < end_ms:
        batch = fetch_futures_klines(
            client=client,
            symbol=symbol,
            interval=interval,
            start_ms=next_start,
            end_ms=end_ms,
            limit=limit,
        )
        if not batch:
            break
        klines.extend(batch)
        next_start = batch[-1][6] + 1
        if len(batch) < limit:
            break
        time.sleep(sleep_s)

    df = pd.DataFrame(klines, columns=COLUMNS)
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("open_time", inplace=True)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download futures candle data for the last 3 years.")
    parser.add_argument("--symbol", default="BTCUSDT", help="Trading pair symbol, e.g. BTCUSDT.")
    parser.add_argument("--interval", default="1h", help="Kline interval, e.g. 1h.")
    parser.add_argument("--output", default="crypto_trading_agent/past_data/futures_3y.csv", help="Output CSV filename.")
    parser.add_argument("--limit", type=int, default=1000, help="Max klines per request (<=1000).")
    parser.add_argument("--sleep", type=float, default=0.6, help="Sleep seconds to respect rate limits.")
    parser.add_argument("--testnet", action="store_true", help="Use Binance testnet credentials.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = build_client(args.testnet)
    df = collect_last_three_years(
        client=client,
        symbol="XRPUSDT",
        interval=args.interval,
        limit=args.limit,
        sleep_s=args.sleep,
    )
    df.to_csv(args.output, index_label="open_time")
    print(f"Saved {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()
