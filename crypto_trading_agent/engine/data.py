import pandas as pd
from binance.client import Client
import os

class DataProvider:
    def latest_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        raise NotImplementedError


class BinanceDataProvider(DataProvider):
    def __init__(self, testnet: bool = True):

        if testnet:
            api_key = os.getenv("API_KEY_TEST")
            api_secret = os.getenv("API_SECRET_TEST")
            self.client = Client(api_key, api_secret, testnet=True)
        else:
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            self.client = Client(api_key, api_secret)

    def latest_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        klines = self.client.futures_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df


class SpotDataProvider(DataProvider):
    def __init__(self, testnet: bool = True):
        if testnet:
            api_key = os.getenv("API_KEY_TEST")
            api_secret = os.getenv("API_SECRET_TEST")
            self.client = Client(api_key, api_secret, testnet=True)
        else:
            api_key = os.getenv("API_KEY")
            api_secret = os.getenv("API_SECRET")
            self.client = Client(api_key, api_secret)

    def latest_df(self, symbol: str, interval: str, limit: int) -> pd.DataFrame:
        klines = self.client.get_klines(symbol=symbol, interval=interval, limit=limit)

        df = pd.DataFrame(klines, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
        ])

        df["open"] = df["open"].astype(float)
        df["high"] = df["high"].astype(float)
        df["low"] = df["low"].astype(float)
        df["close"] = df["close"].astype(float)
        df["volume"] = df["volume"].astype(float)

        df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
        df.set_index("open_time", inplace=True)
        return df
