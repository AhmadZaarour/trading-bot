import os
from binance.client import Client
from dotenv import load_dotenv
# ===============================
# Env / Client (Futures Testnet)
# ===============================
load_dotenv()
API_KEY = os.getenv("API_KEY_testnet")
API_SECRET = os.getenv("API_SECRET_testnet")

if not API_KEY or not API_SECRET:
    raise RuntimeError("Missing API_KEY_testnet / API_SECRET_testnet in .env")

client = Client(API_KEY, API_SECRET, testnet=True)
client.FUTURES_URL = "https://testnet.binancefuture.com/fapi/v1"
client.FUTURES_DATA_URL = "https://testnet.binancefuture.com/fapi/v1"
client.API_URL = "https://testnet.binancefuture.com/fapi/v1"

# ===============================
# Parameters
# ===============================
SYMBOL = "XRPUSDT"
INTERVAL = Client.KLINE_INTERVAL_1MINUTE  # 1m
LOOKBACK = 250            # candles to feed strategy
RISK_PER_TRADE = 0.02
MAX_LEVERAGE = 5
MAX_BARS_PER_TRADE = 20   # exit if exceeded
POLL_SECONDS = 30         # check every 30s for a new closed candle