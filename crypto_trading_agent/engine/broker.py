import os
from binance.client import Client
from typing import Dict, Any
import numpy as np
from dotenv import load_dotenv

class Broker:
    def get_filters(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def get_balance(self) -> float:
        raise NotImplementedError

    def get_position(self, symbol: str) -> Dict[str, Any]:
        raise NotImplementedError

    def market_order(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        raise NotImplementedError

    def place_exit_orders(self, symbol: str, side: str, qty: float, sl: float, tp: float) -> None:
        raise NotImplementedError

    def close_all_positions(self, symbol: str) -> None:
        raise NotImplementedError
    
    def round_step(self, qty: float, step: float) -> float:
        raise NotImplementedError

    def round_tick(self, price: float, tick: float) -> float:
        raise NotImplementedError
    


class BinanceFuturesBroker(Broker):
    """
    Binance Futures Testnet broker adapter.
    """

    def __init__(self, testnet: bool = True):
        
        load_dotenv()
        api_key = os.getenv("API_KEY_testnet")
        api_secret = os.getenv("API_SECRET_testnet")

        if testnet:
            self.client = Client(api_key, api_secret, testnet=True)
        else:
            self.client = Client(api_key, api_secret)

    def get_filters(self, symbol: str) -> Dict[str, Any]:
        info = self.client.futures_exchange_info()
        for s in info["symbols"]:
            if s["symbol"] == symbol:
                return {
                    "stepSize": float(s["filters"][1]["stepSize"]),
                    "tickSize": float(s["filters"][0]["tickSize"]),
                    "minQty": float(s["filters"][1]["minQty"]),
                }
        raise ValueError(f"Symbol {symbol} not found")

    def get_balance(self) -> float:
        balances = self.client.futures_account_balance()
        usdt = next(b for b in balances if b["asset"] == "USDT")
        return float(usdt["balance"])

    def get_position(self, symbol: str) -> Dict[str, Any]:
        positions = self.client.futures_position_information(symbol=symbol)
        if not positions:
            return {"amt": 0.0, "entryPrice": 0.0}
        p = positions[0]
        return {
            "amt": float(p["positionAmt"]),
            "entryPrice": float(p["entryPrice"]),
        }

    def market_order(self, symbol: str, side: str, qty: float) -> Dict[str, Any]:
        return self.client.futures_create_order(
            symbol=symbol,
            side=side,
            type="MARKET",
            quantity=qty
        )

    def place_exit_orders(self, symbol: str, side: str, qty: float, sl: float, tp: float) -> None:
        opp_side = "SELL" if side == "BUY" else "BUY"

        # Stop loss
        self.client.futures_create_order(
            symbol=symbol,
            side=opp_side,
            type="STOP_MARKET",
            stopPrice=str(sl),
            quantity=str(qty),
            reduceOnly=True
        )

        # Take profit
        self.client.futures_create_order(
            symbol=symbol,
            side=opp_side,
            type="TAKE_PROFIT_MARKET",
            stopPrice=str(tp),
            quantity=str(qty),
            reduceOnly=True
        )

    def close_all_positions(self, symbol: str) -> None:
        self.client.futures_cancel_all_open_orders(symbol=symbol)

    def round_step(self, qty, step):
        if step <= 0:
            return qty
        precision = int(round(-np.log10(step)))  # digits allowed
        return float(np.round(np.floor(qty / step) * step, precision))
    
    def round_tick(self, price, tick):
        if tick <= 0:
            return price
        precision = int(round(-np.log10(tick)))  # digits allowed
        return float(np.round(np.floor(price / tick) * tick, precision))
