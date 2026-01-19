from typing import Any, Dict

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