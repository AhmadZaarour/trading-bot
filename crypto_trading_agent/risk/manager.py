from binance.enums import *
from typing import Dict, Optional
from utils.helpers import get_symbol_filters, round_step

def ensure_leverage(symbol: str, leverage: int, client):
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except Exception as e:
        print("Leverage change warning:", e)

def safe_position_size(entry_price: float, sl: float, balance: float, filters: Dict[str, float],
                       leverage: int, risk_per_trade: float, safety: float = 0.95) -> float:
    """
    Returns a *quantity* (base units) sized by risk and capped by leverage.
    """
    step = filters["stepSize"]
    min_qty = filters["minQty"]
    min_notional = float(filters.get("minNotional", 0.0))

    sl_dist = abs(entry_price - sl)
    if sl_dist <= 0:
        return 0.0

    risk_amount = balance * risk_per_trade            # USDT
    qty_risk = risk_amount / sl_dist                  # base units (USDT / USDT-per-unit)

    # Leverage cap (with safety buffer for fees/slippage)
    max_qty_leverage = (balance * leverage * safety) / entry_price
    qty = (min(qty_risk, max_qty_leverage)) * 0.02

    # Round to step and enforce exchange minimums
    qty = round_step(qty, step)
    if qty < min_qty:
        return 0.0
    if min_notional > 0 and (qty * entry_price) < min_notional:
        return 0.0
    return qty

def try_market_order_with_backoff(client, symbol: str, side: str, qty: float, max_retries: int = 4) -> Optional[dict]:
    """
    On insufficient margin, reduce qty by 20% and retry a few times.
    """
    attempt = 0
    while attempt <= max_retries and qty > 0:
        try:
            return client.futures_create_order(
                symbol=symbol,
                side=side,
                type=ORDER_TYPE_MARKET,
                quantity=str(qty),
            )
        except Exception as e:
            msg = str(e)
            print(f"Order attempt {attempt+1} failed: {msg}")
            if "-2019" in msg or "insufficient" in msg.lower():
                qty = round_step(qty * 0.8, get_symbol_filters(symbol)["stepSize"])
                attempt += 1
                continue
            # other errors: give up
            raise
    return None