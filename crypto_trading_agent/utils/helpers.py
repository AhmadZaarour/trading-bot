import numpy as np

# ===============================
# Binance utilities (futures)
# ===============================
def get_symbol_filters(symbol: str, client):
    info = client.futures_exchange_info()
    sym = next(s for s in info["symbols"] if s["symbol"] == symbol)
    lot = next(f for f in sym["filters"] if f["filterType"] in ("LOT_SIZE","MARKET_LOT_SIZE"))
    price = next(f for f in sym["filters"] if f["filterType"] == "PRICE_FILTER")
    # MIN_NOTIONAL is present on many symbols; guard if not
    min_notional = 0.0
    for f in sym["filters"]:
        if f["filterType"] in ("MIN_NOTIONAL","NOTIONAL"):
            min_notional = float(f.get("notional", f.get("minNotional", 0.0)))
            break
    return {
        "stepSize": float(lot["stepSize"]),
        "minQty": float(lot["minQty"]),
        "tickSize": float(price["tickSize"]),
        "minNotional": min_notional,
    }


def round_step(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    precision = int(round(-np.log10(step)))  # digits allowed
    return float(np.round(np.floor(qty / step) * step, precision))

def round_tick(price: float, tick: float) -> float:
    if tick <= 0:
        return price
    precision = int(round(-np.log10(tick)))  # digits allowed
    return float(np.round(np.floor(price / tick) * tick, precision))


def get_usdt_available(client) -> float:
    # Use available balance, not wallet balance
    acct = client.futures_account()
    usdt = next(a for a in acct["assets"] if a["asset"] == "USDT")
    return float(usdt["availableBalance"])

def get_position_amt(symbol: str, client) -> float:
    try:
        pos = client.futures_position_information(symbol=symbol)
        if not pos:  # empty list
            return 0.0
        amt = float(pos[0].get("positionAmt", 0.0))
        return amt
    except Exception as e:
        print(f"[WARN] get_position_amt failed: {e}")
        return 0.0