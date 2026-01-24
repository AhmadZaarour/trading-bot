from typing import Dict, Optional

class RiskManager():
    def __init__(self, broker):
        self.broker = broker

    def ensure_leverage(self, symbol: str, leverage: int):

        try:
            self.broker.set_leverage(symbol=symbol, leverage=leverage)
            print(f"Leverage set to {leverage}x for {symbol}.")

        except Exception as e:
            print("Leverage change warning:", e)

    def safe_position_size(
        self,
        entry_price: float,
        sl: float,
        balance: float,
        filters: Dict[str, float],
        leverage: int,
        risk_per_trade: float,
        safety: float = 0.95,
        taker_fee: float = 0.0004,      # match your backtest
        slippage_pct: float = 0.0005,   # e.g. 5 bps => 0.0005
        gap_buffer_mult: float = 1.1,   # assume stop can slip ~10% worse than planned risk
        use_market_step: bool = True,
    ) -> float:
        """
        Quantity sized by risk (SL distance) and capped by leverage/margin.
        Adds realistic buffers for fees/slippage and stop gaps.
        """
        # Prefer market stepSize if your broker provides it; else fallback
        step = float(filters.get("marketStepSize" if use_market_step else "stepSize", filters["stepSize"]))
        min_qty = float(filters["minQty"])
        min_notional = float(filters.get("minNotional", 0.0))

        sl_dist = abs(entry_price - sl)
        if sl_dist <= 0 or balance <= 0 or leverage <= 0 or risk_per_trade <= 0:
            return 0.0

        # Risk sizing (buffered for stop slippage/gaps)
        risk_amount = balance * float(risk_per_trade)
        qty_risk = risk_amount / (sl_dist * gap_buffer_mult)

        # Cost-aware leverage cap
        # Budget costs on both entry and exit (approx taker + taker) + slippage
        cost_buffer = (2 * taker_fee) + (2 * slippage_pct)
        effective_safety = safety * (1.0 - cost_buffer)
        if effective_safety <= 0:
            effective_safety = safety * 0.9

        max_qty_leverage = (balance * leverage * effective_safety) / entry_price
        qty = min(qty_risk, max_qty_leverage)

        qty = self.broker.round_step(qty, step)
        if qty < min_qty:
            return 0.0

        # Use a slightly worse execution price for notional check
        worse_entry = entry_price * (1.0 + slippage_pct)
        if min_notional > 0 and (qty * worse_entry) < min_notional:
            return 0.0

        return qty

    def try_market_order_with_backoff(self, symbol: str, side: str, qty: float, max_retries: int = 4) -> Optional[dict]:
        attempt = 0
        filters = self.broker.get_symbol_filters(symbol)  # do once
        step = filters["stepSize"]

        while attempt < max_retries and qty > 0:
            try:
                return self.broker.market_order(symbol, side, qty)
            except Exception as e:
                msg = str(e).lower()
                print(f"Order attempt {attempt+1} failed: {e}")

                if "-2019" in msg or "insufficient" in msg:
                    qty = self.broker.round_step(qty * 0.8, step)
                    attempt += 1
                    continue
                raise

        return None