from __future__ import annotations

from .broker_base import Broker
from dotenv import load_dotenv
from binance.client import Client

import os
from decimal import Decimal, ROUND_DOWN
from typing import Any, Dict, Optional, Tuple


class SpotBroker(Broker):
    """
    Spot broker for USDT quote pairs.
    Assumes your strategy decides symbol + TP/SL.
    Broker is responsible for sizing, rounding, placing market/oco orders, and cancellations.
    """

    def __init__(self, testnet: bool = True):
        load_dotenv()
        if testnet:
            self.client = Client(
                os.getenv("API_KEY_TEST"),
                os.getenv("API_SECRET_TEST"),
                testnet=True,
            )
        else:
            self.client = Client(
                os.getenv("API_KEY"),
                os.getenv("API_SECRET"),
            )

        self._symbol_info_cache: Dict[str, Dict[str, Any]] = {}

    # ---------- balances / positions ----------

    def get_usdt_free(self) -> float:
        bal = self.client.get_asset_balance(asset="USDT")
        return float(bal["free"]) if bal else 0.0

    def get_asset_balance(self, asset: str) -> Tuple[float, float]:
        bal = self.client.get_asset_balance(asset=asset)
        if not bal:
            return 0.0, 0.0
        return float(bal["free"]), float(bal["locked"])

    def get_asset_free(self, asset: str) -> float:
        free, _ = self.get_asset_balance(asset)
        return free

    def get_position(self, symbol: str) -> Dict[str, Any]:
        """
        Spot 'position' = base asset free (locked may be in open orders like OCO).
        entryPrice is not provided by spot account; you should track it from fills.
        """
        info = self.get_symbol_info(symbol)
        base = info["baseAsset"]
        free, locked = self.get_asset_balance(base)
        total = free + locked
        return {"amt": total, "entryPrice": None}

    # ---------- symbol info / filters ----------

    def get_symbol_info(self, symbol: str) -> Dict[str, Any]:
        if symbol not in self._symbol_info_cache:
            info = self.client.get_symbol_info(symbol)
            if not info:
                raise ValueError(f"Symbol {symbol} not found")
            self._symbol_info_cache[symbol] = info
        return self._symbol_info_cache[symbol]

    def get_filters(self, symbol: str) -> Dict[str, Any]:
        info = self.get_symbol_info(symbol)
        filt = {f["filterType"]: f for f in info["filters"]}

        lot = filt.get("LOT_SIZE", {})
        price = filt.get("PRICE_FILTER", {})
        notional = filt.get("MIN_NOTIONAL") or filt.get("NOTIONAL") or {}

        return {
            "baseAsset": info["baseAsset"],
            "quoteAsset": info["quoteAsset"],
            "stepSize": Decimal(lot.get("stepSize", "0")),
            "minQty": Decimal(lot.get("minQty", "0")),
            "tickSize": Decimal(price.get("tickSize", "0")),
            "minNotional": (
                Decimal(notional.get("minNotional", "0"))
                if "minNotional" in notional
                else None
            ),
        }

    # ---------- rounding / validation ----------

    @staticmethod
    def _round_down(value: Decimal, step: Decimal) -> Decimal:
        if step == 0:
            return value
        return (value / step).to_integral_value(rounding=ROUND_DOWN) * step

    @staticmethod
    def _format_decimal(value: Decimal) -> str:
        """
        Format Decimal without scientific notation (Binance rejects 1E-8 style strings).
        """
        return format(value, "f")

    def round_qty_price(
        self, symbol: str, qty: Decimal, price: Optional[Decimal] = None
    ) -> Tuple[Decimal, Optional[Decimal]]:
        f = self.get_filters(symbol)
        q = self._round_down(qty, f["stepSize"])
        p = self._round_down(price, f["tickSize"]) if price is not None else None
        return q, p

    def _ensure_min_rules_for_sell(
        self, symbol: str, qty: Decimal, price_for_notional: Decimal
    ) -> None:
        f = self.get_filters(symbol)
        if qty < f["minQty"]:
            raise ValueError(f"qty {qty} < minQty {f['minQty']}")
        if f["minNotional"] is not None:
            notional = qty * price_for_notional
            if notional < f["minNotional"]:
                raise ValueError(
                    f"notional {notional} < minNotional {f['minNotional']} (symbol={symbol})"
                )

    # ---------- orders ----------

    def get_last_price(self, symbol: str) -> Decimal:
        ticker = self.client.get_symbol_ticker(symbol=symbol)
        return Decimal(str(ticker["price"]))

    def market_buy_by_quote(self, symbol: str, quote_usdt: float) -> Dict[str, Any]:
        """
        Market BUY spending quote asset (USDT) via quoteOrderQty.
        """
        f = self.get_filters(symbol)
        if f["quoteAsset"] != "USDT":
            raise ValueError(f"Expected USDT quote, got {f['quoteAsset']} for {symbol}")

        quote = Decimal(str(quote_usdt))
        if f["minNotional"] is not None and quote < f["minNotional"]:
            raise ValueError(f"quote {quote} < minNotional {f['minNotional']}")

        return self.client.create_order(
            symbol=symbol,
            side="BUY",
            type="MARKET",
            quoteOrderQty=self._format_decimal(quote),
            newOrderRespType="FULL",
        )

    def market_buy_by_usdt_percent(
        self,
        symbol: str,
        pct: float,
        fee_buffer_pct: float = 0.002,  # keep ~0.2% USDT unused for fees/slippage
    ) -> Dict[str, Any]:
        usdt_free = Decimal(str(self.get_usdt_free()))
        p = Decimal(str(pct))
        buf = Decimal(str(fee_buffer_pct))

        if p <= 0 or p > 1:
            raise ValueError("pct must be in (0, 1]")

        quote_to_spend = usdt_free * p * (Decimal("1") - buf)
        return self.market_buy_by_quote(symbol, float(quote_to_spend))

    @staticmethod
    def parse_filled_buy(order: Dict[str, Any]) -> Tuple[float, float]:
        """
        Returns (filled_qty_base, avg_price).
        Works for MARKET order responses with executedQty/cummulativeQuoteQty.
        """
        executed_qty = Decimal(order.get("executedQty", "0"))
        cumm_quote = Decimal(order.get("cummulativeQuoteQty", "0"))
        if executed_qty <= 0:
            return 0.0, 0.0
        avg_price = cumm_quote / executed_qty
        return float(executed_qty), float(avg_price)

    def market_sell(self, symbol: str, qty_base: float) -> Dict[str, Any]:
        """
        Market SELL base qty (rounded down to step size).
        """
        q = Decimal(str(qty_base))
        q_rounded, _ = self.round_qty_price(symbol, q)

        if q_rounded <= 0:
            raise ValueError("Rounded quantity is zero.")
        last_price = self.get_last_price(symbol)
        self._ensure_min_rules_for_sell(symbol, q_rounded, price_for_notional=last_price)

        return self.client.create_order(
            symbol=symbol,
            side="SELL",
            type="MARKET",
            quantity=self._format_decimal(q_rounded),
            newOrderRespType="FULL",
        )

    def place_sell_oco(
        self,
        symbol: str,
        qty_base: float,
        take_profit_price: float,
        stop_price: float,
        stop_limit_offset: float = 0.002,  # 0.2% below stopPrice
    ) -> Dict[str, Any]:
        """
        Places a SELL OCO (TP limit + SL stop-limit) for a long spot position.

        You should compute TP/SL in your strategy, then broker rounds to tick/step.
        """
        f = self.get_filters(symbol)

        qty = Decimal(str(qty_base))
        tp = Decimal(str(take_profit_price))
        sp = Decimal(str(stop_price))

        qty, _ = self.round_qty_price(symbol, qty, None)
        if qty <= 0:
            raise ValueError("Rounded quantity is zero.")
        _, tp = self.round_qty_price(symbol, qty, tp)
        _, sp = self.round_qty_price(symbol, qty, sp)

        if tp is None or sp is None:
            raise ValueError("Failed to round prices")

        # stopLimitPrice must be below stopPrice for SELL OCO
        slp = sp * (Decimal("1") - Decimal(str(stop_limit_offset)))
        _, slp = self.round_qty_price(symbol, qty, slp)
        if slp is None:
            raise ValueError("Failed to round stopLimitPrice")
        if slp >= sp:
            # force one tick below if rounding collapsed them
            slp = sp - f["tickSize"]

        # basic sanity (you can tighten these to entry/current checks in your bot)
        if tp <= sp:
            raise ValueError(f"OCO invalid: takeProfit {tp} must be > stopPrice {sp}")

        # strict min rules for the SELL legs (helps avoid rejections)
        # Use TP for notional check; stop leg uses stopLimitPrice
        self._ensure_min_rules_for_sell(symbol, qty, tp)
        self._ensure_min_rules_for_sell(symbol, qty, slp)

        return self.client.create_oco_order(
            symbol=symbol,
            side="SELL",
            quantity=self._format_decimal(qty),
            price=self._format_decimal(tp),
            stopPrice=self._format_decimal(sp),
            stopLimitPrice=self._format_decimal(slp),
            stopLimitTimeInForce="GTC",
        )

    # ---------- “one-call” convenience: buy then protect with OCO ----------

    def enter_long_with_oco(
        self,
        symbol: str,
        usdt_pct: float,
        take_profit_price: float,
        stop_price: float,
        fee_buffer_pct: float = 0.002,
        stop_limit_offset: float = 0.002,
    ) -> Dict[str, Any]:
        """
        Buys using % of USDT, then places a SELL OCO using the filled qty.
        Returns dict with buy_order, filled_qty, avg_price, oco_order.
        """
        buy = self.market_buy_by_usdt_percent(symbol, usdt_pct, fee_buffer_pct=fee_buffer_pct)
        qty, avg_price = self.parse_filled_buy(buy)
        if qty <= 0:
            raise RuntimeError("Buy order did not fill (executedQty=0)")

        oco = self.place_sell_oco(
            symbol=symbol,
            qty_base=qty,
            take_profit_price=take_profit_price,
            stop_price=stop_price,
            stop_limit_offset=stop_limit_offset,
        )

        return {"buy_order": buy, "filled_qty": qty, "avg_price": avg_price, "oco_order": oco}

    # ---------- order management helpers ----------

    def get_open_orders(self, symbol: str) -> list[Dict[str, Any]]:
        return self.client.get_open_orders(symbol=symbol)

    def cancel_order(self, symbol: str, order_id: int) -> Dict[str, Any]:
        return self.client.cancel_order(symbol=symbol, orderId=order_id)

    def cancel_all_orders(self, symbol: str) -> Any:
        """
        Cancels open orders for symbol (includes legs of OCO once created, depending on API behavior).
        """
        return self.client.cancel_open_orders(symbol=symbol)

    def cancel_oco(self, symbol: str, order_list_id: Optional[int] = None, list_client_order_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Cancels an OCO order list. Depending on python-binance version, args may vary.
        Provide either order_list_id or list_client_order_id.
        """
        kwargs = {"symbol": symbol}
        if order_list_id is not None:
            kwargs["orderListId"] = order_list_id
        if list_client_order_id is not None:
            kwargs["listClientOrderId"] = list_client_order_id
        return self.client.cancel_oco_order(**kwargs)
