"""
Transaction costs and P&L for option legs — shared by the backtester and the
paper trader so both report the same numbers.

Every leg is a round trip: opened at entry_price, closed at exit_price, same
quantity. A BUY leg buys first and sells later; a SELL leg sells first and buys
back later. Charges are computed on the actual buy and sell turnover of that
leg, so credit spreads and condors are costed correctly (STT falls on the sell
side whichever comes first).
"""
from __future__ import annotations

OPTION_TICK = 0.05  # NSE/BSE index options tick size (₹)


def leg_pnl(action: str, entry_price: float, exit_price: float, qty: int) -> float:
    """Gross P&L of one leg in ₹. qty = lot_size × lots."""
    sign = 1 if action == "BUY" else -1
    return sign * (exit_price - entry_price) * qty


def leg_charges(
    action: str,
    entry_price: float,
    exit_price: float,
    qty: int,
    cfg: dict,
    exchange: str = "NSE",
) -> float:
    """Round-trip statutory charges + brokerage for one leg, in ₹."""
    c = cfg.get("costs", {})
    buy_price, sell_price = (entry_price, exit_price) if action == "BUY" else (exit_price, entry_price)
    buy_t, sell_t = buy_price * qty, sell_price * qty
    turnover = buy_t + sell_t

    exch_pct = c.get("exchange_txn_options_pct", {})
    if isinstance(exch_pct, dict):
        exch_pct = exch_pct.get(exchange, exch_pct.get("NSE", 0.03553))

    brokerage = c.get("brokerage_per_order", 20.0) * 2          # open + close
    stt       = sell_t * c.get("stt_sell_options_pct", 0.15) / 100
    exch      = turnover * exch_pct / 100
    sebi      = turnover * c.get("sebi_turnover_pct", 0.0001) / 100
    gst       = (brokerage + exch + sebi) * c.get("gst_pct", 18.0) / 100
    stamp     = buy_t * c.get("stamp_duty_buy_pct", 0.003) / 100
    return brokerage + stt + exch + sebi + gst + stamp


def leg_slippage(qty: int, cfg: dict) -> float:
    """Slippage cost for one leg in ₹: `slippage_ticks` against you on entry and on exit."""
    ticks = cfg.get("costs", {}).get("slippage_ticks", 1)
    return ticks * OPTION_TICK * qty * 2


def trade_result(legs: list[dict], lot_size: int, cfg: dict, exchange: str = "NSE") -> dict:
    """
    Net result of a multi-leg trade.

    legs: [{action, lots, entry_price, exit_price, (lot_size)}]; a leg's own
          lot_size (from the contract) wins over the instrument default.
    """
    raw = charges = slip = 0.0
    for leg in legs:
        qty = int(leg.get("lot_size") or lot_size) * int(leg.get("lots", 1))
        raw     += leg_pnl(leg["action"], leg["entry_price"], leg["exit_price"], qty)
        charges += leg_charges(leg["action"], leg["entry_price"], leg["exit_price"], qty, cfg, exchange)
        slip    += leg_slippage(qty, cfg)
    return {
        "raw_pnl_rs": round(raw, 2),
        "charges_rs": round(charges + slip, 2),
        "net_pnl_rs": round(raw - charges - slip, 2),
    }
