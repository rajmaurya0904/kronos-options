"""Tests for costs.py — per-leg P&L and charges for multi-leg option trades."""
import pytest
from src.costs import leg_pnl, leg_charges, trade_result

CFG = {
    "costs": {
        "brokerage_per_order": 20.0,
        "stt_sell_options_pct": 0.15,
        "exchange_txn_options_pct": {"NSE": 0.03553, "BSE": 0.0325},
        "gst_pct": 18.0,
        "sebi_turnover_pct": 0.0001,
        "stamp_duty_buy_pct": 0.003,
        "slippage_ticks": 0,
    }
}


def test_long_leg_pnl():
    assert leg_pnl("BUY", 100.0, 120.0, 65) == pytest.approx(1300.0)


def test_short_leg_pnl():
    assert leg_pnl("SELL", 100.0, 80.0, 65) == pytest.approx(1300.0)


def test_spread_pnl_is_not_multiplied_by_leg_count():
    # Bull put spread: sell 24000 PE @100 -> 70, buy 23950 PE @80 -> 60
    legs = [
        {"action": "SELL", "lots": 1, "entry_price": 100.0, "exit_price": 70.0},
        {"action": "BUY",  "lots": 1, "entry_price": 80.0,  "exit_price": 60.0},
    ]
    res = trade_result(legs, 65, CFG)
    # +30 on the short, -20 on the long = +10 per unit
    assert res["raw_pnl_rs"] == pytest.approx(10 * 65)


def test_stt_only_on_sell_side():
    qty = 65
    buy = leg_charges("BUY", 100.0, 100.0, qty, CFG)
    no_stt = dict(CFG, costs=dict(CFG["costs"], stt_sell_options_pct=0.0))
    assert buy - leg_charges("BUY", 100.0, 100.0, qty, no_stt) == pytest.approx(100.0 * qty * 0.0015)


def test_leg_lot_size_overrides_default():
    legs = [{"action": "BUY", "lots": 1, "lot_size": 75, "entry_price": 10.0, "exit_price": 11.0}]
    assert trade_result(legs, 65, CFG)["raw_pnl_rs"] == pytest.approx(75.0)


def test_charges_positive():
    assert leg_charges("SELL", 50.0, 40.0, 65, CFG, "BSE") > 40.0  # at least brokerage
