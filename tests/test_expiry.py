"""Tests for expiry-day detection."""
from datetime import date
from src.data_cleaner import is_expiry_day

CFG = {"instruments": {
    "NIFTY": {"expiry_day": "Tuesday"},
    "BANKNIFTY": {"expiry_day": "Tuesday", "expiry_cycle": "monthly"},
}}


def test_broker_list_wins_over_weekday():
    # A holiday-shifted Monday expiry is still an expiry day
    assert is_expiry_day(date(2026, 3, 30), "NIFTY", CFG, ["2026-03-30"])
    assert not is_expiry_day(date(2026, 3, 31), "NIFTY", CFG, ["2026-03-30"])


def test_weekly_fallback():
    assert is_expiry_day(date(2026, 6, 9), "NIFTY", CFG)       # a Tuesday
    assert not is_expiry_day(date(2026, 6, 10), "NIFTY", CFG)


def test_monthly_fallback_only_last_tuesday():
    assert is_expiry_day(date(2026, 6, 30), "BANKNIFTY", CFG)      # last Tuesday of June
    assert not is_expiry_day(date(2026, 6, 23), "BANKNIFTY", CFG)
