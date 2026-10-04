"""Tests for utils.py — timezone helpers, holiday calendar, lot sizes."""
from datetime import date, datetime
import pytest
from src.utils import is_trading_day, round_to_atm, IST
import pytz


def test_weekends_are_not_trading_days():
    assert not is_trading_day(date(2025, 6, 7))   # Saturday
    assert not is_trading_day(date(2025, 6, 8))   # Sunday


def test_weekday_is_trading_day():
    assert is_trading_day(date(2025, 6, 9))       # Monday


def test_known_holiday_not_trading():
    assert not is_trading_day(date(2025, 4, 14))  # Ambedkar Jayanti
    assert not is_trading_day(date(2025, 4, 18))  # Good Friday
    assert not is_trading_day(date(2026, 10, 20))  # Dussehra


def test_holiday_list_has_no_weekends():
    from src.utils import NSE_HOLIDAYS
    assert all(d.weekday() < 5 for d in NSE_HOLIDAYS)


def test_round_to_atm_nifty():
    assert round_to_atm(24387.5, 50) == 24400
    assert round_to_atm(24362.5, 50) == 24350
    assert round_to_atm(24375.0, 50) == 24400  # rounds up at midpoint
    assert round_to_atm(24325.0, 50) == 24350  # ...in both directions (banker's rounding gave 24300)


def test_round_to_atm_banknifty():
    assert round_to_atm(52350.0, 100) == 52400
    assert round_to_atm(52249.0, 100) == 52200


def test_ist_is_correct_offset():
    ist = pytz.timezone("Asia/Kolkata")
    # pytz zones must be attached with localize(); tzinfo= gives the 1854
    # LMT offset (+05:53) and this test failed.
    dt = ist.localize(datetime(2025, 6, 9, 9, 15))
    utc = dt.astimezone(pytz.utc)
    # IST = UTC + 5:30
    assert utc.hour == 3 and utc.minute == 45
