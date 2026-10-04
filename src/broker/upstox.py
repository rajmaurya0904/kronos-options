"""
Upstox broker implementation (v2 REST, v3 for candles).

API reference: https://upstox.com/developer/api-documentation/
All endpoints require Bearer token in Authorization header.
Token is valid for one trading day — refresh daily via login flow.

Key instrument key formats:
  NSE index:  "NSE_INDEX|Nifty 50"
  BSE index:  "BSE_INDEX|SENSEX"
  NSE equity: "NSE_EQ|INFY"
  NSE option: "NSE_FO|<token>" (look up via /option/contract or the
              expired-instruments API — never build it by hand)
"""
from __future__ import annotations
import time
import urllib.parse
import logging
from typing import Optional

import requests
import pandas as pd

from .base import BrokerInterface

logger = logging.getLogger(__name__)

BASE_URL = "https://api.upstox.com/v2"
BASE_V3  = "https://api.upstox.com/v3"

# v2 interval name -> v3 (unit, interval). The v2 candle endpoint is deprecated
# and only serves one month of 1-minute history; v3 goes back to Jan 2022.
_V3_INTERVALS = {
    "1minute":  ("minutes", 1),
    "5minute":  ("minutes", 5),
    "30minute": ("minutes", 30),
    "day":      ("days", 1),
    "week":     ("weeks", 1),
    "month":    ("months", 1),
}

_CANDLE_COLS = ["timestamp", "open", "high", "low", "close", "volume", "oi"]


class UpstoxBroker(BrokerInterface):
    """
    Upstox REST API implementation.

    Usage:
        broker = UpstoxBroker(access_token=os.getenv("UPSTOX_ACCESS_TOKEN"))
    """

    def __init__(
        self,
        access_token: str,
        request_delay_s: float = 0.35,
        timeout_s: int = 10,
        max_retries: int = 3,
    ):
        self.access_token = access_token
        self.delay = request_delay_s
        self.timeout = timeout_s
        self.max_retries = max_retries
        # Contract lists change at most once a day; the backtester asks for the
        # same expiry's contracts for every leg of every trade, so cache them.
        self._contracts_cache: dict[tuple, list] = {}
        self._expired_contracts_cache: dict[tuple, list] = {}
        self._expired_expiries_cache: dict[str, list] = {}

    # ── Internal helpers ───────────────────────────────────────────────

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self.access_token}",
            "Accept": "application/json",
        }

    def _get(self, url: str) -> dict:
        """GET with retry and rate-limit delay. Returns parsed JSON, or {} on failure.

        Retries only what can succeed on retry (network errors, 429, 5xx).
        A 401/403/404 is final: retrying an expired token just burns quota.
        """
        time.sleep(self.delay)
        for attempt in range(self.max_retries):
            try:
                r = requests.get(url, headers=self._headers(), timeout=self.timeout)
                if r.status_code == 200:
                    return r.json()
                if r.status_code == 401:
                    logger.error("Upstox rejected the access token (401). "
                                 "Refresh UPSTOX_ACCESS_TOKEN in .env.")
                    return {}
                if r.status_code != 429 and r.status_code < 500:
                    logger.warning("GET %s → HTTP %s: %s", url, r.status_code, r.text[:200])
                    return {}
                logger.warning("GET %s → HTTP %s (attempt %d)", url, r.status_code, attempt + 1)
            except requests.RequestException as e:
                logger.warning("GET %s failed: %s (attempt %d)", url, e, attempt + 1)
            time.sleep(1.0 * (attempt + 1))
        logger.error("All retries exhausted for %s", url)
        return {}

    @staticmethod
    def _encode(instrument_key: str) -> str:
        return urllib.parse.quote(instrument_key, safe="")

    @staticmethod
    def _candles_to_df(candles: list) -> pd.DataFrame:
        if not candles:
            return pd.DataFrame()
        df = pd.DataFrame(candles, columns=_CANDLE_COLS[: len(candles[0])])
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert("Asia/Kolkata")
        df.set_index("timestamp", inplace=True)
        return df.sort_index()

    # ── Market data ────────────────────────────────────────────────────

    def get_historical_candles(
        self,
        instrument_key: str,
        interval: str,
        from_date: str,
        to_date: str,
    ) -> pd.DataFrame:
        """
        Fetch completed-day OHLCV candles (today's bars are NOT included —
        use get_intraday_candles for those).
        interval: "1minute" | "5minute" | "30minute" | "day" | "week" | "month"
        Returns DataFrame with columns [open, high, low, close, volume, oi]
        and a DatetimeIndex in IST timezone.
        """
        unit, n = _V3_INTERVALS[interval]
        encoded = self._encode(instrument_key)
        url = f"{BASE_V3}/historical-candle/{encoded}/{unit}/{n}/{to_date}/{from_date}"
        data = self._get(url)
        candles = data.get("data", {}).get("candles", [])
        if not candles:
            logger.debug("No candles for %s %s %s→%s", instrument_key, interval, from_date, to_date)
        return self._candles_to_df(candles)

    def get_intraday_candles(self, instrument_key: str, interval: str = "1minute") -> pd.DataFrame:
        """Today's candles so far (the historical endpoint never returns today)."""
        unit, n = _V3_INTERVALS[interval]
        encoded = self._encode(instrument_key)
        data = self._get(f"{BASE_V3}/historical-candle/intraday/{encoded}/{unit}/{n}")
        return self._candles_to_df(data.get("data", {}).get("candles", []))

    # ── Live (unexpired) option contracts ──────────────────────────────

    def get_option_contracts(self, instrument_key: str, expiry: Optional[str] = None) -> list[dict]:
        """All listed option contracts for an underlying (optionally one expiry)."""
        key = (instrument_key, expiry)
        if key not in self._contracts_cache:
            url = f"{BASE_URL}/option/contract?instrument_key={self._encode(instrument_key)}"
            if expiry:
                url += f"&expiry_date={expiry}"
            self._contracts_cache[key] = self._get(url).get("data", []) or []
        return self._contracts_cache[key]

    def get_expiries(self, instrument_key: str) -> list[str]:
        """Current and future expiry dates, 'YYYY-MM-DD', ascending."""
        return sorted({c["expiry"] for c in self.get_option_contracts(instrument_key) if c.get("expiry")})

    def get_option_contract(
        self, instrument_key: str, expiry: str, strike: int, option_type: str,
    ) -> Optional[dict]:
        """The listed contract for (expiry, strike, CE/PE), or None."""
        for c in self.get_option_contracts(instrument_key, expiry):
            if float(c.get("strike_price", -1)) == float(strike) and c.get("instrument_type") == option_type:
                return c
        return None

    # ── Expired contracts (backtesting) ────────────────────────────────

    def get_expired_expiries(self, instrument_key: str) -> list[str]:
        """Return list of past expiry dates as 'YYYY-MM-DD' strings, sorted ascending."""
        if instrument_key not in self._expired_expiries_cache:
            encoded = self._encode(instrument_key)
            data = self._get(f"{BASE_URL}/expired-instruments/expiries?instrument_key={encoded}")
            self._expired_expiries_cache[instrument_key] = sorted(data.get("data", []) or [])
        return self._expired_expiries_cache[instrument_key]

    def get_expired_option_contract(
        self, instrument_key: str, expiry: str, strike: int, option_type: str,
    ) -> Optional[dict]:
        """The expired contract for (expiry, strike, CE/PE) — includes its lot_size."""
        key = (instrument_key, expiry)
        if key not in self._expired_contracts_cache:
            url = (
                f"{BASE_URL}/expired-instruments/option/contract"
                f"?instrument_key={self._encode(instrument_key)}&expiry_date={expiry}"
            )
            self._expired_contracts_cache[key] = self._get(url).get("data", []) or []
        for contract in self._expired_contracts_cache[key]:
            sp = float(contract.get("strike_price") or contract.get("strikePrice") or -1)
            itype = contract.get("instrument_type") or contract.get("instrumentType", "")
            if sp == float(strike) and itype == option_type:
                return contract
        return None

    def get_expired_option_key(
        self,
        instrument_key: str,
        expiry: str,
        strike: int,
        option_type: str,
    ) -> Optional[str]:
        """
        Find the Upstox instrument key for an expired option contract.
        Returns None if not found.
        """
        c = self.get_expired_option_contract(instrument_key, expiry, strike, option_type)
        if c is None:
            return None
        return c.get("instrument_key") or c.get("instrumentKey")

    def get_expired_option_candles(
        self,
        expired_option_key: str,
        interval: str,
        date_str: str,
    ) -> list:
        """
        Fetch raw candles for an expired option on a specific date.
        Returns raw list of [timestamp, open, high, low, close, volume, oi].
        """
        encoded = self._encode(expired_option_key)
        url = (
            f"{BASE_URL}/expired-instruments/historical-candle"
            f"/{encoded}/{interval}/{date_str}/{date_str}"
        )
        data = self._get(url)
        return data.get("data", {}).get("candles", [])

    def get_option_chain(self, instrument_key: str, expiry: str) -> pd.DataFrame:
        """
        Fetch live option chain for given expiry.
        Returns DataFrame with columns: strike, CE_ltp, CE_iv, CE_oi, PE_ltp, PE_iv, PE_oi
        NOTE: Only works for non-expired (current) expiries.
        """
        encoded = self._encode(instrument_key)
        url = f"{BASE_URL}/option/chain?instrument_key={encoded}&expiry_date={expiry}"
        data = self._get(url)
        rows = []
        for item in data.get("data", []) or []:
            ce = item.get("call_options") or {}
            pe = item.get("put_options") or {}
            rows.append({
                "strike": item.get("strike_price"),
                "CE_ltp": (ce.get("market_data") or {}).get("ltp"),
                "CE_iv":  (ce.get("option_greeks") or {}).get("iv"),
                "CE_oi":  (ce.get("market_data") or {}).get("oi"),
                "CE_key": ce.get("instrument_key"),
                "PE_ltp": (pe.get("market_data") or {}).get("ltp"),
                "PE_iv":  (pe.get("option_greeks") or {}).get("iv"),
                "PE_oi":  (pe.get("market_data") or {}).get("oi"),
                "PE_key": pe.get("instrument_key"),
            })
        return pd.DataFrame(rows)

    def get_live_quote(self, instrument_key: str) -> dict:
        """Fetch latest LTP and OHLCV for an instrument."""
        encoded = self._encode(instrument_key)
        url = f"{BASE_URL}/market-quote/quotes?instrument_key={encoded}"
        data = self._get(url)
        quotes = data.get("data", {})
        # Upstox returns dict keyed by instrument_key
        for key, val in quotes.items():
            return {
                "ltp":    val.get("last_price"),
                "open":   val.get("ohlc", {}).get("open"),
                "high":   val.get("ohlc", {}).get("high"),
                "low":    val.get("ohlc", {}).get("low"),
                "close":  val.get("ohlc", {}).get("close"),
                "volume": val.get("volume"),
            }
        return {}

    # ── Order management ───────────────────────────────────────────────

    def place_order(
        self,
        instrument_key: str,
        transaction_type: str,
        quantity: int,
        order_type: str,
        price: float = 0.0,
        tag: str = "",
    ) -> str:
        """
        Place a real order via Upstox.
        ONLY called by live_trader.py — paper_trader.py logs without calling this.
        """
        if "_INDEX|" in instrument_key or not instrument_key:
            # An index is not tradable; this would only ever be a resolution bug.
            raise ValueError(f"Refusing to place an order on {instrument_key!r}")
        payload = {
            "quantity": quantity,
            "product": "I",                    # Intraday (MIS). "D" is delivery/carry-forward.
            "validity": "DAY",
            "price": price,
            "tag": tag,
            "instrument_token": instrument_key,
            "order_type": order_type,
            "transaction_type": transaction_type,
            "disclosed_quantity": 0,
            "trigger_price": 0,
            "is_amo": False,
        }
        r = requests.post(
            f"{BASE_URL}/order/place",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout,
        )
        if r.status_code == 200:
            oid = (r.json().get("data") or {}).get("order_id", "")
            if not oid:
                raise RuntimeError(f"Order accepted without an order_id: {r.text[:300]}")
            return oid
        raise RuntimeError(f"Order placement failed: {r.status_code} {r.text[:300]}")

    def get_positions(self) -> pd.DataFrame:
        data = self._get(f"{BASE_URL}/portfolio/short-term-positions")
        return pd.DataFrame(data.get("data", []))

    def get_order_status(self, order_id: str) -> dict:
        data = self._get(f"{BASE_URL}/order/details?order_id={urllib.parse.quote(order_id)}")
        return data.get("data", {})
