"""
Abstract broker interface.
All concrete brokers (Upstox, Zerodha, …) must implement this.
Swap brokers by changing broker.primary in config.yaml.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
import pandas as pd
from typing import Optional, List


class BrokerInterface(ABC):
    """Common contract for all broker implementations."""

    # ── Market data ────────────────────────────────────────────────────

    @abstractmethod
    def get_historical_candles(
        self,
        instrument_key: str,
        interval: str,          # "1minute" | "5minute" | "day"
        from_date: str,         # "YYYY-MM-DD"
        to_date: str,
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame with DatetimeIndex (IST, tz-aware)."""
        ...

    @abstractmethod
    def get_expired_expiries(self, instrument_key: str) -> list[str]:
        """Return sorted list of past expiry dates as 'YYYY-MM-DD' strings."""
        ...

    @abstractmethod
    def get_expired_option_key(
        self,
        instrument_key: str,
        expiry: str,
        strike: int,
        option_type: str,       # "CE" | "PE"
    ) -> Optional[str]:
        """Return the broker-specific instrument key for an expired option contract."""
        ...

    @abstractmethod
    def get_expired_option_candles(
        self,
        expired_option_key: str,
        interval: str,
        date_str: str,
    ) -> list:
        """Return raw candle list for an expired option on a given date."""
        ...

    @abstractmethod
    def get_option_chain(
        self,
        instrument_key: str,
        expiry: str,
    ) -> pd.DataFrame:
        """Return live option chain with columns: strike, CE_ltp, PE_ltp, CE_iv, PE_iv, …"""
        ...

    @abstractmethod
    def get_live_quote(self, instrument_key: str) -> dict:
        """Return latest quote dict with keys: ltp, open, high, low, close, volume."""
        ...

    def get_ltp(self, instrument_keys: List[str]) -> dict:
        """
        Return {instrument_key: ltp} for each key in the list.
        Default implementation calls get_live_quote() per key.
        Concrete brokers may override for a batched API call.
        """
        result = {}
        for key in instrument_keys:
            try:
                q = self.get_live_quote(key)
                result[key] = float(q.get("ltp", 0.0))
            except Exception:
                result[key] = 0.0
        return result

    # ── Order management (paper trader uses these as no-ops or logs) ───

    @abstractmethod
    def place_order(
        self,
        instrument_key: str,
        transaction_type: str,  # "BUY" | "SELL"
        quantity: int,
        order_type: str,        # "MARKET" | "LIMIT"
        price: float = 0.0,
        tag: str = "",
    ) -> str:
        """Place order. Returns broker order_id. Raises on failure."""
        ...

    @abstractmethod
    def get_positions(self) -> pd.DataFrame:
        """Return open positions DataFrame."""
        ...

    @abstractmethod
    def get_order_status(self, order_id: str) -> dict:
        """Return order status dict."""
        ...

    @abstractmethod
    def get_available_margin(self) -> float:
        """Return available cash margin for F&O segment (in ₹)."""
        ...
