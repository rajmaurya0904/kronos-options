"""
Data fetcher — wraps broker API, fetches and caches 5-min OHLCV bars.

Key design decisions:
- Always fetches in 1-min bars from Upstox, resamples to 5-min locally
  (Upstox's 5-min endpoint sometimes has gaps; 1-min is more reliable).
- Caches raw CSVs in data/historical/ so you don't burn API calls on reruns.
- For backtesting, call fetch_date_range(); for live, call fetch_latest_bars().

Usage:
    from src.data_fetcher import DataFetcher
    from src.utils import get_broker
    fetcher = DataFetcher(broker=get_broker())
    df = fetcher.fetch_date_range("NIFTY", "2025-01-01", "2025-12-31")
"""
from __future__ import annotations
import logging
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.broker.base import BrokerInterface
from src.utils import IST, is_trading_day, load_config, get_instrument_key

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/historical")


class DataFetcher:
    def __init__(self, broker: BrokerInterface, config_path: str = "config.yaml"):
        self.broker = broker
        self.cfg = load_config(config_path)
        self.config_path = config_path
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _instrument_key(self, symbol: str) -> str:
        """Return instrument key for the active broker (Upstox or Zerodha)."""
        return get_instrument_key(symbol, self.config_path)

    def _cache_path(self, symbol: str, resolution: str) -> Path:
        return CACHE_DIR / f"{symbol}_{resolution}.csv"

    # ── Core fetch ─────────────────────────────────────────────────────

    def fetch_date_range(
        self,
        symbol: str,
        start: str,
        end: str,
        resolution: str = "5min",
        use_cache: bool = True,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV bars for symbol between start and end (inclusive).
        Merges cached data with any new dates to minimise API calls.
        Returns 5-min (or requested resolution) DataFrame, IST tz-aware index.
        """
        cache_path = self._cache_path(symbol, resolution)
        existing = pd.DataFrame()

        if use_cache and cache_path.exists():
            existing = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if not existing.empty:
                existing.index = pd.to_datetime(existing.index, utc=True).tz_convert(IST)

        start_dt = datetime.strptime(start, "%Y-%m-%d").date()
        end_dt   = datetime.strptime(end,   "%Y-%m-%d").date()

        # Find which dates are missing from cache
        dates_needed = [
            start_dt + timedelta(days=i)
            for i in range((end_dt - start_dt).days + 1)
            if is_trading_day(start_dt + timedelta(days=i))
        ]

        if not existing.empty:
            cached_dates = set(existing.index.date)
            dates_needed = [d for d in dates_needed if d not in cached_dates]

        if dates_needed:
            logger.info("Fetching %d missing dates for %s", len(dates_needed), symbol)
            new_frames = []
            for d in dates_needed:
                df_day = self._fetch_one_day(symbol, d.strftime("%Y-%m-%d"), resolution)
                if df_day is not None and not df_day.empty:
                    new_frames.append(df_day)
                time.sleep(self.cfg["broker"].get("request_delay_s", 0.35))

            if new_frames:
                new_data = pd.concat(new_frames).sort_index()
                existing = pd.concat([existing, new_data]).sort_index()
                existing = existing[~existing.index.duplicated(keep="last")]
                existing.to_csv(cache_path)
                logger.info("Cached %d new bars for %s → %s", len(new_data), symbol, cache_path)

        if existing.empty:
            logger.warning("No data returned for %s %s→%s", symbol, start, end)
            return pd.DataFrame()

        # Slice to requested range
        mask = (existing.index.date >= start_dt) & (existing.index.date <= end_dt)
        return existing[mask].copy()

    def _fetch_one_day(
        self,
        symbol: str,
        date_str: str,
        resolution: str = "5min",
    ) -> Optional[pd.DataFrame]:
        """Fetch 1-min bars for one day, resample to target resolution."""
        instrument_key = self._instrument_key(symbol)
        try:
            df = self.broker.get_historical_candles(
                instrument_key, interval="1minute",
                from_date=date_str, to_date=date_str,
            )
        except Exception as e:
            logger.error("Failed to fetch %s on %s: %s", symbol, date_str, e)
            return None

        if df is None or df.empty:
            logger.debug("No data for %s on %s", symbol, date_str)
            return None

        # Resample to requested resolution
        if resolution == "1min":
            return df
        return self._resample(df, resolution)

    @staticmethod
    def _resample(df: pd.DataFrame, resolution: str) -> pd.DataFrame:
        """Resample 1-min OHLCV to a coarser bar (e.g., '5min', '15min')."""
        rule = resolution  # pandas 2.2+ uses "5min" not "5T"
        agg = {
            "open":   "first",
            "high":   "max",
            "low":    "min",
            "close":  "last",
            "volume": "sum",
        }
        if "oi" in df.columns:
            agg["oi"] = "last"
        resampled = df.resample(rule, label="left", closed="left").agg(agg)
        return resampled.dropna(subset=["open", "close"])

    # ── Live bar fetch ─────────────────────────────────────────────────

    def fetch_latest_bars(
        self,
        symbol: str,
        n_bars: int,
        resolution: str = "5min",
    ) -> pd.DataFrame:
        """
        Fetch the most recent n_bars for use in forecasting.
        Combines cached history with today's live bars.
        """
        cache_path = self._cache_path(symbol, resolution)
        if cache_path.exists():
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            df.index = pd.to_datetime(df.index, utc=True).tz_convert(IST)
        else:
            df = pd.DataFrame()

        today = date.today().strftime("%Y-%m-%d")
        today_df = self._fetch_one_day(symbol, today, resolution)
        if today_df is not None and not today_df.empty:
            df = pd.concat([df, today_df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]

        return df.tail(n_bars)

    # ── Option data helpers (for backtester) ──────────────────────────

    def get_option_candle_at(
        self,
        symbol: str,
        date_str: str,
        strike: int,
        option_type: str,
        time_str: str = "09:16",
        expiry: Optional[str] = None,
    ) -> Optional[dict]:
        """
        Fetch the 1-min candle for an option contract at a specific time.
        Used by the backtester to get entry/exit prices from real option data.

        Returns dict: {open, high, low, close, volume} or None if unavailable.
        """
        instrument_key = self._instrument_key(symbol)

        if expiry is None:
            expiries = self.broker.get_expired_expiries(instrument_key)
            valid = [e for e in expiries if e >= date_str]
            if not valid:
                logger.warning("No expiry found for %s on %s", symbol, date_str)
                return None
            expiry = sorted(valid)[0]

        opt_key = self.broker.get_expired_option_key(instrument_key, expiry, strike, option_type)

        if opt_key is None:
            atm_step = self.cfg["instruments"][symbol]["atm_step"]
            for adj in [atm_step, -atm_step, atm_step * 2, -atm_step * 2]:
                opt_key = self.broker.get_expired_option_key(
                    instrument_key, expiry, strike + adj, option_type
                )
                if opt_key:
                    logger.info("Adjusted strike for %s %s to %d", symbol, date_str, strike + adj)
                    break
                time.sleep(self.cfg["broker"].get("request_delay_s", 0.35))

        if opt_key is None:
            return None

        candles = self.broker.get_expired_option_candles(opt_key, "1minute", date_str)
        for c in candles:
            if time_str in str(c[0]):
                return {
                    "timestamp": c[0],
                    "open":      float(c[1]),
                    "high":      float(c[2]),
                    "low":       float(c[3]),
                    "close":     float(c[4]),
                    "volume":    c[5] if len(c) > 5 else 0,
                }
        return None
