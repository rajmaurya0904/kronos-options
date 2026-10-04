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
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

from src.broker.base import BrokerInterface
from src.utils import IST, is_trading_day, load_config, today_ist

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/historical")

# Calendar days of history to make sure exist before a live forecast:
# 400 five-minute bars ≈ 5.3 sessions, so three weeks covers holidays too.
LIVE_HISTORY_DAYS = 21


def _hhmm_ist(ts) -> str:
    """Candle timestamp (ISO string, offset or not) -> 'HH:MM' in IST."""
    t = pd.Timestamp(ts)
    t = t.tz_localize(IST) if t.tzinfo is None else t.tz_convert(IST)
    return t.strftime("%H:%M")


class DataFetcher:
    def __init__(self, broker: BrokerInterface, config_path: str = "config.yaml"):
        self.broker = broker
        self.cfg = load_config(config_path)
        self._option_day_cache: dict[tuple, list] = {}
        CACHE_DIR.mkdir(parents=True, exist_ok=True)

    def _instrument_key(self, symbol: str) -> str:
        return self.cfg["instruments"][symbol]["upstox_key"]

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

        # Find which dates are missing from cache. Today is never fetched here:
        # the historical endpoint only serves completed sessions, and caching a
        # half-built day would make it look "done" to every later run.
        today = today_ist()
        dates_needed = [
            start_dt + timedelta(days=i)
            for i in range((end_dt - start_dt).days + 1)
            if is_trading_day(start_dt + timedelta(days=i))
            and start_dt + timedelta(days=i) < today
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
        # Completed sessions: top up the cache so the lookback window is full
        # even on a first run (previously an empty cache meant forecasting off
        # a handful of bars).
        today = today_ist()
        start = (today - timedelta(days=LIVE_HISTORY_DAYS)).strftime("%Y-%m-%d")
        end   = (today - timedelta(days=1)).strftime("%Y-%m-%d")
        df = self.fetch_date_range(symbol, start, end, resolution)

        # Today's bars come from the intraday endpoint — the historical one
        # never includes the current session, so without this every forecast
        # was made off yesterday's close.
        try:
            intraday = self.broker.get_intraday_candles(self._instrument_key(symbol), "1minute")
        except Exception as e:
            logger.error("Intraday fetch failed for %s: %s", symbol, e)
            intraday = pd.DataFrame()
        if intraday is not None and not intraday.empty:
            today_df = intraday if resolution == "1min" else self._resample(intraday, resolution)
            # Drop the bar still forming: its OHLC would change after the forecast.
            now = pd.Timestamp.now(tz=IST)
            bar_len = pd.Timedelta(resolution)
            today_df = today_df[today_df.index + bar_len <= now]
            df = pd.concat([df, today_df]).sort_index()
            df = df[~df.index.duplicated(keep="last")]
        else:
            logger.warning("%s: no intraday bars for today yet", symbol)

        return df.tail(n_bars)

    # ── Option data helpers (for backtester) ──────────────────────────

    def get_option_price_at(
        self,
        symbol: str,
        date_str: str,
        strike: int,
        option_type: str,
        time_str: str,
        expiry: str,
    ) -> Optional[tuple[float, int]]:
        """
        Real traded price of an expired option at HH:MM on date_str.

        Uses the open of the 1-min candle starting at time_str; if that minute
        did not trade, the close of the last candle before it (never a later
        one — that would be lookahead).
        Returns (price, contract_lot_size) or None if the contract/candles are
        unavailable. The strike is never shifted: pricing a spread leg at a
        neighbouring strike silently changes the trade.
        """
        instrument_key = self._instrument_key(symbol)
        contract = self.broker.get_expired_option_contract(instrument_key, expiry, strike, option_type)
        if not contract:
            return None
        opt_key = contract.get("instrument_key") or contract.get("instrumentKey")
        lot_size = int(contract.get("lot_size") or contract.get("lotSize") or 0)

        cache_key = (opt_key, date_str)
        if cache_key not in self._option_day_cache:
            candles = self.broker.get_expired_option_candles(opt_key, "1minute", date_str)
            # Normalise to ("HH:MM", open, close), ascending by time.
            rows = sorted(
                (_hhmm_ist(c[0]), float(c[1]), float(c[4]))
                for c in candles
            )
            self._option_day_cache[cache_key] = rows
        rows = self._option_day_cache[cache_key]

        price = None
        for hhmm, open_, close in rows:
            if hhmm == time_str:
                price = open_
                break
            if hhmm > time_str:
                break
            price = close
        if price is None or price <= 0:
            return None
        return price, lot_size
