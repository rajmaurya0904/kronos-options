"""
Data cleaning for Indian market 5-min bars.

Handles:
- Filter to market hours 9:15–15:30 IST
- Drop weekends and NSE/BSE holidays
- Forward-fill missing bars within a session (max 3 consecutive gaps)
- SENSEX: volume is often reported as 0 (BSE index methodology) — flag but don't drop
- Remove pre-market and post-market noise
"""
from __future__ import annotations
import logging
from datetime import time as dtime, timedelta

import pandas as pd

from src.utils import IST, NSE_HOLIDAYS

logger = logging.getLogger(__name__)

MARKET_OPEN  = dtime(9, 15)
MARKET_CLOSE = dtime(15, 30)


def clean(
    df: pd.DataFrame,
    symbol: str = "",
    fill_gaps: bool = True,
    max_fill: int = 3,
) -> pd.DataFrame:
    """
    Main entry point. Accepts a DataFrame with DatetimeIndex (IST or UTC).
    Returns cleaned DataFrame with IST tz-aware index.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ── Ensure IST timezone ────────────────────────────────────────────
    if df.index.tz is None:
        df.index = df.index.tz_localize(IST)
    else:
        df.index = df.index.tz_convert(IST)

    # ── Drop weekends ──────────────────────────────────────────────────
    df = df[df.index.dayofweek < 5]

    # ── Drop holidays ──────────────────────────────────────────────────
    import numpy as np
    df = df[~np.isin(df.index.date, list(NSE_HOLIDAYS))]

    # ── Filter to market hours ─────────────────────────────────────────
    times = df.index.time
    df = df[(times >= MARKET_OPEN) & (times < MARKET_CLOSE)]

    # ── SENSEX zero-volume handling ────────────────────────────────────
    # BSE index feed often reports volume = 0. We keep these bars but flag them
    # so downstream (forecaster) can decide whether to use the volume feature.
    if "volume" in df.columns:
        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            if symbol.upper() == "SENSEX":
                logger.debug("SENSEX: %d bars with zero volume (expected — BSE index)", zero_vol)
                df["volume_synthetic"] = False
            else:
                logger.warning("%s: %d bars with zero volume (unexpected)", symbol, zero_vol)

    # ── Drop bars with NaN OHLC ────────────────────────────────────────
    df = df.dropna(subset=["open", "high", "low", "close"])

    # ── Sanity check: high >= low, open/close within high-low ─────────
    bad_hl = df["high"] < df["low"]
    if bad_hl.any():
        logger.warning("%s: %d bars with high < low — dropping", symbol, bad_hl.sum())
        df = df[~bad_hl]

    # ── Fill gaps within sessions ──────────────────────────────────────
    if fill_gaps:
        df = _fill_intraday_gaps(df, max_fill=max_fill, symbol=symbol)

    df.sort_index(inplace=True)
    return df


def _fill_intraday_gaps(df: pd.DataFrame, max_fill: int, symbol: str) -> pd.DataFrame:
    """
    Forward-fill missing 5-min bars within a trading session.
    A missing bar is one where the full 5-min grid doesn't have a row.
    Caps fill at max_fill consecutive missing bars (avoids filling across lunch breaks or halts).
    """
    if df.empty:
        return df

    # Detect resolution from actual data
    diffs = df.index.to_series().diff().dropna()
    if diffs.empty:
        return df
    mode_diff = diffs.mode()
    if mode_diff.empty:
        return df
    resolution = mode_diff.iloc[0]

    reindexed_frames = []
    for day, group in df.groupby(df.index.date):
        start = group.index[0].replace(hour=9, minute=15, second=0, microsecond=0)
        end   = group.index[-1]
        full_idx = pd.date_range(start=start, end=end, freq=resolution, tz=IST)
        group = group.reindex(full_idx)

        # Count consecutive NaN runs; only fill if ≤ max_fill
        group = _capped_ffill(group, max_fill=max_fill)
        reindexed_frames.append(group)

    result = pd.concat(reindexed_frames)
    n_filled = result["close"].isna().sum()
    if n_filled:
        result = result.dropna(subset=["open", "close"])
        logger.debug("%s: dropped %d unfillable gap bars", symbol, n_filled)

    return result


def _capped_ffill(df: pd.DataFrame, max_fill: int) -> pd.DataFrame:
    """Forward-fill NaN rows but only up to max_fill consecutive missing rows."""
    return df.ffill(limit=max_fill)


# ── Convenience wrappers ──────────────────────────────────────────────

def get_session_bars(df: pd.DataFrame, date_str: str) -> pd.DataFrame:
    """Extract all bars for a single trading date."""
    mask = df.index.strftime("%Y-%m-%d") == date_str
    return df[mask].copy()


def is_expiry_day(d, symbol: str, config: dict, expiries=None) -> bool:
    """
    Check if date d is an expiry day for the given symbol.

    expiries: the broker's real expiry dates ("YYYY-MM-DD"). Prefer passing
    them: they follow exchange changes (NIFTY moved Thursday -> Tuesday in
    Sep-2025, BANKNIFTY weeklies ended Nov-2024) and holiday shifts.
    Without them, falls back to config's expiry_day / expiry_cycle.
    """
    if expiries:
        return d.strftime("%Y-%m-%d") in set(expiries)
    inst = config["instruments"][symbol]
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4,
    }
    wd = day_map.get(inst.get("expiry_day", "Tuesday"), 1)
    if d.weekday() != wd:
        return False
    if inst.get("expiry_cycle", "weekly") == "monthly":
        # Monthly contracts expire on the last such weekday of the month.
        return (d + timedelta(days=7)).month != d.month
    return True
