"""
zerodha_historical.py — Expired option data helper for Zerodha backtesting.

Problem:
  Zerodha's NFO instruments endpoint only returns ACTIVE contracts.
  Expired option contracts disappear from kite.instruments("NFO") after expiry,
  so get_expired_option_key() in zerodha.py returns None for old contracts.

Solution:
  1. Save instrument snapshots daily to data/zerodha_instruments/ (CSV per date).
  2. When backtesting, load the snapshot for the trade date to resolve tokens.
  3. Use Kite historical_data() with the resolved token to fetch candles.
  4. Fall back gracefully to Black-Scholes if snapshot is missing.

Usage (automatic via DataFetcher — no manual calls needed):
  fetcher = DataFetcher(broker=get_broker())   # broker = ZerodhaBroker
  candle = fetcher.get_option_candle_at("NIFTY", "2026-06-05", 24000, "CE", "09:16")

Save today's snapshot (run once daily before market open, or after auth):
  python -m src.broker.zerodha_historical --save-snapshot

The snapshot is also saved automatically when zerodha_auth.py completes.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

SNAPSHOT_DIR = Path("data/zerodha_instruments")


# ── Snapshot management ────────────────────────────────────────────────

def save_snapshot(kite, date_str: Optional[str] = None) -> Path:
    """
    Fetch full NFO instruments list from Zerodha and save as CSV snapshot.
    Call this once per day (ideally from zerodha_auth.py after token refresh).

    Args:
        kite:      KiteConnect instance (already authenticated).
        date_str:  Date string 'YYYY-MM-DD'. Defaults to today.
    Returns:
        Path to saved CSV.
    """
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = date_str or datetime.now().strftime("%Y-%m-%d")
    path = SNAPSHOT_DIR / f"nfo_{date_str}.csv"

    if path.exists():
        logger.info("Snapshot already exists: %s", path)
        return path

    logger.info("Fetching NFO instruments snapshot for %s...", date_str)
    instruments = kite.instruments(exchange="NFO")
    df = pd.DataFrame(instruments)
    df.to_csv(path, index=False)
    logger.info("Snapshot saved: %s (%d instruments)", path, len(df))
    return path


def load_snapshot(date_str: str) -> Optional[pd.DataFrame]:
    """
    Load saved NFO instruments snapshot for a given date.
    Returns None if no snapshot exists for that date.
    """
    path = SNAPSHOT_DIR / f"nfo_{date_str}.csv"
    if not path.exists():
        # Try finding nearest earlier snapshot
        available = sorted(SNAPSHOT_DIR.glob("nfo_*.csv"), reverse=True)
        for snap in available:
            snap_date = snap.stem.replace("nfo_", "")
            if snap_date <= date_str:
                logger.debug("Using nearest snapshot %s for date %s", snap_date, date_str)
                path = snap
                break
        else:
            logger.warning(
                "No instrument snapshot found for %s or earlier. "
                "Run: python -m src.broker.zerodha_historical --save-snapshot",
                date_str,
            )
            return None

    df = pd.read_csv(path)
    return df


# ── Token resolution from snapshot ─────────────────────────────────────────

def resolve_option_token(
    snapshot: pd.DataFrame,
    name: str,          # e.g. "NIFTY", "BANKNIFTY"
    expiry: str,        # "YYYY-MM-DD"
    strike: int,
    option_type: str,   # "CE" | "PE"
) -> Optional[int]:
    """
    Look up instrument_token for an option contract from a snapshot DataFrame.
    Returns integer token, or None if not found.
    """
    name = name.upper()
    option_type = option_type.upper()

    # Normalise expiry column format
    snap = snapshot.copy()
    snap["expiry_str"] = pd.to_datetime(snap["expiry"]).dt.strftime("%Y-%m-%d")

    mask = (
        (snap["name"] == name) &
        (snap["instrument_type"] == option_type) &
        (snap["expiry_str"] == expiry) &
        (snap["strike"].astype(float) == float(strike))
    )
    matched = snap[mask]
    if matched.empty:
        return None
    return int(matched.iloc[0]["instrument_token"])


def get_expiries_from_snapshot(
    snapshot: pd.DataFrame,
    name: str,
    before_date: Optional[str] = None,
) -> list[str]:
    """
    Return sorted list of expiry dates from snapshot for a given instrument.
    If before_date is set, only return expiries on or before that date.
    """
    name = name.upper()
    snap = snapshot.copy()
    snap["expiry_str"] = pd.to_datetime(snap["expiry"]).dt.strftime("%Y-%m-%d")
    options = snap[
        (snap["name"] == name) &
        (snap["instrument_type"].isin(["CE", "PE"]))
    ]
    expiries = sorted(options["expiry_str"].unique().tolist())
    if before_date:
        expiries = [e for e in expiries if e <= before_date]
    return expiries


# ── ZerodhaBroker extension methods ──────────────────────────────────────

class ZerodhaBrokerWithHistory:
    """
    Mixin-style wrapper that extends ZerodhaBroker with snapshot-based
    expired option data methods for backtesting.

    Usage:
        from src.broker.zerodha import ZerodhaBroker
        from src.broker.zerodha_historical import ZerodhaBrokerWithHistory

        class ZerodhaBrokerFull(ZerodhaBrokerWithHistory, ZerodhaBroker):
            pass

        broker = ZerodhaBrokerFull(api_key=..., access_token=...)
    """

    def get_expired_expiries(self, instrument_key: str) -> list[str]:
        """
        Return sorted list of past expiry dates using saved instrument snapshots.
        Falls back to live instruments cache if no snapshot available.
        """
        # instrument_key format: "NSE_INDEX|Nifty 50" → extract name
        name = _extract_name(instrument_key)
        today = datetime.now().strftime("%Y-%m-%d")

        # Try most recent snapshot first
        available = sorted(SNAPSHOT_DIR.glob("nfo_*.csv"), reverse=True)
        if available:
            snap = pd.read_csv(available[0])
            expiries = get_expiries_from_snapshot(snap, name, before_date=today)
            if expiries:
                return expiries

        # Fall back to live instruments cache (only active contracts)
        logger.warning(
            "No snapshot found for get_expired_expiries(%s). "
            "Using live instruments — expired contracts may be missing.",
            instrument_key,
        )
        return super().get_expired_expiries(instrument_key)  # type: ignore[misc]

    def get_expired_option_key(
        self,
        instrument_key: str,
        expiry: str,
        strike: int,
        option_type: str,
    ):
        """
        Resolve instrument_token for an expired option using saved snapshots.
        Searches snapshots from nearest date before expiry backwards.
        """
        name = _extract_name(instrument_key)

        # Search snapshots in reverse chronological order up to expiry date
        available = sorted(SNAPSHOT_DIR.glob("nfo_*.csv"), reverse=True)
        for snap_path in available:
            snap_date = snap_path.stem.replace("nfo_", "")
            if snap_date > expiry:
                continue  # snapshot after expiry won't have this contract
            snap = pd.read_csv(snap_path)
            token = resolve_option_token(snap, name, expiry, strike, option_type)
            if token is not None:
                logger.debug(
                    "Resolved %s %s%s expiry=%s token=%d from snapshot %s",
                    name, strike, option_type, expiry, token, snap_date,
                )
                return str(token)

        logger.warning(
            "Could not resolve token for %s %s%s expiry=%s — no matching snapshot.",
            name, strike, option_type, expiry,
        )
        return None

    def get_expired_option_candles(
        self,
        expired_option_key: str,
        interval: str,
        date_str: str,
    ) -> list:
        """
        Fetch historical candles for an expired option using its instrument_token.
        expired_option_key: integer token as string (from get_expired_option_key).
        """
        try:
            df = self.get_historical_candles(  # type: ignore[attr-defined]
                instrument_key=expired_option_key,
                interval=interval,
                from_date=date_str,
                to_date=date_str,
            )
        except Exception as e:
            logger.warning(
                "Failed to fetch expired option candles for token %s on %s: %s",
                expired_option_key, date_str, e,
            )
            return []

        if df.empty:
            return []

        df.reset_index(inplace=True)
        # Return list of [timestamp, open, high, low, close, volume] for compatibility
        records = []
        for _, row in df.iterrows():
            records.append([
                row.get("timestamp", row.get("index", "")),
                row["open"], row["high"], row["low"], row["close"],
                row.get("volume", 0),
            ])
        return records


# ── Full broker class (import this for backtesting) ─────────────────────

def get_zerodha_backtest_broker():
    """
    Returns a ZerodhaBroker instance with full snapshot-based expired option
    data support. Use this instead of get_broker() for backtesting.

    Example:
        from src.broker.zerodha_historical import get_zerodha_backtest_broker
        broker = get_zerodha_backtest_broker()
        backtester = Backtester(broker=broker)
    """
    from dotenv import load_dotenv
    load_dotenv()

    from src.broker.zerodha import ZerodhaBroker

    class ZerodhaBrokerFull(ZerodhaBrokerWithHistory, ZerodhaBroker):
        """ZerodhaBroker + snapshot-based expired option data for backtesting."""
        pass

    return ZerodhaBrokerFull(
        api_key=os.environ["ZERODHA_API_KEY"],
        access_token=os.environ["ZERODHA_ACCESS_TOKEN"],
    )


# ── Internal helpers ──────────────────────────────────────────────────

_NAME_MAP = {
    "NSE_INDEX|Nifty 50":   "NIFTY",
    "NSE_INDEX|Nifty Bank": "BANKNIFTY",
    "BSE_INDEX|SENSEX":     "SENSEX",
}

def _extract_name(instrument_key: str) -> str:
    """Extract Zerodha NFO 'name' field from upstox-style instrument_key."""
    return _NAME_MAP.get(instrument_key, instrument_key.split("|")[-1].upper().replace(" ", ""))


# ── CLI: save today's snapshot ─────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from dotenv import load_dotenv
    load_dotenv()

    parser = argparse.ArgumentParser(description="Zerodha NFO instrument snapshot manager")
    parser.add_argument("--save-snapshot", action="store_true", help="Save today's NFO instruments snapshot")
    parser.add_argument("--date", default=None, help="Date for snapshot (YYYY-MM-DD, default: today)")
    args = parser.parse_args()

    if args.save_snapshot:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=os.environ["ZERODHA_API_KEY"])
        kite.set_access_token(os.environ["ZERODHA_ACCESS_TOKEN"])
        path = save_snapshot(kite, args.date)
        print(f"✅ Snapshot saved: {path}")
    else:
        parser.print_help()
